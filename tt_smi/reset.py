# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Reset-related functions for tt-smi: PCI board reset, galaxy 6U tray reset, and helpers.

Device-selection parsing for ``-r`` / ``--reset`` lives in
:mod:`tt_smi.device_input`; this module focuses on the reset mechanism itself.
"""

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from tt_tools_common.ui_common.themes import CMD_LINE_COLOR
from tt_tools_common.reset_common.wh_reset import WHChipReset
from tt_tools_common.reset_common.bh_reset import BHChipReset
from tt_smi.utils import (
    get_dev_id_from_bdf,
    get_driver_version,
    is_driver_version_at_least,
    IoctlResetFlags,
    reset_device_ioctl,
)
from tt_smi.constants import (
    get_default_discovery_options,
    BH_UBB_BUS_IDS,
    WH_UBB_BUS_IDS,
)
from tt_smi.device_input import SmiDeviceInput, SmiDeviceTargetKind
from pyluwen import (
    PciChip,
    pci_scan,
    run_ubb_wait_for_driver_load,
)
from tt_umd import (
    WarmReset,
    PCIDevice,
    TopologyDiscovery,
)
from tt_tools_common.utils_common.tools_utils import (
    detect_chips_with_callback,
)

MINIMUM_DRIVER_VERSION_GALAXY_SECONDARY_BUS_RESET = "2.7.0"
# BMC v0.05.22 adds op_mode 0x3: UBB reset without RST_RETIMER_PERST_N.
# op_mode 0x0 is the legacy command that also asserts RST_RETIMER_PERST_N.
MINIMUM_BMC_VERSION_NO_RETIMER_RESET = "0.05.22"
GLX_IPMI_OP_MODE_WITH_RETIMER = "0x0"
GLX_IPMI_OP_MODE_NO_RETIMER = "0x3"
# Holds the UBB in reset until the next reset command releases it.
GLX_IPMI_OP_MODE_HOLD = "0x1"


class ResetFailedError(Exception):
    """Warm reset failed; ``bdfs`` lists the chips that did not come back."""

    def __init__(self, bdfs: List[str]):
        self.bdfs = bdfs
        names = ", ".join(bdfs) if bdfs else "no chip named"
        super().__init__(
            f"Warm reset failed ({len(bdfs)} chip(s) did not come back: {names})"
        )


# Set by the KMD before the reset; a chip that really reset reads it back as 0.
PCI_COMMAND_RESET_MARKER = 0x40
SYSFS_PCI_DEVICES = "/sys/bus/pci/devices"
SYSFS_PCI_RESCAN = "/sys/bus/pci/rescan"

GALAXY_SUBSYSTEM_IDS = {0x35, 0x47, 0x202}
GALAXY_TRAYS = (1, 2, 3, 4)
GALAXY_CHIPS_PER_TRAY = 8
WORMHOLE_DEVICE_ID = 0x401E
# Rounds of tray re-power + PCI rescan for chips still missing after the warm reset.
RECOVERY_MAX_ROUNDS = 2
TRAY_REPOWER_WAIT_S = 30
CHIP_RETURN_TIMEOUT_S = 60
# Recount after POST_RESET: chips already back read ok at once.
POST_RESET_RECHECK_S = 10
# Wait for a held tray's chips to drop off PCI before the release pulse.
TRAY_HOLD_TIMEOUT_S = 30


def chip_reset_state(bdf: str, check_marker: bool = True) -> str:
    """Return "ok" if the chip at ``bdf`` came back from reset, else why not.

    After a successful reset the KMD restores the saved PCI config, which may
    set the marker bit again, so the marker is checked only on a failed reset.
    """
    try:
        with open(os.path.join(SYSFS_PCI_DEVICES, bdf, "config"), "rb") as f:
            config = f.read(8)
    except OSError as e:
        return f"gone from PCI ({e.strerror or e})"
    if len(config) < 8:
        return "config space unreadable"
    if config[0:2] == b"\xff\xff":
        return "config space reads all ones (chip not responding)"
    command = int.from_bytes(config[4:6], "little")
    if check_marker and command & PCI_COMMAND_RESET_MARKER:
        return "reset marker still set (chip did not reset)"
    return "ok"


def tray_of_bdf(bdf: str, wormhole: bool = False) -> Optional[int]:
    """Galaxy tray 1-4 of the chip at ``bdf`` (PCI bus & 0xF0), or None."""
    try:
        bus = int(bdf.split(":")[-2], 16) & 0xF0
    except (IndexError, ValueError):
        return None
    for tray, bus_id in (WH_UBB_BUS_IDS if wormhole else BH_UBB_BUS_IDS).items():
        if bus_id == bus:
            return tray
    return None


def _chip_label(bdf: str, is_galaxy: bool, wormhole: bool) -> str:
    tray = tray_of_bdf(bdf, wormhole) if is_galaxy else None
    return f"{bdf} (tray {tray})" if tray else bdf


def _present_chips() -> Dict[int, str]:
    """/dev/tenstorrent id -> BDF of every chip the driver sees now."""
    try:
        return {
            i: info.pci_bdf for i, info in PCIDevice.enumerate_devices_info().items()
        }
    except Exception as e:
        print(f"Warning: could not enumerate chips: {e}", file=sys.stderr)
        return {}


def _short_trays(present_bdfs, wormhole: bool) -> Dict[int, int]:
    """Galaxy trays with fewer than 8 chips: tray -> chips present."""
    counts = {tray: 0 for tray in GALAXY_TRAYS}
    for bdf in present_bdfs:
        tray = tray_of_bdf(bdf, wormhole)
        if tray in counts:
            counts[tray] += 1
    return {t: n for t, n in counts.items() if n < GALAXY_CHIPS_PER_TRAY}


def _sysfs_write(path: str, value: str) -> bool:
    try:
        with open(path, "w") as f:
            f.write(value)
        return True
    except PermissionError:
        pass
    except OSError as e:
        print(f"Warning: could not write {path}: {e}", file=sys.stderr)
        return False
    try:
        result = subprocess.run(
            ["sudo", "-n", "tee", path], input=value, capture_output=True, text=True
        )
    except FileNotFoundError:
        result = None
    if result is None or result.returncode != 0:
        detail = (
            "sudo not found"
            if result is None
            else (result.stderr.strip() or f"exit code {result.returncode}")
        )
        print(
            f"Warning: could not write {path} (needs root): {detail}", file=sys.stderr
        )
        return False
    return True


def rescan_pci(bdfs: List[str]) -> None:
    """Remove the listed chips still on PCI and rescan the bus (UMD's rescan_pci_bdfs, else sysfs).

    UMD's rescan needs a writable /sys/bus/pci/rescan and returns the chips still missing
    (all of them, without raising, when it is not writable); as non-root, or for chips UMD
    did not bring back, the sysfs path runs (through ``sudo -n tee`` when needed).
    """
    rescan = getattr(WarmReset, "rescan_pci_bdfs", None)
    if rescan is not None and os.access(SYSFS_PCI_RESCAN, os.W_OK):
        try:
            left = rescan(list(bdfs), timeout_s=10.0)
            if not left:
                return
            bdfs = list(left)
            print(
                f"Warning: still missing after UMD PCI rescan: {', '.join(bdfs)}; using the sysfs rescan.",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"Warning: UMD PCI rescan failed ({e}); using the sysfs rescan.",
                file=sys.stderr,
            )
    for bdf in bdfs:
        remove = os.path.join(SYSFS_PCI_DEVICES, bdf, "remove")
        if os.path.exists(remove):
            _sysfs_write(remove, "1")
    _sysfs_write(SYSFS_PCI_RESCAN, "1")


def _tray_ipmi_reset(trays: List[int]) -> None:
    """Re-power only ``trays`` through this host's BMC (in-band ipmitool).

    UBB n on the BMC is tray n+1, so tray t is bit t-1 of the UBB bitmap.
    If the BMC rejects the per-tray command, reset all four trays (as -glx_reset).
    """
    bitmap = 0
    for tray in trays:
        bitmap |= 1 << (tray - 1)
    if bitmap != 0xF:
        result = _run_ipmitool(
            _galaxy_ipmi_cmd(
                hex(bitmap).upper().replace("X", "x"),
                "0xFF",
                GLX_IPMI_OP_MODE_NO_RETIMER,
                "0xF",
            )
        )
        failure = _ipmi_failure_detail(result)
        if not failure:
            return
        print(
            f"{CMD_LINE_COLOR.YELLOW}WARNING!{CMD_LINE_COLOR.ENDC} "
            f"Per-tray IPMI reset of tray(s) {', '.join(map(str, trays))} failed:\n{failure}\n"
            f"{CMD_LINE_COLOR.BLUE}Resetting all four trays of this host instead...{CMD_LINE_COLOR.ENDC}"
        )
        # The per-tray failure may be the BMC going away: check again before the 0xF reset.
        if not _ipmi_answers():
            raise IpmiNotAnswering(IPMI_SKIPPED)
    run_galaxy_ipmi_reset("0xF", "0xFF", GLX_IPMI_OP_MODE_NO_RETIMER, "0xF")


def _tray_ipmi_hold(trays: List[int], bdfs: List[str]) -> None:
    """Hold ``trays`` in reset through this host's BMC until ``bdfs`` drop off PCI.

    The caller's re-power pulse then releases them. Raises RuntimeError if the BMC
    rejects the hold; a hold that leaves chips on PCI only warns (the pulse still runs).
    """
    bitmap = 0
    for tray in trays:
        bitmap |= 1 << (tray - 1)
    result = _run_ipmitool(
        _galaxy_ipmi_cmd(
            hex(bitmap).upper().replace("X", "x"), "0xFF", GLX_IPMI_OP_MODE_HOLD, "0xF"
        )
    )
    failure = _ipmi_failure_detail(result)
    if failure:
        raise RuntimeError(
            f"hold of tray(s) {', '.join(map(str, trays))} failed:\n{failure}"
        )
    deadline = time.monotonic() + TRAY_HOLD_TIMEOUT_S
    while True:
        up = [b for b in bdfs if chip_reset_state(b, check_marker=False) == "ok"]
        if not up or time.monotonic() >= deadline:
            break
        time.sleep(1)
    if up:
        print(
            f"{CMD_LINE_COLOR.YELLOW}WARNING!{CMD_LINE_COLOR.ENDC} "
            f"{len(up)} chip(s) still read on PCI {TRAY_HOLD_TIMEOUT_S} s into the hold "
            f"of tray(s) {', '.join(map(str, trays))}: {', '.join(up)}"
        )


IPMI_CHECK_TIMEOUT_S = 10
IPMI_SKIPPED = "in-band IPMI not answering; tray re-power skipped"


class IpmiNotAnswering(RuntimeError):
    """The BMC did not answer ``ipmitool mc info`` before a re-power step."""


def _ipmi_answers() -> bool:
    """True if this host's BMC answers in-band (``ipmitool mc info``, 10 s limit)."""
    cmd = ["sudo", "ipmitool", "mc", "info"]
    print(f"Executing command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=IPMI_CHECK_TIMEOUT_S
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _repower_trays(
    trays: List[int],
    wormhole: bool,
    user_reset: Optional[List[str]] = None,
    hold: Tuple[int, ...] = (),
) -> None:
    """SBR + USER_RESET on the trays' chips, re-power the trays, then wait.

    Trays in ``hold`` are held in reset until their chips drop off PCI before the re-power
    pulse. The BDFs sent USER_RESET are added to ``user_reset`` before the re-power, so a
    caller whose re-power raises still knows which chips need POST_RESET.
    """
    present = _present_chips()
    in_trays = [i for i, bdf in present.items() if tray_of_bdf(bdf, wormhole) in trays]
    if in_trays and should_use_secondary_bus_reset(True):
        for interface_id in parallel_reset_device_ioctl(
            in_trays, IoctlResetFlags.RESET_PCIE_LINK
        ):
            print(
                f"Warning: Secondary bus reset not completed for device /dev/tenstorrent/{interface_id}. Continuing..."
            )
    for interface_id in parallel_reset_device_ioctl(
        in_trays, IoctlResetFlags.USER_RESET
    ):
        print(
            f"Warning: USER_RESET did not complete for device /dev/tenstorrent/{interface_id}. Continuing..."
        )
    if user_reset is not None:
        user_reset.extend(present[i] for i in in_trays)
    if hold:
        _tray_ipmi_hold(
            list(hold),
            [b for b in present.values() if tray_of_bdf(b, wormhole) in hold],
        )
    try:
        _tray_ipmi_reset(trays)
    except RuntimeError as e:
        if hold:
            bitmap = 0
            for tray in hold:
                bitmap |= 1 << (tray - 1)
            release = " ".join(
                _galaxy_ipmi_cmd(
                    hex(bitmap).upper().replace("X", "x"),
                    "0xFF",
                    GLX_IPMI_OP_MODE_NO_RETIMER,
                    "0xF",
                )
            )
            print(
                CMD_LINE_COLOR.RED,
                f"Tray(s) {', '.join(map(str, hold))} are still held in reset: the release "
                f"pulse failed ({e}). Release them in-band with: {release}",
                CMD_LINE_COLOR.ENDC,
                file=sys.stderr,
            )
        raise
    timed_wait(TRAY_REPOWER_WAIT_S)


def _recount(
    missing: List[str],
    known: Optional[List[str]],
    count_trays: bool,
    is_galaxy: bool,
    wormhole: bool,
    timeout_s: float,
) -> Tuple[Dict[int, str], List[str], Dict[int, int]]:
    """Wait up to ``timeout_s`` for the ``missing`` and ``known`` chips to read back on PCI.

    Returns the chips the driver sees (id -> BDF), the chips still not back (not on PCI,
    or config not reading back) and the short Galaxy trays (counted when ``count_trays``).
    """
    deadline = time.monotonic() + timeout_s
    check = set(missing) | set(known or [])
    while True:
        present = _present_chips()
        bdfs = set(present.values())
        still = sorted(
            b
            for b in check
            if b not in bdfs or chip_reset_state(b, check_marker=False) != "ok"
        )
        short = _short_trays(bdfs, wormhole) if is_galaxy and count_trays else {}
        if (not still and not short) or time.monotonic() >= deadline:
            return present, still, short
        time.sleep(1)


def recover_missing_chips(
    missing: List[str],
    short_trays: Dict[int, int],
    is_galaxy: bool,
    wormhole: bool,
    known: Optional[List[str]] = None,
    repower: bool = True,
) -> Tuple[List[str], Dict[int, int], Optional[str]]:
    """Bring back chips that did not come back from the warm reset.

    Each round re-powers only the trays with a missing chip (Galaxy with ``repower`` only;
    all four trays only for a chip outside the tray map), rescans PCI, sends POST_RESET and
    recounts the missing chips and, after a re-power, every chip of ``known`` (a re-power can
    knock other trays off PCI too), so those feed the next round. Each chip is judged by the
    recount after POST_RESET: a chip on PCI whose config reads back is present, even if its
    POST_RESET failed (the re-init then checks it), unless its reset marker is still set: the
    re-power did not reset it, so it stays missing and the next round holds its tray in reset
    before the re-power pulse. Once every chip is present, no further round runs.
    Before each re-power (and before the all-tray fallback) the BMC must answer
    ``ipmitool mc info``; if it does not, or a re-power fails, recovery stops there: no
    further SBR, USER_RESET or re-power. A stop after a round's SBR + USER_RESET still
    rescans PCI, sends POST_RESET to the chips that got USER_RESET and recounts every chip.
    Returns the chips (by BDF) and short trays (tray -> chips present) still missing afterwards,
    and why recovery stopped early (None if it did not).
    """
    allow_repower = repower
    stopped = None
    # A chip whose reset marker is set after the warm reset was not reset, and the plain
    # 0x3 pulse does not reset a live tray (task #3791, 2/2): round 1 holds its tray.
    hold: set = {
        tray_of_bdf(b, wormhole)
        for b in missing
        if chip_reset_state(b).startswith("reset marker")
    }
    hold.discard(None)
    for round_num in range(1, RECOVERY_MAX_ROUNDS + 1):
        trays = sorted(
            {tray_of_bdf(b, wormhole) for b in missing} | set(short_trays), key=str
        )
        if is_galaxy and allow_repower:
            # Only trays with a missing chip: a tray whose chips are all present is never
            # re-powered. A chip on a bus outside the tray map gets all four trays re-powered.
            repower = list(GALAXY_TRAYS) if None in trays else trays
            # A re-power with no BMC answer can take healthy chips down (OM-F3-GBH01: 24/32
            # to 0/32), so check before every round: no answer, no SBR, USER_RESET or re-power.
            if not _ipmi_answers():
                print(
                    CMD_LINE_COLOR.RED,
                    "In-band IPMI (ipmitool mc info) did not answer within "
                    f"{IPMI_CHECK_TIMEOUT_S} s: no tray re-power, no SBR or USER_RESET on any chip.",
                    CMD_LINE_COLOR.ENDC,
                    file=sys.stderr,
                )
                return sorted(missing), short_trays, IPMI_SKIPPED
            print(
                CMD_LINE_COLOR.PURPLE,
                f"Recovery round {round_num}/{RECOVERY_MAX_ROUNDS}: re-powering tray(s) "
                f"{', '.join(map(str, repower))}, then PCI rescan...",
                CMD_LINE_COLOR.ENDC,
            )
            user_reset: List[str] = []
            try:
                _repower_trays(
                    repower, wormhole, user_reset, tuple(sorted(hold & set(repower)))
                )
            except RuntimeError as e:
                # Stop: escalating after a failed re-power can take down healthy trays too.
                stopped = (
                    str(e)
                    if isinstance(e, IpmiNotAnswering)
                    else f"re-power of tray(s) {', '.join(map(str, repower))} failed ({e}); "
                    "no further re-power"
                )
                print(
                    CMD_LINE_COLOR.RED,
                    f"Tray re-power failed: {e}. Recovery stops: no further SBR, USER_RESET or "
                    "re-power; PCI rescan, POST_RESET and recount only.",
                    CMD_LINE_COLOR.ENDC,
                    file=sys.stderr,
                )
        else:
            repower = []
            print(f"Recovery round {round_num}/{RECOVERY_MAX_ROUNDS}: PCI rescan...")
        rescan_pci(missing)
        present, still, short = _recount(
            missing,
            known if repower else None,
            bool(short_trays or repower),
            is_galaxy,
            wormhole,
            CHIP_RETURN_TIMEOUT_S,
        )

        if repower:
            # After a stop the trays may not have been re-powered: POST_RESET goes to the
            # chips that got USER_RESET, not to every chip on those trays.
            post_ids = [
                i
                for i, bdf in present.items()
                if (
                    bdf in user_reset
                    if stopped
                    else tray_of_bdf(bdf, wormhole) in repower
                )
            ]
            post_failed = list(
                parallel_reset_device_ioctl(post_ids, IoctlResetFlags.POST_RESET)
            )
            # Judge every chip again after POST_RESET (OM-E4-GBH02, tray 3 held in reset:
            # round 1 counted c1-c8:00.0 missing on the check before POST_RESET, so round 2
            # re-powered all four trays of a box that recounted 32/32 and exited 1).
            present, still, short = _recount(
                missing, known, True, is_galaxy, wormhole, POST_RESET_RECHECK_S
            )
            hold = set()
            for interface_id in post_failed:
                bdf = present.get(interface_id)
                back = bdf is not None and bdf not in still
                # The KMD fails POST_RESET while the marker is set and then restores nothing,
                # so here the marker means the chip never reset (OM-E4-GBH02, live tray 3: the
                # 0x3 pulse left c1-c8:00.0 unreset and re-init timed out on their firmware).
                if back and chip_reset_state(bdf).startswith("reset marker"):
                    still = sorted(set(still) | {bdf})
                    hold.add(tray_of_bdf(bdf, wormhole))
                    note = (
                        " POST_RESET failed and the reset marker reads set "
                        "(chip may not have reset); it stays missing."
                    )
                elif back:
                    note = " The chip reads back on PCI; the re-init checks it."
                else:
                    note = ""
                print(
                    CMD_LINE_COLOR.RED,
                    f"Error: POST_RESET failed for device /dev/tenstorrent/{interface_id} "
                    f"({_chip_label(bdf, is_galaxy, wormhole) if bdf else 'gone'}).{note}",
                    CMD_LINE_COLOR.ENDC,
                    file=sys.stderr,
                )
            hold.discard(None)
        bdfs = set(present.values())
        missing = sorted(set(still))
        short_trays = short
        if is_galaxy:
            # Chips on PCI whose config reads back, not only the driver's device count.
            back = len(bdfs - set(still))
            print(
                f"Recount after round {round_num}: {back}/{len(GALAXY_TRAYS) * GALAXY_CHIPS_PER_TRAY} chips"
            )
        if stopped or (not missing and not short_trays):
            break
    return missing, short_trays, stopped if (missing or short_trays) else None


def _ipmi_output_is_param_out_of_range(output: str) -> bool:
    text = output.lower()
    return "rsp=0xc9" in text or "parameter out of range" in text


def _galaxy_ipmi_cmd(
    ubb_num: str, dev_num: str, op_mode: str, reset_time: str
) -> List[str]:
    return [
        "sudo",
        "ipmitool",
        "raw",
        "0x30",
        "0x8B",
        ubb_num,
        dev_num,
        op_mode,
        reset_time,
    ]


def _run_ipmitool(cmd: List[str]) -> subprocess.CompletedProcess:
    print(f"Executing command: {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Failed to execute ipmitool. Ensure ipmitool is installed and sudo is available."
        ) from e


def _ipmi_failure_detail(result: subprocess.CompletedProcess) -> str:
    if result.returncode == 0:
        return ""
    return f"{result.stdout}{result.stderr}".strip() or f"exit code {result.returncode}"


def run_galaxy_ipmi_reset(
    ubb_num: str = "0xF",
    dev_num: str = "0xFF",
    op_mode: str = GLX_IPMI_OP_MODE_NO_RETIMER,
    reset_time: str = "0xF",
) -> None:
    """
    Issue the Galaxy UBB IPMI reset, preferring no-retimer op_mode 0x3.

    BMC firmware v0.05.22 or later accepts 0x3. Older BMCs reject it with
    IPMI completion code 0xc9 (Parameter out of range) and do not reset.
    On any 0x3 failure, print a warning and fall back to legacy op_mode 0x0
    so the tray reset still goes through.
    """
    result = _run_ipmitool(_galaxy_ipmi_cmd(ubb_num, dev_num, op_mode, reset_time))
    failure = _ipmi_failure_detail(result)
    if not failure:
        return

    # Already on the legacy command (or a caller overrode op_mode): do not recurse.
    if op_mode == GLX_IPMI_OP_MODE_WITH_RETIMER:
        raise RuntimeError(f"IPMI command failed: {failure}")

    if _ipmi_output_is_param_out_of_range(failure):
        print(
            f"{CMD_LINE_COLOR.YELLOW}WARNING! "
            f"This Galaxy is on BMC < v{MINIMUM_BMC_VERSION_NO_RETIMER_RESET}, "
            "so reset without PCIe retimer (op_mode 0x3) is not supported. "
            f"Upgrade BMC firmware to v{MINIMUM_BMC_VERSION_NO_RETIMER_RESET} or later to enable it.{CMD_LINE_COLOR.ENDC}\n"
            f"{CMD_LINE_COLOR.BLUE}Falling back to legacy retimer reset (op_mode {GLX_IPMI_OP_MODE_WITH_RETIMER})...{CMD_LINE_COLOR.ENDC}"
        )
    else:
        print(
            f"{CMD_LINE_COLOR.YELLOW}WARNING!{CMD_LINE_COLOR.ENDC} "
            f"Galaxy no-retimer IPMI reset (op_mode {GLX_IPMI_OP_MODE_NO_RETIMER}) failed:\n"
            f"{failure}\n"
            f"{CMD_LINE_COLOR.BLUE}Falling back to legacy retimer reset (op_mode {GLX_IPMI_OP_MODE_WITH_RETIMER})...{CMD_LINE_COLOR.ENDC}"
        )

    fallback = _run_ipmitool(
        _galaxy_ipmi_cmd(ubb_num, dev_num, GLX_IPMI_OP_MODE_WITH_RETIMER, reset_time)
    )
    fallback_failure = _ipmi_failure_detail(fallback)
    if fallback_failure:
        raise RuntimeError(f"IPMI command failed: {fallback_failure}")


def should_use_secondary_bus_reset(is_galaxy: bool) -> bool:
    """
    Whether to issue secondary bus reset (RESET_PCIE_LINK IOCTL) before ASIC reset.

    Non-Galaxy systems always use secondary bus reset. Galaxy systems use it only
    when KMD is >= 2.7.0.
    """
    if not is_galaxy:
        return True
    driver_version = get_driver_version()
    if driver_version is None:
        return False
    return is_driver_version_at_least(
        driver_version, MINIMUM_DRIVER_VERSION_GALAXY_SECONDARY_BUS_RESET
    )


def parallel_reset_device_ioctl(device_ids: List[int], flag: int) -> List[int]:
    """
    Issue reset ioctl on each device in parallel.

    Returns interface IDs where the ioctl did not complete successfully.
    """
    if not device_ids:
        return []

    def ioctl_one(interface_id: int) -> tuple[int, bool]:
        return interface_id, reset_device_ioctl(interface_id, flag)

    failed: List[int] = []
    with ThreadPoolExecutor(max_workers=len(device_ids)) as pool:
        for interface_id, ok in pool.map(ioctl_one, device_ids):
            if not ok:
                failed.append(interface_id)
    return failed


def timed_wait(seconds):
    print("\033[93mWaiting for {} seconds: 0\033[0m".format(seconds), end="")
    sys.stdout.flush()

    for i in range(1, seconds + 1):
        time.sleep(1)
        print("\r\033[93mWaiting for {} seconds: {}\033[0m".format(seconds, i), end="")
        sys.stdout.flush()
    print()


# Keep this function for now, but it's not used anywhere in the codebase.
# It is the check that is used in Funtest for Galaxy systems and we might need to reference it later.
def check_wh_galaxy_eth_link_status(devices):
    """
    Check the WH Galaxy Ethernet link status.
    Returns True if the link is up, False otherwise.
    """
    noc_id = 0
    DEBUG_BUF_ADDR = 0x12C0  # For eth fw 5.0.0 and above
    eth_locations_noc_0 = [
        (9, 0),
        (1, 0),
        (8, 0),
        (2, 0),
        (7, 0),
        (3, 0),
        (6, 0),
        (4, 0),
        (9, 6),
        (1, 6),
        (8, 6),
        (2, 6),
        (7, 6),
        (3, 6),
        (6, 6),
        (4, 6),
    ]
    LINK_INACTIVE_FAIL_DUMMY_PACKET = 10
    if len(devices) != 32:
        print(
            CMD_LINE_COLOR.RED,
            f"Error: Expected 32 devices for WH Galaxy Ethernet link status check, seeing {len(devices)}, please try reset again or cold boot the system.",
            CMD_LINE_COLOR.ENDC,
        )
        sys.exit(1)

    link_errors = {}
    for i, device in enumerate(devices):
        for eth in range(16):
            eth_x, eth_y = eth_locations_noc_0[eth]
            link_error = device.noc_read32(
                noc_id, eth_x, eth_y, DEBUG_BUF_ADDR + 0x4 * 96
            )
            if link_error == LINK_INACTIVE_FAIL_DUMMY_PACKET:
                link_errors[i] = eth

    if link_errors:
        for board_idx, eth in link_errors.items():
            print(
                CMD_LINE_COLOR.RED,
                f"Board {board_idx} has link error on eth port {eth}",
                CMD_LINE_COLOR.ENDC,
            )
        raise Exception("WH Galaxy Ethernet link errors detected!")


def umd_ubb_wait_for_driver_load():
    """
    Wait for the driver to reload for UMD, try 100 times.
    Similar to luwen's ubb_wait_for_driver_load but uses PCIDevice.enumerate_devices.
    """
    attempts = 0
    expected_chip_count = 32

    while attempts < 100:
        device_count = 0
        try:
            devices = PCIDevice.enumerate_devices()
            device_count = len(devices)
            if device_count == expected_chip_count:
                print(f"Driver loaded with {device_count} devices")
                return
        except Exception:
            pass

        print(
            f"Waiting for driver load ... {attempts} seconds (found {device_count} devices)"
        )
        time.sleep(1)
        attempts += 1

    raise Exception(
        f"Driver not loaded with {expected_chip_count} devices after 100 seconds... giving up"
    )


def _target_bdfs(reset_input: SmiDeviceInput, chips) -> List[str]:
    """BDFs of the chips a targeted reset names (ids not enumerated are skipped)."""
    if reset_input.type == SmiDeviceTargetKind.DEV_TENSTORRENT_ID:
        return [chips[i].pci_bdf for i in reset_input.value if i in chips]
    if reset_input.type == SmiDeviceTargetKind.UMD_LOGICAL_ID:
        # UMD logical id n is the n-th enumerated /dev/tenstorrent id, as in warm_reset_chip_id.
        ids = sorted(chips)
        return [chips[ids[i]].pci_bdf for i in reset_input.value if 0 <= i < len(ids)]
    if reset_input.type == SmiDeviceTargetKind.PCI_BDF:
        return [b.lower() for b in reset_input.value]
    return [info.pci_bdf for info in chips.values()]


def _recover_after_failed_reinit() -> bool:
    """Recount every chip after a failed re-init; re-power and rescan the lost ones.

    Returns True once all chips read back again (re-init may be retried), False if
    no chip reads as lost (the re-init failure has another cause) or chips stay lost.
    """
    chips = PCIDevice.enumerate_devices_info()
    is_galaxy = any(
        info.subsystem_id in GALAXY_SUBSYSTEM_IDS for info in chips.values()
    )
    wormhole = any(info.device_id == WORMHOLE_DEVICE_ID for info in chips.values())
    bdfs = sorted({info.pci_bdf for info in chips.values()})
    lost = [b for b in bdfs if chip_reset_state(b, check_marker=False) != "ok"]
    short_trays = _short_trays(bdfs, wormhole) if is_galaxy else {}
    if not lost and not short_trays:
        return False
    for bdf in lost:
        print(
            CMD_LINE_COLOR.RED,
            f"Chip {_chip_label(bdf, is_galaxy, wormhole)} lost after the recount: "
            f"{chip_reset_state(bdf)}",
            CMD_LINE_COLOR.ENDC,
            file=sys.stderr,
        )
    still, still_short, stopped = recover_missing_chips(
        lost, short_trays, is_galaxy, wormhole, known=bdfs, repower=True
    )
    if not still and not still_short:
        print(
            CMD_LINE_COLOR.GREEN,
            f"Recovered: all {len(lost)} lost chip(s) are back; re-initializing again.",
            CMD_LINE_COLOR.ENDC,
        )
        return True
    how = f"({stopped})" if stopped else "after tray re-power and PCI rescan"
    for bdf in still:
        print(
            CMD_LINE_COLOR.RED,
            f"Hardware fault: chip {_chip_label(bdf, is_galaxy, wormhole)} still missing {how}: "
            f"{chip_reset_state(bdf)}",
            CMD_LINE_COLOR.ENDC,
            file=sys.stderr,
        )
    for tray, count in sorted(still_short.items()):
        print(
            CMD_LINE_COLOR.RED,
            f"Hardware fault: tray {tray} has {count}/{GALAXY_CHIPS_PER_TRAY} chips {how}",
            CMD_LINE_COLOR.ENDC,
            file=sys.stderr,
        )
    return False


def umd_pci_warm_reset(
    reset_input: SmiDeviceInput,
):
    """
    Reset the PCI devices using UMD warm reset.
    """
    chips = PCIDevice.enumerate_devices_info()
    # check if any of the chips are galaxy
    is_galaxy = any(
        info.subsystem_id in GALAXY_SUBSYSTEM_IDS for info in chips.values()
    )
    wormhole = any(info.device_id == WORMHOLE_DEVICE_ID for info in chips.values())
    if is_galaxy:
        print(
            CMD_LINE_COLOR.YELLOW,
            "CPLD FW v1.16 or higher is required to use tt-smi -r on Galaxy systems.",
            "If tt-smi -r fails, please continue to use tt-smi -glx_reset instead and contact your system administrator to request a CPLD update.",
            CMD_LINE_COLOR.ENDC,
        )
    secondary_bus_reset = should_use_secondary_bus_reset(is_galaxy)

    # UMD with warm_reset_with_result names the chips that did not come back.
    with_result = getattr(WarmReset, "warm_reset_with_result", None)
    umd_missing: List[str] = []
    reset_indices = reset_input.value
    if reset_input.type in (
        SmiDeviceTargetKind.ALL,
        SmiDeviceTargetKind.DEV_TENSTORRENT_ID,
    ):
        if reset_input.type == SmiDeviceTargetKind.ALL:
            reset_indices = list(chips.keys())
            print(f"Resetting all PCI devices: {reset_indices}")
        else:
            print(f"Resetting /dev/tenstorrent IDs: {reset_input.value}")
        if with_result is not None:
            result = with_result(reset_indices, secondary_bus_reset=secondary_bus_reset)
            ok = bool(result.success)
            umd_missing = list(result.missing_pci_bdfs)
        else:
            ok = WarmReset.warm_reset(
                reset_indices, secondary_bus_reset=secondary_bus_reset
            )
    elif reset_input.type == SmiDeviceTargetKind.UMD_LOGICAL_ID:
        print(f"Resetting UMD logical IDs: {reset_input.value}")
        ok = WarmReset.warm_reset_chip_id(
            reset_indices, secondary_bus_reset=secondary_bus_reset
        )
    elif reset_input.type == SmiDeviceTargetKind.PCI_BDF:
        print(f"Resetting PCI BDFs: {reset_input.value}")
        ok = WarmReset.warm_reset_pci_bdfs(
            reset_indices, secondary_bus_reset=secondary_bus_reset
        )
    else:
        raise ValueError(f"Invalid reset type: {reset_input.type}")

    # A full reset judges every enumerated chip; a targeted one only its targets.
    full = reset_input.type == SmiDeviceTargetKind.ALL
    all_bdfs = sorted({info.pci_bdf for info in chips.values()} | set(umd_missing))
    if full:
        bdfs = all_bdfs
    else:
        bdfs = sorted(_target_bdfs(reset_input, chips))
        umd_missing = [b for b in umd_missing if b in bdfs]
        others = [
            b
            for b in all_bdfs
            if b not in bdfs and chip_reset_state(b, check_marker=False) != "ok"
        ]
        if others:
            print(
                CMD_LINE_COLOR.YELLOW,
                "Note: untargeted chip(s) not reading back (not judged by this reset): "
                + ", ".join(_chip_label(b, is_galaxy, wormhole) for b in others),
                CMD_LINE_COLOR.ENDC,
                file=sys.stderr,
            )
    states = {bdf: chip_reset_state(bdf, check_marker=not ok) for bdf in bdfs}
    for bdf in umd_missing:
        if states[bdf] == "ok":
            states[bdf] = "UMD warm reset reports it did not come back"
    failed = [bdf for bdf in bdfs if states[bdf] != "ok"]
    # A Galaxy has 8 chips on each of its 4 trays; a tray short before the reset is missing chips too.
    short_trays = _short_trays(bdfs, wormhole) if is_galaxy and full else {}
    if ok and not failed and not short_trays:
        return
    for bdf in failed:
        print(
            CMD_LINE_COLOR.RED,
            f"Chip {_chip_label(bdf, is_galaxy, wormhole)} did not come back from reset: {states[bdf]}",
            CMD_LINE_COLOR.ENDC,
            file=sys.stderr,
        )
    for tray, count in sorted(short_trays.items()):
        print(
            CMD_LINE_COLOR.RED,
            f"Tray {tray} has {count}/{GALAXY_CHIPS_PER_TRAY} chips",
            CMD_LINE_COLOR.ENDC,
            file=sys.stderr,
        )
    if not failed and not short_trays:
        print(
            CMD_LINE_COLOR.RED,
            "UMD warm reset reported failure; no chip reads back as failed (the reset may not have run).",
            CMD_LINE_COLOR.ENDC,
            file=sys.stderr,
        )
        raise ResetFailedError(failed)

    # Only a full reset re-powers trays: a targeted reset rescans PCI at most.
    still, still_short, stopped = recover_missing_chips(
        failed, short_trays, is_galaxy, wormhole, known=bdfs, repower=full
    )
    if not still and not still_short:
        print(
            CMD_LINE_COLOR.GREEN,
            f"Recovered: all {len(failed)} missing chip(s) are back"
            + (
                f"; trays {', '.join(map(str, sorted(short_trays)))} full again"
                if short_trays
                else ""
            )
            + ".",
            CMD_LINE_COLOR.ENDC,
        )
        return
    how = (
        f"({stopped})"
        if stopped
        else "after "
        + ("tray re-power and PCI rescan" if is_galaxy and full else "PCI rescan")
    )
    for bdf in still:
        print(
            CMD_LINE_COLOR.RED,
            f"Hardware fault: chip {_chip_label(bdf, is_galaxy, wormhole)} still missing {how}: "
            f"{chip_reset_state(bdf)}",
            CMD_LINE_COLOR.ENDC,
            file=sys.stderr,
        )
    if still and is_galaxy and not full:
        print(
            CMD_LINE_COLOR.YELLOW,
            "A targeted reset does not re-power trays; a full `tt-smi -r` or `tt-smi -glx_reset` "
            "can re-power the tray.",
            CMD_LINE_COLOR.ENDC,
            file=sys.stderr,
        )
    named = {tray_of_bdf(b, wormhole) for b in still}
    for tray, count in sorted(still_short.items()):
        if tray not in named:
            print(
                CMD_LINE_COLOR.RED,
                f"Hardware fault: tray {tray} has {count}/{GALAXY_CHIPS_PER_TRAY} chips {how} "
                "(the missing chip was not on PCI before the reset, so its BDF is not known)",
                CMD_LINE_COLOR.ENDC,
                file=sys.stderr,
            )
    raise ResetFailedError(
        still + [f"tray {t}" for t in sorted(still_short) if t not in named]
    )


def luwen_pci_reset(
    reset_input: SmiDeviceInput,
):
    """
    Reset the PCI devices using luwen (pyluwen): discover board type per device
    and call WHChipReset or BHChipReset as appropriate.
    """
    if reset_input.type == SmiDeviceTargetKind.ALL:
        reset_indices = pci_scan()
    elif reset_input.type == SmiDeviceTargetKind.UMD_LOGICAL_ID:
        print(
            CMD_LINE_COLOR.RED,
            "UMD ID reset not supported for luwen. Please use tt-smi -r /dev/tenstorrent/<id> or tt-smi -r <PCI BDF> instead.",
            CMD_LINE_COLOR.ENDC,
        )
        sys.exit(1)
    elif reset_input.type == SmiDeviceTargetKind.PCI_BDF:
        print(f"Resetting PCI BDFs: {reset_input.value}")
        reset_indices = [get_dev_id_from_bdf(bdf) for bdf in reset_input.value]
    elif reset_input.type == SmiDeviceTargetKind.DEV_TENSTORRENT_ID:
        print(f"Resetting /dev/tenstorrent IDs: {reset_input.value}")
        reset_indices = list(reset_input.value)
    else:
        raise ValueError(f"Invalid reset type: {reset_input.type}")

    reset_wh_pci_idx: List[int] = []
    reset_bh_pci_idx: List[int] = []
    board_types = set()
    for pci_idx in reset_indices:
        try:
            chip = PciChip(pci_interface=pci_idx)
        except Exception as e:
            print(e, file=sys.stderr)
            print(
                CMD_LINE_COLOR.RED,
                f"Error accessing board at PCI index {pci_idx}! Use -ls to see all devices available to reset",
                CMD_LINE_COLOR.ENDC,
            )
            continue
        try:
            if chip.as_wh():
                reset_wh_pci_idx.append(pci_idx)
                board_types.add(chip.as_wh().pci_board_type())
            elif chip.as_bh():
                reset_bh_pci_idx.append(pci_idx)
                board_types.add(chip.as_bh().pci_board_type())
            else:
                print(
                    CMD_LINE_COLOR.RED,
                    "Unknown chip type detected. Exiting...",
                    CMD_LINE_COLOR.ENDC,
                )
                sys.exit(1)
        finally:
            del chip

    is_galaxy = board_types <= {0x35, 0x47, 0x202}
    if is_galaxy:
        print(
            CMD_LINE_COLOR.YELLOW,
            "CPLD FW v1.16 or higher is required to use tt-smi -r on Galaxy systems.",
            "If tt-smi -r fails, please continue to use tt-smi -glx_reset instead and contact your system administrator to request a CPLD update.",
            CMD_LINE_COLOR.ENDC,
        )

    secondary_bus_reset = should_use_secondary_bus_reset(is_galaxy)
    if reset_wh_pci_idx:
        WHChipReset().full_lds_reset(
            pci_interfaces=reset_wh_pci_idx, secondary_bus_reset=secondary_bus_reset
        )
    if reset_bh_pci_idx:
        BHChipReset().full_lds_reset(
            pci_interfaces=reset_bh_pci_idx, secondary_bus_reset=secondary_bus_reset
        )


def pci_board_reset(
    reset_input: SmiDeviceInput,
    reinit: bool = False,
    print_status: bool = True,
    use_umd: bool = False,
    eth_train_skip: bool = False,
):
    """Given a ``SmiDeviceInput`` ``reset_input``, reset the PCI devices using UMD warm reset or luwen (pyluwen)."""

    if use_umd:
        try:
            umd_pci_warm_reset(reset_input)
        except ResetFailedError as e:
            print(CMD_LINE_COLOR.RED, e, CMD_LINE_COLOR.ENDC, file=sys.stderr)
            sys.exit(1)
    else:
        luwen_pci_reset(reset_input)

    if reinit:
        print(
            CMD_LINE_COLOR.PURPLE,
            "Re-initializing boards after reset....",
            CMD_LINE_COLOR.ENDC,
        )

        def reinit_chips():
            if use_umd:
                options = get_default_discovery_options()
                if eth_train_skip:
                    options.discover_remote_devices = False
                    options.wait_on_ethernet_link_training = False
                TopologyDiscovery.discover(options=options)
            else:
                os.environ["RUST_BACKTRACE"] = "full"
                detect_chips_with_callback(
                    print_status=print_status, ignore_ethernet=eth_train_skip
                )

        try:
            reinit_chips()
        except Exception as e:
            print(
                CMD_LINE_COLOR.RED,
                f"Error when re-initializing chips!\n {e}",
                CMD_LINE_COLOR.ENDC,
            )
            # A chip lost after the recount (it reads all ones at re-init) gets the
            # recovery rounds once more, then one more re-init, before giving up.
            if not (
                use_umd
                and reset_input.type == SmiDeviceTargetKind.ALL
                and _recover_after_failed_reinit()
            ):
                sys.exit(1)
            try:
                reinit_chips()
            except Exception as e2:
                print(
                    CMD_LINE_COLOR.RED,
                    f"Error when re-initializing chips after recovery!\n {e2}",
                    CMD_LINE_COLOR.ENDC,
                )
                sys.exit(1)


def glx_6u_trays_reset(
    reinit: bool = True,
    ubb_num: str = "0xF",
    dev_num: str = "0xFF",
    op_mode: str = GLX_IPMI_OP_MODE_NO_RETIMER,
    reset_time: str = "0xF",
    print_status: bool = True,
    use_umd: bool = False,
):
    """
    Reset the ASICs on the galaxy systems with the following steps:
    1. Perform USER_RESET ioctl on all chips
    2. Reset the trays with IPMI (op_mode 0x3, no PCIe retimer)
    3. Wait for 30s
    4. Perform POST_RESET ioctl on all chips
    5. Reinit all chips

    Args:
        reinit: Whether to reinitialize the chips after reset.
        ubb_num: The UBB number to reset. 0x0~0xF (bit map)
        dev_num: The device number to reset. 0x0~0xFF(bit map)
        op_mode: IPMI operation mode. 0x3 = reset without PCIe retimer (BMC v0.05.22+).
        reset_time: The reset time to use. resolution 10ms (ex. 0xF => 15 => 150ms)
        print_status: Whether to print out animations while detecting chips.
        use_umd: Whether to enumerate/reinit with UMD or pyluwen.
    """
    # First, check if we're trying to do anything other than a full reset
    if (
        ubb_num != "0xF"
        or dev_num != "0xFF"
        or op_mode != GLX_IPMI_OP_MODE_NO_RETIMER
        or reset_time != "0xF"
    ):
        print(
            CMD_LINE_COLOR.RED,
            "Error: Galaxy 6U IPMI reset only supports full Galaxy reset ",
            f"(ubb_num=0xF, dev_num=0xFF, op_mode={GLX_IPMI_OP_MODE_NO_RETIMER}, reset_time=0xF)",
            CMD_LINE_COLOR.ENDC,
        )
        sys.exit(1)
    print(
        CMD_LINE_COLOR.PURPLE,
        "Resetting WH Galaxy trays with reset command...",
        CMD_LINE_COLOR.ENDC,
    )

    if use_umd:
        device_ids = list(PCIDevice.enumerate_devices())
    else:
        device_ids = pci_scan()

    if should_use_secondary_bus_reset(True):
        print(
            CMD_LINE_COLOR.BLUE,
            f"Issuing RESET_PCIE_LINK on {len(device_ids)} devices before IPMI reset...",
            CMD_LINE_COLOR.ENDC,
        )
        for interface_id in parallel_reset_device_ioctl(
            device_ids, IoctlResetFlags.RESET_PCIE_LINK
        ):
            print(
                CMD_LINE_COLOR.YELLOW,
                f"Warning: Secondary bus reset not completed for device /dev/tenstorrent/{interface_id}. Continuing...",
                CMD_LINE_COLOR.ENDC,
            )

    print(
        CMD_LINE_COLOR.BLUE,
        f"Issuing USER_RESET on {len(device_ids)} devices before IPMI reset...",
        CMD_LINE_COLOR.ENDC,
    )
    for interface_id in parallel_reset_device_ioctl(
        device_ids, IoctlResetFlags.USER_RESET
    ):
        print(
            CMD_LINE_COLOR.YELLOW,
            f"Warning: USER_RESET did not complete for device /dev/tenstorrent/{interface_id}. Continuing...",
            CMD_LINE_COLOR.ENDC,
        )

    # IPMI reset without PCIe retimer (BMC v0.05.22+). Do not fall back to
    # op_mode 0x0 — that asserts RST_RETIMER_PERST_N and can drop NVMe.
    run_galaxy_ipmi_reset(ubb_num, dev_num, op_mode, reset_time)
    timed_wait(30)
    # This function waits for all 32 chips to reappear on the bus.
    run_ubb_wait_for_driver_load()

    # Issue POST_RESET ioctl on all devices after they reappear
    if use_umd:
        post_reset_ids = list(PCIDevice.enumerate_devices())
    else:
        post_reset_ids = pci_scan()
    print(
        CMD_LINE_COLOR.BLUE,
        f"Issuing POST_RESET on {len(post_reset_ids)} devices after IPMI reset...",
        CMD_LINE_COLOR.ENDC,
    )
    post_reset_failed = parallel_reset_device_ioctl(
        post_reset_ids, IoctlResetFlags.POST_RESET
    )
    for interface_id in post_reset_failed:
        print(
            CMD_LINE_COLOR.RED,
            f"Error: POST_RESET failed for device /dev/tenstorrent/{interface_id}.",
            CMD_LINE_COLOR.ENDC,
        )
    if post_reset_failed:
        sys.exit(1)

    print(
        CMD_LINE_COLOR.PURPLE,
        "Re-initializing boards after reset....",
        CMD_LINE_COLOR.ENDC,
    )
    if not reinit:
        print(
            CMD_LINE_COLOR.GREEN,
            "Exiting after galaxy reset without re-initializing chips.",
            CMD_LINE_COLOR.ENDC,
        )
        sys.exit(0)
    try:
        if use_umd:
            options = get_default_discovery_options()
            options.discover_remote_devices = False
            options.wait_on_ethernet_link_training = False
            _, devices = TopologyDiscovery.discover(options=options)
            chip_count = len(devices)
        else:
            os.environ["RUST_BACKTRACE"] = "full"
            chips = detect_chips_with_callback(
                local_only=True, ignore_ethernet=True, print_status=print_status
            )
            chip_count = len(chips)
    except Exception as e:
        print(
            CMD_LINE_COLOR.RED,
            f"Error when re-initializing chips!\n {e}",
            CMD_LINE_COLOR.ENDC,
        )
        sys.exit(1)

    print(
        CMD_LINE_COLOR.GREEN,
        f"Re-initialized {chip_count} boards after reset. Exiting...",
        CMD_LINE_COLOR.ENDC,
    )
    sys.exit(0)
