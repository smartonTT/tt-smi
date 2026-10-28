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
from typing import List

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
from tt_smi import constants
from tt_smi.constants import get_default_discovery_options
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


# Exit codes of glx_6u_trays_reset beyond 0/1 (see GalaxyResetError).
GLX_EXIT_BMC_FAILED = 3  # IPMI step failed after USER_RESET: chips held in reset
GLX_EXIT_CHIPS_MISSING = 4  # reset done, but some chips did not come back on PCIe
GLX_EXIT_NO_HEARTBEAT = 5  # chips back on PCIe, but an ARC heartbeat is stuck
GLX_EXPECTED_CHIPS = 32
GLX_HEARTBEAT_GAP_S = 1.0
BMC_PREFLIGHT_TIMEOUT_S = 15
BMC_RECOVER_TIMEOUT_S = 300
BMC_POLL_INTERVAL_S = 10
GLX_RECOUNT_TIMEOUT_S = 120
GLX_RESCAN_INTERVAL_S = 10
TT_PCI_VENDOR = 0x1E52
WH_PCI_DEVICE = 0x401E
SYSFS_ROOT = "/sys"


class GalaxyResetError(RuntimeError):
    """A Galaxy reset failure that callers must not simply retry; carries the exit code."""

    def __init__(self, message: str, exit_code: int):
        super().__init__(message)
        self.exit_code = exit_code


class GalaxyIpmiParamError(RuntimeError):
    """The BMC rejected op_mode 0x3 with completion code 0xc9 (BMC older than v0.05.22)."""


def _ipmi_output_is_param_out_of_range(output: str) -> bool:
    text = output.lower()
    return "rsp=0xc9" in text or "parameter out of range" in text


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
    allow_legacy_retimer_reset: bool = False,
) -> None:
    """
    Issue the Galaxy UBB IPMI reset with no-retimer op_mode 0x3.

    BMC firmware v0.05.22 or later accepts 0x3. Older BMCs reject it with
    IPMI completion code 0xc9 (Parameter out of range) and do not reset.
    Only that rejection may lead to legacy op_mode 0x0, which also asserts
    RST_RETIMER_PERST_N, and only when allow_legacy_retimer_reset is set.
    Any other failure raises, so the caller can retry op_mode 0x3.
    """
    result = _run_ipmitool(_galaxy_ipmi_cmd(ubb_num, dev_num, op_mode, reset_time))
    failure = _ipmi_failure_detail(result)
    if not failure:
        return

    # Already on the legacy command (or a caller overrode op_mode): do not recurse.
    if op_mode == GLX_IPMI_OP_MODE_WITH_RETIMER:
        raise RuntimeError(f"IPMI command failed: {failure}")

    if not _ipmi_output_is_param_out_of_range(failure):
        raise RuntimeError(f"IPMI command failed: {failure}")

    if not allow_legacy_retimer_reset:
        raise GalaxyIpmiParamError(
            f"This Galaxy is on BMC < v{MINIMUM_BMC_VERSION_NO_RETIMER_RESET}: it rejected "
            f"reset without PCIe retimer (op_mode {GLX_IPMI_OP_MODE_NO_RETIMER}):\n{failure}\n"
            f"Upgrade BMC firmware to v{MINIMUM_BMC_VERSION_NO_RETIMER_RESET} or later, or rerun with "
            f"--allow_legacy_retimer_reset to use op_mode {GLX_IPMI_OP_MODE_WITH_RETIMER}, "
            "which also resets the PCIe retimers (and can drop other devices, such as NVMe, behind them)."
        )

    print(
        f"{CMD_LINE_COLOR.YELLOW}WARNING! "
        f"This Galaxy is on BMC < v{MINIMUM_BMC_VERSION_NO_RETIMER_RESET}, "
        "so reset without PCIe retimer (op_mode 0x3) is not supported. "
        f"Upgrade BMC firmware to v{MINIMUM_BMC_VERSION_NO_RETIMER_RESET} or later to enable it.{CMD_LINE_COLOR.ENDC}\n"
        f"{CMD_LINE_COLOR.BLUE}--allow_legacy_retimer_reset given: falling back to legacy retimer reset "
        f"(op_mode {GLX_IPMI_OP_MODE_WITH_RETIMER}), which also resets the PCIe retimers...{CMD_LINE_COLOR.ENDC}"
    )
    fallback = _run_ipmitool(
        _galaxy_ipmi_cmd(ubb_num, dev_num, GLX_IPMI_OP_MODE_WITH_RETIMER, reset_time)
    )
    fallback_failure = _ipmi_failure_detail(fallback)
    if fallback_failure:
        raise RuntimeError(f"IPMI command failed: {fallback_failure}")


def bmc_answers(timeout_s: int = BMC_PREFLIGHT_TIMEOUT_S) -> str:
    """
    Send in-band Get Device ID (ipmitool mc info). Return "" if the BMC
    answered, else what went wrong.
    """
    cmd = ["sudo", "ipmitool", "mc", "info"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return f"no answer within {timeout_s} s"
    except FileNotFoundError as e:
        raise RuntimeError(
            "Failed to execute ipmitool. Ensure ipmitool is installed and sudo is available."
        ) from e
    return _ipmi_failure_detail(result)


def run_galaxy_ipmi_reset_with_bmc_retry(
    ubb_num: str = "0xF",
    dev_num: str = "0xFF",
    op_mode: str = GLX_IPMI_OP_MODE_NO_RETIMER,
    reset_time: str = "0xF",
    allow_legacy_retimer_reset: bool = False,
    timeout_s: int = BMC_RECOVER_TIMEOUT_S,
    interval_s: int = BMC_POLL_INTERVAL_S,
) -> None:
    """
    Issue the Galaxy IPMI reset. If it fails (other than a 0xc9 rejection),
    poll Get Device ID until the BMC answers again and re-send op_mode 0x3,
    for at most timeout_s. Raises the last failure when time runs out.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            run_galaxy_ipmi_reset(
                ubb_num, dev_num, op_mode, reset_time, allow_legacy_retimer_reset
            )
            return
        except GalaxyIpmiParamError:
            raise
        except RuntimeError as e:
            if "Failed to execute ipmitool" in str(e):
                raise
            last = e
            print(
                CMD_LINE_COLOR.YELLOW,
                f"Warning: IPMI reset failed: {e}\nWaiting up to {timeout_s} s for the BMC to answer, then retrying op_mode {op_mode}...",
                CMD_LINE_COLOR.ENDC,
            )
        while True:
            if time.monotonic() >= deadline:
                raise last
            time.sleep(interval_s)
            detail = bmc_answers()
            if not detail:
                break
            print(f"BMC does not answer Get Device ID yet: {detail}")


def _ubb_bus_ids(device_ids) -> dict:
    """Tray number -> UBB bus id, by chip architecture (any Wormhole chip seen -> WH map)."""
    if WH_PCI_DEVICE in device_ids:
        return constants.WH_UBB_BUS_IDS
    return constants.BH_UBB_BUS_IDS


def tray_of_bdf(bdf: str, ubb_bus_ids: dict = None) -> str:
    """Tray 1-4 of a Galaxy chip from its PCI bus, or "?"."""
    ubb_bus_ids = ubb_bus_ids or constants.BH_UBB_BUS_IDS
    try:
        bus = int(bdf.split(":")[1], 16) & 0xF0
    except (IndexError, ValueError):
        return "?"
    for tray, bus_id in ubb_bus_ids.items():
        if bus_id == bus:
            return str(tray)
    return "?"


def _read_hex(path: str):
    try:
        with open(path) as f:
            return int(f.read().strip(), 16)
    except (OSError, ValueError):
        return None


def _is_bdf(name: str) -> bool:
    parts = name.split(":")
    return len(parts) == 3 and "." in parts[2]


def _pci_dev_dir(bdf: str, sysfs: str) -> str:
    return os.path.join(sysfs, "bus", "pci", "devices", bdf)


def galaxy_expected_chips(sysfs: str = None) -> dict:
    """
    Read-only snapshot of the Galaxy chips the reset must bring back.

    Returns {chip BDF: parent bridge (root port) BDF}. It holds every Tenstorrent
    chip now on PCIe and, while fewer than GLX_EXPECTED_CHIPS are seen, the empty
    sibling ports of the same model next to them (a chip already lost before the
    reset), named by their secondary bus.
    """
    sysfs = sysfs or SYSFS_ROOT
    devices_dir = os.path.join(sysfs, "bus", "pci", "devices")
    try:
        names = sorted(os.listdir(devices_dir))
    except OSError:
        return {}
    chips = {}
    for name in names:
        d = os.path.join(devices_dir, name)
        if _read_hex(os.path.join(d, "vendor")) != TT_PCI_VENDOR:
            continue
        parent = os.path.basename(os.path.dirname(os.path.realpath(d)))
        chips[name] = parent if _is_bdf(parent) else ""
    if len(chips) >= GLX_EXPECTED_CHIPS:
        return chips

    # Chips already off the bus: empty ports of the same model beside known chip ports.
    known_ports = {p for p in chips.values() if p}
    for port in sorted(known_ports):
        port_dir = os.path.realpath(_pci_dev_dir(port, sysfs))
        model = (
            _read_hex(os.path.join(port_dir, "vendor")),
            _read_hex(os.path.join(port_dir, "device")),
        )
        parent_dir = os.path.dirname(port_dir)
        try:
            siblings = sorted(os.listdir(parent_dir))
        except OSError:
            continue
        for sib in siblings:
            if not _is_bdf(sib) or sib in known_ports:
                continue
            sib_dir = os.path.join(parent_dir, sib)
            if (
                _read_hex(os.path.join(sib_dir, "vendor")),
                _read_hex(os.path.join(sib_dir, "device")),
            ) != model:
                continue
            if any(_is_bdf(c) for c in os.listdir(sib_dir)):
                continue
            try:  # decimal in sysfs
                with open(os.path.join(sib_dir, "secondary_bus_number")) as f:
                    sec = int(f.read().strip())
            except (OSError, ValueError):
                continue
            domain = sib.split(":")[0]
            chips.setdefault(f"{domain}:{sec:02x}:00.0", sib)
            known_ports.add(sib)
    return chips


def _chip_present(bdf: str, sysfs: str) -> bool:
    """A chip counts once it is on PCIe, answers config reads and has its driver bound."""
    d = _pci_dev_dir(bdf, sysfs)
    if not os.path.isdir(d):
        return False
    try:
        with open(os.path.join(d, "config"), "rb") as f:
            raw = f.read(2)
        if len(raw) == 2 and int.from_bytes(raw, "little") != TT_PCI_VENDOR:
            return False  # all ones: the chip is not answering
    except OSError:
        if _read_hex(os.path.join(d, "vendor")) != TT_PCI_VENDOR:
            return False
    driver = os.path.join(d, "driver")
    return (
        os.path.islink(driver)
        and os.path.basename(os.readlink(driver)) == "tenstorrent"
    )


def _tt_chips_present(sysfs: str) -> List[str]:
    devices_dir = os.path.join(sysfs, "bus", "pci", "devices")
    try:
        names = sorted(os.listdir(devices_dir))
    except OSError:
        return []
    return [
        n
        for n in names
        if _read_hex(os.path.join(devices_dir, n, "vendor")) == TT_PCI_VENDOR
        and _chip_present(n, sysfs)
    ]


def _rescan_port(port: str, sysfs: str) -> bool:
    """Rescan the bus below one root port. Returns False (and warns) if it could not."""
    path = os.path.join(_pci_dev_dir(port, sysfs), "rescan")
    try:
        with open(path, "w") as f:
            f.write("1")
        return True
    except PermissionError:
        try:
            r = subprocess.run(
                ["sudo", "tee", path],
                input="1",
                capture_output=True,
                text=True,
                timeout=60,
            )
            if r.returncode == 0:
                return True
            detail = (r.stderr or r.stdout).strip() or f"exit code {r.returncode}"
        except (OSError, subprocess.TimeoutExpired) as e:
            detail = str(e)
    except OSError as e:
        detail = str(e)
    print(
        CMD_LINE_COLOR.YELLOW,
        f"Warning: rescan of root port {port} failed: {detail}. Continuing...",
        CMD_LINE_COLOR.ENDC,
    )
    return False


def galaxy_rescan_and_recount(
    expected: dict,
    ubb_bus_ids: dict = None,
    expected_total: int = GLX_EXPECTED_CHIPS,
    timeout_s: int = GLX_RECOUNT_TIMEOUT_S,
    interval_s: int = GLX_RESCAN_INTERVAL_S,
    sysfs: str = None,
) -> tuple:
    """
    After the tray reset: recount the expected chips by BDF, and rescan the root
    port of each missing one, until all are back (and at least expected_total
    chips are on PCIe) or timeout_s passes. A port that fails to rescan does not
    stop the others. Returns (BDFs still missing sorted by tray, chips on PCIe).
    """
    sysfs = sysfs or SYSFS_ROOT
    deadline = time.monotonic() + timeout_s
    rescans = 0
    while True:
        missing = sorted(b for b in expected if not _chip_present(b, sysfs))
        present = len(_tt_chips_present(sysfs))
        print(
            f"Recount after reset: {len(expected) - len(missing)}/{len(expected)} expected chips back, "
            f"{present} on PCIe (want {expected_total}), {rescans} rescan rounds"
        )
        if not missing and present >= expected_total:
            return [], present
        if time.monotonic() >= deadline:
            break
        for port in sorted({expected[b] for b in missing if expected[b]}):
            _rescan_port(port, sysfs)
        rescans += 1
        time.sleep(interval_s)
    return sorted(missing, key=lambda b: (tray_of_bdf(b, ubb_bus_ids), b)), present


def _chip_bdf(chip, use_umd: bool) -> str:
    if use_umd:
        return chip.get_pci_device().get_device_info().pci_bdf
    return chip.get_pci_bdf()


def _read_arc_heartbeat(chip, use_umd: bool) -> int:
    """One ARC heartbeat read, through the telemetry read tt-smi already uses."""
    if use_umd:
        if chip.get_arch() == ARCH.WORMHOLE_B0:
            reader = SmBusArcTelemetryReader(chip)
            tag = wormhole.TelemetryTag.TIMER_HEARTBEAT
        else:
            reader = chip.get_arc_telemetry_reader()
            tag = TelemetryTag.TIMER_HEARTBEAT
        return int(reader.read_entry(tag.value))
    telem = chip.as_bh() or chip.as_wh()
    return int(telem.get_telemetry().timer_heartbeat)


def galaxy_heartbeat_check(
    chips: list, use_umd: bool, ubb_bus_ids: dict = None, gap_s: float = None
) -> list:
    """
    Read each chip's ARC heartbeat twice, gap_s apart. A chip whose value does not
    advance, or whose read fails, is a hardware fault. A chip's failure does not
    stop the others. Returns the BDFs that did not respond, sorted by tray.
    """
    gap_s = GLX_HEARTBEAT_GAP_S if gap_s is None else gap_s
    bdfs = {}
    first = {}
    for i, chip in enumerate(chips):
        try:
            bdf = _chip_bdf(chip, use_umd)
        except Exception:
            bdf = f"unknown-{i}"
        bdfs[i] = bdf
        try:
            first[i] = _read_arc_heartbeat(chip, use_umd)
        except Exception:
            first[i] = None
    time.sleep(gap_s)
    dead = []
    for i, chip in enumerate(chips):
        try:
            second = _read_arc_heartbeat(chip, use_umd)
        except Exception:
            second = None
        if first[i] is None or second is None or second == first[i]:
            dead.append(bdfs[i])
    return sorted(dead, key=lambda b: (tray_of_bdf(b, ubb_bus_ids), b))


def _chip_lines(bdfs, expected: dict, ubb_bus_ids: dict) -> str:
    return "\n".join(
        f"  {b} (tray {tray_of_bdf(b, ubb_bus_ids)}"
        + (f", root port {expected[b]}" if expected.get(b) else "")
        + ")"
        for b in bdfs
    )


def _bdf_of_interface(interface_id: int, sysfs: str = None) -> str:
    """BDF of /dev/tenstorrent/<interface_id>, or "" if sysfs does not say."""
    sysfs = sysfs or SYSFS_ROOT
    link = os.path.join(
        sysfs, "class", "tenstorrent", f"tenstorrent!{interface_id}", "device"
    )
    try:
        bdf = os.path.basename(os.readlink(link))
    except OSError:
        return ""
    return bdf if _is_bdf(bdf) else ""


def _interface_label(interface_id: int, ubb_bus_ids: dict) -> str:
    bdf = _bdf_of_interface(interface_id)
    where = f"{bdf} (tray {tray_of_bdf(bdf, ubb_bus_ids)})" if bdf else "BDF unknown"
    return f"/dev/tenstorrent/{interface_id}, {where}"


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
        # One chip that cannot be opened or errors in the ioctl must not stop
        # the reset of the others: count it as failed and go on.
        try:
            return interface_id, reset_device_ioctl(interface_id, flag)
        except Exception as e:
            print(
                CMD_LINE_COLOR.YELLOW,
                f"Warning: reset ioctl on /dev/tenstorrent/{interface_id} raised: {e}",
                CMD_LINE_COLOR.ENDC,
            )
            return interface_id, False

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


def umd_pci_warm_reset(
    reset_input: SmiDeviceInput,
):
    """
    Reset the PCI devices using UMD warm reset.
    """
    chips = PCIDevice.enumerate_devices_info()
    # check if any of the chips are galaxy
    is_galaxy = False
    for info in chips.values():
        if info.subsystem_id in {0x35, 0x47, 0x202}:
            is_galaxy = True
            break
    if is_galaxy:
        print(
            CMD_LINE_COLOR.YELLOW,
            "CPLD FW v1.16 or higher is required to use tt-smi -r on Galaxy systems.",
            "If tt-smi -r fails, please continue to use tt-smi -glx_reset instead and contact your system administrator to request a CPLD update.",
            CMD_LINE_COLOR.ENDC,
        )
    secondary_bus_reset = should_use_secondary_bus_reset(is_galaxy)

    reset_indices = reset_input.value
    if reset_input.type == SmiDeviceTargetKind.ALL:
        reset_indices = list(chips.keys())
        print(f"Resetting all PCI devices: {reset_indices}")
        WarmReset.warm_reset(reset_indices, secondary_bus_reset=secondary_bus_reset)
        return
    if reset_input.type == SmiDeviceTargetKind.UMD_LOGICAL_ID:
        print(f"Resetting UMD logical IDs: {reset_input.value}")
        WarmReset.warm_reset_chip_id(
            reset_indices, secondary_bus_reset=secondary_bus_reset
        )
        return
    if reset_input.type == SmiDeviceTargetKind.PCI_BDF:
        print(f"Resetting PCI BDFs: {reset_input.value}")
        WarmReset.warm_reset_pci_bdfs(
            reset_indices, secondary_bus_reset=secondary_bus_reset
        )
        return
    if reset_input.type == SmiDeviceTargetKind.DEV_TENSTORRENT_ID:
        print(f"Resetting /dev/tenstorrent IDs: {reset_input.value}")
        WarmReset.warm_reset(reset_indices, secondary_bus_reset=secondary_bus_reset)
        return
    raise ValueError(f"Invalid reset type: {reset_input.type}")


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
        umd_pci_warm_reset(reset_input)
    else:
        luwen_pci_reset(reset_input)

    if reinit:
        print(
            CMD_LINE_COLOR.PURPLE,
            "Re-initializing boards after reset....",
            CMD_LINE_COLOR.ENDC,
        )
        try:
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
        except Exception as e:
            print(
                CMD_LINE_COLOR.RED,
                f"Error when re-initializing chips!\n {e}",
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
    allow_legacy_retimer_reset: bool = False,
):
    """
    Reset the ASICs on the galaxy systems with the following steps:
    1. Check that the BMC answers in-band (Get Device ID); if not, reset nothing
    2. Perform RESET_PCIE_LINK and USER_RESET ioctls on all chips
    3. Reset the trays with IPMI (op_mode 0x3, no PCIe retimer); if that fails,
       wait for the BMC to answer and re-send it, for a bounded time
    4. Wait for 30s, then rescan the root port of each missing chip and recount
       by BDF until all 32 chips are back or a bounded timeout passes
    5. Perform POST_RESET ioctl on all chips that came back
    6. Reinit all chips

    Exit codes: 0 all chips back; 1 other error; GLX_EXIT_BMC_FAILED (3, raised as
    GalaxyResetError) the IPMI step failed after USER_RESET, so the chips are held
    in reset and need a host power cycle; GLX_EXIT_CHIPS_MISSING (4) the reset ran
    but some chips did not come back (printed by BDF and tray).

    Args:
        reinit: Whether to reinitialize the chips after reset.
        ubb_num: The UBB number to reset. 0x0~0xF (bit map)
        dev_num: The device number to reset. 0x0~0xFF(bit map)
        op_mode: IPMI operation mode. 0x3 = reset without PCIe retimer (BMC v0.05.22+).
        reset_time: The reset time to use. resolution 10ms (ex. 0xF => 15 => 150ms)
        print_status: Whether to print out animations while detecting chips.
        use_umd: Whether to enumerate/reinit with UMD or pyluwen.
        allow_legacy_retimer_reset: On a BMC that rejects op_mode 0x3 (0xc9),
            fall back to legacy op_mode 0x0, which also resets the PCIe retimers.
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

    # Read-only: the chips (and the ports of chips already lost) to bring back.
    expected = galaxy_expected_chips()
    ubb_bus_ids = _ubb_bus_ids(
        {
            _read_hex(os.path.join(_pci_dev_dir(b, SYSFS_ROOT), "device"))
            for b in expected
        }
    )

    # Do not put any chip in reset unless the BMC can finish the reset.
    preflight = bmc_answers(BMC_PREFLIGHT_TIMEOUT_S)
    if preflight:
        raise RuntimeError(
            "BMC does not answer in-band Get Device ID (ipmitool mc info): "
            f"{preflight}\nNo chip was reset."
        )

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
                f"Warning: Secondary bus reset not completed for device {_interface_label(interface_id, ubb_bus_ids)}. Continuing...",
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
            f"Warning: USER_RESET did not complete for device {_interface_label(interface_id, ubb_bus_ids)}. Continuing...",
            CMD_LINE_COLOR.ENDC,
        )

    # IPMI reset without PCIe retimer (BMC v0.05.22+). op_mode 0x0 asserts
    # RST_RETIMER_PERST_N and can drop NVMe: only on a 0xc9 rejection, by opt-in.
    try:
        run_galaxy_ipmi_reset_with_bmc_retry(
            ubb_num, dev_num, op_mode, reset_time, allow_legacy_retimer_reset
        )
    except RuntimeError as e:
        held = sorted(expected, key=lambda b: (tray_of_bdf(b, ubb_bus_ids), b))
        msg = (
            f"IPMI tray reset failed after USER_RESET: {e}\n"
            f"These {len(held)} chips were put in reset and are held there; "
            "they need a host power cycle (do not retry the reset):\n"
            + _chip_lines(held, expected, ubb_bus_ids)
        )
        print(CMD_LINE_COLOR.RED, msg, CMD_LINE_COLOR.ENDC)
        raise GalaxyResetError(msg, GLX_EXIT_BMC_FAILED) from e
    timed_wait(30)

    # Rescan the root port of each chip still missing and recount by BDF.
    missing, on_pci = galaxy_rescan_and_recount(expected, ubb_bus_ids)

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
            f"Error: POST_RESET failed for device {_interface_label(interface_id, ubb_bus_ids)}.",
            CMD_LINE_COLOR.ENDC,
        )

    def report_and_exit(code: int):
        # Missing chips are reported last, so every chip that came back still gets
        # its POST_RESET and reinit first.
        if missing or on_pci < GLX_EXPECTED_CHIPS:
            lines = [
                f"Error: {on_pci}/{GLX_EXPECTED_CHIPS} chips on PCIe after reset and rescan."
            ]
            if missing:
                lines.append(
                    "Still missing:\n" + _chip_lines(missing, expected, ubb_bus_ids)
                )
            unnamed = GLX_EXPECTED_CHIPS - on_pci - len(missing)
            if unnamed > 0:
                lines.append(
                    f"{unnamed} more chips missing with no known BDF (lost before this reset)."
                )
            print(CMD_LINE_COLOR.RED, "\n".join(lines), CMD_LINE_COLOR.ENDC)
            sys.exit(GLX_EXIT_CHIPS_MISSING)
        if post_reset_failed:
            sys.exit(1)
        sys.exit(code)

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
        report_and_exit(0)
    try:
        if use_umd:
            options = get_default_discovery_options()
            options.discover_remote_devices = False
            options.wait_on_ethernet_link_training = False
            _, devices = TopologyDiscovery.discover(options=options)
            chip_count = len(devices)
            hb_chips = list(devices.values())
        else:
            os.environ["RUST_BACKTRACE"] = "full"
            chips = detect_chips_with_callback(
                local_only=True, ignore_ethernet=True, print_status=print_status
            )
            chip_count = len(chips)
            hb_chips = list(chips)
    except Exception as e:
        print(
            CMD_LINE_COLOR.RED,
            f"Error when re-initializing chips!\n {e}",
            CMD_LINE_COLOR.ENDC,
        )
        report_and_exit(1)

    print(
        CMD_LINE_COLOR.GREEN,
        f"Re-initialized {chip_count} boards after reset. Exiting...",
        CMD_LINE_COLOR.ENDC,
    )
    try:
        no_heartbeat = galaxy_heartbeat_check(hb_chips, use_umd, ubb_bus_ids)
    except Exception as e:
        print(CMD_LINE_COLOR.RED, f"Heartbeat check failed: {e}", CMD_LINE_COLOR.ENDC)
        report_and_exit(1)
    if no_heartbeat:
        print(
            CMD_LINE_COLOR.RED,
            "Hardware fault: ARC heartbeat did not advance (or could not be read) on:\n"
            + _chip_lines(no_heartbeat, expected, ubb_bus_ids),
            CMD_LINE_COLOR.ENDC,
        )
        report_and_exit(GLX_EXIT_NO_HEARTBEAT)
    report_and_exit(0)
