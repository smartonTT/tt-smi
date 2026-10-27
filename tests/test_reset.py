# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import itertools
import json
import subprocess
import sys
import pytest

from typing import List, Tuple
from unittest.mock import MagicMock, patch

from pyluwen import pci_scan
from tt_umd import PCIDevice
from tt_smi.backend import TTSMIBackend
from tt_smi.utils import get_dev_id_from_bdf, IoctlResetFlags
from tt_smi.reset import (
    _sysfs_write,
    tray_of_bdf,
    pci_board_reset,
    glx_6u_trays_reset,
    run_galaxy_ipmi_reset,
    GLX_IPMI_OP_MODE_HOLD,
    GLX_IPMI_OP_MODE_NO_RETIMER,
    GLX_IPMI_OP_MODE_RELEASE,
    TRAY_HOLD_RELEASE_WAIT_S,
    IPMI_RECOVER_TIMEOUT_S,
    GLX_IPMI_OP_MODE_WITH_RETIMER,
    BH_GALAXY_INBOX_MESH,
    check_inbox_links_after_repower,
    _links_down,
)
from tt_smi.device_input import (
    classify_single_input,
    parse_smi_device_input,
    SmiDeviceInput,
    SmiDeviceTargetKind,
)

NUM_RESETS_STRESS_TEST = 10

BMC_C9_STDERR = (
    "Unable to send RAW command (channel=0x0 netfn=0x30 lun=0x0 cmd=0x8b rsp=0xc9): "
    "Parameter out of range"
)
# ipmitool when the BMC's in-band (KCS) interface stops answering (OM-E4-GBH02, task #3981).
KCS_SILENT_STDERR = (
    "No data available\n"
    "Unable to send RAW command (channel=0x0 netfn=0x30 lun=0x0 cmd=0x8b)"
)


class TestGalaxyIpmiReset:
    """Unit tests for the no-retimer Galaxy IPMI reset (no hardware)."""

    def test_issues_op_mode_no_retimer(self):
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        with patch("tt_smi.reset.subprocess.run", return_value=mock_result) as mock_run:
            run_galaxy_ipmi_reset()
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == [
            "sudo",
            "ipmitool",
            "raw",
            "0x30",
            "0x8B",
            "0xF",
            "0xFF",
            GLX_IPMI_OP_MODE_NO_RETIMER,
            "0xF",
        ]

    def test_c9_falls_back_to_legacy_retimer_reset(self):
        fail_new = MagicMock(returncode=1, stdout="", stderr=BMC_C9_STDERR)
        ok_old = MagicMock(returncode=0, stdout="", stderr="")
        with patch(
            "tt_smi.reset.subprocess.run", side_effect=[fail_new, ok_old]
        ) as mock_run:
            run_galaxy_ipmi_reset()
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0][0][0][7] == GLX_IPMI_OP_MODE_NO_RETIMER
        assert mock_run.call_args_list[1][0][0][7] == GLX_IPMI_OP_MODE_WITH_RETIMER

    def test_other_ipmi_failure_falls_back_to_legacy_retimer_reset(self):
        fail_new = MagicMock(returncode=1, stdout="", stderr="Could not open device")
        ok_old = MagicMock(returncode=0, stdout="", stderr="")
        with patch(
            "tt_smi.reset.subprocess.run", side_effect=[fail_new, ok_old]
        ) as mock_run:
            run_galaxy_ipmi_reset()
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[1][0][0][7] == GLX_IPMI_OP_MODE_WITH_RETIMER

    def test_fallback_failure_raises_runtime_error(self):
        fail_new = MagicMock(returncode=1, stdout="", stderr=BMC_C9_STDERR)
        fail_old = MagicMock(returncode=1, stdout="", stderr="Could not open device")
        with patch("tt_smi.reset.subprocess.run", side_effect=[fail_new, fail_old]):
            with pytest.raises(RuntimeError, match="IPMI command failed"):
                run_galaxy_ipmi_reset()

    def test_missing_ipmitool_raises_runtime_error(self):
        with patch(
            "tt_smi.reset.subprocess.run", side_effect=FileNotFoundError("sudo")
        ):
            with pytest.raises(RuntimeError, match="ipmitool"):
                run_galaxy_ipmi_reset()

    def test_rejects_legacy_retimer_op_mode(self):
        with pytest.raises(SystemExit) as exc:
            glx_6u_trays_reset(op_mode="0x0")
        assert exc.value.code == 1


# Fake BH Galaxy: tray 1 buses 0x01-0x08, tray 2 0x41-0x48, tray 3 0xC1-0xC8, tray 4 0x81-0x88.
GALAXY_BDFS = [
    f"0000:{base + n:02x}:00.0"
    for base in (0x00, 0x40, 0xC0, 0x80)
    for n in range(1, 9)
]


def _link(a, ca, b, cb):
    return tuple(sorted(((a, ca), (b, cb))))


def _bdf(base, n):
    return f"0000:{base + n:02x}:00.0"


# The in-box ETH links tt-umd discovery read on OM-F3-GBH02 after a glx_reset (task #4008,
# results/t4008/post/disc-post.txt, 120/120), as "bus:ch-bus:ch".
F3_GBH02_LINKS_READ = """
01:0-41:0 01:1-41:1 01:4-02:2 01:5-02:3 01:6-05:0 01:7-05:1 02:0-42:0 02:1-42:1
02:4-03:2 02:5-03:3 02:6-06:0 02:7-06:1 03:0-43:0 03:1-43:1 03:4-04:2 03:5-04:3
03:6-07:0 03:7-07:1 04:0-44:0 04:1-44:1 04:4-c4:4 04:5-c4:5 04:6-08:0 04:7-08:1
05:4-06:2 05:5-06:3 05:6-45:6 05:7-45:7 06:4-07:2 06:5-07:3 06:6-46:6 06:7-46:7
07:4-08:2 07:5-08:3 07:6-47:6 07:7-47:7 08:4-c8:4 08:5-c8:5 08:6-48:6 08:7-48:7
41:4-42:2 41:5-42:3 41:6-45:0 41:7-45:1 42:4-43:2 42:5-43:3 42:6-46:0 42:7-46:1
43:4-44:2 43:5-44:3 43:6-47:0 43:7-47:1 44:4-84:4 44:5-84:5 44:6-48:0 44:7-48:1
45:4-46:2 45:5-46:3 46:4-47:2 46:5-47:3 47:4-48:2 47:5-48:3 48:4-88:4 48:5-88:5
81:0-c1:0 81:1-c1:1 81:4-82:2 81:5-82:3 81:6-85:0 81:7-85:1 82:0-c2:0 82:1-c2:1
82:4-83:2 82:5-83:3 82:6-86:0 82:7-86:1 83:0-c3:0 83:1-c3:1 83:4-84:2 83:5-84:3
83:6-87:0 83:7-87:1 84:0-c4:0 84:1-c4:1 84:6-88:0 84:7-88:1 85:4-86:2 85:5-86:3
85:6-c5:6 85:7-c5:7 86:4-87:2 86:5-87:3 86:6-c6:6 86:7-c6:7 87:4-88:2 87:5-88:3
87:6-c7:6 87:7-c7:7 88:6-c8:6 88:7-c8:7 c1:4-c2:2 c1:5-c2:3 c1:6-c5:0 c1:7-c5:1
c2:4-c3:2 c2:5-c3:3 c2:6-c6:0 c2:7-c6:1 c3:4-c4:2 c3:5-c4:3 c3:6-c7:0 c3:7-c7:1
c4:6-c8:0 c4:7-c8:1 c5:4-c6:2 c5:5-c6:3 c6:4-c7:2 c6:5-c7:3 c7:4-c8:2 c7:5-c8:3
"""
GALAXY_LINKS = {
    _link(f"0000:{a[:2]}:00.0", int(a[3:]), f"0000:{b[:2]}:00.0", int(b[3:]))
    for a, b in (w.split("-") for w in F3_GBH02_LINKS_READ.split())
}
# OM-F3-GBH02, task #4008: tray 2 re-powered alone left 05-45, 06-46, 07-47 ch 6/7 down.
F3_GBH02_LINKS_DOWN = {
    _link(_bdf(0x00, n), 6 + k, _bdf(0x40, n), 6 + k) for n in (5, 6, 7) for k in (0, 1)
}


class FakeGalaxy:
    """A host's chips for ``tt-smi -r`` unit tests: fake sysfs, enumeration, IPMI and rescan.

    ``states``: bdf -> ok, marker, ones or gone (default ok). ``heal``: bdf -> how a chip
    that is not ok comes back: "tray" (its tray re-powered), "all" (all four trays
    re-powered), "rescan" (UMD PCI rescan), "sysfs" (sysfs PCI rescan), "post" (POST_RESET
    sent to it), "hold" (its tray held in reset, then re-powered) or "never" (default).
    A hold (op 0x1) makes the tray's chips read all ones until its next re-power.
    A release (op 0x2) gives held chips back their state from before the hold.
    ``hold_needs_release``: as on OM-E4-GBH02 (task #3850), a hold not preceded by a
    release of its tray (since its last hold) leaves the tray's chips on PCI.
    ``events``: ("ipmi", ubb, op) and ("sleep", seconds) in the order they happened.
    ``drop_on_repower``: chips lost
    when their tray is re-powered. ``knock_others``: the first per-tray re-power leaves the
    chips of every other tray reading all ones until their own tray is re-powered.
    ``lose_at_reinit``: chips that read all ones at the first re-init (healed as
    ``lose_at_reinit_heal`` says, default their tray re-powered); with it, re-init fails while
    any chip is not ok. ``post_reset_fail``: chips whose POST_RESET ioctl fails.
    ``eth_drop``: the ``eth_fragile`` in-box links go down when a re-power pulse covers the
    tray of one end only, and come back when a pulse covers both ends' trays ("both"), only on
    an all-tray pulse ("all") or never ("never"). ``eth_unreadable``: the link read (tt-umd
    discovery in a child process) fails "after_first" read or "always". A link reads only
    while both its chips are up. ``link_reads`` counts the reads.
    """

    lose_at_reinit_heal = "tray"
    reject_hold = False
    reject_pulse_after_hold = False
    hold_leaves_chips_up = False
    hold_needs_release = False
    reject_release = False
    # The next N re-power pulses after a hold fail with KCS_SILENT_STDERR. With
    # ``kcs_pulse_acts`` the pulse still re-powers the tray (only the reply was lost).
    kcs_silent_pulses = 0
    kcs_pulse_acts = False
    # The KCS silence also hits pulses with no hold before them (natural loss, #4132).
    kcs_silent_without_hold = False

    def __init__(
        self,
        tmp_path,
        states=None,
        heal=None,
        bdfs=None,
        subsystem_id=0x47,
        absent=(),
        drop_on_repower=(),
        reject_per_tray=False,
        reject_all=False,
        knock_others=False,
        lose_at_reinit=(),
        reinit_error=False,
        post_reset_fail=(),
        bar_ones=None,
        eth_drop=None,
        eth_fragile=F3_GBH02_LINKS_DOWN,
        eth_unreadable=None,
    ):
        self.released = set()
        self.eth_links = set(GALAXY_LINKS)
        self.eth_down = set()
        self.eth_drop = eth_drop
        self.eth_fragile = set(eth_fragile)
        self.eth_unreadable = eth_unreadable
        self.link_reads = 0
        # bdf -> "tray", "all" or "never": tt-umd discovery reads 0xffffffff from the chip (its config
        # space reads fine) until its tray, or all four trays, are re-powered (task #3973).
        self.bar_ones = dict(bar_ones or {})
        self.events = []
        self.root = tmp_path / "devices"
        self.root.mkdir()
        self.rescan_file = tmp_path / "rescan"
        self.rescan_file.write_text("")
        self.bdfs = list(bdfs or GALAXY_BDFS)
        self.subsystem_id = subsystem_id
        self.heal = dict(heal or {})
        self.drop_on_repower = set(drop_on_repower)
        self.reject_per_tray = reject_per_tray
        self.reject_all = reject_all
        self.knock_others = knock_others
        self.lose_at_reinit = list(lose_at_reinit)
        self.discovers = 0
        self.reinit_error = reinit_error
        self.post_reset_fail = set(post_reset_fail)
        self.ipmi = []
        self.held = {}
        self.ioctls = []
        self.state = {b: "ok" for b in self.bdfs}
        self.state.update(states or {})
        self.enumerated = [b for b in self.bdfs if b not in absent]
        for b in absent:
            self.state[b] = "gone"
        for bdf in self.bdfs:
            self._write(bdf)

    def _write(self, bdf):
        d = self.root / bdf
        state = self.state[bdf]
        if state == "gone":
            if d.exists():
                (d / "config").unlink()
                d.rmdir()
            return
        d.mkdir(exist_ok=True)
        command = 0x46 if state == "marker" else 0x06
        data = b"\x3e\x1e\x00\xb1" + command.to_bytes(2, "little") + b"\x10\x00"
        if state == "ones":
            data = b"\xff" * 8
        (d / "config").write_bytes(data)

    def _set(self, bdf, state):
        self.state[bdf] = state
        self._write(bdf)

    def _heal(self, how):
        for bdf, mode in self.heal.items():
            if mode in how and self.state[bdf] != "ok":
                self._set(bdf, "ok")
                self.heal[bdf] = "done"

    def tray(self, bdf):
        return {0x00: 1, 0x40: 2, 0xC0: 3, 0x80: 4}[int(bdf.split(":")[1], 16) & 0xF0]

    def enumerate_devices_info(self):
        if self.rescan_file.read_text() == "1":
            self._heal(("sysfs",))
        bdfs = (
            self.enumerated
            if self.enumerated is not None
            else [b for b in self.bdfs if self.state[b] not in ("gone",)]
        )
        self.enumerated = (
            None  # Only the first enumeration (before the reset) may hide chips.
        )
        return {
            self.bdfs.index(b): MagicMock(
                pci_bdf=b, subsystem_id=self.subsystem_id, device_id=0xB140
            )
            for b in bdfs
        }

    # "ok", "fail" or "timeout"; a list gives one answer per call, its last one repeating.
    mc_info = "ok"

    def read_links(self):
        self.link_reads += 1
        fail = {"after_first": self.link_reads > 1, "always": True}.get(
            self.eth_unreadable, False
        )
        if fail:
            return MagicMock(returncode=1, stdout="", stderr="discovery failed")
        ends = [
            [a, ca, b, cb]
            for (a, ca), (b, cb) in sorted(self.eth_links - self.eth_down)
            if self.state[a] == self.state[b] == "ok"
        ]
        return MagicMock(
            returncode=0, stdout="LINKS " + json.dumps(ends) + "\n", stderr=""
        )

    def _pulse_links(self, trays, bitmap):
        if not self.eth_drop:
            return
        for link in self.eth_links:
            ends = {self.tray(b) for b, _ in link}
            if link in self.eth_down and ends <= trays:
                if self.eth_drop == "both" or (
                    self.eth_drop == "all" and bitmap == 0xF
                ):
                    self.eth_down.discard(link)
            elif link in self.eth_fragile and len(ends & trays) == 1:
                self.eth_down.add(link)

    def run(self, cmd, **kwargs):
        if cmd[:2] == [sys.executable, "-c"]:
            return self.read_links()
        if cmd[:4] == ["sudo", "ipmitool", "mc", "info"]:
            self.mc_calls = getattr(self, "mc_calls", []) + [kwargs.get("timeout")]
            answer = self.mc_info
            if isinstance(answer, list):
                answer = answer[min(len(self.mc_calls), len(answer)) - 1]
            if answer == "timeout":
                raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
            return MagicMock(
                returncode=0 if answer == "ok" else 1, stdout="", stderr=""
            )
        if cmd[:3] == ["sudo", "ipmitool", "raw"]:
            bitmap = int(cmd[5], 16)
            self.ipmi.append((cmd[5], cmd[7]))
            self.events.append(("ipmi", cmd[5], cmd[7]))
            trays = {t for t in (1, 2, 3, 4) if bitmap & (1 << (t - 1))}
            if cmd[7] == GLX_IPMI_OP_MODE_RELEASE:
                if self.reject_release:
                    return MagicMock(returncode=1, stdout="", stderr=BMC_C9_STDERR)
                for bdf in [b for b in self.held if self.tray(b) in trays]:
                    self._set(bdf, self.held.pop(bdf))
                self.released |= trays
                return MagicMock(returncode=0, stdout="", stderr="")
            silent = False
            if (
                self.kcs_silent_pulses
                and cmd[7] == GLX_IPMI_OP_MODE_NO_RETIMER
                and (
                    self.kcs_silent_without_hold
                    or any(op == GLX_IPMI_OP_MODE_HOLD for _, op in self.ipmi[:-1])
                )
            ):
                self.kcs_silent_pulses -= 1
                if not self.kcs_pulse_acts:
                    return MagicMock(returncode=1, stdout="", stderr=KCS_SILENT_STDERR)
                silent = True
            if self.reject_pulse_after_hold and cmd[7] != GLX_IPMI_OP_MODE_HOLD:
                if any(op == GLX_IPMI_OP_MODE_HOLD for _, op in self.ipmi[:-1]):
                    return MagicMock(returncode=1, stdout="", stderr=BMC_C9_STDERR)
            if self.reject_all or (self.reject_per_tray and bitmap != 0xF):
                return MagicMock(returncode=1, stdout="", stderr=BMC_C9_STDERR)
            if cmd[7] == GLX_IPMI_OP_MODE_HOLD:
                if self.reject_hold:
                    return MagicMock(returncode=1, stdout="", stderr=BMC_C9_STDERR)
                unreleased = self.hold_needs_release and not trays <= self.released
                self.released -= trays
                for bdf in self.bdfs:
                    if self.hold_leaves_chips_up or unreleased:
                        break
                    if self.tray(bdf) in trays and self.state[bdf] != "gone":
                        self.held[bdf] = self.state[bdf]
                        self._set(bdf, "ones")
                return MagicMock(returncode=0, stdout="", stderr="")
            self._pulse_links(trays, bitmap)
            # Releasing a held tray: a "hold" chip resets, the others read as before the hold.
            for bdf in [b for b in self.held if self.tray(b) in trays]:
                before = self.held.pop(bdf)
                if self.heal.get(bdf) == "hold":
                    before = "ok"
                    self.heal[bdf] = "done"
                self._set(bdf, before)
            for bdf in self.drop_on_repower:
                if self.tray(bdf) in trays:
                    self._set(bdf, "gone")
            if self.knock_others and bitmap != 0xF:
                self.knock_others = False
                for bdf in self.bdfs:
                    if self.tray(bdf) not in trays and self.state[bdf] == "ok":
                        self._set(bdf, "ones")
                        self.heal[bdf] = "tray"
            self._heal(("all",) if bitmap == 0xF else ())
            for bdf, mode in list(self.bar_ones.items()):
                if mode == "never":
                    continue
                if bitmap == 0xF or (mode == "tray" and self.tray(bdf) in trays):
                    del self.bar_ones[bdf]
            for bdf, mode in self.heal.items():
                if mode == "tray" and self.tray(bdf) in trays:
                    self._set(bdf, "ok")
                    self.heal[bdf] = "done"
            if silent:
                return MagicMock(returncode=1, stdout="", stderr=KCS_SILENT_STDERR)
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    def rescan_pci_bdfs(self, bdfs, timeout_s=10.0):
        self._heal(("rescan",))
        return [b for b in bdfs if self.state.get(b) != "ok"]

    def discover(self, *args, **kwargs):
        self.discovers += 1
        if self.discovers == 1:
            for bdf in self.lose_at_reinit:
                self._set(bdf, "ones")
                self.heal[bdf] = self.lose_at_reinit_heal
        bad = [b for b in self.bdfs if self.state[b] != "ok"]
        if self.reinit_error:
            raise RuntimeError("Ethernet training timed out")
        for bdf in self.bar_ones:
            raise RuntimeError(
                f"Read 0xffffffff over PCIe ID {self.bdfs.index(bdf)}: the board should be reset"
            )
        if bad:
            raise RuntimeError(f"Read 0xffffffff from PCIE: {bad[0]}")

    def ioctl(self, ids, flag):
        self.ioctls.append((flag, sorted(ids)))
        if flag != IoctlResetFlags.POST_RESET:
            return []
        for i in ids:
            bdf = self.bdfs[i]
            if self.heal.get(bdf) == "post" and bdf not in self.post_reset_fail:
                self._set(bdf, "ok")
                self.heal[bdf] = "done"
        # Like the KMD: POST_RESET fails while the reset marker is still set.
        return [
            i
            for i in ids
            if self.bdfs[i] in self.post_reset_fail
            or self.state[self.bdfs[i]] == "marker"
        ]

    def ioctl_ids(self, flag):
        return [ids for f, ids in self.ioctls if f == flag]

    def reset(
        self,
        ok=True,
        missing=(),
        reset_input=None,
        reinit=True,
        with_result=True,
        rescan=True,
        eth_train_skip=False,
    ):
        """Run pci_board_reset(use_umd=True). Returns (exit code or None, reinit ran, WarmReset mock)."""
        reset_input = reset_input or SmiDeviceInput(SmiDeviceTargetKind.ALL, [])
        methods = ["warm_reset", "warm_reset_pci_bdfs", "warm_reset_chip_id"]
        if with_result:
            methods.append("warm_reset_with_result")
        if rescan:
            methods.append("rescan_pci_bdfs")
        warm = MagicMock(spec=methods)
        warm.warm_reset.return_value = ok
        warm.warm_reset_pci_bdfs.return_value = ok
        warm.warm_reset_chip_id.return_value = ok
        if with_result:
            warm.warm_reset_with_result.return_value = MagicMock(
                success=ok, missing_pci_bdfs=list(missing)
            )
        if rescan:
            warm.rescan_pci_bdfs.side_effect = self.rescan_pci_bdfs
        clock = MagicMock()
        clock.monotonic.side_effect = itertools.count(0, 10)
        clock.sleep.side_effect = lambda s: self.events.append(("sleep", s))
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("tt_smi.reset.SYSFS_PCI_DEVICES", str(self.root)))
            stack.enter_context(
                patch("tt_smi.reset.SYSFS_PCI_RESCAN", str(self.rescan_file))
            )
            stack.enter_context(
                patch(
                    "tt_smi.reset.PCIDevice.enumerate_devices_info",
                    side_effect=self.enumerate_devices_info,
                )
            )
            stack.enter_context(patch("tt_smi.reset.WarmReset", warm))
            stack.enter_context(
                patch("tt_smi.reset.subprocess.run", side_effect=self.run)
            )
            stack.enter_context(
                patch(
                    "tt_smi.reset.parallel_reset_device_ioctl", side_effect=self.ioctl
                )
            )
            stack.enter_context(patch("tt_smi.reset.timed_wait"))
            stack.enter_context(patch("tt_smi.reset.time", clock))
            stack.enter_context(
                patch("tt_smi.reset.should_use_secondary_bus_reset", return_value=True)
            )
            topo = stack.enter_context(patch("tt_smi.reset.TopologyDiscovery"))
            stack.enter_context(patch("tt_smi.reset.get_default_discovery_options"))
            if self.lose_at_reinit or self.reinit_error or self.bar_ones:
                topo.discover.side_effect = self.discover
            try:
                pci_board_reset(
                    reset_input,
                    reinit=reinit,
                    use_umd=True,
                    eth_train_skip=eth_train_skip,
                )
                code = None
            except SystemExit as e:
                code = e.code
        return code, topo.discover.called, warm


class TestUmdWarmResetResult:
    """``tt-smi -r`` checks UMD warm_reset's result and names chips that did not come back (no hardware)."""

    BDFS = ["0000:02:00.0", "0000:03:00.0", "0000:04:00.0"]

    def _reset(self, tmp_path, states, ok, reset_input=None, reinit=True, heal=None):
        host = FakeGalaxy(tmp_path, states=states, heal=heal)
        return host.reset(
            ok=ok, reset_input=reset_input, reinit=reinit, with_result=False
        )

    def test_failed_reset_names_chips_and_exits_nonzero(self, tmp_path, capsys):
        states = {self.BDFS[0]: "ok", self.BDFS[1]: "marker", self.BDFS[2]: "ones"}
        code, reinit_ran, _ = self._reset(
            tmp_path, states, ok=False, heal={self.BDFS[1]: "tray"}
        )
        err = capsys.readouterr().err
        assert code == 1
        assert not reinit_ran
        assert (
            "Chip 0000:03:00.0 (tray 1) did not come back from reset: reset marker still set"
            in err
        )
        assert (
            "Chip 0000:04:00.0 (tray 1) did not come back from reset: config space reads all ones"
            in err
        )
        assert "0000:02:00.0 (tray 1) did not come back" not in err
        # The tray re-power resets 03:00.0; 04:00.0 stays dead.
        assert (
            "Hardware fault: chip 0000:04:00.0 (tray 1) still missing after tray re-power"
            in err
        )
        assert "1 chip(s) did not come back: 0000:04:00.0" in err

    def test_chip_gone_from_pci_is_named(self, tmp_path, capsys):
        states = {self.BDFS[0]: "ok", self.BDFS[1]: "gone", self.BDFS[2]: "ok"}
        code, _, _ = self._reset(tmp_path, states, ok=False)
        err = capsys.readouterr().err
        assert code == 1
        assert (
            "Chip 0000:03:00.0 (tray 1) did not come back from reset: gone from PCI"
            in err
        )

    def test_failed_reset_with_no_chip_named_still_fails(self, tmp_path, capsys):
        code, reinit_ran, _ = self._reset(
            tmp_path, {b: "ok" for b in self.BDFS}, ok=False
        )
        err = capsys.readouterr().err
        assert code == 1
        assert not reinit_ran
        assert "UMD warm reset reported failure" in err
        assert "no chip reads back as failed" in err
        assert "every chip reads back" not in err

    def test_success_reinits(self, tmp_path):
        code, reinit_ran, warm = self._reset(
            tmp_path, {b: "ok" for b in self.BDFS}, ok=True
        )
        assert code is None
        assert reinit_ran
        warm.warm_reset.assert_called_once_with(
            list(range(32)), secondary_bus_reset=True
        )

    def test_success_ignores_marker_restored_by_kmd(self, tmp_path):
        # After POST_RESET the KMD restores saved config, which may carry bit 6.
        states = {b: "marker" for b in self.BDFS}
        code, reinit_ran, _ = self._reset(tmp_path, states, ok=True)
        assert code is None
        assert reinit_ran

    def test_success_but_chip_gone_fails(self, tmp_path, capsys):
        states = {self.BDFS[0]: "ok", self.BDFS[1]: "ok", self.BDFS[2]: "gone"}
        code, reinit_ran, _ = self._reset(tmp_path, states, ok=True)
        assert code == 1
        assert not reinit_ran
        assert "Chip 0000:04:00.0 (tray 1) did not come back" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "kind,value,method",
        [
            (SmiDeviceTargetKind.PCI_BDF, ["0000:03:00.0"], "warm_reset_pci_bdfs"),
            (SmiDeviceTargetKind.UMD_LOGICAL_ID, [1], "warm_reset_chip_id"),
            (SmiDeviceTargetKind.DEV_TENSTORRENT_ID, [1], "warm_reset"),
        ],
    )
    def test_every_target_kind_checks_result(self, tmp_path, kind, value, method):
        states = {self.BDFS[0]: "ok", self.BDFS[1]: "ones", self.BDFS[2]: "ok"}
        code, _, warm = self._reset(
            tmp_path, states, ok=False, reset_input=SmiDeviceInput(kind, value)
        )
        assert code == 1
        getattr(warm, method).assert_called_once_with(value, secondary_bus_reset=True)

    def test_no_reinit_still_fails(self, tmp_path):
        states = {self.BDFS[0]: "ones", self.BDFS[1]: "ok", self.BDFS[2]: "ok"}
        code, _, _ = self._reset(tmp_path, states, ok=False, reinit=False)
        assert code == 1


class TestFullRecoveryReset:
    """``tt-smi -r`` re-powers the tray of a chip that did not come back, rescans and recounts (no hardware)."""

    LOST = "0000:47:00.0"  # tray 2
    TRAY2 = [b for b in GALAXY_BDFS if b.startswith("0000:4")]

    def test_bdf_to_tray(self):
        assert tray_of_bdf("0000:02:00.0") == 1
        assert tray_of_bdf("0000:47:00.0") == 2
        assert tray_of_bdf("0000:c1:00.0") == 3
        assert tray_of_bdf("0000:85:00.0") == 4
        assert tray_of_bdf("0000:02:00.0", wormhole=True) == 3
        assert tray_of_bdf("0000:c1:00.0", wormhole=True) == 1
        assert tray_of_bdf("0000:21:00.0") is None
        assert tray_of_bdf("garbage") is None

    def test_resets_every_chip_in_one_call(self, tmp_path):
        host = FakeGalaxy(tmp_path)
        code, reinit_ran, warm = host.reset(ok=True)
        assert code is None and reinit_ran
        warm.warm_reset_with_result.assert_called_once_with(
            list(range(32)), secondary_bus_reset=True
        )
        warm.warm_reset.assert_not_called()
        assert host.ipmi == []

    def test_tray_repower_brings_back_missing_chip(self, tmp_path, capsys):
        host = FakeGalaxy(
            tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "tray"}
        )
        code, reinit_ran, warm = host.reset(ok=False, missing=[self.LOST])
        out = capsys.readouterr()
        assert code is None and reinit_ran
        # Only tray 2 (UBB bit 1) is re-powered, without the retimer reset.
        assert host.ipmi == [("0x2", GLX_IPMI_OP_MODE_NO_RETIMER)]
        tray2_ids = sorted(GALAXY_BDFS.index(b) for b in self.TRAY2)
        assert host.ioctl_ids(IoctlResetFlags.USER_RESET) == [
            [i for i in tray2_ids if GALAXY_BDFS[i] != self.LOST]
        ]
        assert host.ioctl_ids(IoctlResetFlags.POST_RESET) == [tray2_ids]
        warm.rescan_pci_bdfs.assert_called_once_with([self.LOST], timeout_s=10.0)
        assert f"Chip {self.LOST} (tray 2) did not come back from reset" in out.err
        assert "Recount after round 1: 32/32 chips" in out.out
        assert "Recovered" in out.out
        assert "UBB" not in out.out + out.err

    def test_ipmi_answer_allows_repower(self, tmp_path):
        host = FakeGalaxy(
            tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "tray"}
        )
        code, _, _ = host.reset(ok=False, missing=[self.LOST])
        assert code is None
        assert host.mc_calls == [10]
        assert host.ipmi == [("0x2", GLX_IPMI_OP_MODE_NO_RETIMER)]

    @pytest.mark.parametrize("mode", ["fail", "timeout"])
    def test_no_ipmi_answer_touches_nothing(self, tmp_path, capsys, mode):
        host = FakeGalaxy(
            tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "tray"}
        )
        host.mc_info = mode
        code, reinit_ran, _ = host.reset(ok=False, missing=[self.LOST])
        out = capsys.readouterr()
        assert code == 1 and not reinit_ran
        assert host.ipmi == []
        assert host.ioctl_ids(IoctlResetFlags.RESET_PCIE_LINK) == []
        assert host.ioctl_ids(IoctlResetFlags.USER_RESET) == []
        assert f"Hardware fault: chip {self.LOST} (tray 2)" in out.err
        assert "UBB" not in out.out + out.err

    def assert_stopped_after_round_1(self, host, code, reinit_ran, out):
        tray2_ids = sorted(GALAXY_BDFS.index(b) for b in self.TRAY2 if b != self.LOST)
        assert code == 1 and not reinit_ran
        # Round 1's SBR + USER_RESET went to tray 2 only; no reset or re-power after that,
        # but those chips still got POST_RESET and every chip was recounted.
        assert host.ioctl_ids(IoctlResetFlags.RESET_PCIE_LINK) == [tray2_ids]
        assert host.ioctl_ids(IoctlResetFlags.USER_RESET) == [tray2_ids]
        assert host.ioctl_ids(IoctlResetFlags.POST_RESET) == [tray2_ids]
        assert "Recount after round 1: 31/32 chips" in out.out
        assert "Recovery round 2" not in out.out
        assert "UBB" not in out.out + out.err

    @pytest.mark.parametrize("mode", ["fail", "timeout"])
    def test_bmc_gone_after_round_1_stops_before_round_2(self, tmp_path, capsys, mode):
        # OM-F3-GBH01 24/32 -> 0/32: the BMC answered once, then round 2 would have sent
        # SBR + USER_RESET to every present chip and re-powered all four trays.
        host = FakeGalaxy(tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "all"})
        host.mc_info = ["ok", mode]
        code, reinit_ran, _ = host.reset(ok=False, missing=[self.LOST])
        out = capsys.readouterr()
        assert host.mc_calls == [10, 10]
        assert host.ipmi == [("0x2", GLX_IPMI_OP_MODE_NO_RETIMER)]  # no 0xF
        self.assert_stopped_after_round_1(host, code, reinit_ran, out)
        assert (
            f"Hardware fault: chip {self.LOST} (tray 2) still missing "
            f"(in-band IPMI not answering; tray re-power skipped): gone from PCI"
            in out.err
        )
        assert "after tray re-power" not in out.err

    def test_bmc_gone_after_per_tray_failure_skips_all_tray_fallback(
        self, tmp_path, capsys
    ):
        host = FakeGalaxy(
            tmp_path,
            states={self.LOST: "gone"},
            heal={self.LOST: "all"},
            reject_per_tray=True,
        )
        host.mc_info = ["ok", "fail"]
        code, reinit_ran, _ = host.reset(ok=False, missing=[self.LOST])
        out = capsys.readouterr()
        # Pulse sent, reply lost: waits IPMI_RECOVER_TIMEOUT_S for the BMC, then stops.
        assert len(host.mc_calls) > 2
        assert host.ipmi == [("0x2", GLX_IPMI_OP_MODE_NO_RETIMER)]  # no 0xF fallback
        self.assert_stopped_after_round_1(host, code, reinit_ran, out)
        assert (
            f"Hardware fault: chip {self.LOST} (tray 2) still missing "
            "(re-power pulse of tray(s) 2 sent, reply lost" in out.err
        )

    def test_failed_round_1_repower_stops_without_escalating(self, tmp_path, capsys):
        host = FakeGalaxy(
            tmp_path,
            states={self.LOST: "gone"},
            heal={self.LOST: "all"},
            reject_all=True,
        )
        code, reinit_ran, _ = host.reset(ok=False, missing=[self.LOST])
        out = capsys.readouterr()
        # Per-tray 0x2, then the 0xF fallback (BMC still answers) and its legacy retry fail.
        assert host.ipmi == [
            ("0x2", GLX_IPMI_OP_MODE_NO_RETIMER),
            ("0xF", GLX_IPMI_OP_MODE_NO_RETIMER),
            ("0xF", GLX_IPMI_OP_MODE_WITH_RETIMER),
        ]
        assert host.mc_calls == [10, 10]
        self.assert_stopped_after_round_1(host, code, reinit_ran, out)
        assert (
            f"Hardware fault: chip {self.LOST} (tray 2) still missing "
            "(re-power of tray(s) 2 failed (IPMI command failed:" in out.err
        )

    class Round2Rejected(FakeGalaxy):
        """Round 1's per-tray re-power works but leaves the chip gone; round 2's re-power
        (and its 0xF fallback) is rejected, and its USER_RESET leaves ``knocked`` reading all
        ones."""

        knocked = ()

        def run(self, cmd, **kwargs):
            if cmd[:3] == ["sudo", "ipmitool", "raw"] and self.ipmi:
                self.reject_all = True
            return super().run(cmd, **kwargs)

        def ioctl(self, ids, flag):
            if flag == IoctlResetFlags.USER_RESET and self.ioctl_ids(flag):
                for bdf in self.knocked:
                    self._set(bdf, "ones")
            return super().ioctl(ids, flag)

    def test_failed_round_2_repower_still_post_resets_and_names_every_chip(
        self, tmp_path, capsys
    ):
        knocked = ["0000:41:00.0", "0000:42:00.0"]  # tray 2, like the lost chip
        host = self.Round2Rejected(
            tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "all"}
        )
        host.knocked = knocked
        code, reinit_ran, _ = host.reset(ok=False, missing=[self.LOST])
        out = capsys.readouterr()
        present = sorted(GALAXY_BDFS.index(b) for b in self.TRAY2 if b != self.LOST)
        assert code == 1 and not reinit_ran
        # Round 2: tray 2 again, then the 0xF fallback and its legacy retry, all rejected.
        assert [b for b, _ in host.ipmi] == ["0x2", "0x2", "0xF", "0xF"]
        # Round 2's SBR + USER_RESET reached tray 2's 7 present chips, and so does POST_RESET.
        assert host.ioctl_ids(IoctlResetFlags.USER_RESET)[-1] == present
        assert host.ioctl_ids(IoctlResetFlags.POST_RESET)[-1] == present
        assert len(host.ioctl_ids(IoctlResetFlags.USER_RESET)) == 2  # nothing after
        assert "Recount after round 2: 29/32 chips" in out.out
        how = "(re-power of tray(s) 2 failed (IPMI command failed:"
        for bdf, tray in [(self.LOST, 2), (knocked[0], 2), (knocked[1], 2)]:
            assert f"Hardware fault: chip {bdf} (tray {tray}) still missing {how}" in (
                out.err
            )
        assert out.err.count("Hardware fault: chip") == 3
        assert "UBB" not in out.out + out.err

    def test_failed_repower_with_chip_back_by_rescan_recovers(self, tmp_path):
        # Every re-power is rejected, but the PCI rescan brings the chip back: it reads ok
        # and got POST_RESET, so the box is whole and the reset succeeds.
        host = FakeGalaxy(
            tmp_path,
            states={self.LOST: "gone"},
            heal={self.LOST: "rescan"},
            reject_all=True,
        )
        code, reinit_ran, _ = host.reset(ok=False, missing=[self.LOST])
        assert code is None and reinit_ran
        assert host.ioctl_ids(IoctlResetFlags.POST_RESET) == [
            sorted(GALAXY_BDFS.index(b) for b in self.TRAY2 if b != self.LOST)
        ]

    def test_reinit_path_bmc_gone_after_round_1_stops(self, tmp_path, capsys):
        lost = "0000:45:00.0"  # tray 2; read all ones at re-init, never heals
        host = FakeGalaxy(tmp_path, lose_at_reinit=[lost])
        host.lose_at_reinit_heal = "never"
        host.mc_info = ["ok", "fail"]
        # Round 1 re-powers tray 2 only; the chip stays lost, round 2 finds no BMC answer.
        code, _, _ = host.reset(ok=True)
        err = capsys.readouterr().err
        assert code == 1
        assert host.discovers == 1
        assert [b for b, _ in host.ipmi] == ["0x2"]
        assert (
            f"Hardware fault: chip {lost} (tray 2) still missing "
            "(in-band IPMI not answering; tray re-power skipped)" in err
        )

    def test_umd_missing_list_alone_triggers_recovery(self, tmp_path, capsys):
        # UMD names the chip even though its sysfs config still reads fine.
        host = FakeGalaxy(tmp_path)
        code, _, _ = host.reset(ok=False, missing=[self.LOST])
        assert code is None
        assert host.ipmi == [("0x2", GLX_IPMI_OP_MODE_NO_RETIMER)]
        assert "UMD warm reset reports it did not come back" in capsys.readouterr().err

    def test_per_tray_reset_rejected_resets_all_trays(self, tmp_path):
        host = FakeGalaxy(
            tmp_path,
            states={self.LOST: "gone"},
            heal={self.LOST: "all"},
            reject_per_tray=True,
        )
        code, _, _ = host.reset(ok=False, missing=[self.LOST])
        assert code is None
        assert host.ipmi == [
            ("0x2", GLX_IPMI_OP_MODE_NO_RETIMER),
            ("0xF", GLX_IPMI_OP_MODE_NO_RETIMER),
        ]

    TRAY3 = [b for b in GALAXY_BDFS if b.startswith("0000:c")]

    def test_round_1_recount_full_runs_no_round_2(self, tmp_path, capsys):
        # OM-E4-GBH02 try 2 (01:48 PDT 10-27): tray 3 held in reset, c1-c8:00.0 read all ones.
        # After round 1's tray 3 re-power they read back only once POST_RESET was sent; the
        # check before POST_RESET counted them missing, so round 2 re-powered all four trays,
        # printed "Hardware fault ... still missing ...: ok" and exited 1 on a 32/32 box.
        host = FakeGalaxy(
            tmp_path,
            states={b: "ones" for b in self.TRAY3},
            heal={b: "post" for b in self.TRAY3},
        )
        code, reinit_ran, _ = host.reset(ok=False, missing=self.TRAY3)
        out = capsys.readouterr()
        assert code is None and reinit_ran
        assert host.ipmi == [("0x4", GLX_IPMI_OP_MODE_NO_RETIMER)]  # tray 3 only, once
        tray3_ids = sorted(GALAXY_BDFS.index(b) for b in self.TRAY3)
        assert host.ioctl_ids(IoctlResetFlags.USER_RESET) == [tray3_ids]
        assert host.ioctl_ids(IoctlResetFlags.POST_RESET) == [tray3_ids]
        assert "Recount after round 1: 32/32 chips" in out.out
        assert "Recovery round 2" not in out.out
        assert "Hardware fault" not in out.err
        assert "Recovered: all 8 missing chip(s) are back" in out.out
        assert all(state == "ok" for state in host.state.values())

    def test_marker_set_chips_seed_the_hold_in_round_1(self, tmp_path, capsys):
        # Mock only. Task #3791 showed the plain 0x3 pulse does not reset a live tray (2/2):
        # chips whose marker reads set after the warm reset get their tray held in round 1.
        host = FakeGalaxy(
            tmp_path,
            states={b: "marker" for b in self.TRAY3},
            heal={b: "hold" for b in self.TRAY3},
        )
        host.hold_needs_release = True
        code, reinit_ran, _ = host.reset(ok=False, missing=self.TRAY3)
        out = capsys.readouterr()
        assert code is None and reinit_ran
        assert host.ipmi == [
            ("0x4", GLX_IPMI_OP_MODE_RELEASE),
            ("0x4", GLX_IPMI_OP_MODE_HOLD),
            ("0x4", GLX_IPMI_OP_MODE_NO_RETIMER),
        ]
        assert "Recount after round 1: 32/32 chips" in out.out
        assert "Recount after round 2" not in out.out
        assert all(state == "ok" for state in host.state.values())

    def test_post_reset_failure_with_marker_set_is_labelled(self, tmp_path, capsys):
        # A marker-set chip that comes back unreset after a pulse with no hold (marker not
        # set at entry, set after the re-power): the message says the chip may not have reset.
        host = FakeGalaxy(tmp_path, states={b: "ones" for b in self.TRAY3})
        orig = host.run

        def run(cmd, **kw):
            res = orig(cmd, **kw)
            if (
                cmd[:3] == ["sudo", "ipmitool", "raw"]
                and cmd[7] != GLX_IPMI_OP_MODE_HOLD
            ):
                for b in self.TRAY3:
                    host._set(b, "marker")
            return res

        host.run = run
        host.reset(ok=False, missing=self.TRAY3)
        err = capsys.readouterr().err
        assert (
            "POST_RESET failed and the reset marker reads set "
            "(chip may not have reset)" in err
        )

    def test_hold_accepted_pulse_rejected_bmc_gone_names_the_held_tray(
        self, tmp_path, capsys
    ):
        host = FakeGalaxy(tmp_path, states={b: "marker" for b in self.TRAY3})
        host.reject_pulse_after_hold = True
        host.mc_info = ["ok", "fail"]
        code, reinit_ran, _ = host.reset(ok=False, missing=self.TRAY3)
        err = capsys.readouterr().err
        assert code == 1
        assert not reinit_ran
        assert "Tray(s) 3 may still be held in reset" in err
        assert "raw 0x30 0x8B 0x4 0xFF 0x3 0xF" in err

    # Task #3981 (OM-E4-GBH02, 1/1): from 31/32, c1:00.0 (tray 3) off PCI and c2-c8:00.0
    # marker-set after the warm reset. Round 1 held tray 3, the pulse failed with the in-band
    # BMC silent, and the reset stopped after one 10 s mc info check: tray 3 stayed held, 24/32.
    def _e3963(self, tmp_path):
        lost = self.TRAY3[0]
        host = FakeGalaxy(
            tmp_path,
            states={lost: "ones", **{b: "marker" for b in self.TRAY3[1:]}},
            heal={b: "hold" for b in self.TRAY3[1:]},
        )
        host.hold_needs_release = True
        host.kcs_silent_pulses = 1
        return host, lost

    def test_kcs_silent_pulse_after_hold_waits_and_releases_the_held_tray(
        self, tmp_path, capsys
    ):
        # Mock only. The BMC answers again after a few silent polls: the pulse did not go
        # through (no held chip back), so tray 3 alone gets it once more; the 7 held chips
        # come back and only the chip already off PCI is named, by BDF and tray.
        host, lost = self._e3963(tmp_path)
        host.mc_info = ["ok", "fail", "fail", "fail", "ok"]
        code, reinit_ran, _ = host.reset(ok=False, missing=self.TRAY3)
        out = capsys.readouterr()
        assert code == 1 and not reinit_ran
        assert host.ipmi[:4] == [
            ("0x4", GLX_IPMI_OP_MODE_RELEASE),
            ("0x4", GLX_IPMI_OP_MODE_HOLD),
            ("0x4", GLX_IPMI_OP_MODE_NO_RETIMER),
            ("0x4", GLX_IPMI_OP_MODE_NO_RETIMER),
        ]
        assert all(b == "0x4" for b, _ in host.ipmi)  # never the all-tray fallback
        assert "Resetting all four trays" not in out.out
        assert "sending the pulse once more" in out.out
        assert "Recount after round 1: 31/32 chips" in out.out
        assert all(host.state[b] == "ok" for b in self.TRAY3[1:])
        assert f"Hardware fault: chip {lost} (tray 3) still missing" in out.err
        assert out.err.count("Hardware fault: chip") == 1
        assert "may still be held in reset" not in out.err

    def test_kcs_silent_pulse_that_went_through_is_not_sent_again(
        self, tmp_path, capsys
    ):
        # Mock only. The pulse re-powered tray 3 but its reply was lost: once the BMC answers,
        # the held chips read back, so round 1 re-powers tray 3 once, not twice.
        host, lost = self._e3963(tmp_path)
        host.kcs_pulse_acts = True
        host.mc_info = ["ok", "fail", "ok"]
        code, _, _ = host.reset(ok=False, missing=self.TRAY3)
        out = capsys.readouterr()
        assert code == 1
        assert host.ipmi[:3] == [
            ("0x4", GLX_IPMI_OP_MODE_RELEASE),
            ("0x4", GLX_IPMI_OP_MODE_HOLD),
            ("0x4", GLX_IPMI_OP_MODE_NO_RETIMER),
        ]
        # Round 2 (for the chip already off PCI) starts with its own pulse, no hold.
        assert host.ipmi[3:] == [("0x4", GLX_IPMI_OP_MODE_NO_RETIMER)]
        assert "the pulse went through: not sent again" in out.out
        assert "Recount after round 1: 31/32 chips" in out.out
        assert out.err.count("Hardware fault: chip") == 1
        assert f"Hardware fault: chip {lost} (tray 3) still missing" in out.err

    def test_kcs_silent_held_chips_back_on_second_read_no_second_pulse(
        self, tmp_path, capsys
    ):
        # Mock only. The pulse went through but the held chips need a second read to show
        # ok: the reset polls them before deciding, so the pulse is not sent again.
        import tt_smi.reset as reset_mod

        host, _ = self._e3963(tmp_path)
        host.kcs_pulse_acts = True
        host.mc_info = ["ok", "fail", "ok"]
        real = reset_mod.chip_reset_state
        real_rescan = reset_mod.rescan_pci
        st = {"armed": False, "n": 0}

        def rescan(*a, **k):
            st["armed"] = True
            return real_rescan(*a, **k)

        def late(bdf, *a, **k):
            if st["armed"]:
                st["n"] += 1
                if st["n"] <= len(self.TRAY3):  # first poll of every held chip
                    return "gone"
            return real(bdf, *a, **k)

        with patch("tt_smi.reset.chip_reset_state", side_effect=late), patch(
            "tt_smi.reset.rescan_pci", side_effect=rescan
        ):
            host.reset(ok=False, missing=self.TRAY3)
        out = capsys.readouterr()
        assert "the pulse went through: not sent again" in out.out
        assert "sending the pulse once more" not in out.out
        assert host.ipmi[:3] == [
            ("0x4", GLX_IPMI_OP_MODE_RELEASE),
            ("0x4", GLX_IPMI_OP_MODE_HOLD),
            ("0x4", GLX_IPMI_OP_MODE_NO_RETIMER),
        ]

    def test_kcs_silent_for_good_names_every_held_chip_and_exits_non_zero(
        self, tmp_path, capsys
    ):
        # Mock only. The BMC never answers in-band again (as on OM-E4-GBH02 until the cold
        # cycle): the reset polls for IPMI_RECOVER_TIMEOUT_S, sends nothing more, rescans,
        # recounts and names all 8 tray-3 chips by BDF and tray, rc 1.
        host, _ = self._e3963(tmp_path)
        host.mc_info = ["ok", "fail"]
        code, reinit_ran, _ = host.reset(ok=False, missing=self.TRAY3)
        out = capsys.readouterr()
        assert code == 1 and not reinit_ran
        assert [op for _, op in host.ipmi] == [
            GLX_IPMI_OP_MODE_RELEASE,
            GLX_IPMI_OP_MODE_HOLD,
            GLX_IPMI_OP_MODE_NO_RETIMER,
        ]
        assert len(host.mc_calls) > 3  # it waited, not one 10 s check
        assert (
            f"release pulse sent, reply lost; in-band IPMI silent for "
            f"{IPMI_RECOVER_TIMEOUT_S} s; tray(s) 3 may still be held"
        ) in out.err
        # one announce for the wait's many polls (plus the two earlier checks)
        assert out.out.count("Executing command: sudo ipmitool mc info") == 3
        assert "Tray(s) 3 may still be held in reset" in out.err
        for bdf in self.TRAY3:
            assert f"Hardware fault: chip {bdf} (tray 3) still missing" in out.err
        assert "Recovery round 2" not in out.out

    # Task #4132 (OM-E4-GBH02, natural loss of tray 3, c1-c8:00.0, 0/2): no hold; the pulse
    # 0x4 0xFF 0x3 0xF got "No data available" and the BMC went silent in-band. ecd5c5e
    # stopped after one 10 s check ("tray re-power skipped") although the pulse was sent;
    # fix-d1 ended at 0/32 (trays 1, 2 and 4 all ones), fix-n2 at 24/32.
    def _natural_tray3(self, tmp_path, acts):
        host = FakeGalaxy(
            tmp_path,
            states={b: "ones" for b in self.TRAY3},
            heal={b: "tray" for b in self.TRAY3},
        )
        host.kcs_silent_pulses = 1
        host.kcs_silent_without_hold = True
        host.kcs_pulse_acts = acts
        return host

    @pytest.mark.parametrize("acts", [True, False])
    def test_kcs_silent_pulse_without_hold_waits_and_recounts(
        self, tmp_path, capsys, acts
    ):
        # Mock only. The BMC answers again after a few silent polls: the reset recounts every
        # chip. If the pulse went through, round 1 ends at 32/32; if not, round 2 sends tray 3
        # its own pulse. Never the all-tray fallback.
        host = self._natural_tray3(tmp_path, acts)
        host.mc_info = ["ok", "fail", "fail", "fail", "ok"]
        code, reinit_ran, _ = host.reset(ok=False, missing=self.TRAY3)
        out = capsys.readouterr()
        assert code is None and reinit_ran
        assert len(host.mc_calls) > 3  # it waited, not one 10 s check
        assert "pulse sent, reply lost" in out.out
        assert "Resetting all four trays" not in out.out
        if acts:
            assert host.ipmi == [("0x4", GLX_IPMI_OP_MODE_NO_RETIMER)]
            assert "Recount after round 1: 32/32 chips" in out.out
            assert "Recovery round 2" not in out.out
        else:
            assert host.ipmi == [("0x4", GLX_IPMI_OP_MODE_NO_RETIMER)] * 2
            assert "Recount after round 1: 24/32 chips" in out.out
            assert "Recount after round 2: 32/32 chips" in out.out
        assert "Hardware fault" not in out.err
        assert all(state == "ok" for state in host.state.values())

    def test_kcs_silent_pulse_that_knocks_other_trays_gets_round_2(
        self, tmp_path, capsys
    ):
        # Mock only. fix-d1's shape: the tray-3 pulse went out, its reply was lost and trays
        # 1, 2 and 4 read all ones afterwards. Once the BMC answers, the recount finds them
        # and round 2 re-powers only the trays still short (1, 2, 4; tray 3 is back).
        host = self._natural_tray3(tmp_path, acts=True)
        host.knock_others = True
        host.mc_info = ["ok", "fail", "fail", "ok"]
        code, reinit_ran, _ = host.reset(ok=False, missing=self.TRAY3)
        out = capsys.readouterr()
        assert host.ipmi == [
            ("0x4", GLX_IPMI_OP_MODE_NO_RETIMER),
            ("0xB", GLX_IPMI_OP_MODE_NO_RETIMER),
        ]
        assert "Recount after round 1: 8/32 chips" in out.out
        assert "Recount after round 2: 32/32 chips" in out.out
        assert code is None and reinit_ran
        assert all(state == "ok" for state in host.state.values())

    def test_kcs_silent_for_good_without_hold_names_every_chip_and_exits_non_zero(
        self, tmp_path, capsys
    ):
        # Mock only. The BMC never answers in-band again (as in fix-d1, still silent 8 min
        # later): the reset polls for IPMI_RECOVER_TIMEOUT_S, sends nothing more, rescans,
        # recounts and names every missing chip by BDF and tray 1-4, rc 1.
        host = self._natural_tray3(tmp_path, acts=False)
        host.mc_info = ["ok", "fail"]
        code, reinit_ran, _ = host.reset(ok=False, missing=self.TRAY3)
        out = capsys.readouterr()
        assert code == 1 and not reinit_ran
        assert host.ipmi == [("0x4", GLX_IPMI_OP_MODE_NO_RETIMER)]
        assert len(host.mc_calls) > 3
        reason = (
            "re-power pulse of tray(s) 3 sent, reply lost; in-band IPMI silent for "
            f"{IPMI_RECOVER_TIMEOUT_S} s"
        )
        assert reason in out.err
        assert "tray re-power skipped" not in out.err
        for bdf in self.TRAY3:
            assert (
                f"Hardware fault: chip {bdf} (tray 3) still missing ({reason})"
                in out.err
            )
        assert out.err.count("Hardware fault: chip") == 8
        assert "Recovery round 2" not in out.out
        assert "UBB" not in out.out + out.err

    # Review #4176 follow-up (mock only): all four trays short, so round 1 sends the 0xF pulse.
    def _all_trays_short(self, tmp_path, acts):
        host = FakeGalaxy(
            tmp_path,
            states={b: "ones" for b in GALAXY_BDFS},
            heal={b: "tray" for b in GALAXY_BDFS},
        )
        host.kcs_silent_pulses = 1
        host.kcs_silent_without_hold = True
        host.kcs_pulse_acts = acts
        return host

    def test_all_tray_pulse_lost_reply_waits_and_recounts_no_op0_fallback(
        self, tmp_path, capsys
    ):
        host = self._all_trays_short(tmp_path, acts=True)
        host.mc_info = ["ok", "fail", "fail", "ok"]
        code, reinit_ran, _ = host.reset(ok=False, missing=list(GALAXY_BDFS))
        out = capsys.readouterr()
        assert host.ipmi == [("0xF", GLX_IPMI_OP_MODE_NO_RETIMER)]
        assert len(host.mc_calls) > 3
        assert "pulse sent, reply lost" in out.out
        assert "Recount after round 1: 32/32 chips" in out.out
        assert code is None and reinit_ran
        assert all(state == "ok" for state in host.state.values())

    def test_all_tray_pulse_bmc_silent_for_good_names_every_chip_no_op0(
        self, tmp_path, capsys
    ):
        host = self._all_trays_short(tmp_path, acts=False)
        host.mc_info = ["ok", "fail"]
        code, reinit_ran, _ = host.reset(ok=False, missing=list(GALAXY_BDFS))
        out = capsys.readouterr()
        assert code == 1 and not reinit_ran
        assert host.ipmi == [("0xF", GLX_IPMI_OP_MODE_NO_RETIMER)]
        assert len(host.mc_calls) > 3
        assert f"in-band IPMI silent for {IPMI_RECOVER_TIMEOUT_S} s" in out.err
        assert out.err.count("Hardware fault: chip") == 32
        for bdf in GALAXY_BDFS:
            assert f"Hardware fault: chip {bdf} (tray " in out.err

    def test_rejected_pulse_bmc_silent_without_hold_waits_no_all_tray_fallback(
        self, tmp_path, capsys
    ):
        # Mock only. The per-tray pulse is rejected and the BMC is silent after it: wait, then
        # recount; no 0xF fallback right after the silent BMC. Round 2 starts with its own
        # mc info check and per-tray pulse (its usual 0xF fallback once the BMC answers).
        host = FakeGalaxy(
            tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "tray"}
        )
        host.reject_per_tray = True
        host.mc_info = ["ok", "fail", "fail", "ok"]
        host.reset(ok=False, missing=[self.LOST])
        out = capsys.readouterr()
        assert host.ipmi[:2] == [("0x2", GLX_IPMI_OP_MODE_NO_RETIMER)] * 2
        assert len(host.mc_calls) > 3
        assert "pulse sent, reply lost" in out.out
        assert "Recount after round 1: 31/32 chips" in out.out
        assert "Recovery round 2/2: re-powering tray(s) 2" in out.out

    def test_hold_leaves_chips_up_pulse_fails_exits_non_zero(self, tmp_path, capsys):
        # The hold timed out with the chips still on PCI and the pulse raised: the recount
        # finds every chip, but the tray may still be held, so the reset must not exit 0.
        host = FakeGalaxy(tmp_path, states={b: "marker" for b in self.TRAY3})
        host.hold_leaves_chips_up = True
        host.reject_pulse_after_hold = True
        code, reinit_ran, _ = host.reset(ok=False, missing=self.TRAY3)
        err = capsys.readouterr().err
        assert code == 1
        assert not reinit_ran
        assert "may still be held in reset" in err
        assert "tray 3) still missing" in err
        assert "(chip may not have reset)" in err

    def test_hold_accepted_fallback_fails_names_the_held_tray(self, tmp_path, capsys):
        host = FakeGalaxy(tmp_path, states={b: "marker" for b in self.TRAY3})
        host.reject_pulse_after_hold = True
        code, _, _ = host.reset(ok=False, missing=self.TRAY3)
        err = capsys.readouterr().err
        assert code == 1
        assert "Tray(s) 3 may still be held in reset" in err

    def test_live_tray_the_hold_does_not_reset_is_a_hardware_fault(
        self, tmp_path, capsys
    ):
        host = FakeGalaxy(tmp_path, states={b: "marker" for b in self.TRAY3})
        code, reinit_ran, _ = host.reset(ok=False, missing=self.TRAY3)
        err = capsys.readouterr().err
        assert code == 1
        assert not reinit_ran
        assert host.ipmi == [
            ("0x4", GLX_IPMI_OP_MODE_RELEASE),
            ("0x4", GLX_IPMI_OP_MODE_HOLD),
            ("0x4", GLX_IPMI_OP_MODE_NO_RETIMER),
            ("0x4", GLX_IPMI_OP_MODE_RELEASE),
            ("0x4", GLX_IPMI_OP_MODE_HOLD),
            ("0x4", GLX_IPMI_OP_MODE_NO_RETIMER),
        ]
        assert (
            "Hardware fault: chip 0000:c1:00.0 (tray 3) still missing after tray re-power "
            "and PCI rescan: reset marker still set (chip may not have reset)" in err
        )

    def test_rejected_hold_stops_recovery(self, tmp_path, capsys):
        host = FakeGalaxy(tmp_path, states={b: "marker" for b in self.TRAY3})
        host.reject_hold = True
        code, reinit_ran, _ = host.reset(ok=False, missing=self.TRAY3)
        err = capsys.readouterr().err
        assert code == 1
        assert not reinit_ran
        assert host.ipmi == [
            ("0x4", GLX_IPMI_OP_MODE_RELEASE),
            ("0x4", GLX_IPMI_OP_MODE_HOLD),
        ]
        assert "hold of tray(s) 3 failed" in err

    def test_hold_sends_release_waits_then_holds(self, tmp_path, capsys):
        # Mock only. Task #3850: a 0x1 hold sent straight after the 0x3 pulse left a live
        # tray 3 on PCI (0/3); 0x2, ~10 s, then 0x1 dropped it every time. The release, the
        # wait and the hold come back to back, before the pulse.
        host = FakeGalaxy(
            tmp_path,
            states={b: "marker" for b in self.TRAY3},
            heal={b: "hold" for b in self.TRAY3},
        )
        host.hold_needs_release = True
        code, reinit_ran, _ = host.reset(ok=False, missing=self.TRAY3)
        capsys.readouterr()
        assert code is None and reinit_ran
        i = host.events.index(("ipmi", "0x4", GLX_IPMI_OP_MODE_RELEASE))
        assert host.events[i : i + 3] == [
            ("ipmi", "0x4", GLX_IPMI_OP_MODE_RELEASE),
            ("sleep", TRAY_HOLD_RELEASE_WAIT_S),
            ("ipmi", "0x4", GLX_IPMI_OP_MODE_HOLD),
        ]
        assert TRAY_HOLD_RELEASE_WAIT_S == 10
        pulse = host.events.index(("ipmi", "0x4", GLX_IPMI_OP_MODE_NO_RETIMER))
        assert pulse > i + 2

    def test_hold_with_no_release_first_leaves_the_live_tray_on_pci(
        self, tmp_path, capsys
    ):
        # The mock models #3850: with the release rejected, the hold leaves tray 3 on PCI.
        # The hold warns, the pulse still runs, and each chip still unreset is a
        # hardware fault (exit 1, no re-init).
        host = FakeGalaxy(
            tmp_path,
            states={b: "marker" for b in self.TRAY3},
            heal={b: "hold" for b in self.TRAY3},
        )
        host.hold_needs_release = True
        host.reject_release = True
        code, reinit_ran, _ = host.reset(ok=False, missing=self.TRAY3)
        out = capsys.readouterr()
        assert code == 1 and not reinit_ran
        assert "release of tray(s) 3 before the hold failed" in out.out
        assert "8 chip(s) still read on PCI 30 s into the hold of tray(s) 3" in out.out
        for b in self.TRAY3:
            assert f"Hardware fault: chip {b} (tray 3) still missing" in out.err

    def test_rejected_release_still_holds(self, tmp_path, capsys):
        # A BMC that rejects 0x2 but takes 0x1 (and drops the tray on it): warn, hold anyway.
        host = FakeGalaxy(
            tmp_path,
            states={b: "marker" for b in self.TRAY3},
            heal={b: "hold" for b in self.TRAY3},
        )
        host.reject_release = True
        code, reinit_ran, _ = host.reset(ok=False, missing=self.TRAY3)
        out = capsys.readouterr()
        assert code is None and reinit_ran
        assert "release of tray(s) 3 before the hold failed" in out.out
        assert host.ipmi == [
            ("0x4", GLX_IPMI_OP_MODE_RELEASE),
            ("0x4", GLX_IPMI_OP_MODE_HOLD),
            ("0x4", GLX_IPMI_OP_MODE_NO_RETIMER),
        ]
        assert all(state == "ok" for state in host.state.values())

    def test_tray_that_stays_down_is_the_only_tray_repowered(self, tmp_path, capsys):
        # A whole tray that never comes back: both rounds re-power tray 3 alone (trays 1, 2
        # and 4 have every chip), and every chip of it is named by BDF and tray.
        host = FakeGalaxy(tmp_path, states={b: "ones" for b in self.TRAY3})
        code, reinit_ran, _ = host.reset(ok=False, missing=self.TRAY3)
        out = capsys.readouterr()
        assert code == 1 and not reinit_ran
        assert [b for b, _ in host.ipmi] == ["0x4", "0x4"]
        tray3_ids = sorted(GALAXY_BDFS.index(b) for b in self.TRAY3)
        assert host.ioctl_ids(IoctlResetFlags.USER_RESET) == [tray3_ids, tray3_ids]
        assert "Recount after round 2: 24/32 chips" in out.out
        for bdf in self.TRAY3:
            assert (
                f"Hardware fault: chip {bdf} (tray 3) still missing after tray re-power "
                "and PCI rescan: config space reads all ones" in out.err
            )
        assert out.err.count("Hardware fault: chip") == 8
        assert "UBB" not in out.out + out.err

    def test_round_two_repowers_only_trays_still_missing_a_chip(self, tmp_path, capsys):
        # Round 1 brings tray 2 back but not tray 3's dead chip: round 2 re-powers tray 3 only.
        dead = "0000:c3:00.0"
        host = FakeGalaxy(
            tmp_path, states={self.LOST: "gone", dead: "ones"}, heal={self.LOST: "tray"}
        )
        code, _, _ = host.reset(ok=False, missing=[self.LOST, dead])
        assert code == 1
        assert [b for b, _ in host.ipmi] == ["0x6", "0x4"]

    def test_chip_that_never_returns_is_a_hardware_fault(self, tmp_path, capsys):
        host = FakeGalaxy(tmp_path, states={self.LOST: "gone"})
        code, reinit_ran, _ = host.reset(ok=False, missing=[self.LOST])
        out = capsys.readouterr()
        assert code == 1
        assert not reinit_ran
        assert [b for b, _ in host.ipmi] == ["0x2", "0x2"]  # at most two rounds, tray 2
        assert (
            f"Hardware fault: chip {self.LOST} (tray 2) still missing after tray re-power and PCI rescan: gone from PCI"
            in out.err
        )
        assert "Recount after round 2: 31/32 chips" in out.out
        assert "UBB" not in out.out + out.err

    def test_one_dead_chip_does_not_stop_the_others(self, tmp_path, capsys):
        dead = "0000:c3:00.0"  # tray 3
        host = FakeGalaxy(
            tmp_path, states={self.LOST: "gone", dead: "ones"}, heal={self.LOST: "tray"}
        )
        code, _, _ = host.reset(ok=False, missing=[self.LOST, dead])
        err = capsys.readouterr().err
        assert code == 1
        assert host.ipmi[0][0] == "0x6"  # trays 2 and 3 together
        assert host.state[self.LOST] == "ok"
        assert f"Hardware fault: chip {dead} (tray 3) still missing" in err
        assert f"Hardware fault: chip {self.LOST}" not in err
        assert f"1 chip(s) did not come back: {dead}" in err

    def test_chip_lost_by_the_tray_repower_is_caught(self, tmp_path, capsys):
        other = "0000:41:00.0"  # also tray 2, healthy before the re-power
        host = FakeGalaxy(
            tmp_path,
            states={self.LOST: "gone"},
            heal={self.LOST: "tray"},
            drop_on_repower=[other],
        )
        code, _, _ = host.reset(ok=False, missing=[self.LOST])
        assert code == 1
        assert (
            f"Hardware fault: chip {other} (tray 2) still missing"
            in capsys.readouterr().err
        )

    def test_repower_that_knocks_other_trays_off_gets_round_2(self, tmp_path, capsys):
        # OM-E4-GBH02, tray 3 held in reset: its re-power brought tray 3 back but left
        # trays 1, 2 and 4 reading all ones; round 1 must not count them as present.
        lost = "0000:c1:00.0"
        host = FakeGalaxy(
            tmp_path, states={lost: "gone"}, heal={lost: "tray"}, knock_others=True
        )
        code, reinit_ran, _ = host.reset(ok=False, missing=[lost])
        out = capsys.readouterr().out
        assert code is None and reinit_ran
        assert [m for m, _ in host.ipmi] == ["0x4", "0xB"]  # trays 1, 2 and 4
        assert "Recount after round 2: 32/32 chips" in out
        assert all(state == "ok" for state in host.state.values())

    def test_chip_lost_at_reinit_is_recovered_and_reinit_retried(
        self, tmp_path, capsys
    ):
        lost = "0000:05:00.0"
        host = FakeGalaxy(tmp_path, lose_at_reinit=[lost])
        code, _, _ = host.reset(ok=True)
        out = capsys.readouterr()
        assert code is None
        assert host.discovers == 2
        assert f"Chip {lost} (tray 1) lost after the recount" in out.err
        assert "re-initializing again" in out.out
        assert host.ipmi == [("0x1", GLX_IPMI_OP_MODE_NO_RETIMER)]  # tray 1 only

    def test_umd_ffff_chip_gets_tray_then_all_tray_repower(self, tmp_path, capsys):
        # Mock only. OM-F3-GBH02 47:00.0 (tray 2), task #3973, 08:05-08:10 PDT 10-27: back on
        # PCI after a tray 2 re-power, but tt-umd discovery read 0xffffffff from it; only the
        # all-tray re-power (glx_reset) gave 32/32. Config space alone must not count it back.
        chip = "0000:47:00.0"
        host = FakeGalaxy(tmp_path, bar_ones={chip: "all"})
        code, _, _ = host.reset(ok=True)
        out = capsys.readouterr()
        assert code is None
        assert host.discovers == 3
        assert host.ipmi == [
            ("0x2", GLX_IPMI_OP_MODE_NO_RETIMER),
            ("0xF", GLX_IPMI_OP_MODE_NO_RETIMER),
        ]
        assert (
            f"Chip {chip} (tray 2) lost after the recount: tt-umd discovery read "
            "0xffffffff (config space reads fine)" in out.err
        )

    def test_umd_ffff_chip_healed_by_its_tray_needs_no_all_tray(self, tmp_path):
        chip = "0000:47:00.0"
        host = FakeGalaxy(tmp_path, bar_ones={chip: "tray"})
        code, _, _ = host.reset(ok=True)
        assert code is None
        assert host.discovers == 2
        assert host.ipmi == [("0x2", GLX_IPMI_OP_MODE_NO_RETIMER)]

    def test_umd_ffff_chip_that_never_heals_exits_1(self, tmp_path, capsys):
        chip = "0000:47:00.0"
        host = FakeGalaxy(tmp_path, bar_ones={chip: "never"})
        code, _, _ = host.reset(ok=True)
        assert code == 1
        assert host.discovers == 3
        assert [b for b, _ in host.ipmi] == ["0x2", "0xF"]
        assert (
            "Error when re-initializing chips after recovery" in capsys.readouterr().out
        )

    def test_reinit_failure_with_no_lost_chip_exits_without_repower(self, tmp_path):
        host = FakeGalaxy(tmp_path, reinit_error=True)
        code, _, _ = host.reset(ok=True)
        assert code == 1
        assert host.discovers == 1
        assert host.ipmi == []

    def test_tray_short_before_reset_is_repowered(self, tmp_path, capsys):
        host = FakeGalaxy(tmp_path, absent=[self.LOST], heal={self.LOST: "tray"})
        code, reinit_ran, _ = host.reset(ok=True)
        out = capsys.readouterr()
        assert code is None and reinit_ran
        assert host.ipmi == [("0x2", GLX_IPMI_OP_MODE_NO_RETIMER)]
        assert "Tray 2 has 7/8 chips" in out.err
        assert "Recount after round 1: 32/32 chips" in out.out

    def test_tray_short_that_stays_short_is_a_hardware_fault(self, tmp_path, capsys):
        host = FakeGalaxy(tmp_path, absent=[self.LOST])
        code, _, _ = host.reset(ok=True)
        err = capsys.readouterr().err
        assert code == 1
        assert "Hardware fault: tray 2 has 7/8 chips" in err
        assert "BDF is not known" in err

    def test_old_umd_recounts_and_uses_sysfs_rescan(self, tmp_path):
        host = FakeGalaxy(
            tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "tray"}
        )
        code, _, warm = host.reset(ok=True, with_result=False, rescan=False)
        assert code is None
        warm.warm_reset.assert_called_once_with(
            list(range(32)), secondary_bus_reset=True
        )
        assert host.ipmi == [("0x2", GLX_IPMI_OP_MODE_NO_RETIMER)]
        assert host.rescan_file.read_text() == "1"

    def test_umd_rescan_error_falls_back_to_sysfs(self, tmp_path):
        host = FakeGalaxy(
            tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "tray"}
        )
        host.rescan_pci_bdfs = MagicMock(side_effect=RuntimeError("needs root"))
        code, _, _ = host.reset(ok=False, missing=[self.LOST])
        assert code is None
        assert host.rescan_file.read_text() == "1"

    def test_nonroot_umd_rescan_noop_falls_back_to_sysfs(self, tmp_path):
        # Paired tt-umd returns its input unchanged (no exception) when it may not rescan.
        host = FakeGalaxy(
            tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "sysfs"}
        )
        host.rescan_pci_bdfs = MagicMock(
            side_effect=lambda bdfs, timeout_s=10.0: list(bdfs)
        )
        code, reinit_ran, _ = host.reset(ok=False, missing=[self.LOST])
        assert code is None and reinit_ran
        host.rescan_pci_bdfs.assert_called_once_with([self.LOST], timeout_s=10.0)
        assert host.rescan_file.read_text() == "1"
        assert host.ipmi == [("0x2", GLX_IPMI_OP_MODE_NO_RETIMER)]

    def test_unwritable_rescan_skips_umd_rescan(self, tmp_path):
        host = FakeGalaxy(
            tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "sysfs"}
        )
        with patch("tt_smi.reset.os.access", return_value=False):
            code, _, warm = host.reset(ok=False, missing=[self.LOST])
        assert code is None
        warm.rescan_pci_bdfs.assert_not_called()
        assert host.rescan_file.read_text() == "1"

    @pytest.mark.parametrize(
        "kind,value",
        [
            (
                SmiDeviceTargetKind.DEV_TENSTORRENT_ID,
                [GALAXY_BDFS.index("0000:47:00.0")],
            ),
            (SmiDeviceTargetKind.PCI_BDF, ["0000:47:00.0"]),
            (SmiDeviceTargetKind.UMD_LOGICAL_ID, [GALAXY_BDFS.index("0000:47:00.0")]),
        ],
    )
    def test_targeted_reset_never_repowers_trays(self, tmp_path, capsys, kind, value):
        host = FakeGalaxy(
            tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "tray"}
        )
        code, _, _ = host.reset(
            ok=False, missing=[self.LOST], reset_input=SmiDeviceInput(kind, value)
        )
        err = capsys.readouterr().err
        assert code == 1
        assert host.ipmi == []
        assert host.rescan_file.read_text() == "1"
        assert (
            "Hardware fault: chip 0000:47:00.0 (tray 2) still missing after PCI rescan"
            in err
        )
        assert "full `tt-smi -r` or `tt-smi -glx_reset` can re-power the tray" in err

    @pytest.mark.parametrize(
        "kind,value",
        [
            (
                SmiDeviceTargetKind.DEV_TENSTORRENT_ID,
                [GALAXY_BDFS.index("0000:47:00.0")],
            ),
            (SmiDeviceTargetKind.PCI_BDF, ["0000:47:00.0"]),
        ],
    )
    def test_targeted_reset_heals_by_rescan(self, tmp_path, kind, value):
        host = FakeGalaxy(
            tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "rescan"}
        )
        code, reinit_ran, _ = host.reset(
            ok=False, missing=[self.LOST], reset_input=SmiDeviceInput(kind, value)
        )
        assert code is None and reinit_ran
        assert host.ipmi == []

    @pytest.mark.parametrize(
        "kind,value",
        [
            (SmiDeviceTargetKind.DEV_TENSTORRENT_ID, [0]),
            (SmiDeviceTargetKind.PCI_BDF, ["0000:01:00.0"]),
            (SmiDeviceTargetKind.UMD_LOGICAL_ID, [0]),
        ],
    )
    def test_targeted_reset_ignores_untargeted_dead_chip(
        self, tmp_path, capsys, kind, value
    ):
        host = FakeGalaxy(tmp_path, states={"0000:c3:00.0": "ones", self.LOST: "gone"})
        code, reinit_ran, _ = host.reset(
            ok=True, missing=[self.LOST], reset_input=SmiDeviceInput(kind, value)
        )
        err = capsys.readouterr().err
        assert code is None and reinit_ran
        assert host.ipmi == []
        assert "Hardware fault" not in err
        assert (
            "untargeted chip(s) not reading back" in err
            and "0000:c3:00.0 (tray 3)" in err
        )

    def test_sysfs_rescan_without_root_uses_sudo(self, tmp_path):
        target = tmp_path / "rescan"
        target.write_text("")
        target.chmod(0o444)
        with patch(
            "tt_smi.reset.subprocess.run", return_value=MagicMock(returncode=0)
        ) as run:
            assert _sysfs_write(str(target), "1")
        assert run.call_args[0][0] == ["sudo", "-n", "tee", str(target)]
        assert run.call_args[1]["input"] == "1"

    def test_non_galaxy_only_rescans(self, tmp_path, capsys):
        bdfs = ["0000:01:00.0", "0000:02:00.0"]
        host = FakeGalaxy(
            tmp_path,
            bdfs=bdfs,
            subsystem_id=0x40,
            states={bdfs[1]: "gone"},
            heal={bdfs[1]: "rescan"},
        )
        code, reinit_ran, _ = host.reset(ok=False, missing=[bdfs[1]])
        assert code is None and reinit_ran
        assert host.ipmi == []
        assert "Chip 0000:02:00.0 did not come back" in capsys.readouterr().err

    def test_non_galaxy_chip_that_stays_gone_fails(self, tmp_path, capsys):
        bdfs = ["0000:01:00.0", "0000:02:00.0"]
        host = FakeGalaxy(
            tmp_path, bdfs=bdfs, subsystem_id=0x40, states={bdfs[1]: "gone"}
        )
        code, _, _ = host.reset(ok=False, missing=[bdfs[1]])
        assert code == 1
        assert host.ipmi == []
        assert (
            "Hardware fault: chip 0000:02:00.0 still missing after PCI rescan"
            in capsys.readouterr().err
        )

    @pytest.mark.parametrize("reinit_error", [False, True])
    def test_post_reset_failure_on_a_chip_that_reads_back_is_left_to_reinit(
        self, tmp_path, capsys, reinit_error
    ):
        # A chip back on PCI whose POST_RESET fails is present: no round 2, no hardware
        # fault; the re-init judges it, and a failed re-init still exits non-zero.
        host = FakeGalaxy(
            tmp_path,
            states={self.LOST: "gone"},
            heal={self.LOST: "tray"},
            post_reset_fail=[self.LOST],
            reinit_error=reinit_error,
        )
        code, reinit_ran, _ = host.reset(ok=False, missing=[self.LOST])
        out = capsys.readouterr()
        assert reinit_ran
        assert code == (1 if reinit_error else None)
        assert host.ipmi == [("0x2", GLX_IPMI_OP_MODE_NO_RETIMER)]
        assert "Recovery round 2" not in out.out
        assert "Hardware fault" not in out.err
        assert f"({self.LOST} (tray 2)). The chip reads back on PCI" in out.err
        assert "Recount after round 1: 32/32 chips" in out.out

    def test_post_reset_failure_on_a_chip_still_gone_is_not_a_recovery(
        self, tmp_path, capsys
    ):
        host = FakeGalaxy(
            tmp_path, states={self.LOST: "gone"}, post_reset_fail=[self.LOST]
        )
        code, reinit_ran, _ = host.reset(ok=False, missing=[self.LOST])
        assert code == 1
        assert not reinit_ran
        assert (
            f"Hardware fault: chip {self.LOST} (tray 2) still missing"
            in capsys.readouterr().err
        )


class TestInboxLinksAfterRepower:
    """After a tray re-power, ``tt-smi -r`` checks the in-box ETH links and re-powers to bring
    them back (mock only; OM-F3-GBH02 47:00.0, task #4008: tray 2 re-powered alone left
    05-45, 06-46, 07-47 ch 6/7 down, 114/120 links, until a glx_reset)."""

    LOST = "0000:47:00.0"  # tray 2
    NR = GLX_IPMI_OP_MODE_NO_RETIMER

    def _host(self, tmp_path, **kwargs):
        return FakeGalaxy(
            tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "tray"}, **kwargs
        )

    def test_links_down_after_tray_repower_get_both_sides_repowered(
        self, tmp_path, capsys
    ):
        host = self._host(tmp_path, eth_drop="both")
        code, reinit_ran, _ = host.reset(ok=False, missing=[self.LOST])
        out = capsys.readouterr()
        assert code is None and reinit_ran
        # Tray 2, then trays 1 and 2 (both sides of 05-45, 06-46, 07-47); no all-tray re-power.
        assert host.ipmi == [("0x2", self.NR), ("0x3", self.NR)]
        assert "before the reset" not in out.out
        assert "In-box ETH links after re-powering tray(s) 2: 114/120" in out.out
        assert "(both sides of each down link)" in out.out
        assert "In-box ETH links after re-powering tray(s) 1, 2: 120/120" in out.out
        assert "In-box ETH links back: 120/120" in out.out

    def test_links_only_all_trays_bring_back_get_all_tray_repower(
        self, tmp_path, capsys
    ):
        host = self._host(tmp_path, eth_drop="all")
        code, _, _ = host.reset(ok=False, missing=[self.LOST])
        out = capsys.readouterr()
        assert code is None
        assert host.ipmi == [("0x2", self.NR), ("0x3", self.NR), ("0xF", self.NR)]
        assert "In-box ETH links after re-powering tray(s) 1, 2: 114/120" in out.out
        assert (
            "In-box ETH links after re-powering tray(s) 1, 2, 3, 4: 120/120" in out.out
        )

    def test_links_that_stay_down_exit_nonzero_naming_chip_tray_and_link(
        self, tmp_path, capsys
    ):
        host = self._host(tmp_path, eth_drop="never")
        code, reinit_ran, _ = host.reset(ok=False, missing=[self.LOST])
        out = capsys.readouterr()
        assert code == 1 and not reinit_ran
        assert host.ipmi == [("0x2", self.NR), ("0x3", self.NR), ("0xF", self.NR)]
        assert (
            "Fault: chip 0000:05:00.0 (tray 1) in-box ETH link(s) down after the all-tray "
            "re-power (114/120): ch 6 -> 0000:45:00.0 (tray 2) ch 6; "
            "ch 7 -> 0000:45:00.0 (tray 2) ch 7" in out.err
        )
        assert "Fault: chip 0000:47:00.0 (tray 2) in-box ETH link(s) down" in out.err
        assert "0000:04:00.0" not in out.err
        assert "in-box ETH link(s) down, all chips on PCI" in out.err

    def test_healthy_reset_reads_no_links_and_repowers_nothing(self, tmp_path):
        host = FakeGalaxy(tmp_path, eth_drop="never")
        code, _, _ = host.reset(ok=True)
        assert code is None
        assert host.link_reads == 0 and host.ipmi == []

    def test_full_reset_without_repower_runs_no_discovery(self, tmp_path):
        # A chip lost on a box whose BMC does not answer: no tray re-power, so no link read
        # either (review #4054: no tt-umd discovery on a box that may be hung).
        host = self._host(tmp_path, eth_drop="never")
        host.mc_info = "fail"
        code, reinit_ran, _ = host.reset(ok=False, missing=[self.LOST])
        assert code == 1 and not reinit_ran
        assert host.ipmi == [] and host.link_reads == 0

    def test_eth_train_skip_skips_the_link_check(self, tmp_path, capsys):
        host = self._host(tmp_path, eth_drop="never")
        code, _, _ = host.reset(ok=False, missing=[self.LOST], eth_train_skip=True)
        assert code is None
        assert host.link_reads == 0 and host.ipmi == [("0x2", self.NR)]
        assert (
            "In-box ETH links not checked after the tray re-power (--eth_train_skip)"
            in capsys.readouterr().err
        )

    def test_eth_train_skip_skips_the_link_check_on_reinit_recovery(self, tmp_path):
        host = FakeGalaxy(tmp_path, bar_ones={self.LOST: "tray"}, eth_drop="never")
        code, _, _ = host.reset(ok=True, eth_train_skip=True)
        assert code is None
        assert host.link_reads == 0 and [b for b, _ in host.ipmi] == ["0x2"]

    def test_links_intact_after_tray_repower_need_no_more_repower(
        self, tmp_path, capsys
    ):
        host = self._host(tmp_path)
        code, _, _ = host.reset(ok=False, missing=[self.LOST])
        assert code is None
        assert host.ipmi == [("0x2", self.NR)]
        assert (
            "In-box ETH links after re-powering tray(s) 2: 120/120"
            in capsys.readouterr().out
        )

    def test_tray_short_before_reset_has_its_links_checked_against_the_mesh(
        self, tmp_path, capsys
    ):
        # 47:00.0 is off PCI before the reset (tray 2 short), so a link set read before the
        # reset has none of its links (review #4054, dd91570). Tray 2 comes back from its
        # re-power with 6 links to tray 1 down, 07-47 ch 6/7 among them.
        host = FakeGalaxy(
            tmp_path, absent=[self.LOST], heal={self.LOST: "tray"}, eth_drop="never"
        )
        code, reinit_ran, _ = host.reset(ok=True)
        out = capsys.readouterr()
        assert code == 1 and not reinit_ran
        # Tray 2, then both sides (trays 1 and 2), then all trays.
        assert host.ipmi == [("0x2", self.NR), ("0x3", self.NR), ("0xF", self.NR)]
        assert "In-box ETH links after re-powering tray(s) 2: 114/120" in out.out
        assert (
            "Fault: chip 0000:47:00.0 (tray 2) in-box ETH link(s) down after the all-tray "
            "re-power (114/120): ch 6 -> 0000:07:00.0 (tray 1) ch 6; "
            "ch 7 -> 0000:07:00.0 (tray 1) ch 7"
        ) in out.err
        for bdf in ("05", "06", "07", "45", "46"):
            assert f"Fault: chip 0000:{bdf}:00.0" in out.err
        assert "6 in-box ETH link(s) down, all chips on PCI" in out.err
        assert "0000:07:00.0 (tray 1) ch 6 <-> 0000:47:00.0 (tray 2) ch 6" in out.err
        assert "chip(s) did not come back" not in out.err

    def test_both_sides_first_even_if_other_repowered_trays_cover_all(self, capsys):
        # Trays 3 and 4 re-powered, links down between trays 1 and 2: trays 1 and 2 first.
        down = {((1, 5, 6), (2, 5, 6))}
        reads = iter([(True, down), (True, set())])
        calls = []

        def recover(missing, short, *a, **k):
            calls.append(sorted(short))
            return [], {}, None

        with patch("tt_smi.reset._wait_links", lambda *a, **k: next(reads)), patch(
            "tt_smi.reset.recover_missing_chips", recover
        ), patch("tt_smi.reset._present_chips", return_value={}):
            check_inbox_links_after_repower({3, 4}, True, False)
        assert calls == [[1, 2]]
        assert "In-box ETH links back" in capsys.readouterr().out

    def test_links_unreadable_after_repower_escalate_then_fail(self, tmp_path, capsys):
        host = self._host(tmp_path, eth_drop="never", eth_unreadable="after_first")
        code, _, _ = host.reset(ok=False, missing=[self.LOST])
        out = capsys.readouterr()
        assert code == 1
        assert host.ipmi == [("0x2", self.NR), ("0x3", self.NR), ("0xF", self.NR)]
        assert "In-box ETH links after re-powering tray(s) 1, 2: unreadable" in out.out
        assert "could not be read after the all-tray re-power" in out.err

    def test_expected_mesh_is_the_link_set_read_on_om_f3_gbh02(self):
        assert len(BH_GALAXY_INBOX_MESH) == 120
        assert _links_down(GALAXY_LINKS) == set()
        down = _links_down(GALAXY_LINKS - F3_GBH02_LINKS_DOWN)
        assert down == {
            ((1, n, 6 + k), (2, n, 6 + k)) for n in (5, 6, 7) for k in (0, 1)
        }

    def test_wormhole_galaxy_reads_no_links(self, capsys):
        with patch("tt_smi.reset.read_inbox_links") as read:
            check_inbox_links_after_repower({2}, True, True)
        assert not read.called
        assert "no expected in-box link map" in capsys.readouterr().err

    def test_links_never_readable_are_not_judged(self, tmp_path, capsys):
        host = self._host(tmp_path, eth_unreadable="always")
        code, _, _ = host.reset(ok=False, missing=[self.LOST])
        assert code is None
        assert host.ipmi == [("0x2", self.NR)]
        assert "in-box ETH links not checked" in capsys.readouterr().err

    def test_bmc_gone_before_link_repower_exits_1(self, tmp_path, capsys):
        host = self._host(tmp_path, eth_drop="both")
        host.mc_info = ["ok", "fail"]
        code, _, _ = host.reset(ok=False, missing=[self.LOST])
        out = capsys.readouterr()
        assert code == 1
        assert host.ipmi == [("0x2", self.NR)]
        assert "In-box ETH links not recovered" in out.err

    def test_reinit_recovery_also_checks_links(self, tmp_path, capsys):
        # The 0xffffffff re-init path (task #3973) re-powers tray 2 alone too.
        host = FakeGalaxy(tmp_path, bar_ones={self.LOST: "tray"}, eth_drop="both")
        code, _, _ = host.reset(ok=True)
        assert code is None
        assert host.discovers == 2
        assert host.ipmi == [("0x2", self.NR), ("0x3", self.NR)]

    def test_reinit_recovery_links_stay_down_exits_1(self, tmp_path, capsys):
        host = FakeGalaxy(tmp_path, bar_ones={self.LOST: "tray"}, eth_drop="never")
        code, _, _ = host.reset(ok=True)
        assert code == 1
        assert [b for b, _ in host.ipmi] == ["0x2", "0x3", "0xF"]
        assert "in-box ETH link(s) down after the all-tray re-power" in (
            capsys.readouterr().err
        )

    @pytest.mark.parametrize(
        "kind,value",
        [
            (SmiDeviceTargetKind.PCI_BDF, ["0000:47:00.0"]),
            (SmiDeviceTargetKind.DEV_TENSTORRENT_ID, [14]),
        ],
    )
    def test_targeted_reset_reads_no_links(self, tmp_path, kind, value):
        host = self._host(tmp_path, eth_drop="never")
        host.heal[self.LOST] = "rescan"
        host.reset(ok=False, reset_input=SmiDeviceInput(kind, value))
        assert host.link_reads == 0 and host.ipmi == []


class TestParseResetInput:
    """Unit tests for device-selection parsing used by ``tt-smi -r`` (no hardware)."""

    # --- classify_single_input: UMD logical IDs (ints as strings) ---

    def test_classify_umd_logical_id_single(self):
        assert classify_single_input("0") == (SmiDeviceTargetKind.UMD_LOGICAL_ID, 0)
        assert classify_single_input(" 42 ") == (SmiDeviceTargetKind.UMD_LOGICAL_ID, 42)
        assert classify_single_input("-1") == (SmiDeviceTargetKind.UMD_LOGICAL_ID, -1)

    # --- PCI BDF ---

    def test_classify_pci_bdf(self):
        assert classify_single_input("0000:0a:00.0") == (
            SmiDeviceTargetKind.PCI_BDF,
            "0000:0a:00.0",
        )
        assert classify_single_input("ABCD:EF:01.2") == (
            SmiDeviceTargetKind.PCI_BDF,
            "ABCD:EF:01.2",
        )

    def test_classify_bdf_rejects_short_domain(self):
        with pytest.raises(ValueError, match="Invalid device target"):
            classify_single_input("0a:00.0")

    # --- /dev/tenstorrent/<id> ---

    def test_classify_dev_tenstorrent_path(self):
        assert classify_single_input("/dev/tenstorrent/0") == (
            SmiDeviceTargetKind.DEV_TENSTORRENT_ID,
            0,
        )
        assert classify_single_input("/dev/tenstorrent/12") == (
            SmiDeviceTargetKind.DEV_TENSTORRENT_ID,
            12,
        )

    def test_classify_dev_path_rejects_non_numeric_suffix(self):
        with pytest.raises(ValueError, match="Invalid path"):
            classify_single_input("/dev/tenstorrent/abc")

    def test_classify_dev_path_case_sensitive_prefix(self):
        """Only lowercase /dev/tenstorrent/ is accepted (not a filesystem path check)."""
        with pytest.raises(ValueError, match="Invalid device target"):
            classify_single_input("/dev/Tenstorrent/0")

    # --- invalid / special inputs ---

    def test_classify_empty_input(self):
        with pytest.raises(ValueError, match="Empty input"):
            classify_single_input("")

    def test_classify_all_is_reserved(self):
        with pytest.raises(ValueError, match="only argument"):
            classify_single_input("all")

    def test_classify_garbage(self):
        with pytest.raises(ValueError, match="Invalid device target"):
            classify_single_input("not-a-target")

    # --- parse_smi_device_input: ALL ---

    def test_parse_all_empty_or_none(self):
        assert parse_smi_device_input(None) == SmiDeviceInput(
            type=SmiDeviceTargetKind.ALL, value=None
        )
        assert parse_smi_device_input([]) == SmiDeviceInput(
            type=SmiDeviceTargetKind.ALL, value=None
        )

    def test_parse_all_explicit(self):
        assert parse_smi_device_input(["all"]) == SmiDeviceInput(
            type=SmiDeviceTargetKind.ALL, value=None
        )
        assert parse_smi_device_input([" ALL "]) == SmiDeviceInput(
            type=SmiDeviceTargetKind.ALL, value=None
        )

    # --- parse_smi_device_input: homogeneous types ---

    def test_parse_umd_logical_ids_sorted_deduped(self):
        r = parse_smi_device_input(["2", "0", "1", "0"])
        assert r.type == SmiDeviceTargetKind.UMD_LOGICAL_ID
        assert r.value == [0, 1, 2]

    def test_parse_umd_comma_separated_in_one_arg(self):
        r = parse_smi_device_input(["0, 1 ,2"])
        assert r.type == SmiDeviceTargetKind.UMD_LOGICAL_ID
        assert r.value == [0, 1, 2]

    def test_parse_pci_bdf_list(self):
        r = parse_smi_device_input(["0000:0a:00.0", "0000:0b:00.0"])
        assert r.type == SmiDeviceTargetKind.PCI_BDF
        assert r.value == ["0000:0a:00.0", "0000:0b:00.0"]

    def test_parse_pci_bdf_comma_separated(self):
        r = parse_smi_device_input(["0000:0a:00.0,0000:0b:00.0"])
        assert r.type == SmiDeviceTargetKind.PCI_BDF
        assert r.value == ["0000:0a:00.0", "0000:0b:00.0"]

    def test_parse_dev_tenstorrent_ids_sorted(self):
        r = parse_smi_device_input(["/dev/tenstorrent/2", "/dev/tenstorrent/0"])
        assert r.type == SmiDeviceTargetKind.DEV_TENSTORRENT_ID
        assert r.value == [0, 2]

    def test_parse_dev_tenstorrent_comma_separated(self):
        r = parse_smi_device_input(["/dev/tenstorrent/1,/dev/tenstorrent/0"])
        assert r.type == SmiDeviceTargetKind.DEV_TENSTORRENT_ID
        assert r.value == [0, 1]

    # --- parse_smi_device_input: mixed kinds -> exit 1 ---

    def test_parse_mixed_int_and_bdf_exits(self):
        with pytest.raises(SystemExit) as exc:
            parse_smi_device_input(["0", "0000:0a:00.0"])
        assert exc.value.code == 1

    def test_parse_mixed_int_and_dev_path_exits(self):
        with pytest.raises(SystemExit) as exc:
            parse_smi_device_input(["0", "/dev/tenstorrent/0"])
        assert exc.value.code == 1

    def test_parse_mixed_bdf_and_dev_path_exits(self):
        with pytest.raises(SystemExit) as exc:
            parse_smi_device_input(["0000:0a:00.0", "/dev/tenstorrent/0"])
        assert exc.value.code == 1

    def test_parse_invalid_input_exits(self):
        with pytest.raises(SystemExit) as exc:
            parse_smi_device_input(["bogus"])
        assert exc.value.code == 1


def redetect_devices(use_umd: bool) -> List[int]:
    if use_umd:
        return list(PCIDevice.enumerate_devices_info())
    return pci_scan()


@pytest.mark.requires_hardware
class TestPciDriverReset:
    def test_pci_reset_all_devices(self, reset_test_config):
        """
        Test resetting all PCI devices (invoked by tt-smi -r).

        Passes if the reset is successful and the same number of devices are
        detected before and after.
        """
        pci_indices, use_umd = reset_test_config
        # Match `tt-smi -r` / `tt-smi -r all`: reset all devices (SmiDeviceInput), not a raw index list.
        pci_board_reset(
            SmiDeviceInput(type=SmiDeviceTargetKind.ALL, value=None),
            reinit=True,
            use_umd=use_umd,
        )
        post_reset_devices = redetect_devices(use_umd)
        assert len(post_reset_devices) == len(pci_indices)

    def test_pci_reset_umd_id(self, reset_test_config):
        """
        Reset UMD logical device 0 (tt-smi -r 0).

        Only runs under the UMD parametrization; skips if logical id 0 is absent.
        Passes if device count is unchanged after reset.
        """
        pci_indices, use_umd = reset_test_config
        if not use_umd:
            pytest.skip("UMD logical ID reset is only supported with UMD backend")
        if 0 not in pci_indices:
            pytest.skip("No device with UMD logical ID 0")

        pci_board_reset(
            SmiDeviceInput(type=SmiDeviceTargetKind.UMD_LOGICAL_ID, value=[0]),
            reinit=True,
            use_umd=True,
        )
        post_reset_devices = redetect_devices(use_umd)
        assert len(post_reset_devices) == len(pci_indices)

    def test_pci_reset_dev_id(
        self,
        reset_test_config_with_backend: Tuple[Tuple[List[int], bool], TTSMIBackend],
    ):
        """
        Reset the first enumerated device by /dev/tenstorrent/<N> (tt-smi -r /dev/tenstorrent/N).

        Resolves N from sysfs for board index 0; runs for both Luwen and UMD.
        """
        (pci_indices, use_umd), backend = reset_test_config_with_backend
        if not pci_indices:
            pytest.skip("No Tenstorrent devices")
        bdf = backend.get_pci_bdf(0)
        if bdf == "N/A":
            pytest.skip("No local PCI BDF for first device (remote)")
        dev_id = get_dev_id_from_bdf(bdf)
        pci_board_reset(
            SmiDeviceInput(type=SmiDeviceTargetKind.DEV_TENSTORRENT_ID, value=[dev_id]),
            reinit=True,
            use_umd=use_umd,
        )
        post_reset_devices = redetect_devices(use_umd)
        assert len(post_reset_devices) == len(pci_indices)

    def test_pci_reset_pci_bdf(
        self,
        reset_test_config_with_backend: Tuple[Tuple[List[int], bool], TTSMIBackend],
    ):
        """
        Reset the first enumerated device by /dev/tenstorrent/<N> (tt-smi -r /dev/tenstorrent/N).

        Resolves N from sysfs for board index 0; runs for both Luwen and UMD.
        """
        (pci_indices, use_umd), backend = reset_test_config_with_backend
        if not pci_indices:
            pytest.skip("No Tenstorrent devices")
        bdf = backend.get_pci_bdf(0)
        if bdf == "N/A":
            pytest.skip("No local PCI BDF for first device (remote)")
        pci_board_reset(
            SmiDeviceInput(type=SmiDeviceTargetKind.PCI_BDF, value=[bdf]),
            reinit=True,
            use_umd=use_umd,
        )
        post_reset_devices = redetect_devices(use_umd)
        assert len(post_reset_devices) == len(pci_indices)

    def test_pci_reset_all_devices_stress(self, reset_test_config):
        """
        Test resetting all PCI devices NUM_RESETS_STRESS_TEST times in a row.

        Passes if the resets are successful and the same number of devices are
        detected before and after each reset.
        """
        pci_indices, use_umd = reset_test_config
        for _ in range(NUM_RESETS_STRESS_TEST):
            pci_board_reset(
                SmiDeviceInput(type=SmiDeviceTargetKind.ALL, value=None),
                reinit=True,
                use_umd=use_umd,
            )
            post_reset_devices = redetect_devices(use_umd)
            assert len(post_reset_devices) == len(pci_indices)


@pytest.mark.requires_hardware
@pytest.mark.requires_galaxy
class TestGalaxyReset:
    def test_glx_reset(self, reset_test_config):
        """
        Test galaxy 6U trays reset (invoked by tt-smi -glx_reset).

        Passes if the reset is successful and the same number of devices are
        detected before and after.
        """
        pci_indices, use_umd = reset_test_config
        # Expect SystemExit with return code 0 on successful reset
        with pytest.raises(SystemExit) as exc_info:
            glx_6u_trays_reset(use_umd=use_umd)

        assert exc_info.value.code == 0

        post_reset_devices = redetect_devices(use_umd)
        assert len(post_reset_devices) == len(pci_indices)

    def test_glx_reset_stress(self, reset_test_config):
        """
        Test galaxy 6U trays reset NUM_RESETS_STRESS_TEST times in a row.

        Passes if the resets are successful and the same number of devices are
        detected before and after each reset.
        """
        pci_indices, use_umd = reset_test_config
        for _ in range(NUM_RESETS_STRESS_TEST):
            # Expect SystemExit with return code 0 on successful reset
            with pytest.raises(SystemExit) as exc_info:
                glx_6u_trays_reset(use_umd=use_umd)

            assert exc_info.value.code == 0

            post_reset_devices = redetect_devices(use_umd)
            assert len(post_reset_devices) == len(pci_indices)
