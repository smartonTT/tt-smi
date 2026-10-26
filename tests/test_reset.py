# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import itertools
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
    GLX_IPMI_OP_MODE_NO_RETIMER,
    GLX_IPMI_OP_MODE_WITH_RETIMER,
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
    f"0000:{base + n:02x}:00.0" for base in (0x00, 0x40, 0xC0, 0x80) for n in range(1, 9)
]


class FakeGalaxy:
    """A host's chips for ``tt-smi -r`` unit tests: fake sysfs, enumeration, IPMI and rescan.

    ``states``: bdf -> ok, marker, ones or gone (default ok). ``heal``: bdf -> how a chip
    that is not ok comes back: "tray" (its tray re-powered), "all" (all four trays
    re-powered), "rescan" (UMD PCI rescan), "sysfs" (sysfs PCI rescan) or "never" (default). ``drop_on_repower``: chips lost
    when their tray is re-powered.
    """

    def __init__(self, tmp_path, states=None, heal=None, bdfs=None, subsystem_id=0x47,
                 absent=(), drop_on_repower=(), reject_per_tray=False):
        self.root = tmp_path / "devices"
        self.root.mkdir()
        self.rescan_file = tmp_path / "rescan"
        self.rescan_file.write_text("")
        self.bdfs = list(bdfs or GALAXY_BDFS)
        self.subsystem_id = subsystem_id
        self.heal = dict(heal or {})
        self.drop_on_repower = set(drop_on_repower)
        self.reject_per_tray = reject_per_tray
        self.ipmi = []
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
        bdfs = self.enumerated if self.enumerated is not None else [
            b for b in self.bdfs if self.state[b] not in ("gone",)
        ]
        self.enumerated = None  # Only the first enumeration (before the reset) may hide chips.
        return {
            self.bdfs.index(b): MagicMock(pci_bdf=b, subsystem_id=self.subsystem_id, device_id=0xB140)
            for b in bdfs
        }

    def run(self, cmd, **kwargs):
        if cmd[:3] == ["sudo", "ipmitool", "raw"]:
            bitmap = int(cmd[5], 16)
            self.ipmi.append((cmd[5], cmd[7]))
            if self.reject_per_tray and bitmap != 0xF:
                return MagicMock(returncode=1, stdout="", stderr=BMC_C9_STDERR)
            trays = {t for t in (1, 2, 3, 4) if bitmap & (1 << (t - 1))}
            for bdf in self.drop_on_repower:
                if self.tray(bdf) in trays:
                    self._set(bdf, "gone")
            self._heal(("all",) if bitmap == 0xF else ())
            for bdf, mode in self.heal.items():
                if mode == "tray" and self.tray(bdf) in trays:
                    self._set(bdf, "ok")
                    self.heal[bdf] = "done"
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    def rescan_pci_bdfs(self, bdfs, timeout_s=10.0):
        self._heal(("rescan",))
        return [b for b in bdfs if self.state.get(b) != "ok"]

    def ioctl(self, ids, flag):
        self.ioctls.append((flag, sorted(ids)))
        return []

    def ioctl_ids(self, flag):
        return [ids for f, ids in self.ioctls if f == flag]

    def reset(self, ok=True, missing=(), reset_input=None, reinit=True, with_result=True,
              rescan=True):
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
        with patch("tt_smi.reset.SYSFS_PCI_DEVICES", str(self.root)), \
             patch("tt_smi.reset.SYSFS_PCI_RESCAN", str(self.rescan_file)), \
             patch("tt_smi.reset.PCIDevice.enumerate_devices_info", side_effect=self.enumerate_devices_info), \
             patch("tt_smi.reset.WarmReset", warm), \
             patch("tt_smi.reset.subprocess.run", side_effect=self.run), \
             patch("tt_smi.reset.parallel_reset_device_ioctl", side_effect=self.ioctl), \
             patch("tt_smi.reset.timed_wait"), \
             patch("tt_smi.reset.time", clock), \
             patch("tt_smi.reset.should_use_secondary_bus_reset", return_value=True), \
             patch("tt_smi.reset.TopologyDiscovery") as topo, \
             patch("tt_smi.reset.get_default_discovery_options"):
            try:
                pci_board_reset(reset_input, reinit=reinit, use_umd=True)
                code = None
            except SystemExit as e:
                code = e.code
        return code, topo.discover.called, warm


class TestUmdWarmResetResult:
    """``tt-smi -r`` checks UMD warm_reset's result and names chips that did not come back (no hardware)."""

    BDFS = ["0000:02:00.0", "0000:03:00.0", "0000:04:00.0"]

    def _reset(self, tmp_path, states, ok, reset_input=None, reinit=True):
        host = FakeGalaxy(tmp_path, states=states)
        return host.reset(ok=ok, reset_input=reset_input, reinit=reinit, with_result=False)

    def test_failed_reset_names_chips_and_exits_nonzero(self, tmp_path, capsys):
        states = {self.BDFS[0]: "ok", self.BDFS[1]: "marker", self.BDFS[2]: "ones"}
        code, reinit_ran, _ = self._reset(tmp_path, states, ok=False)
        err = capsys.readouterr().err
        assert code == 1
        assert not reinit_ran
        assert "Chip 0000:03:00.0 (tray 1) did not come back from reset: reset marker still set" in err
        assert "Chip 0000:04:00.0 (tray 1) did not come back from reset: config space reads all ones" in err
        assert "0000:02:00.0 (tray 1) did not come back" not in err
        # The tray re-power resets 03:00.0 (its marker is not checked after a re-power); 04:00.0 stays dead.
        assert "Hardware fault: chip 0000:04:00.0 (tray 1) still missing after tray re-power" in err
        assert "1 chip(s) did not come back: 0000:04:00.0" in err

    def test_chip_gone_from_pci_is_named(self, tmp_path, capsys):
        states = {self.BDFS[0]: "ok", self.BDFS[1]: "gone", self.BDFS[2]: "ok"}
        code, _, _ = self._reset(tmp_path, states, ok=False)
        err = capsys.readouterr().err
        assert code == 1
        assert "Chip 0000:03:00.0 (tray 1) did not come back from reset: gone from PCI" in err

    def test_failed_reset_with_no_chip_named_still_fails(self, tmp_path, capsys):
        code, reinit_ran, _ = self._reset(tmp_path, {b: "ok" for b in self.BDFS}, ok=False)
        err = capsys.readouterr().err
        assert code == 1
        assert not reinit_ran
        assert "UMD warm reset reported failure" in err
        assert "no chip reads back as failed" in err
        assert "every chip reads back" not in err

    def test_success_reinits(self, tmp_path):
        code, reinit_ran, warm = self._reset(tmp_path, {b: "ok" for b in self.BDFS}, ok=True)
        assert code is None
        assert reinit_ran
        warm.warm_reset.assert_called_once_with(list(range(32)), secondary_bus_reset=True)

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
        code, _, warm = self._reset(tmp_path, states, ok=False, reset_input=SmiDeviceInput(kind, value))
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
        warm.warm_reset_with_result.assert_called_once_with(list(range(32)), secondary_bus_reset=True)
        warm.warm_reset.assert_not_called()
        assert host.ipmi == []

    def test_tray_repower_brings_back_missing_chip(self, tmp_path, capsys):
        host = FakeGalaxy(tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "tray"})
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

    def test_umd_missing_list_alone_triggers_recovery(self, tmp_path, capsys):
        # UMD names the chip even though its sysfs config still reads fine.
        host = FakeGalaxy(tmp_path)
        code, _, _ = host.reset(ok=False, missing=[self.LOST])
        assert code is None
        assert host.ipmi == [("0x2", GLX_IPMI_OP_MODE_NO_RETIMER)]
        assert "UMD warm reset reports it did not come back" in capsys.readouterr().err

    def test_per_tray_reset_rejected_resets_all_trays(self, tmp_path):
        host = FakeGalaxy(tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "all"},
                          reject_per_tray=True)
        code, _, _ = host.reset(ok=False, missing=[self.LOST])
        assert code is None
        assert host.ipmi == [("0x2", GLX_IPMI_OP_MODE_NO_RETIMER), ("0xF", GLX_IPMI_OP_MODE_NO_RETIMER)]

    def test_round_two_repowers_all_trays(self, tmp_path, capsys):
        host = FakeGalaxy(tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "all"})
        code, reinit_ran, _ = host.reset(ok=False, missing=[self.LOST])
        assert code is None and reinit_ran
        assert [b for b, _ in host.ipmi] == ["0x2", "0xF"]
        assert host.ioctl_ids(IoctlResetFlags.POST_RESET)[-1] == list(range(32))
        assert "Recount after round 2: 32/32 chips" in capsys.readouterr().out

    def test_chip_that_never_returns_is_a_hardware_fault(self, tmp_path, capsys):
        host = FakeGalaxy(tmp_path, states={self.LOST: "gone"})
        code, reinit_ran, _ = host.reset(ok=False, missing=[self.LOST])
        out = capsys.readouterr()
        assert code == 1
        assert not reinit_ran
        assert [b for b, _ in host.ipmi] == ["0x2", "0xF"]  # at most two rounds
        assert (
            f"Hardware fault: chip {self.LOST} (tray 2) still missing after tray re-power and PCI rescan: gone from PCI"
            in out.err
        )
        assert "Recount after round 2: 31/32 chips" in out.out
        assert "UBB" not in out.out + out.err

    def test_one_dead_chip_does_not_stop_the_others(self, tmp_path, capsys):
        dead = "0000:c3:00.0"  # tray 3
        host = FakeGalaxy(tmp_path, states={self.LOST: "gone", dead: "ones"},
                          heal={self.LOST: "tray"})
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
        host = FakeGalaxy(tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "tray"},
                          drop_on_repower=[other])
        code, _, _ = host.reset(ok=False, missing=[self.LOST])
        assert code == 1
        assert f"Hardware fault: chip {other} (tray 2) still missing" in capsys.readouterr().err

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
        host = FakeGalaxy(tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "tray"})
        code, _, warm = host.reset(ok=True, with_result=False, rescan=False)
        assert code is None
        warm.warm_reset.assert_called_once_with(list(range(32)), secondary_bus_reset=True)
        assert host.ipmi == [("0x2", GLX_IPMI_OP_MODE_NO_RETIMER)]
        assert host.rescan_file.read_text() == "1"

    def test_umd_rescan_error_falls_back_to_sysfs(self, tmp_path):
        host = FakeGalaxy(tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "tray"})
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
        with patch("tt_smi.reset.subprocess.run", return_value=MagicMock(returncode=0)) as run:
            assert _sysfs_write(str(target), "1")
        assert run.call_args[0][0] == ["sudo", "-n", "tee", str(target)]
        assert run.call_args[1]["input"] == "1"

    def test_non_galaxy_only_rescans(self, tmp_path, capsys):
        bdfs = ["0000:01:00.0", "0000:02:00.0"]
        host = FakeGalaxy(tmp_path, bdfs=bdfs, subsystem_id=0x40, states={bdfs[1]: "gone"},
                          heal={bdfs[1]: "rescan"})
        code, reinit_ran, _ = host.reset(ok=False, missing=[bdfs[1]])
        assert code is None and reinit_ran
        assert host.ipmi == []
        assert "Chip 0000:02:00.0 did not come back" in capsys.readouterr().err

    def test_non_galaxy_chip_that_stays_gone_fails(self, tmp_path, capsys):
        bdfs = ["0000:01:00.0", "0000:02:00.0"]
        host = FakeGalaxy(tmp_path, bdfs=bdfs, subsystem_id=0x40, states={bdfs[1]: "gone"})
        code, _, _ = host.reset(ok=False, missing=[bdfs[1]])
        assert code == 1
        assert host.ipmi == []
        assert "Hardware fault: chip 0000:02:00.0 still missing after PCI rescan" in capsys.readouterr().err

    def test_post_reset_failure_is_not_a_recovery(self, tmp_path):
        host = FakeGalaxy(tmp_path, states={self.LOST: "gone"}, heal={self.LOST: "tray"})
        bad = GALAXY_BDFS.index(self.LOST)
        host.ioctl = lambda ids, flag: (
            host.ioctls.append((flag, sorted(ids))) or ([bad] if flag == IoctlResetFlags.POST_RESET else [])
        )
        code, reinit_ran, _ = host.reset(ok=False, missing=[self.LOST])
        assert code == 1
        assert not reinit_ran


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
        self, reset_test_config_with_backend: Tuple[Tuple[List[int], bool], TTSMIBackend]
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
        self, reset_test_config_with_backend: Tuple[Tuple[List[int], bool], TTSMIBackend]
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
