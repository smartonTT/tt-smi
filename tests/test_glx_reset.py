# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mock-only tests for tt-smi -glx_reset (no hardware, no ipmitool, no device nodes):
- #278: op_mode 0x0 only after a 0xc9 rejection of op_mode 0x3, and only by opt-in
- #279: BMC preflight before any chip reset; bounded BMC retry; chips held in reset reported
- rescan of each missing chip's root port and recount by BDF until 32/32 or timeout
"""

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from tt_smi import reset
from tt_smi import tt_smi as cli
from tt_smi.constants import BH_UBB_BUS_IDS, WH_UBB_BUS_IDS
from tt_smi.utils import IoctlResetFlags
from tt_smi.reset import (
    GLX_EXIT_BMC_FAILED,
    GLX_EXIT_CHIPS_MISSING,
    GLX_IPMI_OP_MODE_NO_RETIMER,
    GLX_IPMI_OP_MODE_WITH_RETIMER,
    GalaxyIpmiParamError,
    GalaxyResetError,
    bmc_answers,
    galaxy_expected_chips,
    galaxy_rescan_and_recount,
    glx_6u_trays_reset,
    run_galaxy_ipmi_reset,
    run_galaxy_ipmi_reset_with_bmc_retry,
    tray_of_bdf,
)

C9 = (
    "Unable to send RAW command (channel=0x0 netfn=0x30 lun=0x0 cmd=0x8b rsp=0xc9): "
    "Parameter out of range"
)
NO_DATA = "Unable to send RAW command (channel=0x0 netfn=0x30 lun=0x0 cmd=0x8b): No data available"


def ok():
    return MagicMock(returncode=0, stdout="", stderr="")


def fail(text):
    return MagicMock(returncode=1, stdout="", stderr=text)


def op_modes(mock_run):
    """op_mode of each raw 0x30 0x8B call."""
    return [
        c[0][0][7]
        for c in mock_run.call_args_list
        if c[0][0][2:5] == ["raw", "0x30", "0x8B"]
    ]


# ---------------------------------------------------------------- fake sysfs

BH_CHIPS = {
    # tray 1 (bus 0x0X), tray 2 (0x4X), tray 3 (0xCX), tray 4 (0x8X): 8 chips each
    **{f"0000:{b:02x}:00.0": (f"0000:00:01.{b}", "pci0000:00") for b in range(1, 9)},
    **{
        f"0000:{b:02x}:00.0": (f"0000:40:01.{b - 0x40}", "pci0000:40")
        for b in range(0x41, 0x49)
    },
    **{
        f"0000:{b:02x}:00.0": (f"0000:c0:01.{b - 0xC0}", "pci0000:c0")
        for b in range(0xC1, 0xC9)
    },
    **{
        f"0000:{b:02x}:00.0": (f"0000:80:01.{b - 0x80}", "pci0000:80")
        for b in range(0x81, 0x89)
    },
}


class FakeSysfs:
    """A sysfs tree with root ports and Tenstorrent chips, in tmp_path."""

    def __init__(self, root, chips=BH_CHIPS, device="0xb140"):
        self.root = str(root)
        self.device = device
        self.chips = dict(chips)
        os.makedirs(os.path.join(self.root, "bus", "pci", "devices"))
        os.makedirs(os.path.join(self.root, "bus", "pci", "drivers", "tenstorrent"))
        for bdf, (port, rc) in self.chips.items():
            self._add_port(port, rc, int(bdf.split(":")[1], 16))
            self.add_chip(bdf)

    def _write(self, path, text):
        with open(path, "w") as f:
            f.write(text)

    def _link(self, bdf, target):
        link = os.path.join(self.root, "bus", "pci", "devices", bdf)
        if not os.path.lexists(link):
            os.symlink(target, link)

    def _port_dir(self, port, rc):
        return os.path.join(self.root, "devices", rc, port)

    def _add_port(self, port, rc, sec_bus):
        d = self._port_dir(port, rc)
        if os.path.isdir(d):
            return
        os.makedirs(d)
        self._write(os.path.join(d, "vendor"), "0x1022\n")
        self._write(os.path.join(d, "device"), "0x1483\n")
        self._write(os.path.join(d, "secondary_bus_number"), f"{sec_bus}\n")
        self._write(os.path.join(d, "rescan"), "")
        self._link(port, d)

    def add_chip(self, bdf, answers=True):
        port, rc = self.chips[bdf]
        d = os.path.join(self._port_dir(port, rc), bdf)
        os.makedirs(d, exist_ok=True)
        self._write(os.path.join(d, "vendor"), "0x1e52\n")
        self._write(os.path.join(d, "device"), f"{self.device}\n")
        with open(os.path.join(d, "config"), "wb") as f:
            f.write((b"\x52\x1e" if answers else b"\xff\xff") + b"\x00" * 62)
        drv = os.path.join(d, "driver")
        if not os.path.lexists(drv):
            os.symlink(
                os.path.join(self.root, "bus", "pci", "drivers", "tenstorrent"), drv
            )
        link = os.path.join(self.root, "bus", "pci", "devices", bdf)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(d, link)

    def remove_chip(self, bdf):
        port, rc = self.chips[bdf]
        d = os.path.join(self._port_dir(port, rc), bdf)
        for name in os.listdir(d):
            os.remove(os.path.join(d, name))
        os.rmdir(d)
        os.remove(os.path.join(self.root, "bus", "pci", "devices", bdf))


@pytest.fixture
def no_sleep():
    with patch("tt_smi.reset.time.sleep"):
        yield


# ---------------------------------------------------------------- #278


class TestLegacyFallbackOnlyOnC9OptIn:
    def test_non_c9_failure_raises_without_op0(self):
        with patch("tt_smi.reset.subprocess.run", side_effect=[fail(NO_DATA)]) as run:
            with pytest.raises(RuntimeError, match="No data available"):
                run_galaxy_ipmi_reset()
        assert op_modes(run) == [GLX_IPMI_OP_MODE_NO_RETIMER]

    @pytest.mark.parametrize(
        "text", ["Could not open device", "timeout", "rsp=0xc0 Node busy", ""]
    )
    def test_other_failures_never_reach_op0(self, text):
        res = MagicMock(returncode=2, stdout="", stderr=text)
        with patch("tt_smi.reset.subprocess.run", side_effect=[res]) as run:
            with pytest.raises(RuntimeError, match="IPMI command failed"):
                run_galaxy_ipmi_reset(allow_legacy_retimer_reset=True)
        assert op_modes(run) == [GLX_IPMI_OP_MODE_NO_RETIMER]

    def test_c9_without_opt_in_raises_and_names_flag(self):
        with patch("tt_smi.reset.subprocess.run", side_effect=[fail(C9)]) as run:
            with pytest.raises(
                GalaxyIpmiParamError, match="--allow_legacy_retimer_reset"
            ):
                run_galaxy_ipmi_reset()
        assert op_modes(run) == [GLX_IPMI_OP_MODE_NO_RETIMER]

    def test_c9_with_opt_in_falls_back_to_op0(self):
        with patch("tt_smi.reset.subprocess.run", side_effect=[fail(C9), ok()]) as run:
            run_galaxy_ipmi_reset(allow_legacy_retimer_reset=True)
        assert op_modes(run) == [
            GLX_IPMI_OP_MODE_NO_RETIMER,
            GLX_IPMI_OP_MODE_WITH_RETIMER,
        ]


# ---------------------------------------------------------------- #279: BMC retry


class TestBmcRetry:
    def test_bmc_answers_ok_and_failures(self):
        with patch("tt_smi.reset.subprocess.run", return_value=ok()) as run:
            assert bmc_answers() == ""
        assert run.call_args[0][0] == ["sudo", "ipmitool", "mc", "info"]
        assert run.call_args[1]["timeout"] == reset.BMC_PREFLIGHT_TIMEOUT_S
        with patch("tt_smi.reset.subprocess.run", return_value=fail(NO_DATA)):
            assert "No data available" in bmc_answers()
        with patch(
            "tt_smi.reset.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ipmitool", timeout=15),
        ):
            assert "no answer within" in bmc_answers()

    def test_resends_op3_once_bmc_answers(self, no_sleep):
        calls = [fail(NO_DATA), fail("mc info: no data"), ok(), ok()]
        with patch("tt_smi.reset.subprocess.run", side_effect=calls) as run:
            run_galaxy_ipmi_reset_with_bmc_retry(timeout_s=600, interval_s=1)
        assert op_modes(run) == [
            GLX_IPMI_OP_MODE_NO_RETIMER,
            GLX_IPMI_OP_MODE_NO_RETIMER,
        ]
        assert run.call_count == 4

    def test_gives_up_after_timeout_without_op0(self, no_sleep):
        clock = iter(range(0, 10_000, 100))
        with patch(
            "tt_smi.reset.time.monotonic", side_effect=lambda: next(clock)
        ), patch(
            "tt_smi.reset.subprocess.run", side_effect=lambda *a, **k: fail(NO_DATA)
        ) as run:
            with pytest.raises(RuntimeError, match="No data available"):
                run_galaxy_ipmi_reset_with_bmc_retry(timeout_s=300, interval_s=10)
        assert GLX_IPMI_OP_MODE_WITH_RETIMER not in op_modes(run)

    def test_c9_is_not_retried(self, no_sleep):
        with patch("tt_smi.reset.subprocess.run", side_effect=[fail(C9)]) as run:
            with pytest.raises(GalaxyIpmiParamError):
                run_galaxy_ipmi_reset_with_bmc_retry()
        assert run.call_count == 1

    def test_missing_ipmitool_is_not_retried(self, no_sleep):
        with patch(
            "tt_smi.reset.subprocess.run", side_effect=FileNotFoundError("sudo")
        ) as run:
            with pytest.raises(RuntimeError, match="ipmitool"):
                run_galaxy_ipmi_reset_with_bmc_retry()
        assert run.call_count == 1


# ---------------------------------------------------------------- trays and recount


class TestTrayOf:
    @pytest.mark.parametrize(
        "bdf,tray",
        [
            ("0000:01:00.0", "1"),
            ("0000:47:00.0", "2"),
            ("0000:c1:00.0", "3"),
            ("0000:84:00.0", "4"),
            ("0000:85:00.0", "4"),
            ("0000:20:00.0", "?"),
            ("junk", "?"),
        ],
    )
    def test_bh(self, bdf, tray):
        assert tray_of_bdf(bdf, BH_UBB_BUS_IDS) == tray

    def test_wh(self):
        assert tray_of_bdf("0000:c1:00.0", WH_UBB_BUS_IDS) == "1"
        assert tray_of_bdf("0000:01:00.0", WH_UBB_BUS_IDS) == "3"


class TestExpectedChips:
    def test_all_32_with_root_ports(self, tmp_path):
        fs = FakeSysfs(tmp_path)
        exp = galaxy_expected_chips(fs.root)
        assert len(exp) == 32
        assert exp["0000:47:00.0"] == "0000:40:01.7"

    def test_chip_lost_before_reset_named_from_empty_port(self, tmp_path):
        fs = FakeSysfs(tmp_path)
        fs.remove_chip("0000:47:00.0")
        exp = galaxy_expected_chips(fs.root)
        assert len(exp) == 32
        assert exp["0000:47:00.0"] == "0000:40:01.7"


class TestRescanRecount:
    def test_all_back_at_once(self, tmp_path, no_sleep):
        fs = FakeSysfs(tmp_path)
        exp = galaxy_expected_chips(fs.root)
        with patch("tt_smi.reset._rescan_port") as rescan:
            assert galaxy_rescan_and_recount(exp, sysfs=fs.root) == ([], 32)
        rescan.assert_not_called()

    def test_missing_chip_back_after_rescan_of_its_port(self, tmp_path, no_sleep):
        fs = FakeSysfs(tmp_path)
        exp = galaxy_expected_chips(fs.root)
        fs.remove_chip("0000:47:00.0")
        ports = []

        def rescan(port, sysfs):
            ports.append(port)
            fs.add_chip("0000:47:00.0")
            return True

        with patch("tt_smi.reset._rescan_port", side_effect=rescan):
            assert galaxy_rescan_and_recount(exp, sysfs=fs.root, timeout_s=60) == (
                [],
                32,
            )
        assert ports == ["0000:40:01.7"]

    def test_all_ones_chip_counts_missing(self, tmp_path, no_sleep):
        fs = FakeSysfs(tmp_path)
        exp = galaxy_expected_chips(fs.root)
        fs.add_chip("0000:84:00.0", answers=False)
        clock = iter(range(0, 10_000, 50))
        with patch(
            "tt_smi.reset.time.monotonic", side_effect=lambda: next(clock)
        ), patch("tt_smi.reset._rescan_port", return_value=True):
            missing, on_pci = galaxy_rescan_and_recount(
                exp, sysfs=fs.root, timeout_s=120
            )
        assert missing == ["0000:84:00.0"] and on_pci == 31

    def test_one_failed_tray_does_not_stop_others(self, tmp_path, no_sleep):
        """Tray 1 never comes back and its rescans fail; tray 2's chip still comes back."""
        fs = FakeSysfs(tmp_path)
        exp = galaxy_expected_chips(fs.root)
        tray1 = [b for b in exp if tray_of_bdf(b) == "1"]
        for b in tray1 + ["0000:47:00.0"]:
            fs.remove_chip(b)

        def rescan(port, sysfs):
            if port.startswith("0000:00:"):
                return False
            fs.add_chip("0000:47:00.0")
            return True

        clock = iter(range(0, 10_000, 30))
        with patch(
            "tt_smi.reset.time.monotonic", side_effect=lambda: next(clock)
        ), patch("tt_smi.reset._rescan_port", side_effect=rescan):
            missing, on_pci = galaxy_rescan_and_recount(
                exp, sysfs=fs.root, timeout_s=120
            )
        assert missing == sorted(tray1) and on_pci == 24

    def test_rescan_port_failure_warns_and_returns_false(self, tmp_path, capsys):
        with patch(
            "tt_smi.reset.open", side_effect=OSError("No such device"), create=True
        ):
            assert reset._rescan_port("0000:40:01.7", str(tmp_path)) is False
        assert "rescan of root port 0000:40:01.7 failed" in capsys.readouterr().out


# ---------------------------------------------------------------- glx_6u_trays_reset flow


_REAL_READ = reset._read_arc_heartbeat


def _desc(fs):
    desc = MagicMock()
    desc.get_chip_pci_bdfs.return_value = dict(enumerate(sorted(fs.chips)))
    return desc


def _ticking():
    seen = {}

    def read(chip, use_umd):
        seen[chip] = seen.get(chip, 0) + 1
        return seen[chip]

    return read


@pytest.fixture
def glx(tmp_path, no_sleep):
    fs = FakeSysfs(tmp_path)
    ioctl_calls = []

    def ioctl(ids, flag):
        ioctl_calls.append(flag)
        return []

    clock = iter(range(0, 1_000_000, 50))  # never wait on real time
    with patch.object(reset, "SYSFS_ROOT", fs.root), patch(
        "tt_smi.reset.time.monotonic", side_effect=lambda: next(clock)
    ), patch(
        "tt_smi.reset.PCIDevice.enumerate_devices", return_value=list(range(32))
    ), patch(
        "tt_smi.reset.parallel_reset_device_ioctl", side_effect=ioctl
    ), patch(
        "tt_smi.reset.should_use_secondary_bus_reset", return_value=True
    ), patch(
        "tt_smi.reset.timed_wait"
    ), patch(
        "tt_smi.reset.TopologyDiscovery.discover",
        return_value=(_desc(fs), {i: f"chip{i}" for i in range(len(fs.chips))}),
    ), patch(
        "tt_smi.reset._read_arc_heartbeat", side_effect=_ticking()
    ):
        yield fs, ioctl_calls


def mc_info(aux="0x23"):
    text = (
        "Device ID : 32\nFirmware Revision : 0.05\nIPMI Version : 2.0\n"
        f"Aux Firmware Rev Info     : \n    {aux}\n    0x00\n    0x00\n    0x00\n"
    )
    return MagicMock(returncode=0, stdout=text, stderr="")


class TestBmcVersionDecision:
    """Old BMC: decided before any chip is reset. Mock only, 0 hardware tries."""

    C9 = "Unable to send RAW command (rsp=0xc9): Parameter out of range"

    def test_parse(self):
        assert reset.parse_bmc_version(mc_info("0x23").stdout) == (0, 5, 23)
        assert reset.parse_bmc_version(mc_info("0x21").stdout) == (0, 5, 21)
        assert reset.parse_bmc_version("") is None

    def test_old_bmc_no_flag_matches_main(self, glx):
        fs, ioctl_calls = glx

        def run(cmd, **kw):
            if cmd[2:5] == ["raw", "0x30", "0x8B"]:
                return fail(self.C9) if cmd[7] == "0x3" else ok()
            return mc_info("0x21")

        with patch("tt_smi.reset.subprocess.run", side_effect=run) as m:
            with pytest.raises(SystemExit) as exc:
                glx_6u_trays_reset(use_umd=True)
        assert exc.value.code == 0
        assert op_modes(m) == ["0x3", GLX_IPMI_OP_MODE_WITH_RETIMER]

    def test_old_bmc_without_flag_decided_before_resets(self, glx, capsys):
        fs, ioctl_calls = glx
        with patch("tt_smi.reset.subprocess.run", side_effect=[mc_info("0x21")]):
            with patch.object(
                reset, "run_galaxy_ipmi_reset_with_bmc_retry", side_effect=SystemExit(0)
            ) as retry:
                with pytest.raises(SystemExit):
                    glx_6u_trays_reset(use_umd=True)
        assert retry.call_args[0][4] is True
        assert "older than v0.05.22" in capsys.readouterr().out

    def test_old_bmc_with_flag_reaches_op0(self, glx):
        def run(cmd, **kw):
            if cmd[2:5] == ["raw", "0x30", "0x8B"]:
                return fail(self.C9) if cmd[7] == "0x3" else ok()
            return mc_info("0x21")

        with patch("tt_smi.reset.subprocess.run", side_effect=run) as m:
            with pytest.raises(SystemExit):
                glx_6u_trays_reset(use_umd=True, allow_legacy_retimer_reset=True)
        assert op_modes(m)[-1] == GLX_IPMI_OP_MODE_WITH_RETIMER

    def test_new_bmc_reaches_op3(self, glx):
        with patch(
            "tt_smi.reset.subprocess.run", side_effect=[mc_info("0x23"), ok()]
        ) as m:
            with pytest.raises(SystemExit):
                glx_6u_trays_reset(use_umd=True)
        assert op_modes(m) == ["0x3"]

    def test_unparseable_version_warns_then_current_path(self, glx, capsys):
        with patch("tt_smi.reset.subprocess.run", side_effect=[ok(), ok()]) as m:
            with pytest.raises(SystemExit):
                glx_6u_trays_reset(use_umd=True)
        assert "version was not read" in capsys.readouterr().out
        assert op_modes(m) == ["0x3"]


class TestGlxResetFlow:
    def test_bmc_silent_resets_nothing(self, glx):
        fs, ioctl_calls = glx
        with patch("tt_smi.reset.subprocess.run", side_effect=[fail(NO_DATA)]) as run:
            with pytest.raises(RuntimeError, match="No chip was reset"):
                glx_6u_trays_reset(use_umd=True)
        assert ioctl_calls == [] and op_modes(run) == []

    def test_ipmi_failure_after_user_reset_reports_held_chips(self, glx, capsys):
        fs, ioctl_calls = glx
        clock = iter(range(0, 100_000, 100))
        with patch(
            "tt_smi.reset.time.monotonic", side_effect=lambda: next(clock)
        ), patch(
            "tt_smi.reset.subprocess.run",
            side_effect=[ok()] + [fail(NO_DATA)] * 100,
        ) as run:
            with pytest.raises(GalaxyResetError) as exc:
                glx_6u_trays_reset(use_umd=True)
        assert exc.value.exit_code == GLX_EXIT_BMC_FAILED
        out = capsys.readouterr().out
        assert "0000:47:00.0 (tray 2, root port 0000:40:01.7)" in out
        assert "host power cycle" in out and "UBB" not in out
        assert GLX_IPMI_OP_MODE_WITH_RETIMER not in op_modes(run)

    def test_all_back_exits_0(self, glx):
        with patch("tt_smi.reset.subprocess.run", side_effect=[ok(), ok()]):
            with pytest.raises(SystemExit) as exc:
                glx_6u_trays_reset(use_umd=True)
        assert exc.value.code == 0

    def test_lost_chip_comes_back_after_rescan_exits_0(self, glx):
        fs, _ = glx
        fs.remove_chip(
            "0000:47:00.0"
        )  # lost before the reset (stays down after warm reset)

        def rescan(port, sysfs):
            fs.add_chip("0000:47:00.0")
            return True

        with patch("tt_smi.reset.subprocess.run", side_effect=[ok(), ok()]), patch(
            "tt_smi.reset._rescan_port", side_effect=rescan
        ):
            with pytest.raises(SystemExit) as exc:
                glx_6u_trays_reset(use_umd=True)
        assert exc.value.code == 0

    @pytest.mark.parametrize("reinit", [True, False])
    def test_chip_still_missing_exits_4_after_post_reset_of_others(
        self, glx, capsys, reinit
    ):
        fs, ioctl_calls = glx
        fs.remove_chip("0000:47:00.0")
        clock = iter(range(0, 100_000, 50))
        with patch(
            "tt_smi.reset.time.monotonic", side_effect=lambda: next(clock)
        ), patch("tt_smi.reset.subprocess.run", side_effect=[ok(), ok()]), patch(
            "tt_smi.reset._rescan_port", return_value=True
        ):
            with pytest.raises(SystemExit) as exc:
                glx_6u_trays_reset(use_umd=True, reinit=reinit)
        assert exc.value.code == GLX_EXIT_CHIPS_MISSING
        out = capsys.readouterr().out
        assert "31/32 chips on PCIe" in out
        assert "0000:47:00.0 (tray 2, root port 0000:40:01.7)" in out
        assert ioctl_calls[-1] == IoctlResetFlags.POST_RESET


class TestOneChipFailureDoesNotStopOthers:
    def test_ioctl_that_raises_counts_as_failed_and_others_still_reset(self, capsys):
        called = []

        def ioctl(interface_id, flag):
            called.append(interface_id)
            if interface_id == 5:
                raise OSError(19, "No such device")
            return interface_id != 9

        with patch("tt_smi.reset.reset_device_ioctl", side_effect=ioctl):
            failed = reset.parallel_reset_device_ioctl(
                list(range(32)), IoctlResetFlags.USER_RESET
            )
        assert sorted(called) == list(range(32))
        assert sorted(failed) == [5, 9]
        assert "No such device" in capsys.readouterr().out

    def test_bdf_of_interface_from_sysfs(self, tmp_path):
        cls = tmp_path / "class" / "tenstorrent" / "tenstorrent!3"
        cls.mkdir(parents=True)
        os.symlink(
            "../../../devices/pci0000:40/0000:40:01.7/0000:47:00.0", cls / "device"
        )
        assert reset._bdf_of_interface(3, str(tmp_path)) == "0000:47:00.0"
        assert reset._bdf_of_interface(4, str(tmp_path)) == ""

    def test_post_reset_failure_named_by_bdf_and_tray_exits_1(self, glx, capsys):
        fs, _ = glx
        cls = os.path.join(fs.root, "class", "tenstorrent", "tenstorrent!7")
        os.makedirs(cls)
        os.symlink("/x/0000:47:00.0", os.path.join(cls, "device"))

        def ioctl(ids, flag):
            return [7] if flag == IoctlResetFlags.POST_RESET else []

        with patch("tt_smi.reset.subprocess.run", side_effect=[ok(), ok()]), patch(
            "tt_smi.reset.parallel_reset_device_ioctl", side_effect=ioctl
        ):
            with pytest.raises(SystemExit) as exc:
                glx_6u_trays_reset(use_umd=True)
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "POST_RESET failed for device" in out
        assert "0000:47:00.0 (tray 2)" in out and "UBB" not in out


class TestCliExitCodes:
    @pytest.mark.parametrize("flag", ["-glx_reset", "-glx_reset_auto"])
    def test_galaxy_reset_error_exit_code_and_no_retry(self, flag):
        err = GalaxyResetError("held", GLX_EXIT_BMC_FAILED)
        with patch("sys.argv", ["tt-smi", flag]), patch(
            "tt_smi.tt_smi.get_driver_version", return_value="2.7.0"
        ), patch("tt_smi.tt_smi.glx_6u_trays_reset", side_effect=err) as fn:
            with pytest.raises(SystemExit) as exc:
                cli.main()
        assert exc.value.code == GLX_EXIT_BMC_FAILED
        assert fn.call_count == 1
        assert fn.call_args[1]["allow_legacy_retimer_reset"] is False

    def test_allow_legacy_flag_passed(self):
        with patch(
            "sys.argv", ["tt-smi", "-glx_reset", "--allow_legacy_retimer_reset"]
        ), patch("tt_smi.tt_smi.get_driver_version", return_value="2.7.0"), patch(
            "tt_smi.tt_smi.glx_6u_trays_reset", side_effect=SystemExit(0)
        ) as fn:
            with pytest.raises(SystemExit):
                cli.main()
        assert fn.call_args[1]["allow_legacy_retimer_reset"] is True


class TestHeartbeatCheck:
    """ARC heartbeat after the recount (#281 item 2). Mock only."""

    BDFS = ["0000:01:00.0", "0000:47:00.0", "0000:c1:00.0"]

    def run(self, beats, sleep=None):
        """beats: bdf -> [first, second]; an Exception entry is a failed read."""
        chips = list(self.BDFS)
        seen = {b: 0 for b in chips}

        def read(chip, use_umd):
            v = beats[chip][seen[chip]]
            seen[chip] += 1
            if isinstance(v, Exception):
                raise v
            return v

        with patch("tt_smi.reset._chip_bdf", side_effect=lambda c, u: c), patch(
            "tt_smi.reset._read_arc_heartbeat", side_effect=read
        ), patch("tt_smi.reset.time.sleep") as slp:
            dead = reset.galaxy_heartbeat_check(chips, False, gap_s=sleep)
        return dead, slp

    def test_all_advance(self):
        dead, slp = self.run({b: [10, 11] for b in self.BDFS})
        assert dead == []
        slp.assert_called_once_with(reset.GLX_HEARTBEAT_GAP_S)

    def test_stuck_chip_named_by_bdf_and_tray(self):
        beats = {b: [10, 11] for b in self.BDFS}
        beats["0000:47:00.0"] = [7, 7]
        dead, _ = self.run(beats)
        assert dead == ["0000:47:00.0"]
        assert reset._chip_lines(dead, {}, None) == "  0000:47:00.0 (tray 2)"

    def test_read_failure_counts_as_stuck(self):
        beats = {b: [10, 11] for b in self.BDFS}
        beats["0000:c1:00.0"] = [10, OSError("read failed")]
        beats["0000:01:00.0"] = [OSError("read failed"), 11]
        dead, _ = self.run(beats)
        assert dead == ["0000:01:00.0", "0000:c1:00.0"]

    def _flow(self, glx, beats):
        fs, _ = glx
        chips = {i: b for i, b in enumerate(sorted(fs.chips))}
        seen = {b: 0 for b in chips.values()}
        beats = {**{b: [1, 2] for b in chips.values()}, **beats}

        def read(chip, use_umd):
            v = beats[chip][seen[chip]]
            seen[chip] += 1
            return v

        return (
            patch(
                "tt_smi.reset.TopologyDiscovery.discover",
                return_value=(_desc(fs), chips),
            ),
            patch("tt_smi.reset._chip_bdf", side_effect=lambda c, u: c),
            patch("tt_smi.reset._read_arc_heartbeat", side_effect=read),
        )

    def test_full_flow_exit_5_names_chip(self, glx, capsys):
        beats = {b: [10, 11] for b in self.BDFS}
        beats["0000:47:00.0"] = [7, 7]
        d, b, r = self._flow(glx, beats)
        with patch("tt_smi.reset.subprocess.run", side_effect=[ok(), ok()]), d, b, r:
            with pytest.raises(SystemExit) as exc:
                glx_6u_trays_reset(use_umd=True)
        assert exc.value.code == reset.GLX_EXIT_NO_HEARTBEAT
        out = capsys.readouterr().out
        assert "0000:47:00.0 (tray 2" in out and "UBB" not in out
        assert "0000:01:00.0" not in out

    def test_full_flow_exit_0_when_all_respond(self, glx):
        d, b, r = self._flow(glx, {x: [1, 2] for x in self.BDFS})
        with patch("tt_smi.reset.subprocess.run", side_effect=[ok(), ok()]), d, b, r:
            with pytest.raises(SystemExit) as exc:
                glx_6u_trays_reset(use_umd=True)
        assert exc.value.code == 0


class TestHeartbeatRealReads:
    """Real _read_arc_heartbeat and _chip_bdf on fake UMD chips (#4628). Mock only."""

    def _chip(self, arch, ticking=True):
        n = [0]

        class Reader:
            def __init__(self, *a):
                pass

            def read_entry(self, tag):
                if ticking:
                    n[0] += 1
                return n[0]

        chip = MagicMock()
        chip.get_arch.return_value = arch
        chip.get_arc_telemetry_reader.return_value = Reader()
        return chip, Reader

    def _flow(self, fs, skip=None, raises=False, stuck=None):
        bdfs = sorted(fs.chips)
        devices, bdf_map = {}, {}
        Reader = None
        for i, b in enumerate(bdfs):
            arch = reset.ARCH.WORMHOLE_B0 if i % 2 else reset.ARCH.BLACKHOLE
            chip, Reader = self._chip(arch, ticking=(b != stuck))
            bdf_map[i] = b
            if b != skip:
                devices[i] = chip
        desc = MagicMock()
        desc.get_chip_pci_bdfs.return_value = bdf_map
        # WH chips are read through SmBusArcTelemetryReader(chip): give it a ticking reader.
        counters = {}

        def wh_reader(chip):
            counters[id(chip)] = counters.get(id(chip), 0) + 1
            return MagicMock(read_entry=lambda tag: counters[id(chip)] * 0 + next(tick))

        tick = iter(range(1, 10**6))
        discover = (
            patch("tt_smi.reset.TopologyDiscovery.discover", side_effect=RuntimeError("no"))
            if raises
            else patch(
                "tt_smi.reset.TopologyDiscovery.discover", return_value=(desc, devices)
            )
        )
        return discover, patch("tt_smi.reset.SmBusArcTelemetryReader", side_effect=wh_reader)

    def _run(self, glx, capsys, **kw):
        fs, _ = glx
        d, r = self._flow(fs, **kw)
        with patch("tt_smi.reset.subprocess.run", side_effect=[ok(), ok()]), d, r, patch(
            "tt_smi.reset.time.sleep"
        ), patch("tt_smi.reset._read_arc_heartbeat", new=_REAL_READ):
            with pytest.raises(SystemExit) as exc:
                glx_6u_trays_reset(use_umd=True)
        return exc.value.code, capsys.readouterr().out, fs

    def test_healthy_box_exit_0(self, glx, capsys):
        code, out, _ = self._run(glx, capsys)
        assert code == 0
        assert "heartbeat" not in out.lower()

    def test_skipped_chip_named_exit_5(self, glx, capsys):
        fs, _ = glx
        skip = sorted(fs.chips)[3]
        code, out, _ = self._run(glx, capsys, skip=skip)
        assert code == reset.GLX_EXIT_NO_HEARTBEAT
        assert f"{skip} (tray" in out and "UBB" not in out
        assert out.count("(tray") == 1

    def test_stuck_chip_named_exit_5(self, glx, capsys):
        fs, _ = glx
        stuck = sorted(fs.chips)[4]  # a BH chip
        code, out, _ = self._run(glx, capsys, stuck=stuck)
        assert code == reset.GLX_EXIT_NO_HEARTBEAT
        assert f"{stuck} (tray" in out

    def test_discovery_raises_names_present_unverified(self, glx, capsys):
        fs, _ = glx
        code, out, _ = self._run(glx, capsys, raises=True)
        assert code == 1
        assert "unverified" in out
        assert all(f"{b} (tray" in out for b in fs.chips)
        assert "UBB" not in out
