# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from conftest import HW_TESTS_ENV, pytest_collection_modifyitems


def _item(*keywords):
    item = SimpleNamespace(keywords=set(keywords), markers=[])
    item.add_marker = item.markers.append
    return item


@pytest.mark.parametrize(
    "env,gha,skipped",
    [
        (None, None, True),
        ("0", None, True),
        ("", None, True),
        ("1", None, False),
        (None, "true", False),
        (None, "false", True),
        ("0", "true", True),
        ("", "true", True),
        ("1", "true", False),
    ],
)
def test_hardware_tests_need_opt_in(monkeypatch, env, gha, skipped):
    for name, value in ((HW_TESTS_ENV, env), ("GITHUB_ACTIONS", gha)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    hw, galaxy, plain = (
        _item("requires_hardware"),
        _item("requires_galaxy"),
        _item(),
    )
    pytest_collection_modifyitems(None, [hw, galaxy, plain])
    assert bool(hw.markers) is skipped
    assert bool(galaxy.markers) is skipped
    assert not plain.markers
