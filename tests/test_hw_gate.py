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
    "env,skipped", [(None, True), ("0", True), ("", True), ("1", False)]
)
def test_hardware_tests_need_opt_in(monkeypatch, env, skipped):
    if env is None:
        monkeypatch.delenv(HW_TESTS_ENV, raising=False)
    else:
        monkeypatch.setenv(HW_TESTS_ENV, env)
    hw, galaxy, plain = (
        _item("requires_hardware"),
        _item("requires_galaxy"),
        _item(),
    )
    pytest_collection_modifyitems(None, [hw, galaxy, plain])
    assert bool(hw.markers) is skipped
    assert bool(galaxy.markers) is skipped
    assert not plain.markers
