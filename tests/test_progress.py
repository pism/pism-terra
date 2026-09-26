# Copyright (C) 2026 Andy Aschwanden
#
# This file is part of pism-terra.
#
# PISM-TERRA is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PISM-TERRA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software

"""
Tests for :mod:`pism_terra.progress`.

The displays themselves are not asserted on; what matters is that the
helpers hand back the same values with or without a terminal, and that
the distributed path is taken only when a client is active.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pism_terra import progress


def test_progress_bar_yields_the_items(monkeypatch):
    """
    The bar is a pass-through, on a terminal and off it.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture, used to force the terminal check.
    """
    for tty in (True, False):
        monkeypatch.setattr(progress, "show_progress", lambda: tty)  # pylint: disable=cell-var-from-loop
        assert list(progress.progress_bar(range(3), desc="x")) == [0, 1, 2]
        assert list(progress.progress_bar(iter("ab"), total=2)) == ["a", "b"]


def test_compute_matches_dask_compute(monkeypatch):
    """
    Lazy and eager inputs come back computed, in order, whether or not a bar is drawn.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture, used to force the terminal check.
    """
    lazy = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=["a", "b"]).chunk({"a": 1})
    eager = xr.DataArray([1.0, 2.0], dims=["a"])
    for tty in (True, False):
        monkeypatch.setattr(progress, "show_progress", lambda: tty)  # pylint: disable=cell-var-from-loop
        total, twice, plain = progress.compute(lazy.sum("b"), lazy * 2, eager, desc="test")
        np.testing.assert_allclose(total, [3.0, 12.0])
        np.testing.assert_allclose(twice, lazy.values * 2)
        assert twice.chunks is None
        np.testing.assert_allclose(plain, [1.0, 2.0])


def test_compute_is_silent_for_eager_inputs(monkeypatch, capsys):
    """
    In-memory inputs have nothing to wait for, so no label is printed even on a terminal.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture, used to force the terminal check.
    capsys : pytest.CaptureFixture
        Pytest fixture capturing standard error.
    """
    monkeypatch.setattr(progress, "show_progress", lambda: True)
    (out,) = progress.compute(xr.DataArray([1.0, 2.0], dims=["a"]), desc="should not print")
    np.testing.assert_allclose(out, [1.0, 2.0])
    assert capsys.readouterr().err == ""
    lazy = xr.DataArray([1.0, 2.0], dims=["a"]).chunk({"a": 1})
    progress.compute(lazy, desc="should print")
    assert "should print" in capsys.readouterr().err


def test_distributed_client_is_none_without_a_cluster():
    """
    On the local scheduler there is no client to follow.
    """
    assert progress.distributed_client() is None


@pytest.mark.skipif(
    pytest.importorskip("dask.distributed", reason="distributed not installed") is None, reason="no distributed"
)
def test_compute_uses_an_active_client(monkeypatch):
    """
    With a client in the process the graph is persisted there and followed.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture, used to force the terminal check.
    """
    from dask.distributed import Client  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(progress, "show_progress", lambda: True)
    lazy = xr.DataArray(np.arange(4.0), dims=["a"]).chunk({"a": 2})
    with Client(processes=False, n_workers=1, threads_per_worker=1, dashboard_address=None) as client:
        assert progress.distributed_client() is client
        (total,) = progress.compute(lazy.sum(), desc="on the cluster")
        assert float(total) == 6.0
