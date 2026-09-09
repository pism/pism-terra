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
Tests for :mod:`pism_terra.inverse_plot`.

Builds a small synthetic inversion ensemble carrying the fields the maps
need, and covers:

- ``read_member`` masking the design variable to the free cells and the
  residual to the misfit area, and skipping unusable files.
- ``shared_limits`` pooling across members, honouring overrides, and its
  guards for a log scale and a fully-masked row.
- ``plot_members`` writing a figure for one and for many members.
- the ``main`` entry point end to end, including its exit on an empty
  ensemble.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from pism_terra.inverse_plot import main, plot_members, read_member, shared_limits

PENALTY = "inverse.tikhonov.penalty_weight"


def write_member(
    path: Path,
    penalty_weight: float,
    *,
    design: str = "tauc",
    tauc_free: float = 1.0e5,
    tauc_fixed: float = 1.4e5,
    residual: float = 20.0,
    complete: bool = True,
) -> Path:
    """
    Write a synthetic inversion output with distinguishable masked regions.

    The left half of the domain is the inverted region (``zeta_fixed_mask``
    0, ``vel_misfit_weight`` 1); the right half is fixed and unobserved, and
    carries different values so masking is detectable.

    Parameters
    ----------
    path : pathlib.Path
        File to write.
    penalty_weight : float
        Value stored as ``inverse.tikhonov.penalty_weight``.
    design : str, optional
        Design variable to write, with its ``<design>_prior``.
    tauc_free : float, optional
        Design-variable value on the free (inverted) cells.
    tauc_fixed : float, optional
        Design-variable value on the fixed cells, which masking must drop.
    residual : float, optional
        ``inv_residual`` value on the misfit area; the rest gets twice this,
        which masking must drop.
    complete : bool, optional
        If False, the residual is left out, as in a member still running.

    Returns
    -------
    pathlib.Path
        The path written, for convenience.
    """
    ny, nx = 4, 6
    free = np.zeros((ny, nx), dtype=bool)
    free[:, : nx // 2] = True

    data: dict[str, Any] = {
        "pism_config": ((), np.int8(0), {PENALTY: np.float64(penalty_weight)}),
        design: (("time", "y", "x"), np.where(free, tauc_free, tauc_fixed)[np.newaxis, ...]),
        f"{design}_prior": (("time", "y", "x"), np.full((1, ny, nx), tauc_fixed)),
        "zeta_inv": (("time", "y", "x"), np.zeros((1, ny, nx))),
        "zeta_fixed_mask": (("time", "y", "x"), np.where(free, 0.0, 1.0)[np.newaxis, ...]),
        "vel_misfit_weight": (("time", "y", "x"), free.astype(float)[np.newaxis, ...]),
    }
    if complete:
        data["inv_residual"] = (("time", "y", "x"), np.where(free, residual, 2 * residual)[np.newaxis, ...])
    coords = {"x": np.arange(nx, dtype=float) * 100.0, "y": np.arange(ny, dtype=float) * 100.0}
    xr.Dataset(data, coords=coords).to_netcdf(path)
    return path


@pytest.fixture(name="ensemble")
def fixture_ensemble(tmp_path: Path) -> list[Path]:
    """
    Three-member sweep plus one member that has not written its residual.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    list of pathlib.Path
        The member files, deliberately not ordered by penalty weight.
    """
    files = [
        write_member(tmp_path / f"inv_{p:g}.nc", p, tauc_free=t, residual=r)
        for p, t, r in [(10.0, 3.0e5, 5.0), (0.1, 1.0e5, 40.0), (1.0, 2.0e5, 12.0)]
    ]
    files.append(write_member(tmp_path / "inv_running.nc", 100.0, complete=False))
    return files


def test_read_member_masks_to_the_inverted_region(ensemble: list[Path]) -> None:
    """
    Keep the free cells of the design variable and the fit cells of residual.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.

    Returns
    -------
    None
        Asserts only.
    """
    member = read_member(ensemble[0], [PENALTY], None)
    assert member is not None
    assert member["design_name"] == "tauc"
    assert member["penalty_weight"] == 10.0
    # Left half kept, right half masked out, on both rows.
    assert np.array_equal(np.isnan(member["design"]), np.isnan(member["inv_residual"]))
    assert np.nanmin(member["design"]) == np.nanmax(member["design"]) == 3.0e5
    assert np.nanmin(member["inv_residual"]) == np.nanmax(member["inv_residual"]) == 5.0
    assert np.isnan(member["design"]).sum() == member["design"].size // 2


def test_read_member_unmasked_keeps_everything(ensemble: list[Path]) -> None:
    """
    Skip masking entirely when asked, so the prior stays visible.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.

    Returns
    -------
    None
        Asserts only.
    """
    member = read_member(ensemble[0], [PENALTY], None, mask=False)
    assert member is not None
    assert not np.isnan(member["design"]).any()
    assert set(np.unique(member["design"])) == {3.0e5, 1.4e5}


def test_read_member_skips_incomplete(ensemble: list[Path]) -> None:
    """
    Return None for a member that has not written its residual.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.

    Returns
    -------
    None
        Asserts only.
    """
    assert read_member(ensemble[-1], [PENALTY], None) is None


def test_read_member_skips_unknown_parameter(ensemble: list[Path]) -> None:
    """
    Return None when the swept parameter is absent from ``pism_config``.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.

    Returns
    -------
    None
        Asserts only.
    """
    assert read_member(ensemble[0], ["surface.pdd.factor_ice"], None) is None


def test_shared_limits_pool_across_members() -> None:
    """
    Span every member, and let explicit limits win over the percentiles.

    Returns
    -------
    None
        Asserts only.
    """
    arrays = [np.array([[1.0, 2.0]]), np.array([[10.0, np.nan]])]
    assert shared_limits(arrays, percentile=0.0) == (1.0, 10.0)
    assert shared_limits(arrays, percentile=0.0, vmin=0.5) == (0.5, 10.0)
    assert shared_limits(arrays, percentile=0.0, vmax=99.0) == (1.0, 99.0)
    # Trimming the tails pulls both ends in.
    low, high = shared_limits(arrays, percentile=25.0)
    assert 1.0 < low and high < 10.0


def test_shared_limits_guards() -> None:
    """
    Refuse a fully-masked row, and drop non-positives for a log scale.

    Returns
    -------
    None
        Asserts only.
    """
    with pytest.raises(ValueError, match="fully masked"):
        shared_limits([np.array([[np.nan, np.nan]])])
    with pytest.raises(ValueError, match="log color scale"):
        shared_limits([np.array([[0.0, -1.0]])], percentile=0.0, positive=True)
    assert shared_limits([np.array([[0.0, 1.0, 4.0]])], percentile=0.0, positive=True) == (1.0, 4.0)
    # A constant field still yields an increasing pair.
    low, high = shared_limits([np.array([[3.0, 3.0]])], percentile=0.0)
    assert low < high


def test_plot_members_writes_figure(ensemble: list[Path], tmp_path: Path) -> None:
    """
    Draw both rows for a multi-member ensemble and for a single member.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None
        Asserts only.
    """
    members = [m for m in (read_member(f, [PENALTY], None) for f in ensemble) if m is not None]
    members.sort(key=lambda m: m["penalty_weight"])
    output_file = tmp_path / "figures" / "maps.png"
    plot_members(members, "penalty_weight", output_file)
    assert output_file.exists()

    single = tmp_path / "one.pdf"
    plot_members(members[:1], "penalty_weight", single, log=False)
    assert single.exists()


def test_main_end_to_end(ensemble: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Write the figure from the command line, skipping the unusable member.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to set ``sys.argv``.

    Returns
    -------
    None
        Asserts only.
    """
    output_file = tmp_path / "out" / "maps.png"
    monkeypatch.setattr(
        "sys.argv",
        ["pism-inverse-plot", "--parameters", PENALTY, "-o", str(output_file)] + [str(f) for f in ensemble],
    )
    main()
    assert output_file.exists()


def test_main_exits_when_nothing_is_usable(
    ensemble: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Exit with a message rather than an empty figure.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to set ``sys.argv``.

    Returns
    -------
    None
        Asserts only.
    """
    monkeypatch.setattr(
        "sys.argv",
        ["pism-inverse-plot", "-o", str(tmp_path / "none.png"), str(ensemble[-1])],
    )
    with pytest.raises(SystemExit, match="yielded a panel"):
        main()
