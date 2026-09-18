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
Tests for the observed-velocity COGs written by S4F planning.

The ITS_LIVE fetch is replaced by a stub returning a synthetic clipped
dataset on the planning grid, so only the COG writing is exercised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from shapely.geometry import box

from pism_terra.domain import create_domain
from pism_terra.glacier import s4f
from pism_terra.glacier.s4f import VELOCITY_VARIABLES, write_velocity_cogs

CRS = "EPSG:32607"
GEOM_UTM = box(615_000.0, 6_665_000.0, 625_000.0, 6_675_000.0)
RGI_ID = "RGI2000-v7.0-C-01-99999"


@pytest.fixture(name="grid")
def fixture_grid() -> xr.Dataset:
    """
    A 200 m target grid covering the synthetic outline.

    Returns
    -------
    xarray.Dataset
        Grid with CRS EPSG:32607.
    """
    return create_domain([612_000.0, 628_000.0], [6_662_000.0, 6_678_000.0], resolution=200.0, crs=CRS)


def synthetic_velocities(grid: xr.Dataset) -> xr.Dataset:
    """
    Build a stand-in for the clipped ITS_LIVE dataset.

    Constant fields on the grid; the western half carries a misfit weight
    of 0 (unobserved) and, like the real obs file, zeroed velocities there.

    Parameters
    ----------
    grid : xarray.Dataset
        Planning grid.

    Returns
    -------
    xarray.Dataset
        Velocity fields plus ``vel_misfit_weight``.
    """
    shape = (grid.sizes["y"], grid.sizes["x"])
    observed = np.zeros(shape, dtype=int)
    observed[:, shape[1] // 2 :] = 1
    values = {"v": 5.0, "vx": 3.0, "vy": 4.0, "vx_error": 0.5, "vy_error": 0.25}
    ds = xr.Dataset(
        {name: (("y", "x"), np.where(observed == 1, value, 0.0)) for name, value in values.items()},
        coords={"x": grid.x, "y": grid.y},
    )
    ds["vel_misfit_weight"] = (("y", "x"), observed)
    return ds.rio.write_crs(CRS)


def test_velocity_cogs_are_written_per_field(tmp_path: Path, grid: xr.Dataset, monkeypatch):
    """
    One COG per velocity field, on the grid, with unobserved cells masked.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    grid : xarray.Dataset
        Target grid fixture.
    monkeypatch : pytest.MonkeyPatch
        Replaces the ITS_LIVE fetch.
    """
    calls = []

    def fake_fetch(target_grid, _geometries, **kwargs):
        """
        Record the call and return the synthetic dataset.

        Parameters
        ----------
        target_grid : xarray.Dataset
            Planning grid the stub builds its fields on.
        _geometries : iterable
            Ignored outline.
        **kwargs
            Keyword arguments the helper passes through; recorded.

        Returns
        -------
        xarray.Dataset
            Synthetic velocity dataset.
        """
        calls.append((kwargs["product_name"], Path(kwargs["path"]).name, kwargs["rgi_id"]))
        return synthetic_velocities(target_grid)

    monkeypatch.setattr(s4f, "glacier_velocities_from_grid", fake_fetch)
    out = tmp_path / "input"
    out.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()

    written = write_velocity_cogs({"velocity": "its_live"}, RGI_ID, grid, [GEOM_UTM], out, staging)

    assert calls == [("its_live", f"obs_{RGI_ID}.nc", RGI_ID)]
    assert set(written) == {f"{RGI_ID}_its_live_{var}" for var in VELOCITY_VARIABLES}
    with rasterio.open(written[f"{RGI_ID}_its_live_vx"]) as src:
        assert src.crs == rasterio.CRS.from_user_input(CRS)
        assert (src.width, src.height) == (grid.sizes["x"], grid.sizes["y"])
        assert np.isnan(src.nodata)
        vx = src.read(1, masked=True)
    # Observed half carries the value; the unobserved half is masked, not 0.
    assert vx.mask[:, : grid.sizes["x"] // 2].all()
    np.testing.assert_allclose(vx[:, grid.sizes["x"] // 2 :].compressed(), 3.0, rtol=1e-6)


def test_without_a_velocity_product_nothing_is_written(tmp_path: Path, grid: xr.Dataset, monkeypatch):
    """
    A campaign without ``velocity`` (or with ``"none"``) neither fetches nor writes.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    grid : xarray.Dataset
        Target grid fixture.
    monkeypatch : pytest.MonkeyPatch
        Fails the test if the fetch is reached.
    """

    def must_not_run(*_args, **_kwargs):
        """
        Fail if called.

        Parameters
        ----------
        *_args
            Ignored.
        **_kwargs
            Ignored.
        """
        raise AssertionError("velocity fetch should not run")

    monkeypatch.setattr(s4f, "glacier_velocities_from_grid", must_not_run)

    assert not write_velocity_cogs({"velocity": "none"}, RGI_ID, grid, [GEOM_UTM], tmp_path, tmp_path)
    assert not write_velocity_cogs({}, RGI_ID, grid, [GEOM_UTM], tmp_path, tmp_path)
    assert not list(tmp_path.glob("*.tif"))
