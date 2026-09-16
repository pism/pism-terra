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
Tests for :func:`pism_terra.workflow.region_id`.

The region dimension has to be numeric for CDO, and the number has to mean
the same region in every file: numbering by position merges unrelated
regions when two files carry different region sets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from pism_terra.workflow import make_cdo_readable, region_id


def test_region_id_is_stable_and_distinct():
    """
    The same label always gives the same id, different labels different ids.
    """
    assert region_id("GIS") == region_id("GIS")
    assert region_id("GIS") != region_id("GIS_CE")
    labels = [f"RGI2000-v7.0-G-01-{n:05d}" for n in range(2000)]
    ids = {region_id(label) for label in labels}
    assert len(ids) == len(labels), "collision among 2000 realistic labels"


def test_region_id_fits_a_positive_int32():
    """
    The id is written as ``int32``, so it must not overflow or go negative.
    """
    values = [region_id(f"region-{n}") for n in range(5000)]
    assert all(0 <= v <= np.iinfo(np.int32).max for v in values)


def _write(path: Path, labels: list[str], value: float) -> Path:
    """
    Write a per-region scalar file the way the postprocessors do.

    Parameters
    ----------
    path : pathlib.Path
        Destination file.
    labels : list of str
        Region labels.
    value : float
        Constant to fill the variable with, so the regions are tellable apart.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    ds = xr.Dataset(
        {"ice_mass": (("RGIid", "time"), np.full((len(labels), 4), value))},
        coords={"RGIid": labels, "time": np.arange(4.0)},
    )
    make_cdo_readable(ds, "RGIid").to_netcdf(path)
    return path


def test_files_with_different_region_sets_do_not_merge(tmp_path: Path):
    """
    A complex file and a per-glacier file stay distinct when concatenated.

    With positional ids the complex sat at index 0 and so merged into the
    first glacier -- the same index meaning two different regions. Ids
    derived from the labels cannot collide that way, so the mismatch shows
    up as sparsity instead of a wrong number.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    _write(tmp_path / "scalar_C.nc", ["RGI2000-v7.0-C-01-04374"], 100.0)
    _write(tmp_path / "scalar_G.nc", [f"RGI2000-v7.0-G-01-{n:05d}" for n in range(5)], 1.0)

    ds = xr.open_mfdataset(sorted(tmp_path.glob("*.nc")), combine="nested", concat_dim="exp_id")
    assert ds.sizes["RGIid"] == 6, "1 complex + 5 glaciers should stay 6 regions"
    values = ds["ice_mass"].isel(time=0).values
    # The complex keeps its own value rather than being averaged or replaced.
    assert int(np.nansum(values == 100.0)) == 1
    assert int(np.isnan(values).sum()) == 6


def test_the_id_matches_the_label_it_came_from():
    """
    The written id is the label's, so a reader can recompute it.
    """
    labels = ["GIS", "GIS_CE", "GIS_NW"]
    ds = xr.Dataset(
        {"mass": (("basin", "time"), np.zeros((3, 2)))},
        coords={"basin": labels, "time": np.arange(2.0)},
    )
    out = make_cdo_readable(ds, "basin")
    assert out["basin"].dtype == np.int32
    for value, name in zip(out["basin"].values, out["basin_name"].values):
        assert int(value) == region_id(name)
    # Selecting by name still works the documented way.
    assert out.set_index(basin="basin_name").sel(basin="GIS_CE")["mass"].shape == (2,)
