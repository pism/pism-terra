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
Tests for :mod:`pism_terra.postprocess_dh`.

Reuses the synthetic PISM-like Greenland dataset and the real Mouginot
outlines from ``test_postprocess_scalar``, stamped with a cftime ``365_day``
time axis as real PISM output carries.

Covers:
- ``_nearest_time_index`` nearest selection and clamping outside the record.
- ``compute_dh`` against a plain end-minus-start oracle, time-less
  passthrough, and its error cases.
- ``process_file_dh`` end-to-end: file round-trip, variable subsetting,
  CRS and ``pism_config`` carried through, time bounds spanning the interval.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # pylint: disable=unused-import
import xarray as xr
from test_postprocess_scalar import synthetic_greenland

from pism_terra.postprocess_dh import _nearest_time_index, compute_dh, process_file_dh


@pytest.fixture(name="basins")
def fixture_basins() -> gpd.GeoDataFrame:
    """
    Mouginot basin outlines shipped with the package.

    Returns
    -------
    geopandas.GeoDataFrame
        The seven Greenland drainage basins, in file order.
    """
    return gpd.read_file(str(files("pism_terra.data").joinpath("mouginot_basins_w_shelves.gpkg")))


def dated_greenland(basins: gpd.GeoDataFrame, n_time: int = 5, calendar: str = "365_day") -> xr.Dataset:
    """
    Synthetic Greenland dataset with yearly cftime steps from 2000-01-01.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Outlines used to size the grid.
    n_time : int, default 5
        Length of the time dimension.
    calendar : str, default "365_day"
        Calendar of the time axis, as PISM runs use.

    Returns
    -------
    xarray.Dataset
        The ``synthetic_greenland`` dataset with a datable time axis.
    """
    ds = synthetic_greenland(basins, n_time=n_time)
    times = xr.date_range("2000-01-01", periods=n_time, freq="YS", calendar=calendar, use_cftime=True)
    return ds.assign_coords(time=("time", np.asarray(times)))


def test_nearest_time_index_picks_and_clamps(basins):
    """
    Nearest selection lands on the closest step and clamps outside the record.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = dated_greenland(basins, n_time=5)

    assert _nearest_time_index(ds, "2000-01-01") == 0
    assert _nearest_time_index(ds, "2002-03-01") == 2
    assert _nearest_time_index(ds, "2002-09-01") == 3
    assert _nearest_time_index(ds, "1990-01-01") == 0
    assert _nearest_time_index(ds, "2050-01-01") == 4


def test_nearest_time_index_standard_calendar(basins):
    """
    A proleptic-gregorian axis (cftime under ``use_cftime=True``) works too.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = dated_greenland(basins, n_time=3, calendar="proleptic_gregorian")

    assert _nearest_time_index(ds, "2001-01-01") == 1


def test_compute_dh_matches_plain_difference(basins):
    """
    The differenced fields equal end-slice minus start-slice, names kept.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = dated_greenland(basins, n_time=5).drop_vars(["pism_config", "mapping"])

    dh = compute_dh(ds, "2001-01-01", "2004-01-01", variables=["thk", "ice_mass"])

    assert list(dh.data_vars) == ["thk", "ice_mass", "time_bnds"]
    assert dh.sizes["time"] == 1
    expected = ds["thk"].isel(time=4).values - ds["thk"].isel(time=1).values
    np.testing.assert_array_equal(dh["thk"].isel(time=0).values, expected)
    assert dh["thk"].dtype == ds["thk"].dtype
    assert dh["thk"].attrs["units"] == "1"
    assert dh["time"].values[0] == ds["time"].values[4]
    np.testing.assert_array_equal(dh["time_bnds"].values, [[ds["time"].values[1], ds["time"].values[4]]])
    assert dh["time"].attrs["bounds"] == "time_bnds"


def test_compute_dh_carries_timeless_variables_through(basins):
    """
    A variable without a time dimension is copied, not differenced.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = dated_greenland(basins, n_time=3).drop_vars(["pism_config", "mapping"])
    ds["topg"] = ds["thk"].isel(time=0).drop_vars("time") * -1.0

    dh = compute_dh(ds, "2000-01-01", "2002-01-01")

    np.testing.assert_array_equal(dh["topg"].values, ds["topg"].values)


def test_compute_dh_error_cases(basins):
    """
    Unknown variables and a degenerate interval raise ``ValueError``.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = dated_greenland(basins, n_time=3).drop_vars(["pism_config", "mapping"])

    with pytest.raises(ValueError, match="unknown variables"):
        compute_dh(ds, "2000-01-01", "2002-01-01", variables=["usurf"])
    with pytest.raises(ValueError, match="both select time step"):
        compute_dh(ds, "2000-01-01", "2000-02-01")


def test_process_file_dh_end_to_end(tmp_path: Path, basins):
    """
    File in, differenced file out: values, CRS, config and bounds survive.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = dated_greenland(basins, n_time=5)
    infile = tmp_path / "spatial.nc"
    outfile = tmp_path / "dh_2001-01-01_2004-01-01.nc"
    ds.to_netcdf(infile, engine="h5netcdf")

    written = process_file_dh(infile, outfile, start="2001-01-01", end="2004-01-01", variables=["ice_mass"])

    assert written == outfile
    with xr.open_dataset(outfile, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as out:
        assert set(out.data_vars) >= {"ice_mass", "time_bnds", "pism_config"}
        assert "thk" not in out.data_vars
        assert out.sizes["time"] == 1
        expected = ds["ice_mass"].isel(time=4).values - ds["ice_mass"].isel(time=1).values
        np.testing.assert_allclose(out["ice_mass"].isel(time=0).values, expected)
        assert out["time"].values[0] == ds["time"].values[4]
        np.testing.assert_array_equal(out["time_bnds"].values, [[ds["time"].values[1], ds["time"].values[4]]])
        assert out.rio.crs is not None
        assert out.attrs["dh_start_time"] == str(ds["time"].values[1])
        assert out["pism_config"].attrs["note"] == "synthetic"


def test_process_file_dh_unknown_variable_raises(tmp_path: Path, basins):
    """
    Asking for a variable the file lacks fails with the available list.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = dated_greenland(basins, n_time=3)
    infile = tmp_path / "spatial.nc"
    ds.to_netcdf(infile, engine="h5netcdf")

    with pytest.raises(ValueError, match="unknown variables"):
        process_file_dh(infile, tmp_path / "dh.nc", start="2000-01-01", end="2002-01-01", variables=["usurf"])
