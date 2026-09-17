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
Tests for :mod:`pism_terra.ismip7.greenland.postprocess_dh`.

What distinguishes this from the glacier tool is what it reduces and what it
reduces *from*: ice thickness rather than surface elevation, out of a
submission tree that keeps one file per variable, and optionally as a
cumulative series so it lines up with the observed products.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pism_terra.ismip7.greenland.postprocess_dh import (
    DEFAULT_VARIABLES,
    compute_cumulative_dh,
    postprocess_dh,
    source_files,
)

STEM = "GrIS_UAF_PISM_m001_CESM2-WACCM_f001_historical_C001_1985-2014"


def _spatial(n_time: int = 5, thinning: float = -1.0) -> xr.Dataset:
    """
    A spatial dataset thinning by a known amount each year.

    Parameters
    ----------
    n_time : int, optional
        Number of yearly steps starting 2000.
    thinning : float, optional
        Thickness change per step, metres.

    Returns
    -------
    xr.Dataset
        Dataset with ``lithk``, a time-less ``topg`` and a grid mapping.
    """
    times = np.array([np.datetime64(f"{2000 + i}-01-01") for i in range(n_time)])
    thk = np.stack([np.full((2, 2), 1000.0 + i * thinning) for i in range(n_time)])
    return xr.Dataset(
        {
            "lithk": (("time", "y", "x"), thk, {"units": "m", "standard_name": "land_ice_thickness"}),
            "topg": (("y", "x"), np.zeros((2, 2)), {"units": "m"}),
            "mapping": ((), np.int8(0), {"grid_mapping_name": "polar_stereographic", "proj_params": "EPSG:3413"}),
        },
        coords={"time": times, "y": [0.0, 1000.0], "x": [0.0, 1000.0]},
    )


def test_cumulative_is_change_since_the_start_at_every_step():
    """
    One record per step, each measured from the start of the record.

    That is the shape the observed products carry, so the two subtract
    directly instead of one having to be re-derived.
    """
    ds = _spatial(n_time=5, thinning=-2.0)

    dh = compute_cumulative_dh(ds, "2001-01-01", ["lithk"])

    # From 2001 onward: 4 steps, thinning 0, -2, -4, -6.
    assert dh.sizes["time"] == 4
    np.testing.assert_allclose(dh["lithk"].isel(y=0, x=0).values, [0.0, -2.0, -4.0, -6.0])
    # Every record's lower bound is the start of the record.
    assert (dh["time_bnds"].values[:, 0] == np.datetime64("2001-01-01")).all()
    assert dh["time_bnds"].values[-1, 1] == np.datetime64("2004-01-01")
    assert dh["time"].attrs["bounds"] == "time_bnds"
    # Time-less variables are carried through, not differenced.
    np.testing.assert_array_equal(dh["topg"].values, ds["topg"].values)


def test_cumulative_rejects_a_variable_the_file_does_not_have():
    """
    Name what is available rather than failing obscurely later.
    """
    with pytest.raises(ValueError, match="orog"):
        compute_cumulative_dh(_spatial(), "2000-01-01", ["orog"])


def test_source_files_finds_one_file_per_variable(tmp_path: Path):
    """
    A submission tree keeps a file per variable, keyed on the leading token.

    ``lithk`` must not pick up ``dlithkdt``, which is a different quantity
    that happens to share the substring.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    for variable in ("lithk", "dlithkdt", "orog"):
        _spatial().to_netcdf(tmp_path / f"{variable}_{STEM}.nc")

    found = source_files(tmp_path, ["lithk"])
    assert set(found) == {"lithk"}
    assert found["lithk"].name.startswith("lithk_")


def test_a_flat_spatial_file_is_accepted(tmp_path: Path):
    """
    A run with ISMIP7 naming off writes one combined file.

    The same command has to serve C011's submission tree and a plain OCX
    run, so a file is as valid an input as a directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    path = tmp_path / "spatial_g900m_id_OCX.nc"
    _spatial().to_netcdf(path)

    found = source_files(path, ["lithk"])
    assert found == {"lithk": path}


def test_an_empty_directory_is_an_error(tmp_path: Path):
    """
    Say so rather than writing nothing and reporting success.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    with pytest.raises(FileNotFoundError, match="no file for lithk"):
        source_files(tmp_path, ["lithk"])


def test_default_is_thickness_not_surface_elevation():
    """
    The observed dH/dt is thickness in ice equivalent, so ``lithk`` it is.

    The glacier tool differences ``usurf`` because Hugonnet measures surface
    elevation; using that here would compare two different quantities.
    """
    assert DEFAULT_VARIABLES == ("lithk",)


def test_end_to_end_single_interval_and_cumulative(tmp_path: Path):
    """
    Both modes write a georeferenced file from a submission directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    experiment = tmp_path / "C001"
    experiment.mkdir()
    _spatial(n_time=5, thinning=-2.0).to_netcdf(experiment / f"lithk_{STEM}.nc")

    single = postprocess_dh(experiment, tmp_path / "one", "2001-01-01", end="2003-01-01")
    assert len(single) == 1
    with xr.open_dataset(single[0]) as ds:
        assert ds.sizes["time"] == 1
        # A synthetic input carries no time encoding, so the bounds come back
        # as raw offsets rather than dates; the real files decode.
        bounds = ds["time_bnds"].values[0].astype("datetime64[ns]")
        np.testing.assert_array_equal(bounds, np.array(["2001-01-01", "2003-01-01"], dtype="datetime64[ns]"))
        np.testing.assert_allclose(ds["lithk"].values.ravel(), -4.0)
        assert ds.rio.crs is not None

    series = postprocess_dh(experiment, tmp_path / "many", "2001-01-01")
    with xr.open_dataset(series[0]) as ds:
        assert ds.sizes["time"] == 4
        np.testing.assert_allclose(ds["lithk"].isel(y=0, x=0).values, [0.0, -2.0, -4.0, -6.0])
        assert ds.rio.crs is not None
