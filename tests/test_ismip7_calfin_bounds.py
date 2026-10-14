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
Tests for :func:`pism_terra.ismip7.greenland.forcing.extend_final_time_bound`.

PISM applies a prescribed front-retreat record over its time bounds, so the
last CALFIN record has to stay valid for the whole run rather than expiring
with the observations in 2019.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pism_terra.ismip7.greenland.forcing import (
    FRONT_RETREAT_END,
    extend_final_time_bound,
)


def write_retreat_file(path: Path, units: str = "days since 1972-10-01 00:00:00") -> Path:
    """
    Write a miniature front-retreat file shaped like CALFIN's.

    Monthly records with CF ``time_bnds``, as ``cdo settbounds,1mon`` leaves
    them.

    Parameters
    ----------
    path : pathlib.Path
        Destination file.
    units : str, optional
        Time units attribute.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    # Contiguous, as ``cdo settbounds,1mon`` writes them: each record's upper
    # bound is the next record's timestamp.
    time = np.array([16983.0, 17013.0, 17044.0])
    bounds = np.stack([time, np.append(time[1:], time[-1] + 30.0)], axis=1)
    ds = xr.Dataset(
        {
            "land_ice_area_fraction_retreat": (("time", "y", "x"), np.ones((3, 2, 2))),
            "time_bnds": (("time", "bnds"), bounds),
        },
        coords={"time": ("time", time, {"units": units, "calendar": "proleptic_gregorian", "bounds": "time_bnds"})},
    )
    ds.to_netcdf(path)
    return path


def test_final_bound_reaches_the_projection_end(tmp_path: Path):
    """
    The last record stays valid until 2500, the others are untouched.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    path = write_retreat_file(tmp_path / "retreat.nc")
    extend_final_time_bound(path)

    with xr.open_dataset(path, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as ds:
        assert str(ds["time_bnds"].values[-1][1])[:10] == FRONT_RETREAT_END
        # The record's own timestamp does not move -- only its validity.
        assert str(ds["time"].values[-1])[:10] == "2019-06-01"
        # Earlier records keep their one-month intervals, still contiguous.
        assert (ds["time_bnds"].values[0][1] - ds["time_bnds"].values[0][0]).days == 30
        assert str(ds["time_bnds"].values[-2][1])[:10] == "2019-06-01"
        assert ds["time_bnds"].values[-2][1] == ds["time_bnds"].values[-1][0]


def test_extending_twice_changes_nothing(tmp_path: Path):
    """
    Idempotent, so it can run on a file that was already prepared.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    path = write_retreat_file(tmp_path / "retreat.nc")
    extend_final_time_bound(path)
    with xr.open_dataset(path, decode_times=False) as ds:
        once = ds["time_bnds"].values.copy()
    extend_final_time_bound(path)
    with xr.open_dataset(path, decode_times=False) as ds:
        np.testing.assert_array_equal(ds["time_bnds"].values, once)


def test_a_different_epoch_still_lands_on_2500(tmp_path: Path):
    """
    The bound is a date, not an offset, so the file's epoch cannot skew it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    path = write_retreat_file(tmp_path / "retreat.nc", units="days since 1900-01-01 00:00:00")
    extend_final_time_bound(path)
    with xr.open_dataset(path, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as ds:
        assert str(ds["time_bnds"].values[-1][1])[:10] == FRONT_RETREAT_END


def test_a_bound_before_the_last_record_is_refused(tmp_path: Path):
    """
    Refuse to leave the file with a backwards final interval.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    path = write_retreat_file(tmp_path / "retreat.nc")
    with pytest.raises(ValueError, match="not after the last record"):
        extend_final_time_bound(path, end="1990-01-01")


def test_a_file_without_bounds_is_an_error(tmp_path: Path):
    """
    Say which file lacks bounds rather than raising a bare KeyError.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    path = tmp_path / "nobounds.nc"
    xr.Dataset(
        {"thing": ("time", [1.0, 2.0])},
        coords={"time": ("time", [0.0, 1.0], {"units": "days since 2000-01-01"})},
    ).to_netcdf(path)
    with pytest.raises(KeyError, match="no time bounds"):
        extend_final_time_bound(path)
