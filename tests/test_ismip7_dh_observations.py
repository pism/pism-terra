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
Tests for the observed thickness-change products.

The observation file states dH/dt; a run reports elevation change. Turning
one into the other hinges on the interval each rate covers, and the file
gives bounds for neither source -- so the intervals are what these test.
"""

from __future__ import annotations

from pathlib import Path

import cftime
import numpy as np
import pytest
import xarray as xr

from pism_terra.ismip7.greenland.forcing import (
    DAYS_PER_YEAR,
    DH_SMITH_END,
    DH_SMITH_START,
    OBS_CALENDAR,
    OBS_TIME_UNITS,
    annual_bounds,
    cumulative_dh,
    prepare_dh_observations,
)


def _days(year: int, month: int = 1, day: int = 1) -> float:
    """
    Encode a date in the observation file's time units.

    Parameters
    ----------
    year, month, day : int
        Calendar date.

    Returns
    -------
    float
        Days since the reference epoch.
    """
    return float(
        cftime.date2num(cftime.datetime(year, month, day, calendar=OBS_CALENDAR), OBS_TIME_UNITS, OBS_CALENDAR)
    )


def test_annual_bounds_brackets_mid_year_stamps_with_calendar_years():
    """
    A rate stamped 1 July 2003 is the mean for calendar year 2003.

    Taking midpoints between stamps instead would shift every interval by
    half a year, and with it every cumulative total.
    """
    times = np.array([_days(y, 7, 1) for y in (2003, 2004, 2005)])
    bounds = annual_bounds(times)

    assert bounds[0].tolist() == [_days(2003), _days(2004)]
    assert bounds[-1].tolist() == [_days(2005), _days(2006)]
    # Consecutive intervals meet, leaving no gap for change to vanish into.
    assert bounds[0][1] == bounds[1][0]


def test_annual_bounds_rejects_stamps_that_are_not_annual():
    """
    Two stamps in one year would mean the mid-year reading is wrong.
    """
    times = np.array([_days(2003, 1, 1), _days(2003, 7, 1)])
    with pytest.raises(ValueError, match="one rate per year"):
        annual_bounds(times)


def test_cumulative_dh_accumulates_and_keeps_the_start_in_bounds():
    """
    Each record is the change since the first interval began.

    That is what a run's ``usurf(t) - usurf(t0)`` gives, so the two can be
    compared without re-deriving one of them.
    """
    times = np.array([_days(y, 7, 1) for y in (2003, 2004, 2005)])
    bounds = annual_bounds(times)
    rate = xr.DataArray(
        np.array([[-1.0], [-2.0], [-3.0]]), dims=("t", "x"), coords={"t": times}, attrs={"units": "m/yr"}
    )

    out = cumulative_dh(rate, bounds, "t")

    spans = (bounds[:, 1] - bounds[:, 0]) / DAYS_PER_YEAR
    expected = np.cumsum([-1.0, -2.0, -3.0] * spans)
    np.testing.assert_allclose(out["dh"].values.ravel(), expected)
    # Every record's lower bound is the start of the record, not of its own interval.
    assert (out["time_bnds"].values[:, 0] == bounds[0, 0]).all()
    assert out["time_bnds"].values[-1, 1] == bounds[-1, 1]
    # ``time`` sits at the end of the interval, as postprocess_dh writes it.
    assert out["time"].values.tolist() == bounds[:, 1].tolist()
    assert out["dh"].attrs["units"] == "m"


def test_cumulative_dh_does_not_turn_gaps_into_zeros():
    """
    A cell the survey never saw stays NaN.

    ``cumsum`` treats NaN as zero, which would report confident no-change
    over exactly the places with no observation.
    """
    times = np.array([_days(y, 7, 1) for y in (2003, 2004)])
    rate = xr.DataArray(np.array([[-1.0, np.nan], [-1.0, np.nan]]), dims=("t", "x"), coords={"t": times})

    out = cumulative_dh(rate, annual_bounds(times), "t")

    assert np.isnan(out["dh"].values[:, 1]).all()
    assert np.isfinite(out["dh"].values[:, 0]).all()


def _write_obs(path: Path, n_years: int = 3) -> Path:
    """
    Write a miniature observation file shaped like the ISMIP7 one.

    Parameters
    ----------
    path : pathlib.Path
        Destination file.
    n_years : int, optional
        Number of Khan epochs.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    times = np.array([_days(2003 + i, 7, 1) for i in range(n_years)])
    ds = xr.Dataset(
        {
            "dhdt_khan": (
                ("khan_dhdt_time", "y1km", "x1km"),
                np.full((n_years, 2, 2), -1.0),
                {"units": "m/yr ice equivalent", "source": "Khan et al. 2025"},
            ),
            "dhdt_smith": (
                ("y1km", "x1km"),
                np.full((2, 2), -0.5),
                {"units": "m/yr ice equivalent", "source": "Smith et al. 2020", "time": _days(2019)},
            ),
            "mapping": ((), np.int8(0), {"grid_mapping_name": "polar_stereographic"}),
        },
        coords={
            "khan_dhdt_time": ("khan_dhdt_time", times, {"units": OBS_TIME_UNITS}),
            "y1km": [0.0, 1000.0],
            "x1km": [0.0, 1000.0],
        },
    )
    ds.to_netcdf(path)
    return path


def test_prepare_writes_one_product_per_source(tmp_path: Path):
    """
    Two files, because the two sources have different time axes.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    obs = _write_obs(tmp_path / "GreenlandObsISMIP7-v1.3.nc")
    products = prepare_dh_observations(obs, tmp_path / "out")

    assert set(products) == {"khan", "smith"}
    assert all(p.exists() for p in products.values())

    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(products["smith"], decode_times=coder) as smith:
        assert smith.sizes["time"] == 1
        assert f"{smith['time_bnds'].values[0][0]:%Y-%m-%d}" == DH_SMITH_START
        assert f"{smith['time_bnds'].values[0][1]:%Y-%m-%d}" == DH_SMITH_END
        # -0.5 m/yr over 16 years.
        span = (_days(2019) - _days(2003)) / DAYS_PER_YEAR
        np.testing.assert_allclose(smith["dh"].values.ravel(), -0.5 * span, rtol=1e-6)

    with xr.open_dataset(products["khan"], decode_times=coder) as khan:
        assert khan.sizes["time"] == 3
        # Named x/y, not y1km/x1km: this grid is the ISMIP7 submission grid,
        # so a run aligns against it without either being renamed or flipped.
        assert list(khan["dh"].dims) == ["time", "y", "x"]
        assert "mapping" in khan
        assert khan["dh"].attrs["grid_mapping"] == "mapping"
        # -1 m/yr accumulating over three calendar years.
        assert khan["dh"].values[-1].mean() < khan["dh"].values[0].mean() < 0


def test_the_products_align_with_a_run_without_reindexing(tmp_path: Path):
    """
    A submission file minus these is a plain subtraction.

    The observations run y north-to-south and a run south-to-north; xarray
    reconciles that on the coordinate values, but only if the dimensions are
    named the same. They were ``y1km``/``x1km``, which alignment cannot see
    through.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    obs = _write_obs(tmp_path / "GreenlandObsISMIP7-v1.3.nc")
    products = prepare_dh_observations(obs, tmp_path / "out")

    with xr.open_dataset(products["smith"]) as smith:
        # A run's grid: same coordinates, opposite y order.
        run = xr.DataArray(
            np.zeros((2, 2)),
            dims=("y", "x"),
            coords={"y": smith["y"].values[::-1], "x": smith["x"].values},
        )
        diff = run - smith["dh"].isel(time=0)
        assert diff.sizes == {"y": 2, "x": 2}
        # -0.5 m/yr over the Smith period, subtracted from zero.
        assert float(diff.mean()) > 0


def test_prepare_reuses_existing_products(tmp_path: Path):
    """
    A second call does not rebuild; ``force_overwrite`` does.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    obs = _write_obs(tmp_path / "GreenlandObsISMIP7-v1.3.nc")
    first = prepare_dh_observations(obs, tmp_path / "out")
    stamps = {k: p.stat().st_mtime_ns for k, p in first.items()}

    prepare_dh_observations(obs, tmp_path / "out")
    assert {k: p.stat().st_mtime_ns for k, p in first.items()} == stamps

    prepare_dh_observations(obs, tmp_path / "out", force_overwrite=True)
    assert {k: p.stat().st_mtime_ns for k, p in first.items()} != stamps
