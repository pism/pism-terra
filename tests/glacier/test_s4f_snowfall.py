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
Tests for the S4F planning COG of mean annual CARRA2 snowfall.

CARRA2's monthly snowfall is a daily rate. The planning product is the
annual total, in metres water equivalent, averaged over 1990-2019. These pin
the arithmetic -- every month weighted by its own days -- and the COG that
lands on the planning grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import rioxarray  # noqa: F401  # pylint: disable=unused-import
import xarray as xr

from pism_terra.domain import create_domain
from pism_terra.glacier.s4f import (
    PLANNING_PRODUCTS,
    SNOWFALL_YEARS,
    mean_annual_snowfall,
    write_snowfall_cogs,
)

CRS = "EPSG:3413"


def monthly_snowfall(first_year: int, last_year: int, rate: float = 1.0) -> xr.DataArray:
    """
    Build a monthly snowfall series with the same daily rate in every month.

    Parameters
    ----------
    first_year, last_year : int
        First and last year of the series.
    rate : float, optional
        Snowfall in kg m^-2 day^-1.

    Returns
    -------
    xarray.DataArray
        Snowfall on ``time`` (stamped mid-month, as CARRA2 is), ``y`` and ``x``.
    """
    time = pd.date_range(f"{first_year}-01-15", f"{last_year}-12-15", freq="MS") + pd.Timedelta(days=14)
    return xr.DataArray(
        np.full((len(time), 2, 3), rate),
        coords={"time": time, "y": [0.0, 2500.0], "x": [0.0, 2500.0, 5000.0]},
        dims=("time", "y", "x"),
        name="snowfall",
    )


def carra2_store(path: Path, rate: float = 4.0) -> Path:
    """
    Write a CARRA2-shaped store with a constant snowfall rate.

    Parameters
    ----------
    path : pathlib.Path
        Destination Zarr store.
    rate : float, optional
        Snowfall in kg m^-2 day^-1.

    Returns
    -------
    pathlib.Path
        The store.
    """
    time = pd.date_range("1986-01-15", "2025-12-15", freq="MS") + pd.Timedelta(days=14)
    x = np.arange(0.0, 60_001.0, 2500.0)
    y = np.arange(-2_000_000.0, -1_939_999.0, 2500.0)
    ds = xr.Dataset(
        {"snowfall": (("time", "y", "x"), np.full((len(time), len(y), len(x)), rate, dtype="float32"))},
        coords={"time": time, "y": y, "x": x},
    )
    ds = ds.rio.write_crs(CRS).rio.write_grid_mapping()
    ds.to_zarr(path, mode="w", consolidated=True)
    return path


def test_snowfall_is_a_planning_product_averaged_over_1990_2019():
    """Ship the snowfall COG by default, averaged over the CARRA2 climatology's years."""
    assert "snowfall" in PLANNING_PRODUCTS
    assert list(SNOWFALL_YEARS) == list(range(1990, 2020))


def test_each_month_counts_its_own_days():
    """
    Weight every month by its number of days, leap-year Februaries included.

    1 kg m^-2 day^-1 is 365 mm in 1991 and 366 mm in 1992, so the mean over
    the two is 0.3655 m.
    """
    series = monthly_snowfall(1990, 1993)

    mean = mean_annual_snowfall(series, [1991, 1992])

    assert "time" not in mean.dims
    assert np.allclose(mean, (365 + 366) / 2 / 1000.0)
    assert mean.attrs["units"] == "m yr-1"
    assert mean.attrs["long_name"] == "mean annual snowfall 1991-1992"


def test_a_year_without_all_its_months_is_refused():
    """Refuse a year that would add up to too little snowfall."""
    series = monthly_snowfall(1990, 1992).isel(time=slice(0, -1))

    with pytest.raises(ValueError, match="incomplete: 1992 \\(11\\)"):
        mean_annual_snowfall(series, [1991, 1992])


def test_the_cog_covers_the_planning_grid(tmp_path: Path):
    """
    Write the mean on the planning grid, in metres water equivalent per year.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    store = carra2_store(tmp_path / "carra2.zarr", rate=4.0)
    grid = create_domain([20_000.0, 40_000.0], [-1_990_000.0, -1_970_000.0], resolution=1000.0, crs=CRS)

    written = write_snowfall_cogs({}, "RGI_TEST", grid, tmp_path, store=store)

    assert list(written) == ["RGI_TEST_carra2_snowfall_1990_2019"]
    path = written["RGI_TEST_carra2_snowfall_1990_2019"]
    assert path.name == "RGI_TEST_carra2_snowfall_1990_2019.tif"
    with rasterio.open(path) as src:
        values = src.read(1)
        assert src.crs.to_epsg() == 3413
        assert src.shape == (grid.sizes["y"], grid.sizes["x"])
    # 4 kg m^-2 day^-1 over the 10957 days of 1990-2019 (seven leap years),
    # averaged per year.
    assert np.allclose(values, 4.0 * (30 * 365 + 7) / 30 / 1000.0, rtol=1e-6)


def test_a_complex_outside_carra2_is_skipped(tmp_path: Path):
    """
    Write nothing when the store does not reach the planning grid.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    store = carra2_store(tmp_path / "carra2.zarr")
    grid = create_domain([500_000.0, 520_000.0], [0.0, 20_000.0], resolution=1000.0, crs=CRS)

    assert not write_snowfall_cogs({}, "RGI_FAR", grid, tmp_path, store=store)
    assert not list(tmp_path.glob("*.tif"))
