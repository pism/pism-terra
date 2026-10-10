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
Tests for the ERA5 ``air_temp_sd``.

The monthly means ERA5 forcing is built from carry no day-to-day variability,
so it comes from a second download of daily means. These pin what is asked of
CDS and what is made of the answer, with the download itself replaced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401  # pylint: disable=unused-import
import xarray as xr

from pism_terra.domain import create_domain
from pism_terra.glacier import climate

#: [South, West, North, East], the slot order ``transform_bounds`` hands over.
AREA = [60.4, -144.6, 62.6, -140.9]
YEARS = [1990, 1991]


def _grid(area: list[float], spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the latitudes and longitudes of a regular grid covering a box.

    Parameters
    ----------
    area : list of float
        ``[South, West, North, East]``.
    spacing : float
        Grid spacing in degrees.

    Returns
    -------
    tuple of numpy.ndarray
        Increasing latitudes and longitudes.
    """
    south, west, north, east = area
    return np.arange(south, north + spacing / 2, spacing), np.arange(west, east + spacing / 2, spacing)


def _georeferenced(ds: xr.Dataset) -> xr.Dataset:
    """
    Stamp a dataset the way ``download_request`` returns one.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset on ``latitude``/``longitude``.

    Returns
    -------
    xarray.Dataset
        The dataset with spatial dims and an EPSG:4326 CRS.
    """
    return ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude").rio.write_crs("EPSG:4326")


def daily_temperature(day: pd.DatetimeIndex) -> np.ndarray:
    """
    A daily-mean temperature whose spread differs from month to month.

    Parameters
    ----------
    day : pandas.DatetimeIndex
        The days.

    Returns
    -------
    numpy.ndarray
        Temperature in kelvin, uniform in space.
    """
    month, dom, year = (np.asarray(part, dtype=float) for part in (day.month, day.day, day.year))
    return 260.0 + month * np.sin(dom) + 0.01 * (year - 1990) * dom


def monthly_fields(area: list[float] | None = None, years: list[int] | None = None) -> xr.Dataset:
    """
    Stand in for the ERA5-Land monthly means: first-of-month stamps, 0.1 degrees.

    Parameters
    ----------
    area : list of float, optional
        ``[South, West, North, East]``; :data:`AREA` by default.
    years : list of int, optional
        Years covered; :data:`YEARS` by default.

    Returns
    -------
    xarray.Dataset
        ``t2m`` and ``tp`` on ``valid_time``/``latitude``/``longitude``.
    """
    years = YEARS if years is None else years
    months = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-01", freq="MS")
    lat, lon = _grid(AREA if area is None else area, 0.1)
    shape = (len(months), len(lat), len(lon))
    ds = xr.Dataset(
        {
            "t2m": (("valid_time", "latitude", "longitude"), np.full(shape, 265.0)),
            "tp": (("valid_time", "latitude", "longitude"), np.full(shape, 0.002)),
        },
        coords={"valid_time": months, "latitude": lat, "longitude": lon},
    )
    return _georeferenced(ds)


class FakeCDS:
    """
    Answer ``download_request`` from memory, recording what was asked.

    Parameters
    ----------
    daily_years : list of int, optional
        Years the daily product holds.
    """

    def __init__(self, daily_years: list[int] | None = None):
        """
        Set up the stand-in.

        Parameters
        ----------
        daily_years : list of int, optional
            Years the daily product holds.
        """
        self.daily_years = YEARS if daily_years is None else daily_years
        self.requests: list[tuple[str, dict | None]] = []

    def __call__(
        self, dataset, area=None, year=None, *, variable=None, file_path=None, request_override=None, **kwargs
    ):
        """
        Return the monthly fields, the geopotential or the daily means.

        Parameters
        ----------
        dataset : str
            CDS dataset name.
        area : list of float, optional
            Requested box.
        year : list of int, optional
            Requested years.
        variable : list of str, optional
            Requested variables.
        file_path : pathlib.Path, optional
            Cache file, unused.
        request_override : dict, optional
            Verbatim request, as the daily statistics use.
        **kwargs
            Anything else, ignored like ``download_request`` does.

        Returns
        -------
        xarray.Dataset
            The stand-in data.
        """
        self.requests.append((dataset, request_override))
        if dataset == climate.ERA5_DAILY_DATASET:
            north, west, south, east = request_override["area"]
            days = pd.date_range(f"{self.daily_years[0]}-01-01", f"{self.daily_years[-1]}-12-31", freq="D")
            lat, lon = _grid([south, west, north, east], 0.25)
            values = daily_temperature(days)[:, None, None] * np.ones((1, len(lat), len(lon)))
            ds = xr.Dataset(
                {"t2m": (("valid_time", "latitude", "longitude"), values)},
                coords={"valid_time": days, "latitude": lat, "longitude": lon},
            )
            return _georeferenced(ds)
        if variable == ["geopotential"]:
            lat, lon = _grid(area, 0.1)
            z = np.full((1, len(lat), len(lon)), 9806.65)
            ds = xr.Dataset(
                {"z": (("time", "latitude", "longitude"), z)},
                coords={"time": [pd.Timestamp("2013-01-01")], "latitude": lat, "longitude": lon},
            )
            return _georeferenced(ds)
        return monthly_fields(area)


@pytest.fixture(name="cds")
def fixture_cds(monkeypatch) -> FakeCDS:
    """
    Replace the CDS download with the stand-in.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture replacing ``climate.download_request``.

    Returns
    -------
    FakeCDS
        The stand-in, holding the requests it was sent.
    """
    cds = FakeCDS()
    monkeypatch.setattr(climate, "download_request", cds)
    return cds


@pytest.mark.usefixtures("cds")
def test_the_spread_of_the_daily_means_within_each_month(tmp_path: Path):
    """
    Each month gets the population standard deviation of its own days.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    ds = monthly_fields()

    sd = climate.era5_air_temp_sd(ds, AREA, YEARS, tmp_path, "RGI-test")

    assert sd.name == "air_temp_sd" and sd.attrs["units"] == "kelvin"
    assert sd.dims == ("valid_time", "latitude", "longitude")
    # On the grid and the months of the monthly fields, exactly.
    for name in ("valid_time", "latitude", "longitude"):
        assert sd[name].equals(ds[name])
    assert not bool(sd.isnull().any())
    for month in pd.DatetimeIndex(ds["valid_time"].values):
        days = pd.date_range(month, month + pd.offsets.MonthEnd(0), freq="D")
        expected = np.std(daily_temperature(days))  # ddof=0, like ``cdo monstd``
        assert np.allclose(sd.sel(valid_time=month).values, expected, rtol=1e-5)
    # February 1990 and February 1991 differ, so years are not mixed.
    assert float(sd.sel(valid_time="1990-02-01").mean()) != pytest.approx(float(sd.sel(valid_time="1991-02-01").mean()))


def test_the_request_is_one_year_of_daily_means_on_a_padded_box(cds: FakeCDS, tmp_path: Path):
    """
    CDS is asked for daily means of the hourly reanalysis, with a margin around the box.

    Parameters
    ----------
    cds : FakeCDS
        Stand-in for the CDS download.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    climate.era5_air_temp_sd(monthly_fields(), AREA, YEARS, tmp_path, "RGI-test")

    ((dataset, request),) = cds.requests
    assert dataset == "derived-era5-single-levels-daily-statistics"
    assert request is not None
    assert request["variable"] == ["2m_temperature"]
    assert (request["daily_statistic"], request["frequency"], request["time_zone"]) == (
        "daily_mean",
        "1_hourly",
        "utc+00:00",
    )
    assert request["year"] == ["1990", "1991"]
    assert len(request["month"]) == 12 and len(request["day"]) == 31
    # [North, West, South, East], one degree past the monthly fields on every side.
    assert request["area"] == pytest.approx([63.6, -145.6, 59.4, -139.9])


def test_a_missing_month_is_an_error(monkeypatch, tmp_path: Path):
    """
    A year the daily product does not hold fails instead of leaving NaN for PISM.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture replacing ``climate.download_request``.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    monkeypatch.setattr(climate, "download_request", FakeCDS(daily_years=[1990]))

    with pytest.raises(ValueError, match="12 of 24 months"):
        climate.era5_air_temp_sd(monthly_fields(), AREA, YEARS, tmp_path, "RGI-test")


def test_era5_writes_air_temp_sd(cds: FakeCDS, tmp_path: Path):
    """
    The forcing file carries ``air_temp_sd`` beside ``air_temp``, month for month.

    Parameters
    ----------
    cds : FakeCDS
        Stand-in for the CDS download.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    grid = create_domain([460_000.0, 500_000.0], [6_780_000.0, 6_820_000.0], resolution=1000.0, crs="EPSG:32607")

    out = climate.era5(grid, "RGI-test", years=YEARS, path=tmp_path)

    with xr.open_dataset(out) as ds:
        assert {"air_temp", "air_temp_sd", "precipitation", "surface"} <= set(ds.data_vars)
        assert ds["air_temp_sd"].dims == ds["air_temp"].dims
        # Every month of both years: the last one ends on the first of the next.
        assert ds.sizes["time"] == 24
        assert pd.Timestamp(ds["time_bounds"].values[-1, 0]) == pd.Timestamp("1991-12-01")
        assert pd.Timestamp(ds["time_bounds"].values[-1, 1]) == pd.Timestamp("1992-01-01")
        assert ds["air_temp_sd"].attrs["units"] == "kelvin"
        assert not bool(ds["air_temp_sd"].isnull().any())
        january = pd.date_range("1990-01-01", "1990-01-31", freq="D")
        assert np.allclose(ds["air_temp_sd"].isel(time=0).values, np.std(daily_temperature(january)), rtol=1e-5)
    assert [dataset for dataset, _ in cds.requests].count(climate.ERA5_DAILY_DATASET) == 1
