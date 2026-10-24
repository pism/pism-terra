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
Tests for the ERA5 forcing fields.

These pin what is asked of CDS for a glacier's or a region's ERA5 forcing
and what is made of the answer, with the download itself replaced by
:class:`FakeCDS`, which the other ERA5 tests use too. The forcing has no
``air_temp_sd``: the surface models use a constant temperature spread, so
the daily statistics are not downloaded.
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
    """Answer ``download_request`` from memory, recording what was asked."""

    def __init__(self):
        """Set up the stand-in."""
        self.requests: list[tuple[str, dict | None]] = []

    def __call__(
        self, dataset, area=None, year=None, *, variable=None, file_path=None, request_override=None, **kwargs
    ):
        """
        Return the monthly fields or the geopotential.

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
            Verbatim request, recorded.
        **kwargs
            Anything else, ignored like ``download_request`` does.

        Returns
        -------
        xarray.Dataset
            The stand-in data.
        """
        self.requests.append((dataset, request_override))
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
def test_era5_writes_the_monthly_fields_without_a_spread(cds: FakeCDS, tmp_path: Path):
    """
    Write air temperature, precipitation and surface, month for month, and no spread.

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
        assert set(climate.ERA5_VARIABLES) <= set(ds.data_vars)
        assert "air_temp_sd" not in ds
        # Every month of both years: the last one ends on the first of the next.
        assert ds.sizes["time"] == 24
        assert pd.Timestamp(ds["time_bounds"].values[-1, 0]) == pd.Timestamp("1991-12-01")
        assert pd.Timestamp(ds["time_bounds"].values[-1, 1]) == pd.Timestamp("1992-01-01")
    # Monthly means and geopotential only: no daily statistics.
    assert not any("daily" in dataset for dataset, _ in cds.requests)


def test_a_global_patch_with_the_wrong_fields_is_an_error():
    """
    Refuse to patch from a download that holds none of the fields that need it.

    A cached global geopotential that in fact held temperature and
    precipitation patched nothing, without a word, and the holes over water
    were then filled from the nearest land cell.
    """
    lat, lon = _grid(AREA, 0.1)
    surface = np.full((len(lat), len(lon)), 1000.0)
    surface[0, 0] = np.nan
    coords = {"latitude": lat, "longitude": lon}
    ds_geo = _georeferenced(xr.Dataset({"z": (("latitude", "longitude"), surface)}, coords=coords))
    wrong = monthly_fields().isel(valid_time=0, drop=True)

    with pytest.raises(ValueError, match="none of the fields to patch"):
        climate._patch_from_global(ds_geo, wrong)  # pylint: disable=protected-access

    right = _georeferenced(xr.Dataset({"z": (("latitude", "longitude"), np.zeros_like(surface))}, coords=coords))
    patched = climate._patch_from_global(ds_geo, right)  # pylint: disable=protected-access
    assert float(patched["z"][0, 0]) == 0.0 and not bool(patched["z"].isnull().any())
