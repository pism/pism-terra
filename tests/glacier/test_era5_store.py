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
Tests for the regional ERA5 stores.

Asking CDS for a glacier's forcing takes hours, almost all of it the daily
means behind ``air_temp_sd``. ``prepare`` therefore builds the forcing once
per region and staging cuts each glacier out of it. These pin the store that
is written -- in the region's CRS, or in latitude/longitude without one --
and that staging uses it instead of CDS whenever it covers the glacier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401  # pylint: disable=unused-import
import xarray as xr
from pyproj import CRS, Transformer
from test_era5_air_temp_sd import YEARS, FakeCDS, daily_temperature

from pism_terra.domain import create_domain
from pism_terra.glacier import climate

#: ``(west, south, east, north)`` of a region's glaciers.
BOUNDS = (-151.0, 61.5, -149.0, 62.5)
ALASKA = "EPSG:5936"
GLACIER = "RGI2000-v7.0-C-01-00001"


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


def glacier_grid(crs: str = ALASKA, lon: float = -150.0, lat: float = 62.0) -> xr.Dataset:
    """
    Build a 20 km model grid around a point.

    Parameters
    ----------
    crs : str, optional
        CRS of the grid.
    lon : float, optional
        Longitude of its centre.
    lat : float, optional
        Latitude of its centre.

    Returns
    -------
    xarray.Dataset
        The grid, as staging builds it.
    """
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    x, y = round(x, -3), round(y, -3)
    return create_domain([x - 10_000.0, x + 10_000.0], [y - 10_000.0, y + 10_000.0], resolution=1000.0, crs=crs)


def january_1990() -> float:
    """
    Compute the spread of the stand-in's daily means in January 1990.

    Returns
    -------
    float
        The population standard deviation.
    """
    return float(np.std(daily_temperature(pd.date_range("1990-01-01", "1990-01-31", freq="D"))))


@pytest.mark.usefixtures("cds")
def test_a_region_without_a_crs_is_stored_in_latitude_and_longitude(tmp_path: Path):
    """
    The store keeps ERA5-Land's own grid when the setup file names no CRS.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    out = climate.prepare_era5("01_alaska", BOUNDS, tmp_path / "era5_01_alaska.zarr", tmp_path / "staging", years=YEARS)

    store = climate.open_era5_store(out)
    assert set(store.data_vars) == set(climate.ERA5_VARIABLES)
    assert store["air_temp"].dims == ("time", "latitude", "longitude")
    assert store.rio.crs == CRS("EPSG:4326")
    assert store.sizes["time"] == 24 and store.attrs["years"] == "1990-1991"
    # The box of the glaciers, and a margin around it.
    assert float(store["longitude"].min()) <= BOUNDS[0] and float(store["longitude"].max()) >= BOUNDS[2]
    assert float(store["latitude"].min()) <= BOUNDS[1] and float(store["latitude"].max()) >= BOUNDS[3]
    # The stand-in's fields, converted as the per-glacier download converts them.
    assert np.allclose(store["air_temp"], 265.0)
    assert np.allclose(store["precipitation"], 2.0)
    assert np.allclose(store["surface"], 1000.0)
    assert np.allclose(store["air_temp_sd"].isel(time=0), january_1990(), rtol=1e-5)


@pytest.mark.usefixtures("cds")
def test_a_region_with_a_crs_is_stored_in_it(tmp_path: Path):
    """
    The store is a regular 5 km grid in the region's CRS, with no gaps.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    out = climate.prepare_era5(
        "01_alaska", BOUNDS, tmp_path / "era5_01_alaska.zarr", tmp_path / "staging", crs=ALASKA, years=YEARS
    )

    store = climate.open_era5_store(out)
    assert store["air_temp"].dims == ("time", "y", "x")
    assert store["surface"].dims == ("y", "x")
    assert store.rio.crs == CRS(ALASKA)
    assert np.allclose(np.diff(store["x"]), 5000.0) and np.allclose(np.diff(store["y"]), 5000.0)
    # Every glacier of the region lies inside the grid.
    to_crs = Transformer.from_crs("EPSG:4326", ALASKA, always_xy=True)
    x_min, y_min, x_max, y_max = to_crs.transform_bounds(*BOUNDS, densify_pts=21)
    assert float(store["x"].min()) < x_min and float(store["x"].max()) > x_max
    assert float(store["y"].min()) < y_min and float(store["y"].max()) > y_max
    # The corners of the rectangle, outside the requested box, are filled: nothing missing.
    for name in climate.ERA5_VARIABLES:
        assert not bool(store[name].isnull().any()), name
    assert np.allclose(store["air_temp"], 265.0)
    assert np.allclose(store["air_temp_sd"].isel(time=0), january_1990(), rtol=1e-5)


class LatitudeCDS(FakeCDS):
    """
    Answer with a monthly temperature of ``250 + latitude``, so a cell says where it came from.
    """

    def __call__(self, dataset, *args, **kwargs):
        """
        Return the stand-in data, with the monthly temperature following latitude.

        Parameters
        ----------
        dataset : str
            CDS dataset name.
        *args
            Passed on.
        **kwargs
            Passed on.

        Returns
        -------
        xarray.Dataset
            The stand-in data.
        """
        ds = super().__call__(dataset, *args, **kwargs)
        if dataset != climate.ERA5_DAILY_DATASET and "t2m" in ds:
            ds["t2m"] = ds["t2m"] * 0.0 + 250.0 + ds["latitude"]
        return ds


def test_alaska_is_built_without_crossing_the_antimeridian(tmp_path: Path, monkeypatch):
    """
    Georeference Alaska's store correctly while asking CDS only for the glaciers' box.

    In EPSG:5936 the rectangle that holds Alaska's glaciers has corners
    beyond 180 E. ERA5 is requested for the glaciers' own box, which does not
    cross, and the corners are filled from the nearest cell with data.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture replacing ``climate.download_request``.
    """
    cds = LatitudeCDS()
    monkeypatch.setattr(climate, "download_request", cds)
    alaska = (-176.1, 52.1, -128.7, 69.3)  # the S4F glaciers of RGI region 1

    out = climate.prepare_era5(
        "01_alaska", alaska, tmp_path / "era5_01_alaska.zarr", tmp_path / "staging", crs=ALASKA, years=[1990]
    )

    # No request reaches across 180 E: west of east in every box asked for.
    for _, request in cds.requests:
        if request is not None:
            _north, west, _south, east = request["area"]
            assert -180.0 <= west < east <= 180.0

    store = climate.open_era5_store(out)
    for name in climate.ERA5_VARIABLES:
        assert not bool(store[name].isnull().any()), name
    assert "outside the box" in store.attrs["comment"]

    # Where ERA5 was requested, a cell holds the temperature of its own latitude.
    x, y = np.meshgrid(store["x"].values, store["y"].values)
    lon, lat = Transformer.from_crs(ALASKA, "EPSG:4326", always_xy=True).transform(x, y)
    inside = (lon > alaska[0]) & (lon < alaska[2]) & (lat > alaska[1]) & (lat < alaska[3])
    assert inside.sum() > 10_000
    january = store["air_temp"].isel(time=0).values
    assert np.abs(january[inside] - (250.0 + lat[inside])).max() < 0.02


def test_a_complete_store_is_not_built_again(cds: FakeCDS, tmp_path: Path):
    """
    A second run reuses the store; ``force_overwrite`` rebuilds it.

    Parameters
    ----------
    cds : FakeCDS
        Stand-in for the CDS download.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    out = tmp_path / "era5_01_alaska.zarr"
    climate.prepare_era5("01_alaska", BOUNDS, out, tmp_path / "staging", years=YEARS)
    asked = len(cds.requests)

    climate.prepare_era5("01_alaska", BOUNDS, out, tmp_path / "staging", years=YEARS)
    assert len(cds.requests) == asked

    climate.prepare_era5("01_alaska", BOUNDS, out, tmp_path / "staging", years=YEARS, force_overwrite=True)
    assert len(cds.requests) > asked


@pytest.mark.parametrize("crs", [None, ALASKA])
def test_staging_cuts_the_glacier_out_of_the_store(crs: str | None, cds: FakeCDS, tmp_path: Path, monkeypatch):
    """
    With a store that covers it, a glacier is staged without asking CDS for anything.

    Parameters
    ----------
    crs : str or None
        CRS of the store; None for latitude/longitude.
    cds : FakeCDS
        Stand-in for the CDS download.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture replacing the S3 listing.
    """
    store = climate.prepare_era5(
        "01_alaska", BOUNDS, tmp_path / "era5_01_alaska.zarr", tmp_path / "prep", crs=crs, years=YEARS
    )
    monkeypatch.setattr(climate, "_list_era5_stores", lambda bucket, prefix, project_directory=None: [str(store)])
    asked = len(cds.requests)
    grid = glacier_grid()

    out = climate.era5(grid, GLACIER, years=YEARS, path=tmp_path / "stage", bucket="b", prefix="p")

    assert len(cds.requests) == asked
    assert out.name == (f"era5_{GLACIER}.nc" if crs else f"era5_wgs84_{GLACIER}.nc")
    with xr.open_dataset(out, decode_coords="all") as ds:
        assert set(climate.ERA5_VARIABLES) <= set(ds.data_vars)
        # Every month of both years, each with its own bounds.
        assert ds.sizes["time"] == 24
        bounds = ds["time_bounds"].values
        assert pd.Timestamp(bounds[0, 0]) == pd.Timestamp("1990-01-01")
        assert pd.Timestamp(bounds[-1, 1]) == pd.Timestamp("1992-01-01")
        assert (bounds[1:, 0] == bounds[:-1, 1]).all()
        assert not bool(ds["air_temp_sd"].isnull().any())
        assert np.allclose(ds["air_temp_sd"].isel(time=0), january_1990(), rtol=1e-5)
        assert ds.rio.crs == CRS(crs or "EPSG:4326")
        # Cropped to the glacier, with data past every edge of its grid.
        x_dim, y_dim = ("x", "y") if crs else ("longitude", "latitude")
        to_file = Transformer.from_crs(ALASKA, ds.rio.crs, always_xy=True)
        x_min, y_min, x_max, y_max = to_file.transform_bounds(
            float(grid.x_bnds[0, 0]), float(grid.y_bnds[0, 0]), float(grid.x_bnds[-1, -1]), float(grid.y_bnds[-1, -1])
        )
        assert float(ds[x_dim].min()) < x_min and float(ds[x_dim].max()) > x_max
        assert float(ds[y_dim].min()) < y_min and float(ds[y_dim].max()) > y_max
        with climate.open_era5_store(store) as whole:
            assert ds.sizes[x_dim] < whole.sizes[x_dim] and ds.sizes[y_dim] < whole.sizes[y_dim]


def test_a_store_that_does_not_cover_the_glacier_is_not_used(cds: FakeCDS, tmp_path: Path, monkeypatch):
    """
    A glacier outside the store, or a year it lacks, goes to CDS as before.

    Parameters
    ----------
    cds : FakeCDS
        Stand-in for the CDS download.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture replacing the S3 listing.
    """
    store = climate.prepare_era5("01_alaska", BOUNDS, tmp_path / "era5_01_alaska.zarr", tmp_path / "prep", years=YEARS)
    monkeypatch.setattr(climate, "_list_era5_stores", lambda bucket, prefix, project_directory=None: [str(store)])

    assert climate.find_era5_store(glacier_grid(), YEARS, "b", "p", rgi_id=GLACIER) is not None
    assert climate.find_era5_store(glacier_grid(lon=-140.0), YEARS, "b", "p", rgi_id=GLACIER) is None
    assert climate.find_era5_store(glacier_grid(), [1989, 1990], "b", "p", rgi_id=GLACIER) is None

    asked = len(cds.requests)
    out = climate.era5(glacier_grid(lon=-140.0), GLACIER, years=YEARS, path=tmp_path / "stage", bucket="b", prefix="p")
    assert len(cds.requests) > asked
    assert out.name == f"era5_wgs84_{GLACIER}.nc"


def test_without_a_bucket_the_store_is_not_looked_for(cds: FakeCDS, tmp_path: Path, monkeypatch):
    """
    A caller that names no bucket gets the per-glacier download, with no S3 access.

    Parameters
    ----------
    cds : FakeCDS
        Stand-in for the CDS download.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture making any S3 listing fail.
    """

    def no_listing(*args, **kwargs):
        """
        Fail if the stores are listed.

        Parameters
        ----------
        *args
            Ignored.
        **kwargs
            Ignored.
        """
        raise AssertionError("S3 was listed")

    monkeypatch.setattr(climate, "_list_era5_stores", no_listing)

    climate.era5(glacier_grid(), GLACIER, years=YEARS, path=tmp_path)

    assert cds.requests
