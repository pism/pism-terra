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
is written -- ERA5's own latitude/longitude grid, bounded by the region's
glaciers -- and that staging uses it instead of CDS whenever it covers the
glacier. Nothing is resampled on the way: PISM regrids the forcing itself.
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
def test_a_region_is_stored_on_era5s_own_grid(tmp_path: Path):
    """
    The store keeps ERA5-Land's latitude/longitude grid, bounded by the region's glaciers.

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
    # The box of the glaciers and a margin of a degree around it: no more, no less.
    assert np.allclose(np.diff(store["longitude"]), 0.1) and np.allclose(np.diff(store["latitude"]), 0.1)
    assert float(store["longitude"].min()) == pytest.approx(BOUNDS[0] - 1.0, abs=0.11)
    assert float(store["longitude"].max()) == pytest.approx(BOUNDS[2] + 1.0, abs=0.11)
    assert float(store["latitude"].min()) == pytest.approx(BOUNDS[1] - 1.0, abs=0.11)
    assert float(store["latitude"].max()) == pytest.approx(BOUNDS[3] + 1.0, abs=0.11)
    # The stand-in's fields, converted as the per-glacier download converts them.
    assert np.allclose(store["air_temp"], 265.0)
    assert np.allclose(store["precipitation"], 2.0)
    assert np.allclose(store["surface"], 1000.0)
    assert np.allclose(store["air_temp_sd"].isel(time=0), january_1990(), rtol=1e-5)


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


def test_a_store_in_another_layout_is_rebuilt(cds: FakeCDS, tmp_path: Path):
    """
    A store that is not on latitude/longitude is neither reused nor read by staging.

    An earlier version wrote the stores in the region's projected CRS; one
    left behind has the right variables and would otherwise pass for complete.

    Parameters
    ----------
    cds : FakeCDS
        Stand-in for the CDS download.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    out = tmp_path / "era5_01_alaska.zarr"
    projected = xr.Dataset(
        {name: (("y", "x"), np.zeros((2, 2), dtype="float32")) for name in climate.ERA5_VARIABLES},
        coords={"y": [0.0, 5000.0], "x": [0.0, 5000.0]},
    )
    projected.to_zarr(out, mode="w", consolidated=True)
    with pytest.raises(ValueError, match="latitude/longitude"):
        climate.open_era5_store(out)

    climate.prepare_era5("01_alaska", BOUNDS, out, tmp_path / "staging", years=YEARS)

    assert cds.requests
    assert climate.open_era5_store(out)["air_temp"].dims == ("time", "latitude", "longitude")


def test_staging_cuts_the_glacier_out_of_the_store(cds: FakeCDS, tmp_path: Path, monkeypatch):
    """
    With a store that covers it, a glacier is staged without asking CDS for anything.

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
    asked = len(cds.requests)
    grid = glacier_grid()

    out = climate.era5(grid, GLACIER, years=YEARS, path=tmp_path / "stage", bucket="b", prefix="p")

    assert len(cds.requests) == asked
    # The same file, on the same grid, as the per-glacier download writes.
    assert out.name == f"era5_wgs84_{GLACIER}.nc"
    with xr.open_dataset(out, decode_coords="all") as ds:
        assert set(climate.ERA5_VARIABLES) <= set(ds.data_vars)
        assert ds["air_temp"].dims == ("time", "latitude", "longitude")
        assert ds.rio.crs == CRS("EPSG:4326")
        # Every month of both years, each with its own bounds.
        assert ds.sizes["time"] == 24
        bounds = ds["time_bounds"].values
        assert pd.Timestamp(bounds[0, 0]) == pd.Timestamp("1990-01-01")
        assert pd.Timestamp(bounds[-1, 1]) == pd.Timestamp("1992-01-01")
        assert (bounds[1:, 0] == bounds[:-1, 1]).all()
        assert not bool(ds["air_temp_sd"].isnull().any())
        assert np.allclose(ds["air_temp_sd"].isel(time=0), january_1990(), rtol=1e-5)
        # Cropped to the glacier, with data past every edge of its grid.
        to_lonlat = Transformer.from_crs(ALASKA, "EPSG:4326", always_xy=True)
        west, south, east, north = to_lonlat.transform_bounds(
            float(grid.x_bnds[0, 0]), float(grid.y_bnds[0, 0]), float(grid.x_bnds[-1, -1]), float(grid.y_bnds[-1, -1])
        )
        assert float(ds["longitude"].min()) < west and float(ds["longitude"].max()) > east
        assert float(ds["latitude"].min()) < south and float(ds["latitude"].max()) > north
        whole = climate.open_era5_store(store)
        assert ds.sizes["longitude"] < whole.sizes["longitude"] and ds.sizes["latitude"] < whole.sizes["latitude"]
        # Nothing is resampled: the cells are the store's own.
        assert np.isin(ds["longitude"].values, whole["longitude"].values).all()
        assert np.isin(ds["latitude"].values, whole["latitude"].values).all()


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
