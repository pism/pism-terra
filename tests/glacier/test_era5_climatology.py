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
Tests for the ERA5 monthly climatology.

``era5-monthly-mean`` is the forcing a spin-up cycles. It used to be the
twelve months of 1978, downloaded per glacier, without ``air_temp_sd``, and
written under the name of the transient forcing, which it replaced. These pin
the CARRA2 workflow it now follows: ``prepare`` averages a region's store over
a fixed reference period, and staging cuts each glacier out of that into a
file of its own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401  # pylint: disable=unused-import
import xarray as xr
from pyproj import CRS
from test_era5_air_temp_sd import YEARS, FakeCDS, daily_temperature
from test_era5_store import BOUNDS, GLACIER, glacier_grid

from pism_terra.glacier import climate


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


def january_spread() -> float:
    """
    Compute the January ``air_temp_sd`` of the stand-in's climatology.

    Returns
    -------
    float
        The mean, over the stand-in's years, of the spread of the daily means
        within each January.
    """
    return float(
        np.mean(
            [np.std(daily_temperature(pd.date_range(f"{year}-01-01", f"{year}-01-31", freq="D"))) for year in YEARS]
        )
    )


@pytest.fixture(name="stores")
def fixture_stores(cds: FakeCDS, tmp_path: Path) -> tuple[Path, Path]:
    """
    Build a region's store and its climatology as ``prepare`` does.

    Parameters
    ----------
    cds : FakeCDS
        Stand-in for the CDS download.
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    tuple of pathlib.Path
        The series store and the climatology store.
    """
    _ = cds
    series = climate.prepare_era5("01_alaska", BOUNDS, tmp_path / "era5_01_alaska.zarr", tmp_path / "prep", years=YEARS)
    climatology = climate.prepare_era5_monthly_mean(series, tmp_path / "era5_monthly_mean_01_alaska.zarr", years=YEARS)
    return series, climatology


def test_the_reference_period_is_carra2s():
    """Average ERA5 over the same years as CARRA2, 1990-2019."""
    assert list(climate.ERA5_CLIMATOLOGY_YEARS) == list(range(1990, 2020))
    assert list(climate.ERA5_CLIMATOLOGY_YEARS) == list(climate.CARRA2_CLIMATOLOGY_YEARS)


def test_a_store_is_averaged_by_calendar_month(stores: tuple[Path, Path]):
    """
    Write twelve months on the store's own grid, the spread averaged like the rest.

    Parameters
    ----------
    stores : tuple of pathlib.Path
        The series store and the climatology store.
    """
    series, climatology = (climate.open_era5_store(store) for store in stores)

    assert set(climatology.data_vars) == set(climate.ERA5_VARIABLES)
    assert climatology.sizes["time"] == 12 and list(climatology["time"].values) == list(range(1, 13))
    assert climatology.attrs["climatology_period"] == "1990-1991"
    assert climatology.rio.crs == CRS("EPSG:4326")
    assert np.array_equal(climatology["longitude"], series["longitude"])
    assert np.array_equal(climatology["latitude"], series["latitude"])
    assert np.allclose(climatology["air_temp"], 265.0)
    assert np.allclose(climatology["precipitation"], 2.0)
    # The mean of the Januaries' spreads, not the spread of two Januaries.
    assert np.allclose(climatology["air_temp_sd"].isel(time=0), january_spread(), rtol=1e-5)
    # The surface has no time axis to average over.
    assert climatology["surface"].dims == ("latitude", "longitude")
    assert np.allclose(climatology["surface"], 1000.0)


def test_a_period_outside_the_store_is_rejected(stores: tuple[Path, Path], tmp_path: Path):
    """
    Fail rather than write an empty climatology.

    Parameters
    ----------
    stores : tuple of pathlib.Path
        The series store and the climatology store.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    with pytest.raises(ValueError, match="no time steps in 2000-2019; it covers 1990-1991"):
        climate.prepare_era5_monthly_mean(stores[0], tmp_path / "out.zarr", years=range(2000, 2020))


def test_staging_cuts_the_glacier_out_of_the_climatology(
    cds: FakeCDS, stores: tuple[Path, Path], tmp_path: Path, monkeypatch
):
    """
    Stage a glacier's climatology from the regional one without asking CDS.

    Parameters
    ----------
    cds : FakeCDS
        Stand-in for the CDS download.
    stores : tuple of pathlib.Path
        The series store and the climatology store.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture replacing the S3 listing.
    """
    monkeypatch.setattr(
        climate, "_list_era5_climatology_stores", lambda bucket, prefix, project_directory=None: [str(stores[1])]
    )
    asked = len(cds.requests)

    # The years of the run are not the reference period, and are ignored.
    out = climate.era5_monthly_mean(
        glacier_grid(), GLACIER, years=[2000, 2001], path=tmp_path / "stage", bucket="b", prefix="p"
    )

    assert len(cds.requests) == asked
    assert out.name == f"era5_monthly_mean_{GLACIER}.nc"
    with xr.open_dataset(out, decode_times=False, decode_coords="all") as ds:
        assert set(climate.ERA5_VARIABLES) <= set(ds.data_vars)
        assert ds["air_temp"].dims == ("time", "latitude", "longitude")
        assert ds.rio.crs == CRS("EPSG:4326")
        # The periodic axis PISM cycles: a 365-day year counted from year 1.
        assert ds.sizes["time"] == 12
        assert ds["time"].attrs["units"] == "days since 0001-01-01"
        assert ds["time"].attrs["calendar"] == "365_day"
        bounds = ds["time_bounds"].values
        assert bounds[0, 0] == 0.0 and bounds[-1, 1] == 365.0
        assert (bounds[1:, 0] == bounds[:-1, 1]).all()
        assert np.allclose(ds["air_temp_sd"].isel(time=0), january_spread(), rtol=1e-5)
        assert not bool(ds["air_temp_sd"].isnull().any())
        # The glacier's box only, on the store's own cells.
        whole = climate.open_era5_store(stores[1])
        assert ds.sizes["longitude"] < whole.sizes["longitude"] and ds.sizes["latitude"] < whole.sizes["latitude"]
        assert np.isin(ds["longitude"].values, whole["longitude"].values).all()
        assert np.isin(ds["latitude"].values, whole["latitude"].values).all()


def test_the_climatology_does_not_replace_the_transient_forcing(stores: tuple[Path, Path], tmp_path: Path, monkeypatch):
    """
    Keep both files when a run spins up on the climatology of its own forcing.

    Parameters
    ----------
    stores : tuple of pathlib.Path
        The series store and the climatology store.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture replacing the S3 listings.
    """
    monkeypatch.setattr(climate, "_list_era5_stores", lambda bucket, prefix, project_directory=None: [str(stores[0])])
    monkeypatch.setattr(
        climate, "_list_era5_climatology_stores", lambda bucket, prefix, project_directory=None: [str(stores[1])]
    )
    grid, stage = glacier_grid(), tmp_path / "stage"

    transient = climate.era5(grid, GLACIER, years=YEARS, path=stage, bucket="b", prefix="p")
    climatology = climate.era5_monthly_mean(grid, GLACIER, years=YEARS, path=stage, bucket="b", prefix="p")

    assert transient != climatology and transient.parent == climatology.parent
    with xr.open_dataset(transient) as ds:
        assert ds.sizes["time"] == 24
    with xr.open_dataset(climatology, decode_times=False) as ds:
        assert ds.sizes["time"] == 12


def test_without_a_climatology_nothing_is_downloaded(cds: FakeCDS, tmp_path: Path, monkeypatch):
    """
    Refuse, naming the prepare step, when no regional climatology covers the glacier.

    The reference period is thirty years of daily means, one CDS request of
    hours each; a run must not queue that for one glacier by accident.

    Parameters
    ----------
    cds : FakeCDS
        Stand-in for the CDS download.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture replacing the S3 listing.
    """
    monkeypatch.setattr(climate, "_list_era5_climatology_stores", lambda bucket, prefix, project_directory=None: [])

    with pytest.raises(FileNotFoundError, match="pism-glacier-prepare --include era5"):
        climate.era5_monthly_mean(glacier_grid(), GLACIER, years=[2000], path=tmp_path, bucket="b", prefix="p")
    with pytest.raises(FileNotFoundError, match="no bucket given"):
        climate.era5_monthly_mean(glacier_grid(), GLACIER, years=[2000], path=tmp_path)

    assert not cds.requests
    assert not list(tmp_path.glob("*.nc"))


def test_climatologies_are_not_listed_as_forcing_stores(monkeypatch):
    """
    Keep the 12-step stores out of the search for years of forcing.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture replacing the S3 file system.
    """

    class FakeS3:
        """
        Stand in for the bucket listing.

        Parameters
        ----------
        **kwargs
            Ignored.
        """

        keys = [
            "b/p/s4f/climate/era5_01_alaska.zarr",
            "b/p/s4f/climate/era5_monthly_mean_01_alaska.zarr",
            "b/p/s4f/climate/era5_02_western_canada_usa.zarr",
        ]

        def __init__(self, **kwargs):
            """
            Accept the arguments of ``s3fs.S3FileSystem``.

            Parameters
            ----------
            **kwargs
                Ignored.
            """

        def glob(self, pattern: str) -> list[str]:
            """
            Match the keys against a pattern.

            Parameters
            ----------
            pattern : str
                Glob pattern.

            Returns
            -------
            list of str
                The matching keys.
            """
            return [key for key in self.keys if Path(key).match(Path(pattern).name)]

    monkeypatch.setattr(climate.s3fs, "S3FileSystem", FakeS3)

    assert climate._list_era5_stores("b", "p", "s4f") == [  # pylint: disable=protected-access
        "s3://b/p/s4f/climate/era5_01_alaska.zarr",
        "s3://b/p/s4f/climate/era5_02_western_canada_usa.zarr",
    ]
    assert climate._list_era5_climatology_stores("b", "p", "s4f") == [  # pylint: disable=protected-access
        "s3://b/p/s4f/climate/era5_monthly_mean_01_alaska.zarr"
    ]
