# Copyright (C) 2025 Andy Aschwanden
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

# mypy: disable-error-code="call-overload"
# pylint: disable=unused-import


"""
Prepare Climate.
"""

from collections.abc import Iterable
from pathlib import Path

import cdsapi
import cf_xarray
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

from pism_terra.dem import get_glacier_from_rgi_id
from pism_terra.download import extract_archive


def add_time_bounds(ds: xr.Dataset) -> xr.Dataset:
    """
    Add time bounds to a dataset by computing interval start and end times.

    This function computes time bounds for each time step in the dataset
    by pairing each timestamp with the following one, creating a bounds array
    with shape (n_time - 1, 2). The dataset is truncated by one time step to
    ensure alignment with the bounds.

    Parameters
    ----------
    ds : xr.Dataset
        An xarray Dataset with a one-dimensional "time" coordinate.

    Returns
    -------
    xr.Dataset
        A new dataset with:
        - one fewer time step (last one dropped),
        - a new variable "time_bounds" with shape (time, 2),
        - an attribute "bounds" set on the "time" coordinate pointing to "time_bounds".

    Notes
    -----
    - The function assumes that `ds["time"]` is sorted and regularly spaced.
    - The resulting time bounds are left-closed, right-open intervals: [t, t+1).
    """
    time = ds["time"]
    # Compute bounds (start is current time, end is next time)
    start = time.values[:-1]
    end = time.values[1:]

    # Drop the last bound to match shape
    time_bounds = xr.DataArray(np.stack([start, end], axis=1), dims=["time", "nv"], coords={"time": time[:-1]})
    ds = ds.isel({"time": slice(0, -1)})
    ds["time_bounds"] = time_bounds
    ds["time"].attrs["bounds"] = "time_bounds"
    return ds


def era5_reanalysis_from_rgi_id(
    rgi_id: str,
    rgi: gpd.GeoDataFrame | str | Path = "rgi/rgi.gpkg",
    years: list | Iterable = range(1980, 2025),
    dataset: str = "reanalysis-era5-single-levels-monthly-means",
    buffer_distance: float = 0.02,
) -> xr.Dataset:
    """
    Download and return ERA5-Land monthly reanalysis data for a glacier bounding box.

    This function extracts the bounding box of a glacier identified by its RGI ID,
    then queries and downloads monthly mean reanalysis data (2m temperature and total
    precipitation) from the ERA5-Land dataset using the Copernicus Climate Data Store (CDS) API.

    Parameters
    ----------
    rgi_id : str
        The RGI ID of the glacier (e.g., "RGI2000-v7.0-C-01-10853").

    rgi : geopandas.GeoDataFrame or str or Path, optional
        The RGI dataset as a GeoDataFrame or a file path to a GeoPackage (GPKG).
        If a string or Path is provided, it is read using `geopandas.read_file`.
        Default is "rgi/rgi.gpkg".

    years : list or Iterable of int, optional
        Sequence of years to request data for. Default is `range(1980, 2025)`.

    dataset : str, optional
        The ERA5 CDS dataset name to use for the query.
        Default is "reanalysis-era5-single-levels-monthly-means".

    buffer_distance : float, optional
        Buffer distance (in degrees) to expand the glacier bounding box before querying.
        Default is 0.02 degrees.

    Returns
    -------
    xarray.Dataset
        A dataset containing monthly mean ERA5-Land variables over the glacier bounding box:
        - `air_temp`: 2-meter air temperature [K]
        - `precipitation`: total precipitation [kg m^-2 month^-1]
        Includes a `time` coordinate and `time_bounds` following CF conventions.

    See Also
    --------
    cdsapi.Client : CDS API Python client for data access.
    geopandas.read_file : Reads geospatial vector files such as GeoPackages.
    xarray.open_dataset : Opens NetCDF files into xarray Datasets.

    Notes
    -----
    - Requires a valid CDS API key configured in `~/.cdsapirc`.
    - The bounding box is rounded to 0.1° precision for compatibility with CDS queries.
    - Time bounds are added using `cf_xarray` for CF-compliant time axes.
    """

    print("")
    print("Generate historical climate")
    print("-" * 40)

    if isinstance(rgi, str | Path):
        rgi = gpd.read_file(rgi)

    years = list(years)

    glacier = get_glacier_from_rgi_id(rgi, rgi_id)
    minx, miny, maxx, maxy = glacier.iloc[0]["geometry"].buffer(buffer_distance).bounds
    area = [np.ceil(maxy * 10) / 10, np.floor(minx * 10) / 10, np.floor(miny * 10) / 10, np.ceil(maxx * 10) / 10]

    print(f"Bounding box {area}")

    client = cdsapi.Client()

    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["2m_temperature", "total_precipitation"],
        "year": years,
        "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }
    f = client.retrieve(dataset, request).download()

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=False)

    if f.endswith("zip"):
        era_files = extract_archive(f)
        dss = []
        for era_file in era_files:
            ds = xr.open_mfdataset(era_file, decode_times=time_coder, decode_timedelta=True)
            ds["valid_time"] = ds["valid_time"].dt.floor("D")
            dss.append(ds)
        ds = xr.merge(dss)
    else:
        ds = xr.open_dataset(f, decode_times=time_coder, decode_timedelta=True)

    ds = ds.rename({"valid_time": "time"}).drop_vars(["number", "expver"])
    ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    ds.rio.write_crs("EPSG:4326", inplace=True)

    ds = ds.rename_vars({"tp": "precipitation", "t2m": "air_temp"})
    ds["precipitation"].attrs.update({"units": "kg m^-2 day^-1"})
    ds["precipitation"] *= 1000
    ds["time"].encoding["units"] = "hours since 1980-01-01 00:00:00"
    ds["time"].encoding["calendar"] = "standard"

    return add_time_bounds(ds)
