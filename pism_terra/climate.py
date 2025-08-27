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
from pism_terra.download import download_netcdf, extract_archive
from pism_terra.raster import add_time_bounds


def era5_reanalysis_from_rgi_id(
    rgi_id: str,
    rgi: gpd.GeoDataFrame | str | Path = "rgi/rgi.gpkg",
    years: list | Iterable = range(1980, 2025),
    dataset: str = "reanalysis-era5-single-levels-monthly-means",
    buffer_distance: float = 0.1,
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
    print("-" * 80)

    if isinstance(rgi, str | Path):
        rgi = gpd.read_file(rgi)

    years = list(years)

    glacier = get_glacier_from_rgi_id(rgi, rgi_id)
    minx, miny, maxx, maxy = glacier.iloc[0]["geometry"].buffer(buffer_distance).bounds
    area = [np.ceil(maxy * 10) / 10, np.floor(minx * 10) / 10, np.floor(miny * 10) / 10, np.ceil(maxx * 10) / 10]

    print(f"Bounding box {area}")

    ds = download_request(dataset, area, years)
    ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    ds.rio.write_crs("EPSG:4326", inplace=True)
    ds_geo = download_request(dataset, area, [2013], variable=["geopotential"]).mean(dim="time")
    ds_geo_ = ds_geo.rio.write_crs("EPSG:4326").rio.reproject_match(ds).rename({"x": "longitude", "y": "latitude"})

    lon_attrs = ds["longitude"].attrs
    lat_attrs = ds["latitude"].attrs

    if ("GRIB_missingValue" or "missing_value" or "_FillValue") in (ds["tp"].attrs or ds["t2m"].attrs):
        print("Missing values detected, filling with global reanalysis")
        ds_global = download_request("reanalysis-era5-single-levels-monthly-means", area, years)
        ds_global_ = (
            ds_global.rio.write_crs("EPSG:4326").rio.reproject_match(ds).rename({"x": "longitude", "y": "latitude"})
        )
        ds = xr.where(np.isnan(ds), ds_global_, ds)

    ds = xr.merge([ds, ds_geo_])
    ds = ds.rename({"valid_time": "time"}).drop_vars(["number", "expver"])

    ds = ds.rename_vars({"tp": "precipitation", "t2m": "air_temp", "z": "surface"})
    ds["surface"] /= 9.80665
    ds["surface"].attrs.update({"units": "m", "standard_name": "surface_altitude"})
    ds["precipitation"] *= 1000
    ds["precipitation"].attrs.update({"units": "kg m^-2 day^-1"})
    ds["air_temp"].attrs.update({"units": "kelvin"})
    ds["time"].encoding["units"] = "hours since 1980-01-01 00:00:00"
    ds["time"].encoding["calendar"] = "standard"
    ds["longitude"].attrs = lon_attrs
    ds["latitude"].attrs = lat_attrs
    ds.rio.write_crs("EPSG:4326", inplace=True)

    return add_time_bounds(ds)


def jif_cosipy(url: str) -> xr.Dataset:
    """
    Download and prepare COSIPY.

    Parameters
    ----------
    url : str, optional
        The URL to download.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing COSIPY.
    """

    ds = download_netcdf(url)
    ds = ds.rename({"TS": "ice_surface_temp", "T2": "air_temp", "surfMB": "climatic_mass_balance"})
    ds["precipitation"] = ds["SNOWFALL"] + ds["RAIN"]
    ds = ds[["precipitation", "climatic_mass_balance", "air_temp", "ice_surface_temp"]]
    ds["ice_surface_temp"] -= 273.15
    ds["air_temp"] -= 273.15
    ds["climatic_mass_balance"] *= 1000
    ds["precipitation"] *= 1000
    ds["climatic_mass_balance"].attrs.update({"units": "kg m^-2 day^-1"})
    ds["precipitation"].attrs.update({"units": "kg m^-2 day^-1"})
    ds["ice_surface_temp"].attrs.update({"units": "celsius"})
    ds["air_temp"].attrs.update({"units": "celsius"})
    ds = ds.fillna(0)
    ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    ds.rio.write_crs("EPSG:4326", inplace=True)

    return ds


def download_request(
    dataset: str = "reanalysis-era5-single-levels-monthly-means",
    area: list[float] = [90, -90, 45, 90],
    years: list | Iterable = range(1980, 2025),
    variable: list = ["2m_temperature", "total_precipitation"],
) -> xr.Dataset:
    """
    Download ERA5 monthly reanalysis data from the Copernicus Climate Data Store (CDS).

    This function sends a request to the CDS API to retrieve monthly mean 2m temperature
    and total precipitation data for a specified spatial domain and time range.
    The downloaded data are returned as an xarray Dataset.

    Parameters
    ----------
    dataset : str, optional
        The CDS dataset identifier. Defaults to "reanalysis-era5-single-levels-monthly-means".
    area : list of float, optional
        Bounding box [North, West, South, East] in degrees. Defaults to [90, -90, 45, 90].
    years : list or iterable, optional
        List or range of years to download. Defaults to range(1980, 2025).
    variable : list, optional
        List of variables to download.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the merged ERA5 monthly data. The variables include:
        - `t2m` : 2-meter air temperature [K]
        - `tp` : total precipitation [m]
        If multiple NetCDF files are returned, they are merged into a single dataset.

    Notes
    -----
    - Requires a valid CDS API key in `~/.cdsapirc`.
    - Uses the `cdsapi` client to perform the download.
    - If the data is delivered as a ZIP archive, the contents are extracted before loading.
    - The `valid_time` field is floored to daily resolution.
    """
    client = cdsapi.Client()

    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": variable,
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
            ds = xr.open_dataset(era_file, decode_times=time_coder, decode_timedelta=True)
            if "valid_time" in ds.coords:
                ds["valid_time"] = ds["valid_time"].dt.floor("D")
            dss.append(ds)
        ds = xr.merge(dss)
    else:
        ds = xr.open_dataset(f, decode_times=time_coder, decode_timedelta=True)

    return ds
