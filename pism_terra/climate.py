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

from pathlib import Path

import cdsapi
import cf_xarray
import geopandas as gpd
import numpy as np
import rioxarray
import xarray as xr

from pism_terra.dem import get_glacier_from_rgi_id


def era5_reanalysis_from_rgi_id(
    rgi_id: str,
    rgi: gpd.GeoDataFrame | str | Path = "rgi/rgi.gpkg",
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
        The RGI dataset as a GeoDataFrame or a file path to the RGI GeoPackage.
        If a string or Path is provided, the file is read using `geopandas.read_file`.
        Default is "rgi/rgi.gpkg".
    dataset : str
        A valid product name.
    buffer_distance : float, optional
        The buffer_distance distance around the geometry, by default 0.1 degrees.

    Returns
    -------
    xarray.Dataset
        A dataset containing monthly mean ERA5-Land variables over the glacier bounding box,
        including:
        - `air_temp`: 2-meter air temperature [K]
        - `precipitation`: total precipitation [kg m^-2 month^-1]
        The dataset includes a `time` dimension and time bounds.

    See Also
    --------
    cdsapi.Client : The CDS API Python client for data download.
    geopandas.read_file : Reads vector geospatial files such as GeoPackages or shapefiles.
    xarray.open_dataset : Opens NetCDF datasets into xarray objects.

    Notes
    -----
    - Data is downloaded from the Copernicus CDS and may require a valid CDS API key
      configured in `~/.cdsapirc`.
    - The spatial extent of the request is rounded to the nearest 0.1° to match CDS constraints.
    - The returned dataset uses CF conventions and has time bounds added via `cf_xarray`.
    """

    print("")
    print("Generate historical climate")
    print("-" * 20)

    if isinstance(rgi, str | Path):
        rgi = gpd.read_file(rgi)

    glacier = get_glacier_from_rgi_id(rgi, rgi_id)
    minx, miny, maxx, maxy = glacier.iloc[0]["geometry"].buffer(buffer_distance).bounds

    client = cdsapi.Client()

    request = {
        "variable": ["2m_temperature", "total_precipitation"],
        "year": [
            "1980",
            "1981",
            "1982",
            "1983",
            "1984",
            "1985",
            "1986",
            "1987",
            "1988",
            "1989",
            "1990",
            "1991",
            "1992",
            "1993",
            "1994",
            "1995",
            "1996",
            "1997",
            "1998",
            "1999",
            "2000",
            "2001",
            "2002",
            "2003",
            "2004",
            "2005",
            "2006",
            "2007",
            "2008",
            "2009",
            "2010",
            "2011",
            "2012",
            "2013",
            "2014",
            "2015",
            "2016",
            "2017",
            "2018",
            "2019",
            "2020",
            "2021",
            "2022",
            "2023",
            "2024",
            "2025",
        ],
        "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [np.ceil(maxy * 10) / 10, np.floor(minx * 10) / 10, np.floor(miny * 10) / 10, np.ceil(maxx * 10) / 10],
    }

    client = cdsapi.Client()
    f = client.retrieve(dataset, request).download()
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=False)

    ds = (
        xr.open_dataset(f, decode_times=time_coder, decode_timedelta=True)
        .rename({"valid_time": "time"})
        .drop_vars(["number", "expver"])
    )
    ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    ds.rio.write_crs("EPSG:4326", inplace=True)

    ds = ds.rename_vars({"tp": "precipitation", "t2m": "air_temp"})
    # .fillna(0)
    # ds["air_temp"] = xr.where(ds["air_temp"] > 0, ds["air_temp"], 273.0, keep_attrs=True)
    ds["precipitation"].attrs.update({"units": "kg m^-2 month^-1"})
    ds["precipitation"] *= 1000
    ds = ds.cf.add_bounds("time")
    ds["time"].encoding["units"] = "hours since 1980-01-01 00:00:00"
    ds["time"].encoding["calendar"] = "standard"
    return ds
