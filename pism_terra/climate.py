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
import xarray as xr

from pism_terra.dem import get_glacier_by_rgi_id


def era5_reanalysis_from_rgi_id(
    rgi_id: str,
    rgi: gpd.GeoDataFrame | str | Path = "rgi/rgi.gpkg",
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

    Returns
    -------
    xarray.Dataset
        A dataset containing monthly mean ERA5-Land variables over the glacier bounding box,
        including:
        - `t2m`: 2-meter air temperature [K]
        - `tp`: total precipitation [m]
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

    if isinstance(rgi, str | Path):
        rgi = gpd.read_file(rgi)

    glacier = get_glacier_by_rgi_id(rgi, rgi_id)
    minx, miny, maxx, maxy = glacier.iloc[0].geometry.bounds

    client = cdsapi.Client()

    dataset = "reanalysis-era5-land-monthly-means"
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
    ds = xr.open_dataset(f).rename({"valid_time": "time"})
    ds = ds.cf.add_bounds("time")
    return ds
