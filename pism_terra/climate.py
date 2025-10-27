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
# pylint: disable=unused-import,too-many-positional-arguments


"""
Prepare Climate.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import cdsapi
import cf_xarray
import cftime
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import s3fs
import xarray as xr

from pism_terra.dem import get_glacier_from_rgi_id
from pism_terra.download import download_netcdf, extract_archive
from pism_terra.raster import add_time_bounds
from pism_terra.workflow import check_xr, check_xr_sampled


def era5(
    rgi_id: str,
    rgi: gpd.GeoDataFrame | str | Path = "rgi/rgi.gpkg",
    years: list[int] | Iterable[int] = range(1980, 2025),
    dataset: str = "reanalysis-era5-single-levels-monthly-means",
    buffer_distance: float = 0.1,
    path: Path | str = "era5.nc",
) -> Path:
    """
    Download monthly ERA5 reanalysis over a glacier bbox and write a NetCDF.

    Given a glacier ``rgi_id``, this function:
    (1) loads the glacier geometry (GeoDataFrame or GPKG),
    (2) builds a buffered lon/lat bounding box (WGS84),
    (3) requests ERA5 monthly means (2 m air temperature and total precipitation),
    (4) optionally fills missing values from the global product,
    (5) adds a representative geopotential (converted to meters),
    (6) writes a CF-compliant NetCDF to ``path`` and returns its absolute path.

    Parameters
    ----------
    rgi_id : str
        Glacier identifier, e.g., ``"RGI2000-v7.0-C-01-10853"``.
    rgi : geopandas.GeoDataFrame or str or pathlib.Path, default ``"rgi/rgi.gpkg"``
        In-memory RGI table or a path to a GeoPackage readable by
        :func:`geopandas.read_file`.
    years : list of int or Iterable of int, default ``range(1980, 2025)``
        Years to request from ERA5.
    dataset : str, default ``"reanalysis-era5-single-levels-monthly-means"``
        CDS dataset name for monthly single-level means (ERA5). Adjust if you
        intend to query ERA5-Land or other products.
    buffer_distance : float, default ``0.1``
        Buffer (degrees) applied to the glacier footprint before subsetting.
    path : str or pathlib.Path, default ``"era5.nc"``
        Output directory or filename base. The function writes a file named
        ``era5_wgs84_<rgi_id>.nc`` inside ``path`` if ``path`` is a directory,
        otherwise adapts accordingly.

    Returns
    -------
    pathlib.Path
        Absolute path to the written NetCDF file.

    Raises
    ------
    FileNotFoundError
        If the provided RGI path is missing.
    ValueError
        If the glacier ID cannot be found or the geometry is invalid.
    Exception
        Any errors propagated from the CDS request, reprojection, or I/O.

    See Also
    --------
    download_request
        Helper that performs the CDS API query and returns an xarray object.
    geopandas.read_file
        Load the RGI vector layer from disk.
    xarray.Dataset.rio.write_crs
        Record CRS on xarray objects via rioxarray.

    Notes
    -----
    - Output variables:
      - ``air_temp`` (K) from ERA5 ``t2m``.
      - ``precipitation`` (kg m^-2 day^-1) from ERA5 ``tp`` (converted).
      - ``surface`` (m) derived from ERA5 ``z`` / 9.80665 (geopotential to meters).
    - ``time_bounds`` are added for CF-style climatological metadata.
    - If missing values are detected in the regional subset, the function
      optionally patches them from the global reanalysis (same period).
    """
    path = Path(path)

    print("")
    print("Generate historical climate")
    print("-" * 80)

    era5_filename = path / Path(f"era5_wgs84_{rgi_id}.nc")

    if isinstance(rgi, (str, Path)):
        rgi = gpd.read_file(rgi)

    years = list(years)

    glacier = get_glacier_from_rgi_id(rgi, rgi_id).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = glacier.iloc[0]["geometry"].buffer(buffer_distance).bounds
    area = [
        np.ceil(maxy * 10) / 10,
        np.floor(minx * 10) / 10,
        np.floor(miny * 10) / 10,
        np.ceil(maxx * 10) / 10,
    ]

    print(f"Bounding box {area}")

    era5_files = []
    era5_filename_1 = path / Path(f"era5_wgs84_{rgi_id}_tmp_1.nc")
    era5_files.append(era5_filename_1)
    ds = download_request(dataset, area, years, path=era5_filename_1)
    ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    ds.rio.write_crs("EPSG:4326", inplace=True)

    era5_filename_2 = path / Path(f"era5_wgs84_{rgi_id}_tmp_2.nc")
    era5_files.append(era5_filename_2)
    ds_geo = download_request(dataset, area, [2013], variable=["geopotential"], path=era5_filename_2).mean(
        dim="valid_time"
    )
    ds_geo_ = ds_geo.rio.write_crs("EPSG:4326").rio.reproject_match(ds).rename({"x": "longitude", "y": "latitude"})

    lon_attrs = ds["longitude"].attrs
    lat_attrs = ds["latitude"].attrs

    if ("GRIB_missingValue" or "missing_value" or "_FillValue") in (ds["tp"].attrs or ds["t2m"].attrs):
        print("Missing values detected, filling with global reanalysis")
        era5_filename_3 = path / Path(f"era5_wgs84_{rgi_id}_tmp_3.nc")
        era5_files.append(era5_filename_3)
        ds_global = download_request("reanalysis-era5-single-levels-monthly-means", area, years, path=era5_filename_3)
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
    ds = add_time_bounds(ds)
    ds.to_netcdf(era5_filename)

    return era5_filename


def jif_cosipy(url: str, download_path: Path | str, path: Path | str) -> None:
    """
    Download and prepare COSIPY.

    Parameters
    ----------
    url : str
        The URL to download.
    download_path : str, Path
        The path to the original file.
    path : str, Path
        The path to the processed file.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing COSIPY.
    """

    if Path(download_path).exists():
        ds = xr.open_dataset(Path(download_path))
        print(f"{download_path} exists, skipping download")
    else:
        ds = download_netcdf(url)
    ds = ds.rename(
        {
            "TS": "ice_surface_temp",
            "T2": "air_temp",
            "surfMB": "climatic_mass_balance",
            "HGT": "surface",
        }
    )
    # Fun, fun: rainfall is in "mm", snowfall is in "m".
    ds["precipitation"] = ds["SNOWFALL"] * 1000 + ds["RAIN"]
    ds = ds[
        [
            "precipitation",
            "climatic_mass_balance",
            "air_temp",
            "ice_surface_temp",
            "surface",
        ]
    ]
    ds["ice_surface_temp"] -= 273.15
    ds["air_temp"] -= 273.15
    ds["climatic_mass_balance"] *= 1000
    ds["climatic_mass_balance"].attrs.update({"units": "kg m^-2 day^-1"})
    ds["precipitation"].attrs.update({"units": "kg m^-2 day^-1"})
    ds["ice_surface_temp"].attrs.update({"units": "celsius"})
    ds["air_temp"].attrs.update({"units": "celsius"})
    ds["surface"].attrs.update({"standard_name": "surface_altitude"})
    ds = ds.fillna(0)
    ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    ds.rio.write_crs("EPSG:4326", inplace=True)
    ds = add_time_bounds(ds)
    ds.to_netcdf(path)


def download_request(
    dataset: str = "reanalysis-era5-single-levels-monthly-means",
    area: Sequence[float] = (90.0, -90.0, 45.0, 90.0),
    years: Iterable[int] = range(1980, 2025),
    variable: Sequence[str] = ("2m_temperature", "total_precipitation"),
    path: Path | str = "tmp.nc",
) -> xr.Dataset:
    """
    Download monthly ERA5 reanalysis from the CDS and return it as an xarray Dataset.

    Sends a request to the Copernicus Climate Data Store (CDS) API for monthly
    averages of one or more single-level variables over a given bounding box and
    range of years. If the CDS returns multiple NetCDFs (e.g., one per year),
    they are opened and merged. The merged dataset is cached to ``path`` and
    re-used on subsequent calls.

    Parameters
    ----------
    dataset : str, default ``"reanalysis-era5-single-levels-monthly-means"``
        CDS dataset identifier to retrieve. Use ERA5-Land or other collections
        by passing a different dataset name if desired.
    area : sequence of float, default ``(90, -90, 45, 90)``
        Geographic bounding box **[North, West, South, East]** in degrees
        (WGS84). Note the order follows CDS convention.
    years : iterable of int, default ``range(1980, 2025)``
        Years to request (e.g., ``range(1980, 2025)`` or ``[1990, 1991, 1992]``).
    variable : sequence of str, default ``("2m_temperature", "total_precipitation")``
        ERA5 variable names to download (e.g., ``"2m_temperature"``,
        ``"total_precipitation"``, ``"geopotential"``). Variable availability
        depends on the chosen ``dataset``.
    path : str or pathlib.Path, default ``"tmp.nc"``
        Cache file. If it exists and opens cleanly, it is re-used; otherwise,
        a fresh download is performed and written to this path.

    Returns
    -------
    xarray.Dataset
        Dataset containing the requested monthly means. Typical variables include:
        - ``t2m`` : 2 m air temperature [K]
        - ``tp`` : total precipitation [m]
        Coordinates include a monthly ``valid_time`` (renamed or floored below)
        depending on the product.

    Raises
    ------
    cdsapi.api.ClientError
        If the CDS request fails (authentication/parameters/quota).
    OSError
        If the downloaded file cannot be opened or written.
    ValueError
        If the returned files are incompatible for merging.
    Exception
        Any other error from I/O/decoding during dataset assembly.

    Notes
    -----
    - Requires a valid CDS API key in ``~/.cdsapirc``.
    - The request uses: ``product_type="monthly_averaged_reanalysis"``,
      all months (``"01"``–``"12"``), and time ``"00:00"``.
    - If CDS delivers a ZIP, contents are extracted prior to loading and then merged.
    - ``valid_time`` (if present) is floored to daily resolution.

    Examples
    --------
    >>> ds = download_request(
    ...     dataset="reanalysis-era5-single-levels-monthly-means",
    ...     area=(63.9, 214.1, 59.0, 219.7),  # [N, W, S, E]
    ...     years=range(1980, 1985),
    ...     variable=("2m_temperature", "total_precipitation"),
    ...     path="era5_subset.nc",
    ... )
    >>> list(ds.data_vars)
    ['t2m', 'tp']
    """
    path = Path(path)
    client = cdsapi.Client()

    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": list(variable),
        "year": list(years),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": list(area),  # [N, W, S, E]
    }

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=False)

    if not check_xr(path):
        f = client.retrieve(dataset, request).download()

        if f.endswith(".zip"):
            era_files = extract_archive(f)
            dss = []
            for era_file in era_files:
                ds_part = xr.open_dataset(era_file, decode_times=time_coder, decode_timedelta=True)
                if "valid_time" in ds_part.coords:
                    ds_part["valid_time"] = ds_part["valid_time"].dt.floor("D")
                dss.append(ds_part)
            ds = xr.merge(dss)
        else:
            ds = xr.open_dataset(f, decode_times=time_coder, decode_timedelta=True)

        ds.to_netcdf(path)
    else:
        ds = xr.open_dataset(path, decode_times=time_coder, decode_timedelta=True)

    return ds


def pmip4(
    rgi_id: str,
    rgi: gpd.GeoDataFrame | str | Path = "rgi/rgi.gpkg",
    buffer_distance: float = 2.0,
    path: Path | str = ".",
) -> list[Path]:
    """
    Build a PMIP4 LGM monthly climatology for a glacier bounding box and save it.

    For the glacier identified by an RGI ID, this function:
    (1) reads the glacier geometry (from a GeoDataFrame or GPKG),
    (2) constructs a buffered lon/lat bounding box,
    (3) pulls PMIP4/CMIP6 LGM data (tas, pr) from the pangeo-cmip6 S3 catalog,
    (4) subsets to the bounding box,
    (5) computes a monthly climatology (mean over the final 2,400 months),
    and (6) writes one NetCDF per model into ``<path>/pmip4``.

    Parameters
    ----------
    rgi_id : str
        RGI glacier identifier (e.g., ``"RGI2000-v7.0-C-01-10853"``).
    rgi : geopandas.GeoDataFrame or str or pathlib.Path, default ``"rgi/rgi.gpkg"``
        Either an in-memory RGI GeoDataFrame, or a path to a GeoPackage that
        can be read via :func:`geopandas.read_file`.
    buffer_distance : float, default ``2.0``
        Buffer applied to the glacier footprint (in degrees) before subsetting
        the PMIP4 fields.
    path : str or pathlib.Path, default ``"."``
        Base directory for outputs. Files are written to
        ``<path>/pmip4/{source_id}_rgi_id_{rgi_id}.nc``.

    Returns
    -------
    list
        Returns a list with str or Path-like objects.
    """

    print("")
    print("Generate PMIP4 LGM climate")
    print("-" * 80)

    if isinstance(rgi, str | Path):
        rgi = gpd.read_file(rgi)

    glacier = get_glacier_from_rgi_id(rgi, rgi_id).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = glacier.iloc[0]["geometry"].buffer(buffer_distance).bounds

    minx = (np.floor(minx * 10) / 10) % 360
    maxx = (np.ceil(maxx * 10) / 10) % 360
    miny = np.floor(miny * 10) / 10
    maxy = np.ceil(maxy * 10) / 10

    print(f"Bounding box {minx}, {maxx}, {miny}, {maxy}")

    fs = s3fs.S3FileSystem(anon=True)
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    cmip6_df = pd.read_csv("https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv")
    lgm_df = cmip6_df.query(
        "activity_id=='PMIP' & table_id=='Amon' & experiment_id=='lgm' & (variable_id=='tas' | variable_id=='pr')"
    )

    responses = []
    for source_id, df in lgm_df.groupby(by="source_id"):
        path = Path(path)
        p = path / f"{source_id}_rgi_id_{rgi_id}.nc"

        if not check_xr(p):
            dss = []
            for v in ["tas", "pr"]:
                zstore = df[df["variable_id"] == v].zstore.values[0]
                mapper = fs.get_mapper(zstore)

                # open using xarray
                ds = (
                    xr.open_zarr(mapper, consolidated=True, decode_times=time_coder, decode_timedelta=True).drop_vars(
                        ["height"], errors="ignore"
                    )
                ).sel({"lon": slice(minx, maxx), "lat": slice(miny, maxy)})

                dss.append(ds)
            ds = (
                xr.merge(dss)
                .isel({"time": slice(-2401, -1)})
                .groupby("time.month")
                .mean()
                .rename_dims({"month": "time"})
                .rename_vars({"pr": "precipitation", "tas": "air_temp", "month": "time"})
            )
            ds.rio.write_crs("EPSG:4326", inplace=True)
            # Build a CF-compliant time axis: 12 mid-month datetimes in a no-leap year (e.g., 2001)

            base_year = 1
            start = [cftime.DatetimeNoLeap(base_year, m, 1) for m in range(1, 13)]
            # Assign coordinates and bounds

            ds = ds.assign_coords(time=("time", start))
            ds["time"].attrs.update(
                {
                    "standard_name": "time",
                    "long_name": "climatological time (mid-month)",
                    "bounds": "time_bounds",
                }
            )
            # CRS is fine to keep
            ds = ds.rio.write_crs("EPSG:4326", inplace=True)
            ds["time"].attrs.pop("calendar", None)

            # Put calendar/units ONLY in encoding (CF-compliant, and prevents the error)
            enc = {"units": "days since 0001-01-01", "calendar": "365_day"}
            ds.time.encoding.update(enc)
            ds = add_time_bounds(ds)

            ds.to_netcdf(p)

        responses.append(p)
    return responses
