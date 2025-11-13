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
# pylint: disable=unused-import,too-many-positional-arguments,broad-exception-caught


"""
Prepare Climate.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cdsapi
import cf_xarray
import cftime
import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import s3fs
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

from pism_terra.dem import get_glacier_from_rgi_id
from pism_terra.download import (
    FileInfo,
    download_archive,
    download_file,
    download_netcdf,
    download_request,
    extract_archive,
    parse_filename,
    save_netcdf,
)
from pism_terra.raster import add_time_bounds
from pism_terra.workflow import check_xr, check_xr_sampled

xr.set_options(keep_attrs=True)


def _process_one_tif(p: Path, outdir: Path, force_overwrite: bool) -> Path | None:
    """
    Convert a single GeoTIFF to a time-stamped NetCDF, returning the output path.

    The input filename is parsed (via :func:`parse_filename`) to extract
    ``variable``, ``year``, and ``month``. The raster is opened with
    ``rioxarray``, a time dimension is added using a CF/Gregorian date
    (day = 1), and the result is written as NetCDF into ``outdir``.

    If a matching NetCDF already exists and opens successfully (checked with
    :func:`check_xr_sampled`) and ``force_overwrite`` is False, the file is
    reused and simply returned.

    Parameters
    ----------
    p : pathlib.Path
        Path to the source GeoTIFF.
    outdir : pathlib.Path
        Output directory for the generated NetCDF file.
    force_overwrite : bool
        If True, overwrite any existing output file. If False, reuse a
        valid existing file when possible.

    Returns
    -------
    pathlib.Path or None
        Absolute path to the produced (or reused) NetCDF, or ``None`` if
        an error occurred and the file was skipped.

    Notes
    -----
    - The function sets ``_FillValue`` to ``None`` on the DataArray encoding
      to avoid writing a fill value in the output variable.
    - The helper :func:`parse_filename` is expected to provide fields
      ``variable``, ``year``, ``month``, and ``units``.
    - Any exception is caught and results in ``None`` being returned (so the
      caller can skip that entry without failing the whole batch).

    Examples
    --------
    >>> out = _process_one_tif(Path("tas_1990_01.tif"), Path("out"), False)
    >>> out.name.endswith(".nc")
    True
    """
    fi = parse_filename(str(p))
    output_path = outdir / f"{fi.variable}_{fi.year}_{fi.month}.nc"

    try:
        if (not check_xr_sampled(output_path)) or force_overwrite:
            output_path.unlink(missing_ok=True)

            # Build a cftime "standard/gregorian" datetime (day = 1)
            t = cftime.DatetimeGregorian(int(fi.year), int(fi.month), 1)

            da = rxr.open_rasterio(p).squeeze(drop=True)  # drop 'band' if present
            da.name = fi.variable
            da.attrs.update({"units": fi.units})

            da = da.expand_dims(time=[t]).drop_vars("spatial_ref", errors="ignore")
            da.encoding["_FillValue"] = None  # optional: avoid _FillValue in output
            da.to_netcdf(output_path)

        return output_path if output_path.exists() else None
    except Exception:
        return None


def convert_many_tifs_concurrent(
    tifs: Iterable[Path],
    outdir: Path,
    force_overwrite: bool = False,
    max_workers: int | None = None,
) -> list[Path]:
    """
    Convert many GeoTIFFs to NetCDF in parallel using a process pool.

    Each input is handled by :func:`_process_one_tif`. Existing outputs are
    reused unless ``force_overwrite`` is True. Progress is shown with ``tqdm``
    if available.

    Parameters
    ----------
    tifs : iterable of pathlib.Path
        Collection of input GeoTIFF paths to process.
    outdir : pathlib.Path
        Directory where NetCDF outputs will be written. Created if missing.
    force_overwrite : bool, optional
        Overwrite existing outputs if True. Default is False.
    max_workers : int or None, optional
        Number of worker processes. If ``None`` (default), use ``cpu_count()-1``
        (but at least 1).

    Returns
    -------
    list of pathlib.Path
        Paths of NetCDF files that were successfully produced or reused.
        Only existing files are returned.

    Notes
    -----
    - Uses :class:`concurrent.futures.ProcessPoolExecutor`; make sure that any
      objects used inside :func:`_process_one_tif` are pickleable and safe for
      multiprocessing on your platform.
    - Errors for individual files are caught inside :func:`_process_one_tif`
      and reported as ``None`` results, so the batch continues.

    Examples
    --------
    >>> outs = convert_many_tifs_concurrent(
    ...     [Path("tas_1990_01.tif"), Path("tas_1990_02.tif")],
    ...     Path("out"),
    ...     force_overwrite=False,
    ... )
    >>> len(outs) >= 1
    True
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    max_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)

    rets: list[Path] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_process_one_tif, Path(p), outdir, force_overwrite) for p in tifs]
        try:

            for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing files (parallel)"):
                out = fut.result()
                if out is not None:
                    rets.append(out)
        except ImportError:
            for fut in as_completed(futs):
                out = fut.result()
                if out is not None:
                    rets.append(out)

    return [p for p in rets if p.exists()]


def create_offset_file(file_name: str | Path, delta_T: float = 0.0, frac_P: float = 1.0):
    """
    Generate offset file using xarray.

    Parameters
    ----------
    file_name : str
        The name of the file to create.
    delta_T : float, optional
        The temperature offset, by default 0.0.
    frac_P : float, optional
        The precipitation fraction, by default 1.0.
    """

    file_name = Path(file_name)
    dT = [delta_T]
    fP = [frac_P]
    time = [0]
    time_bounds = [[-1, 1]]

    ds = xr.Dataset(
        data_vars={
            "delta_T": (["time"], dT, {"units": "K"}),
            "frac_P": (["time"], fP, {"units": "1"}),
            "time_bounds": (["time", "bnds"], time_bounds, {}),
        },
        coords={
            "time": (
                "time",
                time,
                {
                    "units": "seconds since 01-01-01",
                    "axis": "T",
                    "calendar": "365_day",
                    "bounds": "time_bounds",
                },
            )
        },
    )
    encoding = {v: {"_FillValue": None} for v in ["delta_T", "frac_P"]}

    ds.to_netcdf(file_name, encoding=encoding)


def snap(
    rgi_id: str,
    path: Path | str = ".",
    **kwargs,
) -> list[Path]:
    """
    Build SNAP-derived monthly climate files and 30-year climatologies.

    Downloads SNAP IEM monthly precipitation and temperature archives,
    extracts and converts each GeoTIFF into NetCDF with a monthly time
    coordinate, merges variables (and DEM-derived surface altitude),
    and writes both the full historical stack and three 30-year weighted
    climatologies (1920–1949, 1950–1979, 1980–2009).

    Parameters
    ----------
    rgi_id : str
        Glacier identifier (currently unused in this workflow, kept for
        interface parity).
    path : str or pathlib.Path, default ``"."``
        Output directory. Intermediate and final NetCDFs are written here.
    **kwargs
        Forwarded to :func:`download_file` and :func:`extract_archive`
        (e.g., ``force_overwrite=True``).

    Returns
    -------
    list[pathlib.Path]
        Paths to the three 30-year climatology NetCDF files:
        ``snap_1920_1949.nc``, ``snap_1950_1979.nc``, ``snap_1980_2009.nc``.
        Additionally, the full stack ``snap_1900_2015.nc`` is written in
        ``path`` as a side effect.

    Raises
    ------
    requests.HTTPError
        If SNAP URLs fail to download.
    OSError
        On file I/O errors during extraction or NetCDF writing.
    ValueError
        If extracted files do not match the expected filename pattern.
    Exception
        Propagated errors from xarray/rioxarray/cftime routines.

    Notes
    -----
    - Variables are renamed to:
      ``precipitation`` (kg m^-2 year^-1), ``air_temp`` (celsius),
      and ``surface`` (m).
    - CRS is written as ``EPSG:3338``. Per-variable CF linkage
      (``grid_mapping="spatial_ref"``) is restored for the output climatologies.
    - The 30-year climatologies are month-wise weighted means using calendar
      month lengths within each window.
    - ``_FillValue`` is suppressed on core variables and coordinates at
      write-time via ``encoding={...: {'_FillValue': None}}``.

    Examples
    --------
    >>> out_paths = snap("RGI2000-v7.0-C-01-04374", path="snap_outputs", force_overwrite=True)
    >>> [p.name for p in out_paths]
    ['snap_1920_1949.nc', 'snap_1950_1979.nc', 'snap_1980_2009.nc']
    """

    print("")
    print("Generate SNAP climate")
    print("-" * 80)

    force_overwrite: bool = bool(kwargs.pop("force_overwrite", False))

    path = Path(path)
    dss = []

    for url in [
        "http://data.snap.uaf.edu/data/IEM/Inputs/historical/precipitation/pr_total_mm_iem_cru_TS40_1901_2015.zip",
        "http://data.snap.uaf.edu/data/IEM/Inputs/historical/temperature/tas_mean_C_iem_cru_TS40_1901_2015.zip",
    ]:
        f = Path(url).name
        f_path = path / Path(f)
        print(f"Processing {f_path.resolve()}")
        _ = download_file(url, f_path, force_overwrite=force_overwrite)
        response = extract_archive(f_path, force_overwrite=force_overwrite)
        response_clean = [Path(r) for r in sorted(response) if str(r).endswith("tif")]
        results = convert_many_tifs_concurrent(response_clean, path, force_overwrite=force_overwrite)

        if not results:
            raise RuntimeError("No NetCDF outputs were produced or found; cannot build SNAP dataset.")

        # Concatenate along time and sort (in case paths are unordered)
        out = xr.open_mfdataset(results, parallel=True, chunks="auto", engine="h5netcdf")
        dss.append(out)

    url = "http://data.snap.uaf.edu/data/IEM/Inputs/ancillary/elevation/iem_prism_dem_1km.tif"
    dem_path = Path(path) / Path("iem_prism_dem_1km.tif")
    _ = download_file(url, dem_path, force_overwrite=force_overwrite)
    da = rxr.open_rasterio(dem_path).squeeze(drop=True)  # drop 'band' if present
    da = da.where(da > 0, 0).fillna(0)
    da.name = "surface"
    da.attrs.update({"units": "m", "standard_name": "surface_altitude"})
    dem = da.interp_like(out)
    dss.append(dem)

    ds = xr.merge(dss).rio.write_crs("EPSG:3338").fillna(0)
    ds = ds.rename_vars({"pr": "precipitation", "tas": "air_temp"})
    ds["precipitation"] *= 12
    ds["precipitation"].attrs.update({"units": "kg m^-2 year^-1"})
    ds["air_temp"].attrs.update({"units": "celsius"})

    period_starts = [1920, 1950, 1980]
    ps: list[Path] = []
    delayed_writes = []

    for y in period_starts:
        start = str(y)
        end = str(y + 29)

        p = Path(path) / f"snap_{rgi_id}_{start}_{end}.nc"
        ps.append(p)

        if check_xr_sampled(p) and not force_overwrite:
            # Reuse existing file; no work scheduled
            continue

        # --- compute weighted monthly climatology for this period (all lazy) ---
        ds_sub = ds.sel(time=slice(start, end))
        month_length = ds_sub.time.dt.days_in_month
        weights = month_length.groupby("time.month") / month_length.groupby("time.month").sum()

        ds_weighted = (ds_sub * weights).groupby("time.month").sum(dim="time").rename({"month": "time"})

        # coordinate metadata on x/y
        for c, axis, stdname in (("x", "X", "projection_x_coordinate"), ("y", "Y", "projection_y_coordinate")):
            attrs = {
                "units": "m",
                "axis": axis,
                "standard_name": stdname,
                "long_name": f"{c}-coordinate in projected coordinate system",
            }
            if c in ds_weighted.coords:
                ds_weighted[c].attrs.update(attrs)

        units = "days since 0001-01-01"
        calendar = "365_day"

        base_year = 1
        start_date = [cftime.DatetimeNoLeap(base_year, m, 1) for m in range(1, 13)]

        ds_weighted = ds_weighted.assign_coords(time=("time", start_date))
        ds_weighted["time"].attrs.update(
            {
                "standard_name": "time",
                "long_name": "climatological time (mid-month)",
                "bounds": "time_bounds",
            }
        )
        # keep calendar only in encoding to avoid xarray overwrite error
        ds_weighted["time"].attrs.pop("calendar", None)
        ds_weighted.time.encoding.update({"units": units, "calendar": calendar})

        time_num = cftime.date2num(ds_weighted.time.values, units=units, calendar=calendar).astype("float64")

        ds_weighted = ds_weighted.assign_coords(time=("time", time_num))
        ds_weighted["time"].attrs.update({"units": units, "calendar": calendar})

        ds_weighted = ds_weighted.cf.add_bounds("time")

        # CRS + spatial dims + grid_mapping tags
        ds_weighted = ds_weighted.rio.set_spatial_dims(x_dim="x", y_dim="y")
        ds_weighted.rio.write_crs("EPSG:3338", inplace=True)
        for v in ("precipitation", "air_temp", "surface"):
            if v in ds_weighted:
                ds_weighted[v].attrs["grid_mapping"] = "spatial_ref"

        # encoding: remove _FillValue without nuking other encodings
        encoding = {
            v: {"_FillValue": None}
            for v in ("x", "y", "surface", "air_temp", "precipitation", "time", "time_bounds")
            if v in ds_weighted
        }

        # schedule a lazy NetCDF write for this period
        p.unlink(missing_ok=True)
        delayed_writes.append(ds_weighted.to_netcdf(p, encoding=encoding, compute=False))

    # Kick off all writes in parallel (plus internal dask parallelism)
    if delayed_writes:
        with ProgressBar():
            dask.compute(*delayed_writes)

    return ps


def era5(
    rgi_id: str,
    rgi: gpd.GeoDataFrame | str | Path = "rgi/rgi.gpkg",
    years: list[int] | Iterable[int] = range(1978, 2025),
    dataset: str = "reanalysis-era5-land-monthly-means",
    buffer_distance: float = 0.1,
    path: Path | str = ".",
    **kwargs,
) -> Path:
    """
    Download monthly ERA5 reanalysis over a glacier bounding box and write a NetCDF.

    Given a glacier ``rgi_id``, this function:
    (1) loads the glacier geometry (GeoDataFrame or GPKG),
    (2) builds a buffered lon/lat bounding box (WGS84),
    (3) requests ERA5 monthly means (2 m air temperature and total precipitation),
    (4) optionally fills missing values using a global product,
    (5) adds a representative geopotential (converted to meters),
    (6) writes a CF-compliant NetCDF under ``path`` and returns its absolute path.

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
        ``era5_wgs84_<rgi_id>.nc`` inside ``path`` if ``path`` is a directory;
        otherwise the provided filename is used.
    **kwargs
        Additional keyword arguments forwarded to :func:`download_request`
        (e.g., alternate ``variable`` sequences, custom authentication/session
        options, or client settings). These are passed unchanged to the CDS
        retrieval helper.

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
      - ``surface`` (m) derived from ERA5 ``z`` / 9.80665 (geopotential → meters).
    - ``time_bounds`` are added for CF-style climatological metadata.
    - If missing values are detected in the regional subset, the function
      patches them from the global reanalysis (same period).
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
    ds = download_request(dataset, area, years, file_path=era5_filename_1, **kwargs)

    era5_filename_2 = path / Path(f"era5_wgs84_{rgi_id}_tmp_2.nc")
    era5_files.append(era5_filename_2)
    ds_geo = (
        download_request(dataset, area, [2013], variable=["geopotential"], file_path=era5_filename_2, **kwargs)
        .squeeze()
        .drop_vars("time", errors="ignore")
    )
    ds_geo_ = ds_geo.rio.write_crs("EPSG:4326").rio.reproject_match(ds).rename({"x": "longitude", "y": "latitude"})

    lon_attrs = ds["longitude"].attrs
    lat_attrs = ds["latitude"].attrs

    if bool(ds.to_array().isnull().any().item()):
        print("Missing values detected, filling with global reanalysis")
        era5_filename_3 = path / Path(f"era5_wgs84_{rgi_id}_tmp_3.nc")
        era5_files.append(era5_filename_3)
        ds_global = download_request(
            "reanalysis-era5-single-levels-monthly-means", area, years, file_path=era5_filename_3, **kwargs
        )
        ds_global_ = (
            ds_global.rio.write_crs("EPSG:4326").rio.reproject_match(ds).rename({"x": "longitude", "y": "latitude"})
        )
        ds = xr.where(np.isnan(ds), ds_global_, ds)

    ds = xr.merge([ds, ds_geo_], combine_attrs="override")
    ds = ds.rename({"valid_time": "time"})

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
    for name in ("latitude", "longitude", "surface", "precipitation", "air_temp"):
        if name in ds:
            ds[name].encoding.update({"_FillValue": None})

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


def pmip4(
    rgi_id: str,
    rgi: gpd.GeoDataFrame | str | Path = "rgi/rgi.gpkg",
    buffer_distance: float = 2.0,
    path: Path | str = ".",
    **kwargs,
) -> list[Path]:
    """
    Build PMIP4 LGM monthly climatology over a glacier bbox and write one NetCDF per model.

    For the glacier identified by ``rgi_id``, this function:
    (1) reads the glacier geometry (GeoDataFrame or GPKG) and converts to WGS84,
    (2) constructs a buffered lon/lat bounding box (degrees),
    (3) pulls PMIP4/CMIP6 LGM monthly fields (``tas``, ``pr``) from the
        ``pangeo-cmip6`` S3 catalog,
    (4) subsets to the bounding box, merges variables, and selects the final 2,400 months,
    (5) computes a 12-month climatology (groupby month → mean),
    (6) writes one CF-style NetCDF per ``source_id`` into ``<path>``.

    Parameters
    ----------
    rgi_id : str
        RGI glacier identifier (e.g., ``"RGI2000-v7.0-C-01-10853"``).
    rgi : geopandas.GeoDataFrame or str or pathlib.Path, default ``"rgi/rgi.gpkg"``
        Either an in-memory RGI GeoDataFrame, or a path to a GeoPackage readable by
        :func:`geopandas.read_file`.
    buffer_distance : float, default ``2.0``
        Buffer (degrees) applied to the glacier footprint before subsetting.
    path : str or pathlib.Path, default ``"."``
        Output directory where files are written as
        ``{source_id}_rgi_id_{rgi_id}.nc``.
    **kwargs
        Reserved for future options (e.g., catalog filtering). Currently unused.

    Returns
    -------
    list of pathlib.Path
        Absolute paths to the written NetCDF files, one per CMIP6 ``source_id``.

    Raises
    ------
    FileNotFoundError
        If the RGI path is missing (when ``rgi`` is a path).
    ValueError
        If the glacier ID cannot be found or geometry/CRS is invalid.
    Exception
        Errors propagated from S3 access (``s3fs``), zarr decoding, or file writing.

    Notes
    -----
    - Variables are renamed to:
        - ``air_temp`` (from ``tas``)
        - ``precipitation`` (from ``pr``)
      and the 12-month climatology dimension is renamed to ``time``.
    - The output CRS is set to ``EPSG:4326`` via rioxarray.
    - A CF-compliant 12-month ``time`` coordinate is attached using
      ``cftime.DatetimeNoLeap`` for a nominal year and encoded with
      ``calendar="365_day"`` and ``units="days since 0001-01-01"``. Time bounds
      (``time_bounds``) are added.
    - Longitudinal bounds are wrapped to 0–360 using modulo arithmetic; the
      bbox edges are rounded to tenths of a degree.
    """

    print("")
    print("Generate PMIP4 LGM climate")
    print("-" * 80)

    force_overwrite: bool = bool(kwargs.pop("force_overwrite", False))

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

        if (not check_xr(p)) or force_overwrite:
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
