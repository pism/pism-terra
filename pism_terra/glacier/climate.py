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

# mypy: disable-error-code="call-overload"
# pylint: disable=unused-import,too-many-positional-arguments,broad-exception-caught


"""
Prepare Climate.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import as_completed as cf_as_completed
from pathlib import Path

import cf_xarray
import cftime
import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import s3fs
import xarray as xr
from cdo import Cdo
from dask.diagnostics import ProgressBar
from pyproj import Transformer
from rasterio.enums import Resampling
from tqdm.auto import tqdm

from pism_terra.aws import s3_to_local
from pism_terra.domain import create_domain, get_bounds_from_geometry
from pism_terra.download import (
    FileInfo,
    carra_download_request,
    download_archive,
    download_file,
    download_netcdf,
    download_request,
    extract_archive,
    parse_filename,
    save_netcdf,
)
from pism_terra.grids import load_grid
from pism_terra.raster import add_time_bounds
from pism_terra.vector import get_glacier_from_rgi_id
from pism_terra.workflow import check_xr_fully, check_xr_lazy

logger = logging.getLogger(__name__)

xr.set_options(keep_attrs=True)

carra2_grid = load_grid("carra2")

# CARRA2 is on a polar-stereographic projection on a 6371229 m sphere.
# Mirrors the parameters in pism_terra/grids/carra2.txt.
CARRA2_PROJ = (
    "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=-30 "
    "+x_0=172840.374543307 +y_0=645049.059394855 "
    "+R=6371229 +units=m +no_defs"
)


def _process_one_tif(p: Path, outdir: Path, force_overwrite: bool) -> Path | None:
    """
    Convert a single GeoTIFF to a time-stamped NetCDF, returning the output path.

    The input filename is parsed (via :func:`parse_filename`) to extract
    ``variable``, ``year``, and ``month``. The raster is opened with
    ``rioxarray``, a time dimension is added using a CF/Gregorian date
    (day = 1), and the result is written as NetCDF into ``outdir``.

    If a matching NetCDF already exists and opens successfully (checked with
    :func:`check_xr_lazy`) and ``force_overwrite`` is False, the file is
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
    snapdir = outdir / Path("snap")
    snapdir.mkdir(parents=True, exist_ok=True)

    output_path = snapdir / f"{fi.variable}_{fi.year}_{fi.month}.nc"

    try:
        if (not check_xr_lazy(output_path)) or force_overwrite:
            output_path.unlink(missing_ok=True)

            # Build a cftime "standard/gregorian" datetime (day = 1)
            t = cftime.DatetimeGregorian(int(fi.year), int(fi.month), 1)

            da = rxr.open_rasterio(p).squeeze(drop=True)  # drop 'band' if present
            da.name = fi.variable
            da.attrs.update({"units": fi.units})

            da = da.expand_dims(time=[t]).drop_vars("spatial_ref", errors="ignore")
            da.encoding["_FillValue"] = None  # optional: avoid _FillValue in output
            da.encoding.update({"zlib": True, "complevel": 2})

            da.to_netcdf(output_path)

        return output_path if output_path.exists() else None
    except Exception:
        return None


def prepare_carra2(
    path: str | Path,
    year: list[str | int] = [
        "1986",
        "1987",
        "1988",
        "1991",
        "1992",
        "1993",
        "1996",
        "1997",
        "1998",
        "2001",
        "2002",
        "2003",
        "2006",
        "2007",
        "2008",
        "2011",
        "2012",
        "2013",
        "2016",
        "2017",
        "2018",
        "2021",
        "2022",
        "2023",
    ],
    max_workers: int = 8,
    force_overwrite: bool = False,
    **kwargs,
) -> Path:
    """
    Download monthly CARRA2 reanalysis and write a NetCDF.

    Parameters
    ----------
    path : str or pathlib.Path
        Working/output directory. The final NetCDF and intermediate
        ``carra2/`` cache subfolder are written under this path.
    year : list[str | int]
        List of years to download.
    max_workers : int, default 8
        Maximum number of concurrent CDS download requests.
    force_overwrite : bool, default False
        If ``True``, recompute intermediate and output files even if they exist.
    **kwargs
        Additional keyword arguments forwarded to :func:`download_request`
        (e.g., alternate ``variable`` sequences, custom authentication/session
        options, or client settings). These are passed unchanged to the CDS
        retrieval helper.

    Returns
    -------
    pathlib.Path
        Absolute path to the written NetCDF file.

    Notes
    -----
    - Output variables:
      - ``air_temp`` (K) from CARRA ``t2m``.
      - ``precipitation`` (kg m^-2 day^-1) from CARRA ``tp`` (converted).
    - ``time_bounds`` are added for CF-style climatological metadata.
    - If missing values are detected in the regional subset, the function
      patches them from the global reanalysis (same period).
    """

    print("")
    print("Generate historical climate")
    print("-" * 120)

    path = Path(path)

    carra2_filename = path / "carra2.zarr"

    carra2_grid_path = path / "carra2_grid.txt"
    carra2_grid_path.write_text(carra2_grid)

    orography_dataset = "reanalysis-pan-carra"
    orography_request = {
        "level_type": "single_levels",
        "variable": ["orography"],
        "product_type": "analysis",
        "time": ["00:00"],
        "year": [
            "1986",
        ],
        "month": [
            "01",
        ],
        "day": [
            "01",
        ],
        "data_format": "grib",
    }
    orography_files = carra_download_request(
        orography_dataset,
        orography_request,
        file_path=path / Path("orography.grib"),
        max_workers=max_workers,
        **kwargs,  # pass the full CARRA request dict
    )

    precipitation_dataset = "reanalysis-pan-carra-means"
    precipitation_request = {
        "time_aggregation": "monthly",
        "level_type": "single_levels",
        "variable": ["total_precipitation"],
        "product_type": "forecast_based",
        "year": year,
        "month": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ],
        "data_format": "netcdf",
        "area": [90, -180, 40, 180],
    }

    precipitation_files = carra_download_request(
        precipitation_dataset,
        precipitation_request,
        file_path=path / Path("pr.nc"),
        max_workers=max_workers,
        **kwargs,  # pass the full CARRA request dict
    )

    temperature_dataset = "reanalysis-pan-carra-means"
    temperature_request = {
        "time_aggregation": "daily",
        "level_type": "single_levels",
        "variable": ["2m_temperature"],
        "product_type": "analysis_based",
        "year": year,
        "month": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ],
        "day": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
        ],
        "data_format": "netcdf",
        "area": [90, -180, 40, 180],
    }

    temperature_files = carra_download_request(
        temperature_dataset,
        temperature_request,
        file_path=path / Path("tas.nc"),
        max_workers=max_workers,
        **kwargs,  # pass the full CARRA request dict
    )

    logger.info(
        "Downloaded %d precipitation files, %d temperature files",
        len(precipitation_files),
        len(temperature_files),
    )

    grid = str(carra2_grid_path.resolve())
    cdo = Cdo()
    cdo.debug = True

    # --- Step 1: per-year batches (setgrid, settaxis, monmean/monstd) ---

    pr_sorted = sorted(precipitation_files)
    tas_sorted = sorted(temperature_files)

    batches = []
    for yr, pr_f, tas_f in zip(year, pr_sorted, tas_sorted):
        batch_out = str((path / f"batch_{yr}.nc").resolve())
        batches.append((yr, str(pr_f), str(tas_f), batch_out))

    def _process_carra2_batch(args):
        """
        Process a single year-batch: fix grid/time, compute monmean/monstd, merge.

        Parameters
        ----------
        args : tuple
            A ``(yr, pr_f, tas_f, batch_out)`` tuple with the year string,
            precipitation file, temperature file, and output path.

        Returns
        -------
        str
            Path to the merged output file.
        """
        yr, pr_f, tas_f, batch_out = args
        tmp = path
        cdo_local = Cdo(tempdir=tmp)

        # Precipitation: monthly means (already monthly data, just fix grid + time)
        pr_fixed = os.path.join(tmp, f"pr_{yr}.nc")
        cdo_local.setgrid(
            grid,
            input=f"""-setattribute,precipitation@units="kg m^-2 day^-1" -chname,tp,precipitation """
            f"""-settbounds,1mon -setreftime,{yr}-01-01 -settunits,days -settaxis,{yr}-01-15,00:00:00,1mon {pr_f}""",
            output=pr_fixed,
            options="--reduce_dim -f nc4 -z zip_2",
        )

        # Temperature: aggregate daily -> monthly mean
        tas_mm = os.path.join(tmp, f"tas_mm_{yr}.nc")
        cdo_local.monmean(
            input=f"""-setgrid,{grid} -chname,t2m,air_temp """
            f"""-settbounds,1day -setreftime,{yr}-01-01 -settunits,days -settaxis,{yr}-01-01,00:00:00,1day {tas_f}""",
            output=tas_mm,
            options="--reduce_dim -f nc4 -z zip_2",
        )

        # Temperature: aggregate daily -> monthly std
        tas_mstd = os.path.join(tmp, f"tas_mstd_{yr}.nc")
        cdo_local.setattribute(
            """air_temp_sd@long_name="standard deviation of 2-m air temperature" """,
            input=f"""-chname,air_temp,air_temp_sd -monstd -setgrid,{grid} -chname,t2m,air_temp """
            f"""-settbounds,1day -setreftime,{yr}-01-01 -settunits,days -settaxis,{yr}-01-01,00:00:00,1day {tas_f}""",
            output=tas_mstd,
            options="--reduce_dim -f nc4 -z zip_2",
        )

        # Merge pr + tas_mm + tas_mstd for this year
        cdo_local.merge(
            input=f"{pr_fixed} {tas_mm} {tas_mstd}",
            output=batch_out,
            options="-f nc4 -z zip_2",
        )
        # Clean up per-year intermediate files only (not the shared directory)
        for f in (pr_fixed, tas_mm, tas_mstd):
            Path(f).unlink(missing_ok=True)
        return batch_out

    # Only process batches that don't already exist (unless force_overwrite)
    batches_to_run = [b for b in batches if (not check_xr_lazy(b[3])) or force_overwrite]
    if batches_to_run:
        logger.info(
            "CDO: processing %d year batches (setgrid + monmean/monstd)...",
            len(batches_to_run),
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_carra2_batch, b): b for b in batches_to_run}
            for future in tqdm(
                cf_as_completed(futures),
                total=len(futures),
                desc="Processing CARRA2 batches",
            ):
                future.result()
    else:
        logger.info("CDO: all %d year batches already exist, skipping.", len(batches))

    batch_files = sorted(b[3] for b in batches)

    # --- Step 2: mergetime all year batches  ---
    if (not check_xr_lazy(carra2_filename)) or force_overwrite:
        logger.info("CDO: merging %d year batches...", len(batch_files))
        ds = cdo.mergetime(
            input=" ".join(batch_files),
            options=f"-f nc4 -z zip_2 -P {max_workers}",
            returnXDataset=True,
        )

        # Attach CARRA's static orography (single-step single-level field).
        # ``setgrid`` re-labels the native CARRA grid using the same descriptor
        # the per-year batches use, so the orography lines up on (y, x).
        orog_src = str(Path(orography_files[0]).resolve())
        orog_nc = str((path / "orography.nc").resolve())
        cdo.setgrid(
            grid,
            input=orog_src,
            output=orog_nc,
            options="--reduce_dim -f nc4 -z zip_2",
        )
        orog_ds = xr.open_dataset(orog_nc)
        # Pick the actual 2-D field, not CDO's scalar grid-mapping variable.
        orog_candidates = [n for n, da in orog_ds.data_vars.items() if {"y", "x"}.issubset(da.dims)]
        if not orog_candidates:
            raise ValueError(f"No (y, x) data variable found in orography file {orog_nc}: {list(orog_ds.data_vars)}")
        orog = orog_ds[orog_candidates[0]]
        # Drop only the singleton non-spatial dims (time/step/level), keep y, x.
        for d in list(orog.dims):
            if d not in ("y", "x") and orog.sizes[d] == 1:
                orog = orog.squeeze(d, drop=True)
        orog.attrs.update(
            {
                "standard_name": "surface_altitude",
                "long_name": "surface altitude",
                "units": "m",
            }
        )
        ds["orography"] = orog

        ds = ds.chunk({"time": -1, "y": 256, "x": 256})  # -1 = single chunk along time
        ds = (
            ds.rio.write_crs(CARRA2_PROJ, inplace=True)
            .rio.write_grid_mapping("spatial_ref", inplace=True)
            .rio.write_coordinate_system(inplace=True)
        )

        ds.to_zarr(
            carra2_filename,
            mode="w",
            consolidated=True,
            encoding={
                "time": {"dtype": "int64", "units": "hours since 1850-01-01 00:00:00"},
                "time_bnds": {
                    "dtype": "int64",
                    "units": "hours since 1850-01-01 00:00:00",
                },
                "crs": {"dtype": "int32"},
            },
        )
    return carra2_filename


def prepare_carra2_for_group(
    carra2_zarr: Path | str,
    dst_crs: str,
    geometry,
    geometry_crs: str,
    output_file: Path | str,
    resolution: float = 2500.0,
    force_overwrite: bool = False,
) -> Path:
    """
    Pre-reproject the CARRA2 Zarr to a group's CRS and bbox; write NetCDF.

    Produces a per-group cache that ``carra2()`` (in :mod:`pism_terra.glacier.stage`)
    can download in a single GET instead of fetching the full pan-Arctic Zarr
    and reprojecting on the fly for every glacier. Output stays at CARRA2's
    native ~2.5 km resolution by default (much smaller than the typical
    100 m boot grid) so the per-glacier ``carra2()`` step is then a cheap
    bbox crop + light resample.

    Parameters
    ----------
    carra2_zarr : Path or str
        Local path or ``s3://`` URI of the full CARRA2 Zarr store.
    dst_crs : str
        Target CRS for the group (e.g. ``"EPSG:3338"`` for Alaska).
    geometry : shapely.geometry.base.BaseGeometry
        The group's polygon/multipolygon (typically the aggregated complex's
        ``geometry`` from ``rgi_c.gpkg``).
    geometry_crs : str
        CRS of ``geometry`` (e.g. ``"EPSG:4326"`` for an RGI-v7 entry).
    output_file : Path or str
        Path to write the NetCDF.
    resolution : float, default ``2500.0``
        Target grid spacing in ``dst_crs`` units (meters).
    force_overwrite : bool, default ``False``
        If True, regenerate even if the output already exists.

    Returns
    -------
    pathlib.Path
        Absolute path to the written NetCDF.
    """
    output_file = Path(output_file)
    if output_file.exists() and not force_overwrite and check_xr_lazy(output_file):
        return output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Build a target grid at `resolution` over the group's bounds.
    geom_projected = gpd.GeoSeries([geometry], crs=geometry_crs).to_crs(dst_crs)
    x_bnds, y_bnds = get_bounds_from_geometry(geom_projected, buffer_dist=5_000.0, dx=1_000.0)
    target_grid = create_domain(x_bnds, y_bnds, resolution=resolution, crs=dst_crs)

    # Open the source Zarr (local or s3); anon read works for our public store.
    storage_options = {"anon": True} if str(carra2_zarr).startswith("s3://") else None
    ds = xr.open_zarr(str(carra2_zarr), consolidated=True, storage_options=storage_options, chunks={})
    # Make sure spatial dims and CRS are attached.
    if "x" in ds.dims and "y" in ds.dims:
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    elif "lon" in ds.dims and "lat" in ds.dims:
        ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    if ds.rio.crs is None:
        crs_wkt = None
        for var in ds.data_vars:
            gm = ds[var].attrs.get("grid_mapping") or ds[var].encoding.get("grid_mapping")
            if gm and gm in ds.variables:
                crs_wkt = ds[gm].attrs.get("crs_wkt") or ds[gm].attrs.get("spatial_ref")
                if crs_wkt:
                    break
        if not crs_wkt:
            for name in ds.variables:
                attrs = ds[name].attrs
                crs_wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
                if crs_wkt:
                    break
        if not crs_wkt:
            raise ValueError(f"Could not recover a CRS from CARRA2 Zarr at {carra2_zarr}")
        ds = ds.rio.write_crs(crs_wkt)

    # Clip to group bounds in CARRA2 coords (cheap; just a .sel slice).
    t = Transformer.from_crs(dst_crs, ds.rio.crs, always_xy=True)
    src_minx, src_miny, src_maxx, src_maxy = t.transform_bounds(x_bnds[0], y_bnds[0], x_bnds[1], y_bnds[1])
    x_asc = bool(ds.x[-1] > ds.x[0])
    y_asc = bool(ds.y[-1] > ds.y[0])
    sub = ds.sel(
        x=slice(src_minx, src_maxx) if x_asc else slice(src_maxx, src_minx),
        y=slice(src_miny, src_maxy) if y_asc else slice(src_maxy, src_miny),
    )

    # Drop the grid-mapping placeholder and any broken non-spatial coords.
    grid_mapping_names: set[str] = set()
    for var in sub.data_vars:
        gm = sub[var].attrs.get("grid_mapping") or sub[var].encoding.get("grid_mapping")
        if gm:
            grid_mapping_names.add(gm)
    for c in list(sub.coords):
        if c in ("x", "y", "spatial_ref"):
            continue
        if c in grid_mapping_names:
            sub = sub.drop_vars(c, errors="ignore")
            continue
        try:
            sub = sub.assign_coords({c: sub[c].compute()})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"Coord {c!r} unreadable from Zarr ({exc}); dropping")
            sub = sub.drop_vars(c, errors="ignore")
    if "time" in sub.coords:
        bounds_name = sub["time"].attrs.get("bounds")
        if bounds_name and bounds_name not in sub.coords and bounds_name not in sub.data_vars:
            sub["time"].attrs.pop("bounds", None)

    # rio.reproject_match walks every data variable; time_bnds has dims
    # (time, bnds) and no x/y so it raises MissingSpatialDimensionError.
    sub = sub.drop_vars("time_bnds", errors="ignore")

    # Reproject onto the group's target grid. At 2.5 km × a regional bbox the
    # full dataset fits comfortably in memory, so a single shot is fine.
    out = sub.rio.reproject_match(target_grid, resampling=Resampling.bilinear).astype("float32")
    # inplace=True avoids deep-copying a multi-GiB dataset just to stamp metadata.
    out = (
        out.rio.write_crs(dst_crs, inplace=True)
        .rio.write_grid_mapping(inplace=True)
        .rio.write_coordinate_system(inplace=True)
    )
    out.attrs["Conventions"] = "CF-1.8"

    # Clear stale encoding inherited from the Zarr source so netCDF4 doesn't
    # see half-prescribed datetime encoding.
    for name in list(out.coords) + list(out.data_vars):
        for k in (
            "dtype",
            "_FillValue",
            "units",
            "calendar",
            "chunks",
            "preferred_chunks",
        ):
            out[name].encoding.pop(k, None)

    encoding = {name: {"zlib": True, "complevel": 2, "shuffle": True} for name in out.data_vars}
    output_file.unlink(missing_ok=True)
    out.to_netcdf(output_file, encoding=encoding, engine="h5netcdf")
    return output_file


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

            for fut in tqdm(
                cf_as_completed(futs),
                total=len(futs),
                desc="Processing files (parallel)",
            ):
                out = fut.result()
                if out is not None:
                    rets.append(out)
        except ImportError:
            for fut in cf_as_completed(futs):
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
    ds.to_netcdf(file_name, encoding=encoding, engine="h5netcdf")


def create_step_file(
    file_name: str | Path,
    t_a: float,
    t_b: float,
    delta_T_a: float = 0.0,
    delta_T_b: float = 0.0,
    frac_P_a: float = 1.0,
    frac_P_b: float = 1.0,
):
    """
    Generate a step-function offset file.

    Applies ``delta_T_a`` / ``frac_P_a`` from year 1 to ``t_a`` and
    ``delta_T_b`` / ``frac_P_b`` from ``t_a`` to ``t_b``.

    Parameters
    ----------
    file_name : str or Path
        The name of the file to create.
    t_a : float
        Year at which the step occurs (end of first interval).
    t_b : float
        Final year (end of second interval).
    delta_T_a : float, optional
        Temperature offset for the first interval, by default 0.0.
    delta_T_b : float, optional
        Temperature offset for the second interval, by default 0.0.
    frac_P_a : float, optional
        Precipitation fraction for the first interval, by default 1.0.
    frac_P_b : float, optional
        Precipitation fraction for the second interval, by default 1.0.
    """
    file_name = Path(file_name)

    seconds_per_year = 365 * 24 * 3600

    # Midpoints and bounds in seconds since 01-01-01
    t0 = 0.0
    t_a_sec = (t_a - 1) * seconds_per_year
    t_b_sec = (t_b - 1) * seconds_per_year

    mid_a = (t0 + t_a_sec) / 2.0
    mid_b = (t_a_sec + t_b_sec) / 2.0

    time = [mid_a, mid_b]
    time_bounds = [[t0, t_a_sec], [t_a_sec, t_b_sec]]

    ds = xr.Dataset(
        data_vars={
            "delta_T": (["time"], [delta_T_a, delta_T_b], {"units": "K"}),
            "frac_P": (["time"], [frac_P_a, frac_P_b], {"units": "1"}),
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
    ds.to_netcdf(file_name, encoding=encoding, engine="h5netcdf")


def snap(
    path: Path | str = ".",
    **kwargs,
) -> list[Path]:
    """
    Download process SNAP forcing from PISM Cloud.

    Parameters
    ----------
    path : str or pathlib.Path, default ``"."``
        Output directory. Intermediate and final NetCDFs are written here.
    **kwargs
        E.g. force_overwrite.

    Returns
    -------
    list[pathlib.Path]
        Paths to the three 30-year climatology NetCDF files:
        ``snap_cru_TS40_1920_1949.nc``, ``snap_cru_TS40_1950_1979.nc``,
        ``snap_cru_TS40_1980_2009.nc``.
    """

    force_overwrite: bool = bool(kwargs.pop("force_overwrite", False))

    bucket: str = "pism-cloud-data"
    out_dir = Path(path)

    for sn in (
        "snap_cru_TS40_1920_1949.nc",
        "snap_cru_TS40_1950_1979.nc",
        "snap_cru_TS40_1980_2009.nc",
    ):

        snap_file = out_dir / sn

        if (not check_xr_lazy(snap_file)) or force_overwrite:
            snap_file.unlink(missing_ok=True)
            s3_to_local(bucket, prefix="snap", dest=path)
    snap_files = list(Path(path).rglob("snap_*.nc"))
    return snap_files


def _carra2_fill_years_and_bounds(ds: xr.Dataset, years: Sequence[int]) -> xr.Dataset:
    """
    Expand CARRA2 monthly data over ``years`` and attach monthly ``time_bnds``.

    CARRA2 is only downloaded for a sparse set of source years (see
    :func:`prepare_carra2`). For any target year not in the source, the 12
    months of the *nearest* available source year are copied and re-stamped
    with the target year. Ties are broken toward the earlier year (e.g.
    2004 → 2003). Bounds for each monthly timestamp are written as
    ``[t, next-month-start)`` so PISM can interpret the data as monthly means.

    Parameters
    ----------
    ds : xarray.Dataset
        CARRA2 dataset with a monthly ``time`` coordinate (12 entries per
        available year).
    years : sequence of int
        Target years to materialize in the output.

    Returns
    -------
    xarray.Dataset
        Same variables as ``ds``, with ``time`` expanded over ``years`` and a
        new ``time_bnds`` variable. Any incoming ``time_bnds`` is rebuilt.
    """

    def _year(t):
        """
        Return the calendar year of ``t``.

        Parameters
        ----------
        t : object
            A scalar time value (cftime datetime, pandas Timestamp, or
            ``numpy.datetime64``).

        Returns
        -------
        int
            Calendar year extracted from ``t``.
        """
        return t.year if hasattr(t, "year") else pd.Timestamp(t).year

    def _replace_year(t, new_year):
        """
        Return ``t`` with its year set to ``new_year``.

        Parameters
        ----------
        t : object
            Scalar time value (cftime datetime, pandas Timestamp, or
            ``numpy.datetime64``).
        new_year : int
            Year to assign on the returned value.

        Returns
        -------
        object
            Same dtype family as ``t`` (cftime in, cftime out;
            ``numpy.datetime64`` in, ``numpy.datetime64`` out).
        """
        if hasattr(t, "replace") and not isinstance(t, np.datetime64):
            return t.replace(year=new_year)
        return np.datetime64(pd.Timestamp(t).replace(year=new_year))

    def _next_month(t):
        """
        Return the first instant of the calendar month following ``t``.

        Parameters
        ----------
        t : object
            Scalar time value (cftime datetime, pandas Timestamp, or
            ``numpy.datetime64``).

        Returns
        -------
        object
            ``t`` advanced to the first day of the next month, preserving the
            input dtype family.
        """
        if isinstance(t, np.datetime64) or not hasattr(t, "month"):
            ts = pd.Timestamp(t)
            if ts.month == 12:
                return np.datetime64(ts.replace(year=ts.year + 1, month=1))
            return np.datetime64(ts.replace(month=ts.month + 1))
        if t.month == 12:
            return t.replace(year=t.year + 1, month=1)
        return t.replace(month=t.month + 1)

    ds = ds.drop_vars("time_bnds", errors="ignore")
    src_times = ds["time"].values
    src_year_of = np.array([_year(t) for t in src_times])
    source_years = sorted(set(src_year_of.tolist()))

    pieces = []
    for ty in sorted({int(y) for y in years}):
        nearest = min((abs(sy - ty), sy) for sy in source_years)[1]
        sub = ds.isel(time=np.where(src_year_of == nearest)[0])
        new_times = np.array([_replace_year(t, ty) for t in sub["time"].values])
        pieces.append(sub.assign_coords(time=new_times))

    merged = xr.concat(pieces, dim="time")
    times = merged["time"].values
    bounds = np.stack([times, np.array([_next_month(t) for t in times])], axis=1)
    merged["time_bnds"] = xr.DataArray(bounds, dims=["time", "nv"], coords={"time": merged["time"]})
    merged["time"].attrs["bounds"] = "time_bnds"
    return merged


def carra2(
    target_grid: xr.Dataset,
    rgi_id: str,
    years: list[int] | Iterable[int] = range(1978, 2025),
    path: Path | str = ".",
    bucket: str = "pism-cloud-data",
    prefix: str = "",
    force_overwrite: bool = False,
) -> Path:
    """
    Subset and reproject CARRA2 reanalysis to a glacier's target grid.

    Opens the cloud-hosted CARRA2 Zarr store on S3, lazily clips it to the
    bounding box of ``target_grid`` (after transforming the box into CARRA2
    coordinates), reprojects the subset onto ``target_grid``, and writes a
    compressed NetCDF.

    Parameters
    ----------
    target_grid : xarray.Dataset
        Target grid dataset providing the destination CRS (via ``spatial_ref``)
        and extent. Used to derive the bounding box for the CARRA2 subset.
    rgi_id : str
        Glacier identifier, e.g., ``"RGI2000-v7.0-C-01-10853"``. Used in the
        output filename.
    years : list of int or Iterable of int, default ``range(1978, 2025)``
        Years to materialize in the output. CARRA2 only stores a sparse set
        of source years; any requested year not in the source is filled by
        copying the nearest available source year (ties go to the earlier
        year).
    path : str or pathlib.Path, default ``"."``
        Output directory. The function writes
        ``carra2_<rgi_id>.nc`` inside this directory.
    bucket : str, default ``"pism-cloud-data"``
        S3 bucket hosting the CARRA2 Zarr store.
    prefix : str, default ``""``
        Optional S3 key prefix; the full URI becomes
        ``s3://<bucket>/<prefix>/climate/carra2.zarr``.
    force_overwrite : bool, default ``False``
        If True, regenerate the output even if the cached NetCDF exists.

    Returns
    -------
    pathlib.Path
        Absolute path to the written NetCDF file.

    Raises
    ------
    FileNotFoundError
        If the Zarr store cannot be opened.
    ValueError
        If ``target_grid`` lacks a CRS or the bbox cannot be transformed.

    Notes
    -----
    - Output variables and their units are inherited from the source CARRA2
      Zarr store (typically ``air_temp`` in K and ``precipitation`` in
      kg m^-2 day^-1).
    - Compression: zlib level 2 + shuffle.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    print("")
    print("Generate historical climate")
    print("-" * 120)

    carra2_filename = path / Path(f"carra2_{rgi_id}.nc")
    if carra2_filename.exists() and not force_overwrite and check_xr_lazy(carra2_filename):
        print(f"Using cached {carra2_filename}")
        return carra2_filename

    # Bounding box of the target grid in its own (projected) CRS.
    bounds = (
        float(target_grid.x_bnds.values[0][0]),
        float(target_grid.y_bnds.values[0][0]),
        float(target_grid.x_bnds.values[-1][-1]),
        float(target_grid.y_bnds.values[-1][-1]),
    )
    dst_crs = target_grid.spatial_ref.attrs["crs_wkt"]

    # Fast path: prepare.py pre-reprojects CARRA2 once per S4F aggregate group
    # and uploads ``carra2_<rgi_id>.nc`` (CARRA2 ~2.5 km, already in the
    # group's CRS). If that file exists on S3, fetch it and let PISM handle
    # interpolation onto the model grid at runtime.
    pre_key = f"{prefix}/climate/carra2_{rgi_id}.nc".lstrip("/")
    pre_uri = f"s3://{bucket}/{pre_key}"
    fs = s3fs.S3FileSystem(anon=True)
    if fs.exists(pre_uri):
        print(f"Found precomputed {pre_uri}; downloading")
        tmp_pre = path / f"_carra2_pre_{rgi_id}.nc"
        fs.get(pre_uri, str(tmp_pre))
        with xr.open_dataset(tmp_pre) as pre:
            out = _carra2_fill_years_and_bounds(pre.load(), list(years))
        for v in out.data_vars:
            if np.issubdtype(out[v].dtype, np.floating):
                out[v] = out[v].fillna(0)
        for name in list(out.coords) + list(out.data_vars):
            for k in (
                "dtype",
                "_FillValue",
                "units",
                "calendar",
                "chunks",
                "preferred_chunks",
            ):
                out[name].encoding.pop(k, None)
        encoding: dict[str, dict[str, object]] = {
            name: {"zlib": True, "complevel": 2, "shuffle": True} for name in out.data_vars
        }
        encoding.update(
            {
                "time": {"dtype": "int64", "units": "hours since 1978-01-01 00:00:00"},
                "time_bnds": {
                    "dtype": "int64",
                    "units": "hours since 1978-01-01 00:00:00",
                },
            }
        )
        carra2_filename.unlink(missing_ok=True)
        out.to_netcdf(carra2_filename, encoding=encoding, engine="h5netcdf")
        tmp_pre.unlink(missing_ok=True)
        return carra2_filename

    uri = f"s3://{bucket}/{prefix}/climate/carra2.zarr".replace("//", "/").replace("s3:/", "s3://")
    print(f"Opening {uri}")
    ds = xr.open_zarr(
        uri,
        consolidated=True,
        storage_options={"anon": True},
        chunks={},
    )
    print(f"  opened zarr: vars={list(ds.data_vars)}, dims={dict(ds.sizes)}", flush=True)

    # Make sure rioxarray knows which dims are spatial and what the CRS is.
    # inplace=True avoids deep-copying the full lazy zarr just to stamp metadata.
    if "x" in ds.dims and "y" in ds.dims:
        ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    elif "lon" in ds.dims and "lat" in ds.dims:
        ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

    # Recover CRS from whichever grid-mapping variable the Zarr writer used.
    # CARRA2 stores it as a *data variable* (not a coord) named "crs"; other
    # writers use "spatial_ref", "polar_stereographic", "lambert_conformal_conic", etc.
    if ds.rio.crs is None:
        crs_wkt = None
        # First, follow any data variable's CF grid_mapping pointer.
        for var in ds.data_vars:
            gm = ds[var].attrs.get("grid_mapping") or ds[var].encoding.get("grid_mapping")
            if gm and gm in ds.variables:  # checks both data_vars and coords
                crs_wkt = ds[gm].attrs.get("crs_wkt") or ds[gm].attrs.get("spatial_ref")
                if crs_wkt:
                    break
        # Fall back to scanning every variable for a CRS-shaped attr.
        if not crs_wkt:
            for name in ds.variables:
                attrs = ds[name].attrs
                crs_wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
                if crs_wkt:
                    break
        if not crs_wkt:
            raise ValueError(
                f"Could not recover a CRS from the CARRA2 Zarr store at {uri}. "
                f"Variables present: {list(ds.variables)}"
            )
        # inplace=True avoids a deep copy of the full lazy zarr.
        ds.rio.write_crs(crs_wkt, inplace=True)
    print(f"  CRS resolved: {ds.rio.crs}", flush=True)

    # Transform the target bbox into CARRA2 coordinates and clip there.
    # Use .sel(x=slice, y=slice) instead of rio.clip_box because the latter
    # tries to apply spatial dims to every Dataset variable and trips over
    # non-spatial helpers like ``time_bnds``.
    t = Transformer.from_crs(dst_crs, ds.rio.crs, always_xy=True)
    minx, miny, maxx, maxy = t.transform_bounds(*bounds)
    x_ascending = bool(ds.x[-1] > ds.x[0])
    y_ascending = bool(ds.y[-1] > ds.y[0])
    sub = ds.sel(
        x=slice(minx, maxx) if x_ascending else slice(maxx, minx),
        y=slice(miny, maxy) if y_ascending else slice(maxy, miny),
    )
    print(f"  subset dims: {dict(sub.sizes)}", flush=True)

    # rioxarray.reproject_match eagerly reads every non-spatial coord (via
    # ``coord.values``). The published CARRA2 Zarr has at least one coord
    # (e.g. time_bnds) whose stored chunk bytes don't match its declared
    # dtype, so the lazy read aborts with a numpy-view error. Pre-load each
    # non-spatial coord; drop the ones that don't survive.
    #
    # The grid-mapping placeholder (typically named ``crs`` or
    # ``spatial_ref``) is dropped unconditionally — we already recovered the
    # CRS into a clean rioxarray ``spatial_ref`` above and don't need the
    # original 0-d variable's data.
    grid_mapping_names = set()
    for var in sub.data_vars:
        gm = sub[var].attrs.get("grid_mapping") or sub[var].encoding.get("grid_mapping")
        if gm:
            grid_mapping_names.add(gm)
    for c in list(sub.coords):
        if c in ("x", "y", "spatial_ref"):
            continue
        if c in grid_mapping_names:
            sub = sub.drop_vars(c, errors="ignore")
            continue
        try:
            sub = sub.assign_coords({c: sub[c].compute()})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"Coord {c!r} unreadable from Zarr ({exc}); dropping")
            sub = sub.drop_vars(c, errors="ignore")
    # If `time.bounds` pointed at a coord we just dropped, clear the attr too.
    if "time" in sub.coords:
        bounds_name = sub["time"].attrs.get("bounds")
        if bounds_name and bounds_name not in sub.coords and bounds_name not in sub.data_vars:
            sub["time"].attrs.pop("bounds", None)

    sub = sub.drop_vars("time_bnds", errors="ignore")

    # Reproject the subset onto the target grid. Each reprojected batch is
    # written to disk immediately so we don't hoard them all in RAM — for
    # large aggregates the target grid can be enormous (100 m × 100s of km)
    # and a single batch already costs many GB. We then assemble the per-
    # variable results lazily from disk and merge into `out`.
    #
    # Suppress any registered dask Callback (e.g. ProgressBar from an outer
    # scope) so the [#####] 100% lines don't interleave with the tqdm bar.
    import tempfile  # pylint: disable=import-outside-toplevel

    from dask.callbacks import Callback  # pylint: disable=import-outside-toplevel

    # Size the per-batch reprojected buffer to ~200 MB. Each batch holds
    # time_batch × target_y × target_x float32s in RAM after rasterio.warp,
    # so the cap matters most for high-res, wide-bbox runs like S4F aggregates.
    target_cells = int(target_grid.sizes["x"]) * int(target_grid.sizes["y"])
    time_batch = max(1, min(24, 200_000_000 // (target_cells * 4)))
    print(f"  target_grid: {dict(target_grid.sizes)}, time_batch={time_batch}", flush=True)
    out_vars: dict[str, xr.DataArray] = {}
    saved_callbacks = Callback.active.copy()
    Callback.active.clear()
    try:
        with tempfile.TemporaryDirectory(prefix="carra2_reproj_", dir=str(path)) as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            for var_name in sub.data_vars:
                da = sub[var_name]
                print(f"  processing {var_name} dims={dict(da.sizes)}", flush=True)
                if "time" not in da.dims or da.sizes["time"] <= time_batch:
                    out_vars[var_name] = da.rio.reproject_match(target_grid, resampling=Resampling.bilinear).astype(
                        "float32"
                    )
                    continue
                n = da.sizes["time"]
                var_dir = tmp_dir / var_name
                var_dir.mkdir(parents=True, exist_ok=True)
                batch_files: list[Path] = []
                pbar = tqdm(
                    range(0, n, time_batch),
                    desc=f"Reprojecting {var_name}",
                    unit="batch",
                    leave=False,
                )
                for batch_idx, i in enumerate(pbar):
                    chunk = da.isel(time=slice(i, i + time_batch)).compute()
                    reproj = chunk.rio.reproject_match(target_grid, resampling=Resampling.bilinear).astype("float32")
                    batch_path = var_dir / f"batch_{batch_idx:04d}.nc"
                    # Clear inherited encoding before writing each shard.
                    reproj.encoding = {}
                    reproj.to_netcdf(batch_path, engine="h5netcdf")
                    batch_files.append(batch_path)
                    # Drop the in-memory copy so the next batch starts clean.
                    del chunk, reproj
                pbar.close()
                # Lazy assemble: open each shard chunked, concat along time.
                shards = [xr.open_dataarray(p, chunks={}) for p in batch_files]
                out_vars[var_name] = xr.concat(shards, dim="time")
            out = xr.Dataset(out_vars, attrs=sub.attrs)

            # Stamp CRS metadata while everything is still lazy. inplace=True
            # avoids deep-copying a multi-GiB dataset just to attach attrs.
            out = (
                out.rio.write_crs(dst_crs, inplace=True)
                .rio.write_grid_mapping(inplace=True)
                .rio.write_coordinate_system(inplace=True)
            )
            out.attrs["Conventions"] = "CF-1.8"

            # Expand to all requested years (filling missing ones from the
            # nearest source year) and attach CF time_bnds for PISM.
            out = _carra2_fill_years_and_bounds(out, list(years))
            for v in out.data_vars:
                if np.issubdtype(out[v].dtype, np.floating):
                    out[v] = out[v].fillna(0)

            # Clear stale encoding inherited from the Zarr source so netCDF4
            # doesn't see half-prescribed datetime encoding (e.g. dtype=int64
            # with no units).
            for name in list(out.coords) + list(out.data_vars):
                for k in (
                    "dtype",
                    "_FillValue",
                    "units",
                    "calendar",
                    "chunks",
                    "preferred_chunks",
                ):
                    out[name].encoding.pop(k, None)

            # Compressed NetCDF (zlib level 2 + shuffle for floats).
            encoding_c: dict[str, dict[str, object]] = {
                name: {"zlib": True, "complevel": 2, "shuffle": True} for name in out.data_vars
            }
            encoding_c.update(
                {
                    "time": {"dtype": "int64", "units": "hours since 1978-01-01 00:00:00"},
                    "time_bnds": {
                        "dtype": "int64",
                        "units": "hours since 1978-01-01 00:00:00",
                    },
                }
            )

            # Stream from the per-batch shards straight into the final NetCDF
            # — never materializing the full reprojected dataset in RAM, which
            # for high-res grids over 50 years is easily 10+ GiB per variable.
            # Writing here (inside the temp-dir context) keeps the shards alive
            # for dask to read from.
            carra2_filename.unlink(missing_ok=True)
            out.to_netcdf(carra2_filename, encoding=encoding_c, engine="h5netcdf")
    finally:
        Callback.active.update(saved_callbacks)

    return carra2_filename


def era5(
    target_grid: xr.Dataset,
    rgi_id: str,
    years: list[int] | Iterable[int] = range(1978, 2025),
    dataset: str = "reanalysis-era5-land-monthly-means",
    path: Path | str = ".",
    **kwargs,
) -> Path:
    """
    Download monthly ERA5 reanalysis over a glacier bounding box and write a NetCDF.

    Parameters
    ----------
    target_grid : xarray.Dataset
        Target grid dataset providing the destination CRS (via ``spatial_ref``)
        and extent. Used to derive the geographic bounding box for the ERA5
        request.
    rgi_id : str
        Glacier identifier, e.g., ``"RGI2000-v7.0-C-01-10853"``. Used in the
        output filename.
    years : list of int or Iterable of int, default ``range(1978, 2025)``
        Years to request from ERA5.
    dataset : str, default ``"reanalysis-era5-land-monthly-means"``
        CDS dataset name for monthly single-level means (ERA5). Adjust if you
        intend to query ERA5-Land or other products.
    path : str or pathlib.Path, default ``"."``
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
    print("-" * 120)

    era5_filename = path / Path(f"era5_wgs84_{rgi_id}.nc")

    years = list(years)

    bounds = [
        target_grid.x_bnds.values[0][0],
        target_grid.y_bnds.values[0][0],
        target_grid.x_bnds.values[-1][-1],
        target_grid.y_bnds.values[-1][-1],
    ]
    dst_crs = target_grid.spatial_ref.attrs["crs_wkt"]
    t = Transformer.from_crs(dst_crs, "EPSG:4326")
    area = t.transform_bounds(*bounds)

    print(f"Bounding box {area}")

    era5_files = []
    era5_filename_1 = path / Path(f"era5_wgs84_{rgi_id}_tmp_1.nc")
    era5_files.append(era5_filename_1)
    ds = download_request(dataset, area, years, file_path=era5_filename_1, **kwargs)

    era5_filename_2 = path / Path(f"era5_wgs84_{rgi_id}_tmp_2.nc")
    era5_files.append(era5_filename_2)
    ds_geo = (
        download_request(
            dataset,
            area,
            [2013],
            variable=["geopotential"],
            file_path=era5_filename_2,
            **kwargs,
        )
        .squeeze()
        .drop_vars("time", errors="ignore")
    )
    ds_geo_ = (
        ds_geo.rio.write_crs("EPSG:4326")
        .rio.reproject_match(ds, resampling=Resampling.bilinear)
        .rename({"x": "longitude", "y": "latitude"})
    )

    lon_attrs = ds["longitude"].attrs
    lat_attrs = ds["latitude"].attrs

    if bool(ds.to_array().isnull().any().item()):
        print("Missing values detected, filling with global reanalysis")
        era5_filename_3 = path / Path(f"era5_wgs84_{rgi_id}_tmp_3.nc")
        era5_files.append(era5_filename_3)
        ds_global = download_request(
            "reanalysis-era5-single-levels-monthly-means",
            area,
            years,
            file_path=era5_filename_3,
            **kwargs,
        )
        ds_global_ = (
            ds_global.rio.write_crs("EPSG:4326")
            .rio.reproject_match(ds, resampling=Resampling.bilinear)
            .rename({"x": "longitude", "y": "latitude"})
        )
        common_vars = list(set(ds.data_vars) & set(ds_global_.data_vars))
        for v in common_vars:
            ds[v] = xr.where(np.isnan(ds[v]), ds_global_[v], ds[v])

    ds = xr.merge([ds, ds_geo_], compat="no_conflicts")
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

        if (not check_xr_fully(p)) or force_overwrite:
            dss = []
            for v in ["tas", "pr"]:
                zstore = df[df["variable_id"] == v].zstore.values[0]
                mapper = fs.get_mapper(zstore)

                # open using xarray
                ds = (
                    xr.open_zarr(
                        mapper,
                        consolidated=True,
                        decode_times=time_coder,
                        decode_timedelta=True,
                    ).drop_vars(["height"], errors="ignore")
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

            month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            bounds_start = np.cumsum([0] + month_lengths[:-1]).astype("float64")
            bounds_end = np.cumsum(month_lengths).astype("float64")
            time_mid = (bounds_start + bounds_end) / 2.0

            time_bounds = np.column_stack([bounds_start, bounds_end])

            ds = ds.assign_coords(time=("time", time_mid))
            ds["time"].attrs.update(
                {
                    "units": "days since 0001-01-01",
                    "calendar": "365_day",
                    "bounds": "time_bounds",
                }
            )
            ds["time_bounds"] = (("time", "nv"), time_bounds)

            ds.to_netcdf(p)

        responses.append(p)
    return responses


def prepare_snap(
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
    path : str or pathlib.Path, default ``"."``
        Output directory. Intermediate and final NetCDFs are written here.
    **kwargs
        Forwarded to :func:`download_file` and :func:`extract_archive`
        (e.g., ``force_overwrite=True``).

    Returns
    -------
    list[pathlib.Path]
        Paths to the three 30-year climatology NetCDF files:
        ``snap_cru_TS40_1920_1949.nc``, ``snap_cru_TS40_1950_1979.nc``,
        ``snap_cru_TS40_1980_2009.nc``.

    Raises
    ------
    requests.HTTPError
        If SNAP URLs fail to download.
    OSError
        On file I/O errors during extraction or NetCDF writing.
    ValueError
        If extracted files do not match the expected filename pattern.
    Exception
        Propagated errors from xarray/rioxarray routines.

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
    >>> out_paths = prepare_snap(path="snap_outputs", force_overwrite=True)
    >>> [p.name for p in out_paths]
    ['snap_cru_TS40_1920_1949.nc', 'snap_cru_TS40_1950_1979.nc', 'snap_cru_TS40_1980_2009.nc']
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

        p = Path(path) / f"snap_cru_TS40_{start}_{end}.nc"
        ps.append(p)

        if check_xr_lazy(p) and not force_overwrite:
            # Reuse existing file; no work scheduled
            continue

        # --- compute weighted monthly climatology for this period (all lazy) ---
        ds_sel = ds.sel(time=slice(start, end))
        ds_weighted = ds_sel.mean(dim="time")
        ds_weighted["air_temp_sd"] = ds_sel["air_temp"].std(dim="time")

        # coordinate metadata on x/y
        for c, axis, stdname in (
            ("x", "X", "projection_x_coordinate"),
            ("y", "Y", "projection_y_coordinate"),
        ):
            attrs = {
                "units": "m",
                "axis": axis,
                "standard_name": stdname,
                "long_name": f"{c}-coordinate in projected coordinate system",
            }
            if c in ds_weighted.coords:
                ds_weighted[c].attrs.update(attrs)

        bounds_start = 0.0
        bounds_end = 365.0
        time_mid = (bounds_start + bounds_end) / 2.0

        time_bounds = np.column_stack([bounds_start, bounds_end])

        ds_weighted = ds_weighted.expand_dims({"time": [time_mid]}, axis=0)
        ds_weighted["time"].attrs.update(
            {
                "units": "days since 0001-01-01",
                "calendar": "365_day",
                "bounds": "time_bounds",
            }
        )
        ds_weighted["time_bounds"] = (("time", "nv"), time_bounds)

        # CRS + spatial dims + grid_mapping tags
        ds_weighted = ds_weighted.rio.set_spatial_dims(x_dim="x", y_dim="y")
        ds_weighted.rio.write_crs("EPSG:3338", inplace=True)
        for v in ("precipitation", "air_temp", "surface"):
            if v in ds_weighted:
                ds_weighted[v].attrs["grid_mapping"] = "spatial_ref"

        # encoding: remove _FillValue without nuking other encodings
        encoding = {
            v: {"_FillValue": None}
            for v in (
                "x",
                "y",
                "surface",
                "air_temp",
                "air_temp_sd",
                "precipitation",
                "time",
                "time_bounds",
            )
            if v in ds_weighted
        }

        # schedule a lazy NetCDF write for this period
        p.unlink(missing_ok=True)
        delayed_writes.append(ds_weighted.to_netcdf(p, encoding=encoding, compute=False, engine="h5netcdf"))

    # Kick off all writes in parallel (plus internal dask parallelism)
    if delayed_writes:
        with ProgressBar():
            dask.compute(*delayed_writes)

    return ps
