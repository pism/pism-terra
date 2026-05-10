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

# pylint: disable=too-many-positional-arguments,broad-exception-caught

"""
Prepare ice thickness.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed as cf_as_completed
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import xarray as xr
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from tqdm.auto import tqdm

from pism_terra.aws import download_from_s3, s3_to_local
from pism_terra.download import download_archive, extract_archive
from pism_terra.raster import check_overlap
from pism_terra.vector import glaciers_in_complex
from pism_terra.workflow import check_xr_lazy

logger = logging.getLogger(__name__)


def get_maffezzoli_url(url_template: str, region: str) -> str:
    """
    Format the given URL template with the provided region.

    Parameters
    ----------
    url_template : str
        A string with a `{region}` placeholder to be replaced.
    region : str
        The region code to insert into the template.

    Returns
    -------
    str
        The formatted URL.
    """
    return url_template.format(region=region)


def prepare_ice_thickness_maffezzoli(
    regions: list,
    complexes: gpd.GeoDataFrame,
    glaciers: gpd.GeoDataFrame,
    output_path: Path | str,
    extract_path: Path | str,
    ntasks: int = 8,
    force_overwrite: bool = False,
):
    """
    Download, extract, and merge RGI region ice thickness.

    Parameters
    ----------
    regions : list
        Region codes (e.g. ``["01", "06"]``).
    complexes : geopandas.GeoDataFrame
        Complex outlines with an ``rgi_id`` and ``o1region`` column.
    glaciers : geopandas.GeoDataFrame
        Glacier outlines with ``rgi_id``, ``rgi_id_c``, and ``o1region`` columns.
    output_path : Path or str
        Root directory for output files.
    extract_path : Path or str
        Subdirectory under *output_path* for extracted archives.
    ntasks : int, default 8
        Maximum number of parallel workers.
    force_overwrite : bool, default False
        If True, re-download the file even if it already exists locally.
    """

    url_template: str = "https://zenodo.org/records/17724512/files/RGI70G_rgi{region}.zip?download=1"
    extract_path = Path(extract_path)
    output_path = Path(output_path)

    def _download_and_extract(region):
        """
        Download and extract ice thickness archive for a single region.

        Parameters
        ----------
        region : str
            Region code (e.g. ``"01"``).

        Returns
        -------
        str
            The region code that was processed.
        """
        url = get_maffezzoli_url(url_template, region)
        archive_dest = extract_path / Path(urlparse(url).path).name
        logger.info("Downloading ice thickness for region %s", region)
        archive = download_archive(url, dest=archive_dest, force_overwrite=force_overwrite, verbose=False)
        logger.info("Extracting ice thickness for region %s", region)
        extract_archive(archive, extract_path, force_overwrite=force_overwrite, verbose=False)
        logger.info("Ice thickness download complete for region %s", region)
        return region

    MAX_WORKERS = min(ntasks, len(regions))
    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_download_and_extract, region): region for region in regions}
        pbar = tqdm(cf_as_completed(futures), total=len(futures), desc="Downloading ice thickness")
        for future in pbar:
            region = futures[future]
            try:
                future.result()
                pbar.set_postfix_str(f"region {region} ✓")
            except Exception as e:
                failed.append((region, e))
                pbar.set_postfix_str(f"region {region} ✗")
    for region, err in failed:
        logger.error("Failed ice thickness: region %s with error: %s", region, err)

    logger.info("Starting ice thickness merging for %d regions", len(regions))

    def _reproject_and_merge(input_files, dst_crs, output_file, tmp_label):
        """
        Reproject inputs to ``dst_crs`` and write their mosaic to ``output_file``.

        Parameters
        ----------
        input_files : list[Path]
            Source GeoTIFFs to merge.
        dst_crs : str
            Target CRS string (e.g. ``"EPSG:32606"``).
        output_file : Path
            Target path for the merged GeoTIFF.
        tmp_label : str
            Label embedded in temporary reproject filenames so concurrent
            tasks operating on the same source raster don't collide.

        Returns
        -------
        Path
            ``output_file`` on success.
        """
        reprojected: list[Path] = []
        for fpath in input_files:
            with rasterio.open(fpath) as src:
                if src.crs == rasterio.crs.CRS.from_user_input(dst_crs):
                    reprojected.append(fpath)
                    continue
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                meta = src.meta.copy()
                meta.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})
                tmp_path = fpath.parent / f"{fpath.stem}__{tmp_label}_reproj.tif"
                with rasterio.open(tmp_path, "w", **meta) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                    )
                reprojected.append(tmp_path)

        mosaic, out_transform = merge(reprojected)
        with rasterio.open(reprojected[0]) as src:
            out_meta = src.meta.copy()
        # Cloud-Optimized GeoTIFF: tiled + DEFLATE + overviews, range-readable
        # from S3 (so QGIS via /vsis3/ streams without a full download).
        # predictor=3 for floats, 2 for ints. BIGTIFF=YES for >4 GB outputs.
        predictor = 3 if np.issubdtype(mosaic.dtype, np.floating) else 2
        out_meta.update(
            {
                "driver": "COG",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "compress": "DEFLATE",
                "predictor": predictor,
                "level": 6,
                "blocksize": 512,
                "overview_resampling": "AVERAGE",
                "BIGTIFF": "YES",
                "num_threads": "ALL_CPUS",
            }
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(mosaic)

        for fpath in reprojected:
            if fpath.stem.endswith("_reproj"):
                fpath.unlink(missing_ok=True)
        return output_file

    def _merge_complex(rgi_c_id, all_glaciers, o1region, dst_crs):
        """
        Build a per-complex thickness raster from member-glacier rasters.

        Phase-1 merge: each member glacier's source raster is fetched from
        ``extract_path/rgi<o1region>/<rgi_id>.tif`` (using the glacier's own
        ``o1region``) and merged into one per-complex GeoTIFF.

        Parameters
        ----------
        rgi_c_id : str
            Complex outline identifier, e.g. ``"RGI2000-v7.0-C-01-09429"``.
        all_glaciers : geopandas.GeoDataFrame
            Full glacier outline table with ``rgi_id``, ``rgi_id_c``, and
            ``o1region`` columns (not pre-filtered by region).
        o1region : str or int
            The complex's RGI-region code; used to build the output
            subdirectory path.
        dst_crs : str
            Target CRS string for the merged output.

        Returns
        -------
        pathlib.Path or None
            Path to the merged GeoTIFF, or ``None`` if no member rasters
            were found on disk.
        """
        glaciers_list = glaciers_in_complex(rgi_c_id, all_glaciers)
        if not glaciers_list:
            return None

        g_subset = all_glaciers[all_glaciers["rgi_id"].isin(glaciers_list)]
        glaciers_files = []
        for _, gr in g_subset.iterrows():
            g_region = gr.get("o1region")
            if pd.isna(g_region):
                continue
            f = extract_path / Path(f"rgi{int(g_region)}") / Path(f"{gr['rgi_id']}.tif")
            if f.exists():
                glaciers_files.append(f)
        if not glaciers_files:
            logger.debug("No thickness files found for complex %s", rgi_c_id)
            return None

        merged_path = output_path / Path(f"RGI2000-v7.0-C-{str(o1region).zfill(2)}")
        merged_file = merged_path / Path(f"{rgi_c_id}_thickness.tif")
        return _reproject_and_merge(glaciers_files, dst_crs, merged_file, rgi_c_id)

    def _merge_aggregate(agg_id, all_glaciers, dst_crs):
        """
        Build an aggregate thickness raster from per-complex outputs.

        Phase-2 merge: looks up the parent complexes that own glaciers tagged
        with ``agg_id`` (via the ``rgi_id_c_aggregate`` column), reads each
        parent's already-produced ``<parent>_thickness.tif`` from phase 1,
        and reprojects/mosaics them into one aggregate GeoTIFF. Orders of
        magnitude fewer inputs than re-merging every individual glacier.

        Parameters
        ----------
        agg_id : str
            Aggregate identifier (e.g. ``"S4F_AK"``).
        all_glaciers : geopandas.GeoDataFrame
            Full glacier outline table with ``rgi_id_c`` and
            ``rgi_id_c_aggregate`` columns.
        dst_crs : str
            Target CRS string for the merged output.

        Returns
        -------
        pathlib.Path or None
            Path to the merged GeoTIFF, or ``None`` if no parent thickness
            files (from phase 1) were found on disk.
        """
        # Find the parent complexes that own glaciers tagged with this aggregate.
        agg_col = all_glaciers.get("rgi_id_c_aggregate")
        if agg_col is None:
            return None
        sel = agg_col.fillna("").str.split(";").apply(lambda parts: agg_id in parts)
        parent_ids = all_glaciers.loc[sel, "rgi_id_c"].dropna().unique()
        if len(parent_ids) == 0:
            logger.debug("No parent complexes resolved for aggregate %s", agg_id)
            return None

        parent_files: list[Path] = []
        for parent_id in parent_ids:
            # Per-region subdirs are output_path/RGI2000-v7.0-C-XX/<parent>_thickness.tif
            for o1_dir in output_path.glob("RGI2000-v7.0-C-*"):
                p = o1_dir / f"{parent_id}_thickness.tif"
                if p.exists():
                    parent_files.append(p)
                    break
        if not parent_files:
            logger.debug("No parent thickness files found for aggregate %s", agg_id)
            return None

        merged_file = output_path / Path(agg_id) / Path(f"{agg_id}_thickness.tif")
        return _reproject_and_merge(parent_files, dst_crs, merged_file, agg_id)

    # Split tasks: regular complexes (have an o1region) run first; aggregates
    # depend on their parents' merged outputs and run after.
    regular_tasks = []
    aggregate_tasks = []
    for _, row in complexes.iterrows():
        crs_value = row.get("crs")
        if not isinstance(crs_value, str) or not crs_value:
            logger.warning("Complex %s has no CRS; skipping", row["rgi_id"])
            continue
        if pd.notna(row.get("o1region")):
            regular_tasks.append((row["rgi_id"], glaciers, row.get("o1region"), crs_value))
        else:
            aggregate_tasks.append((row["rgi_id"], glaciers, crs_value))

    failed = []
    if regular_tasks:
        with ThreadPoolExecutor(max_workers=min(ntasks, max(1, len(regular_tasks)))) as executor:
            futures = {
                executor.submit(
                    _merge_complex, rgi_c_id=rgi_c_id, all_glaciers=g, o1region=o1region, dst_crs=dst_crs
                ): rgi_c_id
                for rgi_c_id, g, o1region, dst_crs in regular_tasks
            }
            for future in tqdm(cf_as_completed(futures), total=len(futures), desc="Merging ice thickness"):
                rgi_c_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed.append((rgi_c_id, exc))

    if aggregate_tasks:
        with ThreadPoolExecutor(max_workers=min(ntasks, max(1, len(aggregate_tasks)))) as executor:
            futures = {
                executor.submit(_merge_aggregate, agg_id=agg_id, all_glaciers=g, dst_crs=dst_crs): agg_id
                for agg_id, g, dst_crs in aggregate_tasks
            }
            for future in tqdm(cf_as_completed(futures), total=len(futures), desc="Merging aggregates"):
                agg_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed.append((agg_id, exc))

    for rgi_c_id, err in failed:
        logger.error("Failed merging %s: %s", rgi_c_id, err)
    logger.info("Ice thickness merging complete")


def get_ice_thickness(
    rgi_id: str,
    target_grid: xr.Dataset | xr.DataArray,
    dataset: Literal["millan", "maffezzoli"] = "maffezzoli",
    path: str | Path = "input_files",
    **kwargs,
) -> xr.DataArray:
    """
    Prepare ice thickness data for a given glacier and target grid.

    This function dispatches to a dataset-specific loader to prepare an
    ice thickness field interpolated to the resolution and bounds of a
    specified target grid.

    Parameters
    ----------
    rgi_id : str
        Glacier identifier (e.g., ``"RGI2000-v7.0-C-06-00014"``) used to locate
        the relevant ice thickness tiles.
    target_grid : xarray.Dataset or xarray.DataArray
        Grid to which the output ice thickness will be interpolated.
    dataset : {"millan", "maffezzoli"}, default ``"maffezzoli"``
        The name of the ice thickness dataset to use.
    path : str or pathlib.Path, default ``"input_files"``
        Working directory used by helper routines to cache/write intermediate rasters/grids.
    **kwargs
        Additional keyword arguments passed to the dataset-specific function,
        e.g., `target_crs="EPSG:32641"`.

    Returns
    -------
    xarray.DataArray
        Ice thickness interpolated to the target grid.

    Raises
    ------
    NotImplementedError
        If the specified dataset is not supported.
    """
    logger.info("Getting ice thickness from dataset '%s'", dataset)
    if dataset == "maffezzoli":
        thickness = get_ice_thickness_maffezzoli(rgi_id, target_grid, path=path, **kwargs)
    elif dataset == "millan":
        thickness = get_ice_thickness_millan(rgi_id, target_grid, path=path, **kwargs)
    else:
        raise NotImplementedError(f"Ice thickness dataset '{dataset}' not implemented.")
    thickness = thickness.where(thickness > 0, 0)
    thickness.attrs.update({"standard_name": "land_ice_thickness", "units": "m"})

    return thickness


def get_ice_thickness_maffezzoli(
    rgi_id: str, target_grid: xr.Dataset | xr.DataArray, path: str | Path = "input_files", **kwargs
):
    """
    Load and interpolate Maffezzoli et al. (2026) ice thickness data to a target grid.

    Parameters
    ----------
    rgi_id : str
        Glacier identifier (e.g., ``"RGI2000-v7.0-C-06-00014"``) used to locate
        the overlapping Maffezzoli thickness rasters.
    target_grid : xarray.Dataset or xarray.DataArray
        Target grid to which ice thickness should be interpolated.
    path : str or pathlib.Path, default ``"input_files"``
        Working directory used by helper routines to cache/write intermediate rasters/grids.
    **kwargs
        Additional keyword arguments. Must include:
        - target_crs : str
            CRS to reproject rasters to (e.g., "EPSG:32641").

    Returns
    -------
    xarray.DataArray
        Interpolated ice thickness field on the target grid.

    Notes
    -----
    - Uses `rioxarray` to load and project raster files.
    - Assumes a fixed reprojected resolution of 100 meters.
    """

    force_overwrite: bool = bool(kwargs.pop("force_overwrite", False))
    bucket: str = kwargs.pop("bucket", "pism-cloud-data")
    prefix: str = kwargs.pop("prefix", "glacier")

    out_dir = Path(path)
    thickness_file = out_dir / f"thickness_maffezzoli_{rgi_id}.nc"

    if (not check_xr_lazy(thickness_file)) or force_overwrite:
        thickness_file.unlink(missing_ok=True)

        # Regular complex IDs strip the trailing glacier-number segment to get
        # the per-region subdir; aggregate IDs (no hyphens) live under their
        # own name.
        region = "-".join(rgi_id.split("-")[:-1]) or rgi_id
        s3_uri = f"s3://{bucket}/{prefix}/ice_thickness/maffezzoli/{region}/{rgi_id}_thickness.tif"
        local_tif = out_dir / f"{rgi_id}_thickness.tif"
        print(f"Downloading Maffezzoli thickness from {s3_uri}", flush=True)
        download_from_s3(s3_uri, local_tif)

        logger.info("Reprojecting and aligning thickness to target grid")
        da = rxr.open_rasterio(local_tif).sel(band=1).drop_vars("band")
        thickness = da.rio.reproject_match(target_grid, resampling=Resampling.bilinear)
        thickness.rio.write_nodata(None, inplace=True)
        thickness.name = "thickness"
        thickness = thickness.rio.write_crs(target_grid.rio.crs).rio.write_grid_mapping()
        thickness.to_netcdf(thickness_file)
        logger.info("Thickness saved to %s", thickness_file)
        # Return the in-memory result; CRS doesn't always survive the netCDF
        # round-trip, and we just computed a CRS-correct value.
        return thickness

    logger.info("Using cached thickness file %s", thickness_file)
    with xr.open_dataset(thickness_file) as ds:
        src_crs = ds.rio.crs
        if src_crs is None and "spatial_ref" in ds.coords:
            sr = ds["spatial_ref"]
            src_crs = sr.attrs.get("crs_wkt") or sr.attrs.get("spatial_ref")
        thickness = ds["thickness"].load()
    if src_crs is None:
        # Last resort: trust target_grid's CRS (the cache was generated against
        # an aligned grid, so this is correct by construction).
        src_crs = target_grid.rio.crs
    thickness = thickness.rio.write_crs(src_crs)
    return thickness


def get_ice_thickness_millan(
    glacier, target_grid: xr.Dataset | xr.DataArray, path: str | Path = "input_files", **kwargs
):
    """
    Load and interpolate Millan et al. (2022) ice thickness data to a target grid.

    This function identifies all Millan ice thickness raster files that overlap
    the input glacier geometry, reprojects them to the specified CRS and resolution,
    interpolates them onto the target grid, and returns the summed result.

    Parameters
    ----------
    glacier : geopandas.GeoDataFrame or geopandas.GeoSeries
        Geometry of the glacier to extract overlapping thickness rasters.
    target_grid : xarray.Dataset or xarray.DataArray
        Target grid to which ice thickness should be interpolated.
    path : str or pathlib.Path, default ``"input_files"``
        Working directory used by helper routines to cache/write intermediate rasters/grids.
    **kwargs
        Additional keyword arguments. Must include:
        - target_crs : str
            CRS to reproject rasters to (e.g., "EPSG:32641").

    Returns
    -------
    xarray.DataArray
        Interpolated and summed ice thickness field on the target grid.

    Notes
    -----
    - Uses `rioxarray` to load and project raster files.
    - All overlapping rasters are summed to produce the final thickness field.
    - Assumes a fixed reprojected resolution of 50 meters.
    """

    force_overwrite: bool = bool(kwargs.pop("force_overwrite", False))
    bucket: str = kwargs.pop("bucket", "pism-cloud-data")

    out_dir = Path(path)
    thickness_file = out_dir / "thickness_millan.nc"

    if (not check_xr_lazy(thickness_file)) or force_overwrite:

        thickness_file.unlink(missing_ok=True)
        logger.info("Downloading Millan thickness data from S3")
        # Could tweak this to only pull the relevant regions instead of all of it
        s3_to_local(bucket, prefix="millan", dest="data/ice_thickness/millan")
        ice_thickness_files = list(Path("data/ice_thickness/millan").rglob("THICKNESS_*.tif"))
        logger.info("Found %d Millan thickness tiles, checking overlap", len(ice_thickness_files))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(check_overlap, path, glacier) for path in ice_thickness_files]

            overlapping_rasters = [f.result() for f in cf_as_completed(futures) if f.result() is not None]

        logger.info("Found %d overlapping rasters, reprojecting and interpolating", len(overlapping_rasters))
        thicknesses = []
        for k, p in enumerate(overlapping_rasters):
            if p is not None:
                da = rxr.open_rasterio(p).squeeze().drop_vars("band", errors="ignore")
                thickness = da.rio.reproject_match(target_grid, resampling=Resampling.bilinear)
                thickness.rio.write_nodata(None, inplace=True)
                thickness.name = "thickness"
                thickness["raster"] = k
                thicknesses.append(thickness)

        thickness = xr.concat(thicknesses, dim="raster").sum(dim="raster")
        thickness = thickness.rio.write_crs(target_grid.rio.crs).rio.write_grid_mapping()
        thickness.to_netcdf(thickness_file)
        logger.info("Millan thickness saved to %s", thickness_file)
        return thickness

    logger.info("Using cached thickness file %s", thickness_file)
    with xr.open_dataset(thickness_file) as ds:
        src_crs = ds.rio.crs
        if src_crs is None and "spatial_ref" in ds.coords:
            sr = ds["spatial_ref"]
            src_crs = sr.attrs.get("crs_wkt") or sr.attrs.get("spatial_ref")
        thickness = ds["thickness"].load()
    if src_crs is None:
        src_crs = target_grid.rio.crs
    thickness = thickness.rio.write_crs(src_crs)

    return thickness


def add_malaspina_bed(
    ds: xr.Dataset,
    target_crs: str,
    bed_file: str | Path = "data/ice_thickness/malaspina/malaspina_bed_3338.tif",
    outline_file: str | Path = "data/rgi/rgi-malaspina.shp",
) -> xr.Dataset:
    """
    Replace bed topography in a dataset using the Malaspina Glacier bed dataset.

    This function reads a GeoTIFF file containing bed topography data for the Malaspina Glacier,
    clips it to the glacier outline, reprojects it to match the target dataset's CRS, and
    replaces the corresponding region in the input dataset. It also updates the `thickness`
    and `surface` fields accordingly.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing at least the variables 'bed', 'thickness', and 'surface'.
    target_crs : str
        The target coordinate reference system (CRS) to use for reprojection (e.g., "EPSG:3413").
    bed_file : str or Path, optional
        Path to the GeoTIFF file containing the Malaspina bed topography.
        Default is "data/ice_thickness/malaspina/malaspina_bed_3338.tif".
    outline_file : str or Path, optional
        Path to the glacier outline shapefile used to clip the bed topography.
        Default is "data/rgi/rgi-malaspina.shp".

    Returns
    -------
    xr.Dataset
        Modified dataset with updated 'bed', 'thickness', and 'surface' fields within the Malaspina region.

    Notes
    -----
    - Bed values of -9999.0 are treated as nodata and replaced with NaN.
    - Replaces `bed` where new values are available and recalculates `thickness = surface - bed`.
    - Ensures that thickness and surface are non-negative.
    - Updates CF-convention attributes and CRS metadata.
    """

    outline = gpd.read_file(outline_file).to_crs(target_crs)
    da = (
        rxr.open_rasterio(bed_file, mask=True)
        .squeeze()
        .drop_vars("band", errors="ignore")
        .rio.reproject_match(ds["bed"])
    )
    clipped_da = da.rio.clip(outline.geometry, drop=False)
    clipped_da = clipped_da.where(clipped_da != -9999.0, other=np.nan).drop_vars("spatial_ref")
    ds["bed"] = xr.where(~np.isnan(clipped_da), clipped_da, ds["bed"], keep_attrs=True)
    ds["thickness"] = xr.where(~np.isnan(clipped_da), ds["surface"] - clipped_da, ds["thickness"], keep_attrs=True)

    ds["thickness"] = ds["thickness"].where(ds["thickness"] > 0.0, 0.0)
    ds["surface"] = ds["surface"].where(ds["thickness"] > 0.0, 0.0)
    ds["surface"].attrs.update({"standard_name": "land_ice_elevation", "units": "m"})

    ds["thickness"].attrs.update({"standard_name": "land_ice_thickness", "units": "m"})

    ds["bed"].attrs.update({"standard_name": "bedrock_altitude", "units": "m"})
    return ds
