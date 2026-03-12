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

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed as cf_as_completed
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from tqdm.auto import tqdm

from pism_terra.aws import download_from_s3, s3_to_local
from pism_terra.download import download_archive, extract_archive
from pism_terra.raster import check_overlap, reproject_file
from pism_terra.vector import glaciers_in_complex
from pism_terra.workflow import check_xr_lazy


def get_maffezzoli_url(url_template: str, region: str) -> str:
    """
    Format the given URL template with the provided region.

    Parameters
    ----------
    url_template : str
        A string with `{region}` and `{outline_type}` placeholders to be replaced.
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
    output_path : Path
        Roor directory for output files.
    extract_path : Path or str, optional
        Subdirectory under *output_path* for extracted archives.
        Defaults to ``"ice_thickness"``.
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
        archive = download_archive(url, dest=archive_dest, force_overwrite=force_overwrite, verbose=False)
        extract_archive(archive, extract_path, force_overwrite=force_overwrite, verbose=False)
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
        print(f"✗ Failed ice thickness: {region} with error: {err}")

    def _merge_complex(rgi_c_id, region_g, region_code, dst_crs):
        """
        Merge glacier thickness rasters for a single complex.

        Glacier rasters are reprojected to *dst_crs* before merging so that
        glaciers spanning different UTM zones can be combined.

        Parameters
        ----------
        rgi_c_id : str
            The complex outline identifier.
        region_g : geopandas.GeoDataFrame
            Glacier outlines for the region.
        region_code : str
            Region code (e.g. ``"1"``).
        dst_crs : str
            Target CRS for the output (e.g. ``"EPSG:32606"``).

        Returns
        -------
        str or None
            Path to the merged file, or ``None`` if no files found.
        """
        glaciers_list = glaciers_in_complex(rgi_c_id, region_g)
        glaciers_files = [extract_path / Path(f"rgi{region_code}") / Path(f"{g}.tif") for g in glaciers_list]
        glaciers_files = [f for f in glaciers_files if f.exists()]
        if not glaciers_files:
            return None

        # Reproject each glacier raster to the complex's CRS
        reprojected = []
        for fpath in glaciers_files:
            with rasterio.open(fpath) as src:
                if src.crs == rasterio.crs.CRS.from_user_input(dst_crs):
                    reprojected.append(fpath)
                    continue
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                meta = src.meta.copy()
                meta.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})
                tmp_path = fpath.parent / f"{fpath.stem}_reproj.tif"
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
        out_meta.update(
            {"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_transform}
        )
        merged_path = output_path / Path(f"RGI2000-v7.0-C-{region_code.zfill(2)}")
        merged_path.mkdir(parents=True, exist_ok=True)

        merged_file_path = merged_path / Path(f"{rgi_c_id}_thickness.tif")
        with rasterio.open(merged_file_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        # Clean up temporary reprojected files
        for fpath in reprojected:
            if fpath.stem.endswith("_reproj"):
                fpath.unlink(missing_ok=True)

        return str(merged_path)

    # Build list of all (rgi_c_id, region_g, region_code) tasks across regions
    merge_tasks = []
    for region in regions:
        region_c = complexes[complexes["o1region"] == region.zfill(2)]
        region_g = glaciers[glaciers["o1region"] == region.zfill(2)]
        for _, row in region_c.iterrows():
            merge_tasks.append((row["rgi_id"], region_g, region, row["epsg"]))

    with ThreadPoolExecutor(max_workers=min(ntasks, max(1, len(merge_tasks)))) as executor:
        futures = {
            executor.submit(
                _merge_complex, rgi_c_id=rgi_c_id, region_g=region_g, region_code=region_code, dst_crs=dst_crs
            ): rgi_c_id
            for rgi_c_id, region_g, region_code, dst_crs in merge_tasks
        }
        failed = []
        for future in tqdm(cf_as_completed(futures), total=len(futures), desc="Merging ice thickness"):
            rgi_c_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                failed.append((rgi_c_id, exc))
        for rgi_c_id, err in failed:
            print(f"✗ Failed {rgi_c_id}: {err}")


def get_ice_thickness(
    glacier,
    target_grid: xr.Dataset | xr.DataArray,
    dataset: Literal["millan", "maffezzoli"] = "maffezzoli",
    path: str | Path = "input_files",
    **kwargs,
):
    """
    Prepare ice thickness data for a given glacier and target grid.

    This function dispatches to a dataset-specific loader to prepare an
    ice thickness field interpolated to the resolution and bounds of a
    specified target grid. Currently only the "millan" dataset is supported.

    Parameters
    ----------
    glacier : geopandas.GeoDataFrame or geopandas.GeoSeries
        Glacier geometry to match against ice thickness tiles.
    target_grid : xarray.Dataset or xarray.DataArray
        Grid to which the output ice thickness will be interpolated.
    dataset : str, optional
        The name of the ice thickness dataset to use. Currently only "millan" is implemented.
        Default is "millan".
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
    if dataset == "maffezzoli":
        thickness = get_ice_thickness_maffezzoli(glacier, target_grid, path=path, **kwargs)
    elif dataset == "millan":
        thickness = get_ice_thickness_millan(glacier, target_grid, path=path, **kwargs)
    else:
        raise NotImplementedError(f"Ice thickness dataset '{dataset}' not implemented.")
    thickness = thickness.where(thickness > 0, 0)
    thickness.attrs.update({"standard_name": "land_ice_thickness", "units": "m"})

    return thickness


def get_ice_thickness_maffezzoli(
    glacier, target_grid: xr.Dataset | xr.DataArray, path: str | Path = "input_files", **kwargs
):
    """
    Load and interpolate Maffezzoli et al. (2026) ice thickness data to a target grid.

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
    thickness_file = out_dir / "thickness_maffezzoli.nc"

    if (not check_xr_lazy(thickness_file)) or force_overwrite:
        thickness_file.unlink(missing_ok=True)

        rgi_id = glacier.iloc[0]["rgi_id"]
        o1region = glacier.iloc[0]["o1region"]
        region = f"RGI2000-v7.0-C-{o1region}"
        s3_uri = f"s3://{bucket}/glaciers/ice_thickness/maffezzoli/{region}/{rgi_id}_thickness.tif"
        local_tif = out_dir / f"{rgi_id}_thickness.tif"
        download_from_s3(s3_uri, local_tif)

        projected_file = reproject_file(local_tif, dst_crs=kwargs["target_crs"], resolution=50)
        da = rxr.open_rasterio(projected_file).sel(band=1).drop_vars("band")
        thickness = da.interp_like(target_grid)
        thickness.rio.write_nodata(None, inplace=True)
        thickness.name = "thickness"
        thickness.to_netcdf(thickness_file)

    thickness = xr.open_dataarray(thickness_file)

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
        # Could tweak this to only pull the relevant regions instead of all of it
        s3_to_local(bucket, prefix="millan", dest="data/ice_thickness/millan")
        ice_thickness_files = list(Path("data/ice_thickness/millan").rglob("THICKNESS_*.tif"))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(check_overlap, path, glacier) for path in ice_thickness_files]

            overlapping_rasters = [f.result() for f in cf_as_completed(futures) if f.result() is not None]

        thicknesses = []
        for k, p in enumerate(overlapping_rasters):
            if p is not None:
                projected_file = reproject_file(p, dst_crs=kwargs["target_crs"], resolution=50)
                da = rxr.open_rasterio(projected_file).squeeze().drop_vars("band", errors="ignore")
                thickness = da.interp_like(target_grid)
                thickness.rio.write_nodata(None, inplace=True)
                thickness.name = "thickness"
                thickness["raster"] = k
                thicknesses.append(thickness)

        thickness = xr.concat(thicknesses, dim="raster").sum(dim="raster")
        thickness.to_netcdf(thickness_file)

    thickness = xr.open_dataarray(thickness_file)

    return thickness


def get_ice_thickness_farinotti(glacier):
    """
    Load and interpolate Farniotti et al (2019) ice thickness data to a target grid.

    This function identifies all Millan ice thickness raster files that overlap
    the input glacier geometry, reprojects them to the specified CRS and resolution,
    interpolates them onto the target grid, and returns the summed result.

    Parameters
    ----------
    glacier : geopandas.GeoDataFrame or geopandas.GeoSeries
        Geometry of the glacier to extract overlapping thickness rasters.

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

    path = Path("data/ice_thickness")
    path.mkdir(parents=True, exist_ok=True)

    region = glacier["o1region"]
    url = f"https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/315707/composite_thickness_RGI60-{region}.zip"
    archive = download_archive(url)
    extract_archive(archive, extract_to=path)

    ice_thickness_files = list(Path(f"data/ice_thickness/RGI60-{region}").rglob("*.tif"))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(check_overlap, path, glacier) for path in ice_thickness_files]

        overlapping_rasters = [f.result() for f in cf_as_completed(futures) if f.result() is not None]

    # Step 1: List all .tif files
    tif_files = overlapping_rasters

    # Step 2: Open all files as datasets
    src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files]

    # Step 3: Merge them
    mosaic, out_transform = merge(src_files_to_mosaic)

    # Step 4: Get metadata from first file, update with new shape and transform
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update(
        {"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_transform}
    )

    # Step 5: Write the result to disk
    with rasterio.open("merged.tif", "w", **out_meta) as dest:
        dest.write(mosaic)


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
