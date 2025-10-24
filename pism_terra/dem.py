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

"""
Prepare DEM.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import NamedTemporaryFile

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr
from dem_stitcher import stitch_dem
from prefect import task
from rasterio.merge import merge

from pism_terra.aws import s3_to_local
from pism_terra.domain import create_domain
from pism_terra.download import download_archive, extract_archive
from pism_terra.raster import check_overlap, reproject_file
from pism_terra.vector import get_glacier_from_rgi_id


def get_surface_dem_by_bounds(
    bounds: tuple[float, float, float, float],
    dem_name: str = "glo_30",
) -> str:
    """
    Generate and save a surface DEM for a given bounding box, returning the file path.

    This function uses `stitch_dem` to create a digital elevation model (DEM) for a specified
    bounding box. The resulting DEM is saved as a temporary GeoTIFF file in the specified
    destination CRS, and the file path is returned for further use.

    Parameters
    ----------
    bounds : tuple of float
        The bounding box of the target area in the form (minx, miny, maxx, maxy).
    dem_name : str, optional
        The name of the DEM source to use (e.g., "glo_30", "arcticdem"). Default is "glo_30".

    Returns
    -------
    str
        The file path to the temporary GeoTIFF DEM file.

    Notes
    -----
    - The temporary file is not automatically deleted. The caller is responsible for cleanup.
    - The DEM is written with AREA_OR_POINT="Point" tag and ellipsoidal height.
    """
    X, p = stitch_dem(bounds, dem_name=dem_name, dst_ellipsoidal_height=False, dst_area_or_point="Point")

    # make sure profile fits what you're writing
    p = p.copy()
    p.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": X.dtype,  # e.g. 'float32'
            "BIGTIFF": "YES",  # allow >4GB
        }
    )

    with NamedTemporaryFile(suffix=".tif", delete=False) as geoid_file:
        geoid_path = geoid_file.name

    with rasterio.open(geoid_path, "w", **p) as src:
        src.write(X, 1)  # X can be 2D; this writes band 1
        src.update_tags(AREA_OR_POINT="Point")
    return geoid_path


def prepare_ice_thickness(glacier, target_grid: xr.Dataset | xr.DataArray, dataset: str = "millan", **kwargs):
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
    if dataset == "millan":
        thickness = prepare_ice_thickness_millan(glacier, target_grid, **kwargs)
    else:
        raise NotImplementedError(f"Ice thickness dataset '{dataset}' not implemented.")
    thickness = thickness.where(thickness > 0, 0)
    return thickness


def prepare_ice_thickness_millan(glacier, target_grid: xr.Dataset | xr.DataArray, **kwargs):
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

    bucket: str = "pism-cloud-data"

    s3_to_local(bucket, prefix="millan", dest_dir="data/ice_thickness")

    ice_thickness_files = list(Path("data/ice_thickness/millan").rglob("THICKNESS_*.tif"))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(check_overlap, path, glacier) for path in ice_thickness_files]

        overlapping_rasters = [f.result() for f in as_completed(futures) if f.result() is not None]

    thicknesses = []
    for k, path in enumerate(overlapping_rasters):
        if path is not None:
            projected_file = reproject_file(path, dst_crs=kwargs["target_crs"], resolution=50)
            da = rxr.open_rasterio(projected_file).squeeze().drop_vars("band", errors="ignore")
            thickness = da.interp_like(target_grid)
            thickness.name = "thickness"
            thickness["raster"] = k
            thicknesses.append(thickness)

    thickness = xr.concat(thicknesses, dim="raster").sum(dim="raster")

    return thickness


def prepare_ice_thickness_farinotti(glacier):
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

        overlapping_rasters = [f.result() for f in as_completed(futures) if f.result() is not None]

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


@task(retries=3, retry_delay_seconds=60)
def glacier_dem_from_rgi_id(
    rgi_id: str,
    rgi: gpd.GeoDataFrame | str | Path = "rgi/rgi.gpkg",
    dem_name: str = "glo_30",
    buffer_distance: float = 5000.0,
    resolution: float = 50.0,
) -> xr.Dataset:
    """
    Generate a glacier DEM, ice thickness, and bedrock topography from an RGI ID.

    This function extracts a glacier geometry from an RGI dataset, creates a buffered
    bounding box around it, stitches and reprojects a DEM over that region, interpolates
    it to a regular target grid, and derives ice thickness and bed elevation fields.

    Parameters
    ----------
    rgi_id : str
        The RGI ID of the target glacier (e.g., "RGI2000-v7.0-C-06-00014").
    rgi : geopandas.GeoDataFrame or str or Path, optional
        Either a pre-loaded RGI GeoDataFrame or the path to the RGI file (e.g., a GeoPackage).
        Default is "rgi/rgi.gpkg".
    dem_name : str, optional
        The name of the DEM source to use (e.g., "glo_30", "arcticdem"). Default is "glo_30".
    buffer_distance : float, optional
        Buffer distance in meters applied around the glacier geometry for DEM coverage.
        Default is 2000.0.
    resolution : float, optional
        Target spatial resolution (in meters) for the interpolated DEM and thickness fields.
        Default is 50.0.

    Returns
    -------
    xarray.Dataset
        A dataset containing the following variables on a regular 2D grid:

        - `surface` : Surface elevation (DEM) in meters
        - `thickness` : Ice thickness in meters
        - `bed` : Bedrock elevation (surface - thickness) in meters
        - `land_ice_area_fraction_retreat` : Boolean mask where ice may retreat (1 if ice-free in DEM)

    See Also
    --------
    get_glacier_from_rgi_id : Extract a glacier geometry by RGI ID.
    get_surface_dem_by_bounds : Download and mosaic a DEM from bounding box.
    reproject_file : Reproject and resample a raster to a target CRS and resolution.
    prepare_ice_thickness : Interpolate glacier ice thickness data to a target grid.
    create_domain : Create a regular xarray grid with specified bounds and resolution.

    Notes
    -----
    - This function assumes that the RGI entry contains a valid `epsg` code.
    - All raster reprojection and interpolation is done using `rasterio` and `rioxarray`.
    - The glacier is buffered in projected coordinates before reprojecting to geographic CRS.
    - The result is not written to disk — it is returned as an in-memory xarray.Dataset.
    - The mask `land_ice_area_fraction_retreat` is derived from missing values in the clipped DEM.
    """

    print("")
    print("Generate DEM")
    print("-" * 80)

    if isinstance(rgi, str | Path):
        rgi = gpd.read_file(rgi)
    glacier = get_glacier_from_rgi_id(rgi, rgi_id)
    glacier_series = glacier.iloc[0]
    dst_crs = glacier_series["epsg"]

    glacier_projected = glacier.to_crs(dst_crs)
    geometry_buffered_projected = glacier_projected.geometry.buffer(buffer_distance)
    geometry_buffered_geoid = geometry_buffered_projected.to_crs("EPSG:4326").iloc[0]

    bounds_geoid_buffered = geometry_buffered_geoid.bounds

    geoid_file = get_surface_dem_by_bounds(bounds_geoid_buffered, dem_name=dem_name)
    projected_file = reproject_file(geoid_file, dst_crs, resolution)

    surface = rxr.open_rasterio(projected_file).squeeze().drop_vars("band", errors="ignore")
    surface.name = "surface"
    surface.attrs.update({"standard_name": "land_ice_elevation", "units": "m"})
    x_min = np.ceil((surface.x.min()) / 100) * 100
    x_max = np.floor((surface.x.max()) / 100) * 100
    y_min = np.ceil((surface.y.min()) / 100) * 100
    y_max = np.floor((surface.y.max()) / 100) * 100
    target_grid = create_domain([x_min, x_max], [y_min, y_max], resolution=resolution, crs=dst_crs)

    surface = surface.interp_like(target_grid).fillna(0)

    ice_thickness = prepare_ice_thickness(glacier, target_grid=target_grid, target_crs=dst_crs)
    ice_thickness.attrs.update({"standard_name": "land_ice_thickness", "units": "m"})

    bed = surface - ice_thickness
    bed.name = "bed"
    bed.attrs.update({"standard_name": "bedrock_altitude", "units": "m"})

    liafr = surface.rio.clip(glacier_projected.geometry, drop=False)
    liafr = xr.where(liafr.isnull(), 0, 1)
    liafr.name = "land_ice_area_fraction_retreat"
    liafr.attrs.update({"units": "1"})
    liafr = liafr.astype("bool")

    ftt_mask = surface.rio.clip(glacier_projected.geometry, drop=False)
    ftt_mask = xr.where(ftt_mask.isnull(), 1, 0)
    ftt_mask.name = "ftt_mask"
    ftt_mask.attrs.update({"units": "1"})
    ftt_mask = ftt_mask.astype("bool")

    ds = xr.merge([bed, surface, ice_thickness, liafr, ftt_mask])
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds.rio.write_crs(dst_crs, inplace=True)
    return ds


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
