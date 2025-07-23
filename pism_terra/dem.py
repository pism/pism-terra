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

from pathlib import Path
from tempfile import NamedTemporaryFile

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr
from dem_stitcher import stitch_dem
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import box

from pism_terra.domain import create_domain


def raster_overlaps_glacier(
    raster: rasterio.DatasetBase | str | Path, glacier: gpd.GeoDataFrame | gpd.GeoSeries, glacier_crs: str | None = None
) -> bool:
    """
    Check whether a raster overlaps with a glacier geometry.

    This function determines whether the bounding box of a raster intersects
    with the bounding box of a glacier geometry, after reprojecting the glacier
    to the raster's CRS if needed.

    Parameters
    ----------
    raster : rasterio.DatasetBase or str or Path
        An open rasterio dataset or a path to a raster file.
    glacier : geopandas.GeoDataFrame or geopandas.GeoSeries
        The glacier geometry. Must contain exactly one geometry.
    glacier_crs : str, optional
        The CRS of the input glacier geometry need if glacier is a GeoSeries.

    Returns
    -------
    bool
        True if the raster and glacier bounding boxes intersect, False otherwise.

    Raises
    ------
    ValueError
        If `glacier` contains more than one geometry.

    Notes
    -----
    This function compares bounding boxes only. It does not perform pixel-wise
    or exact geometry intersection.
    """

    # Open raster
    if not isinstance(raster, rasterio.DatasetBase):
        with rasterio.open(raster) as src:
            raster_bounds = src.bounds
            raster_crs = src.crs
    else:
        raster_crs = raster.crs
        raster_bounds = raster.bounds

    # Ensure glacier is a GeoSeries with one geometry
    if isinstance(glacier, gpd.GeoDataFrame):
        if len(glacier) != 1:
            raise ValueError("The glacier input must contain exactly one geometry.")
        glacier = glacier.to_crs(raster_crs)
    elif isinstance(glacier, gpd.GeoSeries):
        if len(glacier) != 1:
            raise ValueError("Expected exactly one geometry in glacier input.")
        glacier = gpd.GeoSeries([glacier.iloc[0]], crs=glacier_crs)
    else:
        geometry = glacier.geometry
        glacier = gpd.GeoSeries([geometry], crs=glacier_crs)
        glacier = glacier.to_crs(raster_crs)

    # Compare bounding boxes
    glacier_box = box(*glacier.total_bounds)
    raster_box = box(*raster_bounds)

    return glacier_box.intersects(raster_box)


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
    X, p = stitch_dem(
        bounds,
        dem_name=dem_name,
        dst_ellipsoidal_height=True,
        dst_area_or_point="Point",
    )
    with NamedTemporaryFile(suffix=".tif", delete=False) as geoid_file:
        geoid_path = geoid_file.name  # save path before file is closed

    with rasterio.open(geoid_path, "w", **p) as src:
        src.write(X, 1)
        src.update_tags(AREA_OR_POINT="Point")

    return geoid_path


def reproject_file(src_file: str | Path, dst_crs: str | dict, resolution: float) -> str:
    """
    Reproject a raster file to a new coordinate reference system and resolution.

    This function opens a source raster file, reprojects its contents to a specified
    destination CRS and resolution using average resampling, and writes the result
    to a temporary GeoTIFF file. The path to this reprojected file is returned.

    Parameters
    ----------
    src_file : str or Path
        Path to the source raster file.
    dst_crs : str or dict
        Destination coordinate reference system (e.g., "EPSG:32633" or a CRS dict).
    resolution : float
        Target resolution for the output raster in units of the destination CRS.

    Returns
    -------
    str
        Path to the temporary reprojected raster file (GeoTIFF).

    Notes
    -----
    - The output file is written to a temporary location and is not automatically deleted.
      It is the caller's responsibility to clean it up.
    - The reprojected data is resampled using `Resampling.average`.
    """
    with rasterio.open(src_file) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})

        with NamedTemporaryFile(suffix=".tif", delete=False, delete_on_close=False) as projected_file:
            with rasterio.open(projected_file.name, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.average,
                        resolution=resolution,
                    )
            return projected_file.name


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
    ice_thickness_files = list(Path("data/ice_thickness").rglob("*.tif"))

    overlapping_rasters = [path for path in ice_thickness_files if raster_overlaps_glacier(path, glacier)]

    thicknesses = []
    for k, path in enumerate(overlapping_rasters):
        projected_file = reproject_file(path, dst_crs=kwargs["target_crs"], resolution=50)
        da = rxr.open_rasterio(projected_file).squeeze().drop_vars("band", errors="ignore")
        thickness = da.interp_like(target_grid)
        thickness.name = "thickness"
        thickness["raster"] = k
        thicknesses.append(thickness)

    thickness = xr.concat(thicknesses, dim="raster").sum(dim="raster")

    return thickness


def glacier_dem_from_rgi_id(
    rgi_id: str,
    rgi: gpd.GeoDataFrame | str | Path = "rgi/rgi.gpkg",
    dem_name: str = "glo_30",
    buffer_distance: float = 2000.0,
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
    print("-" * 20)

    if isinstance(rgi, str | Path):
        rgi = gpd.read_file(rgi)
    glacier = get_glacier_from_rgi_id(rgi, rgi_id)
    glacier_series = glacier.iloc[0]
    dst_crs = glacier_series["epsg"]

    glacier_projected = glacier.to_crs(dst_crs)
    geometry_buffered_projected = glacier_projected.geometry.buffer(buffer_distance)
    geometry_buffered_geoid = geometry_buffered_projected.to_crs("EPSG:4326").iloc[0]

    bounds = geometry_buffered_geoid.bounds

    geoid_file = get_surface_dem_by_bounds(bounds, dem_name=dem_name)
    projected_file = reproject_file(geoid_file, dst_crs, resolution)

    surface = rxr.open_rasterio(projected_file).squeeze().drop_vars("band", errors="ignore").fillna(0)
    surface.name = "surface"
    surface.attrs.update({"standard_name": "land_ice_elevation", "units": "m"})

    x_min = np.ceil((surface.x.min()) / 1000) * 1000
    x_max = np.floor((surface.x.max()) / 1000) * 1000
    y_min = np.ceil((surface.y.min()) / 1000) * 1000
    y_max = np.floor((surface.y.max()) / 1000) * 1000
    target_grid = create_domain([x_min, x_max], [y_min, y_max], resolution=resolution, crs=dst_crs)

    surface = surface.interp_like(target_grid)
    surface = surface.rio.set_spatial_dims(x_dim="x", y_dim="y")
    surface.rio.write_crs(dst_crs, inplace=True)

    ice_thickness = prepare_ice_thickness(glacier, target_grid=target_grid, target_crs=dst_crs)
    ice_thickness.attrs.update({"standard_name": "land_ice_thickness", "units": "m"})

    bed = surface - ice_thickness
    bed.name = "bed"
    bed.attrs.update({"standard_name": "bedrock_altitude", "units": "m"})

    tillwat = xr.zeros_like(surface) + 2
    tillwat.name = "tillwat"
    tillwat.attrs.update({"units": "m"})

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
    return xr.merge([bed, surface, ice_thickness, liafr, ftt_mask, tillwat])


def get_glacier_from_rgi_id(rgi: gpd.GeoDataFrame | str | Path, rgi_id: str) -> gpd.GeoDataFrame:
    """
    Return the row in the GeoDataFrame matching the given RGI ID.

    Parameters
    ----------
    rgi : geopandas.GeoDataFrame
        GeoDataFrame containing glacier data.
    rgi_id : str
        RGI identifier to look up.

    Returns
    -------
    geopandas.GeoSeries
        The matching row.
    """
    if isinstance(rgi, str | Path):
        rgi = gpd.read_file(rgi)

    glacier = rgi[rgi["rgi_id"] == rgi_id]
    return glacier
