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
Prepare observations.
"""

from pathlib import Path

import geopandas as gpd
import rasterio
import rioxarray as rxr
import xarray as xr
from pyproj import Transformer

from pism_terra.vector import get_glacier_from_rgi_id


def get_velocities_by_bounds(
    bounds: tuple[float, float, float, float],
    product_name: str = "its_live",
) -> xr.Dataset:
    """
    Retrieve and subset a velocity product over a specified geographic bounding box.

    This function fetches a global surface velocity dataset (e.g., ITS_LIVE) and returns a
    spatial subset clipped to the specified bounding box.

    Parameters
    ----------
    bounds : tuple of float
        Geographic bounding box in EPSG:4326 coordinates, formatted as
        (minx, miny, maxx, maxy).
    product_name : {"its_live"}, optional
        The name of the velocity product to query. Currently only "its_live" is supported.
        Default is "its_live".

    Returns
    -------
    xarray.Dataset
        A subset of the velocity dataset clipped to the given bounding box.

    Raises
    ------
    NotImplementedError
        If the requested product name is not supported.

    Notes
    -----
    - The returned dataset includes geospatial coordinates and metadata.
    - The CRS of the bounding box is assumed to be EPSG:4326 (longitude/latitude).
    - This function currently only supports the ITS_LIVE global velocity mosaic.
    """

    # Define source CRS
    src_crs = "EPSG:4326"

    # Load dataset
    if product_name == "its_live":
        ds = get_itslive_velocities()
    else:
        raise NotImplementedError(f"Velocity product '{product_name}' is not supported.")

    # Define destination CRS
    dst_crs = ds.rio.crs

    # Transform bounds to destination CRS
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    bbox_out = transformer.transform_bounds(*bounds)

    # Clip dataset
    subset = ds.rio.clip_box(minx=bbox_out[0], miny=bbox_out[1], maxx=bbox_out[2], maxy=bbox_out[3])

    return subset


def get_itslive_velocities(components: list[str] = ["v", "vx", "vy"]) -> xr.Dataset:
    """
    Load the global ITS_LIVE surface velocity mosaic as an xarray dataset.

    This function reads ITS_LIVE VRT-backed raster layers for specified velocity
    components using `rioxarray` with Dask chunking enabled for efficient access.

    Parameters
    ----------
    components : list of str, optional
        List of velocity components to load. Valid entries include:
        - "v": velocity magnitude
        - "vx": x-component of velocity
        - "vy": y-component of velocity
        Defaults to ["v", "vx", "vy"].

    Returns
    -------
    xarray.Dataset
        A dataset with one DataArray per requested velocity component. Each variable
        has shape (y, x) and spatial coordinates in a projected CRS (EPSG:3413).

    Notes
    -----
    - The data are streamed from Amazon S3 using VRTs and are not downloaded locally.
    - Data are chunked using Dask for parallel I/O. Each raster is read with chunk size (1024, 1024).
    - Missing values are represented using a mask (`masked=True`).
    - Coordinate metadata and CRS are read from the VRT headers.
    """
    dss = []
    for c in components:
        url = f"""https://its-live-data.s3-us-west-2.amazonaws.com/velocity_mosaic/v2/static/cog_global/ITS_LIVE_velocity_120m_0000_v02_{c}.vrt"""
        ds = (
            rxr.open_rasterio(url, parse_coordinates=True, chunks={"x": 1024, "y": 1024}, masked=True)
            .isel(band=0)
            .drop_vars("band")
        )
        ds.name = c
        dss.append(ds)

    return xr.merge(dss)


def glacier_velocities_from_rgi_id(
    rgi_id: str,
    rgi: gpd.GeoDataFrame | str | Path = "rgi/rgi.gpkg",
    product_name: str = "its_live",
    buffer_distance: float = 2000.0,
) -> xr.Dataset:
    """
    Generate a observed glacier velocities from an RGI ID.

    This function extracts a glacier geometry from an RGI dataset, creates a buffered
    bounding box around it, stitches and reprojects velocities over that region.

    Parameters
    ----------
    rgi_id : str
        The RGI ID of the target glacier (e.g., "RGI2000-v7.0-C-06-00014").
    rgi : geopandas.GeoDataFrame or str or Path, optional
        Either a pre-loaded RGI GeoDataFrame or the path to the RGI file (e.g., a GeoPackage).
        Default is "rgi/rgi.gpkg".
    product_name : str, optional
        The name of the DEM source to use (e.g., "its_live"). Default is "its_live".
    buffer_distance : float, optional
        Buffer distance in meters applied around the glacier geometry for DEM coverage.
        Default is 2000.0.

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
    print("Generate Velocity Observations")
    print("-" * 80)

    if isinstance(rgi, str | Path):
        rgi = gpd.read_file(rgi)
    glacier = get_glacier_from_rgi_id(rgi, rgi_id)
    glacier_series = glacier.iloc[0]
    dst_crs = glacier_series["epsg"]

    glacier_projected = glacier.to_crs(dst_crs)
    geometry_buffered_projected = glacier_projected.geometry.buffer(buffer_distance)
    geometry_buffered_geoid = geometry_buffered_projected.to_crs("EPSG:4326").iloc[0]

    bounds = geometry_buffered_geoid.bounds

    ds = get_velocities_by_bounds(bounds, product_name=product_name)
    crs = ds.rio.crs
    glacier_projected = glacier.to_crs(crs)
    ds_clipped = ds.rio.clip(glacier_projected.geometry)

    return ds_clipped
