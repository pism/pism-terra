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
Prepare observations.
"""

from pathlib import Path

import geopandas as gpd
import rasterio
import rioxarray as rxr
import xarray as xr
from pyproj import Transformer

from pism_terra.vector import get_glacier_from_rgi_id
from pism_terra.workflow import check_xr_sampled


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


def get_itslive_velocities(components: list[str] = ["v", "vx", "vy", "vx_error", "vy_error"]) -> xr.Dataset:
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
    path: Path | str = "tmp.nc",
    force_overwrite: bool = False,
) -> xr.Dataset:
    """
    Generate observed glacier surface velocities for a glacier by RGI ID.

    Extracts the glacier geometry, builds a buffered extent, fetches a velocity
    product (e.g., ITS_LIVE) over that region, clips it to the glacier outline,
    and returns the result as an xarray dataset. A cached NetCDF at ``path`` is
    reused unless ``force_overwrite=True``.

    Parameters
    ----------
    rgi_id : str
        Glacier identifier (e.g., ``"RGI2000-v7.0-C-06-00014"``).
    rgi : geopandas.GeoDataFrame or str or pathlib.Path, default ``"rgi/rgi.gpkg"``
        In-memory RGI table or a path to a GeoPackage/shape readable by
        :func:`geopandas.read_file`. Must contain the feature with ``rgi_id``
        and an ``epsg`` column for the glacier CRS.
    product_name : str, default ``"its_live"``
        Velocity product to retrieve (e.g., ``"its_live"``). Passed to
        :func:`get_velocities_by_bounds`.
    buffer_distance : float, default ``2000.0``
        Buffer (meters) applied to the glacier polygon in its projected CRS to
        form the query extent for the velocity product.
    path : str or pathlib.Path, default ``"tmp.nc"``
        Cache file for the clipped velocity dataset. When present and valid
        (per :func:`check_xr_sampled`), it is opened instead of re-downloading.
    force_overwrite : bool, default ``False``
        If ``True``, ignore any existing cache at ``path`` and regenerate.

    Returns
    -------
    xarray.Dataset
        Velocity dataset clipped to the glacier outline. Variable names depend
        on the source product but typically include components (e.g., ``u``,
        ``v`` or ``vx``, ``vy``) and possibly speed (e.g., ``v``). CRS is
        recorded via :mod:`rioxarray`.

    Raises
    ------
    FileNotFoundError
        If the provided RGI path does not exist.
    ValueError
        If ``rgi_id`` is missing from the RGI layer or CRS info is invalid.
    Exception
        Propagated I/O, reprojection, or decoding errors from helper functions.

    See Also
    --------
    get_glacier_from_rgi_id
        Look up the glacier feature/CRS from the RGI table.
    get_velocities_by_bounds
        Fetch velocity data for a geographic bounding box.
    check_xr_sampled
        Lightweight validity check for an on-disk xarray dataset.

    Notes
    -----
    - Buffering is performed in the glacier’s projected CRS (meters) and the
      buffered geometry is reprojected to WGS84 only to compute geographic
      bounds for the velocity query.
    - On cache reuse, the code sets the dataset CRS to the glacier CRS; on a
      fresh download, CRS comes from the source product. If you require a
      specific target CRS, reproject with ``.rio.reproject_match(...)``.
    """

    print("")
    print("Generate Velocity Observations")
    print("-" * 80)

    crs = "EPSG:3857"

    if isinstance(rgi, str | Path):
        rgi = gpd.read_file(rgi)
    glacier = get_glacier_from_rgi_id(rgi, rgi_id)
    glacier_series = glacier.iloc[0]
    dst_crs = glacier_series["epsg"]

    if (not check_xr_sampled(path)) or force_overwrite:

        path = Path(path)
        path.unlink(missing_ok=True)

        glacier_projected = glacier.to_crs(dst_crs)
        geometry_buffered_projected = glacier_projected.geometry.buffer(buffer_distance)
        geometry_buffered_geoid = geometry_buffered_projected.to_crs("EPSG:4326").iloc[0]

        bounds = geometry_buffered_geoid.bounds

        ds = get_velocities_by_bounds(bounds, product_name=product_name)
        glacier_projected = glacier.to_crs(crs)
        ds_clipped = ds.rio.clip(glacier_projected.geometry)
        ds_clipped.to_netcdf(path)

    else:
        ds_clipped = xr.open_dataset(path)
        ds_clipped.rio.write_crs(crs, inplace=True)
    return ds_clipped
