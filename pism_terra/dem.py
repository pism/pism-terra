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

"""
Prepare DEM.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray as rxr
from dem_stitcher import stitch_dem
from rasterio.warp import Resampling, calculate_default_transform, reproject

from pism_terra.domain import create_domain


def prepare_dem(
    rgi_id: str,
    rgi_file: str | Path = "rgi/rgi.gpkg",
    buffer_distance: float = 0.1,
    dem_name: str = "glo_30",
    resolution: float = 25.0,
):
    """
    Generate a reprojected and interpolated DEM for a specific glacier from the RGI.

    This function extracts a glacier by its RGI ID, creates a buffered bounding box,
    stitches a DEM over the area, reprojects it to the glacier's CRS, interpolates it
    to a target grid aligned to 100 m, and saves it as a NetCDF file.

    Parameters
    ----------
    rgi_id : str
        The RGI ID of the target glacier (e.g., "RGI2000-v7.0-C-06-00014").
    rgi_file : str or Path, optional
        Path to the RGI GeoPackage file. Default is "rgi/rgi.gpkg".
    buffer_distance : float, optional
        Buffer distance (in degrees) added around the glacier geometry. Default is 0.1.
    dem_name : str, optional
        Name of the DEM to be used in `stitch_dem` (e.g., "glo_30"). Default is "glo_30".
    resolution : float, optional
        Target resolution (in meters) for the resampled DEM. Default is 25.0.

    Returns
    -------
    None
        Saves the reprojected and interpolated DEM as a NetCDF file named "DEM_<rgi_id>.nc".

    See Also
    --------
    get_glacier_by_rgi_id : Function to extract a single glacier row from the RGI dataset.
    stitch_dem : Function to assemble a DEM mosaic over a bounding box.
    create_domain : Generates a regular domain with specified resolution and bounds.

    Notes
    -----
    - The output NetCDF file contains a single variable `"surface"` on a regular grid.
    - Intermediate files like GeoTIFF are cleaned or overwritten as needed.
    - CRS is inferred from the glacier metadata (`rgi["crs"]`) and used for reprojection.
    """

    rgi = gpd.read_file(rgi_file)
    glacier = get_glacier_by_rgi_id(rgi, rgi_id)

    dst_crs = glacier["crs"]
    bounds = glacier.geometry.buffer(buffer_distance).bounds

    X, p = stitch_dem(
        bounds,
        dem_name=dem_name,
        dst_ellipsoidal_height=True,
        dst_area_or_point="Point",
    )
    with NamedTemporaryFile(suffix=".tif") as geoid_file:
        with rasterio.open(geoid_file, "w", **p) as src:
            src.write(X, 1)
            src.update_tags(AREA_OR_POINT="Point")
            transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
            print(transform, width, height)
            kwargs = src.meta.copy()
            kwargs.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})

            projected_file = Path(f"DEM_{rgi_id}.tif")
            nc_file = Path(f"DEM_{rgi_id}.nc")
            if nc_file.exists():
                nc_file.unlink()
            with rasterio.open(projected_file, "w", **kwargs) as dst:
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
            da = rxr.open_rasterio(projected_file)
            da.name = "surface"
            ds = da.squeeze().to_dataset().drop_vars("band", errors="ignore")

            x_min = np.ceil(ds.x.min() / 100) * 100
            x_max = np.floor(ds.x.max() / 100) * 100
            y_min = np.ceil(ds.y.min() / 100) * 100
            y_max = np.floor(ds.y.max() / 100) * 100
            ds_target = create_domain([x_min, x_max], [y_min, y_max], resolution=resolution)
            ds = ds.interp_like(ds_target)
            ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
            ds.rio.write_crs(dst_crs, inplace=True)
            ds.to_netcdf(nc_file)


def get_glacier_by_rgi_id(rgi: gpd.GeoDataFrame, rgi_id: str) -> gpd.GeoSeries:
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
    glacier = rgi[rgi["rgi_id"] == rgi_id]
    return glacier.iloc[0]
