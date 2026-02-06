# Copyright (C) 2025, 2026 Andy Aschwanden
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
# pylint: disable=too-many-positional-arguments,unused-import

"""
Provide raster functions.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import xarray as xr
from geocube.api.core import make_geocube
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import box

from pism_terra.workflow import check_xr_lazy


def create_ds(
    output_file: Path | str,
    date: pd.Timestamp,
    ds1: gpd.GeoDataFrame,
    ds2: gpd.GeoDataFrame,
    geom: dict,
    resolution: float = 450,
    crs: str = "EPSG:3413",
    encoding_time: dict = {"time": {"units": "hours since 1972-01-01 00:00:00", "calendar": "standard"}},
) -> Path | str:
    """
    Create a dataset representing land ice area fraction retreat and save it to a NetCDF file.

    Parameters
    ----------
    output_file : Path or str
        The path to the output NetCDF file.
    date : pd.Timestamp
        The date for which the dataset is created.
    ds1 : gpd.GeoDataFrame
        The first GeoDataFrame containing the initial geometries.
    ds2 : gpd.GeoDataFrame
        The second GeoDataFrame containing the geometries to be compared.
    geom : dict
        The geometry dictionary for the geocube.
    resolution : float, optional
        The resolution of the geocube, by default 450.
    crs : str, optional
        The coordinate reference system, by default "EPSG:3413".
    encoding_time : dict, optional
        The encoding settings for the time variable, by default {"time": {"units": "hours since 1972-01-01 00:00:00", "calendar": "standard"}}.

    Returns
    -------
    Path
        The path to the saved NetCDF file.

    Examples
    --------
    >>> import geopandas as gp
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> date = pd.Timestamp("2023-01-01")
    >>> ds1 = gpd.read_file("path_to_ds1.shp")
    >>> ds2 = gpd.read_file("path_to_ds2.shp")
    >>> geom = {"type": "Polygon", "coordinates": [[[...]]]}
    >>> result_path = create_ds(date, ds1, ds2, geom)
    >>> print(result_path)
    """

    start = date.replace(day=1)

    ds = gpd.GeoDataFrame(ds1, crs=crs)
    geom_valid = ds.geometry.make_valid()
    ds.geometry = geom_valid
    ds_dissolved = ds.dissolve()
    diff = ds2.difference(ds_dissolved.buffer(5))
    n = len(diff)
    diff_df = {"land_ice_area_fraction_retreat": np.ones(n)}
    diff_gp = gpd.GeoDataFrame(data=diff_df, geometry=diff, crs=crs)
    ds = make_geocube(vector_data=diff_gp, geom=geom, resolution=(resolution, resolution))
    ds = ds.fillna(0)
    ds["land_ice_area_fraction_retreat"].attrs["units"] = "1"
    ds["land_ice_area_fraction_retreat"].attrs.pop("coordinates", None)
    ds["land_ice_area_fraction_retreat"].attrs["grid_mapping"] = "spatial_ref"
    ds = ds.expand_dims(time=[start])
    comp = {"zlib": True, "complevel": 2}
    encoding = {var: comp for var in ds.data_vars}
    encoding.update(encoding_time)

    ds.to_netcdf(output_file, encoding=encoding)

    return output_file


def create_ds(
    date: pd.Timestamp,
    ds1: gpd.GeoDataFrame,
    ds2: gpd.GeoDataFrame,
    geom: dict,
    resolution: float = 450,
    crs: str = "EPSG:3413",
    result_dir: Path | str = "front_retreat",
    encoding_time: dict = {"time": {"units": "hours since 1972-01-01 00:00:00", "calendar": "standard"}},
) -> Path:
    """
    Create a dataset representing land ice area fraction retreat and save it to a NetCDF file.

    Parameters
    ----------
    date : pd.Timestamp
        The date for which the dataset is created.
    ds1 : gpd.GeoDataFrame
        The first GeoDataFrame containing the initial geometries.
    ds2 : gpd.GeoDataFrame
        The second GeoDataFrame containing the geometries to be compared.
    geom : dict
        The geometry dictionary for the geocube.
    resolution : float, optional
        The resolution of the geocube, by default 450.
    crs : str, optional
        The coordinate reference system, by default "EPSG:3413".
    result_dir : Union[Path, str], optional
        The directory where the result NetCDF file will be saved, by default "front_retreat".
    encoding_time : dict, optional
        The encoding settings for the time variable, by default {"time": {"units": "hours since 1972-01-01 00:00:00", "calendar": "standard"}}.

    Returns
    -------
    Path
        The path to the saved NetCDF file.

    Examples
    --------
    >>> import geopandas as gp
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> date = pd.Timestamp("2023-01-01")
    >>> ds1 = gpd.read_file("path_to_ds1.shp")
    >>> ds2 = gpd.read_file("path_to_ds2.shp")
    >>> geom = {"type": "Polygon", "coordinates": [[[...]]]}
    >>> result_path = create_ds(date, ds1, ds2, geom)
    >>> print(result_path)
    """
    ds = gpd.GeoDataFrame(ds1, crs=crs)
    geom_valid = ds.geometry.make_valid()
    ds.geometry = geom_valid
    ds_dissolved = ds.dissolve()
    diff = ds2.difference(ds_dissolved.buffer(5))
    n = len(diff)
    diff_df = {"land_ice_area_fraction_retreat": np.ones(n)}
    diff_gp = gpd.GeoDataFrame(data=diff_df, geometry=diff, crs=crs)
    ds = make_geocube(vector_data=diff_gp, geom=geom, resolution=(resolution, resolution))
    ds = ds.fillna(0)
    ds["land_ice_area_fraction_retreat"].attrs["units"] = "1"

    start = date.replace(day=1)

    ds = ds.expand_dims(time=[start])
    p = Path(result_dir)
    p.mkdir(parents=True, exist_ok=True)

    comp = {"zlib": True, "complevel": 2}
    encoding = {var: comp for var in ds.data_vars}
    encoding.update(encoding_time)

    fn = p / Path(f"frontretreat_g{resolution}m_{start.year}-{start.month}-{start.day}.nc")
    ds.to_netcdf(fn, encoding=encoding)
    ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    ds.rio.write_crs(crs, inplace=True)
    return fn


def add_time_bounds(ds: xr.Dataset) -> xr.Dataset:
    """
    Add time bounds to a dataset by computing interval start and end times.

    This function computes time bounds for each time step in the dataset
    by pairing each timestamp with the following one, creating a bounds array
    with shape (n_time - 1, 2). The dataset is truncated by one time step to
    ensure alignment with the bounds.

    Parameters
    ----------
    ds : xr.Dataset
        An xarray Dataset with a one-dimensional "time" coordinate.

    Returns
    -------
    xr.Dataset
        A new dataset with:
        - one fewer time step (last one dropped),
        - a new variable "time_bounds" with shape (time, 2),
        - an attribute "bounds" set on the "time" coordinate pointing to "time_bounds".

    Notes
    -----
    - The function assumes that `ds["time"]` is sorted and regularly spaced.
    - The resulting time bounds are left-closed, right-open intervals: [t, t+1).
    """
    time = ds["time"]
    # Compute bounds (start is current time, end is next time)
    start = time.values[:-1]
    end = time.values[1:]

    # Drop the last bound to match shape
    time_bounds = xr.DataArray(np.stack([start, end], axis=1), dims=["time", "nv"], coords={"time": time[:-1]})
    ds = ds.isel({"time": slice(0, -1)})
    ds["time_bounds"] = time_bounds
    ds["time"].attrs["bounds"] = "time_bounds"
    return ds


def apply_perimeter_band(
    da: xr.DataArray, bounds: list[float] | None = None, width: float = 1000.0, value: float = -1000.0
) -> xr.DataArray:
    """
    Apply a constant-valued band around the perimeter of a 2D xarray DataArray.

    This function sets the values in a rectangular border region along the edges
    of a 2D DataArray to a specified constant. The band width is defined in the
    physical coordinate units (e.g., meters).

    Parameters
    ----------
    da : xarray.DataArray
        A 2D DataArray with 'x' and 'y' coordinates. The array is modified in-place.
    bounds : list of float, optional
        Explicit bounds in the form [x_min, y_min, x_max, y_max]. If None (default),
        the bounds are inferred from `da.x` and `da.y`.
    width : float, optional
        Width of the perimeter band in coordinate units. Default is 1000.0.
    value : float, optional
        The constant value to assign to the perimeter region. Default is -1000.0.

    Returns
    -------
    xarray.DataArray
        The input DataArray with the perimeter band modified.

    Notes
    -----
    - This function modifies the data values directly using `.values`, so if the input
      is backed by Dask, it will be eagerly loaded into memory.
    - It assumes 'x' and 'y' are 1D coordinate arrays and ordered ascending.
    - The operation is not applied to NaN-only borders unless explicitly masked.
    """
    x, y = da.x.values, da.y.values
    if bounds is None:
        # Identify mask for outer border
        x_min, x_max = x[0], x[-1]
        y_min, y_max = y[0], y[-1]
    else:
        x_min, y_min, x_max, y_max = bounds
    mask = ((x < x_min + width) | (x > x_max - width))[np.newaxis, :] | ((y < y_min + width) | (y > y_max - width))[
        :, np.newaxis
    ]

    da.values[mask] = value

    return da


def check_overlap(path: str | Path, glacier: gpd.GeoDataFrame | gpd.GeoSeries) -> str | Path | None:
    """
    Check whether a raster file spatially overlaps a glacier geometry.

    This function determines if the raster at the given path overlaps with the
    provided glacier geometry. If an overlap is found, the input path is returned;
    otherwise, `None` is returned. This is typically used to filter a list of
    raster files based on spatial intersection.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the raster file to be checked for spatial overlap.
    glacier : geopandas.GeoDataFrame or geopandas.GeoSeries
        The glacier geometry (in projected CRS) to test for intersection.

    Returns
    -------
    str or Path or None
        The input path if the raster intersects with the glacier geometry;
        otherwise, `None`.

    Notes
    -----
    - This function is often used in parallel workflows (e.g., with concurrent.futures)
      to efficiently identify overlapping rasters.
    - The CRS of the raster and the glacier geometry must match, or be reprojected
      accordingly within the `raster_overlaps_glacier` implementation.
    """
    return path if raster_overlaps_glacier(path, glacier) else None


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
        # FIXME: likely neds to have the `glacier.to_crs(raster_crs)` here ...  # pylint: disable=W0511
    else:
        geometry = glacier.geometry
        glacier = gpd.GeoSeries([geometry], crs=glacier_crs)
        glacier = glacier.to_crs(raster_crs)

    # Compare bounding boxes
    glacier_box = box(*glacier.total_bounds)
    raster_box = box(*raster_bounds)

    return glacier_box.intersects(raster_box)


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

        with NamedTemporaryFile(suffix=".tif", delete=False) as projected_file:
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
