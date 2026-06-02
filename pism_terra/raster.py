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

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import xarray as xr
from geocube.api.core import make_geocube
from pyproj import CRS, Transformer
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
    encoding.update({var: {"_FillValue": None} for var in list(ds.data_vars) + list(ds.coords)})

    ds.to_netcdf(output_file, encoding=encoding, engine="h5netcdf")

    return output_file


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


def local_scale_factor(
    crs,
    x,
    y,
    eps_deg: float = 1.0e-6,
):
    """
    Local linear scale factor of a projected CRS at a point.

    Returns the dimensionless number ``k`` such that **1 metre of true
    ground displacement at ``(x, y)`` corresponds to ``k`` metres measured
    in ``crs`` map units**. ``k = 1`` for an isometric projection (e.g.
    polar stereographic near its true-scale latitude); ``k > 1`` where the
    projection inflates distances (e.g. Web Mercator at high latitudes;
    ``k ≈ 1/cos(lat)`` there); ``k < 1`` where it shrinks them.

    The factor is computed as ``sqrt(|det J|)``, where ``J`` is the
    Jacobian of the *true-to-map* coordinate transform at the point:

    .. math::

        J = \\begin{pmatrix}
            \\partial x_\\mathrm{crs} / \\partial x_\\mathrm{true} &
            \\partial x_\\mathrm{crs} / \\partial y_\\mathrm{true} \\\\
            \\partial y_\\mathrm{crs} / \\partial x_\\mathrm{true} &
            \\partial y_\\mathrm{crs} / \\partial y_\\mathrm{true}
        \\end{pmatrix}

    The Jacobian is estimated by finite-differencing the
    geographic-to-map transform a small distance from the point. The
    geographic step (in degrees) is converted to a true-metres step using
    the **prime-vertical radius of curvature** ``N(lat)`` along longitude
    and the **meridional radius of curvature** ``M(lat)`` along latitude,
    both evaluated on the source CRS's own ellipsoid (no spherical-Earth
    approximation):

    .. math::

        e^2 = 2f - f^2, \\quad
        N(\\varphi) = \\frac{a}{\\sqrt{1 - e^2 \\sin^2 \\varphi}}, \\quad
        M(\\varphi) = \\frac{a (1 - e^2)}{(1 - e^2 \\sin^2 \\varphi)^{3/2}}

    where ``a`` is the ellipsoid's semi-major axis and ``f`` its
    flattening. One degree of latitude corresponds to ``M·π/180`` metres;
    one degree of longitude at latitude ``φ`` corresponds to
    ``N·cos(φ)·π/180`` metres. Using these to normalise the Jacobian gives
    ``det J`` in units of ``m_crs² / m_true²``; its square root is the
    scalar scale factor.

    For **conformal** CRSs (Mercator, polar stereographic, UTM,
    Lambert conformal conic, …) the projection is locally isotropic, so
    ``k_x = k_y = k`` — the single returned number fully describes the
    distortion. For **non-conformal** CRSs (Albers equal-area, …)
    ``k_x ≠ k_y``; ``sqrt(|det J|)`` is the area-equivalent (geometric
    mean) scale and is good enough for vector-magnitude corrections.

    Parameters
    ----------
    crs : str or pyproj.CRS
        Source CRS for the velocity (or other vector) field. Anything
        :class:`pyproj.CRS` accepts: ``"EPSG:3857"``, WKT, PROJ4, etc.
    x, y : float or numpy.ndarray
        Position(s) in ``crs`` units (typically metres) where the scale
        factor is evaluated. Scalars or broadcastable arrays.
    eps_deg : float, default ``1.0e-6``
        Geographic step (degrees) used for the numerical Jacobian. The
        default keeps the truncation error below 1 part in 10⁵ for the
        projections used in glaciology while staying well above floating
        -point precision.

    Returns
    -------
    float or numpy.ndarray
        Local scale factor ``k`` (dimensionless). Same shape as ``x``/``y``.

    Notes
    -----
    **How to use this to correct a finite-difference velocity transform.**

    Vector products like ITS_LIVE store velocity components as the
    *true ground motion* in m/yr, projected onto the raster CRS's
    coordinate axes. The advect-and-roundtrip pattern used by
    :func:`pism_terra.glacier.observations.glacier_velocities_from_grid`,
    however, implicitly assumes the components describe the *rate of
    change of the map coordinate* (i.e. ``v = dx_crs/dt``). The two
    conventions agree when the source CRS has unit scale factor at the
    point of interest (polar stereographic over Greenland or Antarctica
    near 70°S/70°N, UTM near its central meridian), but diverge wherever
    ``k != 1`` — most visibly with Web Mercator (EPSG:3857), where
    ``k = 1/cos(φ) ≈ 2`` at 60° latitude.

    To make the FD round-trip recover the right magnitude regardless of
    the source CRS, pre-multiply the source-side velocity by the local
    scale factor before advecting:

    .. code-block:: python

        from pism_terra.raster import local_scale_factor

        k = local_scale_factor(src_crs, X_, Y_)   # X_, Y_ in src_crs metres
        vx_pts *= k
        vy_pts *= k
        # ... continue with the existing FD round-trip; the result is now
        # in m/yr aligned with the destination CRS axes, independent of
        # the source CRS's local distortion.

    Equivalently, divide by ``k`` *after* the round-trip. For an isometric
    source (``k ≈ 1``) the correction is a no-op, so it's safe to apply
    unconditionally as a defensive measure.

    Examples
    --------
    Mercator (EPSG:3857) inflates distances by ``1/cos(φ)``. At about 60° N
    the local scale factor is therefore close to 2:

    >>> from pism_terra.raster import local_scale_factor
    >>> round(local_scale_factor("EPSG:3857", 0.0, 8362900.0), 2)
    1.99

    Polar stereographic NSIDC (EPSG:3413) has true scale at 70° N, so the
    factor is close to 1 over most of the Greenland Ice Sheet:

    >>> round(local_scale_factor("EPSG:3413", 0.0, -1_000_000.0), 2)
    0.98

    Vectorised call — pass arrays of points and get an array back:

    >>> import numpy as np
    >>> ks = local_scale_factor("EPSG:3857", np.zeros(3), np.linspace(0, 8e6, 3))
    >>> [round(float(k), 2) for k in ks]
    [1.0, 1.2, 1.89]
    """
    crs_obj = CRS(crs)
    ellps = crs_obj.geodetic_crs.ellipsoid
    a = ellps.semi_major_metre
    inv_f = ellps.inverse_flattening
    f = 1.0 / inv_f if inv_f and not math.isinf(inv_f) else 0.0
    e2 = 2.0 * f - f * f

    to_geo = Transformer.from_crs(crs_obj, "EPSG:4326", always_xy=True)
    from_geo = Transformer.from_crs("EPSG:4326", crs_obj, always_xy=True)

    lon, lat = to_geo.transform(x, y)

    # Numerical Jacobian d(crs) / d(lon, lat) in m_crs / deg.
    xp, yp = from_geo.transform(np.add(lon, eps_deg), lat)
    xm, ym = from_geo.transform(np.subtract(lon, eps_deg), lat)
    xyp, yyp = from_geo.transform(lon, np.add(lat, eps_deg))
    xym, yym = from_geo.transform(lon, np.subtract(lat, eps_deg))

    j_xlon = (np.asarray(xp) - np.asarray(xm)) / (2.0 * eps_deg)
    j_ylon = (np.asarray(yp) - np.asarray(ym)) / (2.0 * eps_deg)
    j_xlat = (np.asarray(xyp) - np.asarray(xym)) / (2.0 * eps_deg)
    j_ylat = (np.asarray(yyp) - np.asarray(yym)) / (2.0 * eps_deg)

    # Convert deg → true-metres using the local radii of curvature.
    sin_lat = np.sin(np.radians(lat))
    cos_lat = np.cos(np.radians(lat))
    w = np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    n_prime = a / w  # prime-vertical radius (m)
    m_meridional = a * (1.0 - e2) / w**3  # meridional radius (m)
    deg_to_rad = math.pi / 180.0
    m_per_deg_lon = n_prime * cos_lat * deg_to_rad
    m_per_deg_lat = m_meridional * deg_to_rad

    # det of the Jacobian in (m_crs / m_true)^2.
    det = np.abs(j_xlon * j_ylat - j_xlat * j_ylon) / (m_per_deg_lon * m_per_deg_lat)

    k = np.sqrt(det)
    return float(k) if np.isscalar(x) and np.isscalar(y) else k
