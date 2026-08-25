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

import collections
import logging
import re
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr
from pyproj import Transformer
from rasterio.enums import Resampling
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import box as _shapely_box

from pism_terra.vector import get_glacier_from_rgi_id
from pism_terra.workflow import (
    check_rio,
    check_xr_lazy,
    drop_geotransform_attr,
    stamp_grid_mapping,
)

logger = logging.getLogger(__name__)

# RGI o1 codes for which ITS_LIVE v2.1 publishes a per-region COG. 13/15/16 are
# absent because they're merged into the High Mountain Asia (14) mosaic.
_ITS_LIVE_REGION_CODES: tuple[str, ...] = (
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
    "14",
    "17",
    "18",
    "19",
)


#: RGI regions ITS_LIVE folds into another mosaic rather than publishing
#: separately: High Mountain Asia is one COG covering 13, 14 and 15/16.
_ITS_LIVE_REGION_ALIASES: dict[str, str] = {"13": "14", "15": "14", "16": "14"}


def region_code_from_rgi_id(rgi_id: str | None) -> str | None:
    """
    Read the ITS_LIVE region code out of an RGI identifier.

    The o1 region is part of the ID (``RGI2000-v7.0-C-03-01124`` is region
    ``03``), which is authoritative in a way a bounding box is not: the
    per-region COGs overlap generously in polar stereographic, so a box over
    Ellesmere Island falls inside both the Arctic Canada and the Greenland
    footprints and geometry alone cannot say which holds the data.

    Parameters
    ----------
    rgi_id : str or None
        RGI v7 identifier, or an aggregate name such as ``"S4F_AK"``.

    Returns
    -------
    str or None
        Two-digit region code with ITS_LIVE's aliases applied, or ``None``
        when *rgi_id* carries no region ITS_LIVE publishes.
    """
    if not rgi_id:
        return None
    match = re.search(r"-[CG]-(\d{2})-", str(rgi_id))
    if not match:
        return None
    code = _ITS_LIVE_REGION_ALIASES.get(match.group(1), match.group(1))
    return code if code in _ITS_LIVE_REGION_CODES else None


@lru_cache(maxsize=None)
def _its_live_region_footprint(region_code: str):
    """
    Return ``(crs, bounds_polygon)`` for an ITS_LIVE per-region COG.

    Opens the COG via ``/vsicurl/`` and reads only the header (no data
    transfer beyond a few KB). Cached so each region is probed once per
    process.

    Parameters
    ----------
    region_code : str
        Two-digit RGI o1 code (e.g. ``"01"`` for Alaska).

    Returns
    -------
    tuple
        ``(rasterio.crs.CRS, shapely.geometry.Polygon)`` — the COG's native
        CRS and its full extent as a rectangular polygon in that CRS.
    """
    url = (
        "/vsicurl/https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2.1/static/cog/"
        f"ITS_LIVE_velocity_120m_RGI{region_code}A_0000_V02.1_v.tif"
    )
    with rasterio.open(url) as src:
        b = src.bounds
        return src.crs, _shapely_box(b.left, b.bottom, b.right, b.top)


def _coverage(code: str, bounds: tuple[float, float, float, float], crs: str) -> float:
    """
    Fraction of ``bounds`` a region's ITS_LIVE COG covers.

    Parameters
    ----------
    code : str
        Two-digit RGI o1 region code.
    bounds : tuple of float
        ``(minx, miny, maxx, maxy)`` in the CRS given by ``crs``.
    crs : str
        CRS of ``bounds``.

    Returns
    -------
    float
        Covered fraction, 0.0 to 1.0.
    """
    cog_crs, cog_poly = _its_live_region_footprint(code)
    transformer = Transformer.from_crs(crs, cog_crs, always_xy=True)
    box = _shapely_box(*transformer.transform_bounds(*bounds))
    if box.area == 0:
        return 0.0
    return cog_poly.intersection(box).area / box.area


def region_code_from_bounds(
    bounds: tuple[float, float, float, float],
    crs: str,
    rgi_id: str | None = None,
) -> str:
    """
    Return the RGI region code whose ITS_LIVE COG covers ``bounds``.

    When ``rgi_id`` names a region ITS_LIVE publishes, that region is used —
    the ID is authoritative, whereas the footprints overlap generously in
    polar stereographic and a box can sit inside several of them. Otherwise
    each published COG header is probed (cached per process) and the region
    covering the most of ``bounds`` wins.

    Full containment is preferred but not required. A staged domain is the
    glacier plus a buffer plus a geographic pad, so it routinely overhangs a
    COG edge by a few kilometres even when the glacier itself is well inside;
    the interpolation downstream already yields NaN outside the tile and zeroes
    it. A shortfall is logged so genuinely uncovered domains are still visible.

    Parameters
    ----------
    bounds : tuple of float
        ``(minx, miny, maxx, maxy)`` in the CRS given by ``crs``.
    crs : str
        CRS of ``bounds`` (anything pyproj accepts: EPSG code, WKT, …).
    rgi_id : str or None, optional
        RGI identifier of the glacier being staged. Its o1 region selects the
        COG directly when ITS_LIVE publishes one for it.

    Returns
    -------
    str
        Two-digit RGI o1 region code (e.g. ``"01"``).

    Raises
    ------
    ValueError
        If no published region COG overlaps ``bounds`` at all.
    """
    preferred = region_code_from_rgi_id(rgi_id)
    if preferred is not None:
        covered = _coverage(preferred, bounds, crs)
        if covered > 0:
            if covered < 1:
                logger.warning(
                    "ITS_LIVE region %s covers %.1f%% of the requested domain; the rest has no "
                    "velocity data and is zeroed. This is normal when the domain buffer overhangs "
                    "a tile edge — check the glacier itself is inside if the shortfall is large.",
                    preferred,
                    100 * covered,
                )
            return preferred
        logger.warning(
            "ITS_LIVE region %s (from %s) does not overlap the domain at all; falling back to a footprint search.",
            preferred,
            rgi_id,
        )

    coverage = {code: _coverage(code, bounds, crs) for code in _ITS_LIVE_REGION_CODES}
    best = max(coverage, key=lambda code: coverage[code])
    if coverage[best] == 0:
        raise ValueError(
            f"No ITS_LIVE per-region COG overlaps bounds {bounds} (crs={crs}); "
            "the domain falls outside published coverage."
        )
    if coverage[best] < 1:
        logger.warning(
            "No ITS_LIVE COG fully contains the domain; using region %s, which covers %.1f%%. "
            "Pass the glacier's rgi_id to select the region directly.",
            best,
            100 * coverage[best],
        )
    return best


def get_velocities_by_bounds(
    bounds: tuple[float, float, float, float],
    product_name: str = "its_live",
    src_crs: str | None = None,
    rgi_id: str | None = None,
) -> xr.Dataset:
    """
    Retrieve and subset a velocity product over a specified geographic bounding box.

    This function fetches a global surface velocity dataset (e.g., ITS_LIVE) and returns a
    spatial subset clipped to the specified bounding box.

    Parameters
    ----------
    bounds : tuple of float
        Bounding box ``(minx, miny, maxx, maxy)`` in the CRS given by ``src_crs``.
    product_name : {"its_live"}, optional
        The name of the velocity product to query. Currently only "its_live" is supported.
        Default is "its_live".
    src_crs : str or None, optional
        CRS of ``bounds`` (e.g., ``"EPSG:3413"``). If ``None``, ``"EPSG:4326"``
        (longitude/latitude) is assumed.
    rgi_id : str or None, optional
        RGI identifier of the glacier, used to pick the ITS_LIVE region
        directly instead of inferring it from the footprints.

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

    # Define source CRS if not given.
    if src_crs is None:
        src_crs = "EPSG:4326"

    # Load dataset
    if product_name == "its_live":
        region_code = region_code_from_bounds(bounds, crs=src_crs, rgi_id=rgi_id)
        ds = get_itslive_velocities_by_region_code(region_code)
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


def get_itslive_velocities_by_region_code(
    region_code: str, components: list[str] = ["v", "vx", "vy", "vx_error", "vy_error", "landice"]
) -> xr.Dataset:
    """
    Load the global ITS_LIVE surface velocity mosaic as an xarray dataset.

    This function reads ITS_LIVE VRT-backed raster layers for specified velocity
    components using `rioxarray` with Dask chunking enabled for efficient access.

    Parameters
    ----------
    region_code : str
        Two-digit RGI o1 code (e.g. ``"01"`` for Alaska).
    components : list of str, optional
        List of velocity components to load. Valid entries include:
        - "v": velocity magnitude
        - "vx": x-component of velocity
        - "vy": y-component of velocity
        - "vx_error": x-component error
        - "vy_error": y-component error
        Defaults to ["v", "vx", "vy", "vx_error", "vy_error", "landice"].

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
    # Per-component CF metadata. Everything in this VRT family is m/yr except
    # the integer ``landice`` mask.
    component_attrs = {
        "v": {"units": "m year^-1", "long_name": "ice speed"},
        "vx": {"units": "m year^-1", "long_name": "x component of ice velocity"},
        "vy": {"units": "m year^-1", "long_name": "y component of ice velocity"},
        "vx_error": {"units": "m year^-1", "long_name": "x component error"},
        "vy_error": {"units": "m year^-1", "long_name": "y component error"},
        "landice": {"units": "1", "long_name": "land ice mask (1=ice)"},
    }
    dss = []
    for c in components:
        url = (
            "https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2.1/static/cog/"
            f"ITS_LIVE_velocity_120m_RGI{region_code}A_0000_V02.1_{c}.tif"
        )
        _ds = (
            rxr.open_rasterio(url, parse_coordinates=True, chunks={"x": 1024, "y": 1024}, masked=True)
            .isel(band=0)
            .drop_vars("band")
        )
        _ds.name = c
        # Drop junk band-level attrs that rioxarray surfaces from the COG.
        for k in ("scale_factor", "add_offset", "AREA_OR_POINT", "_FillValue"):
            _ds.attrs.pop(k, None)
        _ds.attrs.update(component_attrs.get(c, {}))
        dss.append(_ds)

    ds = xr.merge(dss, compat="no_conflicts")

    return ds


def glacier_velocities_from_grid(
    target_grid: xr.Dataset,
    geometries: collections.abc.Iterable,
    product_name: str = "its_live",
    path: Path | str = "tmp.nc",
    force_overwrite: bool = False,
    rgi_id: str | None = None,
) -> xr.Dataset:
    """
    Generate observed glacier surface velocities for a glacier by RGI ID.

    Extracts the glacier geometry, builds an extent, fetches a velocity
    product (e.g., ITS_LIVE) over that region, clips it to the glacier outline,
    and returns the result as an xarray dataset. A cached NetCDF at ``path`` is
    reused unless ``force_overwrite=True``.

    Parameters
    ----------
    target_grid : xarray.Dataset
        Target grid dataset whose ``x``/``y`` extent (in the grid's projected CRS,
        as recorded in ``spatial_ref``) defines the velocity query region. The
        velocity product is reprojected/aligned to this grid.
    geometries : iterable of shapely geometries
        Glacier outline(s) in ``target_grid``'s CRS. Used to clip the velocity
        dataset to the glacier footprint.
    product_name : str, default ``"its_live"``
        Velocity product to retrieve (e.g., ``"its_live"``). Passed to
        :func:`get_velocities_by_bounds`.
    path : str or pathlib.Path, default ``"tmp.nc"``
        Cache file for the clipped velocity dataset. When present and valid
        (per :func:`check_xr_lazy`), it is opened instead of re-downloading.
    force_overwrite : bool, default ``False``
        If ``True``, ignore any existing cache at ``path`` and regenerate.
    rgi_id : str or None, optional
        RGI identifier of the glacier. Its o1 region selects the ITS_LIVE tile
        directly, which the overlapping polar-stereographic footprints cannot
        do unambiguously.

    Returns
    -------
    xarray.Dataset
        Velocity dataset clipped to the glacier outline. Variable names depend
        on the source product but typically include components (e.g., ``u``,
        ``v`` or ``vx``, ``vy``) and possibly speed (e.g., ``v``). CRS is
        recorded via :mod:`rioxarray`.
    """

    print("")
    print("Generate Velocity Observations")
    print("-" * 120)

    EPS = 10.0

    if (not check_xr_lazy(path)) or force_overwrite:

        path = Path(path)
        path.unlink(missing_ok=True)

        xs = [float(target_grid.x.values[0]), float(target_grid.x.values[-1])]
        ys = [float(target_grid.y.values[0]), float(target_grid.y.values[-1])]
        bounds = (min(xs), min(ys), max(xs), max(ys))
        mapping_var = target_grid.rio.grid_mapping
        dst_crs = target_grid[mapping_var].attrs["crs_wkt"]
        t_geo = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)
        geo_bounds = t_geo.transform_bounds(*bounds)
        # Pad the geographic bbox so the clipped ITS_LIVE region fully covers
        # every target-grid point after round-tripping through 4326 → 3413.
        lon_pad = 0.25
        lat_pad = 0.1
        padded = (
            geo_bounds[0] - lon_pad,
            geo_bounds[1] - lat_pad,
            geo_bounds[2] + lon_pad,
            geo_bounds[3] + lat_pad,
        )
        ds = get_velocities_by_bounds(padded, product_name=product_name, rgi_id=rgi_id)

        # The interpolator is built on ITS_LIVE's native coordinates, so the
        # intermediate frame must match. The finite-difference round-trip
        # below recovers vector components aligned with target_grid's axes
        # (handling any rotation between the two CRSs).
        src_crs = ds.rio.crs
        t = Transformer.from_crs(dst_crs, src_crs, always_xy=True)
        t_inv = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        # Define DEM grid
        X, Y = np.meshgrid(target_grid.x, target_grid.y)

        # Project to ITSLive grid
        X_, Y_ = t.transform(X, Y)

        # Build ITSLive interpolants. ``bounds_error=False`` makes points that
        # land just outside the clipped tile (or in ITS_LIVE nodata regions)
        # fall back to NaN instead of raising; they're zeroed later.
        interpolator_vx = RegularGridInterpolator(
            (ds.y, ds.x), ds.vx.values.squeeze(), bounds_error=False, fill_value=np.nan
        )
        interpolator_vy = RegularGridInterpolator(
            (ds.y, ds.x), ds.vy.values.squeeze(), bounds_error=False, fill_value=np.nan
        )

        # Interpolate dem grid points
        vx_pts = interpolator_vx((Y_, X_))
        vy_pts = interpolator_vy((Y_, X_))

        # Finite difference displacement
        X_plus = X_ + EPS * vx_pts
        Y_plus = Y_ + EPS * vy_pts

        X_minus = X_ - EPS * vx_pts
        Y_minus = Y_ - EPS * vy_pts

        # Transform displaced points back to project grid
        X0_plus, Y0_plus = t_inv.transform(X_plus, Y_plus)
        X0_minus, Y0_minus = t_inv.transform(X_minus, Y_minus)

        # Calculate velocities
        vx = (X0_plus - X0_minus) / (2 * EPS)
        vy = (Y0_plus - Y0_minus) / (2 * EPS)

        # Reproject to the glacier's target CRS to match the PISM grid
        ds_clipped = ds.rio.reproject_match(target_grid, resampling=Resampling.bilinear).rio.clip(
            geometries, drop=False
        )
        # Snapshot ITS_LIVE's coverage from the reprojected/clipped ``v`` before
        # we overwrite ``vx``/``vy``/``v`` with the finite-difference field
        # (which is finite everywhere on the mesh and so erases the original
        # NaN pattern that the masks below depend on).
        v_missing = ds_clipped["v"].isnull().copy()
        ds_clipped["vx"].values = vx
        ds_clipped["vy"].values = vy

        # Zero out the velocity fields (and their per-component errors) off
        # ice. ITS_LIVE's ``landice`` is 1 over glacier ice and 0 elsewhere,
        # but the COG declares nodata=0 so reading with ``masked=True`` turns
        # off-ice cells into NaN. Test for on-ice (== 1) so both 0 and NaN
        # count as off-ice; ``where(cond, 0)`` keeps the value when cond is
        # true and writes 0 otherwise.
        if "landice" in ds_clipped:
            on_ice = ds_clipped["landice"] == 1
            for name in ("vx", "vy", "vx_error", "vy_error"):
                if name in ds_clipped:
                    ds_clipped[name] = ds_clipped[name].where(on_ice, 0)

        ds_clipped["v"].values = (ds_clipped["vx"].values ** 2 + ds_clipped["vy"].values ** 2) ** 0.5
        ds_clipped["u_observed"] = ds_clipped["vx"].fillna(0)
        ds_clipped["v_observed"] = ds_clipped["vy"].fillna(0)

        ds_clipped["zeta_fixed_mask"] = xr.where(v_missing, 1, 0).fillna(0).astype(int)
        ds_clipped["vel_misfit_weight"] = xr.where(v_missing, 0, 1).fillna(0).astype(int)
        ds_clipped["vel_misfit_weight"].attrs.update(
            {"units": "1", "long_name": "misfit weight (1=trust obs, 0=ignore)"}
        )
        ds_clipped["zeta_fixed_mask"].attrs.update({"units": "1", "long_name": "fixed zeta mask (1=no obs, fix prior)"})

        # Spatially constant basal-yield-stress prior for the inversion.
        ds_clipped["tauc_prior"] = xr.full_like(ds_clipped["v_observed"], 1.4e5)
        ds_clipped["tauc_prior"].attrs = {
            "units": "Pa",
            "long_name": "prior basal yield stress (constant)",
        }

        # Stamp CF metadata on the projected x/y coords (lost across some
        # rioxarray ops) and suppress the default ``_FillValue=NaN`` netCDF4
        # writes onto coordinate variables.
        ds_clipped["x"].attrs.update(
            {
                "standard_name": "projection_x_coordinate",
                "long_name": "x coordinate of projection",
                "units": "m",
                "axis": "X",
            }
        )
        ds_clipped["y"].attrs.update(
            {
                "standard_name": "projection_y_coordinate",
                "long_name": "y coordinate of projection",
                "units": "m",
                "axis": "Y",
            }
        )
        ds_clipped["x"].encoding["_FillValue"] = None
        ds_clipped["y"].encoding["_FillValue"] = None

        # Strip junk band metadata that rioxarray inherited from the source
        # COGs and propagated through reprojection.
        for name in ds_clipped.data_vars:
            for k in ("scale_factor", "add_offset", "AREA_OR_POINT"):
                ds_clipped[name].attrs.pop(k, None)
                ds_clipped[name].encoding.pop(k, None)

        # Re-attach the CRS + grid_mapping on every data_var. ``.where`` and
        # the ``u_observed``/``v_observed`` reconstructions drop the
        # ``grid_mapping`` encoding key, so only untouched vars (``v``,
        # ``landice``) would otherwise carry it through to the written file.
        mapping_var = target_grid.rio.grid_mapping
        crs = target_grid[mapping_var].attrs["crs_wkt"]
        ds_clipped = ds_clipped.rio.write_crs(crs).rio.write_grid_mapping().rio.write_coordinate_system()

        ds_clipped = stamp_grid_mapping(ds_clipped)
        # Drop the GeoTransform so GDAL/QGIS derive the (top-down) transform from
        # the ascending y coordinate instead of rendering the raster upside-down.
        drop_geotransform_attr(ds_clipped)
        ds_clipped.to_netcdf(path)

    else:
        ds_clipped = xr.open_dataset(path)
        mapping_var = target_grid.rio.grid_mapping
    crs = target_grid[mapping_var].attrs["crs_wkt"]
    ds_clipped = ds_clipped.rio.write_crs(crs).rio.write_grid_mapping().rio.write_coordinate_system()
    return ds_clipped


def bathymetry_from_grid(
    target_grid: xr.Dataset,
    uri: str,
    path: Path | str = "tmp.nc",
    force_overwrite: bool = False,
) -> xr.DataArray:
    """
    Build a glacier-domain bathymetry/elevation field from a cloud raster.

    Opens a remote raster (typically a Cloud Optimized GeoTIFF on S3 referenced
    via ``/vsis3/`` or ``/vsicurl/``), clips it to the geographic bounds of
    ``target_grid``, reprojects to ``target_grid``'s CRS/extent, and returns the
    result as a DataArray. A cached NetCDF at ``path`` is reused unless
    ``force_overwrite=True``.

    Parameters
    ----------
    target_grid : xarray.Dataset
        Target grid dataset whose ``x``/``y`` extent (in the grid's projected
        CRS, as recorded in ``spatial_ref``) defines the query region. The
        bathymetry raster is reprojected/aligned to this grid.
    uri : str
        Path/URI of the source raster. Local paths and GDAL VSI URIs are both
        accepted (e.g., ``"/vsis3/bucket/key.tif"`` or
        ``"/vsicurl/https://.../bathymetry.tif"``).
    path : str or pathlib.Path, default ``"tmp.nc"``
        Cache file for the clipped/reprojected output. When present and valid
        (per :func:`check_rio`), it is opened instead of re-fetching.
    force_overwrite : bool, default ``False``
        If ``True``, ignore any existing cache at ``path`` and regenerate.

    Returns
    -------
    xarray.DataArray
        Bathymetry/elevation values (float32, meters) on ``target_grid`` with
        CRS attached via :mod:`rioxarray`.
    """

    print("")
    print("Generate Bathymetry")
    print("-" * 120)

    if (not check_rio(path)) or force_overwrite:

        path = Path(path)
        path.unlink(missing_ok=True)

        bounds = [target_grid.x.values[0], target_grid.y.values[0], target_grid.x.values[-1], target_grid.y.values[-1]]
        mapping_var = target_grid.rio.grid_mapping
        dst_crs = target_grid[mapping_var].attrs["crs_wkt"]
        t = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)
        geo_bounds = t.transform_bounds(*bounds)

        da = rxr.open_rasterio(uri, masked=True, chunks={"x": 1024, "y": 1024}).squeeze()
        # GEBCO is a global EPSG:4326 COG; if the target grid wraps the
        # antimeridian (pyproj returns xmin > xmax, e.g. RGI region 01),
        # clip the two 4326 halves separately, reproject each into the
        # projected target grid (continuous across the seam), and coalesce.
        if geo_bounds[0] > geo_bounds[2]:
            west = da.rio.clip_box(geo_bounds[0], geo_bounds[1], 180.0, geo_bounds[3], crs=da.rio.crs)
            east = da.rio.clip_box(-180.0, geo_bounds[1], geo_bounds[2], geo_bounds[3], crs=da.rio.crs)
            west_reproj = west.rio.reproject_match(target_grid, resampling=Resampling.bilinear)
            east_reproj = east.rio.reproject_match(target_grid, resampling=Resampling.bilinear)
            out = west_reproj.fillna(east_reproj).astype("float32")
        else:
            sub = da.rio.clip_box(*geo_bounds, crs=da.rio.crs)
            out = sub.rio.reproject_match(target_grid, resampling=Resampling.bilinear).astype("float32")
        out.encoding = {}  # drop stale int16 dtype/fill from the source COG
        out = out.rio.write_crs(dst_crs).rio.write_grid_mapping()
        out.name = "bathymetry"
        # Strip stale per-band attrs that confuse xarray on re-read
        for k in ("scale_factor", "add_offset", "AREA_OR_POINT"):
            out.attrs.pop(k, None)
        # Drop the GeoTransform so GDAL/QGIS derive the (top-down) transform from
        # the ascending y coordinate instead of rendering the raster upside-down.
        drop_geotransform_attr(out)
        out.to_netcdf(path)

    else:
        out = xr.open_dataset(path)["bathymetry"]
    return out
