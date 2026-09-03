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
import zipfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed as cf_as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import box as _shapely_box
from tqdm.auto import tqdm

from pism_terra.aws import download_from_s3, project_prefix
from pism_terra.download import extract_archive
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


# ---------------------------------------------------------------------------
# Observed surface elevation change (dh), Hugonnet et al. (2021)
# ---------------------------------------------------------------------------

#: Source archives of the supported dh datasets. The Hugonnet archive is a
#: zip of per-period zips of 1°x1° GeoTIFF tiles (100 m, UTM, metres of
#: elevation change over the period) plus matching ``_err_dh_`` error tiles.
DH_DATASETS: dict[str, str] = {"hugonnet": "s3://pism-cloud-data/glacier/hugonnet/mb_rgi7.zip"}

#: Interval of the dh extraction; matches the run-script postprocessing
#: (``pism_terra.glacier.run.DH_START``/``DH_END``).
DH_START = "2000-01-01"
DH_END = "2020-01-01"

_DH_CITATION = (
    "Hugonnet et al. (2021), Accelerated global glacier mass loss in the "
    "early twenty-first century (doi:10.1038/s41586-021-03436-z)"
)

#: Tile-name pattern: ``N60W139_dh_2000-01-01_2020-01-01.tif`` names the
#: tile's lower-left (south-west) corner, like SRTM granules.
_DH_TILE_RE = re.compile(r"^(?P<ns>[NS])(?P<lat>\d{2})(?P<ew>[EW])(?P<lon>\d{3})_dh_")


def _dh_region_dir(rgi_id: str) -> str:
    """
    Per-region output subdirectory for a complex, own name for an aggregate.

    Parameters
    ----------
    rgi_id : str
        RGI7 complex identifier or aggregate name (e.g. ``"S4F_AK"``).

    Returns
    -------
    str
        ``"RGI2000-v7.0-C-01"`` style region directory, or the aggregate's
        own name — the same convention the ice-thickness rasters use.
    """
    return "-".join(rgi_id.split("-")[:-1]) or rgi_id


def ensure_dh_tiles(
    extract_path: Path | str,
    source_uri: str,
    start: str = DH_START,
    end: str = DH_END,
    force_overwrite: bool = False,
) -> Path:
    """
    Download the dh archive and extract one period's tiles.

    The outer archive holds one deflated zip per period, so tiles cannot be
    range-read from S3; the requested period is extracted once into
    ``extract_path`` and reused afterwards.

    Parameters
    ----------
    extract_path : str or pathlib.Path
        Working directory for the archive and the extracted tiles.
    source_uri : str
        S3 URI of the outer archive (see :data:`DH_DATASETS`).
    start, end : str
        Period bounds naming the inner zip (``{start}_{end}.zip``).
    force_overwrite : bool, default False
        Re-download and re-extract even when the tiles are already present.

    Returns
    -------
    pathlib.Path
        Directory holding the period's tiles (per-UTM-zone subdirectories).

    Raises
    ------
    KeyError
        If the archive holds no zip for the requested period.
    """
    extract_path = Path(extract_path)
    extract_path.mkdir(parents=True, exist_ok=True)
    period = f"{start}_{end}"
    tiles_dir = extract_path / period
    if tiles_dir.is_dir() and any(tiles_dir.rglob("*_dh_*.tif")) and not force_overwrite:
        logger.info("Using extracted dh tiles in %s", tiles_dir)
        return tiles_dir

    archive = extract_path / Path(source_uri).name
    if not archive.exists() or force_overwrite:
        logger.info("Downloading dh archive from %s", source_uri)
        download_from_s3(source_uri, archive)

    with zipfile.ZipFile(archive) as outer:
        inner_names = [n for n in outer.namelist() if n.endswith(f"{period}.zip")]
        if not inner_names:
            periods = sorted(n for n in outer.namelist() if n.endswith(".zip"))
            raise KeyError(f"{archive} holds no period {period!r}; available: {periods}")
        logger.info("Extracting %s from %s", inner_names[0], archive)
        inner_zip = Path(outer.extract(inner_names[0], extract_path))

    extract_archive(inner_zip, extract_path, force_overwrite=force_overwrite)
    return tiles_dir


def dh_tile_index(tiles_dir: Path | str) -> list[tuple[Path, Path | None, Any]]:
    """
    Index a period's dh tiles by their geographic footprint.

    The 1°x1° footprint is parsed from the file name (the tile's south-west
    corner), so no tile has to be opened; the per-complex overlap test is a
    cheap geometry intersection in EPSG:4326.

    Parameters
    ----------
    tiles_dir : str or pathlib.Path
        Directory holding the extracted tiles (from :func:`ensure_dh_tiles`).

    Returns
    -------
    list of tuple
        ``(dh_path, err_path, footprint)`` triples, ``err_path`` being
        ``None`` when a tile ships without its error companion.
    """
    index = []
    for dh_path in sorted(Path(tiles_dir).rglob("*_dh_*.tif")):
        match = _DH_TILE_RE.match(dh_path.name)
        if not match:
            continue
        lat = int(match["lat"]) * (1 if match["ns"] == "N" else -1)
        lon = int(match["lon"]) * (1 if match["ew"] == "E" else -1)
        err_path = dh_path.with_name(dh_path.name.replace("_dh_", "_err_dh_", 1))
        index.append((dh_path, err_path if err_path.exists() else None, _shapely_box(lon, lat, lon + 1, lat + 1)))
    return index


def build_dh_raster(
    rgi_id: str,
    geometry_4326,
    dst_crs: str,
    tile_index: list[tuple[Path, Path | None, Any]],
    output_file: Path | str,
    resolution: float = 100.0,
    pad: float = 1_000.0,
) -> Path | None:
    """
    Mosaic the dh tiles overlapping one complex into a two-band COG.

    Every overlapping tile is reprojected from its UTM zone into a window of
    the complex's grid and accumulated first-wins (adjacent zones carry
    near-identical values along their shared edge). Band 1 is dh (m over the
    period), band 2 the 1-sigma error; nodata is NaN, like the source tiles.

    Parameters
    ----------
    rgi_id : str
        Complex identifier, for logging.
    geometry_4326 : shapely.geometry.base.BaseGeometry
        Complex outline in EPSG:4326.
    dst_crs : str
        Target CRS of the output raster (the complex's staging CRS).
    tile_index : list of tuple
        Output of :func:`dh_tile_index`.
    output_file : str or pathlib.Path
        Target GeoTIFF path, parent directories created as needed.
    resolution : float, default 100.0
        Output grid spacing in metres (the source tiles' native resolution).
    pad : float, default 1000.0
        Margin added around the outline's bounds, in metres.

    Returns
    -------
    pathlib.Path or None
        The written file, or ``None`` when no tile overlaps the outline.
    """
    # A small geographic pad keeps tiles that only graze the outline after
    # the 4326 round trip.
    probe = geometry_4326.buffer(0.05)
    tiles = [(dh, err) for dh, err, footprint in tile_index if probe.intersects(footprint)]
    if not tiles:
        logger.warning("No dh tiles overlap complex %s; skipping", rgi_id)
        return None

    geom = gpd.GeoSeries([geometry_4326], crs="EPSG:4326").to_crs(dst_crs).iloc[0]
    minx, miny, maxx, maxy = geom.bounds
    minx = np.floor((minx - pad) / resolution) * resolution
    miny = np.floor((miny - pad) / resolution) * resolution
    maxx = np.ceil((maxx + pad) / resolution) * resolution
    maxy = np.ceil((maxy + pad) / resolution) * resolution
    width = int(round((maxx - minx) / resolution))
    height = int(round((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)

    bands = [np.full((height, width), np.nan, dtype=np.float32) for _ in range(2)]
    for dh_path, err_path in tiles:
        for band, path in zip(bands, (dh_path, err_path)):
            if path is None:
                continue
            da = rxr.open_rasterio(path, masked=True).sel(band=1).drop_vars("band")
            # Reproject into the tile's window of the output grid only; a
            # large aggregate's full grid would cost hundreds of MB per tile.
            tb = gpd.GeoSeries([_shapely_box(*da.rio.bounds())], crs=da.rio.crs).to_crs(dst_crs).total_bounds
            col0 = max(0, int((tb[0] - minx) // resolution) - 2)
            row0 = max(0, int((maxy - tb[3]) // resolution) - 2)
            col1 = min(width, int(np.ceil((tb[2] - minx) / resolution)) + 2)
            row1 = min(height, int(np.ceil((maxy - tb[1]) / resolution)) + 2)
            if col0 >= col1 or row0 >= row1:
                continue
            window_transform = from_origin(minx + col0 * resolution, maxy - row0 * resolution, resolution, resolution)
            reprojected = da.rio.reproject(
                dst_crs,
                shape=(row1 - row0, col1 - col0),
                transform=window_transform,
                resampling=Resampling.bilinear,
                nodata=np.nan,
            ).values.astype(np.float32)
            window = band[row0:row1, col0:col1]
            fill = np.isnan(window) & np.isfinite(reprojected)
            window[fill] = reprojected[fill]

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "driver": "COG",
        "dtype": "float32",
        "count": 2,
        "height": height,
        "width": width,
        "crs": dst_crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "DEFLATE",
        "predictor": 3,
        "blocksize": 512,
        "overview_resampling": "AVERAGE",
        "BIGTIFF": "YES",
        "num_threads": "ALL_CPUS",
    }
    with rasterio.open(output_file, "w", **meta) as dst:
        for idx, (band, description) in enumerate(zip(bands, ("dh", "err_dh")), start=1):
            dst.write(band, idx)
            dst.set_band_description(idx, description)
    return output_file


def prepare_dh_hugonnet(
    complexes: gpd.GeoDataFrame,
    output_path: Path | str,
    extract_path: Path | str,
    source_uri: str | None = None,
    start: str = DH_START,
    end: str = DH_END,
    ntasks: int = 8,
    force_overwrite: bool = False,
) -> None:
    """
    Clip the Hugonnet dh tiles into per-complex rasters for one project.

    The counterpart of ``prepare_ice_thickness_maffezzoli`` for the observed
    2000-2020 surface elevation change: the global tile archive is extracted
    once, then every complex (aggregates included — their outlines carry a
    geometry and CRS like any other row) gets a two-band ``{rgi_id}_dh.tif``
    under ``output_path/<region>/``, ready to sync to S3 for
    :func:`add_dh_observations` to fetch at stage time.

    Parameters
    ----------
    complexes : geopandas.GeoDataFrame
        Complex outlines with ``rgi_id``, ``crs``, and ``geometry`` columns.
    output_path : str or pathlib.Path
        Root directory for the per-complex rasters.
    extract_path : str or pathlib.Path
        Working directory for the archive and its extracted tiles.
    source_uri : str or None, optional
        S3 URI of the tile archive; defaults to
        ``DH_DATASETS["hugonnet"]``.
    start, end : str
        Period to extract, naming the inner zip of the archive.
    ntasks : int, default 8
        Maximum number of parallel workers for the per-complex clips.
    force_overwrite : bool, default False
        Re-download, re-extract, and re-clip everything.
    """
    output_path = Path(output_path)
    tiles_dir = ensure_dh_tiles(
        extract_path,
        source_uri or DH_DATASETS["hugonnet"],
        start=start,
        end=end,
        force_overwrite=force_overwrite,
    )
    tile_index = dh_tile_index(tiles_dir)
    logger.info("Indexed %d dh tiles for period %s_%s", len(tile_index), start, end)

    complexes_4326 = complexes.to_crs("EPSG:4326")

    def _one(rgi_id: str, geometry_4326, dst_crs: str) -> Path | None:
        """
        Clip one complex, reusing an existing raster unless forced.

        Parameters
        ----------
        rgi_id : str
            Complex identifier.
        geometry_4326 : shapely.geometry.base.BaseGeometry
            Complex outline in EPSG:4326.
        dst_crs : str
            Target CRS of the output raster.

        Returns
        -------
        pathlib.Path or None
            The per-complex raster, or ``None`` when no tile overlaps.
        """
        output_file = output_path / _dh_region_dir(rgi_id) / f"{rgi_id}_dh.tif"
        if output_file.exists() and not force_overwrite:
            return output_file
        return build_dh_raster(rgi_id, geometry_4326, dst_crs, tile_index, output_file)

    tasks = []
    for _, row in complexes_4326.iterrows():
        dst_crs = row.get("crs")
        if not isinstance(dst_crs, str) or not dst_crs:
            logger.warning("Complex %s has no CRS; skipping dh prep", row["rgi_id"])
            continue
        tasks.append((row["rgi_id"], row.geometry, dst_crs))

    failed: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=min(ntasks, max(1, len(tasks)))) as executor:
        futures = {executor.submit(_one, *task): task[0] for task in tasks}
        for future in tqdm(cf_as_completed(futures), total=len(futures), desc="Clipping dh"):
            rgi_id = futures[future]
            try:
                future.result()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                failed.append((rgi_id, exc))
    for rgi_id, err in failed:
        logger.error("Failed clipping dh for %s: %s", rgi_id, err)
    logger.info("dh clipping complete")


def fetch_dh_raster(
    rgi_id: str,
    cache_dir: Path | str,
    dataset: str = "hugonnet",
    bucket: str = "pism-cloud-data",
    prefix: str = "glacier/input",
    project_directory: str | None = None,
    force_overwrite: bool = False,
) -> Path | None:
    """
    Fetch a complex's prepared dh raster from S3, with a local cache.

    Parameters
    ----------
    rgi_id : str
        Complex identifier, naming the raster to fetch.
    cache_dir : str or pathlib.Path
        Directory the raster is cached in.
    dataset : str, default ``"hugonnet"``
        Key of :data:`DH_DATASETS` selecting the dh product.
    bucket : str, default ``"pism-cloud-data"``
        S3 bucket holding the prepared rasters.
    prefix : str, default ``"glacier/input"``
        Shared key prefix in the bucket.
    project_directory : str or None, optional
        Project subdirectory under ``prefix`` (e.g. ``"s4f"``).
    force_overwrite : bool, default False
        Re-fetch even when a cached copy exists.

    Returns
    -------
    pathlib.Path or None
        The local raster, or ``None`` when the complex has no prepared
        raster (complexes outside the observed tiles).

    Raises
    ------
    NotImplementedError
        If *dataset* is not a known dh dataset.
    """
    if dataset not in DH_DATASETS:
        raise NotImplementedError(f"Unknown dh dataset {dataset!r}; available: {sorted(DH_DATASETS)}")

    local_tif = Path(cache_dir) / f"{rgi_id}_dh.tif"
    if local_tif.exists() and not force_overwrite:
        return local_tif
    s3_uri = f"s3://{bucket}/{project_prefix(prefix, project_directory)}/dh/{dataset}/{_dh_region_dir(rgi_id)}/{rgi_id}_dh.tif"
    print(f"Downloading dh from {s3_uri}", flush=True)
    try:
        download_from_s3(s3_uri, local_tif)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Complexes outside the observed tiles (e.g. tiny Brooks Range
        # glaciers) have no prepared raster; callers must not die on them.
        local_tif.unlink(missing_ok=True)
        logger.warning("No dh raster for %s (%s: %s)", rgi_id, s3_uri, exc)
        return None
    return local_tif


def dh_from_tif(
    tif: Path | str,
    target_grid: xr.Dataset,
    geometries: collections.abc.Iterable,
    start: str = DH_START,
    end: str = DH_END,
) -> xr.Dataset:
    """
    Reproject a per-complex dh raster onto the target grid.

    Parameters
    ----------
    tif : str or pathlib.Path
        Two-band raster from :func:`build_dh_raster` (band 1 dh, band 2
        error).
    target_grid : xarray.Dataset
        Grid (with a projected CRS in ``spatial_ref``) the fields are aligned
        to.
    geometries : iterable of shapely geometries
        Glacier outline(s) in ``target_grid``'s CRS; cells outside stay NaN.
    start, end : str
        Period bounds, recorded in the variable attributes.

    Returns
    -------
    xarray.Dataset
        Variables ``dh`` and ``dh_err`` (metres, NaN where unobserved or
        outside the outline) on ``target_grid``.
    """
    da = rxr.open_rasterio(tif, masked=True)
    fields = {}
    for band, name in ((1, "dh"), (2, "dh_err")):
        field = da.sel(band=band).drop_vars("band")
        field = field.rio.reproject_match(target_grid, resampling=Resampling.bilinear, nodata=np.nan)
        field = field.rio.clip(geometries, drop=False)
        field.attrs = {}
        for key in ("scale_factor", "add_offset", "AREA_OR_POINT", "_FillValue"):
            field.encoding.pop(key, None)
        fields[name] = field
    fields["dh"].attrs = {
        "units": "m",
        "long_name": f"observed ice surface elevation change {start} to {end}",
        "source": _DH_CITATION,
    }
    fields["dh_err"].attrs = {
        "units": "m",
        "long_name": f"1-sigma error of the observed ice surface elevation change {start} to {end}",
        "source": _DH_CITATION,
    }
    return xr.Dataset(fields)


def add_dh_observations(
    obs_file: Path | str,
    target_grid: xr.Dataset,
    geometries: collections.abc.Iterable,
    rgi_id: str,
    dataset: str = "hugonnet",
    staging_path: Path | str | None = None,
    bucket: str = "pism-cloud-data",
    prefix: str = "glacier/input",
    project_directory: str | None = None,
    force_overwrite: bool = False,
) -> xr.Dataset:
    """
    Add the observed 2000-2020 elevation change to a staged obs file.

    Fetches the pre-clipped per-complex raster (from
    :func:`prepare_dh_hugonnet`, synced to
    ``s3://{bucket}/{prefix}/[{project_directory}/]dh/{dataset}/<region>/{rgi_id}_dh.tif``),
    reprojects it onto the target grid, and rewrites ``obs_file`` with the
    ``dh`` and ``dh_err`` variables added. A file that already carries ``dh``
    is left untouched unless ``force_overwrite`` is set.

    Parameters
    ----------
    obs_file : str or pathlib.Path
        The staged observations NetCDF (``obs_{rgi_id}.nc``), rewritten in
        place.
    target_grid : xarray.Dataset
        Grid (with a projected CRS in ``spatial_ref``) the fields are aligned
        to.
    geometries : iterable of shapely geometries
        Glacier outline(s) in ``target_grid``'s CRS.
    rgi_id : str
        Complex identifier, naming the raster to fetch.
    dataset : str, default ``"hugonnet"``
        Key of :data:`DH_DATASETS` selecting the dh product.
    staging_path : str or pathlib.Path or None, optional
        Directory the fetched raster is cached in. Defaults to the directory
        of ``obs_file``.
    bucket : str, default ``"pism-cloud-data"``
        S3 bucket holding the prepared rasters.
    prefix : str, default ``"glacier/input"``
        Shared key prefix in the bucket.
    project_directory : str or None, optional
        Project subdirectory under ``prefix`` (e.g. ``"s4f"``).
    force_overwrite : bool, default False
        Re-fetch the raster and rewrite the variables even when present.

    Returns
    -------
    xarray.Dataset
        The updated observations dataset.

    Raises
    ------
    NotImplementedError
        If *dataset* is not a known dh dataset.
    """
    if dataset not in DH_DATASETS:
        raise NotImplementedError(f"Unknown dh dataset {dataset!r}; available: {sorted(DH_DATASETS)}")

    print("")
    print(f"Add dh Observations ({dataset}) for {rgi_id}")
    print("-" * 120)

    obs_file = Path(obs_file)
    with xr.open_dataset(obs_file) as obs:
        if "dh" in obs.data_vars and not force_overwrite:
            logger.info("%s already carries dh; skipping", obs_file)
            return obs.load()
        obs = obs.load()

    cache_dir = Path(staging_path) if staging_path is not None else obs_file.parent
    local_tif = fetch_dh_raster(
        rgi_id,
        cache_dir,
        dataset=dataset,
        bucket=bucket,
        prefix=prefix,
        project_directory=project_directory,
        force_overwrite=force_overwrite,
    )
    if local_tif is None:
        logger.warning("Obs file %s left without dh", obs_file)
        return obs

    # Read the CRS off the grid-mapping variable and write it back onto the
    # grid: a grid re-opened from disk carries ``crs_wkt`` but does not
    # always expose it through ``rio.crs``, and ``reproject_match`` needs it.
    mapping_var = target_grid.rio.grid_mapping
    crs = target_grid[mapping_var].attrs["crs_wkt"]
    target_grid = target_grid.rio.write_crs(crs).rio.set_spatial_dims(x_dim="x", y_dim="y")

    ds_dh = dh_from_tif(local_tif, target_grid, geometries)
    # ``spatial_ref`` can come back as a data variable on one side and a
    # coordinate on the other, which makes the merge ambiguous; drop it and
    # let ``write_crs`` below restore a single canonical copy.
    obs = obs.drop_vars("spatial_ref", errors="ignore")
    for name in ds_dh.data_vars:
        obs[name] = ds_dh[name].drop_vars("spatial_ref", errors="ignore")

    obs = obs.rio.write_crs(crs).rio.write_grid_mapping().rio.write_coordinate_system()
    obs = stamp_grid_mapping(obs)
    drop_geotransform_attr(obs)
    obs_file.unlink(missing_ok=True)
    obs.to_netcdf(obs_file)
    logger.info("dh observations added to %s", obs_file)
    return obs


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
