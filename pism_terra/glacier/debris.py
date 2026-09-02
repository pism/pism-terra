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

# pylint: disable=too-many-positional-arguments

"""
Supraglacial debris thickness from NSIDC HMA_DTE (Rounce et al.).

The *Global Glacier Debris Thickness Estimates and Sub-Debris Melt Factors,
Version 1* collection (NSIDC short name ``HMA_DTE``, DOI
10.5067/8DQKWY03KJWT) ships one granule per RGI 6.0 glacier, each holding a
debris-thickness GeoTIFF (``*_hdts_m.tif``, meters) and a sub-debris
melt-enhancement-factor GeoTIFF (``*_meltfactor.tif``); glaciers without
direct estimates carry ``*_extrap`` variants instead.

RGI6 and RGI7 numbering schemes are unrelated and the RGI7-shipped
``rgi6_links.csv`` files are a sparse subset (see the Frank ice-thickness
prep), so granules are matched to an RGI7 complex *spatially*: the CMR query
itself is scoped to the glacier domain's geographic bounding box, and the
mosaic is clipped to the complex outline afterwards.
"""

import collections
import logging
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse

import earthaccess
import numpy as np
import rioxarray as rxr
import xarray as xr
from pyproj import Transformer
from rasterio.warp import Resampling

from pism_terra.workflow import (
    check_xr_lazy,
    drop_geotransform_attr,
    stamp_grid_mapping,
)

xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)

# NSIDC collection identifiers for the supported debris dataset.
DEBRIS_DATASETS = {"rounce": {"short_name": "HMA_DTE", "version": "1"}}

# Filename suffix per output variable, preferred (direct estimate) first.
# ``_extrap`` marks glaciers whose fields were extrapolated from regional
# relations rather than inverted from observations.
_VARIABLE_SUFFIXES: Mapping[str, tuple[str, ...]] = {
    "debris_thickness": ("_hdts_m.tif", "_hdts_m_extrap.tif"),
    "debris_melt_factor": ("_meltfactor.tif", "_meltfactor_extrap.tif"),
}

# The estimates integrate observations over 2000-2018; the single record sits
# at the midpoint of that period.
_PERIOD_START = np.datetime64("2000-01-01")
_PERIOD_END = np.datetime64("2019-01-01")

_SOURCE = (
    "Rounce et al. (2021), Global Glacier Debris Thickness Estimates and "
    "Sub-Debris Melt Factors, Version 1 (NSIDC HMA_DTE, doi:10.5067/8DQKWY03KJWT)"
)


def select_debris_links(links: Sequence[str]) -> dict[str, str]:
    """
    Pick one URL per debris variable from a granule's data links.

    Parameters
    ----------
    links : sequence of str
        Data-link URLs of one HMA_DTE granule (tifs plus CSV/XML/manifest
        sidecars, which are ignored).

    Returns
    -------
    dict[str, str]
        Mapping of output variable name (``"debris_thickness"``,
        ``"debris_melt_factor"``) to the chosen URL. A variable is absent when
        the granule carries no matching tif. A direct estimate wins over its
        ``_extrap`` fallback.
    """
    chosen: dict[str, str] = {}
    for variable, suffixes in _VARIABLE_SUFFIXES.items():
        for suffix in suffixes:
            matches = [link for link in links if urlparse(link).path.endswith(suffix)]
            if matches:
                chosen[variable] = matches[0]
                break
    return chosen


def download_debris_tifs(
    bounds: tuple[float, float, float, float],
    cache_dir: Path | str,
    dataset: str = "rounce",
) -> dict[str, list[Path]]:
    """
    Fetch the debris tifs overlapping a geographic bounding box.

    Queries NASA CMR via :mod:`earthaccess` for the granules whose footprints
    intersect *bounds* and downloads their thickness and melt-factor tifs into
    *cache_dir*; files already present are not re-downloaded. Requires
    Earthdata Login credentials (``~/.netrc`` or ``EARTHDATA_USERNAME`` /
    ``EARTHDATA_PASSWORD``), like the ISMIP7 retreat prep.

    Parameters
    ----------
    bounds : tuple of float
        Geographic bounding box ``(west, south, east, north)`` in EPSG:4326.
    cache_dir : str or pathlib.Path
        Directory the tifs are downloaded into, created if needed.
    dataset : str, default ``"rounce"``
        Key of :data:`DEBRIS_DATASETS` selecting the NSIDC collection.

    Returns
    -------
    dict[str, list[pathlib.Path]]
        Local tif paths keyed by output variable name. Empty lists when no
        granule overlaps the box.

    Raises
    ------
    NotImplementedError
        If *dataset* is not a known debris dataset.
    """
    if dataset not in DEBRIS_DATASETS:
        raise NotImplementedError(f"Unknown debris dataset {dataset!r}; available: {sorted(DEBRIS_DATASETS)}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    earthaccess.login()
    results = earthaccess.search_data(**DEBRIS_DATASETS[dataset], bounding_box=tuple(bounds))
    logger.info("HMA_DTE: %d granules intersect %s", len(results), bounds)

    urls: dict[str, list[str]] = {variable: [] for variable in _VARIABLE_SUFFIXES}
    for granule in results:
        for variable, url in select_debris_links(granule.data_links()).items():
            urls[variable].append(url)

    local = {
        variable: [cache_dir / Path(urlparse(url).path).name for url in variable_urls]
        for variable, variable_urls in urls.items()
    }
    missing = [
        url
        for variable, variable_urls in urls.items()
        for url, path in zip(variable_urls, local[variable])
        if not path.exists()
    ]
    if missing:
        earthaccess.download(missing, str(cache_dir))
    return local


def _assemble_variable(
    files: Sequence[Path | str],
    target_grid: xr.Dataset,
    geometries: collections.abc.Iterable,
    fill_value: float,
) -> xr.DataArray:
    """
    Mosaic per-glacier tifs onto the target grid and clip to the outline.

    Each tif covers a single RGI6 glacier in its own (UTM) CRS, so every file
    is reprojected to the target grid first and the stack is then collapsed;
    the footprints are disjoint, so the reduction merely picks the one finite
    value per cell.

    Parameters
    ----------
    files : sequence of str or pathlib.Path
        Per-glacier GeoTIFFs. May be empty, yielding a constant field.
    target_grid : xarray.Dataset
        Grid (with a projected CRS in ``spatial_ref``) the field is aligned to.
    geometries : iterable of shapely geometries
        Glacier outline(s) in ``target_grid``'s CRS; cells outside are set to
        *fill_value*.
    fill_value : float
        Value written wherever no debris estimate covers a cell.

    Returns
    -------
    xarray.DataArray
        The assembled field on ``target_grid``, with no attributes set.
    """
    crs = target_grid.rio.crs
    if files:
        layers = []
        for file in files:
            da = rxr.open_rasterio(file, masked=True).sel(band=1).drop_vars("band")
            layers.append(da.rio.reproject_match(target_grid, resampling=Resampling.bilinear))
        field = xr.concat(layers, dim="raster").max(dim="raster", skipna=True)
    else:
        field = xr.DataArray(
            np.full((target_grid.sizes["y"], target_grid.sizes["x"]), np.nan),
            coords={"y": target_grid["y"], "x": target_grid["x"]},
            dims=("y", "x"),
        )
    field = field.rio.write_crs(crs)
    field = field.rio.clip(geometries, drop=False)
    field = field.fillna(fill_value)
    # Replace inherited raster attrs wholesale; the caller stamps its own.
    field.attrs = {}
    for key in ("scale_factor", "add_offset", "AREA_OR_POINT", "_FillValue"):
        field.encoding.pop(key, None)
    return field


def assemble_debris(
    tifs: Mapping[str, Sequence[Path | str]],
    target_grid: xr.Dataset,
    geometries: collections.abc.Iterable,
) -> xr.Dataset:
    """
    Build the debris dataset on the target grid from downloaded tifs.

    Parameters
    ----------
    tifs : mapping of str to sequence of str or pathlib.Path
        Per-variable tif paths, as returned by :func:`download_debris_tifs`.
        A missing or empty entry yields the variable's neutral constant
        (zero thickness, unit melt factor).
    target_grid : xarray.Dataset
        Grid (with a projected CRS in ``spatial_ref``) the fields are aligned
        to.
    geometries : iterable of shapely geometries
        Glacier outline(s) in ``target_grid``'s CRS.

    Returns
    -------
    xarray.Dataset
        Variables ``debris_thickness`` (``m``) and ``debris_melt_factor``
        (``1``) on ``target_grid``. No ``standard_name`` is set — CF has none
        for these fields, and PISM refuses input files where two variables
        share one.
    """
    thickness = _assemble_variable(tifs.get("debris_thickness", []), target_grid, geometries, fill_value=0.0)
    thickness.attrs = {
        "units": "m",
        "long_name": "supraglacial debris thickness",
        "source": _SOURCE,
    }
    melt_factor = _assemble_variable(tifs.get("debris_melt_factor", []), target_grid, geometries, fill_value=1.0)
    melt_factor = melt_factor.where(thickness > 0, 1)
    melt_factor.attrs = {
        "units": "1",
        "long_name": "sub-debris melt enhancement factor",
        "source": _SOURCE,
    }
    return xr.Dataset({"debris_thickness": thickness, "debris_melt_factor": melt_factor})


def _add_static_time(ds: xr.Dataset) -> xr.Dataset:
    """
    Attach the single-record time axis spanning the estimate period.

    The HMA_DTE estimates are static (integrated over 2000-2018), so the file
    carries one record at the period midpoint with CF ``time_bnds`` covering
    the whole period; PISM reads that as constant-in-time forcing.

    Parameters
    ----------
    ds : xarray.Dataset
        Spatial-only debris dataset.

    Returns
    -------
    xarray.Dataset
        The dataset with a length-1 ``time`` dimension and ``time_bnds``.
    """
    midpoint = _PERIOD_START + (_PERIOD_END - _PERIOD_START) / 2
    ds = ds.expand_dims(time=[midpoint])
    ds["time_bnds"] = xr.DataArray(
        np.array([[_PERIOD_START, _PERIOD_END]]), dims=["time", "nv"], coords={"time": ds["time"]}
    )
    ds["time"].attrs.update({"axis": "T", "bounds": "time_bnds"})
    for name in ("time", "time_bnds"):
        ds[name].encoding.update({"dtype": "int64", "units": "hours since 2000-01-01"})
    return ds


def debris_from_grid(
    target_grid: xr.Dataset,
    geometries: collections.abc.Iterable,
    rgi_id: str | None = None,
    dataset: str = "rounce",
    path: Path | str = "tmp.nc",
    staging_path: Path | str | None = None,
    force_overwrite: bool = False,
) -> xr.Dataset:
    """
    Generate a debris-thickness forcing file for a glacier domain.

    Searches NASA Earthdata for the HMA_DTE granules overlapping the domain's
    geographic bounding box, mosaics their thickness and melt-factor tifs onto
    *target_grid*, clips to the glacier outline, attaches a single-record time
    axis, and writes the result to ``path`` (e.g. ``debris_{rgi_id}.nc``). A
    valid cache at ``path`` is reused unless ``force_overwrite=True``.

    Parameters
    ----------
    target_grid : xarray.Dataset
        Target grid dataset whose ``x``/``y`` extent (in the grid's projected
        CRS, as recorded in ``spatial_ref``) defines the query region.
    geometries : iterable of shapely geometries
        Glacier outline(s) in ``target_grid``'s CRS, used to clip the fields.
    rgi_id : str or None, optional
        RGI identifier of the glacier, used for logging only.
    dataset : str, default ``"rounce"``
        Key of :data:`DEBRIS_DATASETS` selecting the NSIDC collection.
    path : str or pathlib.Path, default ``"tmp.nc"``
        Output/cache NetCDF (e.g. ``debris_{rgi_id}.nc``).
    staging_path : str or pathlib.Path or None, optional
        Directory the raw tifs are cached under (in a ``debris``
        subdirectory) so reruns skip the download. Defaults to the directory
        of ``path``.
    force_overwrite : bool, default ``False``
        If ``True``, ignore any cache at ``path`` and regenerate.

    Returns
    -------
    xarray.Dataset
        Dataset with ``debris_thickness`` (``m``) and ``debris_melt_factor``
        (``1``) on ``target_grid`` and a length-1 unlimited ``time`` axis.
    """
    print("")
    print(f"Generate Debris Thickness ({dataset})" + (f" for {rgi_id}" if rgi_id else ""))
    print("-" * 120)

    if check_xr_lazy(path) and not force_overwrite:
        logger.info("Using cached debris file %s", path)
        return xr.open_dataset(path)

    path = Path(path)
    path.unlink(missing_ok=True)

    xs = [float(target_grid.x.values[0]), float(target_grid.x.values[-1])]
    ys = [float(target_grid.y.values[0]), float(target_grid.y.values[-1])]
    bounds = (min(xs), min(ys), max(xs), max(ys))
    mapping_var = target_grid.rio.grid_mapping
    dst_crs = target_grid[mapping_var].attrs["crs_wkt"]
    t_geo = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)
    geo_bounds = t_geo.transform_bounds(*bounds)
    # Pad the geographic bbox so granules whose footprints only graze the
    # domain after the 4326 round trip are still picked up.
    lon_pad = 0.25
    lat_pad = 0.1
    padded = (
        geo_bounds[0] - lon_pad,
        geo_bounds[1] - lat_pad,
        geo_bounds[2] + lon_pad,
        geo_bounds[3] + lat_pad,
    )

    cache_dir = Path(staging_path) / "debris" if staging_path is not None else path.parent / "debris"
    tifs = download_debris_tifs(padded, cache_dir, dataset=dataset)
    if not any(tifs.values()):
        logger.warning("No HMA_DTE granules overlap %s; writing a debris-free field", padded)

    ds = assemble_debris(tifs, target_grid, geometries)
    ds = _add_static_time(ds)

    # Stamp CF metadata on the projected x/y coords (lost across some
    # rioxarray ops) and suppress the default ``_FillValue=NaN`` writes onto
    # coordinate variables.
    ds["x"].attrs.update(
        {
            "standard_name": "projection_x_coordinate",
            "long_name": "x coordinate of projection",
            "units": "m",
            "axis": "X",
        }
    )
    ds["y"].attrs.update(
        {
            "standard_name": "projection_y_coordinate",
            "long_name": "y coordinate of projection",
            "units": "m",
            "axis": "Y",
        }
    )
    ds["x"].encoding["_FillValue"] = None
    ds["y"].encoding["_FillValue"] = None

    ds = ds.rio.write_crs(dst_crs).rio.write_grid_mapping().rio.write_coordinate_system()
    ds = stamp_grid_mapping(ds)
    # ``time_bnds`` is not a spatial field; the grid mapping does not belong on it.
    ds["time_bnds"].attrs.pop("grid_mapping", None)
    ds["time_bnds"].encoding.pop("grid_mapping", None)
    drop_geotransform_attr(ds)
    ds.to_netcdf(path, engine="h5netcdf", unlimited_dims=["time"])
    logger.info("Debris thickness saved to %s", path)
    return ds
