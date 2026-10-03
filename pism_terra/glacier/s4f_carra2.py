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

"""
Daily CARRA2 forcing for one RGI7 glacier complex, on CARRA2's own grid.

``pism-s4f-carra2`` takes an RGI7 id, finds the outline, and downloads the
daily CARRA2 means (CDS dataset ``reanalysis-pan-carra-means``) over the
outline's bounding box: air temperature on five pressure levels, snow depth,
total precipitation, and the net and downward surface shortwave radiation.
The result is one Zarr store per glacier.

Two things set this apart from the monthly, whole-domain store that
:func:`pism_terra.glacier.climate.prepare_carra2` builds.

**No reprojection.** The store stays on CARRA2's polar-stereographic grid.
CDS delivers the file without projected coordinates -- only the 2-D
latitude/longitude of each cell -- so the ``x``/``y`` axes are rebuilt by
projecting those back onto :data:`CARRA2_PROJ` and snapping to the 2.5 km
lattice. Snapping is checked, not assumed: a cell that lands more
than a few metres off the lattice means the projection or the file is not
what this module expects, and it raises rather than write a subtly shifted
grid.

**The whole domain is downloaded, once.** CDS accepts an ``area`` for this
dataset but returns the full pan-Arctic grid regardless (checked: a 3-degree
box came back as 2869 x 2869 cells). So no area is requested, the per-year
files are cached once and shared by every glacier, and the outline's box is
cut out locally. The first glacier pays for the download; the next ones are
a local clip.

**A vertical axis.** Temperature carries a ``pressure_level`` dimension,
ordered from 1000 hPa up, encoded with CF ``positive = "down"`` so that
tools which read the axis direction get it right.

Time is daily, stamped at the start of each day with ``time_bounds``
spanning it, and ``time`` and ``time_bounds`` are written with one shared
encoding -- xarray otherwise picks their units independently.
"""

from __future__ import annotations

import logging
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rioxarray  # noqa: F401  pylint: disable=unused-import
import xarray as xr
from pyproj import CRS, Transformer

from pism_terra.download import carra_download_request
from pism_terra.glacier.climate import CARRA2_PROJ
from pism_terra.glacier.rgi import prepare_rgi_region
from pism_terra.log import setup_logging
from pism_terra.vector import get_glacier_from_rgi_id

logger = logging.getLogger(__name__)

#: CDS dataset holding the daily and monthly CARRA2 means.
DATASET = "reanalysis-pan-carra-means"

#: Years the dataset spans, per its CDS form.
FIRST_YEAR = 1985
LAST_YEAR = 2025

#: Pressure levels of the temperature profile, hPa, bottom up.
PRESSURE_LEVELS = (1000, 900, 800, 750, 500)

#: Grid spacing and origin of the CARRA2 lattice, metres, from
#: ``pism_terra/grids/carra2.txt``.
GRID_SPACING = 2500.0
GRID_ORIGIN = -3585000.0

#: How far a reprojected cell centre may sit from the lattice before the
#: grid is declared not to be CARRA2's. A few metres of float noise is
#: normal; hundreds means the wrong projection.
SNAP_TOLERANCE = 25.0

#: Margin added around the outline's bounding box, kilometres. CARRA2 cells
#: are 2.5 km; a glacier's forcing needs the cells around it, not just the
#: ones it covers.
DEFAULT_MARGIN_KM = 25.0

#: RGI7 region codes to the NSIDC regional-file names.
RGI7_REGIONS = {
    "01": "01_alaska",
    "02": "02_western_canada_usa",
    "03": "03_arctic_canada_north",
    "04": "04_arctic_canada_south",
    "05": "05_greenland_periphery",
    "06": "06_iceland",
    "07": "07_svalbard_jan_mayen",
    "08": "08_scandinavia",
    "09": "09_russian_arctic",
    "10": "10_asia_north",
    "11": "11_central_europe",
    "12": "12_caucasus_middle_east",
    "13": "13_asia_central",
    "14": "14_asia_south_west",
    "15": "15_asia_south_east",
    "16": "16_low_latitudes",
    "17": "17_southern_andes",
    "18": "18_new_zealand",
    "19": "19_subantarctic_antarctic_islands",
}

RGI_ID_PATTERN = re.compile(r"^RGI2000-v7\.0-([CG])-(\d{2})-\d+$")

MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]

#: Output names, and how each field is recognised in what CDS returns.
#: ECMWF short names are not stable across versions, so a field is matched
#: on its CF metadata first and its short name last.
FIELDS: dict[str, dict[str, Any]] = {
    "air_temp": {"standard_name": "air_temperature", "short": ("t", "temperature")},
    "snow_depth": {"standard_name": "lwe_thickness_of_surface_snow_amount", "short": ("sde", "sd", "snow_depth")},
    "precipitation": {"standard_name": "precipitation_amount", "short": ("tp", "total_precipitation")},
    "surface_net_solar_radiation": {
        "standard_name": "surface_net_downward_shortwave_flux",
        "short": ("ssr", "surface_net_solar_radiation"),
    },
    "surface_solar_radiation_downwards": {
        "standard_name": "surface_downwelling_shortwave_flux_in_air",
        "short": ("ssrd", "surface_solar_radiation_downwards"),
    },
}

TIME_ENCODING = {"dtype": "int64", "units": "hours since 1850-01-01 00:00:00", "calendar": "standard"}
ZARR_CHUNKS = {"time": -1, "pressure_level": -1, "y": 256, "x": 256}


def parse_rgi_id(rgi_id: str) -> tuple[str, str]:
    """
    Split an RGI7 id into its outline type and region code.

    Parameters
    ----------
    rgi_id : str
        E.g. ``"RGI2000-v7.0-C-01-12784"``.

    Returns
    -------
    tuple of str
        Outline type (``"C"`` or ``"G"``) and two-digit region code.

    Raises
    ------
    ValueError
        If the id is not an RGI7 id.
    """
    match = RGI_ID_PATTERN.match(rgi_id)
    if match is None:
        raise ValueError(f"not an RGI7 id: {rgi_id!r}")
    return match.group(1), match.group(2)


def load_outline(rgi_id: str, rgi_file: Path | str | None = None, cache_path: Path | str = ".") -> gpd.GeoDataFrame:
    """
    Find the outline of one RGI7 id.

    Parameters
    ----------
    rgi_id : str
        RGI7 id.
    rgi_file : Path or str or None, optional
        Outline file to look it up in. When None, the id's region is fetched
        from NSIDC and cached under ``cache_path``.
    cache_path : Path or str, optional
        Where the regional archive is cached.

    Returns
    -------
    geopandas.GeoDataFrame
        The single outline, in the file's CRS (RGI7 ships EPSG:4326).

    Raises
    ------
    ValueError
        If the id is not found.
    """
    if rgi_file is not None:
        # Push the id filter down into GDAL: a regional or global GeoPackage
        # is hundreds of MB, and reading all of it to keep one row takes
        # minutes with nothing on the console to show for it.
        outline = pyogrio.read_dataframe(rgi_file, where=f"rgi_id = '{rgi_id}'", use_arrow=False)
    else:
        outline_type, region = parse_rgi_id(rgi_id)
        regional = prepare_rgi_region(
            {"region": RGI7_REGIONS[region]},
            outline_type=outline_type,
            extract_path=Path(cache_path) / "rgi_archive",
            area_threshold=0.0,
        )
        outline = get_glacier_from_rgi_id(regional, rgi_id)
    if outline.empty:
        raise ValueError(f"RGI id not found: {rgi_id}")
    return outline


def area_from_outline(outline: gpd.GeoDataFrame, margin_km: float = DEFAULT_MARGIN_KM) -> list[float]:
    """
    CDS ``area`` -- ``[north, west, south, east]`` -- around an outline.

    Parameters
    ----------
    outline : geopandas.GeoDataFrame
        Outline in any CRS; converted to geographic coordinates.
    margin_km : float, optional
        Margin around the bounding box, kilometres.

    Returns
    -------
    list of float
        ``[north, west, south, east]`` in degrees, rounded outward to 0.01.
    """
    west, south, east, north = outline.to_crs("EPSG:4326").total_bounds
    lat_mid = np.deg2rad(0.5 * (south + north))
    dlat = margin_km / 111.2
    dlon = margin_km / (111.2 * max(np.cos(lat_mid), 0.05))
    return [
        float(min(np.ceil((north + dlat) * 100) / 100, 90.0)),
        float(np.floor((west - dlon) * 100) / 100),
        float(np.floor((south - dlat) * 100) / 100),
        float(np.ceil((east + dlon) * 100) / 100),
    ]


def bbox_from_outline(
    outline: gpd.GeoDataFrame, margin_km: float = DEFAULT_MARGIN_KM
) -> tuple[float, float, float, float]:
    """
    The outline's bounding box in CARRA2 metres, padded.

    Parameters
    ----------
    outline : geopandas.GeoDataFrame
        Outline in any CRS.
    margin_km : float, optional
        Margin on every side, kilometres.

    Returns
    -------
    tuple of float
        ``(x_min, x_max, y_min, y_max)``.
    """
    x_min, y_min, x_max, y_max = outline.to_crs(CRS.from_proj4(CARRA2_PROJ)).total_bounds
    pad = 1e3 * margin_km
    return float(x_min - pad), float(x_max + pad), float(y_min - pad), float(y_max + pad)


def build_requests(years: Iterable[int]) -> dict[str, dict[str, Any]]:
    """
    The CDS requests, one per product type the fields need.

    Analysis-based and forecast-based fields cannot share a request, and
    pressure-level fields cannot share one with single-level fields, so
    three requests it is. The pressure levels go under ``level_location``:
    that is the key the pan-CARRA form uses, not ``pressure_level``. No
    ``area``: CDS ignores it for this dataset (the full grid comes back
    either way) and leaving it out lets every glacier share the same files.

    Parameters
    ----------
    years : iterable of int
        Years to request.

    Returns
    -------
    dict
        Request name to CDS request dict.
    """
    base = {
        "time_aggregation": "daily",
        "year": [str(y) for y in years],
        "month": MONTHS,
        "day": DAYS,
        "data_format": "netcdf",
    }
    return {
        "temperature": {
            **base,
            "level_type": "pressure_levels",
            "product_type": "analysis_based",
            "variable": ["temperature"],
            "level_location": [str(p) for p in PRESSURE_LEVELS],
        },
        "snow": {
            **base,
            "level_type": "single_levels",
            "product_type": "analysis_based",
            "variable": ["snow_depth"],
        },
        "forecast": {
            **base,
            "level_type": "single_levels",
            "product_type": "forecast_based",
            "variable": ["total_precipitation", "surface_net_solar_radiation", "surface_solar_radiation_downwards"],
        },
    }


def _find(ds: xr.Dataset, name: str) -> str:
    """
    Locate the source variable for one output field.

    Parameters
    ----------
    ds : xarray.Dataset
        A CDS file.
    name : str
        Key of :data:`FIELDS`.

    Returns
    -------
    str
        The source variable's name.

    Raises
    ------
    KeyError
        If nothing in the file matches.
    """
    spec = FIELDS[name]
    for var in ds.data_vars:
        attrs = ds[var].attrs
        if attrs.get("standard_name") == spec["standard_name"]:
            return str(var)
    for var in ds.data_vars:
        attrs = ds[var].attrs
        candidates = {
            str(var).lower(),
            str(attrs.get("GRIB_shortName", "")).lower(),
            str(attrs.get("GRIB_cfVarName", "")).lower(),
        }
        if candidates & set(spec["short"]):
            return str(var)
    raise KeyError(f"no variable for {name!r} among {sorted(map(str, ds.data_vars))}")


def projected_axes(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rebuild the CARRA2 ``x``/``y`` axes from a subset's 2-D latitude/longitude.

    CDS returns an ``area`` subset with only the cell centres' geographic
    coordinates. Projecting those onto CARRA2's own CRS recovers the
    projected coordinates; along a lattice row they must agree to float
    noise, and they must fall on the 2.5 km lattice. Both are checked.

    Parameters
    ----------
    lat, lon : numpy.ndarray
        2-D arrays of shape ``(ny, nx)``.

    Returns
    -------
    tuple of numpy.ndarray
        1-D ``x`` (length ``nx``) and ``y`` (length ``ny``), metres.

    Raises
    ------
    ValueError
        If the projected cells are not a rectilinear lattice with CARRA2's
        spacing and origin.
    """
    transformer = Transformer.from_crs("EPSG:4326", CRS.from_proj4(CARRA2_PROJ), always_xy=True)
    xx, yy = transformer.transform(np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64"))
    x_spread = float(np.nanmax(np.nanstd(xx, axis=0)))
    y_spread = float(np.nanmax(np.nanstd(yy, axis=1)))
    if max(x_spread, y_spread) > SNAP_TOLERANCE:
        raise ValueError(
            f"projected cells do not form a rectilinear grid (spread {x_spread:.1f} m in x, {y_spread:.1f} m in y): "
            "is this really CARRA2 on its native projection?"
        )
    x = np.nanmean(xx, axis=0)
    y = np.nanmean(yy, axis=1)
    x_snapped = GRID_ORIGIN + np.round((x - GRID_ORIGIN) / GRID_SPACING) * GRID_SPACING
    y_snapped = GRID_ORIGIN + np.round((y - GRID_ORIGIN) / GRID_SPACING) * GRID_SPACING
    x_off = float(np.max(np.abs(x - x_snapped)))
    y_off = float(np.max(np.abs(y - y_snapped)))
    if max(x_off, y_off) > SNAP_TOLERANCE:
        raise ValueError(
            f"cells sit {x_off:.1f} m (x) / {y_off:.1f} m (y) off the CARRA2 lattice; "
            f"expected multiples of {GRID_SPACING:.0f} m from {GRID_ORIGIN:.0f} m"
        )
    return x_snapped, y_snapped


def _spatial_dims(ds: xr.Dataset) -> tuple[str, str, str, str]:
    """
    Find the latitude/longitude variables and the dims they span.

    Parameters
    ----------
    ds : xarray.Dataset
        A CDS file.

    Returns
    -------
    tuple of str
        ``(lat_name, lon_name, y_dim, x_dim)``.

    Raises
    ------
    ValueError
        If no 2-D latitude/longitude pair is present.
    """
    names = {str(v): v for v in list(ds.coords) + list(ds.data_vars)}
    lat = next((n for n in names if n.lower() in ("latitude", "lat")), None)
    lon = next((n for n in names if n.lower() in ("longitude", "lon")), None)
    if lat is None or lon is None or ds[lat].ndim != 2:
        raise ValueError(f"expected 2-D latitude/longitude in the CDS file, found {sorted(names)}")
    y_dim, x_dim = (str(d) for d in ds[lat].dims)
    return lat, lon, y_dim, x_dim


def _time_dim(ds: xr.Dataset, spatial: set[str], vertical: str | None = None) -> str:
    """
    Find the time dimension of a CDS file, whatever it is called this year.

    Parameters
    ----------
    ds : xarray.Dataset
        A CDS file.
    spatial : set of str
        The two spatial dims.
    vertical : str or None, optional
        The vertical dim, if any.

    Returns
    -------
    str
        Its name.

    Raises
    ------
    ValueError
        If no single remaining dimension is left to be time.
    """
    for candidate in ("time", "valid_time"):
        if candidate in ds.dims:
            return candidate
    rest = [str(d) for d in ds.dims if str(d) not in spatial and str(d) != vertical]
    if len(rest) != 1:
        raise ValueError(f"cannot tell which of {rest} is time")
    return rest[0]


def normalize(ds: xr.Dataset, fields: Sequence[str], bbox: Sequence[float] | None = None) -> xr.Dataset:
    """
    Turn one CDS file into the store's layout.

    Renames the fields, rebuilds projected ``x``/``y``, cuts out ``bbox``,
    names the vertical dimension ``pressure_level`` and gives it CF
    attributes, and reduces time to daily stamps at 00:00.

    Parameters
    ----------
    ds : xarray.Dataset
        A CDS file, times decoded.
    fields : sequence of str
        Keys of :data:`FIELDS` expected in it.
    bbox : sequence of float or None, optional
        ``(x_min, x_max, y_min, y_max)`` in CARRA2 metres to keep; the whole
        domain when None.

    Returns
    -------
    xarray.Dataset
        Fields on ``(time[, pressure_level], y, x)``.
    """
    lat, lon, y_dim, x_dim = _spatial_dims(ds)
    x, y = projected_axes(ds[lat].values, ds[lon].values)
    renamed = {_find(ds, name): name for name in fields}
    out = ds[list(renamed)].rename(renamed)
    if y_dim != "y" or x_dim != "x":
        out = out.rename({y_dim: "y", x_dim: "x"})
    out = out.assign_coords(
        x=(
            "x",
            x,
            {
                "standard_name": "projection_x_coordinate",
                "long_name": "x coordinate of projection",
                "units": "m",
                "axis": "X",
            },
        ),
        y=(
            "y",
            y,
            {
                "standard_name": "projection_y_coordinate",
                "long_name": "y coordinate of projection",
                "units": "m",
                "axis": "Y",
            },
        ),
        latitude=(
            ("y", "x"),
            np.asarray(ds[lat].values, dtype="float64"),
            {"standard_name": "latitude", "units": "degrees_north"},
        ),
        longitude=(
            ("y", "x"),
            np.asarray(ds[lon].values, dtype="float64"),
            {"standard_name": "longitude", "units": "degrees_east"},
        ),
    )

    if bbox is not None:
        # Before anything touches the data: the file is the whole domain and
        # the fields are still lazy, so this is the step that keeps memory
        # and the store small. The axes are ascending, so plain slices work.
        x_min, x_max, y_min, y_max = bbox
        out = out.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))
        if out.sizes["x"] == 0 or out.sizes["y"] == 0:
            raise ValueError(f"bbox {bbox} selects no cells of the CARRA2 domain")

    vertical = next(
        (
            str(d)
            for d in out.dims
            if str(d) not in ("x", "y", "time", "valid_time") and out.sizes[d] == len(PRESSURE_LEVELS)
        ),
        None,
    )
    time_dim = _time_dim(out, {"x", "y"}, vertical)
    if time_dim != "time":
        out = out.rename({time_dim: "time"})
    if vertical is not None:
        out = out.rename({vertical: "pressure_level"})
        levels = np.asarray(out["pressure_level"].values, dtype="float64")
        if not np.allclose(np.sort(levels), np.sort(PRESSURE_LEVELS)):
            raise ValueError(f"pressure levels {levels} are not {PRESSURE_LEVELS}")
        out = out.assign_coords(pressure_level=("pressure_level", levels)).sortby("pressure_level", ascending=False)
        out["pressure_level"].attrs.update(
            {
                "standard_name": "air_pressure",
                "long_name": "pressure level",
                "units": "hPa",
                "positive": "down",
                "axis": "Z",
            }
        )

    # Daily means are stamped by day; stamp them at 00:00 and drop anything
    # else. CDS files carry the forecast reference time as ``time`` and the
    # valid time beside it; the valid time is the day the mean describes,
    # which for a forecast-based field is not the reference time.
    source = ds["valid_time"] if "valid_time" in ds.coords and ds["valid_time"].dims == (time_dim,) else out["time"]
    stamps = np.asarray(source.values).astype("datetime64[D]").astype("datetime64[ns]")
    out = out.assign_coords(time=("time", stamps)).sortby("time")
    out = out.isel(time=~out.get_index("time").duplicated())
    out = out.drop_vars(
        [v for v in out.coords if v not in ("time", "pressure_level", "y", "x", "latitude", "longitude")]
    )
    for name in fields:
        out[name] = out[name].transpose(
            "time", *(["pressure_level"] if "pressure_level" in out[name].dims else []), "y", "x"
        )
    return out


def add_time_bounds(ds: xr.Dataset) -> xr.Dataset:
    """
    Give daily stamps the day they cover.

    Parameters
    ----------
    ds : xarray.Dataset
        Daily data stamped at 00:00.

    Returns
    -------
    xarray.Dataset
        With ``time_bounds`` on ``(time, nv)`` and ``time`` pointing at it.
    """
    start = pd.DatetimeIndex(ds["time"].values)
    bounds = np.stack([start.values, (start + pd.Timedelta(days=1)).values], axis=1)
    ds["time_bounds"] = xr.DataArray(bounds, dims=("time", "nv"))
    ds["time"].attrs.update({"standard_name": "time", "long_name": "time", "axis": "T", "bounds": "time_bounds"})
    return ds


def download(
    years: Iterable[int],
    cache_path: Path | str,
    max_workers: int = 4,
    force_overwrite: bool = False,
) -> dict[str, list[Path]]:
    """
    Fetch the three CDS requests.

    Parameters
    ----------
    years : iterable of int
        Years to request.
    cache_path : Path or str
        Directory the per-year files are cached under. Not per glacier: the
        files are the whole domain, so every glacier shares them.
    max_workers : int, optional
        Concurrent CDS requests.
    force_overwrite : bool, optional
        Re-download even when the cache is complete.

    Returns
    -------
    dict
        Request name to the per-year files it produced.
    """
    cache_path = Path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    files: dict[str, list[Path]] = {}
    for name, request in build_requests(years).items():
        logger.info("CDS %s: %s %s", name, request["product_type"], request["variable"])
        files[name] = [
            Path(f)
            for f in carra_download_request(
                DATASET,
                dict(request),
                file_path=cache_path / f"{name}.nc",
                max_workers=max_workers,
                force_overwrite=force_overwrite,
            )
        ]
    return files


REQUEST_FIELDS = {
    "temperature": ["air_temp"],
    "snow": ["snow_depth"],
    "forecast": ["precipitation", "surface_net_solar_radiation", "surface_solar_radiation_downwards"],
}


def assemble(files: dict[str, list[Path]], bbox: Sequence[float] | None = None) -> xr.Dataset:
    """
    Merge the downloaded files into one dataset on CARRA2's grid.

    Parameters
    ----------
    files : dict
        Request name to per-year files, as :func:`download` returns.
    bbox : sequence of float or None, optional
        ``(x_min, x_max, y_min, y_max)`` in CARRA2 metres to keep.

    Returns
    -------
    xarray.Dataset
        Every field, with time bounds and the CARRA2 CRS attached.
    """
    parts = []
    for name, paths in files.items():
        yearly = [normalize(xr.open_dataset(p, chunks={}), REQUEST_FIELDS[name], bbox) for p in sorted(paths)]
        parts.append(
            xr.concat(yearly, dim="time", data_vars="minimal", coords="minimal", compat="override", join="exact")
        )
    ds = xr.merge(parts, join="exact", compat="override")
    ds = add_time_bounds(ds)
    # rioxarray records the grid mapping in each variable's *encoding*, and
    # xarray writes that out as the CF ``grid_mapping`` attribute; setting the
    # attribute as well makes the writer refuse the duplicate. Bounds are not
    # a field, so they get no grid mapping.
    ds = ds.rio.write_crs(CARRA2_PROJ).rio.write_grid_mapping("spatial_ref").rio.write_coordinate_system()
    ds["time_bounds"].encoding.pop("grid_mapping", None)
    ds["time_bounds"].attrs.pop("grid_mapping", None)
    return ds


def write_store(ds: xr.Dataset, store: Path | str) -> Path:
    """
    Write the dataset as a consolidated Zarr store.

    Parameters
    ----------
    ds : xarray.Dataset
        Assembled dataset.
    store : Path or str
        Store path; replaced if present.

    Returns
    -------
    pathlib.Path
        The store.
    """
    store = Path(store)
    chunks = {k: v for k, v in ZARR_CHUNKS.items() if k in ds.dims}
    ds = ds.chunk(chunks)
    encoding: dict[str, dict[str, Any]] = {"time": dict(TIME_ENCODING), "time_bounds": dict(TIME_ENCODING)}
    for name in ds.data_vars:
        ds[name].encoding.pop("chunks", None)
        ds[name].encoding.pop("preferred_chunks", None)
    ds.to_zarr(store, mode="w", consolidated=True, encoding=encoding)
    logger.info("Wrote %s", store)
    return store


def run(
    rgi_id: str,
    output_path: Path | str,
    *,
    rgi_file: Path | str | None = None,
    start: int = FIRST_YEAR,
    end: int = LAST_YEAR,
    margin_km: float = DEFAULT_MARGIN_KM,
    max_workers: int = 4,
    force_overwrite: bool = False,
) -> Path:
    """
    Download and write the daily CARRA2 store for one glacier.

    Parameters
    ----------
    rgi_id : str
        RGI7 id.
    output_path : Path or str
        Directory the store and download cache go under.
    rgi_file : Path or str or None, optional
        Outline file; the region is fetched from NSIDC when None.
    start, end : int, optional
        First and last year, inclusive.
    margin_km : float, optional
        Margin around the outline, kilometres.
    max_workers : int, optional
        Concurrent CDS requests.
    force_overwrite : bool, optional
        Re-download and rewrite.

    Returns
    -------
    pathlib.Path
        The Zarr store.

    Raises
    ------
    ValueError
        If the years are out of range or reversed.
    """
    if not FIRST_YEAR <= start <= end <= LAST_YEAR:
        raise ValueError(f"years must satisfy {FIRST_YEAR} <= start <= end <= {LAST_YEAR}, got {start}..{end}")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    store = output_path / f"carra2_daily_{rgi_id}.zarr"
    if store.exists() and not force_overwrite:
        logger.info("Using existing %s", store)
        return store

    # INFO goes to the log file only (see setup_logging), and nothing else is
    # printed until CDS starts its progress bars -- so say what is happening
    # in between: the outline lookup alone can take minutes on a large
    # GeoPackage on a network share.
    if rgi_file is not None:
        size_mb = Path(rgi_file).stat().st_size / 1e6
        print(f"Looking up {rgi_id} in {rgi_file} ({size_mb:.0f} MB)...", flush=True)
    else:
        print(f"Fetching the RGI7 region archive for {rgi_id} from NSIDC...", flush=True)
    outline = load_outline(rgi_id, rgi_file, cache_path=output_path)
    area = area_from_outline(outline, margin_km)
    bbox = bbox_from_outline(outline, margin_km)
    logger.info("%s: area N %.2f W %.2f S %.2f E %.2f, years %d-%d", rgi_id, *area, start, end)
    print(f"Area N {area[0]} W {area[1]} S {area[2]} E {area[3]}, years {start}-{end}", flush=True)
    print(
        "Submitting 3 CDS requests (temperature on pressure levels, snow depth, precipitation + radiation).", flush=True
    )
    print(
        "CDS returns the whole pan-Arctic domain (~0.8 GB per year across the three); the files are cached", flush=True
    )
    print(
        f"under {output_path / 'cds'} and shared by every glacier. The queue can take hours -- the bars track it.",
        flush=True,
    )
    files = download(range(start, end + 1), output_path / "cds", max_workers, force_overwrite)
    print("Clipping to the outline and assembling the store...", flush=True)
    ds = assemble(files, bbox)
    ds.attrs.update(
        {
            "title": f"Daily CARRA2 forcing for {rgi_id}",
            "source": f"CDS {DATASET}, time_aggregation=daily",
            "rgi_id": rgi_id,
            "area": f"N {area[0]} W {area[1]} S {area[2]} E {area[3]}",
            "bbox": f"x {bbox[0]:.0f}..{bbox[1]:.0f} y {bbox[2]:.0f}..{bbox[3]:.0f} m (CARRA2 projection)",
            "Conventions": "CF-1.8",
        }
    )
    return write_store(ds, store)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Console entry point.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments without the program name.

    Returns
    -------
    int
        Exit code.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Daily CARRA2 forcing for one RGI7 glacier complex, on CARRA2's own grid, as Zarr."
    parser.add_argument(
        "--output-path", type=Path, default=Path("data/carra2"), help="Directory for the store and cache."
    )
    parser.add_argument(
        "--rgi-file", type=Path, default=None, help="Outline file with an rgi_id column; fetched from NSIDC if omitted."
    )
    parser.add_argument("--start", type=int, default=FIRST_YEAR, help="First year.")
    parser.add_argument("--end", type=int, default=LAST_YEAR, help="Last year (inclusive).")
    parser.add_argument("--margin-km", type=float, default=DEFAULT_MARGIN_KM, help="Margin around the outline.")
    parser.add_argument("--max-workers", type=int, default=4, help="Concurrent CDS requests.")
    parser.add_argument("--force-overwrite", action="store_true", help="Re-download and rewrite.")
    parser.add_argument("RGI_ID", help="RGI7 id, e.g. RGI2000-v7.0-C-01-12784.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_path / f"carra2_daily_{args.RGI_ID}.log")
    store = run(
        args.RGI_ID,
        args.output_path,
        rgi_file=args.rgi_file,
        start=args.start,
        end=args.end,
        margin_km=args.margin_km,
        max_workers=args.max_workers,
        force_overwrite=args.force_overwrite,
    )
    print(store)
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(main())
