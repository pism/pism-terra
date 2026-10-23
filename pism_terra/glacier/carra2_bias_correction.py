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

# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments

"""
How far CARRA2's free-air temperature, carried down to the ground, is from its 2 m temperature.

A daily CARRA2 store from :mod:`pism_terra.glacier.s4f_carra2` holds the air
temperature and the geopotential height on pressure levels, the 2 m air
temperature, and the terrain (``orography``) the 2 m temperature refers to.
For every grid cell and day a straight line ``T = a + b z`` is fitted through
the levels and evaluated at the terrain. The difference to the 2 m
temperature -- the *bias* -- is what a forcing built from the free atmosphere
would get wrong at the surface: over melting ice the near-surface air is held
near 0 degC while the free air warms, so the bias grows on the lower glacier
in summer.

``pism-carra2-bias-correction`` takes one or more stores and the glacier
outlines, splits the cells into on and off glacier by whether the cell
centre lies inside an outline, and bins them by terrain elevation. The bias
changes with elevation and with the season, so per category the daily median
bias of each bin is fitted with a model linear in elevation plus one
seasonal cycle (period one year) whose amplitude and phase may also change
linearly with elevation; see :func:`fit_seasonal`. Several years pool into
one cycle. It writes the medians, the fit and the fitted cycle per bin as
CSV, and figures of the bias against elevation and against the day of the
year. Given point sites as well, it also samples the stores there and plots
the bias per month, pooled and per site.

The bias is never held for the whole domain: the store is read one chunk
of the grid at a time and the bias counted into histograms per day,
category and elevation bin (see :func:`bias_histograms`), so memory depends
on the chunk, not on the domain or the number of years. Cells at sea level
are taken for sea and left out (``--min-elevation``).

Stores, outlines and sites may each be a local path or an ``s3://`` URI.
"""

import logging
import math
import tempfile
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from matplotlib.figure import Figure
from tqdm.auto import tqdm

from pism_terra.grids import sample_points
from pism_terra.log import setup_logging
from pism_terra.ntasks_option import add_ntasks_argument
from pism_terra.workflow import dataset_crs

logger = logging.getLogger(__name__)

#: Variables a CARRA2 store must have.
REQUIRED_VARS = ("air_temp", "geopotential_height", "air_temp_2m", "orography")

#: Category labels, in plotting order, with their colour and violin offset
#: (in bin widths) so a bin's two violins sit side by side.
CATEGORIES = {"off_glacier": ("tab:brown", -0.25), "on_glacier": ("tab:blue", 0.25)}

#: Default width of the elevation bins, metres.
DEFAULT_BIN_WIDTH = 250.0

#: Bins resting on fewer grid cells than this stay out of the fit: one
#: cell's median is a single location, not a distribution.
DEFAULT_MIN_CELLS = 3

#: Width of a bin of the bias histograms, K: a median read off them is good
#: to a fraction of this, far inside the day-to-day scatter.
DEFAULT_HIST_STEP = 0.05

#: Range of the bias histograms, K; values beyond are counted in the end bins.
HIST_RANGE = (-30.0, 50.0)

#: CARRA2's terrain over the sea lies at about 0 m (slightly below, in
#: places); cells below this many metres are taken for sea and left out.
DEFAULT_MIN_ELEVATION = 5.0

#: Tile size in grid cells: the chunk size pism-s4f-carra2 writes, so each
#: tile reads exactly one chunk of every field.
TILE = 256

#: Days whose bias is computed at once within a tile.
DEFAULT_TIME_BLOCK = 31

#: Coefficients of the seasonal model, see :func:`fit_seasonal`.
COEFFICIENTS = ("mean", "mean_per_km", "cos", "sin", "cos_per_km", "sin_per_km")

#: Default number of CARRA2 levels below the terrain each daily fit may use.
#: With five levels a high site has a single level above CARRA2's terrain,
#: so one below it is needed to fit at all; with twelve, 0 fits real air only.
DEFAULT_BELOW = 1

MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def is_s3(path: str | Path) -> bool:
    """
    Tell whether a path is an S3 URI.

    Parameters
    ----------
    path : str or Path
        Local path or URI.

    Returns
    -------
    bool
        True for ``s3://...``.
    """
    return str(path).startswith("s3://")


def open_carra2(path: str | Path, anon: bool = True) -> xr.Dataset:
    """
    Open a daily CARRA2 store, lazily.

    A ``.zarr`` store is read with the Zarr engine, anything else as NetCDF;
    an S3 NetCDF file is streamed through ``s3fs``.

    Parameters
    ----------
    path : str or Path
        Local path or ``s3://`` URI of a store written by
        :mod:`pism_terra.glacier.s4f_carra2`.
    anon : bool, default True
        Read S3 without credentials (the bucket must be public).

    Returns
    -------
    xarray.Dataset
        The store, backed by dask.

    Raises
    ------
    ValueError
        If the store lacks a variable of :data:`REQUIRED_VARS`.
    """
    uri = str(path)
    if uri.rstrip("/").endswith(".zarr"):
        options = {"storage_options": {"anon": anon}} if is_s3(uri) else {}
        ds = xr.open_dataset(uri, engine="zarr", chunks={}, **options)
    elif is_s3(uri):
        ds = xr.open_dataset(s3fs.S3FileSystem(anon=anon).open(uri), engine="h5netcdf", chunks={})
    else:
        ds = xr.open_dataset(uri, chunks={})
    missing = [v for v in REQUIRED_VARS if v not in ds]
    if missing:
        raise ValueError(f"{uri} lacks {missing}; build it with pism-s4f-carra2")
    return ds


def read_vector(path: str | Path, anon: bool = True) -> gpd.GeoDataFrame:
    """
    Read a vector file from a local path or S3.

    An S3 object is copied to a temporary directory first, with the sidecar
    files of a shapefile (``.shx``, ``.dbf``, ``.prj``, ...) alongside it.

    Parameters
    ----------
    path : str or Path
        Local path or ``s3://`` URI of anything GDAL reads (GeoPackage,
        shapefile, GeoJSON, ...).
    anon : bool, default True
        Read S3 without credentials.

    Returns
    -------
    geopandas.GeoDataFrame
        The features, in the file's CRS.

    Raises
    ------
    FileNotFoundError
        If the S3 object does not exist.
    """
    if not is_s3(path):
        return gpd.read_file(path)
    uri = str(path)
    fs = s3fs.S3FileSystem(anon=anon)
    if not fs.exists(uri):
        raise FileNotFoundError(uri)
    stem, suffix = uri.rsplit(".", 1) if "." in uri.rsplit("/", 1)[-1] else (uri, "")
    sources = fs.glob(f"{stem}.*") if suffix.lower() == "shp" else [uri]
    with tempfile.TemporaryDirectory() as tmp:
        for source in sources:
            fs.get(source, str(Path(tmp) / Path(source).name))
        return gpd.read_file(Path(tmp) / Path(uri).name)


def read_vectors(paths: Sequence[str | Path], anon: bool = True) -> gpd.GeoDataFrame:
    """
    Read several vector files into one frame.

    Parameters
    ----------
    paths : sequence of str or Path
        Local paths or ``s3://`` URIs.
    anon : bool, default True
        Read S3 without credentials.

    Returns
    -------
    geopandas.GeoDataFrame
        All features, in the CRS of the first file.
    """
    frames = [read_vector(p, anon) for p in paths]
    crs = frames[0].crs
    return gpd.GeoDataFrame(pd.concat([f.to_crs(crs) for f in frames], ignore_index=True), crs=crs)


def fit_to_ground(
    temp: xr.DataArray,
    height: xr.DataArray,
    ground: xr.DataArray,
    below: int | None = DEFAULT_BELOW,
    dim: str = "pressure_level",
) -> xr.Dataset:
    """
    Fit ``T = a + b z`` through the pressure levels and evaluate it at the ground.

    The fit is ordinary least squares along *dim*, separately for every other
    index (day, cell or site). CARRA2 reports temperatures on levels below
    its terrain too, but those are extrapolated, not air; *below* limits how
    many of them enter the fit.

    Parameters
    ----------
    temp : xarray.DataArray
        Temperature on *dim*, K.
    height : xarray.DataArray
        Height of each level on *dim*, m; the levels must run bottom up, so
        height rises along *dim*.
    ground : xarray.DataArray
        Height the fit is evaluated at, m, broadcastable against the others
        (the orography, on a grid or at sites).
    below : int or None, default :data:`DEFAULT_BELOW`
        Levels below the ground used besides every level above it: 0 uses
        only levels above the ground, None every level.
    dim : str, default "pressure_level"
        The vertical dimension.

    Returns
    -------
    xarray.Dataset
        ``t_ground`` (K), the fit at the ground; ``lapse_rate`` (K km-1);
        ``n_levels``, the levels used. Both are NaN where fewer than two
        levels were usable.
    """
    if below is None:
        use = xr.ones_like(height, dtype=bool)
    else:
        # Levels run bottom up, so the ones below the ground come first: keep
        # every level from index (n_below - below) upward.
        n_below = (height < ground).sum(dim)
        index = xr.DataArray(np.arange(height.sizes[dim]), dims=dim)
        use = index >= n_below - below
    z, t = height.where(use), temp.where(use)
    n_levels = use.sum(dim)
    z_mean, t_mean = z.mean(dim), t.mean(dim)
    slope = ((z - z_mean) * (t - t_mean)).sum(dim) / ((z - z_mean) ** 2).sum(dim)
    slope = slope.where(n_levels >= 2)
    return xr.Dataset(
        {
            "t_ground": (t_mean + slope * (ground - z_mean)).assign_attrs(
                units="K", long_name="level fit evaluated at the ground"
            ),
            "lapse_rate": (1000.0 * slope).assign_attrs(units="K km-1", long_name="fitted lapse rate"),
            "n_levels": n_levels.assign_attrs(long_name="pressure levels used in the fit"),
        }
    )


def ground_bias(ds: xr.Dataset, below: int | None = DEFAULT_BELOW) -> xr.DataArray:
    """
    Compute the level fit at the terrain minus the 2 m air temperature.

    Parameters
    ----------
    ds : xarray.Dataset
        CARRA2 store, or points sampled from one.
    below : int or None, default :data:`DEFAULT_BELOW`
        See :func:`fit_to_ground`.

    Returns
    -------
    xarray.DataArray
        ``bias`` in K on the store's dimensions without the vertical one.
    """
    fit = fit_to_ground(ds["air_temp"], ds["geopotential_height"], ds["orography"], below=below)
    bias = fit["t_ground"] - ds["air_temp_2m"]
    return bias.rename("bias").assign_attrs(units="K", long_name="level fit at the terrain minus 2 m air temperature")


def glacier_mask(ds: xr.Dataset, outlines: gpd.GeoDataFrame) -> xr.DataArray:
    """
    Mark the grid cells whose centre lies inside a glacier outline.

    Parameters
    ----------
    ds : xarray.Dataset
        Store on a projected ``(y, x)`` grid with its grid mapping.
    outlines : geopandas.GeoDataFrame
        Glacier outlines in any CRS.

    Returns
    -------
    xarray.DataArray
        Boolean on ``(y, x)``, True on glacier.
    """
    crs = dataset_crs(ds)
    xx, yy = np.meshgrid(ds["x"].values, ds["y"].values)
    centres = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs=crs)
    polygons = outlines.to_crs(crs)[["geometry"]]
    hits = centres.sjoin(polygons, predicate="within", how="inner")
    inside = np.zeros(centres.shape[0], dtype=bool)
    inside[np.unique(hits.index.to_numpy())] = True
    return xr.DataArray(inside.reshape(xx.shape), dims=("y", "x"), coords={"y": ds["y"], "x": ds["x"]})


def bin_centre(elevation: np.ndarray, width: float) -> np.ndarray:
    """
    Map elevations to the centre of their bin.

    Bins are ``[k w, (k + 1) w)`` for integer ``k``, so they line up across
    stores whatever range each covers.

    Parameters
    ----------
    elevation : numpy.ndarray
        Elevations, m.
    width : float
        Bin width, m.

    Returns
    -------
    numpy.ndarray
        Bin centres, m.
    """
    return (np.floor(np.asarray(elevation, dtype=float) / width) + 0.5) * width


def value_edges(step: float = DEFAULT_HIST_STEP, value_range: tuple[float, float] = HIST_RANGE) -> np.ndarray:
    """
    Build the edges of the bias histograms.

    Parameters
    ----------
    step : float, default :data:`DEFAULT_HIST_STEP`
        Width of a histogram bin, K.
    value_range : tuple of float, default :data:`HIST_RANGE`
        Lowest and highest edge, K.

    Returns
    -------
    numpy.ndarray
        The edges, K.
    """
    low, high = value_range
    return low + step * np.arange(int(round((high - low) / step)) + 1)


def _tile_counts(
    tile: xr.Dataset,
    category: np.ndarray,
    elevation_bin: np.ndarray,
    n_centres: int,
    edges: np.ndarray,
    below: int | None,
    time_block: int,
) -> tuple[np.ndarray, int]:
    """
    Histogram the daily bias of one tile per category and elevation bin.

    Parameters
    ----------
    tile : xarray.Dataset
        The store's :data:`REQUIRED_VARS` over the tile, in memory.
    category : numpy.ndarray
        Category index of each cell on ``(y, x)``, in the order of :data:`CATEGORIES`.
    elevation_bin : numpy.ndarray
        Elevation-bin index of each cell on ``(y, x)``; -1 leaves the cell out.
    n_centres : int
        Number of elevation bins.
    edges : numpy.ndarray
        Histogram edges, from :func:`value_edges`.
    below : int or None
        See :func:`fit_to_ground`.
    time_block : int
        Days computed at once; bounds the temporaries of the fit.

    Returns
    -------
    counts : numpy.ndarray
        ``(time, category, centre, value)`` counts.
    clipped : int
        Values beyond the histogram's range, counted in its end bins.
    """
    n_days, n_values = tile.sizes["time"], edges.size - 1
    n_groups = len(CATEGORIES) * n_centres
    step = float(edges[1] - edges[0])
    keep = elevation_bin >= 0
    group = (category[keep] * n_centres + elevation_bin[keep]).astype(np.int64)
    shape = (n_days, len(CATEGORIES), n_centres, n_values)
    counts = np.zeros(n_days * n_groups * n_values, dtype=np.int64)
    clipped = 0
    if not keep.any():
        return counts.reshape(shape), clipped
    for start in range(0, n_days, time_block):
        block = tile.isel(time=slice(start, start + time_block))
        values = ground_bias(block, below=below).transpose("time", "y", "x").values[:, keep]
        finite = np.isfinite(values)
        index = np.floor((np.where(finite, values, edges[0]) - edges[0]) / step)
        clipped += int((finite & ((index < 0) | (index >= n_values))).sum())
        index = np.clip(index, 0, n_values - 1).astype(np.int64)
        day = start + np.arange(values.shape[0], dtype=np.int64)[:, None]
        flat = ((day * n_groups + group[None, :]) * n_values + index)[finite]
        counts += np.bincount(flat, minlength=counts.size)
    return counts.reshape(shape), clipped


def bias_histograms(
    ds: xr.Dataset,
    on_glacier: xr.DataArray,
    bin_width: float = DEFAULT_BIN_WIDTH,
    below: int | None = DEFAULT_BELOW,
    step: float = DEFAULT_HIST_STEP,
    min_elevation: float | None = DEFAULT_MIN_ELEVATION,
    ntasks: int = 1,
    time_block: int = DEFAULT_TIME_BLOCK,
    tile: int = TILE,
    progress: bool = False,
) -> xr.Dataset:
    """
    Count the daily bias per category and elevation bin, tile by tile.

    The store is read one ``tile x tile`` block of the grid at a time (the
    chunks :mod:`pism_terra.glacier.s4f_carra2` writes hold all days of a
    256 x 256 block), its bias computed in blocks of *time_block* days and
    added to a histogram of :func:`value_edges` bins. Memory is bounded by a
    tile, not by the domain or the number of days kept; the histograms are
    small and every summary -- daily medians for the fit, annual quantiles,
    violin shapes -- is read off them, to within a fraction of *step*.

    Parameters
    ----------
    ds : xarray.Dataset
        CARRA2 store, from :func:`open_carra2`.
    on_glacier : xarray.DataArray
        Mask on ``(y, x)``, from :func:`glacier_mask`.
    bin_width : float, default :data:`DEFAULT_BIN_WIDTH`
        Width of the elevation bins, m.
    below : int or None, default :data:`DEFAULT_BELOW`
        See :func:`fit_to_ground`.
    step : float, default :data:`DEFAULT_HIST_STEP`
        Width of a histogram bin, K.
    min_elevation : float or None, default :data:`DEFAULT_MIN_ELEVATION`
        Cells whose CARRA2 terrain lies below this are taken for sea and
        left out; None keeps them.
    ntasks : int, default 1
        Tiles processed at once. Each needs a few GB.
    time_block : int, default :data:`DEFAULT_TIME_BLOCK`
        Days computed at once within a tile.
    tile : int, default :data:`TILE`
        Tile size in cells; match the store's chunks so each is read once.
    progress : bool, default False
        Show a progress bar over the tiles.

    Returns
    -------
    xarray.Dataset
        ``counts`` on ``(time, category, centre, value)`` and ``cells`` (grid
        cells per category and bin) on ``(category, centre)``; ``value`` holds
        the histogram bins' centres and ``attrs`` the ``step``, the
        ``min_elevation`` and how many cells it left out (``n_sea``).
    """
    orography = np.asarray(ds["orography"].transpose("y", "x").values, dtype=float)
    used = np.isfinite(orography)
    if min_elevation is not None:
        used &= orography >= min_elevation
    centre = bin_centre(orography, bin_width)
    centres = np.unique(centre[used])
    elevation_bin = np.where(used, np.searchsorted(centres, np.where(used, centre, centres[0])), -1)
    category = np.asarray(on_glacier.transpose("y", "x").values, dtype=np.int64)
    edges = value_edges(step)
    fields = ds[list(REQUIRED_VARS)]
    n_y, n_x = orography.shape
    tiles = [(slice(y0, y0 + tile), slice(x0, x0 + tile)) for y0 in range(0, n_y, tile) for x0 in range(0, n_x, tile)]

    def work(rows: slice, cols: slice) -> tuple[np.ndarray, int]:
        """
        Load one tile and histogram it.

        Parameters
        ----------
        rows, cols : slice
            The tile.

        Returns
        -------
        tuple
            See :func:`_tile_counts`.
        """
        part = fields.isel(y=rows, x=cols).load()
        return _tile_counts(
            part, category[rows, cols], elevation_bin[rows, cols], centres.size, edges, below, time_block
        )

    counts = np.zeros((ds.sizes["time"], len(CATEGORIES), centres.size, edges.size - 1), dtype=np.int64)
    clipped = 0
    with ThreadPoolExecutor(max_workers=max(1, ntasks)) as pool:
        futures = [pool.submit(work, rows, cols) for rows, cols in tiles]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tiles", disable=not progress):
            tile_counts, tile_clipped = future.result()
            counts += tile_counts
            clipped += tile_clipped
    if clipped:
        logger.warning("%d bias values beyond %s K counted in the histogram's end bins", clipped, HIST_RANGE)
    cells = np.zeros((len(CATEGORIES), centres.size), dtype=np.int64)
    np.add.at(cells, (category[used], elevation_bin[used]), 1)
    return xr.Dataset(
        {
            "counts": (("time", "category", "centre", "value"), counts),
            "cells": (("category", "centre"), cells),
        },
        coords={
            "time": ds["time"].values,
            "category": list(CATEGORIES),
            "centre": centres,
            "value": 0.5 * (edges[:-1] + edges[1:]),
        },
        attrs={
            "step": step,
            "min_elevation": "none" if min_elevation is None else min_elevation,
            "n_sea": int((~used).sum()),
        },
    )


def histogram_quantiles(counts: np.ndarray, edges: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    """
    Read quantiles off histograms, interpolating linearly within a bin.

    Parameters
    ----------
    counts : numpy.ndarray
        Histograms along the last axis.
    edges : numpy.ndarray
        Their edges.
    quantiles : sequence of float
        Quantiles in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        ``counts.shape[:-1] + (len(quantiles),)``; NaN where a histogram is empty.
    """
    counts = np.asarray(counts, dtype=float)
    cumulative = np.cumsum(counts, axis=-1)
    total = cumulative[..., -1]
    step = edges[1:] - edges[:-1]
    out = np.full(counts.shape[:-1] + (len(quantiles),), np.nan)
    for i, q in enumerate(quantiles):
        target = q * total
        k = np.argmax(cumulative >= target[..., None], axis=-1)
        before = np.where(k > 0, np.take_along_axis(cumulative, np.maximum(k - 1, 0)[..., None], -1)[..., 0], 0.0)
        in_bin = np.take_along_axis(counts, k[..., None], -1)[..., 0]
        fraction = np.divide(target - before, in_bin, out=np.zeros_like(target), where=in_bin > 0)
        out[..., i] = np.where(total > 0, edges[k] + fraction * step[k], np.nan)
    return out


def _edges_of(hist: xr.Dataset) -> np.ndarray:
    """
    Recover the histogram edges from a dataset of :func:`bias_histograms`.

    Parameters
    ----------
    hist : xarray.Dataset
        Histograms with a ``value`` coordinate and a ``step`` attribute.

    Returns
    -------
    numpy.ndarray
        The edges.
    """
    values = hist["value"].values
    return np.append(values - hist.attrs["step"] / 2, values[-1] + hist.attrs["step"] / 2)


def annual_histograms(hists: Sequence[xr.Dataset]) -> xr.Dataset:
    """
    Pool histograms over days and stores.

    Parameters
    ----------
    hists : sequence of xarray.Dataset
        Outputs of :func:`bias_histograms`, all with the same ``step``; their
        elevation bins may differ.

    Returns
    -------
    xarray.Dataset
        ``counts`` on ``(category, centre, value)`` and ``cells`` on
        ``(category, centre)``, over the union of the elevation bins.
    """
    pooled = [h[["counts", "cells"]].assign(counts=h["counts"].sum("time")) for h in hists]
    aligned = xr.align(*pooled, join="outer", fill_value=0)
    total = sum(aligned[1:], aligned[0])
    total.attrs = dict(hists[0].attrs)
    return total


def binned_medians(annual: xr.Dataset) -> pd.DataFrame:
    """
    Summarise the bias per category and elevation bin.

    Parameters
    ----------
    annual : xarray.Dataset
        Output of :func:`annual_histograms`.

    Returns
    -------
    pandas.DataFrame
        ``category``, ``centre``, ``median``, ``q25``, ``q75``, ``count``
        (cell-days) and ``cells`` (grid cells), for bins with data.
    """
    counts = annual["counts"].transpose("category", "centre", "value").values
    q = histogram_quantiles(counts, _edges_of(annual), (0.5, 0.25, 0.75))
    category, centre = np.meshgrid(annual["category"].values, annual["centre"].values, indexing="ij")
    frame = pd.DataFrame(
        {
            "category": category.ravel(),
            "centre": centre.ravel(),
            "median": q[..., 0].ravel(),
            "q25": q[..., 1].ravel(),
            "q75": q[..., 2].ravel(),
            "count": counts.sum(-1).ravel(),
            "cells": annual["cells"].transpose("category", "centre").values.ravel(),
        }
    )
    return frame[frame["count"] > 0].reset_index(drop=True)


def year_fraction(time: Sequence[Any] | np.ndarray | pd.Series) -> np.ndarray:
    """
    Turn daily time stamps into the fraction of the year at the middle of each day.

    Daily means are stamped at 00:00 and cover the whole day, so a day counts
    at its midpoint: 1 January is ``0.5 / 365``. Leap years have 366 days, so
    every year runs from 0 to 1 and several years share one seasonal phase.

    Parameters
    ----------
    time : array-like
        Time stamps.

    Returns
    -------
    numpy.ndarray
        Fractions in ``(0, 1)``.
    """
    times = pd.to_datetime(pd.Series(np.asarray(time)))
    days = np.where(times.dt.is_leap_year, 366.0, 365.0)
    return ((times.dt.dayofyear.to_numpy() - 0.5) / days).astype(float)


def _design(z: np.ndarray, fraction: np.ndarray, elevation_dependent: bool) -> np.ndarray:
    """
    Build the regressors of the seasonal model.

    Parameters
    ----------
    z : numpy.ndarray
        Elevations, m.
    fraction : numpy.ndarray
        Fraction of the year, from :func:`year_fraction`.
    elevation_dependent : bool
        Let the seasonal terms vary linearly with elevation.

    Returns
    -------
    numpy.ndarray
        Columns in the order of :data:`COEFFICIENTS`, the last two only when
        *elevation_dependent*.
    """
    z_km = np.asarray(z, dtype=float) / 1000.0
    angle = 2.0 * np.pi * np.asarray(fraction, dtype=float)
    cos, sin = np.cos(angle), np.sin(angle)
    columns = [np.ones_like(z_km), z_km, cos, sin]
    if elevation_dependent:
        columns += [z_km * cos, z_km * sin]
    return np.column_stack(columns)


def fit_seasonal(
    z: Sequence[float] | np.ndarray | pd.Series,
    fraction: Sequence[float] | np.ndarray | pd.Series,
    y: Sequence[float] | np.ndarray | pd.Series,
    elevation_dependent: bool = True,
) -> dict[str, float] | None:
    """
    Fit a bias that is linear in elevation and follows one seasonal cycle.

    The model, with ``z`` in km and ``f`` the fraction of the year, is::

        b(z, f) = (mean + mean_per_km z)
                + (cos + cos_per_km z) cos(2 pi f)
                + (sin + sin_per_km z) sin(2 pi f)

    Without *elevation_dependent* the two ``_per_km`` seasonal terms are
    zero: every elevation then shares one seasonal cycle and only the
    annual mean changes with height. Either way it is linear least squares.
    The period is one year, so data from several years pool into one cycle.

    Parameters
    ----------
    z : array-like
        Elevations, m.
    fraction : array-like
        Fraction of the year of each value, from :func:`year_fraction`.
    y : array-like
        Values, e.g. daily median bias in K.
    elevation_dependent : bool, default True
        Let the seasonal amplitude and phase vary linearly with elevation.

    Returns
    -------
    dict or None
        The coefficients of :data:`COEFFICIENTS` (K, or K per km), ``rmse``
        (K), ``r2`` and ``n``. None when there are no more values than
        coefficients, or the values span a single elevation or a single day
        of the year.
    """
    z_arr = np.asarray(z, dtype=float)
    f_arr = np.asarray(fraction, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    keep = np.isfinite(z_arr) & np.isfinite(f_arr) & np.isfinite(y_arr)
    z_arr, f_arr, y_arr = z_arr[keep], f_arr[keep], y_arr[keep]
    design = _design(z_arr, f_arr, elevation_dependent)
    if y_arr.size <= design.shape[1] or np.ptp(z_arr) == 0 or np.ptp(f_arr) == 0:
        return None
    coef, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    residual = y_arr - design @ coef
    sse = float(residual @ residual)
    sst = float(((y_arr - y_arr.mean()) ** 2).sum())
    fit = dict(zip(COEFFICIENTS, np.append(coef, np.zeros(len(COEFFICIENTS) - coef.size)).tolist()))
    fit.update(
        {"rmse": math.sqrt(sse / y_arr.size), "r2": 1.0 - sse / sst if sst > 0 else math.nan, "n": float(y_arr.size)}
    )
    return fit


def seasonal_curve(
    fit: dict[str, float], z: Sequence[float] | np.ndarray | float, fraction: Sequence[float] | np.ndarray | float
) -> np.ndarray:
    """
    Evaluate a fit from :func:`fit_seasonal`.

    Parameters
    ----------
    fit : dict
        The fit.
    z : array-like or float
        Elevations, m.
    fraction : array-like or float
        Fractions of the year; broadcast against *z*.

    Returns
    -------
    numpy.ndarray
        The fitted bias, K.
    """
    z_b, f_b = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(fraction, dtype=float))
    coef = np.array([fit[name] for name in COEFFICIENTS])
    return (_design(z_b.ravel(), f_b.ravel(), elevation_dependent=True) @ coef).reshape(z_b.shape)


def seasonal_cycle(fit: dict[str, float], z: Sequence[float] | np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """
    Give the amplitude and timing of the fitted seasonal cycle at an elevation.

    Parameters
    ----------
    fit : dict
        A fit from :func:`fit_seasonal`.
    z : array-like or float
        Elevations, m.

    Returns
    -------
    amplitude : numpy.ndarray
        Half the range of the seasonal cycle, K.
    day_of_max : numpy.ndarray
        Day of the year the bias peaks, 1 to 365.
    """
    z_km = np.asarray(z, dtype=float) / 1000.0
    c = fit["cos"] + fit["cos_per_km"] * z_km
    s = fit["sin"] + fit["sin_per_km"] * z_km
    phase = np.mod(np.arctan2(s, c), 2.0 * np.pi)
    return np.hypot(c, s), phase / (2.0 * np.pi) * 365.0 + 0.5


def daily_medians(hist: xr.Dataset, source: str = "") -> pd.DataFrame:
    """
    Take the median bias per day, category and elevation bin.

    Parameters
    ----------
    hist : xarray.Dataset
        Output of :func:`bias_histograms`.
    source : str, default ""
        Name of the store, kept in a ``source`` column.

    Returns
    -------
    pandas.DataFrame
        ``source``, ``time``, ``category``, ``centre``, ``median`` (K),
        ``cells`` (grid cells behind the median that day) and ``fraction``
        (of the year, see :func:`year_fraction`), for every day and bin with
        data.
    """
    counts = hist["counts"].transpose("time", "category", "centre", "value").values
    median = histogram_quantiles(counts, _edges_of(hist), (0.5,))[..., 0]
    cells = counts.sum(-1)
    day, cat, cen = np.nonzero(cells)
    daily = pd.DataFrame(
        {
            "source": source,
            "time": hist["time"].values[day],
            "category": hist["category"].values[cat],
            "centre": hist["centre"].values[cen],
            "median": median[day, cat, cen],
            "cells": cells[day, cat, cen],
        }
    )
    daily["fraction"] = year_fraction(daily["time"])
    return daily


def fit_categories(
    daily: pd.DataFrame, min_cells: int = DEFAULT_MIN_CELLS, elevation_dependent: bool = True
) -> pd.DataFrame:
    """
    Fit the seasonal model to the daily medians, per category.

    Parameters
    ----------
    daily : pandas.DataFrame
        Output of :func:`daily_medians`.
    min_cells : int, default :data:`DEFAULT_MIN_CELLS`
        Daily medians resting on fewer grid cells are left out.
    elevation_dependent : bool, default True
        See :func:`fit_seasonal`.

    Returns
    -------
    pandas.DataFrame
        One row per category that could be fitted, indexed by category:
        the fields of :func:`fit_seasonal`, the elevation range of the bins
        used (``z_min``, ``z_max``) and ``elevation_dependent``.
    """
    rows = {}
    for category in CATEGORIES:
        used = daily[(daily["category"] == category) & (daily["cells"] >= min_cells)]
        fit = fit_seasonal(used["centre"], used["fraction"], used["median"], elevation_dependent)
        if fit is None:
            logger.warning("%s: too few daily medians with %d or more cells for a seasonal fit", category, min_cells)
            continue
        fit["z_min"] = float(used["centre"].min())
        fit["z_max"] = float(used["centre"].max())
        fit["elevation_dependent"] = float(elevation_dependent)
        rows[category] = fit
    fits = pd.DataFrame.from_dict(rows, orient="index")
    fits.index.name = "category"
    return fits


def seasonal_by_elevation(fits: pd.DataFrame, centres: Sequence[float] | np.ndarray) -> pd.DataFrame:
    """
    Tabulate the fitted annual mean and seasonal cycle at each elevation bin.

    Parameters
    ----------
    fits : pandas.DataFrame
        Output of :func:`fit_categories`.
    centres : array-like
        Bin centres, m; only those within each category's fitted range are kept.

    Returns
    -------
    pandas.DataFrame
        ``category``, ``centre``, ``annual_mean``, ``amplitude``,
        ``day_of_max``, ``seasonal_min`` and ``seasonal_max`` (K, day of year).
    """
    rows = []
    for category, fit_row in fits.iterrows():
        fit = fit_row.to_dict()
        z = np.asarray([c for c in centres if fit["z_min"] <= c <= fit["z_max"]], dtype=float)
        mean = fit["mean"] + fit["mean_per_km"] * z / 1000.0
        amplitude, day_of_max = seasonal_cycle(fit, z)
        rows.append(
            pd.DataFrame(
                {
                    "category": category,
                    "centre": z,
                    "annual_mean": mean,
                    "amplitude": amplitude,
                    "day_of_max": day_of_max,
                    "seasonal_min": mean - amplitude,
                    "seasonal_max": mean + amplitude,
                }
            )
        )
    columns = ["category", "centre", "annual_mean", "amplitude", "day_of_max", "seasonal_min", "seasonal_max"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def _style_violins(parts: dict[str, Any], color: str) -> None:
    """
    Colour a violin plot's bodies and medians.

    Parameters
    ----------
    parts : dict
        What :meth:`matplotlib.axes.Axes.violinplot` returns.
    color : str
        Body colour.
    """
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("k")
        body.set_linewidth(0.5)
        body.set_alpha(0.7)
    parts["cmedians"].set_color("k")


def _histogram_violin(
    ax: Any, values: np.ndarray, counts: np.ndarray, position: float, half_width: float, color: str, step: float
) -> None:
    """
    Draw one horizontal violin from a histogram.

    The histogram is smoothed with a 0.25 K Gaussian, scaled to *half_width*
    at its peak, and trimmed to its 0.1 to 99.9 percent range so lone
    extremes do not draw a hairline across the axis. The median is a black tick.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    values : numpy.ndarray
        Centres of the histogram bins, K.
    counts : numpy.ndarray
        The histogram.
    position : float
        Elevation the violin is centred on, m.
    half_width : float
        Half the violin's height at its widest, m.
    color : str
        Body colour.
    step : float
        Width of a histogram bin, K.
    """
    sigma = max(1.0, 0.25 / step)
    offsets = np.arange(-int(3 * sigma), int(3 * sigma) + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    density = np.convolve(counts.astype(float), kernel / kernel.sum(), mode="same")
    cumulative = np.cumsum(counts) / counts.sum()
    low, high = np.searchsorted(cumulative, 0.001), np.searchsorted(cumulative, 0.999) + 1
    x = values[low:high]
    width = density[low:high] / density.max() * half_width
    ax.fill_between(x, position - width, position + width, facecolor=color, edgecolor="k", lw=0.5, alpha=0.7)
    edges = np.append(values - step / 2, values[-1] + step / 2)
    median = float(histogram_quantiles(counts, edges, (0.5,))[0])
    at_median = float(np.interp(median, x, width))
    ax.plot([median, median], [position - at_median, position + at_median], color="k", lw=1.5)


def plot_bias_by_elevation(
    annual: xr.Dataset,
    medians: pd.DataFrame,
    fits: pd.DataFrame,
    bin_width: float = DEFAULT_BIN_WIDTH,
    title: str | None = None,
) -> Figure:
    """
    Plot the bias distribution per elevation bin, on and off glacier, with the fitted cycle.

    Elevation runs up the y axis; each bin gets one horizontal violin per
    category, offset so the pair sits side by side, drawn from the pooled
    histograms. The number of grid cells behind each violin is printed at
    the right edge. The fit is drawn at the true bin elevations as its
    annual mean, with a band spanning the seasonal cycle's minimum to maximum.

    Parameters
    ----------
    annual : xarray.Dataset
        Output of :func:`annual_histograms`.
    medians : pandas.DataFrame
        Output of :func:`binned_medians`.
    fits : pandas.DataFrame
        Output of :func:`fit_categories`.
    bin_width : float, default :data:`DEFAULT_BIN_WIDTH`
        Width of the bins, m.
    title : str or None, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    centres = np.sort(medians["centre"].unique())
    values = annual["value"].values
    step = float(annual.attrs["step"])
    fig, ax = plt.subplots(figsize=(8, max(5.0, 0.5 * centres.size + 2)), constrained_layout=True)
    for category, (color, shift) in CATEGORIES.items():
        rows = medians[(medians["category"] == category) & (medians["count"] > 1)]
        for centre, cells in zip(rows["centre"], rows["cells"]):
            counts = annual["counts"].sel(category=category, centre=centre).values
            position = centre + shift * bin_width
            _histogram_violin(ax, values, counts, position, 0.225 * bin_width, color, step)
            ax.annotate(
                str(int(cells)),
                (1.0, position),
                xycoords=("axes fraction", "data"),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=color,
            )
        if not rows.empty:
            ax.plot([], [], "s", color=color, alpha=0.7, label=category.replace("_", " "))

        if category in fits.index:
            fit = fits.loc[category].to_dict()
            z_line = np.linspace(fit["z_min"], fit["z_max"], 200)
            mean = fit["mean"] + fit["mean_per_km"] * z_line / 1000.0
            amplitude, _ = seasonal_cycle(fit, z_line)
            ax.fill_betweenx(z_line, mean - amplitude, mean + amplitude, color=color, alpha=0.15, lw=0, zorder=3)
            ax.plot(mean, z_line, color=color, lw=2.2, zorder=4)

    edges = np.append(centres - bin_width / 2, centres[-1] + bin_width / 2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(edges)
    ax.set_ylim(edges[0], edges[-1])
    ax.set_xlabel("Level fit at the terrain − 2 m air temperature (K)")
    ax.set_ylabel(f"CARRA2 terrain elevation (m), {bin_width:g} m bins")
    ax.set_title(title or "Free-air extrapolation vs CARRA2 2 m temperature, every grid cell", fontsize=11)
    ax.legend(loc="upper right", title="cell centre")
    ax.grid(axis="x", alpha=0.3)
    return fig


def plot_seasonal_fit(daily: pd.DataFrame, fits: pd.DataFrame, min_cells: int = DEFAULT_MIN_CELLS) -> Figure:
    """
    Plot the daily median bias against the day of the year, with the fitted cycle.

    One panel per category. Each elevation bin's daily medians are dots and
    its fitted cycle a line, both coloured by the bin's elevation; years
    overlay, since the fit has one cycle for all of them. Only the medians
    the fit used (resting on at least *min_cells* cells) are shown.

    Parameters
    ----------
    daily : pandas.DataFrame
        Output of :func:`daily_medians`.
    fits : pandas.DataFrame
        Output of :func:`fit_categories`.
    min_cells : int, default :data:`DEFAULT_MIN_CELLS`
        Fewest cells behind a median shown.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    used = daily[daily["cells"] >= min_cells]
    norm = matplotlib.colors.Normalize(float(used["centre"].min()), float(used["centre"].max()))
    cmap = matplotlib.colormaps["viridis"]
    fig, axes = plt.subplots(1, len(CATEGORIES), figsize=(12, 4.8), sharey=True, constrained_layout=True)
    days = np.arange(1, 366)
    # Day of the year each month starts on, in a 365-day year.
    month_starts = 1 + np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
    for ax, category in zip(np.atleast_1d(axes), CATEGORIES):
        rows = used[used["category"] == category]
        doy = rows["fraction"].to_numpy() * 365.0 + 0.5
        ax.scatter(doy, rows["median"], c=cmap(norm(rows["centre"])), s=3, alpha=0.35, lw=0)
        if category in fits.index:
            fit = fits.loc[category].to_dict()
            for centre in np.sort(rows["centre"].unique()):
                curve = seasonal_curve(fit, centre, (days - 0.5) / 365.0)
                ax.plot(days, curve, color=cmap(norm(centre)), lw=1.6)
            ax.set_title(f"{category.replace('_', ' ')}: RMSE {fit['rmse']:.2f} K, R² {fit['r2']:.2f}", fontsize=10)
        else:
            ax.set_title(f"{category.replace('_', ' ')}: no fit", fontsize=10)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(month_starts, [m[0] for m in MONTH_LABELS])
        ax.set_xlim(1, 366)
        ax.set_xlabel("Day of the year")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Daily median bias per elevation bin (K)")
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="Bin elevation (m)")
    fig.suptitle("Free-air extrapolation vs CARRA2 2 m temperature: fit linear in elevation with a yearly cycle")
    return fig


def site_bias(ds: xr.Dataset, sites: gpd.GeoDataFrame, below: int | None = DEFAULT_BELOW) -> pd.DataFrame | None:
    """
    Sample a store at point sites and compute the daily bias there.

    Parameters
    ----------
    ds : xarray.Dataset
        CARRA2 store.
    sites : geopandas.GeoDataFrame
        Points with a CRS, labelled by a ``site`` column or their index.
    below : int or None, default :data:`DEFAULT_BELOW`
        See :func:`fit_to_ground`.

    Returns
    -------
    pandas.DataFrame or None
        ``site``, ``time``, ``orography`` and ``bias``; None when no site
        lies inside the store's grid.
    """
    try:
        points = sample_points(ds, sites, variables=list(REQUIRED_VARS))
    except ValueError as exc:
        logger.warning("no site sampled: %s", exc)
        return None
    points = points.load()
    bias = ground_bias(points, below=below)
    frame = bias.to_dataframe().reset_index()[["site", "time", "bias"]]
    orography = points["orography"].to_series()
    frame["orography"] = frame["site"].map(orography)
    return frame


def _monthly_violins(ax: Any, rows: pd.DataFrame) -> None:
    """
    Draw one grey violin per month of a bias table.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    rows : pandas.DataFrame
        Rows with ``time`` and ``bias``.
    """
    months = pd.to_datetime(rows["time"]).dt.month
    data = {m: rows.loc[months == m, "bias"].dropna().to_numpy() for m in range(1, 13)}
    present = [m for m, values in data.items() if values.size > 1]
    if present:
        parts = ax.violinplot(
            [data[m] for m in present], positions=present, widths=0.85, showmedians=True, showextrema=False
        )
        _style_violins(parts, "0.8")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(1, 13), MONTH_LABELS)
    ax.grid(axis="y", alpha=0.3)


def plot_sites_pooled(frame: pd.DataFrame) -> Figure:
    """
    Plot the bias per month for all sites pooled, with each site's monthly median.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`site_bias`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    _monthly_violins(ax, frame)
    sites = list(dict.fromkeys(frame["site"]))
    medians = frame.assign(month=pd.to_datetime(frame["time"]).dt.month).groupby(["site", "month"])["bias"].median()
    offsets = np.linspace(-0.25, 0.25, len(sites)) if len(sites) > 1 else [0.0]
    for color, dx, site in zip(matplotlib.colormaps["tab10"].colors, offsets, sites):
        series = medians.loc[site]
        ax.plot(series.index + dx, series.to_numpy(), "o", ms=5, color=color, mec="k", mew=0.4, label=str(site))
    ax.set_ylabel("Level fit at the terrain − 2 m air temperature (K)")
    ax.set_title("Free-air extrapolation vs CARRA2 2 m temperature, all sites pooled", fontsize=11)
    ax.legend(title="site median", ncol=3, fontsize=8, loc="upper left")
    return fig


def plot_sites(frame: pd.DataFrame, ncols: int = 2) -> Figure:
    """
    Plot the bias per month in one panel per site.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`site_bias`.
    ncols : int, default 2
        Panels per row.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    sites = list(dict.fromkeys(frame["site"]))
    nrows = math.ceil(len(sites) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 3.3 * nrows), sharex=True, sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, site in zip(axes, sites):
        rows = frame[frame["site"] == site]
        _monthly_violins(ax, rows)
        ax.set_title(
            f"{site}  (CARRA2 terrain {rows['orography'].iloc[0]:.0f} m, "
            f"annual median {rows['bias'].median():+.1f} K)",
            fontsize=10,
        )
    for ax in axes[len(sites) :]:
        ax.set_visible(False)
    for ax in axes[::ncols]:
        ax.set_ylabel("Fit at terrain − 2 m T (K)")
    fig.suptitle("Free-air extrapolation vs CARRA2 2 m temperature, per site")
    return fig


def _source_name(path: str | Path) -> str:
    """
    Name a store by its file name, without the extension.

    Parameters
    ----------
    path : str or Path
        Local path or URI.

    Returns
    -------
    str
        E.g. ``carra2_daily_RGI2000-v7.0-C-01-04374``.
    """
    name = str(path).rstrip("/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


def _parse_below(value: str) -> int | None:
    """
    Parse ``--below``: a count of levels, or ``all``.

    Parameters
    ----------
    value : str
        Command-line value.

    Returns
    -------
    int or None
        The count, or None for ``all``.

    Raises
    ------
    ValueError
        If the value is negative.
    """
    if value.lower() == "all":
        return None
    count = int(value)
    if count < 0:
        raise ValueError("--below must be 0 or more, or 'all'")
    return count


def _parse_min_elevation(value: str) -> float | None:
    """
    Parse ``--min-elevation``: metres, or ``none``.

    Parameters
    ----------
    value : str
        Command-line value.

    Returns
    -------
    float or None
        The elevation, or None to keep every cell.
    """
    return None if value.lower() == "none" else float(value)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the bias analysis from the command line.

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
    parser.description = (
        "Compare CARRA2's free-air temperature, fitted through the pressure levels and carried down to the "
        "terrain, with its 2 m air temperature: on and off glacier, by elevation, and optionally at sites."
    )
    parser.add_argument(
        "CARRA2", nargs="+", help="Daily CARRA2 stores from pism-s4f-carra2 (.zarr or NetCDF), local or s3://."
    )
    parser.add_argument(
        "--outlines",
        nargs="+",
        required=True,
        help="Glacier outline files (GeoPackage, shapefile, ...), local or s3://.",
    )
    parser.add_argument("--sites", default=None, help="Point sites to plot the monthly bias at, local or s3://.")
    parser.add_argument("--output-path", type=Path, default=Path("carra2_bias"), help="Directory for the results.")
    parser.add_argument("--bin-width", type=float, default=DEFAULT_BIN_WIDTH, help="Width of the elevation bins (m).")
    parser.add_argument(
        "--below",
        type=_parse_below,
        default=DEFAULT_BELOW,
        help="CARRA2 levels below the terrain each daily fit may use besides those above it; 'all' for every level.",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=DEFAULT_MIN_CELLS,
        help="Daily bin medians resting on fewer grid cells stay out of the fit.",
    )
    parser.add_argument(
        "--season",
        choices=("elevation", "constant"),
        default="elevation",
        help="Seasonal amplitude and phase linear in elevation, or one cycle shared by all elevations.",
    )
    parser.add_argument(
        "--months", type=int, nargs="+", default=None, help="Use only these months, e.g. 6 7 8; all when omitted."
    )
    parser.add_argument(
        "--min-elevation",
        type=_parse_min_elevation,
        default=DEFAULT_MIN_ELEVATION,
        help="Cells whose CARRA2 terrain lies below this (m) are taken for sea and left out; 'none' keeps all.",
    )
    parser.add_argument(
        "--hist-step", type=float, default=DEFAULT_HIST_STEP, help="Bin width of the bias histograms (K)."
    )
    parser.add_argument(
        "--time-block", type=int, default=DEFAULT_TIME_BLOCK, help="Days computed at once within a tile."
    )
    add_ntasks_argument(parser, default=4, help_text="Grid tiles processed at once; each needs a few GB of memory.")
    parser.add_argument(
        "--anon",
        default=True,
        action="store_true",
        help="Read S3 without credentials (the default; the bucket must be public).",
    )
    parser.add_argument("--no-anon", dest="anon", action="store_false", help="Read S3 with your AWS credentials.")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution of the figures.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output = args.output_path
    output.mkdir(parents=True, exist_ok=True)
    setup_logging(output / "carra2_bias_correction.log")

    outlines = read_vectors(args.outlines, anon=args.anon)
    sites = read_vector(args.sites, anon=args.anon) if args.sites else None
    logger.info("%d outlines from %s", len(outlines), args.outlines)

    hists, dailies, site_frames = [], [], []
    for path in args.CARRA2:
        print(f"Reading {path}...", flush=True)
        ds = open_carra2(path, anon=args.anon)
        if args.months:
            ds = ds.sel(time=ds["time"].dt.month.isin(args.months))
            if ds.sizes["time"] == 0:
                raise ValueError(f"{path} has no days in months {args.months}")
        source = _source_name(path)
        on_glacier = glacier_mask(ds, outlines)
        hist = bias_histograms(
            ds,
            on_glacier,
            bin_width=args.bin_width,
            below=args.below,
            step=args.hist_step,
            min_elevation=args.min_elevation,
            ntasks=args.ntasks,
            time_block=args.time_block,
            progress=True,
        )
        logger.info(
            "%s: %d of %d cells on glacier, %d left out as sea",
            source,
            int(on_glacier.sum()),
            on_glacier.size,
            hist.attrs["n_sea"],
        )
        hists.append(hist)
        dailies.append(daily_medians(hist, source))
        if sites is not None:
            frame = site_bias(ds, sites, below=args.below)
            if frame is not None:
                site_frames.append(frame.assign(source=source))

    annual = annual_histograms(hists)
    medians = binned_medians(annual)
    daily = pd.concat(dailies, ignore_index=True)
    fits = fit_categories(daily, min_cells=args.min_cells, elevation_dependent=args.season == "elevation")
    cycle = seasonal_by_elevation(fits, annual["centre"].values)
    medians.to_csv(output / "bias_by_elevation_medians.csv", index=False)
    daily.to_csv(output / "bias_daily_medians.csv", index=False)
    fits.to_csv(output / "bias_seasonal_fit.csv")
    cycle.to_csv(output / "bias_seasonal_by_elevation.csv", index=False)
    for name, figure in (
        ("bias_by_elevation.png", plot_bias_by_elevation(annual, medians, fits, bin_width=args.bin_width)),
        ("bias_seasonal_fit.png", plot_seasonal_fit(daily, fits, min_cells=args.min_cells)),
    ):
        figure.savefig(output / name, dpi=args.dpi)
        plt.close(figure)

    if site_frames:
        frame = pd.concat(site_frames, ignore_index=True)
        frame.to_csv(output / "site_bias.csv", index=False)
        for name, figure in (
            ("site_bias_monthly.png", plot_sites_pooled(frame)),
            ("site_bias_per_site.png", plot_sites(frame)),
        ):
            figure.savefig(output / name, dpi=args.dpi)
            plt.close(figure)

    if not fits.empty:
        print(fits[[*COEFFICIENTS, "rmse", "r2", "n"]].round(3).to_string())
    print(f"Results in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
