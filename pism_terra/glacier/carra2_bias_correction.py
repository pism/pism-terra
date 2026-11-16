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
bias of each bin is fitted with an annual elevation profile plus one
seasonal cycle (period one year) whose amplitude and phase may also change
linearly with elevation; see :func:`fit_seasonal`. Two profiles are fitted
and compared by BIC: a straight line, and a bias decaying exponentially
with elevation toward a constant, which follows the strong excess over the
lower glacier that levels off higher up. The fit follows the mean by
default (least squares): PISM adds the bias to a daily-mean temperature
that PDD and dEBM integrate around, so the warm tail of large bias -- the
melt days -- has to count. ``--fit median`` makes it follow the median
instead (least absolute deviations), the typical day, which that tail does
not pull up. Several years pool into one cycle. It writes the medians, the fit and the fitted cycle per bin as
CSV, and figures of the bias against elevation and against the day of the
year.

The bias is never held for the whole domain: the store is read one chunk
of the grid at a time and the bias counted into histograms per day,
category and elevation bin (see :func:`bias_histograms`), so memory depends
on the chunk, not on the domain or the number of years. Cells at sea level
are taken for sea and left out (``--min-elevation``).

Two diagnostics say how much a fit of this kind can do and what it leaves
to the melt model. ``r2_ceiling`` is the R² of each bin's own smooth
seasonal climatology (see :func:`ceiling_fit`), the best any bias of
elevation and season alone can reach: the rest is weather. ``sigma_extra``
(see :func:`variance_by_elevation`) is the day-to-day spread of the
free-air temperature at the ground that the corrected forcing -- the 2 m
temperature plus the bias, as PISM's ``carra2_bias`` applies it -- lacks,
with anomalies taken per cell from its own seasonal cycle: positive, as
where the 2 m air is held near 0 degC over melting ice, it is variance to
add to a PDD or dEBM ``std_dev`` in quadrature; negative, the 2 m air
varies more than the free air and the corrected forcing already has too much.

A third checks that the fit is a property of the region rather than of
the years it saw: the first and second halves of the record, in whole
years, are fitted apart and their fitted biases compared (see
:func:`half_record_stability`). Coefficients that differ between regions
by much more than between halves are worth setting per region.

Stores and outlines may each be a local path or an ``s3://`` URI.
"""

import logging
import math
import tempfile
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from matplotlib.figure import Figure
from scipy.optimize import linprog, minimize_scalar
from tqdm.auto import tqdm

from pism_terra.log import setup_logging
from pism_terra.ntasks_option import add_ntasks_argument
from pism_terra.workflow import dataset_crs

logger = logging.getLogger("pism_terra.glacier.carra2_bias_correction")

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
DEFAULT_MIN_ELEVATION = 500.0

#: Tile size in grid cells for a store that is not chunked; a chunked one is
#: read in tiles of its own chunks (see :func:`store_tile`), so each tile
#: reads exactly one chunk of every field.
TILE = 256

#: Days whose bias is computed at once within a tile.
DEFAULT_TIME_BLOCK = 31

#: Annual-mean elevation profiles :func:`fit_seasonal` knows; both are fitted.
ELEVATION_MODELS = ("linear", "exponential")
#: What the fit follows: the mean (least squares) or the median (least
#: absolute deviations).
FIT_LOSSES = ("mean", "median")
#: The mean: the bias corrects a daily-mean forcing, so its long warm tail
#: counts, and a median fit would sit about half a kelvin too cold.
DEFAULT_FIT_LOSS = "mean"
#: Decay heights tried before the bounded refinement, per loss. Each median
#: fit is a linear programme, so its scan is coarser; the refinement still
#: finds the optimum within the bracket the scan picks.
SCALE_HEIGHT_SCAN = {"mean": 121, "median": 41}

#: Range of decay heights the exponential profile is scanned over, m. A fit
#: at the upper end has found no curvature: it is a straight line.
SCALE_HEIGHT_RANGE = (500.0, 4_000.0)

#: Line style of each profile in the elevation figure.
PROFILE_STYLES = {"linear": "--", "exponential": "-"}

#: Seasonal coefficients of the model, see :func:`fit_seasonal`.
SEASONAL_COEFFICIENTS = ("cos", "sin", "cos_per_km", "sin_per_km")

#: Harmonics of the year in the per-bin and per-cell climatologies behind the
#: diagnostics (see :func:`ceiling_fit` and :func:`variance_by_elevation`):
#: smooth enough not to fit the weather, flexible enough to follow any
#: realistic seasonal shape.
DEFAULT_HARMONICS = 3

#: Fewer cell-days than this many times the climatology's coefficients leave
#: a cell out of the variance diagnostics: too short to tell weather from season.
MIN_DAYS_PER_COEFFICIENT = 2

#: Temperatures whose day-to-day anomalies :func:`variance_by_elevation`
#: compares: the 2 m air, the free air carried down to the ground, and their
#: difference, the bias.
ANOMALY_QUANTITIES = ("t2m", "free", "bias")

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
    index (day or cell). CARRA2 reports temperatures on levels below
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
        (the orography).
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
    harmonics: int = DEFAULT_HARMONICS,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Histogram the daily bias of one tile per category and elevation bin.

    Alongside, every cell's 2 m and free-air temperatures are regressed on
    *harmonics* harmonics of the year, from sums accumulated block by block,
    so the day-to-day anomalies' sums of squares come without holding a
    cell's whole series (see :func:`variance_by_elevation`).

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
    harmonics : int, default :data:`DEFAULT_HARMONICS`
        Harmonics of each cell's seasonal cycle.

    Returns
    -------
    counts : numpy.ndarray
        ``(time, category, centre, value)`` counts.
    clipped : int
        Values beyond the histogram's range, counted in its end bins.
    anomaly_ss : numpy.ndarray
        ``(quantity, category, centre)`` sums over cells and days of the
        squared anomalies from each cell's cycle, for the quantities of
        :data:`ANOMALY_QUANTITIES`, K².
    cell_days : numpy.ndarray
        ``(category, centre)`` cell-days behind *anomaly_ss*; cells with
        fewer than :data:`MIN_DAYS_PER_COEFFICIENT` days per coefficient are
        left out.
    """
    n_days, n_values = tile.sizes["time"], edges.size - 1
    n_groups = len(CATEGORIES) * n_centres
    step = float(edges[1] - edges[0])
    keep = elevation_bin >= 0
    group = (category[keep] * n_centres + elevation_bin[keep]).astype(np.int64)
    shape = (n_days, len(CATEGORIES), n_centres, n_values)
    counts = np.zeros(n_days * n_groups * n_values, dtype=np.int64)
    clipped = 0
    anomaly_ss = np.zeros((len(ANOMALY_QUANTITIES), n_groups))
    cell_days = np.zeros(n_groups)
    if not keep.any():
        return (
            counts.reshape(shape),
            clipped,
            anomaly_ss.reshape((-1, len(CATEGORIES), n_centres)),
            cell_days.reshape((len(CATEGORIES), n_centres)),
        )
    # Per cell: the Gram matrix of the harmonic basis over the days with data,
    # the basis times each temperature, and the temperatures' (cross) products.
    n_cells, n_basis = int(keep.sum()), 1 + 2 * harmonics
    gram = np.zeros((n_cells, n_basis, n_basis))
    basis_t2m = np.zeros((n_cells, n_basis))
    basis_free = np.zeros((n_cells, n_basis))
    ss_t2m, ss_free, ss_cross, n = (np.zeros(n_cells) for _ in range(4))
    fraction = year_fraction(tile["time"].values)
    for start in range(0, n_days, time_block):
        block = tile.isel(time=slice(start, start + time_block))
        fit = fit_to_ground(block["air_temp"], block["geopotential_height"], block["orography"], below=below)
        # Centred on 0 degC so the sums keep their precision.
        free = fit["t_ground"].transpose("time", "y", "x").values[:, keep].astype(float) - 273.15
        t2m = block["air_temp_2m"].transpose("time", "y", "x").values[:, keep].astype(float) - 273.15
        values = free - t2m
        finite = np.isfinite(values)

        phi = np.column_stack(_harmonic_columns(fraction[start : start + values.shape[0]], harmonics))
        mask = finite.astype(float)
        free0, t2m0 = np.where(finite, free, 0.0), np.where(finite, t2m, 0.0)
        gram += np.einsum("tcp,tq->cpq", mask[:, :, None] * phi[:, None, :], phi)
        basis_t2m += np.einsum("tc,tp->cp", t2m0, phi)
        basis_free += np.einsum("tc,tp->cp", free0, phi)
        ss_t2m += (t2m0**2).sum(0)
        ss_free += (free0**2).sum(0)
        ss_cross += (t2m0 * free0).sum(0)
        n += mask.sum(0)

        index = np.floor((np.where(finite, values, edges[0]) - edges[0]) / step)
        clipped += int((finite & ((index < 0) | (index >= n_values))).sum())
        index = np.clip(index, 0, n_values - 1).astype(np.int64)
        # Count only this block's days: a bincount over the whole record would
        # allocate the full histogram for every block.
        day = np.arange(values.shape[0], dtype=np.int64)[:, None]
        flat = ((day * n_groups + group[None, :]) * n_values + index)[finite]
        offset = start * n_groups * n_values
        size = values.shape[0] * n_groups * n_values
        counts[offset : offset + size] += np.bincount(flat, minlength=size)

    rss = _residual_sums(gram, basis_t2m, basis_free, ss_t2m, ss_free, ss_cross)
    valid = n >= MIN_DAYS_PER_COEFFICIENT * n_basis
    for k, values in enumerate(rss):
        anomaly_ss[k] = np.bincount(group, weights=np.where(valid, np.maximum(values, 0.0), 0.0), minlength=n_groups)
    cell_days[:] = np.bincount(group, weights=np.where(valid, n, 0.0), minlength=n_groups)
    return (
        counts.reshape(shape),
        clipped,
        anomaly_ss.reshape((-1, len(CATEGORIES), n_centres)),
        cell_days.reshape((len(CATEGORIES), n_centres)),
    )


def _residual_sums(
    gram: np.ndarray,
    basis_t2m: np.ndarray,
    basis_free: np.ndarray,
    ss_t2m: np.ndarray,
    ss_free: np.ndarray,
    ss_cross: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Turn accumulated sums into the residual sums of squares of per-cell regressions.

    For values ``y`` regressed on a basis ``phi``, the residual sum of
    squares is ``S - c^T G^+ c`` with ``S = sum y²``, ``c = sum phi y`` and the
    Gram matrix ``G = sum phi phi^T``; the cross term of two series follows alike.

    Parameters
    ----------
    gram : numpy.ndarray
        ``(cell, basis, basis)`` Gram matrices over each cell's days with data.
    basis_t2m, basis_free : numpy.ndarray
        ``(cell, basis)`` basis times the 2 m and the free-air temperature.
    ss_t2m, ss_free, ss_cross : numpy.ndarray
        ``(cell,)`` sums of the squares and of the product of the two.

    Returns
    -------
    tuple of numpy.ndarray
        Residual sums of squares of the 2 m temperature, the free-air
        temperature and the bias (free air minus 2 m), per cell, K².
    """
    inverse = np.linalg.pinv(gram, rcond=1e-10, hermitian=True)
    coef_t2m = np.einsum("cpq,cq->cp", inverse, basis_t2m)
    coef_free = np.einsum("cpq,cq->cp", inverse, basis_free)
    rss_t2m = ss_t2m - np.einsum("cp,cp->c", basis_t2m, coef_t2m)
    rss_free = ss_free - np.einsum("cp,cp->c", basis_free, coef_free)
    rss_cross = ss_cross - np.einsum("cp,cp->c", basis_t2m, coef_free)
    return rss_t2m, rss_free, rss_free + rss_t2m - 2.0 * rss_cross


def store_tile(ds: xr.Dataset) -> tuple[int, int]:
    """
    Find the tile that reads every field of a store one chunk at a time.

    Parameters
    ----------
    ds : xarray.Dataset
        CARRA2 store opened lazily, from :func:`open_carra2`.

    Returns
    -------
    tuple of int
        Chunk size along ``y`` and ``x``, the smallest over the fields that
        have both; :data:`TILE` along a dimension no field is chunked in.
    """
    sizes: dict[str, list[int]] = {"y": [], "x": []}
    for var in ds.data_vars.values():
        chunks = var.chunksizes
        for dim, found in sizes.items():
            if dim in chunks:
                found.append(int(chunks[dim][0]))
    return min(sizes["y"], default=TILE), min(sizes["x"], default=TILE)


def bias_histograms(
    ds: xr.Dataset,
    on_glacier: xr.DataArray,
    bin_width: float = DEFAULT_BIN_WIDTH,
    below: int | None = DEFAULT_BELOW,
    step: float = DEFAULT_HIST_STEP,
    min_elevation: float | None = DEFAULT_MIN_ELEVATION,
    ntasks: int = 1,
    time_block: int = DEFAULT_TIME_BLOCK,
    tile: int | None = None,
    progress: bool = False,
    harmonics: int = DEFAULT_HARMONICS,
) -> xr.Dataset:
    """
    Count the daily bias per category and elevation bin, tile by tile.

    The store is read one block of the grid at a time, by default one chunk
    of the store (see :func:`store_tile`), which holds all of its days; the
    bias is computed in blocks of *time_block* days and added to a
    histogram of :func:`value_edges` bins. Each worker adds its tile into
    the totals as soon as it is done, so memory is bounded by *ntasks*
    tiles plus one histogram per worker, not by the domain; a tile grows
    with the days kept, which is why it should be a chunk, not more. Every
    summary -- daily medians for the fit, annual quantiles, violin shapes
    -- is read off the histograms, to within a fraction of *step*.

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
        Tiles processed at once. Each holds one tile of every field, all
        days, and one histogram.
    time_block : int, default :data:`DEFAULT_TIME_BLOCK`
        Days computed at once within a tile.
    tile : int or None, optional
        Tile size in cells along y and x; None takes the store's chunks,
        see :func:`store_tile`.
    progress : bool, default False
        Show a progress bar over the tiles.
    harmonics : int, default :data:`DEFAULT_HARMONICS`
        Harmonics of each cell's seasonal cycle, see :func:`_tile_counts`.

    Returns
    -------
    xarray.Dataset
        ``counts`` on ``(time, category, centre, value)`` and ``cells`` (grid
        cells per category and bin) on ``(category, centre)``; for
        :func:`variance_by_elevation`, ``anomaly_ss`` on ``(quantity,
        category, centre)`` and ``cell_days`` on ``(category, centre)``, see
        :func:`_tile_counts`; ``value`` holds
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
    tile_y, tile_x = store_tile(fields) if tile is None else (tile, tile)
    tiles = [
        (slice(y0, y0 + tile_y), slice(x0, x0 + tile_x)) for y0 in range(0, n_y, tile_y) for x0 in range(0, n_x, tile_x)
    ]
    logger.info("%d tiles of %d x %d cells", len(tiles), tile_y, tile_x)

    counts = np.zeros((ds.sizes["time"], len(CATEGORIES), centres.size, edges.size - 1), dtype=np.int64)
    anomaly_ss = np.zeros((len(ANOMALY_QUANTITIES), len(CATEGORIES), centres.size))
    cell_days = np.zeros((len(CATEGORIES), centres.size))
    lock = Lock()

    def work(rows: slice, cols: slice) -> int:
        """
        Load one tile, histogram it and add it to the totals.

        Adding here rather than in the caller frees the tile's histogram as
        soon as it is counted; a finished future would hold it to the end.

        Parameters
        ----------
        rows, cols : slice
            The tile.

        Returns
        -------
        int
            Values beyond the histogram's range, see :func:`_tile_counts`.
        """
        part = fields.isel(y=rows, x=cols).load()
        tile_counts, tile_clipped, tile_ss, tile_days = _tile_counts(
            part,
            category[rows, cols],
            elevation_bin[rows, cols],
            centres.size,
            edges,
            below,
            time_block,
            harmonics,
        )
        del part
        with lock:
            counts[...] += tile_counts
            anomaly_ss[...] += tile_ss
            cell_days[...] += tile_days
        return tile_clipped

    clipped = 0
    with ThreadPoolExecutor(max_workers=max(1, ntasks)) as pool:
        futures = [pool.submit(work, rows, cols) for rows, cols in tiles]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tiles", disable=not progress):
            clipped += future.result()
    if clipped:
        logger.warning("%d bias values beyond %s K counted in the histogram's end bins", clipped, HIST_RANGE)
    cells = np.zeros((len(CATEGORIES), centres.size), dtype=np.int64)
    np.add.at(cells, (category[used], elevation_bin[used]), 1)
    return xr.Dataset(
        {
            "counts": (("time", "category", "centre", "value"), counts),
            "cells": (("category", "centre"), cells),
            "anomaly_ss": (("quantity", "category", "centre"), anomaly_ss),
            "cell_days": (("category", "centre"), cell_days),
        },
        coords={
            "time": ds["time"].values,
            "quantity": list(ANOMALY_QUANTITIES),
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
        ``counts`` on ``(category, centre, value)``, ``cells`` and
        ``cell_days`` on ``(category, centre)`` and ``anomaly_ss`` on
        ``(quantity, category, centre)``, over the union of the elevation
        bins. Each store's anomalies are taken from its own cycle.
    """
    names = ["counts", "cells", "anomaly_ss", "cell_days"]
    pooled = [h[names].assign(counts=h["counts"].sum("time")) for h in hists]
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


def _harmonic_columns(fraction: np.ndarray, harmonics: int = DEFAULT_HARMONICS) -> list[np.ndarray]:
    """
    Build the regressors of a smooth seasonal climatology.

    Parameters
    ----------
    fraction : numpy.ndarray
        Fraction of the year, from :func:`year_fraction`.
    harmonics : int, default :data:`DEFAULT_HARMONICS`
        Harmonics of the year.

    Returns
    -------
    list of numpy.ndarray
        ``1``, then ``cos(2 pi k f)`` and ``sin(2 pi k f)`` for ``k = 1 ... harmonics``.
    """
    angle = 2.0 * np.pi * np.asarray(fraction, dtype=float)
    columns = [np.ones_like(angle)]
    for k in range(1, harmonics + 1):
        columns += [np.cos(k * angle), np.sin(k * angle)]
    return columns


def _seasonal_columns(z: np.ndarray, fraction: np.ndarray, elevation_dependent: bool) -> list[np.ndarray]:
    """
    Build the seasonal regressors.

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
    list of numpy.ndarray
        ``cos``, ``sin`` and, when *elevation_dependent*, ``z cos`` and ``z sin``
        (z in km), in the order of :data:`SEASONAL_COEFFICIENTS`.
    """
    z_km = np.asarray(z, dtype=float) / 1000.0
    angle = 2.0 * np.pi * np.asarray(fraction, dtype=float)
    columns = [np.cos(angle), np.sin(angle)]
    if elevation_dependent:
        columns += [z_km * columns[0], z_km * columns[1]]
    return columns


def _elevation_columns(z: np.ndarray, model: str, scale_height: float = math.nan) -> list[np.ndarray]:
    """
    Build the regressors of the annual-mean elevation profile.

    Parameters
    ----------
    z : numpy.ndarray
        Elevations, m.
    model : {"linear", "exponential"}
        Profile, see :func:`fit_seasonal`.
    scale_height : float, optional
        Decay height of the exponential, m.

    Returns
    -------
    list of numpy.ndarray
        ``1, z`` (z in km) for linear; ``1, exp(-z / H)`` for exponential.
    """
    z = np.asarray(z, dtype=float)
    if model == "linear":
        return [np.ones_like(z), z / 1000.0]
    return [np.ones_like(z), np.exp(-z / scale_height)]


def _least_squares(columns: list[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Solve a linear least-squares problem.

    Parameters
    ----------
    columns : list of numpy.ndarray
        Regressors.
    y : numpy.ndarray
        Values.

    Returns
    -------
    coef : numpy.ndarray
        Coefficients, in the order of *columns*.
    sse : float
        Sum of squared residuals.
    """
    design = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    return coef, float(residual @ residual)


def _least_absolute(columns: list[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Solve a linear least-absolute-deviations problem, which follows the median.

    Solved exactly through the dual of the linear programme
    ``min sum |y - X b|``: ``max y^T d`` subject to ``X^T d = 0`` and
    ``-1 <= d <= 1``. It has one constraint per coefficient rather than one
    per value, which makes it some fifty times faster than the primal on a
    few thousand daily medians; the coefficients are the negated marginals
    of its equality constraints.

    Parameters
    ----------
    columns : list of numpy.ndarray
        Regressors.
    y : numpy.ndarray
        Values.

    Returns
    -------
    coef : numpy.ndarray
        Coefficients, in the order of *columns*.
    sae : float
        Sum of absolute residuals.

    Raises
    ------
    RuntimeError
        If the solver fails.
    """
    design = np.column_stack(columns)
    result = linprog(-y, A_eq=design.T, b_eq=np.zeros(design.shape[1]), bounds=(-1.0, 1.0), method="highs")
    if not result.success:
        raise RuntimeError(f"least-absolute-deviations fit failed: {result.message}")
    coef = -np.asarray(result.eqlin.marginals, dtype=float)
    return coef, float(np.abs(y - design @ coef).sum())


def fit_seasonal(
    z: Sequence[float] | np.ndarray | pd.Series,
    fraction: Sequence[float] | np.ndarray | pd.Series,
    y: Sequence[float] | np.ndarray | pd.Series,
    elevation_dependent: bool = True,
    elevation: str = "linear",
    loss: str = DEFAULT_FIT_LOSS,
) -> dict[str, Any] | None:
    """
    Fit a bias that changes with elevation and follows one seasonal cycle.

    With ``z`` in km and ``f`` the fraction of the year, the model is::

        b(z, f) = m(z) + (cos + cos_per_km z) cos(2 pi f)
                       + (sin + sin_per_km z) sin(2 pi f)

    with an annual-mean profile ``m(z)`` that is either

    * ``linear``: ``mean + mean_per_km z``, or
    * ``exponential``: ``background + excess exp(-z / H)``, a bias that
      decays with elevation toward a constant ``background``, ``excess``
      above it at sea level, falling by 1/e every ``scale_height`` H.
      Over glaciers the bias is large low down and levels off higher up,
      which a straight line cannot follow.

    Without *elevation_dependent* the ``_per_km`` seasonal terms are zero.
    With *loss* ``"mean"`` the fit minimises the squared residuals and
    follows the mean; with ``"median"`` it minimises the absolute residuals
    and follows the median, which a long tail of large values does not pull
    along. For a given H the problem is linear either way; H is found by a scan
    over :data:`SCALE_HEIGHT_RANGE` refined with a bounded 1-D minimisation;
    an H at the top of that range means the data show no curvature. The period
    is one year, so data from several years pool into one cycle.

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
    elevation : {"linear", "exponential"}, default "linear"
        Annual profile.
    loss : {"mean", "median"}, default :data:`DEFAULT_FIT_LOSS`
        What the fit follows: least squares or least absolute deviations.

    Returns
    -------
    dict or None
        ``elevation_model``, ``loss``; the profile's coefficients (``mean`` and
        ``mean_per_km``, or ``background``, ``excess`` and ``scale_height``
        in m; the other model's are NaN); the coefficients of
        :data:`SEASONAL_COEFFICIENTS`; ``rmse`` and ``mae`` (K), ``r2``,
        ``bic`` (Gaussian residuals for the mean fit, Laplace for the
        median fit, so compare BIC within one loss only), ``n_parameters``
        and ``n``. None when there are no more values than
        coefficients, or the values span a single elevation or a single day
        of the year.

    Raises
    ------
    ValueError
        If *elevation* is not one of :data:`ELEVATION_MODELS`, or *loss* not
        one of :data:`FIT_LOSSES`.
    """
    if elevation not in ELEVATION_MODELS:
        raise ValueError(f"unknown elevation model {elevation!r}; use one of {ELEVATION_MODELS}")
    if loss not in FIT_LOSSES:
        raise ValueError(f"unknown fit {loss!r}; use one of {FIT_LOSSES}")
    solve = _least_squares if loss == "mean" else _least_absolute
    z_arr = np.asarray(z, dtype=float)
    f_arr = np.asarray(fraction, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    keep = np.isfinite(z_arr) & np.isfinite(f_arr) & np.isfinite(y_arr)
    z_arr, f_arr, y_arr = z_arr[keep], f_arr[keep], y_arr[keep]
    seasonal = _seasonal_columns(z_arr, f_arr, elevation_dependent)
    n_parameters = 2 + len(seasonal) + (1 if elevation == "exponential" else 0)
    if y_arr.size <= n_parameters or np.ptp(z_arr) == 0 or np.ptp(f_arr) == 0:
        return None

    scale_height = math.nan
    if elevation == "exponential":

        def sse_at(log_h: float) -> float:
            """
            The fit's objective for a decay height.

            Parameters
            ----------
            log_h : float
                Base-10 logarithm of the decay height in m.

            Returns
            -------
            float
                Sum of squared (mean) or absolute (median) residuals.
            """
            return solve(_elevation_columns(z_arr, "exponential", 10.0**log_h) + seasonal, y_arr)[1]

        grid = np.linspace(
            math.log10(SCALE_HEIGHT_RANGE[0]), math.log10(SCALE_HEIGHT_RANGE[1]), SCALE_HEIGHT_SCAN[loss]
        )
        best = int(np.argmin([sse_at(g) for g in grid]))
        low, high = grid[max(best - 1, 0)], grid[min(best + 1, grid.size - 1)]
        scale_height = 10.0 ** float(minimize_scalar(sse_at, bounds=(low, high), method="bounded").x)

    columns = _elevation_columns(z_arr, elevation, scale_height) + seasonal
    coef, objective = solve(columns, y_arr)
    residual = y_arr - np.column_stack(columns) @ coef
    sse, sae = float(residual @ residual), float(np.abs(residual).sum())
    n = y_arr.size
    sst = float(((y_arr - y_arr.mean()) ** 2).sum())
    # -2 log L up to a constant: Gaussian residuals for the mean fit, Laplace
    # for the median fit, each at its maximum-likelihood scale.
    if objective <= 0:
        bic = -math.inf
    elif loss == "mean":
        bic = n * math.log(sse / n) + n_parameters * math.log(n)
    else:
        bic = 2 * n * math.log(sae / n) + n_parameters * math.log(n)
    fit: dict[str, Any] = {"elevation_model": elevation, "loss": loss}
    if elevation == "linear":
        fit.update({"mean": float(coef[0]), "mean_per_km": float(coef[1])})
        fit.update({"background": math.nan, "excess": math.nan, "scale_height": math.nan})
    else:
        fit.update({"mean": math.nan, "mean_per_km": math.nan})
        fit.update({"background": float(coef[0]), "excess": float(coef[1]), "scale_height": scale_height})
    seasonal_coef = np.append(coef[2:], np.zeros(len(SEASONAL_COEFFICIENTS) - (coef.size - 2)))
    fit.update(dict(zip(SEASONAL_COEFFICIENTS, seasonal_coef.tolist())))
    fit.update(
        {
            "rmse": math.sqrt(sse / n),
            "mae": sae / n,
            "r2": 1.0 - sse / sst if sst > 0 else math.nan,
            "bic": bic,
            "n_parameters": float(n_parameters),
            "n": float(n),
        }
    )
    return fit


def annual_mean(fit: dict[str, Any], z: Sequence[float] | np.ndarray | float) -> np.ndarray:
    """
    Evaluate the annual-mean profile of a fit from :func:`fit_seasonal`.

    Parameters
    ----------
    fit : dict
        The fit.
    z : array-like or float
        Elevations, m.

    Returns
    -------
    numpy.ndarray
        The annual-mean bias, K.
    """
    z = np.asarray(z, dtype=float)
    if fit["elevation_model"] == "exponential":
        return fit["background"] + fit["excess"] * np.exp(-z / fit["scale_height"])
    return fit["mean"] + fit["mean_per_km"] * z / 1000.0


def seasonal_curve(
    fit: dict[str, Any], z: Sequence[float] | np.ndarray | float, fraction: Sequence[float] | np.ndarray | float
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
    seasonal = np.column_stack(_seasonal_columns(z_b.ravel(), f_b.ravel(), elevation_dependent=True))
    coef = np.array([fit[name] for name in SEASONAL_COEFFICIENTS])
    return annual_mean(fit, z_b) + (seasonal @ coef).reshape(z_b.shape)


def seasonal_cycle(fit: dict[str, Any], z: Sequence[float] | np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
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


def _signed(value: float, digits: int = 2) -> str:
    """
    Format a term to follow another, with its sign as the operator.

    Parameters
    ----------
    value : float
        The term's coefficient.
    digits : int, default 2
        Decimals.

    Returns
    -------
    str
        E.g. ``"+ 1.23"`` or ``"− 1.23"``.
    """
    return f"{'−' if value < 0 else '+'} {abs(value):.{digits}f}"


def _number(value: float, digits: int = 2) -> str:
    """
    Format a number with a typographic minus sign.

    Parameters
    ----------
    value : float
        The number.
    digits : int, default 2
        Decimals.

    Returns
    -------
    str
        E.g. ``"−2.62"``.
    """
    return f"{value:.{digits}f}".replace("-", "−")


def equation(fit: dict[str, Any]) -> str:
    """
    Write a fit from :func:`fit_seasonal` as an equation with its fitted values.

    Parameters
    ----------
    fit : dict
        The fit.

    Returns
    -------
    str
        E.g. ``b = 1.92 + 6.18 exp(−z / 1.04) + (−2.62 + 0.82 z) cos 2πt
        + (−0.66 + 0.14 z) sin 2πt``, with z in km and t the fraction of
        the year.
    """
    if fit["elevation_model"] == "exponential":
        height = _number(fit["scale_height"] / 1000)
        profile = f"{_number(fit['background'])} {_signed(fit['excess'])} exp(−z / {height})"
    else:
        profile = f"{_number(fit['mean'])} {_signed(fit['mean_per_km'])} z"
    cos, sin = _number(fit["cos"]), _number(fit["sin"])
    if fit["cos_per_km"]:
        cos = f"({cos} {_signed(fit['cos_per_km'])} z)"
    if fit["sin_per_km"]:
        sin = f"({sin} {_signed(fit['sin_per_km'])} z)"
    return f"b = {profile} + {cos} cos 2πt + {sin} sin 2πt"


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
    daily: pd.DataFrame,
    min_cells: int = DEFAULT_MIN_CELLS,
    elevation_dependent: bool = True,
    models: Sequence[str] = ELEVATION_MODELS,
    loss: str = DEFAULT_FIT_LOSS,
    harmonics: int = DEFAULT_HARMONICS,
) -> pd.DataFrame:
    """
    Fit the seasonal model to the daily medians, per category and elevation profile.

    Parameters
    ----------
    daily : pandas.DataFrame
        Output of :func:`daily_medians`.
    min_cells : int, default :data:`DEFAULT_MIN_CELLS`
        Daily medians resting on fewer grid cells are left out.
    elevation_dependent : bool, default True
        See :func:`fit_seasonal`.
    models : sequence of str, default :data:`ELEVATION_MODELS`
        Elevation profiles to fit, see :func:`fit_seasonal`.
    loss : {"mean", "median"}, default :data:`DEFAULT_FIT_LOSS`
        What the fits follow, see :func:`fit_seasonal`.
    harmonics : int, default :data:`DEFAULT_HARMONICS`
        Harmonics of the climatology behind ``r2_ceiling``, see :func:`ceiling_fit`.

    Returns
    -------
    pandas.DataFrame
        One row per category and profile that could be fitted, indexed by
        ``(category, elevation_model)``: the fields of :func:`fit_seasonal`,
        the elevation range of the bins used (``z_min``, ``z_max``),
        ``elevation_dependent``, ``equation`` (see :func:`equation`),
        ``best``, True for the profile with the lowest BIC in its category,
        and the category's ``r2_ceiling``, ``rmse_ceiling`` and
        ``mae_ceiling`` from :func:`ceiling_fit`.
    """
    rows = []
    for category in CATEGORIES:
        used = daily[(daily["category"] == category) & (daily["cells"] >= min_cells)]
        for model in models:
            fit = fit_seasonal(used["centre"], used["fraction"], used["median"], elevation_dependent, model, loss)
            if fit is None:
                logger.warning(
                    "%s: too few daily medians with %d or more cells for a seasonal fit", category, min_cells
                )
                break
            fit.update(
                {
                    "category": category,
                    "z_min": float(used["centre"].min()),
                    "z_max": float(used["centre"].max()),
                    "elevation_dependent": float(elevation_dependent),
                    "equation": equation(fit),
                }
            )
            rows.append(fit)
    if not rows:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["category", "elevation_model"]))
    fits = pd.DataFrame(rows).set_index(["category", "elevation_model"])
    fits["best"] = fits["bic"] == fits.groupby(level="category")["bic"].transform("min")
    ceiling = ceiling_fit(daily, min_cells=min_cells, loss=loss, harmonics=harmonics)
    category = fits.index.get_level_values("category")
    for name in ceiling.columns:
        fits[name] = category.map(ceiling[name]).astype(float)
    return fits


def ceiling_fit(
    daily: pd.DataFrame,
    min_cells: int = DEFAULT_MIN_CELLS,
    loss: str = DEFAULT_FIT_LOSS,
    harmonics: int = DEFAULT_HARMONICS,
) -> pd.DataFrame:
    """
    Fit every elevation bin its own smooth seasonal cycle: the best a fit in elevation and season can do.

    Each bin's daily medians get a mean and *harmonics* harmonics of the
    year of their own, with the loss of the main fit. Any bias that depends
    on elevation and the time of year alone is at best this one, up to
    sharper seasonal detail than the harmonics follow, so its R² is the
    ceiling of :func:`fit_seasonal`'s; what is left is the weather. With
    the mean loss the main fits are nested in it and never do better; with
    the median loss that holds for the absolute residuals, not strictly
    for R².

    Parameters
    ----------
    daily : pandas.DataFrame
        Output of :func:`daily_medians`, one or several stores concatenated.
    min_cells : int, default :data:`DEFAULT_MIN_CELLS`
        Daily medians resting on fewer grid cells are left out, as in
        :func:`fit_categories`.
    loss : {"mean", "median"}, default :data:`DEFAULT_FIT_LOSS`
        Least squares or least absolute deviations.
    harmonics : int, default :data:`DEFAULT_HARMONICS`
        Harmonics of the year per bin.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``category``: ``r2_ceiling``, ``rmse_ceiling`` and
        ``mae_ceiling`` (K) over the bins with more daily medians than
        coefficients, for the categories with any.
    """
    solve = _least_squares if loss == "mean" else _least_absolute
    rows = {}
    for category in CATEGORIES:
        used = daily[(daily["category"] == category) & (daily["cells"] >= min_cells)]
        used = used[np.isfinite(used["median"]) & np.isfinite(used["fraction"])]
        values, residuals = [], []
        for _, group in used.groupby("centre"):
            columns = _harmonic_columns(group["fraction"].to_numpy(), harmonics)
            y = group["median"].to_numpy(dtype=float)
            if y.size <= len(columns):
                continue
            coef, _ = solve(columns, y)
            values.append(y)
            residuals.append(y - np.column_stack(columns) @ coef)
        if not values:
            continue
        y, residual = np.concatenate(values), np.concatenate(residuals)
        sse, sst = float(residual @ residual), float(((y - y.mean()) ** 2).sum())
        rows[category] = {
            "r2_ceiling": 1.0 - sse / sst if sst > 0 else math.nan,
            "rmse_ceiling": math.sqrt(sse / y.size),
            "mae_ceiling": float(np.abs(residual).mean()),
        }
    columns = ["r2_ceiling", "rmse_ceiling", "mae_ceiling"]
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns).rename_axis("category")


def split_years(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Split the daily medians into the first and the second half of their years.

    Whole calendar years go to each half, so both see every season equally
    often; with an odd number of years the second half gets the extra one.

    Parameters
    ----------
    daily : pandas.DataFrame
        Output of :func:`daily_medians`, one or several stores concatenated.

    Returns
    -------
    tuple of pandas.DataFrame or None
        The earlier and the later years, or None with fewer than two years.
    """
    years = pd.to_datetime(daily["time"]).dt.year
    distinct = np.unique(years)
    if distinct.size < 2:
        return None
    first = years < distinct[distinct.size // 2]
    return daily[first], daily[~first]


def half_record_stability(
    daily: pd.DataFrame,
    min_cells: int = DEFAULT_MIN_CELLS,
    elevation_dependent: bool = True,
    loss: str = DEFAULT_FIT_LOSS,
    harmonics: int = DEFAULT_HARMONICS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit the first and the second half of the record apart and compare the fitted biases.

    The difference between the halves is what the fit changes from one
    stretch of years to the next: a bias that differs between regions by
    much more than this is a regional signal, not interannual weather. Both
    halves' fits are evaluated on every elevation bin both cover and every
    day of a 365-day year, and their difference split into the annual mean
    and the seasonal cycle, so that ``rms_diff² = annual_rms_diff² +
    cycle_rms_diff²``.

    Parameters
    ----------
    daily : pandas.DataFrame
        Output of :func:`daily_medians`, one or several stores concatenated.
    min_cells : int, default :data:`DEFAULT_MIN_CELLS`
        See :func:`fit_categories`.
    elevation_dependent : bool, default True
        See :func:`fit_seasonal`.
    loss : {"mean", "median"}, default :data:`DEFAULT_FIT_LOSS`
        See :func:`fit_seasonal`.
    harmonics : int, default :data:`DEFAULT_HARMONICS`
        See :func:`ceiling_fit`.

    Returns
    -------
    halves : pandas.DataFrame
        The fits of :func:`fit_categories` for each half, indexed by
        ``(half, category, elevation_model)``, with the ``years`` each covers.
    stability : pandas.DataFrame
        Indexed by ``(category, elevation_model)``: ``years_first`` and
        ``years_second``, the elevation range compared (``z_min``,
        ``z_max``), and the second half's bias minus the first's (K):
        ``rms_diff``, ``annual_rms_diff``, ``cycle_rms_diff``,
        ``max_abs_diff``, and ``jja_diff``, the mean over June to August.
        Both are empty with fewer than two years.
    """
    index = pd.MultiIndex.from_arrays([[], []], names=["category", "elevation_model"])
    split = split_years(daily)
    if split is None:
        logger.warning("fewer than two years: no half-record stability check")
        return pd.DataFrame(), pd.DataFrame(index=index)
    fits, years = {}, {}
    for half, part in zip(("first", "second"), split):
        span = pd.to_datetime(part["time"]).dt.year
        years[half] = f"{span.min()}-{span.max()}"
        fits[half] = fit_categories(part, min_cells, elevation_dependent, loss=loss, harmonics=harmonics)
        fits[half]["years"] = years[half]
    halves = pd.concat(fits, names=["half"])

    fraction = (np.arange(365) + 0.5) / 365.0
    summer = (fraction >= 151 / 365) & (fraction < 243 / 365)
    rows = []
    for key in fits["first"].index.intersection(fits["second"].index):
        first = {**fits["first"].loc[key].to_dict(), "elevation_model": key[1]}
        second = {**fits["second"].loc[key].to_dict(), "elevation_model": key[1]}
        lo, hi = max(first["z_min"], second["z_min"]), min(first["z_max"], second["z_max"])
        centres = np.unique(daily.loc[daily["category"] == key[0], "centre"])
        z = centres[(centres >= lo) & (centres <= hi)]
        if z.size == 0:
            continue
        diff = seasonal_curve(second, z[:, None], fraction[None, :]) - seasonal_curve(
            first, z[:, None], fraction[None, :]
        )
        annual = diff.mean(axis=1, keepdims=True)
        rows.append(
            {
                "category": key[0],
                "elevation_model": key[1],
                "years_first": years["first"],
                "years_second": years["second"],
                "z_min": float(z.min()),
                "z_max": float(z.max()),
                "rms_diff": float(np.sqrt((diff**2).mean())),
                "annual_rms_diff": float(np.sqrt((annual**2).mean())),
                "cycle_rms_diff": float(np.sqrt(((diff - annual) ** 2).mean())),
                "max_abs_diff": float(np.abs(diff).max()),
                "jja_diff": float(diff[:, summer].mean()),
            }
        )
    if not rows:
        return halves, pd.DataFrame(index=index)
    return halves, pd.DataFrame(rows).set_index(["category", "elevation_model"])


def _signed_sqrt(value: np.ndarray | float) -> np.ndarray:
    """
    Take the square root of the magnitude, keeping the sign.

    Parameters
    ----------
    value : array-like or float
        A difference of variances.

    Returns
    -------
    numpy.ndarray
        ``sign(value) sqrt(|value|)``.
    """
    value = np.asarray(value, dtype=float)
    return np.sign(value) * np.sqrt(np.abs(value))


def variance_by_elevation(annual: xr.Dataset) -> pd.DataFrame:
    """
    Compare the day-to-day spread of the 2 m and the free-air temperature per elevation bin.

    The anomalies are taken per cell from its own smooth seasonal cycle
    (see :func:`_tile_counts`) and their variances pooled over the bin's
    cell-days. PISM's ``carra2_bias`` adds the bias to a 2 m temperature to
    raise it to the free-air temperature at the ground. A bias that
    depends on elevation and season only is part of that cycle, so the
    corrected forcing keeps the 2 m air's anomalies, and the variance it
    lacks against the free air is::

        sigma_extra² = sigma_free² - sigma_t2m²

    Where it is positive -- the 2 m air held near 0 degC over melting ice
    varies less than the free air -- it may be added in quadrature to a
    melt model's ``std_dev``. This is less than the bias's own spread
    ``sigma_bias`` wherever the bias follows the free air's weather, so
    adding all of that would overshoot. Where it is negative the corrected
    forcing varies more than the free air already. ``sigma_extra`` carries
    the sign of the difference.

    Parameters
    ----------
    annual : xarray.Dataset
        Output of :func:`annual_histograms` (or one of :func:`bias_histograms`).

    Returns
    -------
    pandas.DataFrame
        ``category``, ``centre``, ``cell_days``, ``sigma_t2m``,
        ``sigma_free``, ``sigma_bias`` and ``sigma_extra`` (K), for bins with
        cell-days.
    """
    ss = annual["anomaly_ss"].transpose("quantity", "category", "centre")
    days = annual["cell_days"].transpose("category", "centre").values
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = {q: ss.sel(quantity=q).values / days for q in ANOMALY_QUANTITIES}
    category, centre = np.meshgrid(annual["category"].values, annual["centre"].values, indexing="ij")
    frame = pd.DataFrame(
        {
            "category": category.ravel(),
            "centre": centre.ravel(),
            "cell_days": days.ravel(),
            "sigma_t2m": np.sqrt(variance["t2m"]).ravel(),
            "sigma_free": np.sqrt(variance["free"]).ravel(),
            "sigma_bias": np.sqrt(variance["bias"]).ravel(),
            "sigma_extra": _signed_sqrt(variance["free"] - variance["t2m"]).ravel(),
        }
    )
    return frame[frame["cell_days"] > 0].reset_index(drop=True)


def pooled_sigma_extra(fits: pd.DataFrame, annual: xr.Dataset) -> pd.Series:
    """
    Pool :func:`variance_by_elevation`'s ``sigma_extra`` over each fit's elevation range.

    Parameters
    ----------
    fits : pandas.DataFrame
        Output of :func:`fit_categories`.
    annual : xarray.Dataset
        Output of :func:`annual_histograms`.

    Returns
    -------
    pandas.Series
        ``sigma_extra`` (K, signed) on the index of *fits*, from the
        variances pooled over the cell-days of the bins in
        ``[z_min, z_max]``; NaN without any.
    """
    ss = annual["anomaly_ss"]
    days = annual["cell_days"]
    centre = annual["centre"]
    out = pd.Series(math.nan, index=fits.index, name="sigma_extra")
    for (category, model), fit in fits.iterrows():
        inside = (centre >= fit["z_min"]) & (centre <= fit["z_max"])
        n = float(days.sel(category=category).where(inside, 0.0).sum())
        if n > 0:
            difference = ss.sel(quantity="free", category=category) - ss.sel(quantity="t2m", category=category)
            out.loc[(category, model)] = float(_signed_sqrt(float(difference.where(inside, 0.0).sum()) / n))
    return out


def _fit_rows(fits: pd.DataFrame, category: str) -> list[dict[str, Any]]:
    """
    List a category's fits, best (lowest BIC) first.

    Parameters
    ----------
    fits : pandas.DataFrame
        Output of :func:`fit_categories`.
    category : str
        Category.

    Returns
    -------
    list of dict
        The fits, each with its ``elevation_model``.
    """
    if fits.empty or category not in fits.index.get_level_values("category"):
        return []
    rows = fits.xs(category, level="category").sort_values("bic").reset_index()
    return [{**row.to_dict(), "category": category} for _, row in rows.iterrows()]


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
        ``category``, ``elevation_model``, ``centre``, ``annual_mean``,
        ``amplitude``, ``day_of_max``, ``seasonal_min`` and ``seasonal_max``
        (K, day of year).
    """
    rows = []
    for (category, model), fit_row in fits.iterrows():
        fit = {**fit_row.to_dict(), "elevation_model": model}
        z = np.asarray([c for c in centres if fit["z_min"] <= c <= fit["z_max"]], dtype=float)
        mean = annual_mean(fit, z)
        amplitude, day_of_max = seasonal_cycle(fit, z)
        rows.append(
            pd.DataFrame(
                {
                    "category": category,
                    "elevation_model": model,
                    "centre": z,
                    "annual_mean": mean,
                    "amplitude": amplitude,
                    "day_of_max": day_of_max,
                    "seasonal_min": mean - amplitude,
                    "seasonal_max": mean + amplitude,
                }
            )
        )
    columns = [
        "category",
        "elevation_model",
        "centre",
        "annual_mean",
        "amplitude",
        "day_of_max",
        "seasonal_min",
        "seasonal_max",
    ]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def fit_label(fit: Mapping[str, Any]) -> str:
    """
    Summarise a fit's quality by the statistic it minimised.

    Parameters
    ----------
    fit : Mapping
        A fit from :func:`fit_seasonal` or a row of :func:`fit_categories`.

    Returns
    -------
    str
        RMSE for a mean fit, MAE for a median fit, then R².
    """
    if fit.get("loss", "mean") == "median":
        return f"MAE {fit['mae']:.2f} K, R² {fit['r2']:.2f}"
    return f"RMSE {fit['rmse']:.2f} K, R² {fit['r2']:.2f}"


def _equation_panel(ax: Any, fits: pd.DataFrame) -> None:
    """
    Write every fit's equation and statistics into an empty axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to fill; its frame is hidden.
    fits : pandas.DataFrame
        Output of :func:`fit_categories`.
    """
    ax.set_axis_off()
    rows = [fit for category in CATEGORIES for fit in _fit_rows(fits, category)]
    lines = ["z in km, t fraction of the year; ★ = lowest BIC in its category"]
    for fit in rows:
        note = ""
        if fit["elevation_model"] == "exponential" and fit["scale_height"] >= 0.99 * SCALE_HEIGHT_RANGE[1]:
            note = "   H at its limit: no curvature, a straight line"
        lines.append(
            f"{'★' if fit['best'] else ' '} {fit['category'].replace('_', ' ')}, {fit['elevation_model']}:"
            f"  {fit_label(fit)}, BIC {fit['bic']:.0f}{note}"
        )
        lines.append(f"    {fit['equation']}")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=7, transform=ax.transAxes)


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
    the right edge. Both fitted profiles are drawn at the true bin
    elevations as their annual mean (exponential solid, linear dashed), the
    better one by BIC with a band spanning its seasonal cycle's minimum to
    maximum. A panel below gives every fit's equation, fitted values and
    statistics.

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
    height = max(5.0, 0.5 * centres.size + 2)
    fig, (ax, text_ax) = plt.subplots(
        2, 1, figsize=(10, height + 2.0), height_ratios=[height, 2.0], constrained_layout=True
    )
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

        for fit in _fit_rows(fits, category):
            z_line = np.linspace(fit["z_min"], fit["z_max"], 200)
            mean = annual_mean(fit, z_line)
            if fit["best"]:
                amplitude, _ = seasonal_cycle(fit, z_line)
                ax.fill_betweenx(z_line, mean - amplitude, mean + amplitude, color=color, alpha=0.15, lw=0, zorder=3)
            style = PROFILE_STYLES[fit["elevation_model"]]
            ax.plot(
                mean,
                z_line,
                color=color,
                lw=2.2,
                ls=style,
                zorder=4,
                label=f"{category.replace('_', ' ')}, {fit['elevation_model']}",
            )

    edges = np.append(centres - bin_width / 2, centres[-1] + bin_width / 2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(edges)
    ax.set_ylim(edges[0], edges[-1])
    ax.set_xlabel("Level fit at the terrain − 2 m air temperature (K)")
    ax.set_ylabel(f"CARRA2 terrain elevation (m), {bin_width:g} m bins")
    ax.set_title(title or "Free-air extrapolation vs CARRA2 2 m temperature, every grid cell", fontsize=11)
    loss = str(fits["loss"].iloc[0]) if "loss" in fits and not fits.empty else "mean"
    ax.legend(loc="upper right", title=f"cells and annual {loss} fits", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    _equation_panel(text_ax, fits)
    return fig


def plot_seasonal_fit(daily: pd.DataFrame, fits: pd.DataFrame, min_cells: int = DEFAULT_MIN_CELLS) -> Figure:
    """
    Plot the daily median bias against the day of the year, with the better fit's cycle.

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
    fig = plt.figure(figsize=(13, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, len(CATEGORIES), height_ratios=[4.8, 1.8])
    axes = [fig.add_subplot(grid[0, 0])]
    axes += [fig.add_subplot(grid[0, i], sharey=axes[0]) for i in range(1, len(CATEGORIES))]
    days = np.arange(1, 366)
    # Day of the year each month starts on, in a 365-day year.
    month_starts = 1 + np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
    for ax, category in zip(axes, CATEGORIES):
        rows = used[used["category"] == category]
        doy = rows["fraction"].to_numpy() * 365.0 + 0.5
        ax.scatter(doy, rows["median"], c=cmap(norm(rows["centre"])), s=3, alpha=0.35, lw=0)
        fit_rows = _fit_rows(fits, category)
        if fit_rows:
            fit = fit_rows[0]
            for centre in np.sort(rows["centre"].unique()):
                curve = seasonal_curve(fit, centre, (days - 0.5) / 365.0)
                ax.plot(days, curve, color=cmap(norm(centre)), lw=1.6)
            ax.set_title(
                f"{category.replace('_', ' ')}, {fit['elevation_model']} profile: " f"{fit_label(fit)}",
                fontsize=10,
            )
        else:
            ax.set_title(f"{category.replace('_', ' ')}: no fit", fontsize=10)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(month_starts, [m[0] for m in MONTH_LABELS])
        ax.set_xlim(1, 366)
        ax.set_xlabel("Day of the year")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Daily median bias per elevation bin (K)")
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="Bin elevation (m)")
    _equation_panel(fig.add_subplot(grid[1, :]), fits)
    fig.suptitle("Free-air extrapolation vs CARRA2 2 m temperature: elevation profile with a yearly cycle")
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


def _parser() -> ArgumentParser:
    """
    Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser of :func:`main`.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = (
        "Compare CARRA2's free-air temperature, fitted through the pressure levels and carried down to the "
        "terrain, with its 2 m air temperature: on and off glacier, by elevation."
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
        "--fit",
        choices=FIT_LOSSES,
        default=DEFAULT_FIT_LOSS,
        help="Fit the daily medians' mean (least squares) or their median (least absolute deviations).",
    )
    parser.add_argument(
        "--hist-step", type=float, default=DEFAULT_HIST_STEP, help="Bin width of the bias histograms (K)."
    )
    parser.add_argument(
        "--time-block", type=int, default=DEFAULT_TIME_BLOCK, help="Days computed at once within a tile."
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=None,
        help="Tile size in grid cells; by default the store's chunks, so each chunk is read once.",
    )
    add_ntasks_argument(
        parser,
        default=4,
        help_text="Grid tiles processed at once; each holds one tile of every field for all days.",
    )
    parser.add_argument(
        "--anon",
        default=True,
        action="store_true",
        help="Read S3 without credentials (the default; the bucket must be public).",
    )
    parser.add_argument("--no-anon", dest="anon", action="store_false", help="Read S3 with your AWS credentials.")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution of the figures.")
    parser.add_argument(
        "--harmonics",
        type=int,
        default=DEFAULT_HARMONICS,
        help="Harmonics of the year in the seasonal climatologies behind r2_ceiling and sigma_extra.",
    )
    return parser


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
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    output = args.output_path
    output.mkdir(parents=True, exist_ok=True)
    setup_logging(output / "carra2_bias_correction.log")

    outlines = read_vectors(args.outlines, anon=args.anon)
    logger.info("%d outlines from %s", len(outlines), args.outlines)

    hists, dailies = [], []
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
            tile=args.tile,
            progress=True,
            harmonics=args.harmonics,
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

    annual = annual_histograms(hists)
    medians = binned_medians(annual)
    daily = pd.concat(dailies, ignore_index=True)
    fits = fit_categories(
        daily,
        min_cells=args.min_cells,
        elevation_dependent=args.season == "elevation",
        loss=args.fit,
        harmonics=args.harmonics,
    )
    variance = variance_by_elevation(annual)
    if not fits.empty:
        fits["sigma_extra"] = pooled_sigma_extra(fits, annual)
    cycle = seasonal_by_elevation(fits, annual["centre"].values)
    halves, stability = half_record_stability(
        daily,
        min_cells=args.min_cells,
        elevation_dependent=args.season == "elevation",
        loss=args.fit,
        harmonics=args.harmonics,
    )
    medians.to_csv(output / "bias_by_elevation_medians.csv", index=False)
    daily.to_csv(output / "bias_daily_medians.csv", index=False)
    fits.to_csv(output / "bias_seasonal_fit.csv")
    cycle.to_csv(output / "bias_seasonal_by_elevation.csv", index=False)
    variance.to_csv(output / "bias_variance_by_elevation.csv", index=False)
    if not stability.empty:
        halves.to_csv(output / "bias_seasonal_fit_halves.csv")
        stability.to_csv(output / "bias_half_stability.csv")
    for name, figure in (
        ("bias_by_elevation.png", plot_bias_by_elevation(annual, medians, fits, bin_width=args.bin_width)),
        ("bias_seasonal_fit.png", plot_seasonal_fit(daily, fits, min_cells=args.min_cells)),
    ):
        figure.savefig(output / name, dpi=args.dpi)
        plt.close(figure)

    if not fits.empty:
        columns = ["rmse", "mae", "r2", "r2_ceiling", "sigma_extra", "bic", "best"]
        print(fits[columns].round(3).to_string())
        for (category, model), fit in fits.iterrows():
            print(f"{category:11s} {model:11s} {fit['equation']}")
    if not stability.empty:
        first, second = stability[["years_first", "years_second"]].iloc[0]
        print(f"Second half ({second}) minus first half ({first}) of the fitted bias, K:")
        columns = ["rms_diff", "annual_rms_diff", "cycle_rms_diff", "max_abs_diff", "jja_diff"]
        print(stability[columns].round(3).to_string())
    print(f"Results in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
