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
centre lies inside an outline, bins them by terrain elevation, and fits a
continuous two-segment line through the median bias of each bin per
category. It writes the binned medians and the fits as CSV and a figure of
the bias distributions against elevation. Given point sites as well, it also
samples the stores there and plots the bias per month, pooled and per site.

Stores, outlines and sites may each be a local path or an ``s3://`` URI.
"""

import logging
import math
import tempfile
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence
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

from pism_terra.grids import sample_points
from pism_terra.log import setup_logging
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

#: Fewest bins each segment of the two-segment fit must keep.
DEFAULT_MIN_POINTS = 3

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


def bias_table(
    bias: xr.DataArray,
    on_glacier: xr.DataArray,
    orography: xr.DataArray,
    bin_width: float = DEFAULT_BIN_WIDTH,
    source: str = "",
) -> pd.DataFrame:
    """
    Flatten the gridded bias into one row per cell and day.

    Parameters
    ----------
    bias : xarray.DataArray
        Bias on ``(time, y, x)``, from :func:`ground_bias`.
    on_glacier : xarray.DataArray
        Mask on ``(y, x)``, from :func:`glacier_mask`.
    orography : xarray.DataArray
        Terrain on ``(y, x)``, m.
    bin_width : float, default :data:`DEFAULT_BIN_WIDTH`
        Width of the elevation bins, m.
    source : str, default ""
        Name of the store, kept in a ``source`` column and in the cell id so
        cells of different stores stay apart.

    Returns
    -------
    pandas.DataFrame
        ``source``, ``cell`` (``<source>:<y index>:<x index>``), ``time``,
        ``elevation``, ``centre`` (bin centre), ``category`` and ``bias``,
        without rows whose bias is NaN.
    """
    bias = bias.transpose("time", "y", "x")
    n_time, n_y, n_x = bias.shape
    cells = np.array([f"{source}:{j}:{i}" for j in range(n_y) for i in range(n_x)])
    elevation = np.asarray(orography.transpose("y", "x").values, dtype=float).ravel()
    category = np.where(np.asarray(on_glacier.transpose("y", "x").values).ravel(), "on_glacier", "off_glacier")
    table = pd.DataFrame(
        {
            "source": source,
            "cell": np.tile(cells, n_time),
            "time": np.repeat(bias["time"].values, n_y * n_x),
            "elevation": np.tile(elevation, n_time),
            "centre": np.tile(bin_centre(elevation, bin_width), n_time),
            "category": np.tile(category, n_time),
            "bias": np.asarray(bias.values, dtype=float).ravel(),
        }
    )
    return table.dropna(subset=["bias"]).reset_index(drop=True)


def binned_medians(table: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the bias per category and elevation bin.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows from :func:`bias_table`.

    Returns
    -------
    pandas.DataFrame
        ``category``, ``centre``, ``median``, ``q25``, ``q75``, ``count``
        (cell-days) and ``cells`` (grid cells), sorted by category and centre.
    """
    grouped = table.groupby(["category", "centre"])
    summary = grouped["bias"].agg(
        median="median", q25=lambda b: b.quantile(0.25), q75=lambda b: b.quantile(0.75), count="count"
    )
    summary["cells"] = grouped["cell"].nunique()
    return summary.reset_index().sort_values(["category", "centre"]).reset_index(drop=True)


def fit_piecewise(
    z: Sequence[float] | np.ndarray | pd.Series,
    y: Sequence[float] | np.ndarray | pd.Series,
    min_points: int = DEFAULT_MIN_POINTS,
    n_candidates: int = 400,
) -> dict[str, float] | None:
    """
    Fit a continuous two-segment line ``y(z)`` with a scanned breakpoint.

    The model is ``y = a + k_below min(z - z_b, 0) + k_above max(z - z_b, 0)``.
    For each candidate breakpoint ``z_b`` the rest is linear least squares;
    the candidate with the smallest squared error wins. Candidates are
    limited so that each segment keeps at least *min_points* points. Unlike
    :func:`pism_terra.glacier.usgs_benchmark_stakes.fit_gradient`, the value
    at the breakpoint is free, not zero.

    Parameters
    ----------
    z : array-like
        Abscissae, e.g. bin centres in m.
    y : array-like
        Values, e.g. median bias in K.
    min_points : int, default :data:`DEFAULT_MIN_POINTS`
        Fewest points each segment must keep.
    n_candidates : int, default 400
        Number of breakpoints scanned.

    Returns
    -------
    dict or None
        ``z_break``, ``y_break`` (``a``), ``k_below`` and ``k_above`` (per unit
        of z), ``rmse`` and ``n``. None with fewer than ``2 * min_points``
        finite points.
    """
    z_arr, y_arr = np.asarray(z, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(z_arr) & np.isfinite(y_arr)
    z_arr, y_arr = z_arr[keep], y_arr[keep]
    if z_arr.size < 2 * min_points or np.ptp(z_arr) == 0:
        return None
    ordered = np.sort(z_arr)
    best_sse, best_break, best_coef = math.inf, math.nan, np.full(3, np.nan)
    for z_break in np.linspace(ordered[min_points - 1], ordered[-min_points], n_candidates):
        dz = z_arr - z_break
        design = np.column_stack([np.ones_like(z_arr), np.minimum(dz, 0.0), np.maximum(dz, 0.0)])
        coef, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
        sse = float(((y_arr - design @ coef) ** 2).sum())
        if sse < best_sse:
            best_sse, best_break, best_coef = sse, float(z_break), coef
    return {
        "z_break": best_break,
        "y_break": float(best_coef[0]),
        "k_below": float(best_coef[1]),
        "k_above": float(best_coef[2]),
        "rmse": math.sqrt(best_sse / z_arr.size),
        "n": float(z_arr.size),
    }


def piecewise_line(fit: dict[str, float], z: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Evaluate a fit from :func:`fit_piecewise`.

    Parameters
    ----------
    fit : dict
        The fit.
    z : array-like
        Where to evaluate it.

    Returns
    -------
    numpy.ndarray
        The fitted values.
    """
    dz = np.asarray(z, dtype=float) - fit["z_break"]
    return fit["y_break"] + fit["k_below"] * np.minimum(dz, 0.0) + fit["k_above"] * np.maximum(dz, 0.0)


def fit_categories(
    medians: pd.DataFrame, min_cells: int = DEFAULT_MIN_CELLS, min_points: int = DEFAULT_MIN_POINTS
) -> pd.DataFrame:
    """
    Fit the median bias against elevation, per category.

    Parameters
    ----------
    medians : pandas.DataFrame
        Output of :func:`binned_medians`.
    min_cells : int, default :data:`DEFAULT_MIN_CELLS`
        Bins resting on fewer grid cells are left out.
    min_points : int, default :data:`DEFAULT_MIN_POINTS`
        Fewest bins per segment, see :func:`fit_piecewise`.

    Returns
    -------
    pandas.DataFrame
        One row per category that could be fitted, indexed by category:
        the fields of :func:`fit_piecewise`, the slopes again in K per km
        (``k_below_per_km``, ``k_above_per_km``) and the elevation range of
        the bins used (``z_min``, ``z_max``).
    """
    rows = {}
    for category in CATEGORIES:
        used = medians[(medians["category"] == category) & (medians["cells"] >= min_cells)]
        fit = fit_piecewise(used["centre"], used["median"], min_points=min_points)
        if fit is None:
            logger.warning("%s: too few bins with %d or more cells for a two-segment fit", category, min_cells)
            continue
        fit["k_below_per_km"] = 1000.0 * fit["k_below"]
        fit["k_above_per_km"] = 1000.0 * fit["k_above"]
        fit["z_min"] = float(used["centre"].min())
        fit["z_max"] = float(used["centre"].max())
        rows[category] = fit
    fits = pd.DataFrame.from_dict(rows, orient="index")
    fits.index.name = "category"
    return fits


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


def plot_bias_by_elevation(
    table: pd.DataFrame,
    medians: pd.DataFrame,
    fits: pd.DataFrame,
    bin_width: float = DEFAULT_BIN_WIDTH,
    title: str | None = None,
) -> Figure:
    """
    Plot the bias distribution per elevation bin, on and off glacier, with the fits.

    Elevation runs up the y axis; each bin gets one horizontal violin per
    category, offset so the pair sits side by side. The number of grid cells
    behind each violin is printed at the right edge. The two-segment fits
    are drawn at the true bin elevations, with a diamond at the breakpoint.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows from :func:`bias_table`.
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
    centres = np.sort(table["centre"].unique())
    fig, ax = plt.subplots(figsize=(8, max(5.0, 0.5 * centres.size + 2)), constrained_layout=True)
    for category, (color, shift) in CATEGORIES.items():
        rows = table[table["category"] == category]
        positions = [c for c in centres if (rows["centre"] == c).sum() > 1]
        if not positions:
            continue
        parts = ax.violinplot(
            [rows.loc[rows["centre"] == c, "bias"].to_numpy() for c in positions],
            positions=np.asarray(positions) + shift * bin_width,
            widths=0.45 * bin_width,
            orientation="horizontal",
            showmedians=True,
            showextrema=False,
        )
        _style_violins(parts, color)
        ax.plot([], [], "s", color=color, alpha=0.7, label=category.replace("_", " "))

        cells = medians[medians["category"] == category].set_index("centre")["cells"]
        for c in positions:
            ax.annotate(
                str(int(cells.get(c, 0))),
                (1.0, c + shift * bin_width),
                xycoords=("axes fraction", "data"),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=color,
            )

        if category in fits.index:
            fit = fits.loc[category].to_dict()
            z_line = np.linspace(fit["z_min"], fit["z_max"], 200)
            ax.plot(piecewise_line(fit, z_line), z_line, color=color, lw=2.2, zorder=4)
            ax.plot(fit["y_break"], fit["z_break"], "D", ms=7, color=color, mec="k", zorder=6)

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
        "--min-cells", type=int, default=DEFAULT_MIN_CELLS, help="Bins with fewer grid cells stay out of the fit."
    )
    parser.add_argument(
        "--min-points", type=int, default=DEFAULT_MIN_POINTS, help="Fewest bins per segment of the fit."
    )
    parser.add_argument(
        "--months", type=int, nargs="+", default=None, help="Use only these months, e.g. 6 7 8; all when omitted."
    )
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

    tables, site_frames = [], []
    for path in args.CARRA2:
        print(f"Reading {path}...", flush=True)
        ds = open_carra2(path, anon=args.anon)
        if args.months:
            ds = ds.sel(time=ds["time"].dt.month.isin(args.months))
            if ds.sizes["time"] == 0:
                raise ValueError(f"{path} has no days in months {args.months}")
        source = _source_name(path)
        bias = ground_bias(ds, below=args.below).load()
        on_glacier = glacier_mask(ds, outlines)
        logger.info("%s: %d of %d cells on glacier", source, int(on_glacier.sum()), on_glacier.size)
        tables.append(bias_table(bias, on_glacier, ds["orography"].load(), args.bin_width, source))
        if sites is not None:
            frame = site_bias(ds, sites, below=args.below)
            if frame is not None:
                site_frames.append(frame.assign(source=source))

    table = pd.concat(tables, ignore_index=True)
    medians = binned_medians(table)
    fits = fit_categories(medians, min_cells=args.min_cells, min_points=args.min_points)
    medians.to_csv(output / "bias_by_elevation_medians.csv", index=False)
    fits.to_csv(output / "bias_by_elevation_fits.csv")
    fig = plot_bias_by_elevation(table, medians, fits, bin_width=args.bin_width)
    fig.savefig(output / "bias_by_elevation.png", dpi=args.dpi)
    plt.close(fig)

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
        columns = ["z_break", "y_break", "k_below_per_km", "k_above_per_km", "rmse", "n"]
        print(fits[columns].round(2).to_string())
    print(f"Results in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
