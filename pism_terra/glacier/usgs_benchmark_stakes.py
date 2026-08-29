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

# pylint: disable=unused-import,too-many-locals,too-many-branches,too-many-statements

"""
USGS benchmark-glacier stake measurements against PISM spatial output.

The ScienceBase release behind :mod:`pism_terra.glacier.usgs_benchmark_glaciers`
also ships the point measurements every glacier-wide balance was built from:
a winter balance ``bw`` measured at the spring visit and an annual balance
``ba`` at the fall visit, per site and year, and for some glaciers a
sub-seasonal ``db`` between two visits. A stake records the surface mass
balance at one spot, so its model counterpart is the surface mass balance of
the grid cell the stake sits in — ``surface_accumulation_flux`` minus
``surface_runoff_flux`` — integrated over exactly the interval between the
visits. This module samples every monthly ``spatial_*.nc`` file under a run
directory at the stake locations, integrates the balances, scores them per
site and pooled per glacier, and writes per glacier a figure with a panel per
site, a modelled-against-observed scatter, a balance-against-elevation
gradient plot per season with two-piece linear fits hinged at the ELA, and a
NetCDF file into a ``<rgi_id>`` sub-directory.
"""

import logging
import math
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import cf_xarray
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pint_xarray
import xarray as xr
from matplotlib.ticker import MaxNLocator

from pism_terra.download import download_usgs_benchmark
from pism_terra.glacier.usgs import (
    DEFAULT_DATA_DIR,
    MODEL_COLOR,
    MODEL_SEASON_STYLE,
    OBS_STYLE,
    RHO_WATER,
    SEASON_DATE_COLUMNS,
    SPECIFIC_LABEL,
    SPECIFIC_UNITS,
    find_model_files,
    grid_spacing,
    integrate_rate,
    interval_edges,
    is_monthly,
    load_glacier_wide,
    load_measurements,
    load_rgi_glaciers,
    load_sites,
    match_rgi_ids,
    open_pism,
    rgi_output_dir,
    run_label,
    score,
)
from pism_terra.kitp.analyze import rc_params
from pism_terra.log import setup_logging
from pism_terra.workflow import dataset_crs, drop_grid_mapping

logger = logging.getLogger("pism_terra.glacier.usgs_benchmark_stakes")

#: Accumulation and runoff, whose difference is the surface mass balance.
SMB_VARS = ("surface_accumulation_flux", "surface_runoff_flux")
#: Per-cell surface mass flux, the fallback when :data:`SMB_VARS` are absent.
SMB_TENDENCY_VAR = "tendency_of_ice_mass_due_to_surface_mass_flux"
#: Geometry sampled alongside the balance: the thickness masks deglaciated
#: cells, the surface is compared with the stake's elevation.
POINT_VARS = ("thk", "usurf")
#: Stake variables and their seasons, in plotting order.
STAKE_VARS = {"bw": "winter", "bs": "summer", "ba": "annual", "db": "sub-seasonal"}
STAKE_STYLE = {
    "bw": OBS_STYLE["Bw"],
    "bs": OBS_STYLE["Bs"],
    "ba": OBS_STYLE["Ba"],
    "db": {"color": "#CCBB44", "marker": "s"},
}
STAKE_MODEL_STYLE = {
    "bw": MODEL_SEASON_STYLE["Bw"],
    "bs": MODEL_SEASON_STYLE["Bs"],
    "ba": {"color": MODEL_COLOR, "ls": "-"},
    "db": {"color": "#CCBB44", "ls": "--"},
}
SPATIAL_PATTERN = "spatial_*.nc"
TABLE_COLUMNS = [
    "glacier",
    "site",
    "year",
    "variable",
    "start",
    "end",
    "obs",
    "model",
    "elevation",
    "usurf",
    "visit",
    "run",
]
SKILL_COLUMNS = [
    "glacier",
    "site",
    "variable",
    "season",
    "n",
    "r",
    "mae",
    "bias",
    "elevation_offset",
    "n_ice_free_months",
]
POOLED = "all"
NO_RUN = "observations"


def find_spatial_files(root: Path | str, pattern: str = SPATIAL_PATTERN) -> list[Path]:
    """
    Find PISM spatial output files below a run directory.

    The post-processed time means (``processed_spatial/*_TM.nc``) carry no
    time series and are left out.

    Parameters
    ----------
    root : Path or str
        Directory searched recursively.
    pattern : str, optional
        Glob pattern of the spatial files.

    Returns
    -------
    list of Path
        Matching files, sorted.
    """
    return [
        f
        for f in find_model_files(root, pattern)
        if "processed_spatial" not in f.parts and not f.name.endswith("_TM.nc")
    ]


def stake_points(sites: gpd.GeoDataFrame, tables: Iterable[pd.DataFrame | None]) -> gpd.GeoDataFrame:
    """
    One point per measured site.

    Parameters
    ----------
    sites : geopandas.GeoDataFrame
        Stake locations from :func:`pism_terra.glacier.usgs.load_sites`.
    tables : iterable of pandas.DataFrame or None
        Measurement tables with ``glacier`` and ``site_name`` columns; a
        site appearing in none of them is dropped.

    Returns
    -------
    geopandas.GeoDataFrame
        ``glacier``, ``site_name``, ``site`` (``<glacier>:<site_name>``) and
        an EPSG:4326 point per site, sorted by glacier and site.
    """
    measured = pd.concat(
        [t[["glacier", "site_name"]] for t in tables if t is not None], ignore_index=True
    ).drop_duplicates()
    points = sites.rename(columns={"Glacier": "glacier"})[["glacier", "site_name", "geometry"]]
    points = points.drop_duplicates(subset=["glacier", "site_name"]).merge(measured, on=["glacier", "site_name"])
    points["site"] = points["glacier"] + ":" + points["site_name"]
    points = points.sort_values(["glacier", "site_name"]).reset_index(drop=True)
    return gpd.GeoDataFrame(points, geometry="geometry", crs=sites.crs)


def _nearest_indices(coord: pd.Index, values: np.ndarray) -> np.ndarray:
    """
    Index of the grid coordinate nearest each value.

    Parameters
    ----------
    coord : pandas.Index
        Monotonic grid coordinate.
    values : numpy.ndarray
        Projected stake coordinates.

    Returns
    -------
    numpy.ndarray
        Integer indices into *coord*.
    """
    return np.asarray(coord.get_indexer(values, method="nearest"))


def sample_points(ds: xr.Dataset, points: gpd.GeoDataFrame, method: str = "nearest") -> xr.Dataset | None:
    """
    Sample a spatial file at the stakes that fall inside its grid.

    A stake's cell is read from the file's own projection, so the sites are
    reprojected rather than the grid. The chunks PISM writes span the whole
    grid per time step, so reading any point costs a pass over the file;
    the stakes' bounding box (plus a one-cell margin for the bilinear
    stencil) is therefore loaded once and every site is taken from that.

    Parameters
    ----------
    ds : xarray.Dataset
        Spatial file as opened by :func:`pism_terra.glacier.usgs.open_pism`,
        with its grid-mapping variable still present.
    points : geopandas.GeoDataFrame
        Sites from :func:`stake_points`.
    method : {"nearest", "linear"}, default "nearest"
        Nearest cell, or bilinear interpolation from the four neighbours. A
        bilinear value that touches a masked cell falls back to the nearest
        cell for that site and time step, and ``thk`` — which only masks
        ice-free cells — is always the stake's own cell.

    Returns
    -------
    xarray.Dataset or None
        ``smb`` in m w.e./yr (see :func:`point_smb_rate`) and whichever of
        :data:`POINT_VARS` the file has, on ``(site, time)``, with
        ``glacier``/``site_name``/``x``/``y``/``lon``/``lat`` coordinates
        on ``site`` and ``month_start``/``month_end`` on ``time``. None
        when no stake lies inside the grid or the file has no usable
        balance variables.
    """
    crs = dataset_crs(ds)
    spacing = grid_spacing(ds)
    for dim in ("x", "y"):
        if ds[dim].size > 1 and ds[dim].values[1] < ds[dim].values[0]:
            ds = ds.sortby(dim)
    if spacing is None:
        spacing = (float(abs(np.diff(ds["x"].values).mean())), float(abs(np.diff(ds["y"].values).mean())))
    dx, dy = spacing

    projected = points.to_crs(crs)
    xs = projected.geometry.x.to_numpy()
    ys = projected.geometry.y.to_numpy()
    inside = (
        (xs >= float(ds["x"].min()) - dx / 2)
        & (xs <= float(ds["x"].max()) + dx / 2)
        & (ys >= float(ds["y"].min()) - dy / 2)
        & (ys <= float(ds["y"].max()) + dy / 2)
    )
    if not inside.any():
        return None
    points = points[inside].reset_index(drop=True)
    xs, ys = xs[inside], ys[inside]

    if all(v in ds for v in SMB_VARS):
        balance_vars = list(SMB_VARS)
    elif SMB_TENDENCY_VAR in ds:
        balance_vars = [SMB_TENDENCY_VAR]
    else:
        logger.warning("no surface mass balance variables (%s or %s); skipping", SMB_VARS, SMB_TENDENCY_VAR)
        return None
    variables = balance_vars + [v for v in POINT_VARS if v in ds]

    ix = _nearest_indices(ds.indexes["x"], xs)
    iy = _nearest_indices(ds.indexes["y"], ys)
    crop = ds[variables].isel(
        x=slice(max(int(ix.min()) - 1, 0), int(ix.max()) + 2),
        y=slice(max(int(iy.min()) - 1, 0), int(iy.max()) + 2),
    )
    crop = crop.drop_vars([v for v in crop.coords if v not in ("x", "y", "time")], errors="ignore").load()

    x_sel = xr.DataArray(xs, dims="site")
    y_sel = xr.DataArray(ys, dims="site")
    nearest = crop.sel(x=x_sel, y=y_sel, method="nearest")
    if method == "nearest":
        sampled = nearest
    elif method == "linear":
        sampled = crop.interp(x=x_sel, y=y_sel, method="linear")
        for var in variables:
            gaps = sampled[var].isnull() & nearest[var].notnull()
            if bool(gaps.any()):
                logger.info(
                    "%s: bilinear stencil touches a masked cell at %d site-steps; using nearest", var, int(gaps.sum())
                )
                sampled[var] = sampled[var].where(~gaps, nearest[var])
        # Whether the model has ice at the stake is a property of the stake's
        # cell; a thickness blended across the margin would leave a sliver.
        if "thk" in nearest:
            sampled["thk"] = nearest["thk"]
    else:
        raise ValueError(f"unknown sampling method {method!r}")
    sampled = sampled.drop_vars(["x", "y"], errors="ignore")

    smb = point_smb_rate(sampled, spacing)
    out = xr.Dataset({"smb": smb})
    for var in POINT_VARS:
        if var in sampled:
            out[var] = sampled[var]
    starts, ends = interval_edges(ds)
    out = out.assign_coords(
        site=("site", points["site"].to_numpy()),
        glacier=("site", points["glacier"].to_numpy()),
        site_name=("site", points["site_name"].to_numpy()),
        x=("site", xs),
        y=("site", ys),
        lon=("site", points.geometry.x.to_numpy()),
        lat=("site", points.geometry.y.to_numpy()),
        month_start=("time", starts),
        month_end=("time", ends),
    )
    out.attrs["sampling"] = method
    out.attrs["crs"] = crs
    return out


def point_smb_rate(sampled: xr.Dataset, spacing: tuple[float, float]) -> xr.DataArray:
    """
    Surface mass balance rate at the sampled points, in metres water equivalent per year.

    ``surface_accumulation_flux − surface_runoff_flux`` where the file has
    both; otherwise the per-cell ``tendency_of_ice_mass_due_to_surface_mass_flux``
    spread over the cell area. Cells without ice are masked: PISM reports a
    zero flux there, which would read as a perfectly balanced stake.

    Parameters
    ----------
    sampled : xarray.Dataset
        Point series of the balance variables (and ``thk`` when available).
    spacing : tuple of float
        ``(dx, dy)`` in metres, for the per-cell fallback.

    Returns
    -------
    xarray.DataArray
        ``smb`` in ``m year^-1``.
    """
    rho = xr.DataArray(RHO_WATER).pint.quantify("kg m^-3")
    if all(v in sampled for v in SMB_VARS):
        acc, runoff = (sampled[v].pint.quantify() for v in SMB_VARS)
        rate = ((acc - runoff) / rho).pint.to("m/year").pint.dequantify()
        source = " - ".join(SMB_VARS)
    else:
        dx, dy = spacing
        area = xr.DataArray(dx * dy).pint.quantify("m^2")
        rate = (sampled[SMB_TENDENCY_VAR].pint.quantify().pint.to("kg/year") / area / rho).pint.to("m/year")
        rate = rate.pint.dequantify()
        source = f"{SMB_TENDENCY_VAR} / (dx dy)"
    if "thk" in sampled:
        rate = rate.where(sampled["thk"] > 0)
    rate.name = "smb"
    rate.attrs = {
        "units": SPECIFIC_UNITS,
        "long_name": "surface mass balance at the stake (m w.e.)",
        "source_variable": source,
    }
    return rate


def _mean_over(values: xr.DataArray, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """
    Day-weighted mean of a monthly series over an interval.

    Parameters
    ----------
    values : xarray.DataArray
        Series with ``month_start``/``month_end`` coordinates.
    start, end : pandas.Timestamp
        Interval, ``end`` exclusive.

    Returns
    -------
    float
        The mean, NaN when the interval has no overlap or only NaNs.
    """
    starts = pd.DatetimeIndex(values["month_start"].values).values
    ends = pd.DatetimeIndex(values["month_end"].values).values
    overlap = (np.minimum(ends, np.datetime64(end)) - np.maximum(starts, np.datetime64(start))) / np.timedelta64(1, "D")
    weights = np.clip(overlap.astype(float), 0.0, None)
    data = np.asarray(values.values, dtype=float)
    keep = np.isfinite(data) & (weights > 0)
    if not keep.any():
        return float("nan")
    return float(np.average(data[keep], weights=weights[keep]))


def stake_balances(
    sampled: xr.Dataset | None,
    stakes: pd.DataFrame,
    subseasonal: pd.DataFrame | None = None,
    glacier_wide: xr.Dataset | None = None,
    run: str = NO_RUN,
) -> pd.DataFrame:
    """
    Integrate the modelled balance over each measurement's own interval.

    Parameters
    ----------
    sampled : xarray.Dataset or None
        Point series from :func:`sample_points` for one glacier's sites;
        None yields the observations with NaN model values.
    stakes : pandas.DataFrame
        The glacier's ``Glaciological_Data`` table from
        :func:`pism_terra.glacier.usgs.load_measurements`.
    subseasonal : pandas.DataFrame or None, optional
        Its ``SubSeasonal_Glaciological_Data`` table.
    glacier_wide : xarray.Dataset or None, optional
        The glacier-wide solution, whose ``Ba_Date`` of the previous balance
        year stands in for a missing previous fall visit.
    run : str, optional
        Label of the run the samples come from.

    Returns
    -------
    pandas.DataFrame
        One row per measurement with :data:`TABLE_COLUMNS` (``visit`` is
        the sub-seasonal table's row number, -1 otherwise). ``bw`` runs
        from the site's previous fall visit (else the previous balance
        year's ``Ba_Date``) to the spring visit, ``bs`` from spring to fall
        (observed as ``ba − bw``), ``ba`` from the previous fall to this
        fall, ``db`` between the two visits; ``usurf`` is the modelled
        surface over the same interval.

    Notes
    -----
    Intervals the model record does not fully cover give a NaN model value
    (see :func:`pism_terra.glacier.usgs.integrate_rate`).
    """
    ba_dates: dict[int, pd.Timestamp] = {}
    if glacier_wide is not None and "Ba_Date" in glacier_wide:
        for year, date in zip(glacier_wide["time"].values, pd.to_datetime(glacier_wide["Ba_Date"].values)):
            if pd.notna(date):
                ba_dates[int(year)] = date

    def series(site: str) -> tuple[xr.DataArray | None, xr.DataArray | None]:
        """
        The modelled balance and surface at a site.

        Parameters
        ----------
        site : str
            ``<glacier>:<site_name>``.

        Returns
        -------
        tuple
            ``(smb, usurf)``, None where unavailable.
        """
        if sampled is None or site not in sampled["site"].values:
            return None, None
        at = sampled.sel(site=site)
        return at["smb"], at["usurf"] if "usurf" in at else None

    def integrate(smb, usurf, start, end) -> tuple[float, float]:
        """
        Integrate the balance and average the surface over an interval.

        Parameters
        ----------
        smb : xarray.DataArray or None
            Balance rate.
        usurf : xarray.DataArray or None
            Surface elevation.
        start, end : pandas.Timestamp
            Interval.

        Returns
        -------
        tuple of float
            ``(balance, surface)``.
        """
        if smb is None or pd.isna(start) or pd.isna(end) or end <= start:
            return float("nan"), float("nan")
        balance = integrate_rate(smb, start, end)
        surface = _mean_over(usurf, start, end) if usurf is not None else float("nan")
        return balance, surface

    rows: list[dict[str, Any]] = []
    for (glacier, site_name), group in stakes.groupby(["glacier", "site_name"], sort=True):
        site = f"{glacier}:{site_name}"
        smb, usurf = series(site)
        group = group.sort_values("Year")
        previous_fall = {int(y): d for y, d in zip(group["Year"], group["fall_date"]) if pd.notna(d)}
        for row in group.itertuples(index=False):
            year = int(row.Year)
            spring, fall = row.spring_date, row.fall_date
            start = previous_fall.get(year - 1, ba_dates.get(year - 1, pd.NaT))
            elevation = float(row.elevation) if hasattr(row, "elevation") and pd.notna(row.elevation) else np.nan
            bw_obs = float(row.bw) if hasattr(row, "bw") and pd.notna(row.bw) else np.nan
            ba_obs = float(row.ba) if hasattr(row, "ba") and pd.notna(row.ba) else np.nan
            for var, obs, interval in (
                ("bw", bw_obs, (start, spring)),
                ("bs", ba_obs - bw_obs, (spring, fall)),
                ("ba", ba_obs, (start, fall)),
            ):
                model, surface = integrate(smb, usurf, *interval)
                if np.isnan(obs) and np.isnan(model):
                    continue
                rows.append(
                    {
                        "glacier": glacier,
                        "site": site_name,
                        "year": year,
                        "variable": var,
                        "start": interval[0],
                        "end": interval[1],
                        "obs": obs,
                        "model": model,
                        "elevation": elevation,
                        "usurf": surface,
                        "visit": -1,
                        "run": run,
                    }
                )

    if subseasonal is not None:
        for row in subseasonal.itertuples(index=True):
            site = f"{row.glacier}:{row.site_name}"
            smb, usurf = series(site)
            model, surface = integrate(smb, usurf, row.Date1, row.Date2)
            obs = float(row.db) if pd.notna(row.db) else np.nan
            if np.isnan(obs) and np.isnan(model):
                continue
            rows.append(
                {
                    "glacier": row.glacier,
                    "site": row.site_name,
                    "year": int(row.Year),
                    "variable": "db",
                    "start": row.Date1,
                    "end": row.Date2,
                    "obs": obs,
                    "model": model,
                    "elevation": (
                        float(row.Elevation) if hasattr(row, "Elevation") and pd.notna(row.Elevation) else np.nan
                    ),
                    "usurf": surface,
                    "visit": int(row.Index),
                    "run": run,
                }
            )
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def _model_line(table: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the runs to the series the plots draw: the single run, or the median.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows of one variable at one site.

    Returns
    -------
    pandas.DataFrame
        One row per ``(year, start, visit)`` with ``obs``, ``model`` (median over
        runs), ``low``/``high`` (5-95 % over runs) and ``n_runs``.
    """
    grouped = table.groupby(["year", "start", "visit"], sort=True)
    out = grouped.agg(
        obs=("obs", "first"),
        model=("model", "median"),
        low=("model", lambda s: s.quantile(0.05)),
        high=("model", lambda s: s.quantile(0.95)),
        n_runs=("model", lambda s: int(s.notna().sum())),
        elevation=("elevation", "first"),
        usurf=("usurf", "mean"),
    )
    return out.reset_index()


def stake_skill(table: pd.DataFrame, ice_free: pd.Series | None = None) -> pd.DataFrame:
    """
    Score the modelled balances against the stakes, per site and pooled per glacier.

    Scores use the ensemble median where several runs sampled a site.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows from :func:`stake_balances`, possibly from several runs.
    ice_free : pandas.Series or None, optional
        Months the model had no ice at each site, indexed by
        ``(glacier, site)``; reported alongside the scores.

    Returns
    -------
    pandas.DataFrame
        :data:`SKILL_COLUMNS`; the pooled rows carry ``site`` =
        :data:`POOLED`. Sites without a single modelled value are left out
        unless the model had no ice there (``n_ice_free_months`` > 0).
        Empty when nothing was modelled.
    """
    rows: list[dict[str, Any]] = []
    if table.empty or table["model"].isna().all():
        return pd.DataFrame(columns=SKILL_COLUMNS)
    for (glacier, site, var), group in table.groupby(["glacier", "site", "variable"], sort=True):
        line = _model_line(group)
        offset = float((line["usurf"] - line["elevation"]).mean()) if line["usurf"].notna().any() else np.nan
        months = np.nan
        if ice_free is not None and (glacier, site) in ice_free.index:
            months = int(ice_free.loc[(glacier, site)])
        rows.append(
            {
                "glacier": glacier,
                "site": site,
                "variable": var,
                "season": STAKE_VARS[var],
                **score(line["obs"].to_numpy(), line["model"].to_numpy()),
                "elevation_offset": offset,
                "n_ice_free_months": months,
            }
        )
    for (glacier, var), group in table.groupby(["glacier", "variable"], sort=True):
        pooled = pd.concat([_model_line(g) for _, g in group.groupby("site")], ignore_index=True)
        rows.append(
            {
                "glacier": glacier,
                "site": POOLED,
                "variable": var,
                "season": STAKE_VARS[var],
                **score(pooled["obs"].to_numpy(), pooled["model"].to_numpy()),
                "elevation_offset": float((pooled["usurf"] - pooled["elevation"]).mean()),
                "n_ice_free_months": np.nan,
            }
        )
    skill = pd.DataFrame(rows, columns=SKILL_COLUMNS)
    # A site nothing was modelled at says nothing — unless the model had no
    # ice there, which is worth reporting.
    skill = skill[(skill["n"] > 0) | (skill["n_ice_free_months"].fillna(0) > 0)]
    order = {var: i for i, var in enumerate(STAKE_VARS)}
    skill["_order"] = skill["variable"].map(order)
    skill["_pooled"] = skill["site"] == POOLED
    return (
        skill.sort_values(["glacier", "_pooled", "site", "_order"])
        .drop(columns=["_order", "_pooled"])
        .reset_index(drop=True)
    )


def _site_order(table: pd.DataFrame) -> list[str]:
    """
    Order the sites from the highest to the lowest, by mean measured elevation.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows of one glacier.

    Returns
    -------
    list of str
        Site names.
    """
    elevation = table.groupby("site")["elevation"].mean().sort_values(ascending=False)
    return elevation.index.tolist()


def plot_stakes(
    table: pd.DataFrame, glacier: str, rgi_id: str, output_dir: Path | str, skill: pd.DataFrame | None = None
) -> Path:
    """
    Plot every site of a glacier: observed balances as markers, modelled as lines.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows from :func:`stake_balances` for one glacier (all runs).
    glacier : str
        Glacier name for the title and file name.
    rgi_id : str
        RGI identifier for the title and file name.
    output_dir : Path or str
        Directory the figure is written to.
    skill : pandas.DataFrame or None, optional
        Scores from :func:`stake_skill`; the annual r and MAE of each site
        are written into its panel.

    Returns
    -------
    Path
        The PNG written (a PDF sits next to it).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"usgs_benchmark_stakes_{glacier}_{rgi_id}"
    sites = _site_order(table)
    ncols = min(3, len(sites))
    nrows = math.ceil(len(sites) / ncols)

    with mpl.rc_context(rc=rc_params):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(2.6 * ncols + 0.4, 1.7 * nrows + 0.9),
            sharex=True,
            layout="constrained",
            squeeze=False,
        )
        for ax, site in zip(axes.ravel(), sites):
            rows = table[table["site"] == site]
            for var in ("bw", "bs", "ba"):
                sub = rows[rows["variable"] == var]
                if sub.empty:
                    continue
                line = _model_line(sub)
                style = STAKE_STYLE[var]
                ax.plot(
                    line["year"],
                    line["obs"],
                    color=style["color"],
                    marker=style["marker"],
                    ms=2,
                    lw=0.5,
                    label=f"Observed {STAKE_VARS[var]}",
                )
                if line["model"].notna().any():
                    mstyle = STAKE_MODEL_STYLE[var]
                    n_runs = int(line["n_runs"].max())
                    if n_runs > 1:
                        ax.fill_between(line["year"], line["low"], line["high"], color=mstyle["color"], alpha=0.2, lw=0)
                    ax.plot(
                        line["year"],
                        line["model"],
                        color=mstyle["color"],
                        ls=mstyle["ls"],
                        lw=0.9,
                        label=f"PISM {STAKE_VARS[var]}" + (f" (median, n={n_runs})" if n_runs > 1 else ""),
                    )
            elevation = rows["elevation"].mean()
            title = f"{site} ({elevation:.0f} m)" if np.isfinite(elevation) else site
            ax.set_title(title, fontsize=6)
            ax.axhline(y=0, color="k", ls="dotted", lw=0.5)
            if skill is not None and not skill.empty:
                annual = skill[(skill["site"] == site) & (skill["variable"] == "ba")]
                if not annual.empty and annual.iloc[0]["n"] > 0:
                    row = annual.iloc[0]
                    r = f"r={row['r']:.2f}" if np.isfinite(row["r"]) else "r=n/a"
                    ax.text(
                        0.02,
                        0.04,
                        f"ba {r} MAE={row['mae']:.2f} n={int(row['n'])}",
                        transform=ax.transAxes,
                        fontsize=4.5,
                        family="monospace",
                        ha="left",
                        va="bottom",
                    )
        for ax in axes.ravel()[len(sites) :]:
            ax.set_visible(False)
        # the lowest visible panel of each column carries the axis label
        for column in range(ncols):
            visible = [ax for ax in axes[:, column] if ax.get_visible()]
            if visible:
                visible[-1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
                visible[-1].tick_params(labelbottom=True)
                visible[-1].set_xlabel("Balance year")
        fig.supylabel("Mass balance (m w.e.)")
        fig.suptitle(f"{glacier} ({rgi_id}) stakes")
        handles, labels = [], []
        for ax in axes.ravel():
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        legend = fig.legend(handles, labels, loc="outside lower center", ncol=3)
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)
        png = output_dir / f"{stem}.png"
        fig.savefig(png, dpi=300)
        fig.savefig(output_dir / f"{stem}.pdf")
        plt.close(fig)
    return png


def plot_scatter(
    table: pd.DataFrame, glacier: str, rgi_id: str, output_dir: Path | str, skill: pd.DataFrame | None = None
) -> Path | None:
    """
    Modelled against observed balances, pooled over a glacier's sites.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows from :func:`stake_balances` for one glacier (all runs).
    glacier : str
        Glacier name for the title and file name.
    rgi_id : str
        RGI identifier for the title and file name.
    output_dir : Path or str
        Directory the figure is written to.
    skill : pandas.DataFrame or None, optional
        Scores from :func:`stake_skill`; the pooled scores label each panel.

    Returns
    -------
    Path or None
        The PNG written (a PDF sits next to it); None when nothing was
        modelled.
    """
    lines = {
        var: pd.concat([_model_line(g) for _, g in sub.groupby("site")], ignore_index=True)
        for var, sub in table.groupby("variable")
    }
    lines = {var: line for var, line in lines.items() if line["model"].notna().any() and line["obs"].notna().any()}
    if not lines:
        return None
    output_dir = Path(output_dir)
    stem = f"usgs_benchmark_stakes_{glacier}_{rgi_id}_scatter"
    variables = [var for var in STAKE_VARS if var in lines]

    with mpl.rc_context(rc=rc_params):
        fig, axes = plt.subplots(
            1, len(variables), figsize=(2.2 * len(variables) + 0.4, 2.4), layout="constrained", squeeze=False
        )
        for ax, var in zip(axes[0], variables):
            line = lines[var]
            scatter = ax.scatter(line["obs"], line["model"], c=line["elevation"], cmap="viridis", s=5, lw=0, alpha=0.8)
            both = np.concatenate([line["obs"].to_numpy(), line["model"].to_numpy()])
            both = both[np.isfinite(both)]
            lo, hi = float(both.min()), float(both.max())
            pad = 0.05 * (hi - lo) if hi > lo else 0.5
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="k", lw=0.5, ls="dotted")
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal")
            ax.set_xlabel(f"Observed {var} (m w.e.)")
            ax.set_ylabel(f"PISM {var} (m w.e.)")
            if skill is not None and not skill.empty:
                pooled = skill[(skill["site"] == POOLED) & (skill["variable"] == var)]
                if not pooled.empty:
                    row = pooled.iloc[0]
                    r = f"r={row['r']:.2f}" if np.isfinite(row["r"]) else "r=n/a"
                    ax.text(
                        0.03,
                        0.97,
                        f"{r}\nMAE={row['mae']:.2f}\nbias={row['bias']:+.2f}\nn={int(row['n'])}",
                        transform=ax.transAxes,
                        fontsize=5,
                        family="monospace",
                        ha="left",
                        va="top",
                    )
        colorbar = fig.colorbar(scatter, ax=axes[0].tolist(), shrink=0.8, pad=0.02)
        colorbar.set_label("Stake elevation (m)")
        fig.suptitle(f"{glacier} ({rgi_id}) stakes")
        png = output_dir / f"{stem}.png"
        fig.savefig(png, dpi=300)
        fig.savefig(output_dir / f"{stem}.pdf")
        plt.close(fig)
    return png


GRADIENT_COLUMNS = ["variable", "source", "n", "ela", "ela_in_range", "gradient_below", "gradient_above", "rmse"]
#: Fewest points for a balance gradient fit.
GRADIENT_MIN_POINTS = 4


def fit_gradient(balance: np.ndarray, elevation: np.ndarray, n_candidates: int = 400) -> dict[str, float] | None:
    """
    Fit a two-piece linear balance gradient hinged at the ELA.

    The balance is modelled as ``b(z) = k_below (z - ELA)`` below the ELA and
    ``k_above (z - ELA)`` above it, so both pieces are zero at the ELA. For a
    given ELA the two slopes are a linear least-squares problem; the ELA is
    found by scanning candidates over the stakes' elevation range widened by
    itself on either side, since a winter balance can be positive at every
    stake and its ELA then lies below the lowest one. A side of the hinge
    is either empty or holds at least :data:`GRADIENT_MIN_POINTS` points.

    Parameters
    ----------
    balance : numpy.ndarray
        Balances in m w.e.
    elevation : numpy.ndarray
        Stake elevations in m.
    n_candidates : int, default 400
        Number of candidate ELAs scanned.

    Returns
    -------
    dict or None
        ``n``, ``ela`` (m), ``ela_in_range`` (1.0 when the ELA lies between
        the lowest and highest stake), ``gradient_below`` and
        ``gradient_above`` (m w.e. per m; NaN for a side without points)
        and ``rmse`` (m w.e.). None with fewer than
        :data:`GRADIENT_MIN_POINTS` points or a single elevation.
    """
    b = np.asarray(balance, dtype=float)
    z = np.asarray(elevation, dtype=float)
    keep = np.isfinite(b) & np.isfinite(z)
    b, z = b[keep], z[keep]
    if b.size < GRADIENT_MIN_POINTS or np.ptp(z) == 0:
        return None
    span = np.ptp(z)
    candidates = np.linspace(z.min() - span, z.max() + span, n_candidates)
    best_sse, best_ela, best_slopes = np.inf, float("nan"), np.full(2, np.nan)
    for ela in candidates:
        dz = z - ela
        # A hinge is only allowed where each side has enough points to fix a
        # slope of its own; otherwise a few outlying stakes get a piece to
        # themselves.
        n_below, n_above = int((dz < 0).sum()), int((dz >= 0).sum())
        if 0 < n_below < GRADIENT_MIN_POINTS or 0 < n_above < GRADIENT_MIN_POINTS:
            continue
        design = np.column_stack([np.where(dz < 0, dz, 0.0), np.where(dz >= 0, dz, 0.0)])
        used = design.any(axis=0)
        slopes = np.full(2, np.nan)
        slopes[used], *_ = np.linalg.lstsq(design[:, used], b, rcond=None)
        residual = b - design[:, used] @ slopes[used]
        sse = float(residual @ residual)
        if sse < best_sse:
            best_sse, best_ela, best_slopes = sse, float(ela), slopes
    if not np.isfinite(best_sse):
        return None
    return {
        "n": float(b.size),
        "ela": best_ela,
        "ela_in_range": float(z.min() <= best_ela <= z.max()),
        "gradient_below": float(best_slopes[0]),
        "gradient_above": float(best_slopes[1]),
        "rmse": float(np.sqrt(best_sse / b.size)),
    }


def _gradient_line(fit: dict[str, float], z_min: float, z_max: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a fitted gradient over the stakes' elevation range.

    Parameters
    ----------
    fit : dict
        Output of :func:`fit_gradient`.
    z_min, z_max : float
        Elevation range of the points.

    Returns
    -------
    tuple of numpy.ndarray
        ``(balance, elevation)`` along the two pieces.
    """
    ela = fit["ela"]
    z = np.linspace(z_min, z_max, 40)
    if z_min < ela < z_max:
        z = np.sort(np.append(z, ela))
    below = np.nan_to_num(fit["gradient_below"])
    above = np.nan_to_num(fit["gradient_above"])
    b = np.where(z < ela, below * (z - ela), above * (z - ela))
    return b, z


def gradient_fits(table: pd.DataFrame) -> pd.DataFrame:
    """
    Fit the observed and modelled balance gradient of each season.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows from :func:`stake_balances` for one glacier (all runs).

    Returns
    -------
    pandas.DataFrame
        :data:`GRADIENT_COLUMNS`, one row per season and source (``obs``
        or ``model``, the latter the ensemble median) that could be fitted.
    """
    rows: list[dict[str, Any]] = []
    seasonal = table[(table["variable"] != "db") & table["elevation"].notna()]
    for var in ("bw", "bs", "ba"):
        sub = seasonal[seasonal["variable"] == var]
        if sub.empty:
            continue
        line = pd.concat([_model_line(g) for _, g in sub.groupby("site")], ignore_index=True)
        for source, column in (("obs", "obs"), ("model", "model")):
            fit = fit_gradient(line[column].to_numpy(), line["elevation"].to_numpy())
            if fit is not None:
                rows.append({"variable": var, "source": source, **fit})
    return pd.DataFrame(rows, columns=GRADIENT_COLUMNS)


def plot_gradient(
    table: pd.DataFrame, glacier: str, rgi_id: str, output_dir: Path | str, fits: pd.DataFrame | None = None
) -> Path | None:
    """
    Plot balances against stake elevation, one panel per season, with their gradient fits.

    Observed balances are filled markers and modelled ones hollow; the
    two-piece fits of :func:`fit_gradient` are drawn as a solid (observed)
    and dashed (modelled) line through the ELA, which is marked on the
    elevation axis.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows from :func:`stake_balances` for one glacier (all runs).
    glacier : str
        Glacier name for the title and file name.
    rgi_id : str
        RGI identifier for the title and file name.
    output_dir : Path or str
        Directory the figure is written to.
    fits : pandas.DataFrame or None, optional
        Output of :func:`gradient_fits`; computed here when omitted.

    Returns
    -------
    Path or None
        The PNG written (a PDF sits next to it); None when no measurement
        carries an elevation.
    """
    seasonal = table[(table["variable"] != "db") & table["elevation"].notna()]
    if seasonal.empty:
        return None
    if fits is None:
        fits = gradient_fits(table)
    output_dir = Path(output_dir)
    stem = f"usgs_benchmark_stakes_{glacier}_{rgi_id}_gradient"
    modelled = seasonal["model"].notna().any()
    z_min, z_max = float(seasonal["elevation"].min()), float(seasonal["elevation"].max())

    with mpl.rc_context(rc=rc_params):
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), layout="constrained", sharey=True)
        for ax, var in zip(axes, ("bw", "bs", "ba")):
            season = STAKE_VARS[var]
            style = STAKE_STYLE[var]
            sub = seasonal[seasonal["variable"] == var]
            ax.axvline(x=0, color="k", ls="dotted", lw=0.5)
            ax.set_title(season.capitalize())
            ax.set_xlabel(f"{season.capitalize()} balance (m w.e.)")
            if sub.empty:
                continue
            line = pd.concat([_model_line(g) for _, g in sub.groupby("site")], ignore_index=True)
            notes = []
            for source, column, filled, ls in (("obs", "obs", True, "-"), ("model", "model", False, "--")):
                points = line[line[column].notna()]
                if points.empty:
                    continue
                label = "Observed" if source == "obs" else "PISM"
                kwargs: dict[str, Any] = {"s": 7, "marker": style["marker"], "alpha": 0.7, "label": label}
                if filled:
                    kwargs.update(color=style["color"], lw=0)
                else:
                    kwargs.update(facecolors="none", edgecolors=STAKE_MODEL_STYLE[var]["color"], lw=0.5)
                ax.scatter(points[column], points["elevation"], **kwargs)
                fit = fits[(fits["variable"] == var) & (fits["source"] == source)]
                if fit.empty:
                    continue
                fit = fit.iloc[0]
                b, z = _gradient_line(fit, z_min, z_max)
                ax.plot(b, z, color=style["color"] if filled else STAKE_MODEL_STYLE[var]["color"], ls=ls, lw=1.0)
                if fit["ela_in_range"]:
                    ax.plot(0.0, fit["ela"], marker="_", ms=8, mew=1.0, color=style["color"], ls="none")
                below = f"{100 * fit['gradient_below']:.2f}" if np.isfinite(fit["gradient_below"]) else "n/a"
                above = f"{100 * fit['gradient_above']:.2f}" if np.isfinite(fit["gradient_above"]) else "n/a"
                flag = "" if fit["ela_in_range"] else "*"
                notes.append(f"{label:8s} ELA {fit['ela']:.0f}{flag} m  db/dz {below} | {above}")
            if notes:
                ax.text(
                    0.98,
                    0.02,
                    "\n".join(notes),
                    transform=ax.transAxes,
                    fontsize=4.5,
                    family="monospace",
                    ha="right",
                    va="bottom",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
                )
        axes[0].set_ylabel("Stake elevation (m)")
        note = "db/dz in m w.e. per 100 m, below | above the ELA; * ELA outside the stake range"
        fig.suptitle(f"{glacier} ({rgi_id}) stakes" + ("" if modelled else ", observations only") + "\n" + note)
        handles, labels = axes[-1].get_legend_handles_labels()
        if not labels:
            handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc="outside lower center", ncol=2)
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)
        png = output_dir / f"{stem}.png"
        fig.savefig(png, dpi=300)
        fig.savefig(output_dir / f"{stem}.pdf")
        plt.close(fig)
    return png


def to_dataset(table: pd.DataFrame, sampled: Sequence[xr.Dataset] = ()) -> xr.Dataset:
    """
    Arrange a glacier's rows as arrays for a NetCDF file.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows from :func:`stake_balances` for one glacier (all runs).
    sampled : sequence of xarray.Dataset, optional
        The point series the rows came from, for the site coordinates.

    Returns
    -------
    xarray.Dataset
        ``bw``/``bs``/``ba`` observed on ``(site, time)`` (balance years)
        with ``<var>_model`` and ``usurf`` on ``(run, site, time)``, the
        interval bounds as ``<var>_start``/``<var>_end``, the stake
        ``elevation``; sub-seasonal ``db``/``db_model`` on a ``visit``
        dimension numbered like the rows of the release's sub-seasonal table. Site ``lon``/``lat``/``x``/``y`` are attached when
        *sampled* provides them.
    """
    seasonal = table[table["variable"] != "db"]
    runs = sorted(r for r in table["run"].unique() if r != NO_RUN) or [NO_RUN]
    out = xr.Dataset()
    if not seasonal.empty:
        obs = seasonal.drop_duplicates(subset=["site", "year", "variable"])
        for var in ("bw", "bs", "ba"):
            sub = obs[obs["variable"] == var]
            if sub.empty:
                continue
            out[var] = xr.DataArray.from_series(sub.set_index(["site", "year"])["obs"]).rename({"year": "time"})
            out[var].attrs = {"units": SPECIFIC_UNITS, "long_name": f"measured {STAKE_VARS[var]} balance (m w.e.)"}
            for column in ("start", "end"):
                out[f"{var}_{column}"] = xr.DataArray.from_series(sub.set_index(["site", "year"])[column]).rename(
                    {"year": "time"}
                )
            model = seasonal[seasonal["variable"] == var].set_index(["run", "site", "year"])["model"]
            out[f"{var}_model"] = xr.DataArray.from_series(model).rename({"year": "time"}).reindex(run=runs)
            out[f"{var}_model"].attrs = {
                "units": SPECIFIC_UNITS,
                "long_name": f"modelled {STAKE_VARS[var]} balance (m w.e.)",
            }
        annual = seasonal[seasonal["variable"] == "ba"] if (seasonal["variable"] == "ba").any() else seasonal
        out["elevation"] = xr.DataArray.from_series(
            annual.drop_duplicates(subset=["site", "year"]).set_index(["site", "year"])["elevation"]
        ).rename({"year": "time"})
        out["elevation"].attrs = {"units": "m", "long_name": "stake elevation"}
        out["usurf"] = (
            xr.DataArray.from_series(annual.set_index(["run", "site", "year"])["usurf"])
            .rename({"year": "time"})
            .reindex(run=runs)
        )
        out["usurf"].attrs = {"units": "m", "long_name": "modelled surface elevation over the balance year"}
    visits = table[table["variable"] == "db"]
    if not visits.empty:
        keys = visits.drop_duplicates(subset=["visit"]).sort_values("visit")
        out["db"] = xr.DataArray.from_series(keys.set_index("visit")["obs"])
        out["db"].attrs = {"units": SPECIFIC_UNITS, "long_name": "measured balance between two visits (m w.e.)"}
        out["db_model"] = xr.DataArray.from_series(
            visits.drop_duplicates(subset=["run", "visit"]).set_index(["run", "visit"])["model"]
        ).reindex(run=runs)
        out["db_model"].attrs = {"units": SPECIFIC_UNITS, "long_name": "modelled balance between two visits (m w.e.)"}
        out = out.assign_coords(
            db_site=("visit", keys["site"].to_numpy()),
            db_start=("visit", keys["start"].to_numpy()),
            db_end=("visit", keys["end"].to_numpy()),
        )
    if "site" in out.dims and sampled:
        coords = xr.concat(
            [s[["smb"]].isel(time=0).drop_vars(["time", "month_start", "month_end"], errors="ignore") for s in sampled],
            dim="site",
        )
        coords = coords.drop_duplicates("site").swap_dims({"site": "site_name"}).drop_vars("site")
        names = out["site"].values
        keep = [n for n in names if n in coords["site_name"].values]
        if keep:
            at = coords.sel(site_name=keep).reindex(site_name=names)
            out = out.assign_coords({c: ("site", at[c].values) for c in ("lon", "lat", "x", "y") if c in at.coords})
    if "time" in out.dims:
        out["time"].attrs = {"long_name": "balance year"}
    return drop_grid_mapping(out)


def run_pipeline(
    run_dir: Path | str | None,
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    output_dir: Path | str = ".",
    rgi_file: Path | str | None = None,
    method: str = "nearest",
    force_overwrite: bool = False,
) -> pd.DataFrame:
    """
    Download, match, sample, integrate, score and plot every glacier's stakes.

    Parameters
    ----------
    run_dir : Path or str or None
        Directory searched recursively for ``spatial_*.nc``; None plots
        the observations alone.
    data_dir : Path or str, optional
        Cache for the ScienceBase archives and RGI outlines.
    output_dir : Path or str, optional
        Where the match and skill tables go; each glacier's figures and
        NetCDF file are written to a ``<rgi_id>`` sub-directory of it.
    rgi_file : Path or str or None, optional
        Outline file to match against instead of the downloaded regions.
    method : {"nearest", "linear"}, optional
        How the grid is sampled at a stake (see :func:`sample_points`).
    force_overwrite : bool, default False
        Re-download the archives.

    Returns
    -------
    pandas.DataFrame
        The match table with ``n_stakes`` (sites with measurements),
        ``n_sampled`` (sites some run covered), ``n_runs`` and ``figure``
        columns added.
    """
    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = download_usgs_benchmark(data_dir, force_overwrite=force_overwrite)
    sites = load_sites(paths["sites"])
    rgi = load_rgi_glaciers(data_dir, rgi_file=rgi_file, force_overwrite=force_overwrite)
    matches = match_rgi_ids(sites, rgi)

    stakes: dict[str, pd.DataFrame] = {}
    subseasonal: dict[str, pd.DataFrame] = {}
    glacier_wide: dict[str, xr.Dataset] = {}
    for row in matches.itertuples(index=False):
        if pd.isna(row.rgi_id):
            continue
        table = load_measurements(paths["data"], row.glacier, "stakes")
        if table is None:
            logger.info("%s: no stake measurements; skipping", row.glacier)
            continue
        stakes[row.glacier] = table
        sub = load_measurements(paths["data"], row.glacier, "subseasonal")
        if sub is not None:
            subseasonal[row.glacier] = sub
        wide = load_glacier_wide(paths["data"], row.glacier, fallback_area_km2=row.area_km2)
        if wide is not None:
            glacier_wide[row.glacier] = wide
    points = stake_points(sites, [*stakes.values(), *subseasonal.values()])
    logger.info("%d measured sites on %d glaciers", len(points), len(stakes))

    files = find_spatial_files(run_dir) if run_dir is not None else []
    logger.info("%d spatial files under %s", len(files), run_dir)
    tables: dict[str, list[pd.DataFrame]] = {glacier: [] for glacier in stakes}
    samples: dict[str, list[xr.Dataset]] = {glacier: [] for glacier in stakes}
    ice_free: dict[tuple[str, str], int] = {}
    for file in files:
        label = run_label(file, run_dir)
        with open_pism(file) as ds:
            if not is_monthly(ds["time"].values):
                logger.info("%s: not monthly output; skipping", label)
                continue
            sampled = sample_points(ds, points, method=method)
        if sampled is None:
            logger.info("%s: no stake inside the grid", label)
            continue
        for glacier in np.unique(sampled["glacier"].values):
            at = sampled.sel(site=sampled["glacier"].values == glacier)
            logger.info("%s: %d sites of %s sampled", label, at.sizes["site"], glacier)
            tables[glacier].append(
                stake_balances(at, stakes[glacier], subseasonal.get(glacier), glacier_wide.get(glacier), run=label)
            )
            samples[glacier].append(at)
            if "thk" in at:
                for site_name, months in zip(at["site_name"].values, (at["thk"] <= 0).sum("time").values):
                    key = (str(glacier), str(site_name))
                    ice_free[key] = max(ice_free.get(key, 0), int(months))

    figures: list[str | None] = []
    n_runs: list[int] = []
    n_stakes: list[int] = []
    n_sampled: list[int] = []
    skills: list[pd.DataFrame] = []
    gradients: list[pd.DataFrame] = []
    for row in matches.itertuples(index=False):
        glacier = row.glacier
        if glacier not in stakes:
            figures.append(None)
            n_runs.append(0)
            n_stakes.append(0)
            n_sampled.append(0)
            continue
        table = (
            pd.concat(tables[glacier], ignore_index=True)
            if tables[glacier]
            else stake_balances(None, stakes[glacier], subseasonal.get(glacier), glacier_wide.get(glacier))
        )
        glacier_dir = rgi_output_dir(output_dir, row.rgi_id)
        series = pd.Series(ice_free, dtype=float) if ice_free else None
        if series is not None:
            series.index = pd.MultiIndex.from_tuples(series.index) if len(series) else series.index
        skill = stake_skill(table, series)
        if not skill.empty:
            skills.append(skill.assign(rgi_id=row.rgi_id, units=SPECIFIC_UNITS))
            for line in skill[skill["site"] == POOLED].itertuples(index=False):
                r = f"r={line.r:.2f}" if np.isfinite(line.r) else "r=n/a"
                logger.info("%s: %-12s %s MAE=%.3g %s n=%d", glacier, line.season, r, line.mae, SPECIFIC_LABEL, line.n)
        png = plot_stakes(table, glacier, row.rgi_id, glacier_dir, skill=skill)
        plot_scatter(table, glacier, row.rgi_id, glacier_dir, skill=skill)
        fits = gradient_fits(table)
        plot_gradient(table, glacier, row.rgi_id, glacier_dir, fits=fits)
        if not fits.empty:
            fits.to_csv(glacier_dir / f"usgs_benchmark_stakes_{glacier}_{row.rgi_id}_gradient.csv", index=False)
            gradients.append(fits.assign(glacier=glacier, rgi_id=row.rgi_id))
        out = to_dataset(table, samples[glacier])
        out.attrs.update({"glacier": glacier, "rgi_id": row.rgi_id, "sampling": method})
        out.to_netcdf(glacier_dir / f"usgs_benchmark_stakes_{glacier}_{row.rgi_id}.nc")
        table.to_csv(glacier_dir / f"usgs_benchmark_stakes_{glacier}_{row.rgi_id}.csv", index=False)
        figures.append(str(png))
        n_runs.append(len(tables[glacier]))
        n_stakes.append(int((points["glacier"] == glacier).sum()))
        n_sampled.append(len({str(s) for at in samples[glacier] for s in at["site_name"].values}))

    matches = matches.assign(n_stakes=n_stakes, n_sampled=n_sampled, n_runs=n_runs, figure=figures)
    matches.to_csv(output_dir / "usgs_benchmark_stakes_rgi_match.csv", index=False)
    columns = ["glacier", "rgi_id", *SKILL_COLUMNS[1:], "units"]
    skill_table = pd.concat(skills, ignore_index=True) if skills else pd.DataFrame(columns=columns)
    skill_table[columns].to_csv(output_dir / "usgs_benchmark_stakes_skill.csv", index=False)
    gradient_columns = ["glacier", "rgi_id", *GRADIENT_COLUMNS]
    gradient_table = pd.concat(gradients, ignore_index=True) if gradients else pd.DataFrame(columns=gradient_columns)
    gradient_table[gradient_columns].to_csv(output_dir / "usgs_benchmark_stakes_gradients.csv", index=False)
    return matches


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Compare USGS stake measurements with PISM spatial output.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments (excluding the program name). When
        ``None``, :data:`sys.argv` is used.

    Returns
    -------
    pandas.DataFrame
        The glacier-to-RGI match table.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = (
        "Sample PISM spatial output at the USGS benchmark-glacier stakes and compare the modelled surface "
        "mass balance (m w.e.) with the measured winter, summer, annual and sub-seasonal balances."
    )
    parser.add_argument(
        "RUN_DIR",
        nargs="?",
        default=None,
        help="Directory searched recursively for spatial_*.nc files. Omit to plot the observations alone.",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Cache for the USGS archives and RGI outlines.")
    parser.add_argument(
        "--output-dir", default=".", help="Directory for the tables; figures and NetCDF files go to <rgi_id>/ below it."
    )
    parser.add_argument(
        "--rgi-glacier-file",
        default=None,
        help="RGI v7 glacier (-G-) outlines to match against instead of downloading regions 01 and 02.",
    )
    parser.add_argument(
        "--method",
        choices=("nearest", "linear"),
        default="nearest",
        help="Sample the nearest cell, or interpolate bilinearly from the four neighbours.",
    )
    parser.add_argument("--force-overwrite", action="store_true", default=False, help="Re-download the archives.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "usgs_benchmark_stakes.log")

    matches = run_pipeline(
        args.RUN_DIR,
        data_dir=args.data_dir,
        output_dir=output_dir,
        rgi_file=args.rgi_glacier_file,
        method=args.method,
        force_overwrite=args.force_overwrite,
    )
    print(matches.to_string(index=False))
    return matches


def cli(argv: Sequence[str] | None = None) -> int:
    """
    Console entry point.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments (excluding the program name).

    Returns
    -------
    int
        Exit code (``0`` on success).
    """
    _ = main(argv=argv)
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(cli())
