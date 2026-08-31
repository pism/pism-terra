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

# pylint: disable=unused-import,too-many-positional-arguments

"""
USGS benchmark-glacier mass balances against PISM output.

The USGS ScienceBase release *Compiled Input Data and Glacier-Wide Mass
Balances* (https://doi.org/10.5066/F7HD7SRF) publishes geodetically
calibrated glacier-wide winter, summer and annual balances for a handful of
North American glaciers in metres water equivalent, keyed by glacier *name*
rather than RGI ID. This module downloads the release, finds the RGI v7
glacier (``-G-``) outline each glacier lives in, converts the balances to
Gt/yr with the release's own time-varying glacier area, and plots each
glacier's series with whatever PISM ``tendency_of_ice_mass`` exists for that
RGI ID under a run directory. Outputs are grouped by RGI ID in
sub-directories of the output directory. The per-stake counterpart is
:mod:`pism_terra.glacier.usgs_benchmark_stakes`.
"""

import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cf_xarray
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pint_xarray
import xarray as xr
from matplotlib.ticker import MaxNLocator
from tqdm.auto import tqdm

from pism_terra.download import download_usgs_benchmark
from pism_terra.glacier.usgs import (
    BALANCE_VARS,
    DEFAULT_DATA_DIR,
    MODEL_COLOR,
    MODEL_SEASON_STYLE,
    OBS_STYLE,
    RHO_WATER,
    SEASON_DATE_COLUMNS,
    SKILL_COLUMNS,
    SPECIFIC_LABEL,
    SPECIFIC_UNITS,
    datetime_encoding,
    ensemble_line,
    find_model_files,
    format_skill,
    grid_spacing,
    integrate_rate,
    interval_edges,
    is_monthly,
    load_glacier_wide,
    load_rgi_glaciers,
    load_sites,
    map_files,
    match_rgi_ids,
    mean_years,
    open_pism,
    rgi_output_dir,
    run_label,
    score,
    uncertainty_var,
)
from pism_terra.kitp.analyze import REGION_DIM, rc_params, with_region_labels
from pism_terra.log import setup_logging

logger = logging.getLogger("pism_terra.glacier.usgs_benchmark_glaciers")

MODEL_VAR = "tendency_of_ice_mass"
#: Model counterpart of a glaciological seasonal balance. A stake network
#: measures accumulation minus melt at the surface; within a season flow and
#: discharge move mass around without a stake seeing it, so the surface flux is
#: the like-for-like variable — not the total ``MODEL_VAR`` tendency.
MODEL_SEASONAL_VAR = "tendency_of_ice_mass_due_to_surface_mass_flux"


def to_mass_rate(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert glacier-wide specific balances to mass-change rates.

    A balance of 1 m w.e. over the glacier is a mass change of
    ``1 m x rho_water x area``; with the area in km² that is ``1e-3`` Gt per
    km².

    Parameters
    ----------
    ds : xarray.Dataset
        Output of :func:`load_glacier_wide`.

    Returns
    -------
    xarray.Dataset
        The balance (and uncertainty) variables in Gt/yr, plus ``area``.
    """
    quantified = ds.pint.quantify()
    rho = xr.DataArray(RHO_WATER).pint.quantify("kg m^-3")
    out: dict[str, xr.DataArray] = {}
    for var in list(ds.data_vars):
        if var == "area" or not np.issubdtype(ds[var].dtype, np.number):
            continue
        rate = (quantified[var] * rho * quantified["area"]).pint.to("Gt/year")
        rate.attrs["long_name"] = ds[var].attrs.get("long_name", var).replace("(m w.e.)", "").strip()
        out[var] = rate
    result = xr.Dataset(out).pint.dequantify()
    for var in result.data_vars:
        result[var].attrs["units"] = "Gt year^-1"
    result["area"] = ds["area"]
    for column in SEASON_DATE_COLUMNS:
        if column in ds:
            result[column] = ds[column]
    result.attrs.update(ds.attrs)
    return result


def ice_area(ds: xr.Dataset, rgi_id: str, spacing: tuple[float, float] | None) -> xr.DataArray | None:
    """
    Ice-covered area of a glacier through time, from a per-glacier scalar file.

    The post-processing sums ``sftgif`` (the ice-covered fraction of each
    cell) over the outline, so it is a cell count; times the cell area it is
    the glacierized area. Without it the outline area the post-processing
    stores is used, which does not change in time.

    Parameters
    ----------
    ds : xarray.Dataset
        Scalar dataset with region labels restored.
    rgi_id : str
        RGI ``-G-`` identifier to select.
    spacing : tuple of float or None
        ``(dx, dy)`` in metres from :func:`pism_terra.glacier.usgs.grid_spacing`.

    Returns
    -------
    xarray.DataArray or None
        Area in m² on ``time``; None when the file has neither ``sftgif``
        nor ``area``.
    """
    if "sftgif" in ds and spacing is not None:
        dx, dy = spacing
        area = ds["sftgif"].sel({REGION_DIM: rgi_id}).load() * dx * dy
        area.attrs = {"units": "m^2", "long_name": "ice-covered area", "source": "sftgif * dx * dy"}
    elif "area" in ds:
        static = ds["area"].sel({REGION_DIM: rgi_id}).load()
        area = xr.full_like(ds["time"], float(static), dtype=float)
        area.attrs = {"units": "m^2", "long_name": "outline area", "source": "outline"}
    else:
        return None
    return area.drop_vars(REGION_DIM, errors="ignore")


def specific_balance(values: xr.DataArray, area: xr.DataArray) -> xr.DataArray:
    """
    Turn a mass-change rate into a specific balance in metres water equivalent.

    Parameters
    ----------
    values : xarray.DataArray
        Rates in Gt/yr (``units`` attribute required).
    area : xarray.DataArray
        Glacier area in m² (``units`` attribute required), broadcastable
        against *values*.

    Returns
    -------
    xarray.DataArray
        ``values / (area * rho_water)`` in ``m year^-1``.
    """
    rho = xr.DataArray(RHO_WATER).pint.quantify("kg m^-3")
    out = (values.pint.quantify() / (area.pint.quantify() * rho)).pint.to("m/year").pint.dequantify()
    # the area used is reported separately; as a coordinate it would clash with
    # the observed area when the series are merged into one dataset
    out = out.drop_vars("area", errors="ignore")
    out.attrs = {"units": SPECIFIC_UNITS, "long_name": f"{values.attrs.get('long_name', values.name)} (m w.e.)"}
    return out


def to_specific_balances(
    model: xr.DataArray | None, seasons: xr.Dataset | None, obs: xr.Dataset
) -> tuple[xr.DataArray | None, xr.Dataset | None]:
    """
    Express the modelled series per unit area, like the observations.

    Each series is divided by the model's own ice area for that year (the
    ``area`` coordinate/variable the loaders attach). Runs whose files
    carry no area fall back to the observed area of the year.

    Parameters
    ----------
    model : xarray.DataArray or None
        Annual total rates in Gt/yr from :func:`load_model_series`.
    seasons : xarray.Dataset or None
        Seasonal balances in Gt from :func:`model_seasonal_balances`.
    obs : xarray.Dataset
        Observations from :func:`load_glacier_wide` (``area`` in km²).

    Returns
    -------
    tuple
        ``(model, seasons)`` in ``m year^-1``, None where the input was None.
    """
    obs_area = (obs["area"].pint.quantify().pint.to("m^2").pint.dequantify()).astype(float)
    obs_area.attrs["units"] = "m^2"

    def _area_for(values: xr.DataArray, area: xr.DataArray | None) -> xr.DataArray:
        """
        Area to divide by, filled from the observations where the model has none.

        Parameters
        ----------
        values : xarray.DataArray
            Series being converted (for its ``time`` axis and name).
        area : xarray.DataArray or None
            Model ice area in m², possibly with NaN years.

        Returns
        -------
        xarray.DataArray
            Area in m² on the series' years.
        """
        fallback = obs_area.reindex(time=values["time"])
        if area is None:
            logger.warning("%s: model files carry no ice area; using the observed area", values.name)
            area = fallback
        else:
            area = area.where(np.isfinite(area), fallback).astype(float)
        area.attrs["units"] = "m^2"
        return area

    model_mwe = None
    if model is not None:
        model_mwe = specific_balance(model, _area_for(model, model.coords.get("area")))
        model_mwe.name = model.name
    seasons_mwe = None
    if seasons is not None:
        area = _area_for(seasons["Bw"], seasons["area"] if "area" in seasons else None)
        seasons_mwe = xr.Dataset({var: specific_balance(seasons[var], area) for var in BALANCE_VARS if var in seasons})
    return model_mwe, seasons_mwe


def _season_spec(obs: xr.Dataset) -> tuple[np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex] | None:
    """
    Balance years and season dates of a glacier's observations.

    Parameters
    ----------
    obs : xarray.Dataset
        Observations from :func:`load_glacier_wide` (or their mass rates,
        which carry the same date columns).

    Returns
    -------
    tuple or None
        ``(years, bw_dates, ba_dates)``; None when the release dates no
        season for the glacier.
    """
    if not all(column in obs for column in SEASON_DATE_COLUMNS):
        return None
    years = obs["time"].values.astype(int)
    return years, pd.to_datetime(obs["Bw_Date"].values), pd.to_datetime(obs["Ba_Date"].values)


def _annual_from_ds(ds: xr.Dataset, rgi_id: str, spacing: tuple[float, float] | None) -> xr.DataArray:
    """
    Annual-mean mass-change rate of one glacier from an open scalar dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Scalar dataset with region labels restored; must carry
        :data:`MODEL_VAR` and the glacier.
    rgi_id : str
        RGI ``-G-`` identifier to select.
    spacing : tuple of float or None
        ``(dx, dy)`` in metres from :func:`pism_terra.glacier.usgs.grid_spacing`.

    Returns
    -------
    xarray.DataArray
        Annual-mean rate in Gt/yr on integer years, carrying the year's mean
        ice area (m²) as the ``area`` coordinate (NaN when the file has none).
    """
    da = ds[MODEL_VAR].sel({REGION_DIM: rgi_id}).pint.quantify().pint.to("Gt/year").pint.dequantify().load()
    years = mean_years(ds)
    area = ice_area(ds, rgi_id, spacing)
    if area is None:
        area = xr.full_like(ds["time"], np.nan, dtype=float)
    da = da.assign_coords(year=("time", years)).swap_dims({"time": "year"}).drop_vars("time")
    da = da.groupby("year").mean("year").rename({"year": "time"})
    area = area.assign_coords(year=("time", years)).swap_dims({"time": "year"}).drop_vars("time")
    area = area.groupby("year").mean("year").rename({"year": "time"})
    return da.assign_coords(area=("time", area.values)).drop_vars(REGION_DIM, errors="ignore")


def _seasonal_from_ds(
    ds: xr.Dataset,
    rgi_id: str,
    spacing: tuple[float, float] | None,
    spec: tuple[np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex],
    variable: str,
) -> xr.Dataset | None:
    """
    Seasonal surface-flux balances of one glacier from an open monthly dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Monthly scalar dataset with region labels restored; must carry
        *variable* and the glacier.
    rgi_id : str
        RGI ``-G-`` identifier to select.
    spacing : tuple of float or None
        ``(dx, dy)`` in metres from :func:`pism_terra.glacier.usgs.grid_spacing`.
    spec : tuple
        ``(years, bw_dates, ba_dates)`` from :func:`_season_spec`.
    variable : str
        Model variable to integrate.

    Returns
    -------
    xarray.Dataset or None
        ``Bw``/``Bs`` in Gt and the ice area on the kept balance years; None
        when the record fully spans no measured season.
    """
    years, bw_dates, ba_dates = spec
    area = ice_area(ds, rgi_id, spacing)
    area_by_year = (
        None
        if area is None
        else area.assign_coords(year=("time", mean_years(ds))).swap_dims({"time": "year"}).groupby("year").mean("year")
    )
    rate = (
        ds[variable]
        .sel({REGION_DIM: rgi_id})
        .pint.quantify()
        .pint.to("Gt/year")
        .pint.dequantify()
        .load()
        .drop_vars(REGION_DIM, errors="ignore")
    )
    starts, ends = interval_edges(ds)
    rate = rate.assign_coords(month_start=("time", starts), month_end=("time", ends))

    # Winter starts at the previous balance year's minimum, so a year needs its
    # own two dates and its predecessor's ``Ba_Date``.
    previous = {int(y): ba for y, ba in zip(years, ba_dates)}
    winter, summer, areas, kept = [], [], [], []
    for year, bw_date, ba_date in zip(years, bw_dates, ba_dates):
        year_start = previous.get(int(year) - 1)
        if pd.isna(bw_date) or pd.isna(ba_date) or year_start is None or pd.isna(year_start):
            continue
        b_w = integrate_rate(rate, year_start, bw_date)
        b_s = integrate_rate(rate, bw_date, ba_date)
        if np.isnan(b_w) and np.isnan(b_s):
            continue
        winter.append(b_w)
        summer.append(b_s)
        # the year's mean ice area, for the conversion to metres water equivalent
        areas.append(
            float(area_by_year.sel(year=int(year)))
            if area_by_year is not None and int(year) in area_by_year["year"].values
            else np.nan
        )
        kept.append(int(year))

    if not kept:
        return None
    return xr.Dataset(
        {
            "Bw": ("time", np.array(winter)),
            "Bs": ("time", np.array(summer)),
            "area": ("time", np.array(areas), {"units": "m^2", "long_name": "ice-covered area"}),
        },
        coords={"time": np.array(kept)},
    )


def _extract_one_file(
    file: Path,
    rgi_ids: Sequence[str],
    seasonal_specs: dict[str, tuple[np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]],
    variable: str,
) -> tuple[dict[str, xr.DataArray], dict[str, xr.Dataset]]:
    """
    Extract every requested glacier's series from one scalar file.

    Opens the file once and pulls the annual mass-change rate for every id in
    *rgi_ids* and the seasonal surface-flux balances for every id in
    *seasonal_specs*, so a run directory is read once per file rather than
    twice per glacier. Runs in worker processes via
    :func:`pism_terra.glacier.usgs.map_files`.

    Parameters
    ----------
    file : pathlib.Path
        Post-processed per-glacier scalar file.
    rgi_ids : sequence of str
        RGI ``-G-`` identifiers whose annual series are wanted.
    seasonal_specs : dict
        Season dates per RGI id from :func:`_season_spec`.
    variable : str
        Model variable the seasonal balances integrate.

    Returns
    -------
    tuple of dict
        ``(annual, seasonal)``, each keyed by RGI id; ids the file does not
        cover are absent.
    """
    annual: dict[str, xr.DataArray] = {}
    seasonal: dict[str, xr.Dataset] = {}
    with open_pism(file) as ds:
        spacing = grid_spacing(ds)
        ds = with_region_labels(ds.drop_vars("pism_config", errors="ignore"))
        names = ds[REGION_DIM].values
        if names.dtype.kind == "S":
            ds = ds.assign_coords({REGION_DIM: names.astype(str)})
        present = set(ds[REGION_DIM].values.tolist())
        if MODEL_VAR in ds:
            for rgi_id in rgi_ids:
                if rgi_id in present:
                    annual[rgi_id] = _annual_from_ds(ds, rgi_id, spacing)
        monthly = is_monthly(ds["time"].values)
        if seasonal_specs and variable in ds and not monthly:
            logger.info("%s is not monthly; no modelled seasons from it", file.name)
        if monthly and variable in ds:
            for rgi_id, spec in seasonal_specs.items():
                if rgi_id not in present:
                    continue
                one = _seasonal_from_ds(ds, rgi_id, spacing, spec, variable)
                if one is not None:
                    seasonal[rgi_id] = one
    return annual, seasonal


def load_model_data(
    files: Sequence[Path | str],
    rgi_ids: Sequence[str],
    seasonal_specs: dict[str, tuple[np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]],
    root: Path | str | None = None,
    variable: str = MODEL_SEASONAL_VAR,
    n_jobs: int = 1,
) -> tuple[dict[str, xr.DataArray], dict[str, xr.Dataset]]:
    """
    Collect every glacier's annual and seasonal series in one sweep over the files.

    Each file is opened exactly once (in parallel for ``n_jobs > 1``) and all
    glaciers are extracted from it together; per glacier the runs keep the
    order of *files*, so the result matches per-glacier scans exactly.

    Parameters
    ----------
    files : sequence of Path or str
        Post-processed per-glacier scalar files.
    rgi_ids : sequence of str
        RGI ``-G-`` identifiers whose annual ``tendency_of_ice_mass`` is
        wanted.
    seasonal_specs : dict
        Season dates per RGI id (see :func:`_season_spec`) for the glaciers
        whose seasonal balances are wanted.
    root : Path or str or None, optional
        Run directory the ``run`` labels are made relative to.
    variable : str, optional
        Model variable the seasonal balances integrate. Defaults to
        :data:`MODEL_SEASONAL_VAR`.
    n_jobs : int, optional
        Worker processes used to read the files (see
        :func:`pism_terra.glacier.usgs.map_files`).

    Returns
    -------
    tuple of dict
        ``(annual, seasonal)`` keyed by RGI id: annual rates in Gt/yr on
        ``(run, time)`` (see :func:`load_model_series`) and seasonal balances
        in Gt (see :func:`model_seasonal_balances`). Glaciers no file covers
        are absent.
    """
    results = map_files(
        _extract_one_file, files, list(rgi_ids), seasonal_specs, variable, n_jobs=n_jobs, desc="Reading model files"
    )
    annual_series: dict[str, list[xr.DataArray]] = {}
    annual_labels: dict[str, list[str]] = {}
    seasonal_series: dict[str, list[xr.Dataset]] = {}
    seasonal_labels: dict[str, list[str]] = {}
    for file, (file_annual, file_seasonal) in zip(files, results):
        label = run_label(Path(file), root)
        for rgi_id, da in file_annual.items():
            annual_series.setdefault(rgi_id, []).append(da)
            annual_labels.setdefault(rgi_id, []).append(label)
            logger.info("%s found in %s", rgi_id, label)
        for rgi_id, one in file_seasonal.items():
            seasonal_series.setdefault(rgi_id, []).append(one)
            seasonal_labels.setdefault(rgi_id, []).append(label)
            logger.info("%s: %d modelled seasons from %s", rgi_id, one.sizes["time"], label)

    annual: dict[str, xr.DataArray] = {}
    for rgi_id, series in annual_series.items():
        model = xr.concat(series, dim=pd.Index(annual_labels[rgi_id], name="run"), join="outer").sortby("time")
        model.attrs = {"units": "Gt year^-1", "long_name": "rate of change of the ice mass"}
        model.name = MODEL_VAR
        annual[rgi_id] = model
    seasonal: dict[str, xr.Dataset] = {}
    for rgi_id, per_run in seasonal_series.items():
        out = xr.concat(per_run, dim=pd.Index(seasonal_labels[rgi_id], name="run"), join="outer").sortby("time")
        out["Ba"] = out["Bw"] + out["Bs"]
        for var, season in BALANCE_VARS.items():
            out[var].attrs = {
                "units": "Gt year^-1",
                "long_name": f"modelled {season} surface mass balance",
                "source_variable": variable,
            }
        seasonal[rgi_id] = out
    return annual, seasonal


def load_model_series(
    files: Sequence[Path | str], rgi_id: str, root: Path | str | None = None, n_jobs: int = 1
) -> xr.DataArray | None:
    """
    Collect a glacier's annual ``tendency_of_ice_mass`` from every file that has it.

    A convenience wrapper around :func:`load_model_data` for one glacier; the
    pipeline extracts all glaciers in a single sweep instead.

    Parameters
    ----------
    files : sequence of Path or str
        Post-processed per-glacier scalar files.
    rgi_id : str
        RGI ``-G-`` identifier to select.
    root : Path or str or None, optional
        Run directory the ``run`` labels are made relative to.
    n_jobs : int, optional
        Worker processes used to read the files (see
        :func:`pism_terra.glacier.usgs.map_files`).

    Returns
    -------
    xarray.DataArray or None
        Annual-mean rate in Gt/yr on ``(run, time)`` with integer years,
        carrying the year's mean ice area (m²) as the ``area`` coordinate
        (NaN for files without one); None when no file contains the glacier.
    """
    annual, _ = load_model_data(files, [rgi_id], {}, root=root, n_jobs=n_jobs)
    return annual.get(rgi_id)


def model_seasonal_balances(
    files: Sequence[Path | str],
    rgi_id: str,
    obs: xr.Dataset,
    root: Path | str | None = None,
    variable: str = MODEL_SEASONAL_VAR,
    n_jobs: int = 1,
) -> xr.Dataset | None:
    """
    Integrate the modelled surface mass flux over each measured season.

    A balance year runs from one annual minimum to the next and is split at the
    winter maximum, both dated in the release. The winter balance is therefore
    the mass change from the *previous* year's ``Ba_Date`` to this year's
    ``Bw_Date``, and the summer balance runs from there to this year's
    ``Ba_Date``. Years where the release gives no dates — or that the model
    record does not fully span — are left out, so every value plotted covers
    exactly the interval its observation does.

    Requires monthly model output; annual files carry no seasonal information
    and yield ``None``. A convenience wrapper around :func:`load_model_data`
    for one glacier; the pipeline extracts all glaciers in a single sweep
    instead.

    Parameters
    ----------
    files : sequence of Path or str
        Post-processed per-glacier scalar files.
    rgi_id : str
        RGI ``-G-`` identifier to select.
    obs : xarray.Dataset
        Observations carrying ``Bw_Date``/``Ba_Date`` (see
        :func:`load_glacier_wide`).
    root : Path or str or None, optional
        Run directory the ``run`` labels are made relative to.
    variable : str, optional
        Model variable to integrate. Defaults to
        :data:`MODEL_SEASONAL_VAR`.
    n_jobs : int, optional
        Worker processes used to read the files (see
        :func:`pism_terra.glacier.usgs.map_files`).

    Returns
    -------
    xarray.Dataset or None
        ``Bw``/``Bs`` in Gt on ``(run, time)`` over balance years, plus the
        derived annual ``Ba``. None when no file has monthly output for the
        glacier, or the release dates no season.
    """
    spec = _season_spec(obs)
    if spec is None:
        logger.info("%s: release gives no season dates; no modelled seasons", rgi_id)
        return None
    _, seasonal = load_model_data(files, [], {rgi_id: spec}, root=root, variable=variable, n_jobs=n_jobs)
    return seasonal.get(rgi_id)


def skill_scores(obs: xr.Dataset, model: xr.DataArray | None = None, seasons: xr.Dataset | None = None) -> pd.DataFrame:
    """
    Score the modelled balances against the observations.

    Every comparison uses the series the plot draws — the single run, or the
    ensemble median — over the years both sides have. The seasonal and annual
    surface balances are the natural counterparts of the stake-derived
    observations; the total ``tendency_of_ice_mass`` is scored against the
    annual balance as well, labelled ``total``, since it is what the annual
    model line shows.

    Parameters
    ----------
    obs : xarray.Dataset
        Balances in Gt/yr from :func:`to_mass_rate`.
    model : xarray.DataArray or None, optional
        Annual total mass-change rates from :func:`load_model_series`.
    seasons : xarray.Dataset or None, optional
        Modelled seasonal balances from :func:`model_seasonal_balances`.

    Returns
    -------
    pandas.DataFrame
        One row per available comparison with :data:`SKILL_COLUMNS`;
        empty when there is nothing to compare.
    """
    rows: list[dict[str, Any]] = []
    if seasons is not None:
        for var, season in BALANCE_VARS.items():
            if var in obs and var in seasons:
                rows.append(
                    {
                        "variable": var,
                        "season": season,
                        "source": "surface",
                        **score(obs[var], ensemble_line(seasons[var])),
                    }
                )
    if model is not None and "Ba" in obs:
        rows.append({"variable": "Ba", "season": "annual", "source": "total", **score(obs["Ba"], ensemble_line(model))})
    return pd.DataFrame(rows, columns=SKILL_COLUMNS)


def plot_glacier(
    obs: xr.Dataset,
    model: xr.DataArray | None,
    glacier: str,
    rgi_id: str,
    output_dir: Path | str,
    *,
    seasons: xr.Dataset | None = None,
    skill: pd.DataFrame | None = None,
    area_km2: float | None = None,
) -> Path:
    """
    Plot observed seasonal and annual balances with the modelled mass-change rate.

    Everything is drawn as a specific balance in metres water equivalent per
    year. A second axis on the right reads the same curves in Gt/yr at a
    fixed area — the mean observed area — since one scale factor cannot
    follow the year-to-year area changes that went into the conversion.

    Parameters
    ----------
    obs : xarray.Dataset
        Balances in m w.e. per year from :func:`load_glacier_wide`.
    model : xarray.DataArray or None
        Modelled specific balances on ``(run, time)`` from
        :func:`to_specific_balances`. Several runs are drawn as their median
        and 5-95 % range.
    glacier : str
        Glacier name for the title and file name.
    rgi_id : str
        RGI identifier for the title and file name.
    output_dir : Path or str
        Directory the figure is written to.
    seasons : xarray.Dataset or None, optional
        Modelled seasonal balances from :func:`model_seasonal_balances`,
        drawn as lines in their observation's colour. ``None`` (the case for
        annual model output) leaves the plot as it was.
    skill : pandas.DataFrame or None, optional
        Scores from :func:`skill_scores`, written into the top-left corner.
    area_km2 : float or None, optional
        Glacier area the right-hand Gt/yr axis is scaled with. Defaults to
        the mean of ``obs["area"]``.

    Returns
    -------
    Path
        The PNG written (a PDF sits next to it).
    """
    output_dir = Path(output_dir)
    if area_km2 is None and "area" in obs:
        area_km2 = float(obs["area"].mean())
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"usgs_benchmark_{glacier}_{rgi_id}"

    with mpl.rc_context(rc=rc_params):
        fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.6), layout="constrained")
        for var, season in BALANCE_VARS.items():
            if var not in obs:
                continue
            style = OBS_STYLE[var]
            unc = uncertainty_var(obs, var)
            kwargs: dict[str, Any] = {
                "color": style["color"],
                "marker": style["marker"],
                "ms": 2,
                "lw": 0.75,
                "label": f"Observed {season}",
            }
            if unc is not None:
                ax.errorbar(obs["time"], obs[var], yerr=obs[unc], capsize=1, elinewidth=0.5, **kwargs)
            else:
                ax.plot(obs["time"], obs[var], **kwargs)

        if seasons is not None:
            for var, style in MODEL_SEASON_STYLE.items():
                if var not in seasons:
                    continue
                season = BALANCE_VARS[var]
                values = seasons[var]
                if values.sizes["run"] == 1:
                    ax.plot(
                        values["time"],
                        values.isel(run=0),
                        color=style["color"],
                        ls=style["ls"],
                        lw=1.0,
                        label=f"PISM {season} (surface)",
                    )
                else:
                    ax.fill_between(
                        values["time"],
                        values.quantile(0.05, dim="run"),
                        values.quantile(0.95, dim="run"),
                        color=style["color"],
                        alpha=0.2,
                        lw=0,
                    )
                    ax.plot(
                        values["time"],
                        values.median(dim="run"),
                        color=style["color"],
                        ls=style["ls"],
                        lw=1.0,
                        label=f"PISM {season} (surface, n={values.sizes['run']})",
                    )

        if model is not None:
            if model.sizes["run"] == 1:
                ax.plot(
                    model["time"],
                    model.isel(run=0),
                    color=MODEL_COLOR,
                    lw=1.0,
                    label=f"PISM ({model['run'].values[0]})",
                )
            else:
                low = model.quantile(0.05, dim="run")
                high = model.quantile(0.95, dim="run")
                ax.fill_between(model["time"], low, high, color=MODEL_COLOR, alpha=0.25, lw=0)
                ax.plot(
                    model["time"],
                    model.median(dim="run"),
                    color=MODEL_COLOR,
                    lw=1.0,
                    label=f"PISM median, 5-95% (n={model.sizes['run']})",
                )

        if skill is not None and (text := format_skill(skill, units=SPECIFIC_LABEL)):
            ax.text(
                0.02,
                0.98,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5,
                family="monospace",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )

        ax.axhline(y=0, color="k", ls="dotted", lw=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Balance year")
        ax.set_ylabel("Mass balance (m w.e. yr$^{-1}$)")
        if area_km2 is not None and np.isfinite(area_km2) and area_km2 > 0:
            # 1 m w.e. over 1 km^2 is 1e-3 Gt
            factor = area_km2 * 1e-3
            secondary = ax.secondary_yaxis(
                "right", functions=(lambda m: np.asarray(m) * factor, lambda g: np.asarray(g) / factor)
            )
            secondary.set_ylabel(f"Mass balance (Gt yr$^{{-1}}$ at {area_km2:.3g} km$^2$)")
        ax.set_title(f"{glacier} ({rgi_id})")
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc="outside lower center", ncol=2)
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)
        png = output_dir / f"{stem}.png"
        fig.savefig(png, dpi=300)
        fig.savefig(output_dir / f"{stem}.pdf")
        plt.close(fig)
    return png


def run_pipeline(
    run_dir: Path | str | None,
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    output_dir: Path | str = ".",
    rgi_file: Path | str | None = None,
    uncertainty: float | None = None,
    n_jobs: int = 1,
    force_overwrite: bool = False,
) -> pd.DataFrame:
    """
    Download, match, convert and plot every glacier in the release.

    Parameters
    ----------
    run_dir : Path or str or None
        Directory searched recursively for ``scalar_G_*.nc``; None plots
        the observations alone.
    data_dir : Path or str, optional
        Cache for the ScienceBase archives and RGI outlines.
    output_dir : Path or str, optional
        Where the match and skill tables go; each glacier's figures and
        NetCDF file are written to a ``<rgi_id>`` sub-directory of it.
    rgi_file : Path or str or None, optional
        Outline file to match against instead of the downloaded regions.
    uncertainty : float or None, optional
        Constant one-sigma uncertainty in m w.e. applied to every balance
        when the CSV carries none.
    n_jobs : int, optional
        Worker processes used to read the model files (see
        :func:`pism_terra.glacier.usgs.map_files`).
    force_overwrite : bool, default False
        Re-download the archives.

    Returns
    -------
    pandas.DataFrame
        The match table with ``figure`` and ``n_runs`` columns added.
    """
    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = download_usgs_benchmark(data_dir, force_overwrite=force_overwrite)
    sites = load_sites(paths["sites"])
    rgi = load_rgi_glaciers(data_dir, rgi_file=rgi_file, force_overwrite=force_overwrite)
    matches = match_rgi_ids(sites, rgi)

    model_files = find_model_files(run_dir) if run_dir is not None else []
    logger.info("%d scalar_G files under %s", len(model_files), run_dir)

    # Pass 1: the observations for every matched glacier, so a single sweep
    # over the model files can extract every glacier at once.
    observations: dict[str, xr.Dataset] = {}
    rates: dict[str, xr.Dataset] = {}
    seasonal_specs: dict[str, tuple[np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]] = {}
    rgi_ids: list[str] = []
    for row in matches.itertuples(index=False):
        if pd.isna(row.rgi_id):
            continue
        obs = load_glacier_wide(paths["data"], row.glacier, fallback_area_km2=row.area_km2)
        if obs is None:
            logger.info("%s: point data only, no glacier-wide balances; skipping", row.glacier)
            continue
        if uncertainty is not None:
            for var in BALANCE_VARS:
                if var in obs and uncertainty_var(obs, var) is None:
                    obs[f"{var}_unc"] = xr.full_like(obs[var], uncertainty)
                    obs[f"{var}_unc"].attrs = {
                        "units": "m year^-1",
                        "long_name": f"assumed uncertainty of {var} (m w.e.)",
                    }
        observations[row.glacier] = obs
        rates[row.glacier] = to_mass_rate(obs)
        rgi_ids.append(row.rgi_id)
        spec = _season_spec(obs)
        if spec is None:
            logger.info("%s: release gives no season dates; no modelled seasons", row.rgi_id)
        else:
            seasonal_specs[row.rgi_id] = spec

    # One sweep: every scalar file is opened once, yielding every glacier's
    # annual series and seasonal balances together.
    annual_by_id: dict[str, xr.DataArray] = {}
    seasonal_by_id: dict[str, xr.Dataset] = {}
    if model_files:
        annual_by_id, seasonal_by_id = load_model_data(
            model_files, rgi_ids, seasonal_specs, root=run_dir, n_jobs=n_jobs
        )

    # Pass 2: score, plot and write each glacier from the collected series.
    figures: list[str | None] = []
    n_runs: list[int] = []
    skills: list[pd.DataFrame] = []
    for row in tqdm(matches.itertuples(index=False), total=len(matches), desc="Benchmark glaciers", unit="glacier"):
        if pd.isna(row.rgi_id) or row.glacier not in observations:
            figures.append(None)
            n_runs.append(0)
            continue
        obs = observations[row.glacier]
        rate = rates[row.glacier]
        model = annual_by_id.get(row.rgi_id)
        seasons = seasonal_by_id.get(row.rgi_id)
        model_mwe, seasons_mwe = to_specific_balances(model, seasons, obs)
        skill = skill_scores(obs, model_mwe, seasons_mwe)
        if not skill.empty:
            skills.append(skill.assign(glacier=row.glacier, rgi_id=row.rgi_id, units=SPECIFIC_UNITS))
            for line in format_skill(skill, units=SPECIFIC_LABEL).splitlines():
                logger.info("%s: %s", row.glacier, line)
        glacier_dir = rgi_output_dir(output_dir, row.rgi_id)
        png = plot_glacier(obs, model_mwe, row.glacier, row.rgi_id, glacier_dir, seasons=seasons_mwe, skill=skill)
        figures.append(str(png))
        n_runs.append(0 if model is None else int(model.sizes["run"]))

        mwe = [v for v in obs.data_vars if v != "area" and np.issubdtype(obs[v].dtype, np.number)]
        out = xr.merge([rate, obs[mwe].rename({v: f"{v}_mwe" for v in mwe})])
        if model is not None:
            out[MODEL_VAR] = model.drop_vars("area", errors="ignore")
            out[f"{MODEL_VAR}_mwe"] = model_mwe
            if "area" in model.coords:
                area = model.coords["area"]
                out["model_area"] = xr.DataArray(
                    area.values,
                    dims=area.dims,
                    coords={dim: model.coords[dim] for dim in area.dims},
                    attrs={"units": "m^2", "long_name": "modelled ice-covered area"},
                )
        if seasons is not None and seasons_mwe is not None:
            for var in BALANCE_VARS:
                if var in seasons:
                    out[f"{var}_model"] = seasons[var]
                    out[f"{var}_model_mwe"] = seasons_mwe[var]
        for entry in skill.itertuples(index=False):
            name = f"{MODEL_VAR}_mwe" if entry.source == "total" else f"{entry.variable}_model_mwe"
            if name in out:
                out[name].attrs.update(
                    {
                        "pearson_r": entry.r,
                        "mae": entry.mae,
                        "bias": entry.bias,
                        "n_years": entry.n,
                        "skill_units": SPECIFIC_UNITS,
                    }
                )
        out.attrs["rgi_id"] = row.rgi_id
        out.to_netcdf(glacier_dir / f"usgs_benchmark_{row.glacier}_{row.rgi_id}.nc", encoding=datetime_encoding(out))

    matches = matches.assign(figure=figures, n_runs=n_runs)
    matches.to_csv(output_dir / "usgs_benchmark_rgi_match.csv", index=False)
    columns = ["glacier", "rgi_id", *SKILL_COLUMNS, "units"]
    skill_table = pd.concat(skills, ignore_index=True) if skills else pd.DataFrame(columns=columns)
    skill_table = skill_table[columns]
    skill_table.to_csv(output_dir / "usgs_benchmark_skill.csv", index=False)
    return matches


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Compare USGS benchmark-glacier mass balances with PISM output.

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
    parser.description = "Plot USGS glacier-wide mass balances (Gt/yr) against PISM tendency_of_ice_mass."
    parser.add_argument(
        "RUN_DIR",
        nargs="?",
        default=None,
        help="Directory searched recursively for scalar_G_*.nc files. Omit to plot the observations alone.",
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
        "--uncertainty",
        type=float,
        default=None,
        help="Constant one-sigma uncertainty (m w.e.) to draw as error bars when the CSVs carry none.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Worker processes for reading the model files; 1 runs serially.",
    )
    parser.add_argument("--force-overwrite", action="store_true", default=False, help="Re-download the archives.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "usgs_benchmark.log")

    matches = run_pipeline(
        args.RUN_DIR,
        data_dir=args.data_dir,
        output_dir=output_dir,
        rgi_file=args.rgi_glacier_file,
        uncertainty=args.uncertainty,
        n_jobs=args.n_jobs,
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
