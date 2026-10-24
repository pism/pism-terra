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
Importance-sample an ISMIP7 Greenland ensemble against the observed mass fluxes.

The time-series counterpart of ``pism-ismip7-greenland-importance-sampling-dh``,
which scores the members on a field. Here the members' per-basin flux series
-- the ``region_*.nc`` files ``pism-ismip7-postprocess-flux`` writes, one per
UQ draw -- are weighed against the Mankoff et al. (2021) input-output mass
balance that ``pism-ismip7-greenland-observations`` stages beside a run's
output. For every region both sides report:

* a Gaussian likelihood weight per member from one flux series (the total
  mass balance by default; the surface mass balance or the grounding-line
  flux instead with ``--variable``), summed over the instants of a sampling
  window, for several fudge factors on the observed uncertainty, resolved
  into resampling counts and an effective sample size
  (:func:`pism_terra.calibration.importance_weights`);
* per-member RMSE, MAE and bias over the same window;
* a flux figure per fudge factor: the observed mass balance, surface mass
  balance and grounding-line flux with their uncertainty, the prior ensemble
  band and the posterior band the weights give;
* a posterior histogram of the UQ parameters (the run's ``uq.csv``) per
  fudge factor.

The basins are shared parameter draws, so their log-likelihoods are also
summed into one joint posterior -- over the basins only, since the
ice-sheet total is their sum and would count everything twice.

**Alignment.** The two records are on different clocks: PISM reports one
value per month, stamped at the middle of the month, while Mankoff is
daily. Both are averaged onto the same calendar bins (``--freq``: months or
years, labelled at the start or the end), the bins a record only partly
covers are dropped -- the final, half-reported year of the observations, a
month a run starts in the middle of -- and what is left is joined on the
instants both have. The likelihood then runs over the bins inside
``--start``/``--end``; the flux figures show the whole shared record with
the window marked.

**Sign conventions.** Both sides count mass loss negative for the mass
balance, the surface mass balance and the grounding-line flux (the staged
Mankoff product already flips its discharge). Basal melt is the exception:
Mankoff reports it as a positive loss, PISM as a negative flux, so it is
negated on the observed side before comparing.

Results go to ``<output-path>/<region>/`` per region, the joint posterior
to ``<output-path>/joint/`` and the summary table to the top level.
"""

from __future__ import annotations

import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cf_xarray.units  # pylint: disable=unused-import  # noqa: F401  (teaches pint UDUNITS)
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pint_xarray  # pylint: disable=unused-import  # noqa: F401  (registers the .pint accessor)
import xarray as xr
from cmcrameri import cm

from pism_terra.calibration import (
    importance_weights,
    joint_log_likelihood,
    plot_parameter_histograms,
    posterior_table,
    short_labels,
    weighted_quantiles,
    weights_from_log_likelihood,
)
from pism_terra.log import setup_logging
from pism_terra.plotting import rc_params
from pism_terra.processing import preprocess_netcdf
from pism_terra.progress import progress_bar

# Named after the module even under ``python -m``, where ``__name__`` is
# ``__main__`` and a logger of that name would sit outside the ``pism_terra``
# tree that :func:`pism_terra.log.setup_logging` writes to the log file.
logger = logging.getLogger(
    "pism_terra.ismip7.greenland.importance_sampling_flux" if __name__ == "__main__" else __name__
)

#: Dimension the ensemble members are stacked along: the UQ draw, which is
#: the key into the run's ``uq.csv``.
MEMBER_DIM = "uq_id"

#: The staged observations, relative to a run's output directory.
OBS_FILE = Path("output") / "observations" / "mankoff_greenland_mass_balance.nc"

#: The parameter table the run generator writes, relative to a run directory.
UQ_FILE = Path("output") / "uq.csv"

#: The observed region that is the ice-sheet total. It is the sum of the
#: basins, so it is left out of the joint posterior.
TOTAL_REGION = "GIS"

#: Resampling frequencies and the pandas period each bin is: months or
#: years, labelled at their start (``MS``, ``YS``) or their end (``ME``, ``YE``).
FREQUENCIES: dict[str, str] = {"MS": "M", "ME": "M", "YS": "Y", "YE": "Y"}

#: Inflations of the observed uncertainty, one filter each. Consecutive
#: annual values of a flux are not independent -- a warm decade thins the
#: margin for years -- so an uninflated likelihood is overconfident.
DEFAULT_FUDGE_FACTORS = (1.0, 3.0, 10.0)
DEFAULT_N_SAMPLES = 10_000
DEFAULT_FREQ = "YS"
DEFAULT_REDUCTION = "sum"

#: How the instants of the window are collapsed into one log-likelihood.
#: ``"sum"`` treats every bin as an independent observation; ``"mean"``
#: divides by their number, a strongly tempered posterior.
REDUCTIONS = ("sum", "mean")

#: Half-width of the observed band in the flux figures, in standard deviations.
OBS_BAND_SIGMA = 2.0

#: Quantiles of the prior and posterior bands in the flux figures.
BAND_QUANTILES = (0.05, 0.5, 0.95)

_BATLOW_S = getattr(cm, "batlowS")  # crameri registers its maps dynamically
PALETTE = [mpl.colors.to_hex(_BATLOW_S(i)) for i in (2, 3, 4, 5, 8, 9, 7, 12, 14, 11, 15, 13)]
PRIOR_COLOR = PALETTE[2]
POSTERIOR_COLOR = PALETTE[1]


@dataclass(frozen=True)
class FluxVariable:
    """
    One flux both sides report, and how they name it.

    Attributes
    ----------
    sim_vars : tuple of str
        Names the flux goes by in the model output, in order of preference.
    obs_var : str
        Name in the observations.
    obs_std_var : str
        Name of its uncertainty in the observations.
    sign : float
        Factor the observations are multiplied with so that both sides share
        a sign convention.
    label : str
        Axis label.
    """

    sim_vars: tuple[str, ...]
    obs_var: str
    obs_std_var: str
    sign: float
    label: str


#: The fluxes, keyed by the name they are compared under.
VARIABLES: dict[str, FluxVariable] = {
    "mass_balance": FluxVariable(
        ("tendency_of_ice_mass",), "mass_balance", "mass_balance_uncertainty", 1.0, "Mass balance"
    ),
    "surface_mass_balance": FluxVariable(
        ("tendency_of_ice_mass_due_to_surface_mass_flux",),
        "surface_mass_balance",
        "surface_mass_balance_uncertainty",
        1.0,
        "Surface mass balance",
    ),
    "grounding_line_flux": FluxVariable(
        ("grounding_line_flux", "ice_mass_transport_across_grounding_line"),
        "grounding_line_flux",
        "grounding_line_flux_uncertainty",
        1.0,
        "Grounding line flux",
    ),
    "basal_mass_balance": FluxVariable(
        ("tendency_of_ice_mass_due_to_basal_mass_flux",),
        "basal_mass_balance",
        "basal_mass_balance_uncertainty",
        -1.0,
        "Basal mass balance",
    ),
}

#: The fluxes the flux figure shows, top to bottom.
PLOT_VARIABLES = ("mass_balance", "surface_mass_balance", "grounding_line_flux")


def uncertainty_name(variable: str) -> str:
    """
    Name the uncertainty of a flux is compared under.

    Parameters
    ----------
    variable : str
        Key of :data:`VARIABLES`.

    Returns
    -------
    str
        ``<variable>_uncertainty``.
    """
    return f"{variable}_uncertainty"


def find_region_files(run_dir: Path | str, pattern: str = "region_*.nc") -> list[Path]:
    """
    Collect the per-member region files of an ensemble.

    Parameters
    ----------
    run_dir : Path or str
        Directory searched recursively.
    pattern : str, optional
        Glob the member files match.

    Returns
    -------
    list of pathlib.Path
        Sorted matches.

    Raises
    ------
    FileNotFoundError
        If nothing matches.
    """
    files = sorted(Path(run_dir).expanduser().rglob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} under {run_dir}")
    logger.info("Found %d member file(s) under %s", len(files), run_dir)
    return files


def load_uq_parameters(run_dir: Path | str, uq_file: Path | str | None = None) -> pd.DataFrame:
    """
    Read the parameter values of the ensemble members.

    Parameters
    ----------
    run_dir : Path or str
        Run directory; the run generator writes ``output/uq.csv`` into it.
    uq_file : Path or str or None, optional
        The table itself, when it lives elsewhere.

    Returns
    -------
    pandas.DataFrame
        Parameter columns indexed by member id (``uq_id``, as strings to
        match the ensemble coordinate).

    Raises
    ------
    FileNotFoundError
        If no ``uq.csv`` exists below the run directory.
    """
    run_dir = Path(run_dir).expanduser()
    candidates = (
        [Path(uq_file).expanduser()] if uq_file is not None else [run_dir / UQ_FILE] + sorted(run_dir.rglob("uq.csv"))
    )
    for csv in candidates:
        if csv.is_file():
            df = pd.read_csv(csv)
            df["uq"] = df["uq"].astype(str)
            logger.info("Read %d member(s) from %s", len(df), csv)
            return df.set_index("uq").rename_axis(MEMBER_DIM)
    raise FileNotFoundError(f"no uq.csv below {run_dir}")


def to_units(ds: xr.Dataset, targets: Mapping[str, str]) -> xr.Dataset:
    """
    Convert data variables with pint and hand back plain arrays.

    Only data variables are quantified, so the coordinates keep their plain
    indexes; a variable without a ``units`` attribute is left alone.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset whose variables carry ``units`` attributes.
    targets : mapping of str to str
        Target units per variable name.

    Returns
    -------
    xarray.Dataset
        ``ds`` with the requested variables converted, the target's spelling
        of the unit back in their attrs (pint would write its own, and
        ``Gt year^-1`` and ``gigametric_ton / year`` are the same unit).
    """
    skip = {str(c): None for c in ds.coords if "units" in ds[c].attrs}
    quantified = ds.pint.quantify(**skip)
    conversions = {v: u for v, u in targets.items() if v in quantified and quantified[v].pint.units is not None}
    out = quantified.pint.to(conversions).pint.dequantify()
    for name, units in conversions.items():
        out[name].attrs["units"] = units
    return out


def load_ensemble(files: Sequence[Path]) -> xr.Dataset:
    """
    Open the members' region files as one ensemble of flux series.

    Parameters
    ----------
    files : sequence of pathlib.Path
        Per-member ``region_*.nc`` files; the UQ draw is read from the name.

    Returns
    -------
    xarray.Dataset
        The fluxes of :data:`VARIABLES` that the files carry, under their
        compared names, with dims ``(uq_id, time, region)``; ``region`` is
        labelled by name. A file without a region dimension is one region,
        the ice-sheet total.

    Raises
    ------
    ValueError
        If the files carry none of the fluxes.
    """
    ds = xr.open_mfdataset(
        [str(f) for f in files],
        preprocess=partial(preprocess_netcdf, exp_dim=None, rgi_dim=None, gcm_dim=None, process_config=False),
        combine="nested",
        concat_dim=MEMBER_DIM,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="outer",
        decode_timedelta=True,
    )
    if "region" not in ds.dims:
        ds = ds.expand_dims(region=[TOTAL_REGION])
    elif "region_name" in ds.coords:
        ds = ds.set_index(region="region_name")
    ds["region"] = ds["region"].astype(str)

    names = {}
    for name, spec in VARIABLES.items():
        for candidate in spec.sim_vars:
            if candidate in ds:
                names[candidate] = name
                break
    if not names:
        raise ValueError(f"{files[0]} carries none of the fluxes {[s.sim_vars for s in VARIABLES.values()]}")
    ds = ds[list(names)].rename_vars(names).load()
    logger.info(
        "Ensemble: %d member(s), %d region(s), %d instant(s), fluxes %s",
        ds.sizes[MEMBER_DIM],
        ds.sizes["region"],
        ds.sizes["time"],
        sorted(names.values()),
    )
    return ds.transpose(MEMBER_DIM, "time", "region")


def load_observations(path: Path | str) -> xr.Dataset:
    """
    Open the staged Mankoff mass balance and put it in the model's convention.

    Parameters
    ----------
    path : Path or str
        ``mankoff_greenland_mass_balance.nc`` from ``pism-ismip7-greenland-observations``.

    Returns
    -------
    xarray.Dataset
        The fluxes of :data:`VARIABLES` and their uncertainties under their
        compared names, with dims ``(time, region)``, mass loss negative.
    """
    with xr.open_dataset(path, decode_timedelta=True) as ds:
        found = {}
        for name, spec in VARIABLES.items():
            if spec.obs_var in ds and spec.obs_std_var in ds:
                found[spec.obs_var] = name
                found[spec.obs_std_var] = uncertainty_name(name)
        obs = ds[list(found)].rename_vars(found).load()
    if "region" not in obs.dims:
        obs = obs.expand_dims(region=[TOTAL_REGION])
    obs["region"] = obs["region"].astype(str)
    with xr.set_options(keep_attrs=True):
        for name, spec in VARIABLES.items():
            if name in obs and spec.sign != 1.0:
                obs[name] = spec.sign * obs[name]
    logger.info("Observations: %d region(s), %s to %s", obs.sizes["region"], *time_span(obs))
    return obs.transpose("time", "region")


def time_span(ds: xr.Dataset) -> tuple[str, str]:
    """
    First and last instant of a record, as dates.

    Parameters
    ----------
    ds : xarray.Dataset
        Record with a ``time`` coordinate.

    Returns
    -------
    tuple of str
        ``(first, last)``.
    """
    return str(pd.Timestamp(ds.time.values[0]).date()), str(pd.Timestamp(ds.time.values[-1]).date())


def match_regions(sim_regions: Sequence[str], obs_regions: Sequence[str]) -> dict[str, str]:
    """
    Pair the model's regions with the observed ones.

    The model names a basin after the ice sheet and the basin (``GIS_NW``),
    the observations after the basin alone (``NW``); a model region is paired
    with the observed region of the same name, or else with the one its
    name ends in after its last underscore.

    Parameters
    ----------
    sim_regions : sequence of str
        Model region names.
    obs_regions : sequence of str
        Observed region names.

    Returns
    -------
    dict
        ``{sim_region: obs_region}`` for the regions both sides have.
    """
    available = set(map(str, obs_regions))
    pairs = {}
    for region in map(str, sim_regions):
        short = region.rsplit("_", 1)[-1]
        if region in available:
            pairs[region] = region
        elif short in available:
            pairs[region] = short
        else:
            logger.warning("Region %s has no observed counterpart among %s; skipped", region, sorted(available))
    return pairs


def resample_complete(ds: xr.Dataset, freq: str) -> xr.Dataset:
    """
    Average a record onto calendar bins, keeping the bins it fully covers.

    A bin the record only reaches into -- the half-reported final year of
    the observations, a first month a run starts in the middle of -- would
    otherwise be averaged over part of its span and compared with a full
    one. A bin counts as covered when the record's first instant in it is no
    later than one sampling step after the bin starts and its last no earlier
    than one step before the bin ends, the step being the record's median
    spacing; a monthly record stamped mid-month covers its months, a daily
    one covers its years.

    Parameters
    ----------
    ds : xarray.Dataset
        Record on a ``time`` coordinate.
    freq : str
        One of :data:`FREQUENCIES`.

    Returns
    -------
    xarray.Dataset
        Bin means on the bin labels of ``freq``.

    Raises
    ------
    ValueError
        If ``freq`` is not supported, or the record has fewer than two instants.
    """
    if freq not in FREQUENCIES:
        raise ValueError(f"freq must be one of {sorted(FREQUENCIES)}, got {freq!r}")
    times = pd.DatetimeIndex(ds.time.values)
    if times.size < 2:
        raise ValueError("a record needs at least two instants to be resampled")
    step = pd.Timedelta(np.median(np.diff(times.values)))
    stamps = xr.DataArray(times.values, dims=["time"], coords={"time": ds.time}, name="stamp")
    first = stamps.resample(time=freq).min()
    last = stamps.resample(time=freq).max()
    period = FREQUENCIES[freq]
    starts = np.array([pd.Timestamp(t).to_period(period).start_time.to_datetime64() for t in first.time.values])
    ends = np.array([pd.Timestamp(t).to_period(period).end_time.to_datetime64() for t in first.time.values])
    covered = (first.values <= starts + step.to_timedelta64()) & (last.values >= ends - step.to_timedelta64())
    with xr.set_options(keep_attrs=True):
        binned = ds.resample(time=freq).mean("time")
    kept = binned.isel(time=np.flatnonzero(covered))
    logger.info(
        "Resampled %d instant(s) to %d %s bin(s), %d partly covered dropped",
        times.size,
        kept.sizes["time"],
        freq,
        int((~covered).sum()),
    )
    return kept


def align_ensemble_and_observations(sim: xr.Dataset, obs: xr.Dataset, freq: str) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Put the ensemble and the observations on shared bins, regions and units.

    Parameters
    ----------
    sim : xarray.Dataset
        Ensemble (:func:`load_ensemble`).
    obs : xarray.Dataset
        Observations (:func:`load_observations`).
    freq : str
        Bins both are averaged onto, one of :data:`FREQUENCIES`.

    Returns
    -------
    tuple of xarray.Dataset
        The ensemble and the observations, on the bins both fully cover,
        the observations relabelled with the model's region names and
        converted to the model's units.

    Raises
    ------
    ValueError
        If no region or no instant is shared.
    """
    pairs = match_regions(sim.region.values, obs.region.values)
    if not pairs:
        raise ValueError("the ensemble and the observations share no region")
    obs = obs.sel(region=list(pairs.values())).assign_coords(region=list(pairs))
    sim = sim.sel(region=list(pairs))

    targets = {}
    for name in VARIABLES:
        if name in sim and "units" in sim[name].attrs:
            targets[name] = sim[name].attrs["units"]
            targets[uncertainty_name(name)] = sim[name].attrs["units"]
    obs = to_units(obs, targets)

    sim = resample_complete(sim, freq)
    obs = resample_complete(obs, freq)
    sim, obs = xr.align(sim, obs, join="inner", exclude=[MEMBER_DIM])
    if sim.sizes.get("time", 0) == 0:
        raise ValueError("the ensemble and the observations share no instant once binned")
    logger.info(
        "Aligned %d member(s) and %d region(s) over %d shared %s bin(s), %s to %s",
        sim.sizes[MEMBER_DIM],
        sim.sizes["region"],
        sim.sizes["time"],
        freq,
        *time_span(sim),
    )
    return sim, obs


def sampling_window(ds: xr.Dataset, start: str | None, end: str | None) -> xr.Dataset:
    """
    Cut a record down to the sampling window.

    Parameters
    ----------
    ds : xarray.Dataset
        Record on a ``time`` coordinate.
    start, end : str or None
        Bounds of the window, inclusive; ``None`` leaves that side open.

    Returns
    -------
    xarray.Dataset
        The instants inside the window.

    Raises
    ------
    ValueError
        If no instant falls inside it.
    """
    window = ds.sel(time=slice(start, end))
    if window.sizes.get("time", 0) == 0:
        raise ValueError(f"no instant between {start or 'the start'} and {end or 'the end'} of the shared record")
    return window


def window_span(ds: xr.Dataset, freq: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    First and last day the bins of a record cover.

    Parameters
    ----------
    ds : xarray.Dataset
        Record on the bin labels of ``freq``.
    freq : str
        One of :data:`FREQUENCIES`.

    Returns
    -------
    tuple of pandas.Timestamp
        Start of the first bin and end of the last, whichever way the bins
        are labelled.
    """
    period = FREQUENCIES[freq]
    first = pd.Timestamp(ds.time.values[0]).to_period(period).start_time
    last = pd.Timestamp(ds.time.values[-1]).to_period(period).end_time.normalize()
    return first, last


def error_stats(sim: xr.DataArray, obs: xr.DataArray, dim: str = MEMBER_DIM) -> xr.Dataset:
    """
    Per-member RMSE, MAE and bias over every dimension but the member one.

    Parameters
    ----------
    sim : xarray.DataArray
        Simulated series with a member dimension.
    obs : xarray.DataArray
        Observed series on the same instants.
    dim : str, optional
        Member dimension.

    Returns
    -------
    xarray.Dataset
        ``rmse``, ``mae``, ``bias`` and ``n_instants``, the number of
        instants each was taken over.
    """
    over = [d for d in sim.dims if d != dim]
    error = sim - obs
    units = obs.attrs.get("units", "")
    stats = xr.Dataset(
        {
            "rmse": np.sqrt((error**2).mean(dim=over, skipna=True)),
            "mae": abs(error).mean(dim=over, skipna=True),
            "bias": error.mean(dim=over, skipna=True),
            "n_instants": error.notnull().sum(dim=over),
        }
    )
    for name in ("rmse", "mae", "bias"):
        stats[name].attrs["units"] = units
    return stats


def score_region(
    sim: xr.Dataset,
    obs: xr.Dataset,
    variable: str,
    *,
    fudge_factors: Sequence[float] = DEFAULT_FUDGE_FACTORS,
    reduction: str = DEFAULT_REDUCTION,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = 0,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Weigh the members of one region on one flux over the sampling window.

    Parameters
    ----------
    sim : xarray.Dataset
        One region's ensemble over the window, dims ``(uq_id, time)``.
    obs : xarray.Dataset
        The region's observations over the window, dims ``(time,)``.
    variable : str
        Key of :data:`VARIABLES` to score on.
    fudge_factors : sequence of float, optional
        Inflations of the observed uncertainty, one filter each.
    reduction : {"sum", "mean"}, optional
        How the instants are collapsed into the log-likelihood.
    n_samples : int, optional
        Draws with replacement that resolve the weights into counts.
    seed : int, optional
        Seed of the resampler.

    Returns
    -------
    xarray.Dataset
        ``log_likelihood``, ``weights``, ``counts`` on ``(fudge_factor, uq_id)``
        and ``ess`` on ``fudge_factor``.
    xarray.Dataset
        The error statistics on ``uq_id`` (:func:`error_stats`).

    Raises
    ------
    ValueError
        If ``reduction`` is unknown or the flux is missing on either side.
    """
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
    std_var = uncertainty_name(variable)
    if variable not in sim or variable not in obs or std_var not in obs:
        raise ValueError(f"{variable!r} (and {std_var!r}) must be on both sides to be sampled")
    weighted = importance_weights(
        sim,
        obs,
        variable,
        obs_var=variable,
        obs_std_var=std_var,
        fudge_factors=tuple(fudge_factors),
        n_samples=n_samples,
        seed=seed,
        dim=MEMBER_DIM,
        sum_dims=("time",),
        reduction=reduction,
    )
    stats = error_stats(sim[variable], obs[variable]).compute()
    return weighted, stats


def plot_fluxes(
    sim: xr.Dataset,
    obs: xr.Dataset,
    weights: xr.DataArray | None,
    filename: Path | str,
    *,
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    title: str | None = None,
    sigma: float = OBS_BAND_SIGMA,
) -> None:
    """
    Plot the observed fluxes of one region with the prior and posterior bands.

    One panel per flux of :data:`PLOT_VARIABLES` both sides carry: the
    observations with a ``sigma``-wide band, the ensemble's 5-95 % band and
    median, and -- given weights -- the same band with every member counted
    by its weight. A sampling window narrower than the record is shaded.

    Parameters
    ----------
    sim : xarray.Dataset
        One region's ensemble, dims ``(uq_id, time)``.
    obs : xarray.Dataset
        The region's observations, dims ``(time,)``.
    weights : xarray.DataArray or None
        Posterior weight per member on ``uq_id``; ``None`` draws the prior only.
    filename : Path or str
        Output figure.
    window : tuple of pandas.Timestamp or None, optional
        Start and end of the sampling window.
    title : str or None, optional
        Figure title.
    sigma : float, optional
        Half-width of the observed band, in standard deviations.
    """
    variables = [v for v in PLOT_VARIABLES if v in sim and v in obs]
    if not variables:
        logger.warning("Nothing to plot: none of %s on both sides", PLOT_VARIABLES)
        return
    prior = xr.ones_like(sim[MEMBER_DIM], dtype=float)
    lo, mid, hi = BAND_QUANTILES
    with mpl.rc_context(rc=rc_params):
        fig, axs = plt.subplots(
            len(variables), 1, figsize=(6.2, 1.4 * len(variables) + 0.2), sharex=True, squeeze=False
        )
        for ax, name in zip(axs[:, 0], variables):
            std_var = uncertainty_name(name)
            if std_var in obs:
                ax.fill_between(
                    obs.time.values,
                    obs[name] - sigma * obs[std_var],
                    obs[name] + sigma * obs[std_var],
                    lw=0,
                    color="0.75",
                    alpha=0.5,
                    label=f"Observed ±{sigma:g}σ",
                )
            ax.plot(obs.time.values, obs[name], lw=1.0, color="k", label="Observed")
            bands = [("Prior 5-95%", prior, PRIOR_COLOR)]
            if weights is not None:
                bands.append(("Posterior 5-95%", weights, POSTERIOR_COLOR))
            for label, w, color in bands:
                q = weighted_quantiles(sim[name], w, BAND_QUANTILES, dim=MEMBER_DIM)
                ax.fill_between(
                    q.time.values, q.sel(quantile=lo), q.sel(quantile=hi), lw=0, color=color, alpha=0.4, label=label
                )
                ax.plot(q.time.values, q.sel(quantile=mid), lw=1.0, color=color)
            units = sim[name].attrs.get("units", "")
            ax.set_ylabel(f"{VARIABLES[name].label}\n({units})" if units else VARIABLES[name].label)
            if window is not None and (window[0] > sim.time.values[0] or window[1] < sim.time.values[-1]):
                ax.axvspan(window[0], window[1], color="0.92", zorder=0, lw=0, label="Sampling window")
        axs[0, 0].legend(fontsize=5, frameon=False, ncol=2, loc="lower left")
        axs[-1, 0].set_xlim(sim.time.values[0], sim.time.values[-1])
        axs[-1, 0].set_xlabel("")
        if title:
            fig.suptitle(title, fontsize=7)
        fig.tight_layout()
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=300)
        plt.close(fig)


def write_region(
    region: str,
    sim: xr.Dataset,
    obs: xr.Dataset,
    weighted: xr.Dataset,
    stats: xr.Dataset,
    *,
    uq_df: pd.DataFrame,
    variable: str,
    output_dir: Path,
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Write one region's tables, NetCDF and figures.

    Parameters
    ----------
    region : str
        Region name.
    sim : xarray.Dataset
        The region's ensemble over the whole shared record.
    obs : xarray.Dataset
        The region's observations over the whole shared record.
    weighted : xarray.Dataset
        Weights, counts, ESS and log-likelihood (:func:`score_region`).
    stats : xarray.Dataset
        Error statistics (:func:`score_region`).
    uq_df : pandas.DataFrame
        Parameter values per member.
    variable : str
        The sampled flux.
    output_dir : pathlib.Path
        The region's directory.
    window : tuple of pandas.Timestamp
        Start and end of the sampling window, for the figures and the attributes.

    Returns
    -------
    pandas.DataFrame
        One row per member: the parameters, the error statistics, and the
        weights and counts per fudge factor.
    list of dict
        Summary rows, one per fudge factor.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    members = uq_df.reindex(weighted[MEMBER_DIM].values)
    labels = short_labels(members.columns)
    table = posterior_table(weighted, members, dim=MEMBER_DIM).join(stats.to_dataframe())
    table.insert(0, "variable", variable)
    table.insert(0, "region", region)
    table.to_csv(output_dir / f"importance_sampling_{variable}.csv")
    out = xr.merge([weighted, stats])
    for column in uq_df.columns:
        out[column] = (MEMBER_DIM, members[column].values)
    out.attrs.update(
        {"region": region, "variable": variable, "start": str(window[0].date()), "end": str(window[1].date())}
    )
    out.to_netcdf(output_dir / f"importance_sampling_{variable}.nc")

    rows = []
    for fudge_factor in weighted.fudge_factor.values:
        w = weighted["weights"].sel(fudge_factor=fudge_factor)
        top = str(w.idxmax(dim=MEMBER_DIM).values)
        ess = float(weighted["ess"].sel(fudge_factor=fudge_factor))
        logger.info(
            "%s/%s: fudge %g: ESS %.1f of %d, top member %s (weight %.3f)",
            region,
            variable,
            fudge_factor,
            ess,
            w.sizes[MEMBER_DIM],
            top,
            float(w.max()),
        )
        plot_fluxes(
            sim,
            obs,
            w,
            output_dir / f"fluxes_{region}_{variable}_ff_{fudge_factor:g}.png",
            window=window,
            title=f"{region}: sampled on {variable}, fudge factor {fudge_factor:g}, ESS {ess:.1f} of {w.sizes[MEMBER_DIM]}",
        )
        plot_parameter_histograms(
            members,
            labels,
            weighted["counts"].sel(fudge_factor=fudge_factor).to_pandas(),
            output_dir / f"posterior_{region}_{variable}_ff_{fudge_factor:g}.png",
            prior=True,
            title=f"{region}: sampled on {variable}, fudge factor {fudge_factor:g}, ESS {ess:.1f} of {w.sizes[MEMBER_DIM]}",
        )
        rows.append(
            {
                "region": region,
                "variable": variable,
                "fudge_factor": float(fudge_factor),
                "n_members": int(w.sizes[MEMBER_DIM]),
                "ess": ess,
                "top_uq_id": top,
                "top_weight": float(w.max()),
                "best_rmse_uq_id": str(stats["rmse"].idxmin(dim=MEMBER_DIM).values),
                "best_rmse": float(stats["rmse"].min()),
                "n_instants": int(stats["n_instants"].max()),
            }
        )
    return table, rows


def joint_posterior(
    log_likes: Mapping[str, xr.DataArray],
    uq_df: pd.DataFrame,
    variable: str,
    output_dir: Path,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Sum the basins' log-likelihoods into one posterior and write it.

    Parameters
    ----------
    log_likes : mapping of str to xarray.DataArray
        Per-basin log-likelihood on ``(fudge_factor, uq_id)``.
    uq_df : pandas.DataFrame
        Parameter values per member.
    variable : str
        The sampled flux.
    output_dir : pathlib.Path
        Directory the joint table, NetCDF and figures go to.
    n_samples : int, optional
        Draws with replacement that resolve the weights into counts.
    seed : int, optional
        Seed of the resampler.

    Returns
    -------
    pandas.DataFrame
        One row per member.
    list of dict
        Summary rows, one per fudge factor.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total, member_ids = joint_log_likelihood(log_likes, dim=MEMBER_DIM)
    logger.info("Joint posterior over %d basin(s) and %d shared member(s)", len(log_likes), len(member_ids))
    weighted = weights_from_log_likelihood(total, dim=MEMBER_DIM, n_samples=n_samples, seed=seed)
    members = uq_df.reindex(member_ids)
    labels = short_labels(members.columns)
    table = posterior_table(weighted, members, dim=MEMBER_DIM)
    table.insert(0, "variable", variable)
    table.insert(0, "region", "joint")
    table.to_csv(output_dir / f"importance_sampling_{variable}.csv")
    weighted.attrs.update({"regions": sorted(log_likes), "variable": variable})
    weighted.to_netcdf(output_dir / f"importance_sampling_{variable}.nc")
    rows = []
    for fudge_factor in weighted.fudge_factor.values:
        w = weighted["weights"].sel(fudge_factor=fudge_factor)
        ess = float(weighted["ess"].sel(fudge_factor=fudge_factor))
        plot_parameter_histograms(
            members,
            labels,
            weighted["counts"].sel(fudge_factor=fudge_factor).to_pandas(),
            output_dir / f"posterior_joint_{variable}_ff_{fudge_factor:g}.png",
            prior=True,
            title=f"joint over {len(log_likes)} basins: sampled on {variable}, fudge factor {fudge_factor:g}, "
            f"ESS {ess:.1f} of {w.sizes[MEMBER_DIM]}",
        )
        rows.append(
            {
                "region": "joint",
                "variable": variable,
                "fudge_factor": float(fudge_factor),
                "n_members": int(w.sizes[MEMBER_DIM]),
                "ess": ess,
                "top_uq_id": str(w.idxmax(dim=MEMBER_DIM).values),
                "top_weight": float(w.max()),
                "n_regions": len(log_likes),
            }
        )
    return table, rows


def run_pipeline(
    run_dir: Path | str,
    output_path: Path | str,
    *,
    observations: Path | str | None = None,
    uq_file: Path | str | None = None,
    pattern: str = "region_*.nc",
    variable: str = "mass_balance",
    freq: str = DEFAULT_FREQ,
    start: str | None = None,
    end: str | None = None,
    fudge_factors: Sequence[float] = DEFAULT_FUDGE_FACTORS,
    reduction: str = DEFAULT_REDUCTION,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Weigh an ensemble's members on one flux, region by region, and combine the basins.

    Parameters
    ----------
    run_dir : Path or str
        Run directory searched recursively for the member region files.
    output_path : Path or str
        Directory the results are written to.
    observations : Path or str or None, optional
        The staged Mankoff product; ``<run_dir>/output/observations/`` by default.
    uq_file : Path or str or None, optional
        The run's ``uq.csv``; found below ``run_dir`` by default.
    pattern : str, optional
        Glob the member files match.
    variable : str, optional
        Flux to sample on, a key of :data:`VARIABLES`.
    freq : str, optional
        Bins both records are averaged onto, one of :data:`FREQUENCIES`.
    start, end : str or None, optional
        Bounds of the sampling window; ``None`` uses all the shared record.
    fudge_factors : sequence of float, optional
        Inflations of the observed uncertainty, one filter each.
    reduction : {"sum", "mean"}, optional
        How the instants are collapsed into the log-likelihood.
    n_samples : int, optional
        Draws with replacement that resolve the weights into counts.
    seed : int, optional
        Seed of the resampler.

    Returns
    -------
    pandas.DataFrame
        Summary: one row per region (and ``joint``) and fudge factor.

    Raises
    ------
    ValueError
        If ``variable`` is unknown.
    """
    if variable not in VARIABLES:
        raise ValueError(f"variable must be one of {sorted(VARIABLES)}, got {variable!r}")
    run_dir = Path(run_dir).expanduser()
    output_path = Path(output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    obs_file = Path(observations).expanduser() if observations is not None else run_dir / OBS_FILE

    uq_df = load_uq_parameters(run_dir, uq_file)
    sim = load_ensemble(find_region_files(run_dir, pattern))
    obs = load_observations(obs_file)
    sim, obs = align_ensemble_and_observations(sim, obs, freq)
    sim_window = sampling_window(sim, start, end)
    obs_window = obs.sel(time=sim_window.time)
    window = window_span(sim_window, freq)
    logger.info(
        "Sampling %s over %d %s bin(s), %s to %s",
        variable,
        sim_window.sizes["time"],
        freq,
        window[0].date(),
        window[1].date(),
    )

    tables, summaries = [], []
    log_likes: dict[str, xr.DataArray] = {}
    for region in progress_bar([str(r) for r in sim.region.values], desc="Regions", unit="region"):
        weighted, stats = score_region(
            sim_window.sel(region=region, drop=True),
            obs_window.sel(region=region, drop=True),
            variable,
            fudge_factors=fudge_factors,
            reduction=reduction,
            n_samples=n_samples,
            seed=seed,
        )
        table, rows = write_region(
            region,
            sim.sel(region=region, drop=True),
            obs.sel(region=region, drop=True),
            weighted,
            stats,
            uq_df=uq_df,
            variable=variable,
            output_dir=output_path / region,
            window=window,
        )
        tables.append(table)
        summaries.extend(rows)
        if region.rsplit("_", 1)[-1] != TOTAL_REGION:
            log_likes[region] = weighted["log_likelihood"]

    if len(log_likes) > 1:
        table, rows = joint_posterior(log_likes, uq_df, variable, output_path / "joint", n_samples=n_samples, seed=seed)
        tables.append(table)
        summaries.extend(rows)

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_path / "importance_sampling_summary.csv", index=False)
    pd.concat(tables).to_csv(output_path / "importance_sampling_weights.csv")
    logger.info("Wrote %s", output_path / "importance_sampling_summary.csv")
    return summary


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
        Exit code, ``0`` on success.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Importance-sample an ISMIP7 Greenland ensemble against the observed mass fluxes, per region."
    parser.add_argument(
        "--observations",
        default=None,
        metavar="FILE",
        help="The staged Mankoff mass balance. Default: RUN_DIR/output/observations/mankoff_greenland_mass_balance.nc.",
    )
    parser.add_argument(
        "--uq-file", default=None, metavar="CSV", help="The run's uq.csv. Default: found below RUN_DIR."
    )
    parser.add_argument("--pattern", default="region_*.nc", help="Glob the per-member region files match.")
    parser.add_argument(
        "--variable",
        choices=sorted(VARIABLES),
        default="mass_balance",
        help="Flux the members are weighed on.",
    )
    parser.add_argument(
        "--freq",
        choices=sorted(FREQUENCIES),
        default=DEFAULT_FREQ,
        help="Bins both records are averaged onto: months or years, labelled at the start or the end.",
    )
    parser.add_argument(
        "--start", default=None, help="First date of the sampling window. Default: the shared record's."
    )
    parser.add_argument("--end", default=None, help="Last date of the sampling window. Default: the shared record's.")
    # Comma-separated, not nargs="+": a greedy list would reach past itself into RUN_DIR.
    parser.add_argument(
        "--fudge-factors",
        type=lambda s: [float(part) for part in s.split(",")],
        default=list(DEFAULT_FUDGE_FACTORS),
        metavar="F[,F...]",
        help="Inflations of the observed uncertainty, one filter each.",
    )
    parser.add_argument(
        "--reduction",
        choices=REDUCTIONS,
        default=DEFAULT_REDUCTION,
        help="How the instants of the window are collapsed: 'sum' treats every bin as independent, 'mean' tempers.",
    )
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Draws resolving weights to counts.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the resampler.")
    parser.add_argument("RUN_DIR", nargs=1, help="Run directory searched recursively for the member region files.")
    parser.add_argument("OUTPUT_PATH", nargs=1, help="Directory to write the results into.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_path = Path(args.OUTPUT_PATH[0])
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(output_path / "importance_sampling.log")

    summary = run_pipeline(
        args.RUN_DIR[0],
        output_path,
        observations=args.observations,
        uq_file=args.uq_file,
        pattern=args.pattern,
        variable=args.variable,
        freq=args.freq,
        start=args.start,
        end=args.end,
        fudge_factors=args.fudge_factors,
        reduction=args.reduction,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    logger.info("\n%s", summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(main())
