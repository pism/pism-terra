"""
Importance-sample glacier UQ ensembles against observed elevation change.

For every glacier of a project directory that has post-processed ``dh_*.nc``
files (the surface-elevation change over the Hugonnet et al. (2021) period
written by ``pism-glacier-postprocess-dh``), the ensemble members are weighed
against the observed ``dh`` of the staged ``obs_<rgi_id>.nc``:

* a Gaussian likelihood weight per member for several fudge factors on the
  observed error, resolved into resampling counts and an effective sample
  size (:func:`pism_terra.calibration.importance_weights`);
* a plain per-member RMSE and a block-bootstrap RMSE ranking that honours
  the spatial autocorrelation of the field, with the set of members
  statistically tied with the best (:func:`pism_terra.calibration.rank_by_bootstrap_rmse`);
* posterior histograms of the UQ parameters (read from the run's
  ``uq.csv``) for each fudge factor and for the tied set.

The members are shared parameter draws across glaciers, so their
log-likelihoods are also summed over all glaciers into one joint posterior.

Outputs mirror the USGS benchmark tools: each glacier's figures, tables and
NetCDF file go to ``<output-path>/<rgi_id>/``, the joint posterior to
``<output-path>/joint/`` and the summary tables to the top level.
"""

from __future__ import annotations

import logging
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pint_xarray  # pylint: disable=unused-import
import xarray as xr

from pism_terra.calibration import (
    block_size_from_field,
    importance_weights,
    joint_log_likelihood,
    plot_parameter_histograms,
    posterior_table,
    rank_by_bootstrap_rmse,
    short_labels,
    weights_from_log_likelihood,
)
from pism_terra.glacier.observations import DH_END, DH_START
from pism_terra.glacier.usgs import find_model_files, rgi_output_dir
from pism_terra.likelihood import REDUCTIONS
from pism_terra.processing import preprocess_netcdf

logger = logging.getLogger(__name__)

RGI_PATTERN = re.compile(r"(RGI2000-v7\.0-[A-Z]-\d{2}-\d+)")
MEMBER_DIM = "uq_id"
DEFAULT_VARIABLES: tuple[tuple[str, str, str], ...] = (("usurf", "dh", "dh_err"),)
DEFAULT_FUDGE_FACTORS = (1.0, 3.0, 10.0)
DEFAULT_N_SAMPLES = 10_000
DEFAULT_N_BOOT = 500
DEFAULT_REDUCTION = "blocks"
DEFAULT_ACF_THRESHOLD = 1.0 / np.e


def parse_variable(spec: str) -> tuple[str, str, str]:
    """
    Parse a ``SIM:OBS[:OBS_STD]`` variable specification.

    Parameters
    ----------
    spec : str
        Simulated variable, observed variable and, optionally, the observed
        uncertainty, separated by colons; the uncertainty defaults to
        ``<OBS>_err``.

    Returns
    -------
    tuple of str
        ``(sim_var, obs_var, obs_std_var)``.

    Raises
    ------
    ValueError
        If ``spec`` has fewer than two or more than three fields.
    """
    parts = [p.strip() for p in spec.split(":")]
    if len(parts) == 2:
        parts.append(f"{parts[1]}_err")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"variable must be SIM:OBS[:OBS_STD], got {spec!r}")
    return parts[0], parts[1], parts[2]


def find_dh_files(run_dir: Path | str, start: str = DH_START, end: str = DH_END) -> dict[str, list[Path]]:
    """
    Group the project's ``dh`` files by glacier.

    Parameters
    ----------
    run_dir : Path or str
        Project directory searched recursively.
    start : str, optional
        Start date in the file names.
    end : str, optional
        End date in the file names.

    Returns
    -------
    dict
        ``{rgi_id: [files]}``, sorted by glacier and file name.
    """
    groups: dict[str, list[Path]] = {}
    for file in find_model_files(run_dir, pattern=f"dh_*_id_*_uq_*_{start}_{end}.nc"):
        match = RGI_PATTERN.search(file.name)
        if match:
            groups.setdefault(match.group(1), []).append(file)
    return dict(sorted(groups.items()))


def load_uq_parameters(run_dir: Path | str, rgi_id: str) -> pd.DataFrame:
    """
    Read the parameter values of a glacier's ensemble members.

    Parameters
    ----------
    run_dir : Path or str
        Project directory; the run generator writes ``<rgi_id>/output/uq.csv``.
    rgi_id : str
        Glacier.

    Returns
    -------
    pandas.DataFrame
        Parameter columns indexed by member id (``uq_id``, as strings to match
        the ensemble coordinate).

    Raises
    ------
    FileNotFoundError
        If no ``uq.csv`` exists below the glacier's directory.
    """
    candidates = [Path(run_dir) / rgi_id / "output" / "uq.csv"] + sorted((Path(run_dir) / rgi_id).rglob("uq.csv"))
    for csv in candidates:
        if csv.is_file():
            df = pd.read_csv(csv)
            df["uq"] = df["uq"].astype(str)
            return df.set_index("uq").rename_axis(MEMBER_DIM)
    raise FileNotFoundError(f"no uq.csv below {Path(run_dir) / rgi_id}")


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
        ``ds`` with the requested variables converted, units back in attrs.
    """
    skip = {c: None for c in ds.coords if "units" in ds[c].attrs}
    quantified = ds.pint.quantify(**skip)
    conversions = {v: u for v, u in targets.items() if v in quantified and quantified[v].pint.units is not None}
    return quantified.pint.to(conversions).pint.dequantify()


def load_observations(path: Path | str, variables: Sequence[tuple[str, str, str]]) -> xr.Dataset:
    """
    Open a glacier's observation file and express each uncertainty in its variable's units.

    Parameters
    ----------
    path : Path or str
        ``obs_<rgi_id>.nc`` written by the staging step.
    variables : sequence of tuple of str
        ``(sim_var, obs_var, obs_std_var)`` triples; only the observed
        variables (plus ``landice``, when present) are kept.

    Returns
    -------
    xarray.Dataset
        Observations with plain arrays and ``units`` attributes.
    """
    ds = xr.open_dataset(path)
    keep = {v for _, obs_var, obs_std in variables for v in (obs_var, obs_std)} & set(ds.data_vars)
    if "landice" in ds:
        keep.add("landice")
    ds = ds[sorted(keep)].load()
    targets = {
        obs_std: ds[obs_var].attrs["units"]
        for _, obs_var, obs_std in variables
        if obs_var in ds and obs_std in ds and "units" in ds[obs_var].attrs
    }
    return to_units(ds, targets)


def load_ensemble(files: Sequence[Path], obs: xr.Dataset, variables: Sequence[tuple[str, str, str]]) -> xr.Dataset:
    """
    Open a glacier's ``dh`` files as one ensemble on the observed grid.

    Parameters
    ----------
    files : sequence of Path
        Per-member ``dh`` files of one glacier.
    obs : xarray.Dataset
        Observations whose grid the ensemble is interpolated onto.
    variables : sequence of tuple of str
        ``(sim_var, obs_var, obs_std_var)`` triples; each simulated variable
        is converted to the units of its observed counterpart.

    Returns
    -------
    xarray.Dataset
        Lazy ensemble with dims ``(uq_id, time, y, x)`` on the observed grid.

    Raises
    ------
    ValueError
        If the files span more than one run id: the members must be one ensemble.
    """
    ds = xr.open_mfdataset(
        [str(f) for f in files],
        preprocess=partial(preprocess_netcdf, process_config=False),
        parallel=True,
        join="outer",
        compat="no_conflicts",
    )
    for dim in ("rgi_id", "exp_id"):
        if dim in ds.dims:
            if ds.sizes[dim] != 1:
                raise ValueError(f"{dim} has {ds.sizes[dim]} values; the dh files must belong to one ensemble")
            ds = ds.squeeze(dim, drop=True)
    sim_vars = [sim_var for sim_var, _, _ in variables if sim_var in ds]
    ds = ds[sim_vars].chunk({d: -1 for d in ("x", "y", "time") if d in ds.dims})
    ds = ds.interp_like(obs[[v for v in obs.data_vars if v != "landice"]])
    targets = {
        sim_var: obs[obs_var].attrs["units"]
        for sim_var, obs_var, _ in variables
        if sim_var in ds and obs_var in obs and "units" in obs[obs_var].attrs
    }
    return to_units(ds, targets)


def plain_rmse(sim: xr.DataArray, obs: xr.DataArray, dim: str = MEMBER_DIM) -> xr.DataArray:
    """
    Root-mean-square error of every member over all non-member dimensions.

    Parameters
    ----------
    sim : xarray.DataArray
        Simulated field with a ``dim`` dimension.
    obs : xarray.DataArray
        Observed field on the same grid.
    dim : str, optional
        Member dimension.

    Returns
    -------
    xarray.DataArray
        RMSE per member, NaN cells skipped.
    """
    over = [d for d in sim.dims if d != dim]
    rmse = np.sqrt(((sim - obs) ** 2).mean(dim=over, skipna=True))
    rmse.name = "rmse"
    rmse.attrs["units"] = obs.attrs.get("units", "")
    return rmse


def coverage(obs: xr.DataArray, sim: xr.DataArray, landice: xr.DataArray | None) -> tuple[int, int]:
    """
    Count the cells that enter the likelihood and the cells the glacier has.

    Parameters
    ----------
    obs : xarray.DataArray
        Observed field; NaN where the observations have holes.
    sim : xarray.DataArray
        One simulated member on the same grid.
    landice : xarray.DataArray or None
        Glacier mask (1 on ice) when the observation file carries one.

    Returns
    -------
    tuple of int
        ``(n_valid, n_glacier)``: cells finite in both fields, and cells under
        the mask or with an observation (cells with an observation when there
        is no mask), so the ratio is at most 1 and drops with holes in ``obs``.
    """
    finite_obs = np.isfinite(obs)
    valid = int((finite_obs & np.isfinite(sim)).sum())
    glacier = int(((landice == 1) | finite_obs).sum()) if landice is not None else int(finite_obs.sum())
    return valid, glacier


def plot_best_member(obs: xr.DataArray, best: xr.DataArray, title: str, filename: Path | str) -> None:
    """
    Map the observed field, the best member and their difference.

    Parameters
    ----------
    obs : xarray.DataArray
        Observed field with dims ``(y, x)``.
    best : xarray.DataArray
        Best member on the same grid.
    title : str
        Title of the middle panel.
    filename : Path or str
        Output figure.
    """
    vmax = float(np.nanmax(np.abs(np.concatenate([np.ravel(obs.values), np.ravel(best.values)]))))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
    obs.plot(ax=axes[0], vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    axes[0].set_title("Observed")
    best.plot(ax=axes[1], vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    axes[1].set_title(title, fontsize=8)
    (best - obs).plot(ax=axes[2], vmin=-vmax, vmax=vmax, cmap="RdBu")
    axes[2].set_title("Difference")
    for ax in axes:
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def benchmark_glacier(
    rgi_id: str,
    sim: xr.Dataset,
    obs: xr.Dataset,
    uq_df: pd.DataFrame,
    variables: Sequence[tuple[str, str, str]],
    *,
    output_dir: Path | str,
    fudge_factors: Sequence[float] = DEFAULT_FUDGE_FACTORS,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = 0,
    n_boot: int = DEFAULT_N_BOOT,
    bootstrap: bool = True,
    reduction: str = DEFAULT_REDUCTION,
    acf_threshold: float = DEFAULT_ACF_THRESHOLD,
) -> tuple[xr.Dataset, pd.DataFrame, pd.DataFrame]:
    """
    Importance-sample and rank one glacier's ensemble, writing its outputs.

    Parameters
    ----------
    rgi_id : str
        Glacier.
    sim : xarray.Dataset
        Ensemble on the observed grid (:func:`load_ensemble`).
    obs : xarray.Dataset
        Observations (:func:`load_observations`).
    uq_df : pandas.DataFrame
        Parameter values per member (:func:`load_uq_parameters`).
    variables : sequence of tuple of str
        ``(sim_var, obs_var, obs_std_var)`` triples to evaluate.
    output_dir : Path or str
        Top-level output directory; the glacier's files go to ``<rgi_id>/``.
    fudge_factors : sequence of float, optional
        Multipliers on the observed error.
    n_samples : int, optional
        Draws with replacement that resolve weights into counts.
    seed : int, optional
        Seed of the resampler and the bootstrap.
    n_boot : int, optional
        Bootstrap resamples of the RMSE ranking.
    bootstrap : bool, optional
        Skip the block-bootstrap ranking when False.
    reduction : {"blocks", "mean", "sum"}, optional
        How the likelihood collapses the cells; ``"blocks"`` sums one independent
        sample per decorrelation-length block of the observed field.
    acf_threshold : float, optional
        Autocorrelation level that defines the decorrelation length, and with
        it the block side; lower values give longer blocks.

    Returns
    -------
    xarray.Dataset
        Weights, counts, ESS, log-likelihood, RMSE and ranking on
        ``(variable, fudge_factor, uq_id)``, plus the parameters on ``uq_id``.
    pandas.DataFrame
        Summary rows: one per variable and fudge factor.
    pandas.DataFrame
        Long table: one row per variable and member, parameters included.
    """
    glacier_dir = rgi_output_dir(output_dir, rgi_id)
    labels = short_labels(uq_df.columns)
    landice = obs["landice"] if "landice" in obs else None
    per_variable, rows, tables = [], [], []
    for sim_var, obs_var, obs_std in variables:
        if sim_var not in sim or obs_var not in obs or obs_std not in obs:
            logger.warning("%s: skipping %s:%s:%s, variable missing", rgi_id, sim_var, obs_var, obs_std)
            continue
        members = uq_df.reindex(sim[MEMBER_DIM].values)
        sim_mean = sim[sim_var].mean(dim="time") if "time" in sim[sim_var].dims else sim[sim_var]
        obs_mean = obs[obs_var].mean(dim="time") if "time" in obs[obs_var].dims else obs[obs_var]
        length, block_size = block_size_from_field(obs_mean, threshold=acf_threshold)
        weighted = importance_weights(
            sim,
            obs,
            sim_var,
            obs_var=obs_var,
            obs_std_var=obs_std,
            fudge_factors=fudge_factors,
            n_samples=n_samples,
            seed=seed,
            dim=MEMBER_DIM,
            reduction=reduction,
            block_size=block_size,
        )
        rmse = plain_rmse(sim[sim_var], obs[obs_var]).compute()
        n_valid, n_glacier = coverage(obs_mean, sim_mean.isel({MEMBER_DIM: 0}).compute(), landice)
        result = weighted.assign(rmse=rmse)
        table = posterior_table(weighted, members, dim=MEMBER_DIM).assign(rmse=rmse.to_pandas())
        extra: dict[str, object] = {
            "decorrelation_length": length,
            "block_size": block_size,
            "acf_threshold": acf_threshold,
        }
        if bootstrap:
            ranking = rank_by_bootstrap_rmse(
                sim_mean, obs_mean, n_boot=n_boot, seed=seed, dim=MEMBER_DIM, block_size=block_size
            )
            result = result.assign(**{v: ranking[v] for v in ranking.data_vars})
            table = table.join(ranking.to_dataframe())
            best = ranking.attrs["best"]
            tied = ranking["tied_with_best"].to_pandas().astype(int)
            extra.update({"best_rmse_uq_id": best, "n_tied": int(tied.sum())})
            logger.info(
                "%s/%s: best member %s, %d tied within the 5-95%% CI, block %d px",
                rgi_id,
                sim_var,
                best,
                int(tied.sum()),
                ranking.attrs["block_size"],
            )
            plot_parameter_histograms(
                members,
                labels,
                tied,
                glacier_dir / f"importance_{sim_var}_tied.png",
                prior=True,
                title=f"{rgi_id} {sim_var}: members tied with the best RMSE (n={int(tied.sum())})",
            )
            best_field = sim_mean.sel({MEMBER_DIM: best}).compute()
            params = ", ".join(f"{labels[k]}={members.loc[best, k]:.4g}" for k in members.columns)
            plot_best_member(
                obs_mean,
                best_field,
                f"Best (uq_id={best}, RMSE={float(ranking['rmse_mean'].sel({MEMBER_DIM: best})):.2f})\n{params}",
                glacier_dir / f"importance_{sim_var}_best_rmse.png",
            )
        for fudge_factor in weighted.fudge_factor.values:
            w = weighted["weights"].sel(fudge_factor=fudge_factor)
            top = w.idxmax(dim=MEMBER_DIM).values
            ess = float(weighted["ess"].sel(fudge_factor=fudge_factor))
            logger.info(
                "%s/%s: fudge %g: ESS = %.1f of %d, top member %s (weight %.3f)",
                rgi_id,
                sim_var,
                fudge_factor,
                ess,
                w.sizes[MEMBER_DIM],
                top,
                float(w.max()),
            )
            plot_parameter_histograms(
                members,
                labels,
                weighted["counts"].sel(fudge_factor=fudge_factor).to_pandas(),
                glacier_dir / f"importance_{sim_var}_ff_{fudge_factor:g}.png",
                prior=True,
                title=f"{rgi_id} {sim_var}: fudge factor {fudge_factor:g}, ESS {ess:.1f} of {w.sizes[MEMBER_DIM]}",
            )
            rows.append(
                {
                    "rgi_id": rgi_id,
                    "variable": sim_var,
                    "fudge_factor": float(fudge_factor),
                    "reduction": reduction,
                    "n_members": int(w.sizes[MEMBER_DIM]),
                    "ess": ess,
                    "top_uq_id": str(top),
                    "top_weight": float(w.max()),
                    "n_valid_cells": n_valid,
                    "n_glacier_cells": n_glacier,
                    "coverage": n_valid / n_glacier if n_glacier else np.nan,
                    **extra,
                }
            )
        table.insert(0, "variable", sim_var)
        table.insert(0, "rgi_id", rgi_id)
        table.to_csv(glacier_dir / f"importance_{sim_var}.csv")
        tables.append(table)
        per_variable.append(result.expand_dims(variable=[sim_var]))
    if not per_variable:
        raise ValueError(f"{rgi_id}: none of the requested variables is available")
    out = xr.concat(per_variable, dim="variable")
    for column in uq_df.columns:
        out[column] = ("uq_id", uq_df.reindex(out[MEMBER_DIM].values)[column].values)
    out.attrs.update(
        {
            "rgi_id": rgi_id,
            "fudge_factors": list(map(float, fudge_factors)),
            "n_samples": n_samples,
            "reduction": reduction,
            "acf_threshold": acf_threshold,
        }
    )
    out.to_netcdf(glacier_dir / f"importance_sampling_{rgi_id}.nc")
    return out, pd.DataFrame(rows), pd.concat(tables)


def joint_posterior(
    log_likes: Mapping[str, xr.DataArray],
    uq_frames: Mapping[str, pd.DataFrame],
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = 0,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """
    Combine the glaciers' log-likelihoods into one posterior over the members.

    Parameters
    ----------
    log_likes : mapping of str to xarray.DataArray
        Per-glacier log-likelihood on ``(fudge_factor, uq_id)`` keyed by glacier,
        for one variable.
    uq_frames : mapping of str to pandas.DataFrame
        Parameter values per member, keyed by glacier; the members are shared
        draws, so the frames agree on the common members.
    n_samples : int, optional
        Draws with replacement that resolve weights into counts.
    seed : int, optional
        Seed of the resampler.

    Returns
    -------
    xarray.Dataset
        Joint weights, counts, ESS and log-likelihood on ``(fudge_factor, uq_id)``.
    pandas.DataFrame
        Summary rows, one per fudge factor.
    """
    total, members = joint_log_likelihood(log_likes, dim=MEMBER_DIM)
    weighted = weights_from_log_likelihood(total, dim=MEMBER_DIM, n_samples=n_samples, seed=seed)
    first = next(iter(uq_frames.values())).reindex(members)
    for rgi_id, frame in uq_frames.items():
        other = frame.reindex(members)
        if not np.allclose(first.values.astype(float), other.values.astype(float), equal_nan=True):
            logger.warning("%s: its uq.csv differs from the first glacier's on the common members", rgi_id)
    return weighted, first


def run_pipeline(
    run_dir: Path | str,
    *,
    data_path: Path | str | None = None,
    output_path: Path | str = ".",
    variables: Sequence[tuple[str, str, str]] = DEFAULT_VARIABLES,
    fudge_factors: Sequence[float] = DEFAULT_FUDGE_FACTORS,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = 0,
    n_boot: int = DEFAULT_N_BOOT,
    bootstrap: bool = True,
    min_members: int = 2,
    reduction: str = DEFAULT_REDUCTION,
    acf_threshold: float = DEFAULT_ACF_THRESHOLD,
    start: str = DH_START,
    end: str = DH_END,
) -> pd.DataFrame:
    """
    Importance-sample every glacier of a project and combine them.

    Parameters
    ----------
    run_dir : Path or str
        Project directory searched recursively for ``dh`` files.
    data_path : Path or str or None, optional
        Staging tree holding ``<rgi_id>/input/obs_<rgi_id>.nc``; defaults to
        ``run_dir``, where the run generator puts the inputs when it was not
        given a separate data path.
    output_path : Path or str, optional
        Where the summary tables go; each glacier writes to ``<rgi_id>/`` and
        the joint posterior to ``joint/`` below it.
    variables : sequence of tuple of str, optional
        ``(sim_var, obs_var, obs_std_var)`` triples to evaluate.
    fudge_factors : sequence of float, optional
        Multipliers on the observed error.
    n_samples : int, optional
        Draws with replacement that resolve weights into counts.
    seed : int, optional
        Seed of the resampler and the bootstrap.
    n_boot : int, optional
        Bootstrap resamples of the RMSE ranking.
    bootstrap : bool, optional
        Skip the block-bootstrap ranking when False.
    min_members : int, optional
        Glaciers with fewer finished members are skipped: a single member has no
        posterior, and it would shrink the joint posterior to that member.
    reduction : {"blocks", "mean", "sum"}, optional
        How the likelihood collapses the cells of a field; see
        :func:`pism_terra.likelihood.reduce_log_likelihood`.
    acf_threshold : float, optional
        Autocorrelation level defining the decorrelation length and block side.
    start : str, optional
        Start date of the ``dh`` files.
    end : str, optional
        End date of the ``dh`` files.

    Returns
    -------
    pandas.DataFrame
        Summary: one row per glacier (and ``joint``), variable and fudge factor.
    """
    run_dir = Path(run_dir).expanduser()
    data_root = Path(data_path).expanduser() if data_path is not None else run_dir
    output_path = Path(output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    groups = find_dh_files(run_dir, start=start, end=end)
    if not groups:
        raise FileNotFoundError(f"no dh_*_id_*_uq_*_{start}_{end}.nc below {run_dir}")
    logger.info("%d glacier(s) with dh files below %s", len(groups), run_dir)

    summaries, tables = [], []
    log_likes: dict[str, dict[str, xr.DataArray]] = {}
    uq_frames: dict[str, pd.DataFrame] = {}
    for rgi_id, files in groups.items():
        obs_file = data_root / rgi_id / "input" / f"obs_{rgi_id}.nc"
        if not obs_file.is_file():
            logger.warning("%s: no observations at %s, skipped", rgi_id, obs_file)
            continue
        try:
            uq_df = load_uq_parameters(run_dir, rgi_id)
        except FileNotFoundError as err:
            logger.warning("%s: %s, skipped", rgi_id, err)
            continue
        if len(files) < min_members:
            logger.warning("%s: only %d member(s) finished, fewer than %d, skipped", rgi_id, len(files), min_members)
            continue
        logger.info("%s: %d members", rgi_id, len(files))
        obs = load_observations(obs_file, variables)
        if not any(obs_var in obs and obs_std in obs for _, obs_var, obs_std in variables):
            logger.warning("%s: %s has none of the observed variables, skipped", rgi_id, obs_file.name)
            continue
        sim = load_ensemble(files, obs, variables)
        result, summary, table = benchmark_glacier(
            rgi_id,
            sim,
            obs,
            uq_df,
            variables,
            output_dir=output_path,
            fudge_factors=fudge_factors,
            n_samples=n_samples,
            seed=seed,
            n_boot=n_boot,
            bootstrap=bootstrap,
            reduction=reduction,
            acf_threshold=acf_threshold,
        )
        summaries.append(summary)
        tables.append(table)
        uq_frames[rgi_id] = uq_df
        for variable in result["variable"].values:
            log_likes.setdefault(str(variable), {})[rgi_id] = result["log_likelihood"].sel(variable=variable, drop=True)

    if not summaries:
        raise FileNotFoundError("no glacier had both dh files and observations")

    if len(uq_frames) > 1:
        joint_dir = output_path / "joint"
        joint_dir.mkdir(parents=True, exist_ok=True)
        joint_results = []
        for variable, per_glacier in log_likes.items():
            weighted, members = joint_posterior(per_glacier, uq_frames, n_samples=n_samples, seed=seed)
            labels = short_labels(members.columns)
            table = posterior_table(weighted, members, dim=MEMBER_DIM)
            table.insert(0, "variable", variable)
            table.insert(0, "rgi_id", "joint")
            table.to_csv(joint_dir / f"importance_joint_{variable}.csv")
            tables.append(table)
            for fudge_factor in weighted.fudge_factor.values:
                w = weighted["weights"].sel(fudge_factor=fudge_factor)
                ess = float(weighted["ess"].sel(fudge_factor=fudge_factor))
                plot_parameter_histograms(
                    members,
                    labels,
                    weighted["counts"].sel(fudge_factor=fudge_factor).to_pandas(),
                    joint_dir / f"importance_joint_{variable}_ff_{fudge_factor:g}.png",
                    prior=True,
                    title=f"joint over {len(per_glacier)} glaciers, {variable}: fudge factor {fudge_factor:g}, "
                    f"ESS {ess:.1f} of {w.sizes[MEMBER_DIM]}",
                )
                summaries.append(
                    pd.DataFrame(
                        [
                            {
                                "rgi_id": "joint",
                                "variable": variable,
                                "fudge_factor": float(fudge_factor),
                                "reduction": reduction,
                                "n_members": int(w.sizes[MEMBER_DIM]),
                                "ess": ess,
                                "top_uq_id": str(w.idxmax(dim=MEMBER_DIM).values),
                                "top_weight": float(w.max()),
                                "n_glaciers": len(per_glacier),
                            }
                        ]
                    )
                )
            joint_results.append(weighted.expand_dims(variable=[variable]))
        joint = xr.concat(joint_results, dim="variable")
        joint.attrs.update({"glaciers": sorted(uq_frames), "n_samples": n_samples, "reduction": reduction})
        joint.to_netcdf(joint_dir / "importance_sampling_joint.nc")

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output_path / "importance_sampling_summary.csv", index=False)
    pd.concat(tables).to_csv(output_path / "importance_sampling_weights.csv")
    return summary


def setup_logging(log_file: Path | str) -> None:
    """
    Log to the console and to ``log_file``.

    Parameters
    ----------
    log_file : Path or str
        File the log is appended to.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_file)):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Command-line entry point.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    pandas.DataFrame
        The summary table.
    """
    parser = ArgumentParser(
        description="Importance-sample every glacier UQ ensemble of a project against its observed elevation change.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("RUN_DIR", help="Project directory searched recursively for dh_*_id_*_uq_*.nc files.")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Staging tree with <rgi_id>/input/obs_<rgi_id>.nc; defaults to RUN_DIR.",
    )
    parser.add_argument(
        "--output-path",
        default=".",
        help="Directory for the summary tables; each glacier writes to <rgi_id>/ below it.",
    )
    parser.add_argument(
        "--variable",
        action="append",
        type=parse_variable,
        default=None,
        help="SIM:OBS[:OBS_STD] to evaluate; repeatable. Default usurf:dh:dh_err.",
    )
    parser.add_argument(
        "--fudge-factors",
        type=lambda s: tuple(float(x) for x in s.split(",")),
        default=DEFAULT_FUDGE_FACTORS,
        help="Comma-separated multipliers on the observed error.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Draws with replacement per fudge factor."
    )
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT, help="Bootstrap resamples of the RMSE ranking.")
    parser.add_argument(
        "--no-bootstrap", action="store_true", default=False, help="Skip the block-bootstrap RMSE ranking."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed of the resampler and the bootstrap.")
    parser.add_argument(
        "--reduction",
        choices=REDUCTIONS,
        default=DEFAULT_REDUCTION,
        help="How the likelihood collapses the cells: 'blocks' sums one independent sample per "
        "decorrelation-length block, 'mean' averages (tempered), 'sum' treats every cell as independent.",
    )
    parser.add_argument(
        "--acf-threshold",
        type=float,
        default=DEFAULT_ACF_THRESHOLD,
        help="Autocorrelation level that defines the decorrelation length and hence the block side; "
        "lower values give longer blocks and a softer posterior.",
    )
    parser.add_argument(
        "--min-members", type=int, default=2, help="Skip glaciers with fewer finished ensemble members than this."
    )
    parser.add_argument("--start", default=DH_START, help="Start date in the dh file names.")
    parser.add_argument("--end", default=DH_END, help="End date in the dh file names.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_path = Path(args.output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(output_path / "importance_sampling.log")
    return run_pipeline(
        args.RUN_DIR,
        data_path=args.data_path,
        output_path=output_path,
        variables=tuple(args.variable) if args.variable else DEFAULT_VARIABLES,
        fudge_factors=args.fudge_factors,
        n_samples=args.n_samples,
        seed=args.seed,
        n_boot=args.n_boot,
        bootstrap=not args.no_bootstrap,
        min_members=args.min_members,
        reduction=args.reduction,
        acf_threshold=args.acf_threshold,
        start=args.start,
        end=args.end,
    )


def cli(argv: Sequence[str] | None = None) -> int:
    """
    Console-script wrapper around :func:`main`.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Exit status.
    """
    main(argv)
    return 0


if __name__ == "__main__":
    cli()
