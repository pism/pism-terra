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
Importance-sample an ISMIP7 Greenland ensemble against observed thickness change.

The ice-sheet counterpart of ``pism-glacier-importance-sampling``. The
members of an OCX or historical ensemble -- their cumulative ``dh`` written
by ``pism-ismip7-greenland-postprocess-dh`` -- are weighed against the
observed products staged by ``pism-ismip7-greenland-prepare --include dh``
(Khan's annual series and Smith's 2003-2019 total), giving:

* per-member RMSE, MAE and bias over the cells both sides cover;
* a block-bootstrap RMSE ranking that honours the spatial autocorrelation
  of the field, and the members statistically tied with the best
  (:func:`pism_terra.calibration.rank_by_bootstrap_rmse`);
* Gaussian importance weights for several fudge factors on the observed
  error, resolved into resampling counts and an effective sample size
  (:func:`pism_terra.calibration.importance_weights`);
* one joint posterior over the observed products, since the members are
  shared parameter draws.

Two things differ from the glacier version.

**Conservative regridding, not interpolation.** Members written into the
submission tree already sit on the observations' 1 km grid and are left
alone. A member on another grid -- a 900 m historical run, say -- is
regridded conservatively, which preserves the field's integral across the
change of support; the bilinear ``interp_like`` the glacier version uses
does not, and for a thickness change that sums into a mass budget that
matters.

**The observations carry no uncertainty.** Neither product ships an error
estimate, so the likelihood uses a relative error with an absolute floor
(:func:`pism_terra.calibration.observation_uncertainty`). The floor keeps
cells where nothing changed from dominating the likelihood, which a purely
relative error would let them do.

Results go to ``<output-path>/<product>/`` per product, with the joint
posterior and the summary tables at the top level.
"""

from __future__ import annotations

import logging
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xarray_regrid.methods.conservative  # noqa: F401  pylint: disable=unused-import

from pism_terra.calibration import (
    importance_weights,
    joint_log_likelihood,
    observation_uncertainty,
    rank_by_bootstrap_rmse,
    weights_from_log_likelihood,
)
from pism_terra.likelihood import REDUCTIONS
from pism_terra.log import setup_logging
from pism_terra.processing import preprocess_netcdf

logger = logging.getLogger(__name__)

#: Dimension the ensemble members are stacked along. Named as in the glacier
#: version so the posterior helpers and a run's ``uq.csv`` line up.
MEMBER_DIM = "uq_id"

#: Variable the model dh files carry.
SIM_VAR = "lithk"

#: Variable the observed dh files carry, and the name both are compared under.
OBS_VAR = "dh"

#: The set counter of an ISMIP7 file name, which is what identifies a member:
#: it is the key into the run's ``ismip7_members.csv`` parameter table.
SET_COUNTER_PATTERN = re.compile(r"_([CEP]\d{3})_")

#: A UQ draw, for ensembles named the way the glacier runs are.
UQ_PATTERN = re.compile(r"_uq_(\d+)")

#: Observed product in a staged ``dh_<product>_g1000m_*.nc`` file name.
PRODUCT_PATTERN = re.compile(r"^dh_([a-z0-9]+)_")

#: Relative error and floor of the observed uncertainty, in metres. Below
#: half a metre over the whole record the altimetry is not resolving change,
#: and a purely relative error would make those cells infinitely informative.
DEFAULT_RELATIVE_ERROR = 0.10
DEFAULT_ERROR_FLOOR = 0.5

#: Inflations of the observed error, one filter each. Cells are far from
#: independent, so an uninflated likelihood is overconfident.
DEFAULT_FUDGE_FACTORS = (1.0, 3.0, 10.0)

DEFAULT_N_SAMPLES = 10_000
DEFAULT_N_BOOT = 500


def member_label(path: Path) -> str:
    """
    Identify the ensemble member a file belongs to.

    The set counter is preferred: the ISMIP7 protocol makes it the key into
    the spreadsheet of parameter and modelling choices, which is the table
    ``pism-ismip7-greenland-run`` writes as ``ismip7_members.csv``. A UQ draw
    is the fallback for ensembles named the way the glacier runs are, and the
    stem the last resort.

    Parameters
    ----------
    path : pathlib.Path
        Member file.

    Returns
    -------
    str
        Member label.
    """
    for pattern in (SET_COUNTER_PATTERN, UQ_PATTERN):
        match = pattern.search(path.name)
        if match:
            return match.group(1)
    return path.stem


def product_name(path: Path) -> str:
    """
    Name of an observed product, from its staged file name.

    Parameters
    ----------
    path : pathlib.Path
        Observed ``dh_<product>_g1000m_*.nc`` file.

    Returns
    -------
    str
        Product name, e.g. ``"khan"``; the stem when the name does not follow
        the staged convention.
    """
    match = PRODUCT_PATTERN.match(path.name)
    return match.group(1) if match else path.stem


def find_member_files(run_dir: Path | str, pattern: str = "dh_*.nc") -> list[Path]:
    """
    Collect the per-member ``dh`` files of an ensemble.

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
        If nothing matches, rather than reporting statistics over no members.
    """
    files = sorted(Path(run_dir).rglob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} under {run_dir}")
    logger.info("Found %d member file(s) under %s", len(files), run_dir)
    return files


def same_grid(a: xr.Dataset, b: xr.Dataset) -> bool:
    """
    Whether two datasets share a grid, ignoring the direction of the axes.

    The submission files run ``y`` the other way from the observations, which
    alignment sorts out on its own; only the set of coordinates decides
    whether regridding is needed.

    Parameters
    ----------
    a, b : xarray.Dataset
        Datasets carrying ``x`` and ``y``.

    Returns
    -------
    bool
        ``True`` when both axes hold the same coordinates.
    """
    return all(
        a.sizes.get(axis) == b.sizes.get(axis) and np.allclose(np.sort(a[axis].values), np.sort(b[axis].values))
        for axis in ("x", "y")
    )


def to_observed_grid(ds: xr.Dataset, obs: xr.Dataset) -> xr.Dataset:
    """
    Put an ensemble on the observations' grid, conservatively.

    A no-op when the grids already match, which they do for members written
    into the submission tree -- that grid *is* the observations' 1 km grid.
    Otherwise the field is regridded conservatively, so its integral survives
    the change of support.

    Parameters
    ----------
    ds : xarray.Dataset
        Ensemble to regrid.
    obs : xarray.Dataset
        Observations whose grid is the target.

    Returns
    -------
    xarray.Dataset
        ``ds`` on the observed grid.
    """
    if same_grid(ds, obs):
        logger.info("Ensemble is already on the observed grid; not regridding")
        return ds
    logger.info(
        "Regridding %d x %d -> %d x %d (conservative)",
        ds.sizes["y"],
        ds.sizes["x"],
        obs.sizes["y"],
        obs.sizes["x"],
    )
    # Only the target's grid matters, so hand the regridder bare coordinates
    # rather than the observations and their data.
    target = xr.Dataset(coords={"y": obs["y"], "x": obs["x"]}).reset_coords(drop=True)
    return ds.regrid.conservative(target)


def load_observations(path: Path | str, relative: float, floor: float) -> xr.Dataset:
    """
    Load an observed ``dh`` product and give it an uncertainty.

    Parameters
    ----------
    path : Path or str
        Staged ``dh_khan_*`` or ``dh_smith_*`` file.
    relative : float
        Relative error as a fraction of the absolute value.
    floor : float
        Smallest error, metres.

    Returns
    -------
    xarray.Dataset
        ``dh`` and ``dh_error``.
    """
    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(path, decode_times=coder) as ds:
        obs = ds[[OBS_VAR]].load()
    return observation_uncertainty(obs, relative=relative, floor=floor)


def load_ensemble(files: Sequence[Path]) -> xr.Dataset:
    """
    Open the member files as one ensemble.

    Parameters
    ----------
    files : sequence of pathlib.Path
        Per-member ``dh`` files.

    Returns
    -------
    xarray.Dataset
        ``dh`` with a member dimension labelled by :func:`member_label`.

    Raises
    ------
    ValueError
        If the files do not carry the simulated variable.
    """
    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    # ISMIP7 file names carry none of the identifiers the glacier
    # preprocessor extracts, so every one of its dimensions is switched off
    # and the member dimension is attached here instead.
    ds = xr.open_mfdataset(
        [str(f) for f in files],
        preprocess=partial(
            preprocess_netcdf,
            process_config=False,
            exp_dim=None,
            rgi_dim=None,
            uq_dim=None,
            gcm_dim=None,
        ),
        decode_times=coder,
        combine="nested",
        concat_dim=MEMBER_DIM,
        join="outer",
        compat="no_conflicts",
    )
    if SIM_VAR not in ds:
        raise ValueError(f"{SIM_VAR!r} not among {sorted(ds.data_vars)} in {files[0]}")
    ds = ds[[SIM_VAR]].assign_coords({MEMBER_DIM: [member_label(Path(f)) for f in files]})
    return ds.rename({SIM_VAR: OBS_VAR})


def align_to_observations(sim: xr.Dataset, obs: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Put the ensemble on the observed grid and keep the records both have.

    Both sides are cumulative from the same 2003 reference, so their records
    line up on their dates; an inner join keeps the ones present in both,
    which for Smith's single total is the one date it reports.

    Parameters
    ----------
    sim : xarray.Dataset
        Ensemble.
    obs : xarray.Dataset
        Observations.

    Returns
    -------
    tuple of xarray.Dataset
        Aligned ensemble and observations.

    Raises
    ------
    ValueError
        If the two share no time step, which would otherwise be reported as
        statistics over nothing.
    """
    sim = to_observed_grid(sim, obs)
    sim, obs = xr.align(sim, obs, join="inner")
    if sim.sizes.get("time", 0) == 0:
        raise ValueError("the ensemble and the observations share no time step")
    logger.info("Comparing %d member(s) over %d shared time step(s)", sim.sizes[MEMBER_DIM], sim.sizes["time"])
    return sim, obs


def error_stats(sim: xr.DataArray, obs: xr.DataArray, dim: str = MEMBER_DIM) -> xr.Dataset:
    """
    Per-member RMSE, MAE and bias over every dimension but the member one.

    All three are reported because they answer different questions: RMSE is
    what the Gaussian likelihood is built on and is driven by the worst
    cells, MAE says what a typical cell is off by, and bias says whether the
    member is systematically thick or thin. A member can have a small bias
    and a large RMSE, or the reverse, and only the pair tells a member that
    is wrong everywhere from one that is wrong in one place.

    Parameters
    ----------
    sim : xarray.DataArray
        Simulated field with a member dimension.
    obs : xarray.DataArray
        Observed field on the same grid.
    dim : str, optional
        Member dimension.

    Returns
    -------
    xarray.Dataset
        ``rmse``, ``mae``, ``bias`` and the number of cells each was taken
        over -- the count matters, because members that cover different parts
        of the observations are not being scored on the same thing.
    """
    over = [d for d in sim.dims if d != dim]
    error = sim - obs
    units = obs.attrs.get("units", "")
    stats = xr.Dataset(
        {
            "rmse": np.sqrt((error**2).mean(dim=over, skipna=True)),
            "mae": abs(error).mean(dim=over, skipna=True),
            "bias": error.mean(dim=over, skipna=True),
            "n_cells": error.notnull().sum(dim=over),
        }
    )
    for name in ("rmse", "mae", "bias"):
        stats[name].attrs["units"] = units
    return stats


def score_product(
    sim: xr.Dataset,
    obs: xr.Dataset,
    output_dir: Path,
    *,
    fudge_factors: Sequence[float] = DEFAULT_FUDGE_FACTORS,
    reduction: str = "blocks",
    n_samples: int = DEFAULT_N_SAMPLES,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """
    Score an ensemble against one observed product.

    Parameters
    ----------
    sim : xarray.Dataset
        Ensemble, already on the observed grid.
    obs : xarray.Dataset
        Observations with their uncertainty.
    output_dir : pathlib.Path
        Directory the product's table and NetCDF go to.
    fudge_factors : sequence of float, optional
        Inflations of the observed error, one filter each.
    reduction : str, optional
        How the cells are collapsed into the log-likelihood.
    n_samples : int, optional
        Draws with replacement that resolve the weights into counts.
    n_boot : int, optional
        Bootstrap resamples of the RMSE ranking.
    seed : int, optional
        Seed of the resampler and the bootstrap.

    Returns
    -------
    xarray.Dataset
        Weights, counts, ESS and log-likelihood on ``(fudge_factor, uq_id)``.
    pandas.DataFrame
        One row per member and fudge factor: the error statistics, the
        bootstrap ranking and the weights.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = error_stats(sim[OBS_VAR], obs[OBS_VAR]).compute()
    # The ranking compares fields, not series, so it is taken on the record's
    # mean rather than once per time step.
    ranking = rank_by_bootstrap_rmse(
        sim[OBS_VAR].mean(dim="time"),
        obs[OBS_VAR].mean(dim="time"),
        n_boot=n_boot,
        seed=seed,
        dim=MEMBER_DIM,
    )
    weighted = importance_weights(
        sim,
        obs,
        OBS_VAR,
        fudge_factors=tuple(fudge_factors),
        n_samples=n_samples,
        seed=seed,
        dim=MEMBER_DIM,
        sum_dims=("time", "y", "x"),
        reduction=reduction,
    )

    table = weighted[["log_likelihood", "weights", "counts"]].to_dataframe().reset_index()
    table = table.join(weighted["ess"].to_dataframe(), on="fudge_factor", rsuffix="_ess")
    table = table.merge(stats.to_dataframe().reset_index(), on=MEMBER_DIM)
    table = table.merge(ranking.to_dataframe().reset_index(), on=MEMBER_DIM)
    table = table.sort_values(["fudge_factor", "rmse"])

    xr.merge([weighted, stats, ranking]).to_netcdf(output_dir / "importance_sampling.nc")
    table.to_csv(output_dir / "importance_sampling.csv", index=False)
    logger.info(
        "%s: best RMSE %s (%.3f m), %d member(s) tied",
        output_dir.name,
        ranking.attrs["best"],
        float(ranking["rmse_mean"].min()),
        int(ranking["tied_with_best"].sum()),
    )
    return weighted, table


def joint_posterior(
    log_likes: Mapping[str, xr.DataArray],
    output_dir: Path,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = 0,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """
    Sum the log-likelihoods over the observed products.

    The members are shared parameter draws, so a member's evidence against
    Khan and against Smith is evidence about the same thing and adds.

    Parameters
    ----------
    log_likes : mapping of str to xarray.DataArray
        Per-product log-likelihood on ``(fudge_factor, uq_id)``.
    output_dir : pathlib.Path
        Directory the joint table and NetCDF go to.
    n_samples : int, optional
        Draws with replacement that resolve the weights into counts.
    seed : int, optional
        Seed of the resampler.

    Returns
    -------
    xarray.Dataset
        Joint weights, counts, ESS and log-likelihood.
    pandas.DataFrame
        One row per member and fudge factor.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total, members = joint_log_likelihood(log_likes, dim=MEMBER_DIM)
    logger.info("Joint posterior over %d product(s) and %d shared member(s)", len(log_likes), len(members))
    weighted = weights_from_log_likelihood(total, dim=MEMBER_DIM, n_samples=n_samples, seed=seed)
    table = weighted[["log_likelihood", "weights", "counts"]].to_dataframe().reset_index()
    table = table.join(weighted["ess"].to_dataframe(), on="fudge_factor", rsuffix="_ess")
    table = table.sort_values(["fudge_factor", "weights"], ascending=[True, False])
    weighted.to_netcdf(output_dir / "importance_sampling.nc")
    table.to_csv(output_dir / "importance_sampling.csv", index=False)
    return weighted, table


def run_pipeline(
    run_dir: Path | str,
    observations: Sequence[Path | str],
    output_path: Path | str,
    *,
    pattern: str = "dh_*.nc",
    relative: float = DEFAULT_RELATIVE_ERROR,
    floor: float = DEFAULT_ERROR_FLOOR,
    fudge_factors: Sequence[float] = DEFAULT_FUDGE_FACTORS,
    reduction: str = "blocks",
    n_samples: int = DEFAULT_N_SAMPLES,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Score an ensemble against every observed product and weigh its members.

    Parameters
    ----------
    run_dir : Path or str
        Directory holding the per-member ``dh`` files.
    observations : sequence of Path or str
        Observed ``dh`` products.
    output_path : Path or str
        Directory the results are written to.
    pattern : str, optional
        Glob the member files match.
    relative, floor : float, optional
        Relative error and floor of the observed uncertainty.
    fudge_factors : sequence of float, optional
        Inflations of the observed error, one filter each.
    reduction : str, optional
        How the cells are collapsed into the log-likelihood.
    n_samples : int, optional
        Draws with replacement that resolve the weights into counts.
    n_boot : int, optional
        Bootstrap resamples of the RMSE ranking.
    seed : int, optional
        Seed of the resampler and the bootstrap.

    Returns
    -------
    pandas.DataFrame
        Every product's rows, with a ``product`` column.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    files = find_member_files(run_dir, pattern)

    tables = []
    log_likes: dict[str, xr.DataArray] = {}
    for observed in observations:
        observed = Path(observed)
        product = product_name(observed)
        obs = load_observations(observed, relative, floor)
        sim, obs = align_to_observations(load_ensemble(files), obs)
        weighted, table = score_product(
            sim,
            obs,
            output_path / product,
            fudge_factors=fudge_factors,
            reduction=reduction,
            n_samples=n_samples,
            n_boot=n_boot,
            seed=seed,
        )
        log_likes[product] = weighted["log_likelihood"]
        tables.append(table.assign(product=product))

    if len(log_likes) > 1:
        _, joint = joint_posterior(log_likes, output_path / "joint", n_samples=n_samples, seed=seed)
        tables.append(joint.assign(product="joint"))

    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(output_path / "importance_sampling_summary.csv", index=False)
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
    parser.description = "Importance-sample an ISMIP7 Greenland ensemble against observed thickness change."
    parser.add_argument(
        "--observations",
        nargs="+",
        required=True,
        help="Observed dh products, e.g. the staged dh_khan_* and dh_smith_* files.",
    )
    parser.add_argument("--pattern", default="dh_*.nc", help="Glob the per-member dh files match.")
    parser.add_argument(
        "--relative-error",
        type=float,
        default=DEFAULT_RELATIVE_ERROR,
        help="Observed uncertainty as a fraction of the absolute value.",
    )
    parser.add_argument(
        "--error-floor",
        type=float,
        default=DEFAULT_ERROR_FLOOR,
        help="Smallest observed uncertainty, metres.",
    )
    parser.add_argument(
        "--fudge-factors",
        type=float,
        nargs="+",
        default=list(DEFAULT_FUDGE_FACTORS),
        help="Inflations of the observed error, one filter each.",
    )
    parser.add_argument(
        "--reduction",
        choices=REDUCTIONS,
        default="blocks",
        help="How the cells are collapsed into the log-likelihood.",
    )
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Draws resolving weights to counts.")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT, help="Bootstrap resamples of the RMSE ranking.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the resampler and the bootstrap.")
    parser.add_argument("RUN_DIR", nargs=1, help="Directory searched recursively for the member dh files.")
    parser.add_argument("OUTPUT_PATH", nargs=1, help="Directory to write the results into.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_path = Path(args.OUTPUT_PATH[0])
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(output_path / "importance_sampling.log")

    summary = run_pipeline(
        args.RUN_DIR[0],
        args.observations,
        output_path,
        pattern=args.pattern,
        relative=args.relative_error,
        floor=args.error_floor,
        fudge_factors=args.fudge_factors,
        reduction=args.reduction,
        n_samples=args.n_samples,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    logger.info("\n%s", summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(main())
