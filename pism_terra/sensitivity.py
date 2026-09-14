"""
Variance-based sensitivity indices of ensemble time series to the UQ parameters.

For every instant of an ensemble time series the members' responses are
regressed against the sampled parameters with SALib's *given data* method
(:func:`SALib.analyze.delta.analyze`, Plischke et al. 2013), which needs no
special sampling design and so works on the Latin-hypercube ensembles
pism-terra runs. It returns, per parameter and instant,

* the first-order Sobol index ``S1``: the share of the response variance
  explained by that parameter alone, and
* Borgonovo's moment-independent ``delta``: the shift of the whole response
  distribution when the parameter is fixed, sensitive to more than variance,

each with a bootstrap confidence half-width (``S1_conf``, ``delta_conf``).

The instants are independent, so they are spread over a process pool.
:func:`sensitivity_indices` is the library entry point; ``pism-sensitivity-indices``
runs it on the processed ``scalar_C_*``/``scalar_G_*`` files of one glacier
complex (one group per glacier in the file), and
``pism-glacier-sensitivity-indices`` over every glacier of a project.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import xarray as xr
from threadpoolctl import threadpool_limits

from pism_terra.processing import preprocess_netcdf

logger = logging.getLogger(__name__)

MEMBER_DIM = "uq_id"
PARAM_DIM = "uq_var"
INDICES = ("S1", "S1_conf", "delta", "delta_conf")
DEFAULT_TARGET = "ice_mass"
DEFAULT_N_RESAMPLES = 100
FREQUENCIES = {"yearly": "YS", "monthly": "MS", "none": None}

rc_params = {
    "axes.linewidth": 0.15,
    "xtick.major.size": 2.0,
    "xtick.major.width": 0.15,
    "ytick.major.size": 2.0,
    "ytick.major.width": 0.15,
    "font.size": 6,
}


def sensitivity_problem(uq_df: pd.DataFrame) -> dict[str, Any]:
    """
    Build the SALib problem definition from the sampled parameters.

    Parameters
    ----------
    uq_df : pandas.DataFrame
        Parameter values per member, one column per parameter.

    Returns
    -------
    dict
        ``num_vars``, ``names`` and ``bounds`` (the sampled range of each column).
    """
    names = [str(c) for c in uq_df.columns]
    lo, hi = uq_df.min().values.astype(float), uq_df.max().values.astype(float)
    return {"num_vars": len(names), "names": names, "bounds": [[float(a), float(b)] for a, b in zip(lo, hi)]}


def delta_indices(
    X: np.ndarray, Y: np.ndarray, problem: dict[str, Any], *, num_resamples: int = DEFAULT_N_RESAMPLES, seed: int = 0
) -> np.ndarray:
    """
    Sobol first-order and delta indices of one response vector.

    Parameters
    ----------
    X : numpy.ndarray
        Parameter matrix, shape ``(n_members, n_vars)``.
    Y : numpy.ndarray
        Response per member, shape ``(n_members,)``; NaN members are dropped.
    problem : dict
        SALib problem (:func:`sensitivity_problem`).
    num_resamples : int, optional
        Bootstrap resamples for the confidence half-widths.
    seed : int, optional
        Seed of the bootstrap.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_vars, 4)`` with columns :data:`INDICES`; all NaN when the
        analysis is impossible (fewer members than parameters plus one, a
        constant response, or a singular estimate).
    """
    # SALib's delta module is imported here so the module loads without it.
    from SALib.analyze import delta  # pylint: disable=import-outside-toplevel

    out = np.full((problem["num_vars"], len(INDICES)), np.nan)
    finite = np.isfinite(Y)
    if finite.sum() <= problem["num_vars"] + 1 or np.nanstd(Y[finite]) == 0:
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = delta.analyze(
                problem, X[finite], Y[finite], num_resamples=num_resamples, seed=seed, print_to_console=False
            )
    except Exception:  # pylint: disable=broad-exception-caught
        return out
    for k, name in enumerate(INDICES):
        out[:, k] = np.asarray(res[name], dtype=float)
    return out


def _single_threaded() -> None:
    """
    Pin the BLAS/OpenMP pools of a worker process to one thread.

    Each instant is a small problem; letting every worker spin up a full
    BLAS pool oversubscribes the machine (12 workers x 12 threads) and
    turns the run into system time.
    """
    threadpool_limits(1)


def _analyze_task(
    item: tuple[np.ndarray, int], X: np.ndarray, problem: dict[str, Any], num_resamples: int
) -> np.ndarray:
    """
    Process-pool task: one response vector with its own bootstrap seed.

    Parameters
    ----------
    item : tuple
        ``(Y, seed)``: the response per member and a non-zero seed.
    X : numpy.ndarray
        Parameter matrix.
    problem : dict
        SALib problem.
    num_resamples : int
        Bootstrap resamples.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_vars, 4)``.
    """
    Y, seed = item
    return delta_indices(X, Y, problem, num_resamples=num_resamples, seed=seed)


def sensitivity_indices(
    response: xr.DataArray,
    uq_df: pd.DataFrame,
    *,
    dim: str = MEMBER_DIM,
    iter_dim: str = "time",
    num_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = 0,
    n_jobs: int | None = None,
) -> xr.Dataset:
    """
    Sensitivity indices of an ensemble response at every instant and group.

    Parameters
    ----------
    response : xarray.DataArray
        Ensemble response with the member dimension ``dim``, the instant
        dimension ``iter_dim`` and any further (group) dimensions.
    uq_df : pandas.DataFrame
        Parameter values indexed by member id; members missing from either
        side are dropped.
    dim : str, optional
        Member dimension.
    iter_dim : str, optional
        Dimension the indices are computed along, typically ``time``.
    num_resamples : int, optional
        Bootstrap resamples for the confidence half-widths.
    seed : int, optional
        Seed from which one bootstrap seed per instant is derived, so the
        confidence bands are reproducible and independent between instants.
    n_jobs : int or None, optional
        Worker processes; ``None`` uses every CPU, ``1`` runs in this process.

    Returns
    -------
    xarray.Dataset
        ``S1``, ``S1_conf``, ``delta`` and ``delta_conf`` on the group
        dimensions, ``iter_dim`` and ``uq_var``; attrs record the member count.

    Raises
    ------
    ValueError
        If no member of ``response`` has parameters in ``uq_df``.
    """
    members = [m for m in response[dim].values if str(m) in uq_df.index]
    if not members:
        raise ValueError(f"no member of the ensemble ({dim}) is listed in the parameter table")
    response = response.sel({dim: members}).load()
    X = uq_df.loc[[str(m) for m in members]].to_numpy(dtype=float)
    problem = sensitivity_problem(uq_df)

    other = [d for d in response.dims if d not in (dim, iter_dim)]
    response = response.transpose(*other, iter_dim, dim)
    shape = tuple(response.sizes[d] for d in other) + (response.sizes[iter_dim],)
    Y = np.asarray(response.values, dtype=float).reshape(-1, len(members))
    # One independent, non-zero seed per instant: SALib treats seed=0 as
    # "do not seed", which would make the bootstrap bands irreproducible.
    seeds = (np.random.SeedSequence(seed).generate_state(Y.shape[0]) % (2**31 - 1) + 1).astype(int)
    items = list(zip(Y, seeds.tolist()))
    task = partial(_analyze_task, X=X, problem=problem, num_resamples=num_resamples)
    n_jobs = os.cpu_count() or 1 if n_jobs is None else max(int(n_jobs), 1)
    logger.info(
        "sensitivity indices: %d members, %d parameters, %d responses on %d worker(s)",
        len(members),
        problem["num_vars"],
        Y.shape[0],
        n_jobs,
    )
    if n_jobs == 1 or Y.shape[0] < 4:
        with threadpool_limits(1):
            results = [task(item) for item in items]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_single_threaded) as pool:
            results = list(pool.map(task, items, chunksize=max(1, len(items) // (4 * n_jobs))))
    stacked = np.stack(results).reshape(shape + (problem["num_vars"], len(INDICES)))

    coords = {d: response[d] for d in other + [iter_dim]}
    for c in response.coords:
        if c not in coords and set(response[c].dims) <= set(other + [iter_dim]):
            coords[c] = response[c]
    coords[PARAM_DIM] = problem["names"]
    ds = xr.Dataset(
        {name: (other + [iter_dim, PARAM_DIM], stacked[..., k]) for k, name in enumerate(INDICES)}, coords=coords
    )
    ds["S1"].attrs.update({"long_name": "first-order Sobol index", "units": "1"})
    ds["delta"].attrs.update({"long_name": "Borgonovo delta index", "units": "1"})
    ds.attrs.update({"n_members": len(members), "num_resamples": num_resamples, "response": str(response.name or "")})
    return ds


def resample_response(da: xr.DataArray, freq: str = "yearly") -> xr.DataArray:
    """
    Average a time series to the requested frequency.

    Parameters
    ----------
    da : xarray.DataArray
        Series with a ``time`` dimension.
    freq : {"yearly", "monthly", "none"}, optional
        Target frequency; ``"none"`` leaves the series as it is.

    Returns
    -------
    xarray.DataArray
        The averaged series (attributes kept).

    Raises
    ------
    ValueError
        If ``freq`` is unknown.
    """
    if freq not in FREQUENCIES:
        raise ValueError(f"freq must be one of {sorted(FREQUENCIES)}, got {freq!r}")
    rule = FREQUENCIES[freq]
    if rule is None or "time" not in da.dims:
        return da
    with xr.set_options(keep_attrs=True):
        return da.resample(time=rule).mean()


def short_labels(columns: Iterable[str]) -> dict[str, str]:
    """
    Abbreviate dotted PISM parameter names to their last component.

    Parameters
    ----------
    columns : iterable of str
        Parameter names such as ``surface.pdd.factor_ice``.

    Returns
    -------
    dict
        ``{"surface.pdd.factor_ice": "factor_ice", ...}``; a suffix that is
        not unique keeps one more component.
    """
    columns = list(columns)
    labels = {c: c.split(".")[-1] for c in columns}
    seen: dict[str, int] = {}
    for c in columns:
        seen[labels[c]] = seen.get(labels[c], 0) + 1
    return {c: (".".join(c.split(".")[-2:]) if seen[labels[c]] > 1 else labels[c]) for c in columns}


def plot_sensitivity_indices(ds: xr.Dataset, filename: Path | str, *, title: str | None = None) -> None:
    """
    Plot the Sobol and delta indices of one group as time series with confidence bands.

    Parameters
    ----------
    ds : xarray.Dataset
        Indices on ``(time, uq_var)`` (one group already selected).
    filename : Path or str
        Output figure.
    title : str or None, optional
        Figure title.
    """
    labels = short_labels(ds[PARAM_DIM].values)
    with mpl.rc_context(rc=rc_params):
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6.2, 2.2), layout="constrained")
        for ax, name, label in zip(axes, ("S1", "delta"), ("Sobol first-order index $S_1$", "Borgonovo $\\delta$")):
            for k, param in enumerate(ds[PARAM_DIM].values):
                v = ds[name].sel({PARAM_DIM: param})
                c = ds[f"{name}_conf"].sel({PARAM_DIM: param})
                color = f"C{k % 10}"
                ax.fill_between(v.time.values, (v - c).values, (v + c).values, color=color, alpha=0.2, lw=0)
                ax.plot(v.time.values, v.values, color=color, lw=0.8, label=labels[str(param)])
            ax.set_title(label)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0, color="0.5", lw=0.3)
        axes[0].set_ylabel("index")
        axes[0].legend(fontsize=5, frameon=False, ncol=2)
        if title:
            fig.suptitle(title, fontsize=7)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=300)
        plt.close(fig)


def summarize_indices(ds: xr.Dataset, group_dim: str | None, *, last_years: int = 10) -> pd.DataFrame:
    """
    Tabulate time-mean indices per group and parameter.

    Parameters
    ----------
    ds : xarray.Dataset
        Output of :func:`sensitivity_indices`.
    group_dim : str or None
        Group dimension (e.g. ``glacier``), or ``None`` when there is none.
    last_years : int, optional
        Length of the closing window averaged into the ``*_last`` columns.

    Returns
    -------
    pandas.DataFrame
        One row per group and parameter: ``S1_mean``, ``delta_mean`` over all
        instants, ``S1_last``, ``delta_last`` over the last ``last_years`` instants.
    """
    n = min(last_years, ds.sizes["time"])
    rows = []
    groups = ds[group_dim].values if group_dim else [None]
    for g in groups:
        sub = ds.sel({group_dim: g}) if group_dim else ds
        for param in sub[PARAM_DIM].values:
            s = sub.sel({PARAM_DIM: param})
            row: dict[str, Any] = {"parameter": str(param)}
            if group_dim:
                row[group_dim] = g
            for name in ("S1", "delta"):
                row[f"{name}_mean"] = float(s[name].mean("time"))
                row[f"{name}_last"] = float(s[name].isel(time=slice(-n, None)).mean("time"))
            rows.append(row)
    return pd.DataFrame(rows)


# --- Processed scalar files of one glacier complex ---------------------------------


def load_scalar_ensemble(files: Sequence[Path | str], target: str = DEFAULT_TARGET) -> xr.DataArray:
    """
    Open post-processed scalar files as one ensemble of the target variable.

    Parameters
    ----------
    files : sequence of Path
        ``scalar_C_*`` or ``scalar_G_*`` files of one glacier complex, one per
        member (``_uq_<n>_`` in the name).
    target : str, optional
        Variable to analyze.

    Returns
    -------
    xarray.DataArray
        ``target`` on ``(uq_id, time, glacier)``; the ``glacier`` coordinate
        holds the RGI ids of the file's ``glacier_id_name``.

    Raises
    ------
    KeyError
        If ``target`` is not in the files.
    """
    ds = xr.open_mfdataset(
        [str(f) for f in files],
        preprocess=partial(preprocess_netcdf, process_config=False),
        parallel=True,
        join="outer",
        compat="no_conflicts",
    )
    if target not in ds:
        raise KeyError(f"{target!r} is not in {Path(files[0]).name}; available: {sorted(ds.data_vars)}")
    da = ds[target]
    for d in ("rgi_id", "exp_id"):
        if d in da.dims and da.sizes[d] == 1:
            da = da.squeeze(d, drop=True)
    if "glacier_id_name" in ds.coords and "glacier_id" in da.dims:
        names = ds["glacier_id_name"]
        if names.ndim > 1:
            names = names.isel({d: 0 for d in names.dims if d != "glacier_id"})
        da = da.assign_coords(glacier_id=names.values.astype(str)).rename({"glacier_id": "glacier"})
    da[MEMBER_DIM] = da[MEMBER_DIM].astype(str)
    return da


def find_uq_csv(files: Sequence[Path | str]) -> Path:
    """
    Find the ``uq.csv`` that belongs to a set of processed scalar files.

    Parameters
    ----------
    files : sequence of Path
        The files; their parents are searched upwards.

    Returns
    -------
    Path
        The first ``uq.csv`` found.

    Raises
    ------
    FileNotFoundError
        If none of the parent directories holds one.
    """
    for parent in Path(files[0]).resolve().parents:
        for candidate in (parent / "uq.csv", parent / "output" / "uq.csv"):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"no uq.csv above {files[0]}")


def load_uq_csv(path: Path | str) -> pd.DataFrame:
    """
    Read a run's ``uq.csv`` as parameters indexed by member id.

    Parameters
    ----------
    path : Path or str
        The file written by the run generator.

    Returns
    -------
    pandas.DataFrame
        Parameter columns indexed by ``uq_id`` (strings).
    """
    df = pd.read_csv(path)
    df["uq"] = df["uq"].astype(str)
    return df.set_index("uq").rename_axis(MEMBER_DIM)


def analyze_scalar_files(
    files: Sequence[Path | str],
    uq_df: pd.DataFrame,
    *,
    output_dir: Path | str,
    target: str = DEFAULT_TARGET,
    stem: str = "sensitivity",
    freq: str = "yearly",
    num_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = 0,
    n_jobs: int | None = None,
    title_prefix: str = "",
) -> tuple[xr.Dataset, pd.DataFrame]:
    """
    Sensitivity indices of one glacier complex's scalar files, with outputs.

    Parameters
    ----------
    files : sequence of Path
        Processed scalar files, one per member.
    uq_df : pandas.DataFrame
        Parameters per member.
    output_dir : Path or str
        Where the NetCDF, CSV and one figure per glacier go.
    target : str, optional
        Variable to analyze.
    stem : str, optional
        File-name stem of the outputs.
    freq : str, optional
        Frequency the series is averaged to first (:func:`resample_response`).
    num_resamples : int, optional
        Bootstrap resamples.
    seed : int, optional
        Bootstrap seed.
    n_jobs : int or None, optional
        Worker processes.
    title_prefix : str, optional
        Prefix of the figure titles.

    Returns
    -------
    xarray.Dataset
        Indices on ``(glacier, time, uq_var)``.
    pandas.DataFrame
        :func:`summarize_indices` of it.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    response = resample_response(load_scalar_ensemble(files, target), freq)
    ds = sensitivity_indices(response, uq_df, num_resamples=num_resamples, seed=seed, n_jobs=n_jobs)
    ds.attrs.update({"target": target, "frequency": freq})
    ds.to_netcdf(output_dir / f"{stem}_{target}.nc")
    group = "glacier" if "glacier" in ds.dims else None
    summary = summarize_indices(ds, group)
    summary.to_csv(output_dir / f"{stem}_{target}.csv", index=False)
    for g in ds[group].values if group else [None]:
        sub = ds.sel({group: g}) if group else ds
        name = str(g) if g is not None else "all"
        plot_sensitivity_indices(
            sub, output_dir / f"{stem}_{target}_{name}.png", title=f"{title_prefix}{name}: sensitivity of {target}"
        )
    return ds, summary


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Command-line entry point of ``pism-sensitivity-indices``.

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
        description="Sobol and delta sensitivity indices of a scalar time series to the UQ parameters, "
        "for every glacier in processed scalar_C_* or scalar_G_* files of one complex.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("FILES", nargs="+", help="Processed scalar files, one per ensemble member (_uq_<n>_).")
    parser.add_argument(
        "--uq-csv", default=None, help="Parameter table of the members; found above the files by default."
    )
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Variable to analyze.")
    parser.add_argument("--output-path", default=".", help="Directory for the NetCDF, CSV and figures.")
    parser.add_argument(
        "--freq", choices=sorted(FREQUENCIES), default="yearly", help="Average the series to this frequency first."
    )
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES, help="Bootstrap resamples per instant.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Worker processes; default all CPUs.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    files = [Path(f) for f in args.FILES]
    uq_df = load_uq_csv(args.uq_csv or find_uq_csv(files))
    match = re.search(r"scalar_([CG])_", files[0].name)
    stem = f"sensitivity_{match.group(1)}" if match else "sensitivity"
    _, summary = analyze_scalar_files(
        files,
        uq_df,
        output_dir=args.output_path,
        target=args.target,
        stem=stem,
        freq=args.freq,
        num_resamples=args.n_resamples,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    return summary


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
