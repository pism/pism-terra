"""
Sensitivity indices of every glacier UQ ensemble of a project.

For each glacier of a project directory that has post-processed scalar files
(``output/processed_scalar/scalar_C_*`` for the complex, ``scalar_G_*`` for its
individual glaciers, written by ``pism-postprocess-scalar``), the target
variable's time series is analyzed with :func:`pism_terra.sensitivity.sensitivity_indices`:
first-order Sobol and Borgonovo delta indices of the UQ parameters (read from
the run's ``uq.csv``) at every instant of the yearly-averaged series.

Outputs mirror ``pism-glacier-importance-sampling``: each glacier's NetCDF,
CSV and figures go to ``<output-path>/<rgi_id>/``, one figure per glacier of
the file, and ``sensitivity_indices_summary.csv`` at the top level.
"""

from __future__ import annotations

import logging
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from pism_terra.glacier.importance_sampling import load_uq_parameters, setup_logging
from pism_terra.glacier.usgs import find_model_files, rgi_output_dir
from pism_terra.sensitivity import (
    DEFAULT_N_RESAMPLES,
    DEFAULT_TARGET,
    FREQUENCIES,
    analyze_scalar_files,
)

logger = logging.getLogger(__name__)

RGI_PATTERN = re.compile(r"(RGI2000-v7\.0-[A-Z]-\d{2}-\d+)")
KINDS = ("C", "G")


def find_scalar_files(run_dir: Path | str, kind: str) -> dict[str, list[Path]]:
    """
    Group a project's processed scalar files of one kind by glacier complex.

    Parameters
    ----------
    run_dir : Path or str
        Project directory searched recursively.
    kind : {"C", "G"}
        ``scalar_C_*`` (complex totals) or ``scalar_G_*`` (per glacier).

    Returns
    -------
    dict
        ``{rgi_id: [files]}``, sorted.
    """
    groups: dict[str, list[Path]] = {}
    for file in find_model_files(run_dir, pattern=f"scalar_{kind}_*_id_*_uq_*.nc"):
        match = RGI_PATTERN.search(file.name)
        if match:
            groups.setdefault(match.group(1), []).append(file)
    return dict(sorted(groups.items()))


def run_pipeline(
    run_dir: Path | str,
    *,
    output_path: Path | str = ".",
    kinds: Sequence[str] = KINDS,
    target: str = DEFAULT_TARGET,
    freq: str = "yearly",
    num_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = 0,
    n_jobs: int | None = None,
    min_members: int = 8,
) -> pd.DataFrame:
    """
    Compute the sensitivity indices of every glacier complex of a project.

    Parameters
    ----------
    run_dir : Path or str
        Project directory searched recursively for processed scalar files.
    output_path : Path or str, optional
        Where the summary goes; each complex writes to ``<rgi_id>/`` below it.
    kinds : sequence of str, optional
        Which files to analyze: ``C`` (complex totals) and/or ``G`` (per glacier).
    target : str, optional
        Variable to analyze.
    freq : str, optional
        Frequency the series is averaged to first.
    num_resamples : int, optional
        Bootstrap resamples per instant.
    seed : int, optional
        Bootstrap seed.
    n_jobs : int or None, optional
        Worker processes; default all CPUs.
    min_members : int, optional
        Complexes with fewer finished members are skipped: the given-data
        estimator needs comfortably more members than parameters.

    Returns
    -------
    pandas.DataFrame
        Summary: one row per complex, kind, glacier and parameter.

    Raises
    ------
    FileNotFoundError
        If no complex has enough processed members.
    """
    run_dir = Path(run_dir).expanduser()
    output_path = Path(output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    summaries = []
    for kind in kinds:
        groups = find_scalar_files(run_dir, kind)
        logger.info("%d complex(es) with scalar_%s files below %s", len(groups), kind, run_dir)
        for rgi_id, files in groups.items():
            if len(files) < min_members:
                logger.warning(
                    "%s/%s: only %d member(s), fewer than %d, skipped", rgi_id, kind, len(files), min_members
                )
                continue
            try:
                uq_df = load_uq_parameters(run_dir, rgi_id)
            except FileNotFoundError as err:
                logger.warning("%s: %s, skipped", rgi_id, err)
                continue
            logger.info("%s/%s: %d members", rgi_id, kind, len(files))
            _, summary = analyze_scalar_files(
                files,
                uq_df,
                output_dir=rgi_output_dir(output_path, rgi_id),
                target=target,
                stem=f"sensitivity_{kind}",
                freq=freq,
                num_resamples=num_resamples,
                seed=seed,
                n_jobs=n_jobs,
                title_prefix=f"{rgi_id} ({kind}) ",
            )
            summary.insert(0, "kind", kind)
            summary.insert(0, "rgi_id", rgi_id)
            summaries.append(summary)
    if not summaries:
        raise FileNotFoundError(f"no complex below {run_dir} has {min_members} or more processed members")
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output_path / "sensitivity_indices_summary.csv", index=False)
    return summary


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
        description="Sobol and delta sensitivity indices of every glacier UQ ensemble of a project.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("RUN_DIR", help="Project directory searched recursively for processed scalar files.")
    parser.add_argument(
        "--output-path", default=".", help="Directory for the summary; each complex writes to <rgi_id>/ below it."
    )
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Variable to analyze.")
    parser.add_argument(
        "--kind",
        action="append",
        choices=KINDS,
        default=None,
        help="C (complex) and/or G (per glacier); repeatable. Default both.",
    )
    parser.add_argument(
        "--freq", choices=sorted(FREQUENCIES), default="yearly", help="Average the series to this frequency first."
    )
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES, help="Bootstrap resamples per instant.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Worker processes; default all CPUs.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    parser.add_argument(
        "--min-members", type=int, default=8, help="Skip complexes with fewer processed members than this."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_path = Path(args.output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(output_path / "sensitivity_indices.log")
    return run_pipeline(
        args.RUN_DIR,
        output_path=output_path,
        kinds=tuple(args.kind) if args.kind else KINDS,
        target=args.target,
        freq=args.freq,
        num_resamples=args.n_resamples,
        seed=args.seed,
        n_jobs=args.n_jobs,
        min_members=args.min_members,
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
