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
Annual and cumulative glacier-wide balances of the USGS benchmark glaciers.

Reproduces Fig. 2 of the USGS *2025 Benchmark Glaciers Executive Summary*
(https://www.usgs.gov/programs/ecosystems-land-change-science-program/science/2025-usgs-benchmark-glaciers-executive),
one figure per glacier instead of one stacked panel each: the glacier-wide
annual balance ``Ba`` as bars coloured by sign and magnitude (left axis),
the cumulative balance as a black line (right axis), and the latest year's
balance, its rank (highest to lowest) and percentile (share of years with a
more negative balance). The data are the ScienceBase release
https://doi.org/10.5066/F7HD7SRF, fetched by
:func:`pism_terra.download.download_usgs_benchmark`. Only the observations
are drawn, so no RGI matching is needed and every glacier with a
glacier-wide solution is plotted, including the two outside RGI region 1.
"""

import logging
import math
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import xarray as xr

from pism_terra.download import download_usgs_benchmark
from pism_terra.glacier.usgs import (
    DEFAULT_DATA_DIR,
    apply_plot_years,
    load_glacier_wide,
    plot_year,
)
from pism_terra.log import setup_logging

logger = logging.getLogger("pism_terra.glacier.usgs_benchmark_annual")

#: Bars are shaded on a diverging map clipped at this magnitude (m w.e.), so
#: near-zero years fade out and the extremes saturate, as in the original.
BAR_CMAP = "RdBu"
BAR_CLIP = 2.0
#: Cumulative balance is read on a right axis this many times the left one,
#: which keeps the two zeros level (±4 and ±60 m w.e. in the original).
CUMULATIVE_SCALE = 15.0
MIN_BALANCE_LIMIT = 4.0
SUMMARY_COLUMNS = ["glacier", "year", "Ba", "rank", "n_years", "percentile", "cumulative", "first_year"]


def display_name(glacier: str) -> str:
    """
    Turn the release's directory name into the name printed on the figure.

    Parameters
    ----------
    glacier : str
        Name as used in the release (``LemonCreek``).

    Returns
    -------
    str
        The name with its words apart (``Lemon Creek``).
    """
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", glacier)


def annual_summary(ba: pd.Series) -> dict[str, float | int]:
    """
    Rank the latest balance year against the record.

    Parameters
    ----------
    ba : pandas.Series
        Annual balances (m w.e.) indexed by balance year; missing years are
        dropped.

    Returns
    -------
    dict
        ``year`` (the latest), its balance ``Ba``, ``rank`` from highest to
        lowest balance (ties share the better rank), ``n_years`` measured,
        ``percentile`` — the share of the *other* years ranked below it,
        ``(n_years - rank) / (n_years - 1)``, as USGS reports it, so a record
        low is 0 and a record high 100 — ``cumulative`` balance over the
        record and its ``first_year``.
    """
    ba = ba.dropna().sort_index()
    if ba.empty:
        raise ValueError("no annual balances")
    year = int(ba.index[-1])
    value = float(ba.iloc[-1])
    n = int(ba.size)
    rank = int((ba > value).sum()) + 1
    return {
        "year": year,
        "Ba": value,
        "rank": rank,
        "n_years": n,
        "percentile": 100.0 * (n - rank) / (n - 1) if n > 1 else np.nan,
        "cumulative": float(ba.sum()),
        "first_year": int(ba.index[0]),
    }


def balance_limit(series: Iterable[pd.Series]) -> float:
    """
    Symmetric left-axis limit that holds every balance.

    Parameters
    ----------
    series : iterable of pandas.Series
        Annual balances and cumulative balances divided by
        :data:`CUMULATIVE_SCALE`, of every glacier to be drawn.

    Returns
    -------
    float
        At least :data:`MIN_BALANCE_LIMIT`, rounded up to a whole metre.
    """
    peak = max((float(np.nanmax(np.abs(s.to_numpy()))) for s in series if s.notna().any()), default=0.0)
    return float(max(MIN_BALANCE_LIMIT, math.ceil(peak)))


def plot_annual_balance(
    obs: xr.Dataset,
    glacier: str,
    output_dir: Path | str,
    *,
    ylim: float = MIN_BALANCE_LIMIT,
    plot_years: tuple[float | None, float | None] = (None, None),
) -> tuple[Path, dict[str, float | int]]:
    """
    Draw one glacier's annual balance bars and cumulative balance.

    Parameters
    ----------
    obs : xarray.Dataset
        Balances from :func:`pism_terra.glacier.usgs.load_glacier_wide`.
    glacier : str
        Glacier name as used in the release.
    output_dir : Path or str
        Directory the figure is written to.
    ylim : float, optional
        Half-range of the left axis in m w.e.; the right axis spans
        :data:`CUMULATIVE_SCALE` times as much.
    plot_years : tuple, optional
        ``(left, right)`` balance-year limits from
        :func:`pism_terra.glacier.usgs.plot_year`.

    Returns
    -------
    tuple
        The PNG written (a PDF sits next to it) and the
        :func:`annual_summary` of the record.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ba = obs["Ba"].to_series().dropna().sort_index()
    summary = annual_summary(ba)
    years = ba.index.to_numpy(dtype=float)
    # A bar is centred on its balance year; the cumulative line steps from
    # the start of the first year to the end of each year.
    line_x = np.concatenate([[years[0] - 0.5], years + 0.5])
    line_y = np.concatenate([[0.0], ba.cumsum().to_numpy()])

    norm = mpl.colors.Normalize(vmin=-BAR_CLIP, vmax=BAR_CLIP)
    colors = mpl.colormaps[BAR_CMAP](norm(ba.to_numpy()))

    # Bold is set per text rather than via ``font.weight``, which mathtext
    # cannot honour and warns about for every label.
    with mpl.rc_context(rc={"font.family": "DejaVu Sans", "font.size": 9}):
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.0), layout="constrained")
        ax.bar(years, ba.to_numpy(), width=0.9, color=colors, alpha=0.8, lw=0)
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_ylim(-ylim, ylim)
        ax.set_ylabel("$B_a$ (m w.e.)", fontweight="bold")
        ax.set_xlabel("Year", fontweight="bold")

        cumulative = ax.twinx()
        cumulative.plot(line_x, line_y, color="k", lw=2.5, alpha=0.75, solid_capstyle="round")
        cumulative.set_ylim(-ylim * CUMULATIVE_SCALE, ylim * CUMULATIVE_SCALE)
        cumulative.set_ylabel("Cumulative $B_a$ (m w.e.)", fontweight="bold")

        apply_plot_years(ax, plot_years)
        for axis in (ax, cumulative):
            axis.spines["top"].set_visible(False)
            for label in axis.get_xticklabels() + axis.get_yticklabels():
                label.set_fontweight("bold")

        ax.text(0.05, 0.82, display_name(glacier), transform=ax.transAxes, fontsize=16, fontweight="bold")
        ax.text(
            0.05,
            0.04,
            f"{summary['year']}: $B_a$ = {summary['Ba']:.2f} m w.e.\n"
            f"Rank {summary['rank']} of {summary['n_years']} years ({summary['percentile']:.0f} percentile)\n"
            f"Total mass change: {summary['cumulative']:.1f} m w.e. over {summary['n_years']} yrs",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

        stem = f"usgs_annual_balance_{glacier}"
        png = output_dir / f"{stem}.png"
        fig.savefig(png, dpi=300)
        fig.savefig(output_dir / f"{stem}.pdf")
        plt.close(fig)
    return png, summary


def run_pipeline(
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    output_dir: Path | str = ".",
    glaciers: Sequence[str] | None = None,
    force_overwrite: bool = False,
    plot_years: tuple[float | None, float | None] = (None, None),
) -> pd.DataFrame:
    """
    Download the release and draw every glacier's annual balance figure.

    All figures share their axis ranges, as the stacked panels of the
    original do, so they can be compared side by side.

    Parameters
    ----------
    data_dir : Path or str, optional
        Cache for the ScienceBase archives.
    output_dir : Path or str, optional
        Where the figures and the summary table go.
    glaciers : sequence of str or None, optional
        Glaciers to draw, by the release's name for them; None draws every
        glacier with a glacier-wide solution.
    force_overwrite : bool, default False
        Re-download the archives.
    plot_years : tuple, optional
        ``(left, right)`` balance-year limits; a None end is set to cover
        every glacier's record.

    Returns
    -------
    pandas.DataFrame
        One row per glacier: :data:`SUMMARY_COLUMNS` and ``figure``.
    """
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = download_usgs_benchmark(Path(data_dir).expanduser(), force_overwrite=force_overwrite)["data"]

    names = sorted(p.name for p in data_path.iterdir() if p.is_dir()) if glaciers is None else list(glaciers)
    observations: dict[str, xr.Dataset] = {}
    for glacier in names:
        # The area is not used here, so any fallback will do for glaciers
        # without an area-altitude distribution.
        obs = load_glacier_wide(data_path, glacier, fallback_area_km2=np.nan)
        if obs is None or "Ba" not in obs or not obs["Ba"].notnull().any():
            logger.info("%s: no glacier-wide annual balances; skipping", glacier)
            continue
        observations[glacier] = obs
    if not observations:
        raise FileNotFoundError(f"No glacier-wide annual balances found in {data_path}")

    ba_series = [obs["Ba"].to_series() for obs in observations.values()]
    ylim = balance_limit(ba_series + [s.dropna().cumsum() / CUMULATIVE_SCALE for s in ba_series])
    first = min(int(s.dropna().index.min()) for s in ba_series)
    last = max(int(s.dropna().index.max()) for s in ba_series)
    left, right = plot_years
    plot_years = (
        left if left is not None else 10.0 * math.floor((first - 1) / 10.0),
        right if right is not None else 10.0 * math.ceil((last + 1) / 10.0),
    )

    rows = []
    for glacier, obs in observations.items():
        png, summary = plot_annual_balance(obs, glacier, output_dir, ylim=ylim, plot_years=plot_years)
        logger.info(
            "%s %d: Ba=%.2f m w.e., rank %d of %d (%.0f percentile), cumulative %.1f m w.e.",
            glacier,
            summary["year"],
            summary["Ba"],
            summary["rank"],
            summary["n_years"],
            summary["percentile"],
            summary["cumulative"],
        )
        rows.append({"glacier": glacier, **summary, "figure": str(png)})
    table = pd.DataFrame(rows, columns=[*SUMMARY_COLUMNS, "figure"])
    table.to_csv(output_dir / "usgs_annual_balance_summary.csv", index=False)
    return table


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Plot the annual and cumulative balances of the USGS benchmark glaciers.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments (excluding the program name). When
        ``None``, :data:`sys.argv` is used.

    Returns
    -------
    pandas.DataFrame
        The per-glacier summary table.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = (
        "Plot USGS benchmark-glacier annual and cumulative glacier-wide balances, one figure per glacier "
        "(Fig. 2 of the USGS Benchmark Glaciers Executive Summary)."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_DIR, help="Cache for the USGS archives.")
    parser.add_argument("--output-path", default=".", help="Directory for the figures and the summary table.")
    parser.add_argument(
        "--glaciers",
        nargs="+",
        default=None,
        help="Glaciers to plot, by the release's names (e.g. Gulkana LemonCreek). Default: all.",
    )
    parser.add_argument(
        "--plot-start",
        "--plot_start",
        default=None,
        help="Left end of the year axis, as a date (1950-01-01) or a year (1950). Default: covers every glacier.",
    )
    parser.add_argument(
        "--plot-end",
        "--plot_end",
        default=None,
        help="Right end of the year axis, as a date or a year. Default: covers every glacier.",
    )
    parser.add_argument("--force-overwrite", action="store_true", default=False, help="Re-download the archives.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = Path(args.output_path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "usgs_annual_balance.log")

    table = run_pipeline(
        data_dir=args.data_path,
        output_dir=output_dir,
        glaciers=args.glaciers,
        force_overwrite=args.force_overwrite,
        plot_years=(plot_year(args.plot_start), plot_year(args.plot_end)),
    )
    print(table.drop(columns="figure").to_string(index=False))
    return table


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
