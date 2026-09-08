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
L-curve analysis of a PISM Tikhonov inversion ensemble.

Each member of an inversion ensemble is run with a different regularization
setting (usually ``inverse.tikhonov.penalty_weight``) and trades data fit
against model roughness: weak regularization fits the observed velocities but
produces a noisy design variable, strong regularization produces a smooth
field that misses the data. Plotting the model norm
``N = sqrt(J_design)`` against the data misfit ``M`` — the
misfit-weight-weighted RMS of ``inv_residual``, in m/yr — gives the familiar
L-shaped curve, and the corner of that L (maximum Menger curvature in log-log
space) is the conventional pick for the regularization parameter.

The tool reads the ensemble members named on the command line, tabulates the
regularization parameters straight out of each file's ``pism_config``
attributes, and writes the plot plus the underlying table:

```bash
pism-inverse-lcurve --parameters inverse.tikhonov.penalty_weight \
    -o lcurve.png inv_g*.nc
```

Members that are still running, or that crashed before writing the inversion
diagnostics, lack ``J_design``/``inv_residual`` and are skipped with a
warning rather than aborting the analysis.
"""

from __future__ import annotations

import logging
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import xarray as xr

from pism_terra.log import setup_logging

xr.set_options(keep_attrs=True)
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)

logger = logging.getLogger("pism_terra.lcurve")

DEFAULT_PARAMETERS = ["inverse.tikhonov.penalty_weight"]

REQUIRED_VARS = ("vel_misfit_weight", "inv_residual", "J_design")

fontsize = 6
rc_params = {
    "axes.linewidth": 0.15,
    "xtick.major.size": 2.0,
    "xtick.major.width": 0.15,
    "ytick.major.size": 2.0,
    "ytick.major.width": 0.15,
    "hatch.linewidth": 0.15,
    "font.size": fontsize,
    "font.family": "DejaVu Sans",
}


def short_name(parameter: str) -> str:
    """
    Column name used for a dotted ``pism_config`` parameter.

    Parameters
    ----------
    parameter : str
        Dotted configuration key, e.g. ``"inverse.tikhonov.penalty_weight"``.

    Returns
    -------
    str
        The last dotted component, e.g. ``"penalty_weight"``.
    """
    return parameter.split(".")[-1]


def data_misfit(ds: xr.Dataset) -> float:
    """
    Weighted RMS velocity misfit of an inversion, in m/yr.

    The residual ``inv_residual`` is averaged over the misfit area Omega with
    PISM's own ``vel_misfit_weight`` as the weight, so cells outside the
    observed-velocity mask do not dilute the average.

    Parameters
    ----------
    ds : xarray.Dataset
        Inversion output carrying ``vel_misfit_weight`` and ``inv_residual``.

    Returns
    -------
    float
        Physical RMS misfit ``sqrt(sum(w r^2) / sum(w))`` in m/yr.
    """
    weight = ds["vel_misfit_weight"].squeeze().values.astype(float)
    residual = ds["inv_residual"].squeeze().values.astype(float)
    return float(np.sqrt(np.nansum(weight * residual**2) / np.nansum(weight)))


def model_norm(ds: xr.Dataset) -> float:
    """
    Model norm ``N = sqrt(J_design)`` at the last inversion iteration.

    Parameters
    ----------
    ds : xarray.Dataset
        Inversion output carrying the ``J_design`` iteration history.

    Returns
    -------
    float
        Square root of the design functional of the converged solution.
    """
    return float(np.sqrt(ds["J_design"].isel(inv_iter=-1)))


def collect_lcurve(files: list[Path], parameters: list[str]) -> pd.DataFrame:
    """
    Tabulate misfit, model norm and regularization parameters of an ensemble.

    Files that do not carry the full set of inversion diagnostics — members
    still running, or crashed before the diagnostics were written — are
    skipped with a warning, as are files missing one of ``parameters`` in
    their ``pism_config`` attributes.

    Parameters
    ----------
    files : list of pathlib.Path
        Inversion output files, one per ensemble member.
    parameters : list of str
        Dotted ``pism_config`` keys to read from each file; they become
        columns named after their last dotted component.

    Returns
    -------
    pandas.DataFrame
        One row per usable file with the ``parameters`` columns plus ``M``
        (data misfit, m/yr), ``N`` (model norm) and ``file``, sorted by the
        parameter columns.

    Raises
    ------
    ValueError
        If no file yields a complete row.
    """
    columns = [short_name(p) for p in parameters]
    rows: list[dict[str, Any]] = []
    for path in files:
        try:
            with xr.open_dataset(path) as ds:
                missing = [v for v in REQUIRED_VARS if v not in ds]
                if missing:
                    logger.warning("%s: skipped, missing %s", path.name, ", ".join(missing))
                    continue
                config = ds["pism_config"].attrs
                missing = [p for p in parameters if p not in config]
                if missing:
                    logger.warning("%s: skipped, pism_config has no %s", path.name, ", ".join(missing))
                    continue
                row: dict[str, Any] = {c: float(config[p]) for c, p in zip(columns, parameters)}
                row["M"] = data_misfit(ds)
                row["N"] = model_norm(ds)
                row["file"] = path.name
                rows.append(row)
        except (OSError, KeyError, ValueError) as error:
            logger.warning("%s: skipped, %s", path.name, error)

    if not rows:
        raise ValueError(
            f"none of the {len(files)} given files yielded an L-curve point; "
            f"they need {', '.join(REQUIRED_VARS)} and {', '.join(parameters)}"
        )
    logger.info("collected %d of %d files", len(rows), len(files))
    return pd.DataFrame(rows).sort_values(columns).reset_index(drop=True)


def corner(norm: np.ndarray, misfit: np.ndarray) -> int | None:
    """
    Index of maximum Menger curvature on a (log N, log M) L-curve.

    The curvature of the circle through three consecutive points is
    ``k = 4 A / (a b c)`` with ``A`` the triangle area and ``a``, ``b``, ``c``
    its side lengths; the point where it peaks is the corner of the L.

    Parameters
    ----------
    norm : numpy.ndarray
        Model norms ``N``, ordered along the curve.
    misfit : numpy.ndarray
        Data misfits ``M``, ordered along the curve.

    Returns
    -------
    int or None
        Position of the corner, or ``None`` for fewer than three points (no
        interior point has two neighbours).
    """
    x, y = np.log10(norm), np.log10(misfit)
    if len(x) < 3:
        return None
    k_best, best = -np.inf, None
    for i in range(1, len(x) - 1):
        a = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
        b = np.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
        c = np.hypot(x[i + 1] - x[i - 1], y[i + 1] - y[i - 1])
        area = abs((x[i] - x[i - 1]) * (y[i + 1] - y[i - 1]) - (x[i + 1] - x[i - 1]) * (y[i] - y[i - 1]))
        k = 2 * area / (a * b * c) if a * b * c > 0 else 0.0
        if k > k_best:
            k_best, best = k, i
    return best


def plot_lcurve(
    df: pd.DataFrame,
    parameters: list[str],
    output_file: Path,
    log: bool = False,
    dpi: int = 300,
) -> pd.DataFrame:
    """
    Plot the L-curve of an inversion ensemble and mark its corner.

    Points are joined in order of the *first* parameter, whose value labels
    each marker. Any further parameters split the ensemble into one curve per
    combination of their values, drawn with a legend, so a 2-D sweep stays
    readable. The corner of every curve is marked with a star.

    Parameters
    ----------
    df : pandas.DataFrame
        Table from :func:`collect_lcurve`.
    parameters : list of str
        Dotted ``pism_config`` keys behind the table's parameter columns; the
        first one is the abscissa of the sweep.
    output_file : pathlib.Path
        Where to write the figure; the suffix picks the format.
    log : bool, optional
        Use logarithmic axes, the space the corner is computed in.
    dpi : int, optional
        Resolution of raster output.

    Returns
    -------
    pandas.DataFrame
        The corner rows, one per curve, with the parameter values, ``M`` and
        ``N`` of the picked member.
    """
    columns = [short_name(p) for p in parameters]
    sweep, grouping = columns[0], columns[1:]
    groups = df.groupby(grouping, sort=True) if grouping else [((), df)]

    corners: list[dict[str, Any]] = []
    with mpl.rc_context(rc=rc_params):
        fig, ax = plt.subplots(figsize=(3.2, 2.4))
        for key, group in groups:
            g = group.sort_values(sweep).reset_index(drop=True)
            label = ", ".join(f"{c}={v:g}" for c, v in zip(grouping, np.atleast_1d(key))) if grouping else None
            ax.plot(g["N"], g["M"], "o-", ms=2, lw=0.75, label=label)
            for _, row in g.iterrows():
                ax.annotate(
                    f"{row[sweep]:g}",
                    (row["N"], row["M"]),
                    fontsize=fontsize,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
            index = corner(g["N"].values, g["M"].values)
            if index is None:
                logger.warning("%s: fewer than 3 points, no corner", label or "L-curve")
                continue
            ax.plot(g["N"][index], g["M"][index], "k*", ms=5, zorder=5)
            corners.append(g.loc[index].to_dict())

        if grouping:
            legend = ax.legend(loc="best")
            legend.get_frame().set_linewidth(0.0)
            legend.get_frame().set_alpha(0.0)
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(r"model norm  $N=\sqrt{J_\mathrm{design}}$")
        ax.set_ylabel(r"data misfit  $M$ (m yr$^{-1}$)")
        ax.set_title("L-curve")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=dpi)
        plt.close(fig)
    logger.info("wrote %s", output_file)
    return pd.DataFrame(corners)


def main() -> None:
    """
    Run the ``pism-inverse-lcurve`` command line tool.

    Returns
    -------
    None
        The figure and the ``.csv`` table are written to disk.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = (
        "L-curve of a PISM Tikhonov inversion ensemble: data misfit against model norm, "
        "with the corner of the L marked as the conventional pick for the "
        "regularization parameter."
    )
    parser.add_argument(
        "--parameters",
        help="Comma-separated pism_config keys varied across the ensemble. The first one is "
        "the swept parameter labelling the points; any further ones split the ensemble "
        "into one curve each.",
        type=str,
        default=",".join(DEFAULT_PARAMETERS),
    )
    parser.add_argument(
        "-o",
        "--output-file",
        help="Figure to write; the suffix picks the format. The underlying table is written "
        "alongside it with a .csv suffix.",
        type=str,
        default="lcurve.png",
    )
    parser.add_argument(
        "--log",
        help="Use logarithmic axes, the space the corner is computed in.",
        action="store_true",
    )
    parser.add_argument(
        "--dpi",
        help="Resolution of raster output.",
        type=int,
        default=300,
    )
    parser.add_argument(
        "INFILES",
        help="Inversion output files, one per ensemble member, e.g. inv_g*.nc.",
        nargs="+",
    )

    options = parser.parse_args()
    parameters = [p.strip() for p in options.parameters.split(",") if p.strip()]
    if not parameters:
        parser.error("--parameters needs at least one pism_config key")

    output_file = Path(options.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(output_file.parent / "lcurve.log")

    df = collect_lcurve([Path(f) for f in options.INFILES], parameters)
    table_file = output_file.with_suffix(".csv")
    df.to_csv(table_file, index=False)
    logger.info("wrote %s", table_file)

    corners = plot_lcurve(df, parameters, output_file, log=options.log, dpi=options.dpi)

    print(df.to_string(index=False))
    if not corners.empty:
        print("\nL-curve corner:")
        print(corners.to_string(index=False))


if __name__ == "__main__":
    main()
