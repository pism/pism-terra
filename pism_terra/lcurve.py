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

The model-norm axis is labelled with the field the run inverted for — ``tauc``
or ``hardav``, read off the variables ``pismi`` wrote — and with the units of
the norm. Those are usually *not* the units of that field: ``J_design`` is
evaluated on the parameterized design variable zeta, which is dimensionless
under PISM's default ``inverse.design.param = "exp"``.

The tool reads the ensemble members named on the command line, tabulates the
regularization parameters straight out of each file's ``pism_config``
attributes, and writes the plot plus the underlying table:

```bash
pism-inverse-lcurve --parameters inverse.tikhonov.penalty_weight \
    -o lcurve.png inv_g*.nc
```

An alternating ``tauc``/``hardav`` co-inversion optimizes each phase in turn
and writes a design functional per phase and cycle, so it gives one curve per
phase — ``lcurve_tauc.png`` and ``lcurve_hardav.png``, each with its table —
taking each phase's norm from its last cycle. The misfit is shared: there is
one residual, produced by both design variables together, so the curves
differ only in ``N``. Outputs are suffixed only when there is more than one
curve, so a single-design run keeps the name it was given, and a third figure
— ``lcurve_combined.png`` — overlays the phases on one pair of axes, which
the dimensionless norm of the default ``exp`` parameterization makes
comparable.

Members that are still running, or that crashed before writing the inversion
diagnostics, lack the residual or a phase's design functional and are skipped
with a warning rather than aborting the analysis.
"""

from __future__ import annotations

import logging
import re
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

# Fields every member needs. The design functional is resolved separately:
# a single-design run writes ``J_design``, an alternating co-inversion writes
# ``J_design_c<cycle>_<design>`` per phase (see :func:`j_design_variable`).
REQUIRED_VARS = ("vel_misfit_weight", "inv_residual")

# Design variables ``pismi -inv_design`` can invert for, with the units of the
# *physical* field. ``hardav`` is the vertically-averaged ice hardness
# B = A^(-1/n), so its units carry the Glen exponent; PISM writes the literal
# string ``Pa s^(1/n)`` and marks it as not UDUNITS-validated
# (``set_units_without_validation`` in ``HardnessAverage``), with the exponent
# filled in from the flow law's ``n``.
DESIGN_VARIABLES = {"tauc": "Pa", "hardav": "Pa s^(1/n)"}

# Config keys holding the Glen exponent, most specific stress balance first.
_GLEN_EXPONENT_KEYS = (
    "stress_balance.blatter.Glen_exponent",
    "stress_balance.ssa.Glen_exponent",
    "stress_balance.sia.Glen_exponent",
)

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


def design_variables(ds: xr.Dataset) -> list[str]:
    """
    Which fields the inversion solved for: ``tauc``, ``hardav``, or both.

    ``pismi``'s ``-inv_design`` is a plain option rather than a configuration
    parameter, so it is not recorded in ``pism_config`` and has to be read off
    the variables the run wrote. An alternating co-inversion names its design
    variable per phase (``zeta_inv_tauc`` *and* ``zeta_inv_hardav``); a
    single-design run writes ``zeta_inv`` and a ``<design>_prior``. The plain
    field is the last resort, since a ``tauc`` inversion of a Blatter forward
    problem may carry a prescribed ``hardav`` alongside it, and an alternating
    run carries both fields whichever phase it is in.

    Parameters
    ----------
    ds : xarray.Dataset
        Inversion output.

    Returns
    -------
    list of str
        The design variables, in :data:`DESIGN_VARIABLES` order — two for an
        alternating co-inversion, one for a single-design run, none when the
        file names neither.
    """
    names = set(ds.variables)
    for candidates in (
        [v for v in DESIGN_VARIABLES if f"zeta_inv_{v}" in names],
        [v for v in DESIGN_VARIABLES if f"{v}_prior" in names],
        [v for v in DESIGN_VARIABLES if v in names],
    ):
        if candidates:
            return candidates
    return []


def design_variable(ds: xr.Dataset) -> str | None:
    """
    Give the single field the inversion solved for, if there is just one.

    Parameters
    ----------
    ds : xarray.Dataset
        Inversion output.

    Returns
    -------
    str or None
        ``"tauc"`` or ``"hardav"``, or ``None`` when the file names neither
        or names both — an alternating run inverts for both in turn, so no
        single name describes it. Use :func:`design_variables` there.
    """
    found = design_variables(ds)
    return found[0] if len(found) == 1 else None


def design_units(ds: xr.Dataset, design: str) -> str:
    """
    Give a design variable's units, with the Glen exponent filled in.

    ``hardav`` is the vertically-averaged hardness B = A^(-1/n), so PISM
    writes its units as the literal ``Pa s^(1/n)`` — a string UDUNITS cannot
    parse, which is why PISM sets it without validation. Substitute the flow
    law's ``n`` when the file records one.

    Parameters
    ----------
    ds : xarray.Dataset
        Inversion output carrying ``pism_config``.
    design : str
        Design variable, a key of :data:`DESIGN_VARIABLES`.

    Returns
    -------
    str
        Units, e.g. ``"Pa"`` for ``tauc`` and ``"Pa s^(1/3)"`` for ``hardav``
        under the usual Glen exponent.
    """
    units = DESIGN_VARIABLES[design]
    if "1/n" not in units:
        return units
    config = ds["pism_config"].attrs
    for key in _GLEN_EXPONENT_KEYS:
        if key in config:
            return units.replace("1/n", f"1/{float(config[key]):g}")
    return units


def mathtext_units(units: str) -> str:
    """
    Typeset a units string's ``^(...)`` exponents as mathtext.

    Parameters
    ----------
    units : str
        Units as PISM writes them, e.g. ``"Pa s^(1/3)"``.

    Returns
    -------
    str
        The same units with the exponent in mathtext, so an axis shows
        ``Pa s^{1/3}`` rather than the caret and parentheses.
    """
    return re.sub(r"\^\(([^)]*)\)", r"$^{\1}$", units)


def model_norm_units(ds: xr.Dataset, design: str | None) -> str | None:
    """
    Give the units of the model norm, or ``None`` when it is dimensionless.

    ``J_design`` is the design functional evaluated on the *parameterized*
    design variable zeta, not on the physical field: with
    ``inverse.design.param = "exp"`` (PISM's default, which keeps ``tauc``
    positive via ``tauc = tauc_scale * exp(zeta)``) zeta is dimensionless, and
    so is ``N = sqrt(J_design)``. Only ``param = "ident"`` makes zeta the field
    itself, and only then does the norm carry the field's units.

    Parameters
    ----------
    ds : xarray.Dataset
        Inversion output carrying ``pism_config``.
    design : str or None
        Design variable from :func:`design_variable`.

    Returns
    -------
    str or None
        Units of the norm, or ``None`` when it is dimensionless — either
        because zeta is a transform of the field or because the design
        variable could not be identified.
    """
    config = ds["pism_config"].attrs
    if design is None or str(config.get("inverse.design.param", "")).lower() != "ident":
        return None
    return design_units(ds, design)


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


def j_design_variable(ds: xr.Dataset, design: str | None = None) -> str | None:
    """
    Name the design-functional history of one inversion phase.

    A single-design run writes ``J_design``. An alternating co-inversion runs
    the phases repeatedly and writes one history each, ``J_design_c<cycle>_
    <design>``, so the phase's converged norm is the one from its *last*
    cycle. (``J_design_weighted_...`` is the same functional divided by the
    penalty weight, so it is not the model norm and is skipped.)

    Parameters
    ----------
    ds : xarray.Dataset
        Inversion output.
    design : str or None, optional
        Design variable whose phase is wanted. ``None`` accepts only the
        plain ``J_design``.

    Returns
    -------
    str or None
        Variable name, or ``None`` when the file carries no history for that
        phase — a member that has not reached it yet.
    """
    if "J_design" in ds:
        return "J_design"
    if design is None:
        return None
    pattern = re.compile(rf"^J_design_c(\d+)_{re.escape(design)}$")
    cycles = [(int(m.group(1)), name) for name in ds.variables if (m := pattern.match(str(name)))]
    return max(cycles)[1] if cycles else None


def model_norm(ds: xr.Dataset, design: str | None = None) -> float:
    """
    Model norm ``N = sqrt(J_design)`` at the last inversion iteration.

    Parameters
    ----------
    ds : xarray.Dataset
        Inversion output carrying a design-functional iteration history.
    design : str or None, optional
        Design variable whose phase to read, for an alternating co-inversion.

    Returns
    -------
    float
        Square root of the design functional of the converged solution.

    Raises
    ------
    KeyError
        If the file carries no design functional for that phase.
    """
    name = j_design_variable(ds, design)
    if name is None:
        raise KeyError(f"no design functional for {design or 'the inversion'}")
    # An alternating run names the iteration axis per phase too
    # (``inv_iter_c1_tauc``), so index the history's own dimension.
    history = ds[name]
    return float(np.sqrt(history.isel({history.dims[-1]: -1})))


def collect_lcurve(files: list[Path], parameters: list[str], designs: list[str] | None = None) -> pd.DataFrame:
    """
    Tabulate misfit, model norm and regularization parameters of an ensemble.

    An alternating co-inversion contributes one row per phase: the data
    misfit is shared — one residual, produced by both design variables
    together — while the model norm is that phase's own. Files that do not
    carry the full set of inversion diagnostics, and phases a member has not
    reached, are skipped with a warning, as are files missing one of
    ``parameters`` in their ``pism_config`` attributes.

    Parameters
    ----------
    files : list of pathlib.Path
        Inversion output files, one per ensemble member.
    parameters : list of str
        Dotted ``pism_config`` keys to read from each file; they become
        columns named after their last dotted component.
    designs : list of str or None, optional
        Design variables to tabulate, or ``None`` to detect them per file
        with :func:`design_variables`.

    Returns
    -------
    pandas.DataFrame
        One row per usable (file, design) pair with the ``parameters``
        columns plus ``M`` (data misfit, m/yr), ``N`` (model norm),
        ``design`` (the inverted field), ``norm_units`` (empty when the norm
        is dimensionless) and ``file``, sorted by design then the parameter
        columns.

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

                found = designs or design_variables(ds) or [None]  # type: ignore[list-item]
                misfit = data_misfit(ds)
                for design in found:
                    if j_design_variable(ds, design) is None:
                        logger.warning(
                            "%s: no %s design functional, that phase is not written yet",
                            path.name,
                            design or "J_design",
                        )
                        continue
                    row: dict[str, Any] = {c: float(config[p]) for c, p in zip(columns, parameters)}
                    row["M"] = misfit
                    row["N"] = model_norm(ds, design)
                    row["design"] = design or ""
                    row["norm_units"] = model_norm_units(ds, design) or ""
                    row["file"] = path.name
                    rows.append(row)
        except (OSError, KeyError, ValueError) as error:
            logger.warning("%s: skipped, %s", path.name, error)

    if not rows:
        raise ValueError(
            f"none of the {len(files)} given files yielded an L-curve point; "
            f"they need {', '.join(REQUIRED_VARS)}, a design functional and {', '.join(parameters)}"
        )
    logger.info("collected %d rows from %d files", len(rows), len(files))
    return pd.DataFrame(rows).sort_values(["design"] + columns).reset_index(drop=True)


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


def norm_axis_label(df: pd.DataFrame, mixed_ok: bool = False) -> str:
    """
    Build the model-norm axis label, naming the inverted field and its units.

    Parameters
    ----------
    df : pandas.DataFrame
        Table from :func:`collect_lcurve`, carrying ``design`` and
        ``norm_units``.
    mixed_ok : bool, optional
        Set when several design variables are deliberately on one pair of
        axes, as in :func:`plot_combined`, so that is not warned about.

    Returns
    -------
    str
        Label naming the design variable — ``tauc`` or ``hardav`` — and either
        its units or the fact that the norm is dimensionless. Falls back to
        the bare norm when the ensemble mixes design variables or the field
        could not be identified.
    """
    designs = sorted(set(df["design"]) - {""})
    base = r"model norm  $N=\sqrt{J_\mathrm{design}}$"
    if len(designs) != 1:
        if len(designs) > 1 and not mixed_ok:
            logger.warning("ensemble mixes design variables (%s); labelling generically", ", ".join(designs))
        if len(designs) > 1 and mixed_ok:
            units = sorted(set(df["norm_units"]) - {""})
            shown = mathtext_units(units[0]) if len(units) == 1 else r"dimensionless $\zeta$"
            return f"{base} ({shown})"
        return base
    units = sorted(set(df["norm_units"]) - {""})
    # A dimensionless norm is the common case: it is the norm of the
    # parameterized zeta, not of the field itself (see model_norm_units).
    if len(units) != 1:
        return f"{base}, {designs[0]} (dimensionless $\\zeta$)"
    return f"{base}, {designs[0]} ({mathtext_units(units[0])})"


# One color and marker per design variable, so the combined figure reads at a
# glance and keeps the same assignment across runs.
DESIGN_STYLE = {"tauc": ("C0", "o"), "hardav": ("C1", "s")}


def _draw_curve(
    ax: Any,
    g: pd.DataFrame,
    sweep: str,
    *,
    label: str | None = None,
    color: str | None = None,
    marker: str = "o",
    offset: tuple[int, int] = (3, 3),
) -> dict[str, Any] | None:
    """
    Draw one member sequence as a curve, and star its corner.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    g : pandas.DataFrame
        Rows of one curve, already ordered along it.
    sweep : str
        Column whose value labels each point.
    label : str or None, optional
        Legend entry for the curve.
    color : str or None, optional
        Line color; ``None`` takes the next one from the cycle.
    marker : str, optional
        Point marker.
    offset : tuple of int, optional
        Point-label offset in points, staggered per curve so overlaid curves
        do not collide.

    Returns
    -------
    dict or None
        The corner row, or ``None`` when the curve is too short to have one.
    """
    ax.plot(g["N"], g["M"], marker=marker, ls="-", ms=2, lw=0.75, label=label, color=color)
    for _, row in g.iterrows():
        ax.annotate(
            f"{row[sweep]:g}",
            (row["N"], row["M"]),
            fontsize=fontsize,
            xytext=offset,
            textcoords="offset points",
            color=color,
        )
    index = corner(g["N"].values, g["M"].values)
    if index is None:
        logger.warning("%s: fewer than 3 points, no corner", label or "L-curve")
        return None
    ax.plot(g["N"][index], g["M"][index], "k*", ms=5, zorder=5)
    return g.loc[index].to_dict()


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
    readable. The corner of every curve is marked with a star. The abscissa is
    labelled with the field the inversion solved for (see
    :func:`norm_axis_label`).

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
            found = _draw_curve(ax, g, sweep, label=label)
            if found is not None:
                corners.append(found)

        if grouping:
            legend = ax.legend(loc="best")
            legend.get_frame().set_linewidth(0.0)
            legend.get_frame().set_alpha(0.0)
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(norm_axis_label(df))
        ax.set_ylabel(r"data misfit  $M$ (m yr$^{-1}$)")
        ax.set_title("L-curve")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=dpi)
        plt.close(fig)
    logger.info("wrote %s", output_file)
    return pd.DataFrame(corners)


def plot_combined(
    df: pd.DataFrame,
    parameters: list[str],
    output_file: Path,
    log: bool = False,
    dpi: int = 300,
) -> pd.DataFrame:
    """
    Overlay every phase of an alternating co-inversion on one L-curve.

    The phases of an alternating run share a misfit and differ only in their
    model norm, so putting them on one pair of axes shows directly which
    design variable the regularization is biting on, and whether their
    corners agree on a penalty weight. This is only meaningful when the norms
    are commensurate — under PISM's default ``exp`` parameterization every
    norm is the dimensionless norm of zeta, so they are. Under
    ``param = "ident"`` they carry the fields' own units (Pa against
    Pa s^(1/n)) and the figure is refused rather than drawn misleadingly.

    Parameters
    ----------
    df : pandas.DataFrame
        Table from :func:`collect_lcurve`, covering more than one design
        variable.
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
        The corner rows, one per design variable. Empty when the figure was
        refused because the norms are not comparable.
    """
    units = sorted(set(df["norm_units"]))
    if len(units) > 1:
        logger.warning(
            "not drawing the combined L-curve: the norms are not comparable (%s)",
            ", ".join(u or "dimensionless" for u in units),
        )
        return pd.DataFrame()

    columns = [short_name(p) for p in parameters]
    sweep, grouping = columns[0], columns[1:]
    corners: list[dict[str, Any]] = []
    with mpl.rc_context(rc=rc_params):
        fig, ax = plt.subplots(figsize=(3.2, 2.4))
        for index, (design, rows) in enumerate(df.groupby("design", sort=True)):
            color, marker = DESIGN_STYLE.get(str(design), (f"C{index}", "o"))
            for key, group in rows.groupby(grouping, sort=True) if grouping else [((), rows)]:
                g = group.sort_values(sweep).reset_index(drop=True)
                extra = ", ".join(f"{c}={v:g}" for c, v in zip(grouping, np.atleast_1d(key))) if grouping else ""
                found = _draw_curve(
                    ax,
                    g,
                    sweep,
                    label=f"{design}{', ' + extra if extra else ''}",
                    color=color,
                    marker=marker,
                    # Stagger the point labels so the overlaid curves' do not
                    # land on top of each other.
                    offset=(3, 3) if index % 2 == 0 else (3, -9),
                )
                if found is not None:
                    corners.append(found)

        legend = ax.legend(loc="best")
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(norm_axis_label(df, mixed_ok=True))
        ax.set_ylabel(r"data misfit  $M$ (m yr$^{-1}$)")
        ax.set_title("L-curve, both phases")
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
        A figure and its ``.csv`` table are written to disk, one pair per
        design variable.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = (
        "L-curve of a PISM Tikhonov inversion ensemble: data misfit against model norm, "
        "with the corner of the L marked as the conventional pick for the "
        "regularization parameter. An alternating co-inversion gives one curve per phase."
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
        "--design-variable",
        help=f"Comma-separated design variables to curve ({', '.join(DESIGN_VARIABLES)}). The default "
        "reads them from each file, which yields one curve per phase of an alternating "
        "co-inversion.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--no-combined",
        help="Skip the extra figure overlaying every phase of an alternating co-inversion " "on one pair of axes.",
        action="store_true",
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

    designs = None
    if options.design_variable:
        designs = [d.strip() for d in options.design_variable.split(",") if d.strip()]
        unknown = [d for d in designs if d not in DESIGN_VARIABLES]
        if unknown or not designs:
            parser.error(
                f"--design-variable takes any of {', '.join(DESIGN_VARIABLES)}; "
                f"got {', '.join(unknown) or 'nothing'}"
            )

    df = collect_lcurve([Path(f) for f in options.INFILES], parameters, designs)
    # An alternating co-inversion gives one curve per phase; only then are the
    # outputs suffixed, so a single-design run keeps the name it was given.
    found = list(df["design"].unique())
    for design in found:
        rows = df[df["design"] == design].reset_index(drop=True)
        path = output_file
        if len(found) > 1:
            path = output_file.with_name(f"{output_file.stem}_{design}{output_file.suffix}")
        table_file = path.with_suffix(".csv")
        rows.to_csv(table_file, index=False)
        logger.info("wrote %s", table_file)

        corners = plot_lcurve(rows, parameters, path, log=options.log, dpi=options.dpi)

        if len(found) > 1:
            print(f"\n=== {design} ===")
        print(rows.to_string(index=False))
        if not corners.empty:
            print("L-curve corner:")
            print(corners.to_string(index=False))

    if len(found) > 1 and not options.no_combined:
        combined_file = output_file.with_name(f"{output_file.stem}_combined{output_file.suffix}")
        combined = plot_combined(df, parameters, combined_file, log=options.log, dpi=options.dpi)
        if not combined.empty:
            print("\n=== both phases ===")
            print(f"wrote {combined_file}")


if __name__ == "__main__":
    main()
