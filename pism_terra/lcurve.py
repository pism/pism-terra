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

What a member inverted for is its *strategy*, the value of
``inverse.design.variable``: ``tauc`` or ``hardav`` alone, or a pair,
``tauc_hardav`` or ``hardav_tauc``, which alternates between the two in that
order, each phase holding the other field fixed. One ensemble can sample the
strategy the way it samples the penalty weight, and the tool then draws one
L-curve per strategy on one pair of axes.

A pair writes a design functional per phase and cycle, and each phase's norm
is taken from its last cycle. ``J_design`` measures the departure of zeta from
its prior, and a single-field run leaves the other field at its prior, so the
norm of every strategy is ``N = sqrt(sum of J_design over its phases)`` — for
a single field just ``sqrt(J_design)``. Under PISM's default
``inverse.design.param = "exp"`` every zeta is dimensionless and the curves
share one axis. The misfit is shared by the phases of a pair: there is one
residual, produced by both fields together.

The tool reads the ensemble members named on the command line, tabulates the
regularization parameters and the strategy straight out of each file's
``pism_config`` attributes, and writes the plot plus a table with one row per
member:

```bash
pism-inverse-lcurve --parameters inverse.tikhonov.penalty_weight \
    -o lcurve.png inv_g*.nc
```

Every pair strategy also gets a figure of its phases, e.g.
``lcurve_tauc_hardav.png``, one curve per phase against the shared misfit,
which shows which field the regularization is biting on.

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

# Values of ``inverse.design.variable``: one field, or a pair that ``pismi``
# alternates between in the order given.
STRATEGIES = ("tauc", "hardav", "tauc_hardav", "hardav_tauc")

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


def config_value(value: Any) -> float | str:
    """
    Read a ``pism_config`` attribute as a number where it is one.

    Parameters
    ----------
    value : Any
        Attribute value.

    Returns
    -------
    float or str
        The value as a float, or as a string for a keyword such as
        ``inverse.design.variable``.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def format_value(value: Any) -> str:
    """
    Format a parameter value for a label.

    Parameters
    ----------
    value : Any
        Number or keyword.

    Returns
    -------
    str
        ``"1e+03"`` style for numbers, the keyword itself otherwise.
    """
    return f"{value:g}" if isinstance(value, (int, float, np.number)) else str(value)


def design_strategy(ds: xr.Dataset) -> str | None:
    """
    Which strategy the inversion followed, as a value of :data:`STRATEGIES`.

    ``inverse.design.variable`` names it: ``tauc`` or ``hardav`` for a
    single-field run, or the pair ``tauc_hardav`` / ``hardav_tauc`` for an
    alternating co-inversion in that order. ``pismi`` ignores
    ``inverse.alternating_cycles`` for a single field, so the cycle count
    does not decide it.

    The variables the run wrote override a single-field configuration when
    they show both phases (``zeta_inv_tauc`` *and* ``zeta_inv_hardav``):
    before PISM took pairs, a positive cycle count alternated starting with
    the configured field. Output written before PISM had the parameter at all
    falls back to the variables alone — per-phase zeta for an alternating
    run, a single ``<design>_prior``, or as the last resort a single plain
    field.

    Parameters
    ----------
    ds : xarray.Dataset
        Inversion output.

    Returns
    -------
    str or None
        The strategy, or ``None`` when neither the configuration nor the
        variables say.
    """
    config = ds["pism_config"].attrs if "pism_config" in ds else {}
    names = set(ds.variables)
    alternated = all(f"zeta_inv_{v}" in names for v in DESIGN_VARIABLES)
    configured = str(config.get("inverse.design.variable", "")).lower()
    if configured in STRATEGIES:
        if configured in DESIGN_VARIABLES and alternated:
            other = next(v for v in DESIGN_VARIABLES if v != configured)
            return f"{configured}_{other}"
        return configured
    if alternated:
        return "tauc_hardav"
    for candidates in (
        [v for v in DESIGN_VARIABLES if f"{v}_prior" in names],
        [v for v in DESIGN_VARIABLES if v in names],
    ):
        if len(candidates) == 1:
            return candidates[0]
    return None


def strategy_phases(strategy: str) -> list[str]:
    """
    Split a strategy into the design variables it inverts for, in order.

    Parameters
    ----------
    strategy : str
        A value of :data:`STRATEGIES`.

    Returns
    -------
    list of str
        ``["tauc"]`` for ``tauc``, ``["hardav", "tauc"]`` for ``hardav_tauc``.
    """
    return strategy.split("_")


def design_variables(ds: xr.Dataset) -> list[str]:
    """
    Which fields the inversion solved for, in the order it solved for them.

    Parameters
    ----------
    ds : xarray.Dataset
        Inversion output.

    Returns
    -------
    list of str
        The phases of :func:`design_strategy` — two for an alternating
        co-inversion, one for a single-field run, none when the strategy
        cannot be told.
    """
    strategy = design_strategy(ds)
    return strategy_phases(strategy) if strategy else []


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


def collect_lcurve(files: list[Path], parameters: list[str], strategies: list[str] | None = None) -> pd.DataFrame:
    """
    Tabulate misfit, model norm and regularization parameters of an ensemble.

    An alternating co-inversion contributes one row per phase: the data
    misfit is shared — one residual, produced by both design variables
    together — while the model norm is that phase's own. Files that do not
    carry the full set of inversion diagnostics, and phases a member has not
    reached, are skipped with a warning, as are files missing one of
    ``parameters`` in their ``pism_config`` attributes and files whose
    strategy cannot be told. :func:`member_norms` turns the rows into one per
    member.

    Parameters
    ----------
    files : list of pathlib.Path
        Inversion output files, one per ensemble member.
    parameters : list of str
        Dotted ``pism_config`` keys to read from each file; they become
        columns named after their last dotted component.
    strategies : list of str or None, optional
        Strategies (values of :data:`STRATEGIES`) to keep, or ``None`` for
        every member.

    Returns
    -------
    pandas.DataFrame
        One row per usable (file, phase) pair with the ``parameters``
        columns plus ``M`` (data misfit, m/yr), ``N`` (model norm of the
        phase), ``strategy``, ``design`` (the field of the phase),
        ``norm_units`` (empty when the norm is dimensionless) and ``file``,
        sorted by strategy (in :data:`STRATEGIES` order), then the parameter
        columns, then phase.

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

                strategy = design_strategy(ds)
                if strategy is None:
                    logger.warning("%s: skipped, cannot tell what the inversion solved for", path.name)
                    continue
                if strategies and strategy not in strategies:
                    continue
                misfit = data_misfit(ds)
                for phase, design in enumerate(strategy_phases(strategy)):
                    if j_design_variable(ds, design) is None:
                        logger.warning(
                            "%s: no %s design functional, that phase is not written yet",
                            path.name,
                            design,
                        )
                        continue
                    row: dict[str, Any] = {c: config_value(config[p]) for c, p in zip(columns, parameters)}
                    row["M"] = misfit
                    row["N"] = model_norm(ds, design)
                    row["strategy"] = strategy
                    row["design"] = design
                    row["phase"] = phase
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
    return _in_strategy_order(pd.DataFrame(rows), columns + ["phase"])


def _in_strategy_order(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Sort a table by strategy, in :data:`STRATEGIES` order, then by ``columns``.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with a ``strategy`` column.
    columns : list of str
        Further sort keys.

    Returns
    -------
    pandas.DataFrame
        The sorted table with a fresh index.
    """
    order = df["strategy"].map({s: i for i, s in enumerate(STRATEGIES)})
    return df.assign(_order=order).sort_values(["_order"] + columns).drop(columns="_order").reset_index(drop=True)


def member_norms(df: pd.DataFrame, parameters: list[str]) -> pd.DataFrame:
    """
    Combine the phases of each member into its total model norm.

    ``J_design`` measures the departure of zeta from its prior, and a
    single-field run leaves the other field at its prior, so
    ``N = sqrt(sum of J_design over the phases)`` puts every strategy on the
    same axis. Adding the phases needs every phase to be written and the
    norms to be dimensionless (``inverse.design.param = "exp"``); members that
    fall short are left out with a warning.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-phase table from :func:`collect_lcurve`.
    parameters : list of str
        Dotted ``pism_config`` keys behind the table's parameter columns.

    Returns
    -------
    pandas.DataFrame
        One row per member with the ``parameters`` columns plus
        ``strategy``, ``M``, ``N`` (total norm), ``N_<design>`` per phase,
        ``norm_units`` and ``file``, in strategy then parameter order.
    """
    columns = [short_name(p) for p in parameters]
    rows: list[dict[str, Any]] = []
    for file, group in df.groupby("file", sort=False):
        strategy = str(group["strategy"].iloc[0])
        phases = strategy_phases(strategy)
        norms = dict(zip(group["design"], group["N"]))
        missing = [p for p in phases if p not in norms]
        if missing:
            logger.warning("%s: left out, %s phase not written yet", file, ", ".join(missing))
            continue
        units = set(group["norm_units"])
        if len(phases) > 1 and units != {""}:
            logger.warning("%s: left out, phase norms carry units (%s) and cannot be added", file, ", ".join(units))
            continue
        row: dict[str, Any] = {c: group[c].iloc[0] for c in columns}
        row["strategy"] = strategy
        row["M"] = float(group["M"].iloc[0])
        row["N"] = float(np.sqrt(sum(norms[p] ** 2 for p in phases)))
        for design in DESIGN_VARIABLES:
            row[f"N_{design}"] = norms.get(design, np.nan)
        row["norm_units"] = str(group["norm_units"].iloc[0]) if len(phases) == 1 else ""
        row["file"] = file
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return _in_strategy_order(pd.DataFrame(rows), columns)


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


def total_norm_label(table: pd.DataFrame) -> str:
    """
    Build the model-norm axis label of the per-strategy L-curve.

    Parameters
    ----------
    table : pandas.DataFrame
        Per-member table from :func:`member_norms`.

    Returns
    -------
    str
        For a single single-field strategy the label of :func:`norm_axis_label`;
        otherwise the norm summed over the phases, dimensionless unless every
        member is a single field under ``param = "ident"`` in one unit.
    """
    strategies = list(dict.fromkeys(table["strategy"]))
    units = sorted(set(table["norm_units"]) - {""})
    if len(strategies) == 1 and strategies[0] in DESIGN_VARIABLES:
        return norm_axis_label(table.assign(design=strategies[0]))
    if len(units) > 1:
        logger.warning("strategies carry norms in different units (%s); labelling generically", ", ".join(units))
        return r"model norm  $N=\sqrt{\sum J_\mathrm{design}}$"
    shown = mathtext_units(units[0]) if units and set(table["norm_units"]) == set(units) else r"dimensionless $\zeta$"
    return rf"model norm  $N=\sqrt{{\sum J_\mathrm{{design}}}}$ ({shown})"


# One color and marker per design variable and per strategy, so the figures
# read at a glance and keep the same assignment across runs.
DESIGN_STYLE = {"tauc": ("C0", "o"), "hardav": ("C1", "s")}
STRATEGY_STYLE = {
    "tauc": ("C0", "o"),
    "hardav": ("C1", "s"),
    "tauc_hardav": ("C2", "^"),
    "hardav_tauc": ("C3", "D"),
}

# Point-label offsets, in points, cycled over the curves of one figure so the
# labels of overlaid curves do not land on top of each other.
LABEL_OFFSETS = ((3, 3), (3, -9), (-14, 3), (-14, -9))


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
            format_value(row[sweep]),
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


def _finish(
    ax: Any, fig: Any, *, xlabel: str, title: str, legend: bool, log: bool, output_file: Path, dpi: int
) -> None:
    """
    Label, scale and write an L-curve figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes drawn on.
    fig : matplotlib.figure.Figure
        Figure to write and close.
    xlabel : str
        Model-norm axis label.
    title : str
        Figure title.
    legend : bool
        Draw the legend.
    log : bool
        Use logarithmic axes.
    output_file : pathlib.Path
        Where to write the figure.
    dpi : int
        Resolution of raster output.
    """
    if legend:
        handle = ax.legend(loc="best")
        handle.get_frame().set_linewidth(0.0)
        handle.get_frame().set_alpha(0.0)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"data misfit  $M$ (m yr$^{-1}$)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)
    logger.info("wrote %s", output_file)


def plot_lcurve(
    table: pd.DataFrame,
    parameters: list[str],
    output_file: Path,
    log: bool = False,
    dpi: int = 300,
) -> pd.DataFrame:
    """
    Plot one L-curve per strategy and mark each corner.

    Points are joined in order of the *first* parameter, whose value labels
    each marker. The ensemble splits into one curve per strategy and, within
    it, per combination of any further parameters, drawn with a legend when
    there is more than one curve. The corner of every curve is marked with a
    star.

    Parameters
    ----------
    table : pandas.DataFrame
        Per-member table from :func:`member_norms`.
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
        The corner rows, one per curve, with the strategy, parameter values,
        ``M`` and ``N`` of the picked member.
    """
    columns = [short_name(p) for p in parameters]
    sweep, grouping = columns[0], columns[1:]
    strategies = list(dict.fromkeys(table["strategy"]))

    corners: list[dict[str, Any]] = []
    curves = 0
    with mpl.rc_context(rc=rc_params):
        fig, ax = plt.subplots(figsize=(3.2, 2.4))
        for strategy in strategies:
            rows = table[table["strategy"] == strategy]
            color, marker = STRATEGY_STYLE.get(strategy, (None, "o"))
            for key, group in rows.groupby(grouping, sort=True) if grouping else [((), rows)]:
                g = group.sort_values(sweep).reset_index(drop=True)
                parts = [strategy] if len(strategies) > 1 else []
                parts += [f"{c}={format_value(v)}" for c, v in zip(grouping, np.atleast_1d(key))]
                found = _draw_curve(
                    ax,
                    g,
                    sweep,
                    label=", ".join(parts) or None,
                    color=color if len(strategies) > 1 else None,
                    marker=marker,
                    offset=LABEL_OFFSETS[curves % len(LABEL_OFFSETS)],
                )
                curves += 1
                if found is not None:
                    corners.append(found)
        _finish(
            ax,
            fig,
            xlabel=total_norm_label(table),
            title="L-curve",
            legend=curves > 1,
            log=log,
            output_file=output_file,
            dpi=dpi,
        )
    return pd.DataFrame(corners)


def plot_phases(
    df: pd.DataFrame,
    parameters: list[str],
    output_file: Path,
    log: bool = False,
    dpi: int = 300,
) -> pd.DataFrame:
    """
    Overlay the phases of an alternating strategy on one L-curve.

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
        Per-phase rows of one strategy, from :func:`collect_lcurve`.
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
        The corner rows, one per phase. Empty when the figure was refused
        because the norms are not comparable.
    """
    units = sorted(set(df["norm_units"]))
    if len(units) > 1:
        logger.warning(
            "not drawing the phase L-curve: the norms are not comparable (%s)",
            ", ".join(u or "dimensionless" for u in units),
        )
        return pd.DataFrame()

    columns = [short_name(p) for p in parameters]
    sweep, grouping = columns[0], columns[1:]
    strategy = str(df["strategy"].iloc[0]) if "strategy" in df else ""
    corners: list[dict[str, Any]] = []
    curves = 0
    with mpl.rc_context(rc=rc_params):
        fig, ax = plt.subplots(figsize=(3.2, 2.4))
        for index, design in enumerate(strategy_phases(strategy) if strategy else sorted(set(df["design"]))):
            rows = df[df["design"] == design]
            color, marker = DESIGN_STYLE.get(str(design), (f"C{index}", "o"))
            for key, group in rows.groupby(grouping, sort=True) if grouping else [((), rows)]:
                g = group.sort_values(sweep).reset_index(drop=True)
                extra = ", ".join(f"{c}={format_value(v)}" for c, v in zip(grouping, np.atleast_1d(key)))
                found = _draw_curve(
                    ax,
                    g,
                    sweep,
                    label=f"{design} phase{', ' + extra if extra else ''}",
                    color=color,
                    marker=marker,
                    offset=LABEL_OFFSETS[curves % len(LABEL_OFFSETS)],
                )
                curves += 1
                if found is not None:
                    corners.append(found)
        _finish(
            ax,
            fig,
            xlabel=norm_axis_label(df, mixed_ok=True),
            title=f"L-curve, phases of {strategy}" if strategy else "L-curve, phases",
            legend=True,
            log=log,
            output_file=output_file,
            dpi=dpi,
        )
    return pd.DataFrame(corners)


def main() -> None:
    """
    Run the ``pism-inverse-lcurve`` command line tool.

    Returns
    -------
    None
        The L-curve figure and its ``.csv`` table are written to disk, plus
        one phase figure per alternating strategy.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = (
        "L-curve of a PISM Tikhonov inversion ensemble: data misfit against model norm, "
        "with the corner of the L marked as the conventional pick for the "
        "regularization parameter. One curve per strategy (inverse.design.variable); "
        "every alternating strategy also gets a figure of its phases."
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
        "alongside it with a .csv suffix, and the phase figures with the strategy appended "
        "to the stem.",
        type=str,
        default="lcurve.png",
    )
    parser.add_argument(
        "--strategy",
        help=f"Comma-separated strategies to keep ({', '.join(STRATEGIES)}). The default keeps every " "member.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--no-phases",
        help="Skip the figures of the phases of each alternating strategy.",
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

    strategies = None
    if options.strategy:
        strategies = [d.strip() for d in options.strategy.split(",") if d.strip()]
        unknown = [d for d in strategies if d not in STRATEGIES]
        if unknown or not strategies:
            parser.error(f"--strategy takes any of {', '.join(STRATEGIES)}; got {', '.join(unknown) or 'nothing'}")

    output_file = Path(options.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(output_file.parent / "lcurve.log")

    df = collect_lcurve([Path(f) for f in options.INFILES], parameters, strategies)
    table = member_norms(df, parameters)
    if table.empty:
        raise SystemExit("no member has written every phase of its strategy; nothing to plot")
    table_file = output_file.with_suffix(".csv")
    table.to_csv(table_file, index=False)
    logger.info("wrote %s", table_file)

    corners = plot_lcurve(table, parameters, output_file, log=options.log, dpi=options.dpi)
    print(table.drop(columns=["norm_units", "file"]).to_string(index=False))
    if not corners.empty:
        print("\nL-curve corners:")
        shown = [c for c in ["strategy"] + [short_name(p) for p in parameters] + ["M", "N"] if c in corners]
        print(corners[shown].to_string(index=False))

    if options.no_phases:
        return
    for strategy in dict.fromkeys(df["strategy"]):
        if strategy in DESIGN_VARIABLES:
            continue
        phase_file = output_file.with_name(f"{output_file.stem}_{strategy}{output_file.suffix}")
        rows = df[df["strategy"] == strategy].reset_index(drop=True)
        phase_corners = plot_phases(rows, parameters, phase_file, log=options.log, dpi=options.dpi)
        if not phase_corners.empty:
            print(f"\nwrote {phase_file}")


if __name__ == "__main__":
    main()
