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
Where the time goes in a PISM run, from its ``-profile`` files.

``pism-profile-analysis`` reads the profiles the runners write when
``campaign.profile`` is set (see :mod:`pism_terra.profiling`) and answers two
questions per run: how much of the time-stepping loop the Blatter stress
balance costs next to PISM's other components, and what the Blatter solve
itself spends its time on.

PISM times the Blatter solve under its ``stress_balance.shallow`` event
(the "shallow" slot of the stress balance holds whichever membrane solver is
selected: SSA or Blatter). Inside it, PETSc's own events tell the story of
each Newton step: the residual (``SNESFunctionEval``), the Jacobian
(``SNESJacobianEval``, once per multigrid level when the levels are
rediscretised), the preconditioner setup (``PCSetUp``, which rebuilds the
multigrid hierarchy and the GAMG coarse solver), the linear solve
(``KSPSolve``) and the line search. Those phases overlap (the line search
evaluates the residual), so they are shown side by side, never stacked.

Times are per rank. The stacked figures use the mean over the ranks, the only
statistic that respects the nesting of events; the phase and kernel figures
show mean, slowest and fastest rank, because the spread between them is the
load imbalance.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm

from pism_terra.profiling import STAGE_SUMMARY, load_profiles

# Figures are files, never windows; switching here is safe as long as no
# figure exists yet, and this module creates them all.
matplotlib.use("Agg")

#: PISM's stage holding the model steps.
STAGE = "time-stepping loop"

#: PISM's event around the membrane stress balance; the Blatter solve when
#: ``stress_balance.model = "blatter"``.
BLATTER_EVENT = "stress_balance.shallow"

#: PISM's top-level components of a model step, in plotting order. Any time in
#: the stage that none of them accounts for is reported as ``other``.
COMPONENTS = (
    "stress_balance",
    "basal_hydrology",
    "energy",
    "mass_transport",
    "surface",
    "ocean",
    "calving",
    "bed_deformation",
    "basal_yield_stress",
    "age",
    "fracture_density",
    "io",
)

#: PETSc events of one Newton step of the Blatter solve, with a short label.
BLATTER_PHASES = {
    "SNESFunctionEval": "residual",
    "SNESJacobianEval": "Jacobian",
    "PCSetUp": "preconditioner setup",
    "KSPSolve": "linear solve",
    "SNESLineSearch": "line search",
}

#: Linear-algebra kernels the Blatter phases are made of, with a short label.
BLATTER_KERNELS = {
    "MatMult": "MatMult",
    "MatSOR": "MatSOR (smoother)",
    "MatResidual": "MatResidual",
    "MatPtAPNumeric": "MatPtAP (Galerkin)",
    "PCSetUp_GAMG+": "GAMG setup",
    "KSPGMRESOrthog": "GMRES orthog.",
    "VecNorm": "VecNorm (reduce)",
    "VecScatterEnd": "VecScatterEnd (wait)",
    "MatAssemblyBegin": "MatAssemblyBegin",
}


def _by_run_event(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """
    Reduce a profile table over the ranks, one row per run and event of a stage.

    Parameters
    ----------
    df : pandas.DataFrame
        Table from :func:`pism_terra.profiling.load_profiles`.
    stage : str
        Stage to reduce.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(run, event)`` with ``count`` (max over ranks),
        ``time_mean``, ``time_max``, ``time_min`` and ``balance``
        (``time_min / time_max``).
    """
    grouped = df[df["stage"] == stage].groupby(["run", "event"])
    out = pd.DataFrame(
        {
            "count": grouped["count"].max(),
            "time_mean": grouped["time"].mean(),
            "time_max": grouped["time"].max(),
            "time_min": grouped["time"].min(),
        }
    )
    out["balance"] = (out["time_min"] / out["time_max"]).where(out["time_max"] > 0, 0.0)
    return out


def _pick(table: pd.DataFrame, run: str, event: str, column: str = "time_mean") -> float:
    """
    Read one number off a ``(run, event)`` table, 0 when the event is absent.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`_by_run_event`.
    run : str
        Run label.
    event : str
        Event name.
    column : str, optional
        Column to read.

    Returns
    -------
    float
        The value, or 0.0.
    """
    try:
        return float(table.loc[(run, event), column])
    except KeyError:
        return 0.0


def component_breakdown(df: pd.DataFrame, stage: str = STAGE, blatter_event: str = BLATTER_EVENT) -> pd.DataFrame:
    """
    Split the stage's time into PISM's components, the Blatter solve set apart.

    ``stress_balance`` is reported as the Blatter solve (``blatter_event``)
    plus the rest of the stress balance (SIA modifier, strain heating,
    vertical velocity). Time in the stage outside every component is
    ``other``.

    Parameters
    ----------
    df : pandas.DataFrame
        Table from :func:`pism_terra.profiling.load_profiles`.
    stage : str, optional
        Stage to split.
    blatter_event : str, optional
        Event holding the Blatter solve.

    Returns
    -------
    pandas.DataFrame
        One row per run and component with ``time_mean``, ``time_max``
        (seconds, over the ranks), ``share_mean`` (of the stage's mean total)
        and ``stage_total_mean``, ``stage_total_max``.
    """
    table = _by_run_event(df, stage)
    rows = []
    for run in table.index.get_level_values("run").unique():
        total_mean = _pick(table, run, STAGE_SUMMARY)
        total_max = _pick(table, run, STAGE_SUMMARY, "time_max")
        parts: dict[str, tuple[float, float]] = {}
        blatter = (_pick(table, run, blatter_event), _pick(table, run, blatter_event, "time_max"))
        parts["Blatter solve"] = blatter
        sb = (_pick(table, run, "stress_balance"), _pick(table, run, "stress_balance", "time_max"))
        parts["stress balance, rest"] = (max(sb[0] - blatter[0], 0.0), max(sb[1] - blatter[1], 0.0))
        for component in COMPONENTS[1:]:
            parts[component.replace("_", " ")] = (
                _pick(table, run, component),
                _pick(table, run, component, "time_max"),
            )
        parts["other"] = (
            max(total_mean - sum(m for m, _ in parts.values()), 0.0),
            max(total_max - sum(x for _, x in parts.values()), 0.0),
        )
        for component, (mean, largest) in parts.items():
            rows.append(
                {
                    "run": run,
                    "component": component,
                    "time_mean": mean,
                    "time_max": largest,
                    "share_mean": mean / total_mean if total_mean else np.nan,
                    "stage_total_mean": total_mean,
                    "stage_total_max": total_max,
                }
            )
    return pd.DataFrame(rows)


def _event_table(df: pd.DataFrame, events: dict[str, str], stage: str, parent: str, kind: str) -> pd.DataFrame:
    """
    Reduce a set of events over the ranks and relate them to a parent event.

    Parameters
    ----------
    df : pandas.DataFrame
        Table from :func:`pism_terra.profiling.load_profiles`.
    events : dict of str to str
        Event name to short label.
    stage : str
        Stage to read.
    parent : str
        Event whose mean time the shares refer to.
    kind : str
        Name of the label column (``"phase"`` or ``"kernel"``).

    Returns
    -------
    pandas.DataFrame
        One row per run and event: ``event``, the label column, ``count``,
        ``time_mean``, ``time_max``, ``time_min``, ``balance``,
        ``share_of_parent`` (mean over mean) and ``parent_mean``.
    """
    table = _by_run_event(df, stage)
    rows = []
    for run in table.index.get_level_values("run").unique():
        parent_mean = _pick(table, run, parent)
        for event, label in events.items():
            if (run, event) not in table.index:
                continue
            rec = table.loc[(run, event)]
            rows.append(
                {
                    "run": run,
                    "event": event,
                    kind: label,
                    "count": rec["count"],
                    "time_mean": rec["time_mean"],
                    "time_max": rec["time_max"],
                    "time_min": rec["time_min"],
                    "balance": rec["balance"],
                    "share_of_parent": rec["time_mean"] / parent_mean if parent_mean else np.nan,
                    "parent_mean": parent_mean,
                }
            )
    return pd.DataFrame(rows)


def blatter_phases(df: pd.DataFrame, stage: str = STAGE, blatter_event: str = BLATTER_EVENT) -> pd.DataFrame:
    """
    The Newton-step phases of the Blatter solve, per run.

    Parameters
    ----------
    df : pandas.DataFrame
        Table from :func:`pism_terra.profiling.load_profiles`.
    stage : str, optional
        Stage to read.
    blatter_event : str, optional
        Event holding the Blatter solve; the shares refer to its mean time.

    Returns
    -------
    pandas.DataFrame
        See :func:`_event_table`; the label column is ``phase``. The phases
        overlap, so their shares need not add up to one.
    """
    return _event_table(df, BLATTER_PHASES, stage, blatter_event, "phase")


def blatter_kernels(df: pd.DataFrame, stage: str = STAGE, blatter_event: str = BLATTER_EVENT) -> pd.DataFrame:
    """
    The linear-algebra kernels under the Blatter solve, per run.

    Parameters
    ----------
    df : pandas.DataFrame
        Table from :func:`pism_terra.profiling.load_profiles`.
    stage : str, optional
        Stage to read.
    blatter_event : str, optional
        Event holding the Blatter solve; the shares refer to its mean time.

    Returns
    -------
    pandas.DataFrame
        See :func:`_event_table`; the label column is ``kernel``.
    """
    return _event_table(df, BLATTER_KERNELS, stage, blatter_event, "kernel")


def solver_counts(df: pd.DataFrame, stage: str = STAGE) -> pd.DataFrame:
    """
    Iteration counts of the Blatter solve, per run.

    Parameters
    ----------
    df : pandas.DataFrame
        Table from :func:`pism_terra.profiling.load_profiles`.
    stage : str, optional
        Stage to read.

    Returns
    -------
    pandas.DataFrame
        One row per run: ``steps`` (``SNESSolve`` calls), ``newton_steps``
        (``KSPSolve`` calls), ``newton_per_step``, ``ksp_iterations`` (GMRES
        orthogonalisations), ``ksp_per_newton``, ``jacobian_evals``,
        ``jacobian_per_newton`` (the number of multigrid levels when each is
        rediscretised), ``residual_evals`` and ``residual_per_newton``.
    """
    table = _by_run_event(df, stage)
    rows = []
    for run in table.index.get_level_values("run").unique():
        steps = _pick(table, run, "SNESSolve", "count")
        newton = _pick(table, run, "KSPSolve", "count")
        ksp = _pick(table, run, "KSPGMRESOrthog", "count")
        jac = _pick(table, run, "SNESJacobianEval", "count")
        res = _pick(table, run, "SNESFunctionEval", "count")
        rows.append(
            {
                "run": run,
                "steps": steps,
                "newton_steps": newton,
                "newton_per_step": newton / steps if steps else np.nan,
                "ksp_iterations": ksp,
                "ksp_per_newton": ksp / newton if newton else np.nan,
                "jacobian_evals": jac,
                "jacobian_per_newton": jac / newton if newton else np.nan,
                "residual_evals": res,
                "residual_per_newton": res / newton if newton else np.nan,
            }
        )
    return pd.DataFrame(rows)


def rank_times(df: pd.DataFrame, events: Sequence[str], stage: str = STAGE) -> pd.DataFrame:
    """
    Return seconds per rank for a few events, to look at the load balance.

    Parameters
    ----------
    df : pandas.DataFrame
        Table from :func:`pism_terra.profiling.load_profiles`.
    events : sequence of str
        Events to keep.
    stage : str, optional
        Stage to read.

    Returns
    -------
    pandas.DataFrame
        Long table with ``run``, ``event``, ``rank`` and ``time``.
    """
    sel = df[(df["stage"] == stage) & df["event"].isin(list(events))]
    return sel[["run", "event", "rank", "time"]].sort_values(["run", "event", "rank"], ignore_index=True)


def find_profiles(paths: Sequence[str | Path]) -> list[Path]:
    """
    Expand run directories into their profile files.

    Parameters
    ----------
    paths : sequence of str or pathlib.Path
        Profile files, or directories searched recursively for
        ``profile/profile_*.py``.

    Returns
    -------
    list of pathlib.Path
        The files, sorted, without duplicates.

    Raises
    ------
    FileNotFoundError
        If no profile file is found.
    """
    found: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found.extend(sorted(p.rglob("profile/profile_*.py")))
        elif p.exists():
            found.append(p)
    files = sorted(set(found))
    if not files:
        raise FileNotFoundError(f"no profile_*.py under {[str(p) for p in paths]}")
    return files


def run_labels(files: Sequence[Path]) -> dict[str, str]:
    """
    Short labels for the runs, from the UQ tables next to the profiles.

    A runner-written profile sits in ``<run>/output/profile/`` and its tag
    carries ``uq_<n>``; ``<run>/output/uq.csv`` names the parameters of
    member ``n``. The label is then ``uq_<n>: key=value, ...``. Without a UQ
    table the tag itself is the label.

    Parameters
    ----------
    files : sequence of pathlib.Path
        Profile files.

    Returns
    -------
    dict of str to str
        Tag (the default ``run`` label of :func:`pism_terra.profiling.load_profile`)
        to display label.
    """
    labels: dict[str, str] = {}
    for f in files:
        tag = f.stem.removeprefix("profile_")
        labels[tag] = tag
        uq_csv = f.parent.parent / "uq.csv"
        member = pd.Series([tag]).str.extract(r"uq_(\d+)")[0].iloc[0]
        if member is None or not uq_csv.exists():
            continue
        uq = pd.read_csv(uq_csv, keep_default_na=False)
        row = uq[uq["uq"] == int(member)]
        if row.empty:
            continue
        params = ", ".join(f"{k}={row.iloc[0][k]}" for k in uq.columns if k != "uq")
        labels[tag] = f"uq_{member}: {params}"
    return labels


def load_runs(paths: Sequence[str | Path]) -> pd.DataFrame:
    """
    Load every profile under ``paths`` with readable run labels.

    Parameters
    ----------
    paths : sequence of str or pathlib.Path
        Profile files or run directories (see :func:`find_profiles`).

    Returns
    -------
    pandas.DataFrame
        The concatenated profile table with ``run`` relabelled through
        :func:`run_labels`.
    """
    files = find_profiles(paths)
    df = load_profiles(files)
    df["run"] = df["run"].map(run_labels(files))
    return df


# Categorical colors: crameri's batlowS in its own order, which is designed
# for colour-vision deficiency (adjacent pairs keep a CVD ΔE ≥ 16 in OKLab).
# Its chroma is deliberately muted, so every mark is also labelled directly.
_BATLOW_S = getattr(cm, "batlowS")  # crameri registers its maps dynamically
_PALETTE = [matplotlib.colors.to_hex(_BATLOW_S(i)) for i in (2, 3, 4, 5, 8, 9, 7, 12, 14, 11, 15, 13)]
_OTHER = "#9a9a97"
_INK = "#2b2b2b"


def _style() -> None:
    """
    Set recessive axes and grid and thin marks, once per figure.
    """
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#c8c8c5",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e6e6e3",
            "grid.linewidth": 0.6,
            "font.size": 9,
            "text.color": _INK,
            "axes.labelcolor": _INK,
            "xtick.color": _INK,
            "ytick.color": _INK,
            "legend.frameon": False,
        }
    )


def _wrap(label: str, width: int = 34) -> str:
    """
    Break a run label at commas so it fits a tick.

    Parameters
    ----------
    label : str
        Run label.
    width : int, optional
        Characters per line.

    Returns
    -------
    str
        The label with line breaks.
    """
    if len(label) <= width:
        return label
    return label.replace(", ", ",\n")


def plot_components(breakdown: pd.DataFrame, path: Path, title: str = "Where a model step goes") -> None:
    """
    Stacked bars per run: the time-stepping loop split into PISM's components.

    Uses the mean over the ranks, so the segments add up to the stage. Only
    components above 0.5 % of any run are drawn on their own; the rest is
    folded into ``other``. Each segment above 4 % carries its share.

    Parameters
    ----------
    breakdown : pandas.DataFrame
        Output of :func:`component_breakdown`.
    path : pathlib.Path
        Figure file.
    title : str, optional
        Figure title.
    """
    _style()
    wide = breakdown.pivot(index="run", columns="component", values="time_mean").fillna(0.0)
    order = [c for c in breakdown["component"].unique() if c != "other"]
    keep = [c for c in order if (wide[c] / wide.sum(axis=1)).max() >= 0.005]
    folded = [c for c in order if c not in keep]
    wide["other"] = wide["other"] + wide[folded].sum(axis=1)
    wide = wide[keep + ["other"]]
    colors = dict(zip(keep, _PALETTE)) | {"other": _OTHER}

    fig, ax = plt.subplots(figsize=(8, 0.9 * len(wide) + 1.6))
    left = np.zeros(len(wide))
    y = np.arange(len(wide))
    totals = wide.sum(axis=1).to_numpy()
    for component in wide.columns:
        vals = wide[component].to_numpy()
        ax.barh(
            y, vals, left=left, height=0.55, color=colors[component], edgecolor="white", linewidth=1.5, label=component
        )
        for yi, (x0, v, t) in enumerate(zip(left, vals, totals)):
            if t and v / t >= 0.04:
                ax.text(x0 + v / 2, yi, f"{100 * v / t:.0f}%", ha="center", va="center", fontsize=8, color=_INK)
        left = left + vals
    for yi, t in enumerate(totals):
        ax.text(t, yi, f"  {t:,.0f} s", va="center", fontsize=8, color=_INK)
    ax.set_yticks(y, [_wrap(r) for r in wide.index])
    ax.invert_yaxis()
    ax.set_xlabel("seconds per rank (mean over ranks)")
    ax.set_xlim(0, totals.max() * 1.18)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(
        ncol=min(len(wide.columns), 4),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28 / max(len(wide) * 0.6, 1) - 0.05),
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_phases(table: pd.DataFrame, label_column: str, path: Path, title: str) -> None:
    """
    Grouped bars per phase (or kernel): mean over ranks, with the slowest and fastest rank as whiskers.

    The bars are side by side, not stacked, because the events overlap. The
    share of the Blatter solve is written after each bar.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`blatter_phases` or :func:`blatter_kernels`.
    label_column : str
        ``"phase"`` or ``"kernel"``.
    path : pathlib.Path
        Figure file.
    title : str
        Figure title.
    """
    _style()
    runs = list(table["run"].unique())
    # Largest first, by the biggest mean over the runs, so the eye lands on
    # what matters; the phases keep their Newton-step order.
    if label_column == "kernel":
        labels = list(table.groupby(label_column)["time_mean"].max().sort_values(ascending=False).index)
    else:
        labels = list(table[label_column].unique())
    n_runs = len(runs)
    height = 0.8 / n_runs
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(labels) * max(n_runs, 1) + 1.8))
    y = np.arange(len(labels))
    for i, run in enumerate(runs):
        sub = table[table["run"] == run].set_index(label_column).reindex(labels)
        pos = y - 0.4 + height * (i + 0.5)
        mean = sub["time_mean"].fillna(0).to_numpy()
        lo = (mean - sub["time_min"].fillna(0).to_numpy()).clip(min=0)
        hi = (sub["time_max"].fillna(0).to_numpy() - mean).clip(min=0)
        ax.barh(
            pos, mean, height=height * 0.9, color=_PALETTE[i], edgecolor="white", linewidth=1.0, label=_wrap(run, 60)
        )
        ax.errorbar(mean, pos, xerr=[lo, hi], fmt="none", ecolor=_INK, elinewidth=0.8, capsize=2)
        for yi, (m, x_hi, share) in enumerate(
            zip(mean, sub["time_max"].fillna(0).to_numpy(), sub["share_of_parent"].to_numpy())
        ):
            if m > 0:
                ax.text(x_hi, pos[yi], f"  {100 * share:.0f}%", va="center", fontsize=7.5, color=_INK)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("seconds per rank: mean, whiskers fastest to slowest rank; label = share of the Blatter solve")
    ax.set_xlim(0, table["time_max"].max() * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_rank_balance(times: pd.DataFrame, path: Path, title: str = "Seconds per rank") -> None:
    """
    Small multiples, one per run: seconds per rank for a few events.

    A flat line is a balanced event; a sawtooth means some ranks work while
    others wait. Waiting shows up in the scatter/reduction events.

    Parameters
    ----------
    times : pandas.DataFrame
        Output of :func:`rank_times`.
    path : pathlib.Path
        Figure file.
    title : str, optional
        Figure title.
    """
    _style()
    runs = list(times["run"].unique())
    events = list(times["event"].unique())
    fig, axes = plt.subplots(len(runs), 1, figsize=(8, 2.4 * len(runs) + 0.8), sharex=True, squeeze=False)
    for ax, run in zip(axes[:, 0], runs):
        sub = times[times["run"] == run]
        for j, event in enumerate(events):
            e = sub[sub["event"] == event].sort_values("rank")
            ax.plot(e["rank"], e["time"], color=_PALETTE[j], linewidth=1.8, marker="o", markersize=3, label=event)
            ax.text(e["rank"].iloc[-1], e["time"].iloc[-1], f" {event}", fontsize=7.5, va="center", color=_INK)
        ax.set_title(_wrap(run, 80), loc="left", fontsize=9)
        ax.set_ylabel("seconds")
        ax.set_xlim(-0.5, times["rank"].max() + 0.5 + 0.28 * times["rank"].max())
    axes[-1, 0].set_xlabel("MPI rank")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, ncol=len(events), fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_summary(
    components: pd.DataFrame, phases: pd.DataFrame, kernels: pd.DataFrame, counts: pd.DataFrame, path: Path
) -> str:
    """
    Write the headline numbers as Markdown and return the text.

    Parameters
    ----------
    components : pandas.DataFrame
        Output of :func:`component_breakdown`.
    phases : pandas.DataFrame
        Output of :func:`blatter_phases`.
    kernels : pandas.DataFrame
        Output of :func:`blatter_kernels`.
    counts : pandas.DataFrame
        Output of :func:`solver_counts`.
    path : pathlib.Path
        Markdown file.

    Returns
    -------
    str
        The Markdown text.
    """
    lines = ["# Profile analysis", ""]
    for run in components["run"].unique():
        comp = components[components["run"] == run].set_index("component")
        total = comp["stage_total_mean"].iloc[0]
        blatter = comp.loc["Blatter solve"]
        lines += [f"## {run}", ""]
        lines.append(
            f"- time-stepping loop: {total:,.0f} s per rank (mean), {comp['stage_total_max'].iloc[0]:,.0f} s on the slowest rank"
        )
        lines.append(f"- Blatter solve: {blatter['time_mean']:,.0f} s, {100 * blatter['share_mean']:.0f}% of the loop")
        rest = comp.drop(index=["Blatter solve"]).sort_values("time_mean", ascending=False)
        top = ", ".join(f"{c} {100 * r.share_mean:.0f}%" for c, r in rest.head(3).iterrows() if r.share_mean >= 0.005)
        lines.append(f"- next largest: {top or 'nothing above 0.5%'}")
        cnt = counts[counts["run"] == run].iloc[0]
        lines.append(
            f"- {cnt.steps:.0f} steps, {cnt.newton_per_step:.1f} Newton iterations per step, "
            f"{cnt.ksp_per_newton:.0f} GMRES iterations per Newton iteration, "
            f"{cnt.jacobian_per_newton:.0f} Jacobian assemblies per Newton iteration (one per multigrid level)"
        )
        lines += [
            "",
            "| Blatter phase | mean s | slowest s | fastest s | balance | share |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for _, r in phases[phases["run"] == run].iterrows():
            lines.append(
                f"| {r.phase} | {r.time_mean:,.0f} | {r.time_max:,.0f} | {r.time_min:,.0f} | {r.balance:.2f} | {100 * r.share_of_parent:.0f}% |"
            )
        lines += ["", "| kernel | mean s | slowest s | fastest s | balance | share |", "|---|---:|---:|---:|---:|---:|"]
        for _, r in kernels[kernels["run"] == run].sort_values("time_mean", ascending=False).iterrows():
            lines.append(
                f"| {r.kernel} | {r.time_mean:,.0f} | {r.time_max:,.0f} | {r.time_min:,.0f} | {r.balance:.2f} | {100 * r.share_of_parent:.0f}% |"
            )
        lines.append("")
    lines += [
        "Shares of the Blatter solve are mean over mean; the phases overlap (the line search evaluates the residual),",
        "so they need not add up to 100%. Balance is fastest over slowest rank; 1 is perfect.",
        "",
    ]
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def analyze(
    paths: Sequence[str | Path], output_path: str | Path, stage: str = STAGE, blatter_event: str = BLATTER_EVENT
) -> dict[str, Path]:
    """
    Run the whole analysis: tables, figures and summary into ``output_path``.

    Parameters
    ----------
    paths : sequence of str or pathlib.Path
        Profile files or run directories.
    output_path : str or pathlib.Path
        Directory for the outputs; created if needed.
    stage : str, optional
        Stage to analyse.
    blatter_event : str, optional
        Event holding the Blatter solve.

    Returns
    -------
    dict of str to pathlib.Path
        Written files by name.
    """
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    df = load_runs(paths)
    components = component_breakdown(df, stage, blatter_event)
    phases = blatter_phases(df, stage, blatter_event)
    kernels = blatter_kernels(df, stage, blatter_event)
    counts = solver_counts(df, stage)
    balance_events = [
        e for e in ("SNESJacobianEval", "SNESFunctionEval", "PCSetUp", "VecScatterEnd") if e in set(df["event"])
    ]
    ranks = rank_times(df, balance_events, stage)

    written = {
        "profiles": out / "profiles.csv",
        "components": out / "components.csv",
        "blatter_phases": out / "blatter_phases.csv",
        "blatter_kernels": out / "blatter_kernels.csv",
        "solver_counts": out / "solver_counts.csv",
        "rank_times": out / "rank_times.csv",
        "components_png": out / "components.png",
        "blatter_phases_png": out / "blatter_phases.png",
        "blatter_kernels_png": out / "blatter_kernels.png",
        "rank_balance_png": out / "rank_balance.png",
        "summary": out / "summary.md",
    }
    df.to_csv(written["profiles"], index=False)
    components.to_csv(written["components"], index=False)
    phases.to_csv(written["blatter_phases"], index=False)
    kernels.to_csv(written["blatter_kernels"], index=False)
    counts.to_csv(written["solver_counts"], index=False)
    ranks.to_csv(written["rank_times"], index=False)
    plot_components(components, written["components_png"])
    plot_phases(phases, "phase", written["blatter_phases_png"], "Inside the Blatter solve: Newton-step phases")
    plot_phases(kernels, "kernel", written["blatter_kernels_png"], "Inside the Blatter solve: linear-algebra kernels")
    if not ranks.empty:
        plot_rank_balance(ranks, written["rank_balance_png"])
    print(write_summary(components, phases, kernels, counts, written["summary"]))
    for name, p in written.items():
        print(f"{name}: {p}")
    return written


def cli(argv: Sequence[str] | None = None) -> int:
    """
    Run ``pism-profile-analysis PATHS... --output-path OUT``.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Exit status.
    """
    from argparse import ArgumentParser  # pylint: disable=import-outside-toplevel

    parser = ArgumentParser(
        description=(
            "Cost of the Blatter solve next to PISM's other components, and its breakdown into Newton-step "
            "phases and linear-algebra kernels, from the profile_*.py files written with campaign.profile."
        )
    )
    parser.add_argument(
        "PATHS", nargs="+", help="Profile files, or run directories searched for output/profile/profile_*.py."
    )
    parser.add_argument(
        "--output-path", default="profile_analysis", help="Directory for the CSVs, figures and summary.md."
    )
    parser.add_argument("--stage", default=STAGE, help=f"PETSc stage to analyse (default {STAGE!r}).")
    parser.add_argument(
        "--blatter-event", default=BLATTER_EVENT, help=f"Event holding the Blatter solve (default {BLATTER_EVENT!r})."
    )
    options = parser.parse_args(argv)
    analyze(options.PATHS, options.output_path, options.stage, options.blatter_event)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
