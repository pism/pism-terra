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
Read the profiling files PISM writes with ``-profile``.

``pism -profile FILE`` saves PETSc's detailed log at the end of the run in
the ``ascii_info_detail`` format: a Python script that fills nested dicts
``Stages[stage][event][rank]`` with per-rank counters, plus per-rank totals
(``LocalTimes``, ``LocalMemory``, ...). The runners write one such file per
``pism`` leg when ``campaign.profile`` is set, named after the leg's state
file (see :func:`pism_terra.workflow.add_profile_option`).

The script cannot be imported as a module: PETSc assigns into events it never
declared in the preamble (``PetscBarrier``, ``MatMult MF``), so a plain import
fails with a ``KeyError``. :func:`load_profile` executes it with
auto-vivifying dicts instead, the trick PISM's own ``pism_plot_profiling``
uses, and flattens the result into one tidy table.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

#: Per-rank counters PETSc writes for an event, and the column each becomes.
_COUNTERS = {
    "count": "count",
    "time": "time",
    "syncTime": "sync_time",
    "numMessages": "num_messages",
    "messageLength": "message_length",
    "numReductions": "num_reductions",
    "flop": "flop",
}

#: The pseudo-event carrying a stage's totals.
STAGE_SUMMARY = "summary"


def _autodict() -> collections.defaultdict:
    """
    Make a dict that creates missing entries, recursively, on first access.

    Returns
    -------
    collections.defaultdict
        Nested auto-vivifying dictionary.
    """
    return collections.defaultdict(_autodict)


def _execute(path: Path) -> dict:
    """
    Run a PETSc ``ascii_info_detail`` script and return its namespace.

    Parameters
    ----------
    path : pathlib.Path
        The profiling file.

    Returns
    -------
    dict
        Names defined by the script (``size``, ``Stages``, ``LocalTimes``, ...).
    """
    # Empty-dict initialisers ("= {}") become auto-vivifying dicts so that
    # assignments into undeclared stages/events succeed; the non-empty value
    # literals ("= {...}") are left untouched.
    source = path.read_text(encoding="utf-8").replace("= {}", "= _autodict()")
    namespace: dict = {"_autodict": _autodict}
    exec(compile(source, str(path), "exec"), namespace)  # pylint: disable=exec-used
    return namespace


def load_profile(path: str | Path, run: str | None = None) -> pd.DataFrame:
    """
    Load one ``-profile`` file as a long table, one row per stage, event and rank.

    Parameters
    ----------
    path : str or pathlib.Path
        File written by ``pism -profile``.
    run : str or None, optional
        Label stored in the ``run`` column, to tell files apart once several
        are concatenated. Defaults to the file stem without its ``profile_``
        prefix, which for runner-written files is the leg's state-file tag.

    Returns
    -------
    pandas.DataFrame
        Columns ``run``, ``stage``, ``event``, ``rank``, ``count``, ``time``,
        ``sync_time``, ``num_messages``, ``message_length``,
        ``num_reductions`` and ``flop``. Times are seconds on that rank.
        Every stage also carries the pseudo-event :data:`STAGE_SUMMARY` with
        the stage's totals (its ``count`` is NaN); :func:`event_summary`
        uses it for the time shares and drops it from its output.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if run is None:
        run = path.stem.removeprefix("profile_")
    namespace = _execute(path)
    rows: list[dict] = []
    for stage, events in namespace["Stages"].items():
        for event, ranks in events.items():
            for rank, counters in ranks.items():
                row: dict = {"run": run, "stage": stage, "event": event, "rank": int(rank)}
                row.update({column: counters.get(key) for key, column in _COUNTERS.items()})
                rows.append(row)
    columns = ["run", "stage", "event", "rank", *_COUNTERS.values()]
    df = pd.DataFrame(rows, columns=columns)
    for column in _COUNTERS.values():
        df[column] = pd.to_numeric(df[column])
    return df.sort_values(["run", "stage", "event", "rank"], ignore_index=True)


def load_profiles(paths: Iterable[str | Path]) -> pd.DataFrame:
    """
    Load several ``-profile`` files into one table.

    Parameters
    ----------
    paths : iterable of str or pathlib.Path
        Files written by ``pism -profile``; each is labelled by its own
        default ``run`` label (see :func:`load_profile`).

    Returns
    -------
    pandas.DataFrame
        The concatenated tables, in the order given.
    """
    frames = [load_profile(p) for p in paths]
    if not frames:
        return pd.DataFrame(columns=["run", "stage", "event", "rank", *_COUNTERS.values()])
    return pd.concat(frames, ignore_index=True)


def event_summary(df: pd.DataFrame, stage: str = "time-stepping loop") -> pd.DataFrame:
    """
    Reduce a profile table over the ranks, one row per run and event of a stage.

    Event times are per rank, so the figures of merit are the slowest and the
    fastest rank, their ratio (the load balance) and the slowest rank's share
    of the stage, the numbers PISM's ``pism_plot_profiling`` draws.

    Parameters
    ----------
    df : pandas.DataFrame
        Table from :func:`load_profile` or :func:`load_profiles`.
    stage : str, optional
        Stage to reduce; PISM's main one is ``"time-stepping loop"``.

    Returns
    -------
    pandas.DataFrame
        One row per ``run`` and ``event`` (the stage summary excluded), with
        ``count`` (max over ranks), ``time_max``, ``time_min``, ``time_mean``
        (seconds), ``balance`` (``time_min / time_max``, 1 is perfect) and
        ``share`` (``time_max`` over the stage's slowest-rank total), sorted
        by ``time_max`` descending within each run.

    Raises
    ------
    ValueError
        If ``stage`` is not in the table.
    """
    in_stage = df[df["stage"] == stage]
    if in_stage.empty:
        raise ValueError(f"stage {stage!r} not in profile; available: {sorted(df['stage'].unique())}")
    totals = in_stage[in_stage["event"] == STAGE_SUMMARY].groupby("run")["time"].max()
    events = in_stage[in_stage["event"] != STAGE_SUMMARY]
    grouped = events.groupby(["run", "event"])
    out = pd.DataFrame(
        {
            "count": grouped["count"].max(),
            "time_max": grouped["time"].max(),
            "time_min": grouped["time"].min(),
            "time_mean": grouped["time"].mean(),
        }
    )
    out["balance"] = (out["time_min"] / out["time_max"]).where(out["time_max"] > 0, 0.0)
    out["share"] = out["time_max"] / out.index.get_level_values("run").map(totals).to_numpy()
    return out.reset_index().sort_values(["run", "time_max"], ascending=[True, False], ignore_index=True)
