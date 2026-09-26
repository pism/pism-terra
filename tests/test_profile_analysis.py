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
Tests for the Blatter cost analysis of ``-profile`` files.
"""

from pathlib import Path

import pytest

from pism_terra.profile_analysis import (
    analyze,
    blatter_phases,
    component_breakdown,
    find_profiles,
    load_runs,
    run_labels,
    solver_counts,
)

# Two ranks; rank 1 does the ice work, rank 0 waits. Times in seconds.
_EVENTS = {
    "summary": (1000.0, 1000.0),
    "stress_balance": (800.0, 800.0),
    "stress_balance.shallow": (780.0, 780.0),
    "SNESSolve": (780.0, 780.0),
    "SNESFunctionEval": (100.0, 300.0),
    "SNESJacobianEval": (50.0, 250.0),
    "PCSetUp": (200.0, 200.0),
    "KSPSolve": (300.0, 300.0),
    "SNESLineSearch": (40.0, 40.0),
    "MatMult": (150.0, 150.0),
    "VecScatterEnd": (390.0, 10.0),
    "KSPGMRESOrthog": (5.0, 5.0),
    "basal_hydrology": (150.0, 150.0),
    "io": (20.0, 20.0),
}
_COUNTS = {"SNESSolve": 10, "KSPSolve": 100, "KSPGMRESOrthog": 5000, "SNESJacobianEval": 300, "SNESFunctionEval": 350}


def _profile_text(scale: float = 1.0) -> str:
    """
    Build a PETSc ascii_info_detail script with the events above.

    Parameters
    ----------
    scale : float, optional
        Factor on every time, to tell two runs apart.

    Returns
    -------
    str
        The script.
    """
    lines = ["size = 2", "Stages = {}", 'Stages["time-stepping loop"] = {}']
    for event, times in _EVENTS.items():
        lines.append(f'Stages["time-stepping loop"]["{event}"] = {{}}')
        for rank, t in enumerate(times):
            count = "" if event == "summary" else f'"count" : {_COUNTS.get(event, 1)}, '
            lines.append(
                f'Stages["time-stepping loop"]["{event}"][{rank}] = {{{count}"time" : {t * scale}, '
                '"numMessages" : 0., "messageLength" : 0., "numReductions" : 0., "flop" : 0.}'
            )
    return "\n".join(lines) + "\n"


@pytest.fixture(name="run_dir")
def fixture_run_dir(tmp_path: Path) -> Path:
    """
    A runner-style output tree with two members and their UQ table.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest scratch directory.

    Returns
    -------
    pathlib.Path
        The run directory.
    """
    out = tmp_path / "run" / "RGI" / "output"
    (out / "profile").mkdir(parents=True)
    (out / "profile" / "profile_g200m_RGI_id_0_uq_0_1986-01-01_1987-01-01.py").write_text(_profile_text())
    (out / "profile" / "profile_g200m_RGI_id_0_uq_1_1986-01-01_1987-01-01.py").write_text(_profile_text(0.5))
    (out / "uq.csv").write_text("uq,hydrology.model\n0,routing\n1,null\n")
    return tmp_path / "run"


def test_runs_are_found_and_labelled_from_the_uq_table(run_dir: Path) -> None:
    """
    Expand directories to their profiles and label them from the UQ table, ``null`` kept as text.

    Parameters
    ----------
    run_dir : pathlib.Path
        The run directory.
    """
    files = find_profiles([run_dir])
    assert len(files) == 2
    labels = run_labels(files)
    assert sorted(labels.values()) == ["uq_0: hydrology.model=routing", "uq_1: hydrology.model=null"]
    assert set(load_runs([run_dir])["run"]) == set(labels.values())
    with pytest.raises(FileNotFoundError):
        find_profiles([run_dir / "nowhere"])


def test_component_breakdown_sets_the_blatter_solve_apart(run_dir: Path) -> None:
    """
    Blatter, the rest of the stress balance, the other components and ``other`` add up to the stage.

    Parameters
    ----------
    run_dir : pathlib.Path
        The run directory.
    """
    df = load_runs([run_dir])
    comp = component_breakdown(df)
    one = comp[comp["run"].str.startswith("uq_0")].set_index("component")
    assert one.loc["Blatter solve", "time_mean"] == 780.0
    assert one.loc["Blatter solve", "share_mean"] == pytest.approx(0.78)
    assert one.loc["stress balance, rest", "time_mean"] == pytest.approx(20.0)
    assert one.loc["basal hydrology", "time_mean"] == 150.0
    assert one.loc["other", "time_mean"] == pytest.approx(1000.0 - 780 - 20 - 150 - 20)
    assert one["time_mean"].sum() == pytest.approx(1000.0)


def test_phases_and_counts(run_dir: Path) -> None:
    """
    Phase shares refer to the Blatter mean; counts give iterations per step and per Newton iteration.

    Parameters
    ----------
    run_dir : pathlib.Path
        The run directory.
    """
    df = load_runs([run_dir])
    phases = blatter_phases(df)
    jac = phases[(phases["run"].str.startswith("uq_0")) & (phases["phase"] == "Jacobian")].iloc[0]
    assert jac["time_mean"] == 150.0
    assert jac["time_max"] == 250.0
    assert jac["balance"] == pytest.approx(0.2)
    assert jac["share_of_parent"] == pytest.approx(150 / 780)
    counts = solver_counts(df).set_index("run")
    row = counts.loc["uq_0: hydrology.model=routing"]
    assert row["newton_per_step"] == 10.0
    assert row["ksp_per_newton"] == 50.0
    assert row["jacobian_per_newton"] == 3.0
    assert row["residual_per_newton"] == 3.5


def test_analyze_writes_tables_figures_and_summary(run_dir: Path, tmp_path: Path) -> None:
    """
    The end-to-end run leaves every advertised file behind.

    Parameters
    ----------
    run_dir : pathlib.Path
        The run directory.
    tmp_path : pathlib.Path
        Pytest scratch directory.
    """
    written = analyze([run_dir], tmp_path / "analysis")
    assert all(p.exists() for p in written.values()), [n for n, p in written.items() if not p.exists()]
    summary = written["summary"].read_text()
    assert "Blatter solve: 780 s, 78% of the loop" in summary
    assert "3 Jacobian assemblies per Newton iteration" in summary
