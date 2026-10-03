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
Tests for reading PISM ``-profile`` files.
"""

from pathlib import Path

import pytest

from pism_terra.profiling import (
    STAGE_SUMMARY,
    event_summary,
    load_profile,
    load_profiles,
)

# A trimmed PETSc ascii_info_detail script, as pism -profile writes it. The
# ``PetscBarrier`` event is deliberately never declared with ``= {}``: PETSc
# does that, and it is what breaks a plain import of the file.
PROFILE = """size = 2
LocalTimes = {}
LocalTimes[0] = 100.
LocalTimes[1] = 100.
Stages = {}
Stages["Main Stage"] = {}
Stages["Main Stage"]["summary"] = {}
Stages["Main Stage"]["io.model_state"] = {}
Stages["time-stepping loop"] = {}
Stages["time-stepping loop"]["summary"] = {}
Stages["time-stepping loop"]["stress_balance"] = {}
Stages["Main Stage"]["summary"][0] = {"time" : 5., "numMessages" : 0., "messageLength" : 0., "numReductions" : 0., "flop" : 0.}
Stages["Main Stage"]["summary"][1] = {"time" : 5., "numMessages" : 0., "messageLength" : 0., "numReductions" : 0., "flop" : 0.}
Stages["Main Stage"]["io.model_state"][0] = {"count" : 1, "time" : 2., "syncTime" : 0., "numMessages" : 0., "messageLength" : 0., "numReductions" : 0., "flop" : 0.}
Stages["Main Stage"]["io.model_state"][1] = {"count" : 1, "time" : 2., "syncTime" : 0., "numMessages" : 0., "messageLength" : 0., "numReductions" : 0., "flop" : 0.}
Stages["time-stepping loop"]["summary"][0] = {"time" : 100., "numMessages" : 4., "messageLength" : 8., "numReductions" : 2., "flop" : 0.}
Stages["time-stepping loop"]["summary"][1] = {"time" : 90., "numMessages" : 4., "messageLength" : 8., "numReductions" : 2., "flop" : 0.}
Stages["time-stepping loop"]["stress_balance"][0] = {"count" : 5, "time" : 60., "syncTime" : 0., "numMessages" : 4., "messageLength" : 8., "numReductions" : 2., "flop" : 1e6}
Stages["time-stepping loop"]["stress_balance"][1] = {"count" : 5, "time" : 30., "syncTime" : 0., "numMessages" : 4., "messageLength" : 8., "numReductions" : 2., "flop" : 1e6}
Stages["time-stepping loop"]["PetscBarrier"][0] = {"count" : 0, "time" : 0., "syncTime" : 0., "numMessages" : 0., "messageLength" : 0., "numReductions" : 0., "flop" : 0.}
Stages["time-stepping loop"]["PetscBarrier"][1] = {"count" : 0, "time" : 0., "syncTime" : 0., "numMessages" : 0., "messageLength" : 0., "numReductions" : 0., "flop" : 0.}
"""


@pytest.fixture(name="profile_file")
def fixture_profile_file(tmp_path: Path) -> Path:
    """
    Write the sample profile under a runner-style name.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest scratch directory.

    Returns
    -------
    pathlib.Path
        The profile file.
    """
    path = tmp_path / "profile_g1000m_RGI_id_0_1986-01-01_2025-01-01.py"
    path.write_text(PROFILE, encoding="utf-8")
    return path


def test_load_profile_flattens_every_stage_event_and_rank(profile_file: Path) -> None:
    """
    One row per stage, event and rank, undeclared events included.

    Parameters
    ----------
    profile_file : pathlib.Path
        The sample profile.
    """
    df = load_profile(profile_file)
    assert len(df) == 2 * 5
    assert set(df["stage"]) == {"Main Stage", "time-stepping loop"}
    assert "PetscBarrier" in set(df["event"])
    assert df["run"].unique().tolist() == ["g1000m_RGI_id_0_1986-01-01_2025-01-01"]
    row = df[(df["event"] == "stress_balance") & (df["rank"] == 1)].iloc[0]
    assert row["count"] == 5
    assert row["time"] == 30.0
    assert row["flop"] == 1e6
    summary = df[(df["stage"] == "time-stepping loop") & (df["event"] == STAGE_SUMMARY)]
    assert summary["time"].tolist() == [100.0, 90.0]
    assert summary["count"].isna().all()


def test_load_profile_takes_a_run_label(profile_file: Path) -> None:
    """
    An explicit label replaces the file-derived one.

    Parameters
    ----------
    profile_file : pathlib.Path
        The sample profile.
    """
    assert load_profile(profile_file, run="ntasks_40")["run"].unique().tolist() == ["ntasks_40"]


def test_load_profiles_concatenates_with_distinct_labels(profile_file: Path, tmp_path: Path) -> None:
    """
    Several files stack into one table, told apart by ``run``.

    Parameters
    ----------
    profile_file : pathlib.Path
        The sample profile.
    tmp_path : pathlib.Path
        Pytest scratch directory.
    """
    other = tmp_path / "profile_other.py"
    other.write_text(PROFILE, encoding="utf-8")
    df = load_profiles([profile_file, other])
    assert len(df) == 2 * 10
    assert set(df["run"]) == {"g1000m_RGI_id_0_1986-01-01_2025-01-01", "other"}
    assert load_profiles([]).empty


def test_event_summary_reduces_over_the_ranks(profile_file: Path) -> None:
    """
    Slowest and fastest rank, their ratio and the share of the stage, per event.

    Parameters
    ----------
    profile_file : pathlib.Path
        The sample profile.
    """
    summary = event_summary(load_profile(profile_file)).set_index("event")
    assert STAGE_SUMMARY not in summary.index
    assert summary.index.tolist() == ["stress_balance", "PetscBarrier"]  # by time_max, descending
    sb = summary.loc["stress_balance"]
    assert sb["count"] == 5
    assert sb["time_max"] == 60.0
    assert sb["time_min"] == 30.0
    assert sb["time_mean"] == 45.0
    assert sb["balance"] == 0.5
    assert sb["share"] == pytest.approx(0.6)  # 60 s of the slowest rank's 100 s
    assert summary.loc["PetscBarrier", "balance"] == 0.0


def test_event_summary_rejects_an_unknown_stage(profile_file: Path) -> None:
    """
    A stage the file does not carry is reported with the available ones.

    Parameters
    ----------
    profile_file : pathlib.Path
        The sample profile.
    """
    with pytest.raises(ValueError, match="Main Stage"):
        event_summary(load_profile(profile_file), stage="nope")
