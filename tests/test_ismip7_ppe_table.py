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
Tests for the PPE appendix workbook.

The workbook is built from the files an ensemble leaves behind: the member
tables the runner writes, the config and UQ copies beside them, and the
ISMIP7 output tree. These tests lay out a miniature PISM-Cloud ensemble
(one job directory per run, every job carrying the whole member table) and
check what the sheets say about it.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from pism_terra.ismip7 import ppe_table
from pism_terra.ismip7.ppe_table import (
    MEMBERS_CSV,
    build_table,
    main,
    physics_legend,
    prior_bounds,
    run_years,
)


@pytest.fixture(autouse=True)
def no_default_posterior(monkeypatch, tmp_path):
    """
    Point the default posterior at a file that does not exist.

    A development checkout ships the 1e6-row posterior sample, which the
    builder would otherwise read for every test.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Patches the module constant.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    monkeypatch.setattr(ppe_table, "DEFAULT_POSTERIOR", tmp_path / "no_posterior.csv.gz")


CONFIG = """
[campaign]
init_start = "1980-01-01"
init_end = "1985-01-01"
retreat_file = "pism_g900m_frontretreat_calfin_1972_2019_MS.nc"

[time]
'time.start' = "1985-01-01"
'time.end' = "2101-01-01"

[grid]
resolution = "900m"

[stress_balance]
model = "hybrid"

[energy]
model = "enthalpy"

[ocean]
model = "plume_sgd"

[frontal_melt]
model = "routing"

[bed_deformation]
model = "lc"

[hydrology]
model = "routing"

[surface]
model = "ismip7"

[calving]
'calving.methods' = "vonmises_calving,thickness_calving"
'calving.thickness_calving.threshold' = 100
"""

UQ = """
samples = 2

['calving.vonmises_calving.sigma_max']
loc = 350000
scale = 400000
distribution = "uniform"

['ocean.plume.melt_rate_parameter']
loc = 1
scale = 4
distribution = "uniform"
"""

MEMBERS = pd.DataFrame(
    {
        "set_counter": ["P001", "P002", "P003", "P004"],
        "ism_member_id": ["m001", "m002", "m001", "m002"],
        "forcing_member_id": ["f001"] * 4,
        "esm_id": ["CESM2-WACCM", "CESM2-WACCM", "MRI-ESM2-0", "MRI-ESM2-0"],
        "experiment_id": ["ssp370"] * 4,
        "set_id": ["PPE"] * 4,
        "uq_draw": [0, 1, 0, 1],
        "stress_balance.sia.enhancement_factor": [2.3, 2.1, 2.3, 2.1],
        "calving.vonmises_calving.sigma_max": [641616.7, 577422.9, 641616.7, 577422.9],
        "ocean.plume.melt_rate_parameter": [1.68, 4.23, 1.68, 4.23],
    }
)


def _touch(path: Path) -> None:
    """
    Create an empty file and its directories.

    Parameters
    ----------
    path : pathlib.Path
        File to create.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def make_ensemble(root: Path) -> Path:
    """
    Lay out a miniature PISM-Cloud ensemble: one job per run, all carrying the member table.

    P001 completed (output through 2100), P002 aborted in the projection leg
    (historical output and a failed state), P003 was resubmitted and completed
    on the second try, P004 never produced output.

    Parameters
    ----------
    root : pathlib.Path
        Directory to fill.

    Returns
    -------
    pathlib.Path
        ``root``.
    """
    jobs = {"job-a": "P001", "job-b": "P002", "job-c": "P003", "job-c2": "P003", "job-d": "P004"}
    for job in jobs:
        d = root / job
        d.mkdir(parents=True)
        MEMBERS.to_csv(d / MEMBERS_CSV, index=False)
        (d / "config").mkdir()
        (d / "config" / "run.toml").write_text(CONFIG)
        (d / "uq").mkdir()
        (d / "uq" / "uq.toml").write_text(UQ)
    out = "output/GrIS/UAF/PISM/PPE"
    for job, counter in (("job-a", "P001"), ("job-c2", "P003")):
        for span, exp in (("1985-2014", "historical"), ("2015-2100", "ssp370")):
            _touch(root / job / out / counter / f"scalar_GrIS_UAF_PISM_m001_ESM_f001_{exp}_{counter}_{span}.nc")
            _touch(root / job / out / counter / f"lithk_GrIS_UAF_PISM_m001_ESM_f001_{exp}_{counter}_{span}.nc")
    _touch(root / "job-b" / out / "P002" / "scalar_GrIS_UAF_PISM_m002_CESM2-WACCM_f001_historical_P002_1985-2014.nc")
    _touch(
        root
        / "job-b"
        / "output/state/state_g900m_id_CESM2-WACCM_uq_1_ssp370_1985-01-01_2015-01-01_mass_transport_failed.nc"
    )
    _touch(
        root / "job-c" / "output/state/state_g900m_id_MRI-ESM2-0_uq_0_1980-01-01_1985-01-01_mass_transport_failed.nc"
    )
    return root


def test_run_years_reads_spinup_start_and_inclusive_end():
    """
    The start is the spin-up leg's first year; an end on 1 January names the previous year.
    """
    import tomllib  # pylint: disable=import-outside-toplevel

    assert run_years(tomllib.loads(CONFIG)) == (1980, 2100)
    assert run_years({"time": {"time.start": "1985-01-01", "time.end": "2100-12-31"}}) == (1985, 2100)
    assert run_years({}) == (None, None)


def test_physics_legend_names_every_model_choice():
    """
    Each model section of the config becomes one readable legend line.
    """
    import tomllib  # pylint: disable=import-outside-toplevel

    lines = physics_legend(tomllib.loads(CONFIG))
    joined = "\n".join(lines)
    assert "hybrid SIA+SSA dynamics" in joined
    assert "PLUME sub-shelf melt with subglacial discharge" in joined
    assert "von Mises calving plus thickness calving (100 m)" in joined
    assert "prescribed front retreat" in joined
    assert "900m horizontal grid" in joined
    assert "Lingle-Clark bed deformation" in joined


def test_prior_bounds():
    """
    Uniform priors span [loc, loc + scale]; other distributions report their extra parameters.
    """
    assert prior_bounds({"distribution": "uniform", "loc": 1, "scale": 4}) == (1.0, 5.0, "")
    lower, upper, extra = prior_bounds({"distribution": "truncnorm", "loc": 0, "scale": 1, "a": -1, "b": 1})
    assert (lower, upper) == (None, None) and extra == "a=-1, b=1"


def test_workbook_describes_the_ensemble_and_its_completion(tmp_path):
    """
    The PPE sheet lists every run with its status; Members and Priors follow the member table.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    root = make_ensemble(tmp_path / "ppe")
    out = build_table(root, tmp_path / "ppe.xlsx")
    wb = openpyxl.load_workbook(out)

    ws = wb["PPE"]
    header = [ws.cell(2, j).value for j in range(1, 10)]
    assert header == [
        "counter_id",
        "experiment_id",
        "Start year",
        "End year",
        "ESM_id",
        "ism_member_id",
        "forcing_member_id",
        "Status",
        "Last output year",
    ]
    rows = {ws.cell(r, 1).value: [ws.cell(r, j).value for j in range(2, 10)] for r in range(3, 7)}
    assert rows["P001"] == ["SSP370", 1980, 2100, "CESM2-WACCM", "m001", "f001", "completed", 2100]
    assert rows["P002"] == ["SSP370", 1980, 2100, "CESM2-WACCM", "m002", "f001", "failed", 2014]
    assert rows["P003"][6:] == ["completed", 2100], "a resubmitted run that finished counts as completed"
    assert rows["P004"][6:] == ["no output", None]
    caption = ws.cell(7, 1).value
    assert caption.startswith("2 of 4 runs ended successfully (output through 2100)")
    assert "2-member perturbed parameter ensemble" in caption
    assert "ism_member_id = m001–m002" in caption

    wm = wb["Members"]
    head = [wm.cell(2, j).value for j in range(1, 9)]
    assert head[:5] == [
        "ism_member_id",
        "uq_draw",
        "counter_id (CESM2-WACCM)",
        "counter_id (MRI-ESM2-0)",
        "Completed runs",
    ]
    # posterior-drawn (not in the UQ TOML) parameters come first
    assert head[5:] == [
        "stress_balance.sia.enhancement_factor",
        "calving.vonmises_calving.sigma_max",
        "ocean.plume.melt_rate_parameter",
    ]
    members = {wm.cell(r, 1).value: [wm.cell(r, j).value for j in range(2, 9)] for r in (4, 5)}
    assert members["m001"] == [0, "P001", "P003", 2, 2.3, 641616.7, 1.68]
    assert members["m002"] == [1, "P002", "P004", 0, 2.1, 577422.9, 4.23]

    wp = wb["Priors"]
    priors = {wp.cell(r, 1).value: [wp.cell(r, j).value for j in range(4, 15)] for r in (3, 4, 5)}
    sigma = priors["calving.vonmises_calving.sigma_max"]
    assert sigma[0] == "uniform" and sigma[1:3] == [350000.0, 400000.0]
    assert sigma[4:6] == [350000.0, 750000.0]  # lower, upper
    assert sigma[9:11] == [577422.9, 641616.7]  # ensemble min, max
    enh = priors["stress_balance.sia.enhancement_factor"]
    assert enh[0] == "posterior" and enh[6:9] == [None, None, None], "no posterior file: no percentiles"
    assert enh[9:11] == [2.1, 2.3]
    assert (
        sum(
            1
            for ws_ in wb.worksheets
            for row in ws_.iter_rows()
            for c in row
            if isinstance(c.value, str) and c.value.startswith("=")
        )
        == 0
    )


def test_completed_only_drops_unfinished_runs(tmp_path):
    """
    With --completed-only the PPE sheet lists the finished runs and Members the members that have one.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    root = make_ensemble(tmp_path / "ppe")
    out = tmp_path / "done.xlsx"
    assert main([str(root), "--output", str(out), "--completed-only"]) == 0
    wb = openpyxl.load_workbook(out)
    ws = wb["PPE"]
    counters = [ws.cell(r, 1).value for r in range(3, 5)]
    assert counters == ["P001", "P003"]
    assert ws.cell(5, 1).value.startswith("Only the 2 of 4 runs that ended successfully")
    wm = wb["Members"]
    assert [wm.cell(4, 1).value, wm.cell(5, 1).value] == ["m001", None]


def test_disagreeing_member_tables_are_an_error(tmp_path):
    """
    Two jobs that disagree about one counter's parameters cannot be merged silently.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    root = make_ensemble(tmp_path / "ppe")
    other = MEMBERS.copy()
    other.loc[0, "ocean.plume.melt_rate_parameter"] = 9.9
    other.to_csv(root / "job-d" / MEMBERS_CSV, index=False)
    with pytest.raises(ValueError, match="disagree about set_counter \\['P001'\\]"):
        build_table(root, tmp_path / "x.xlsx")
