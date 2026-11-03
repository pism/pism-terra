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
Tests for the annual and cumulative balance figures of the USGS benchmark glaciers.

The release comes from ``conftest.py``; nothing touches the network.
"""

import numpy as np
import pandas as pd
import pytest

from pism_terra.glacier import usgs_benchmark_annual as ua


def test_annual_summary_ranks_like_usgs():
    """
    Rank and percentile follow the USGS convention, ties included.
    """
    # The last year ties another; neither counts as ranked above the other.
    ba = pd.Series([0.5, -0.71, -2.0, np.nan, -0.71], index=[2000, 2001, 2002, 2003, 2004])
    summary = ua.annual_summary(ba)
    assert summary["year"] == 2004 and summary["n_years"] == 4 and summary["first_year"] == 2000
    assert summary["rank"] == 2
    assert summary["percentile"] == pytest.approx(100.0 * 2 / 3)
    assert summary["cumulative"] == pytest.approx(-2.92)

    assert ua.annual_summary(pd.Series([1.0, -3.0], index=[1, 2]))["percentile"] == 0.0
    assert ua.annual_summary(pd.Series([-1.0, 3.0], index=[1, 2]))["percentile"] == 100.0


def test_display_name():
    """
    The release's run-together names get their spaces back.
    """
    assert ua.display_name("LemonCreek") == "Lemon Creek"
    assert ua.display_name("Gulkana") == "Gulkana"


def test_cli_end_to_end(tmp_path, usgs_release, monkeypatch):
    """
    The command writes one figure per glacier with balances, plus the summary table.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    usgs_release : dict of str to Path
        The synthetic release directories.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the download.
    """
    monkeypatch.setattr(ua, "download_usgs_benchmark", lambda data_dir, force_overwrite=False: usgs_release)
    output_dir = tmp_path / "out"
    assert ua.cli(["--data-path", str(tmp_path), "--output-path", str(output_dir)]) == 0

    table = pd.read_csv(output_dir / "usgs_annual_balance_summary.csv").set_index("glacier")
    # Bar has point data only.
    assert table.index.tolist() == ["Foo"]
    row = table.loc["Foo"]
    assert row["year"] == 2002 and row["rank"] == 1 and row["n_years"] == 3
    assert row["cumulative"] == pytest.approx(-3.5)
    assert (output_dir / "usgs_annual_balance_Foo.png").exists()
    assert (output_dir / "usgs_annual_balance_Foo.pdf").exists()
