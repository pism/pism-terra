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
Tests for the ``--samples`` override of the ensemble runners.

The number of ensemble members lives in the UQ TOML; ``--samples`` lets a
command line trim or enlarge the ensemble without editing the file.
"""

from pathlib import Path

import pandas as pd
import pytest

from pism_terra.glacier import run as glacier_run
from pism_terra.ismip7.greenland import run as ismip7_run

UQ_TOML = """
samples = 8

['surface.pdd.factor_ice']
loc = 0.008
scale = 0.004
distribution = "norm"
"""


@pytest.fixture(name="uq_file")
def fixture_uq_file(tmp_path: Path) -> Path:
    """
    Write a one-parameter UQ file asking for eight samples.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        The UQ TOML.
    """
    uq_file = tmp_path / "uq.toml"
    uq_file.write_text(UQ_TOML)
    return uq_file


@pytest.mark.parametrize("module", [glacier_run, ismip7_run], ids=["glacier", "ismip7"])
def test_samples_override_trims_the_ensemble(module, uq_file, tmp_path):
    """
    ``samples`` wins over the UQ file; omitted, the file's count is used.

    Parameters
    ----------
    module : module
        Runner module under test.
    uq_file : Path
        UQ TOML asking for eight samples.
    tmp_path : Path
        Pytest temporary directory.
    """
    df = pd.DataFrame({"sample": [0], "boot": ["boot.nc"]})

    default = module._build_ensemble_df(df, uq_file, tmp_path, None)  # pylint: disable=protected-access
    assert len(default) == 8

    trimmed = module._build_ensemble_df(df, uq_file, tmp_path, None, samples=3)  # pylint: disable=protected-access
    assert len(trimmed) == 3
    assert len(pd.read_csv(tmp_path / "uq.csv")) == 3
    assert trimmed["sample"].tolist() == ["0_uq_0", "0_uq_1", "0_uq_2"]


@pytest.mark.parametrize("module", [glacier_run, ismip7_run], ids=["glacier", "ismip7"])
def test_samples_flag_is_parsed(module):
    """
    ``--samples`` is an optional integer that defaults to None.

    Parameters
    ----------
    module : module
        Runner module under test.
    """
    positionals = ["RGI2000-v7.0-C-01-00001"] if module is glacier_run else []
    positionals += ["config.toml", "template.j2", "uq.toml"]
    parser = module._build_cli_parser("test", supports_execute=True)  # pylint: disable=protected-access
    assert parser.parse_args(positionals).samples is None
    assert parser.parse_args(["--samples", "5", *positionals]).samples == 5
