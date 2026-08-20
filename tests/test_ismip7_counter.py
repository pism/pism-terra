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
Tests for the ISMIP7 Core Experiment counter → config expansion.

Covers:
- The ``CORE_EXPERIMENTS`` mapping / ``resolve_counter`` helper.
- ``PismConfig`` expanding ``run_info.counter`` into experiment/pathway/gcms/time.end
  for every shipped ``ismip7_greenland_c00N.toml`` config.
- The staged-forcing filename derived from the expanded fields.
- ``ISMIP7Names`` producing the counter-driven submission path/name.
- ``time.end`` being required only when no counter is set.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pism_terra.config import PismConfig, load_config
from pism_terra.ismip7.experiments import CORE_EXPERIMENTS, resolve_counter
from pism_terra.ismip7.naming import ISMIP7Names

CONFIG_DIR = Path(__file__).resolve().parents[1] / "pism_terra" / "config"

ALL_COUNTERS = [f"C{n:03d}" for n in range(1, 9)]


@pytest.mark.parametrize("counter", ALL_COUNTERS)
def test_config_expands_counter(counter):
    """
    Check each shipped counter config expands to the mapping's derived fields.

    Parameters
    ----------
    counter : str
        ISMIP7 Core Experiment counter id under test (e.g. ``"C003"``).
    """
    spec = CORE_EXPERIMENTS[counter]
    cfg = load_config(CONFIG_DIR / f"ismip7_greenland_{counter.lower()}.toml")

    assert cfg.run_info.counter == counter
    assert cfg.run_info.experiment == spec.experiment_id
    assert cfg.campaign.pathway == spec.pathway
    assert cfg.campaign.gcms == [spec.esm_id]
    assert cfg.time.time_end == f"{spec.proj_end_year}-01-01"


@pytest.mark.parametrize("counter", ALL_COUNTERS)
def test_forcing_filename_from_expanded_fields(counter):
    """
    Check the expanded pathway/gcm/version reproduce the staged forcing filenames.

    Parameters
    ----------
    counter : str
        ISMIP7 Core Experiment counter id under test (e.g. ``"C003"``).
    """
    spec = CORE_EXPERIMENTS[counter]
    cfg = load_config(CONFIG_DIR / f"ismip7_greenland_{counter.lower()}.toml")
    assert isinstance(cfg.campaign.gcms, list)
    gcm = cfg.campaign.gcms[0]
    expected = f"ismip7_greenland_climate_{spec.pathway}_{spec.esm_id}_{cfg.campaign.version}.nc"
    assert f"ismip7_greenland_climate_{cfg.campaign.pathway}_{gcm}_{cfg.campaign.version}.nc" == expected


def test_resolve_counter_unknown_raises():
    """An unsupported counter id raises a helpful ValueError."""
    with pytest.raises(ValueError, match="unknown ISMIP7 counter"):
        resolve_counter("C099")


def test_resolve_counter_case_insensitive():
    """Counter lookup is case-insensitive and trims whitespace."""
    assert resolve_counter(" c003 ") is CORE_EXPERIMENTS["C003"]


def test_historical_counters_use_ssp585_to_2100():
    """C001/C002 keep the historical product but continue on ssp585 to 2100."""
    for counter in ("C001", "C002"):
        spec = CORE_EXPERIMENTS[counter]
        assert spec.experiment_id == "historical"
        assert spec.product_leg == "historical"
        assert spec.pathway == "ssp585"
        assert spec.proj_end_year == 2100


def test_ismip7_names_use_counter():
    """Build the counter-driven ISMIP7 submission directory and file name."""
    names = ISMIP7Names(
        domain_id="GrIS",
        source_id="UAF",
        ism_id="PISM",
        ism_member_id="m001",
        esm_id="CESM2-WACCM",
        forcing_member_id="f001",
        experiment_id="ssp370",
        set_id="CORE",
        set_counter="C003",
        time_range="2015-2100",
    )
    assert names.directory("/root").as_posix().endswith("GrIS/UAF/PISM/CORE/C003")
    assert "_ssp370_C003_" in names.filename("acabf")


def test_time_end_required_without_counter():
    """A config with neither counter nor time.end fails validation."""
    data = {
        "run_info": {"run_info.title": "no end"},
        "campaign": {},
        "time": {"time.start": "1985-01-01"},
        "grid": {"resolution": "900m"},
        "energy": {"model": "enthalpy", "options": {"enthalpy": {}}},
        "stress_balance": {"model": "blatter", "options": {"blatter": {}}},
        "atmosphere": {"model": "given", "options": {"given": {}}},
        "ocean": {"model": "th", "options": {"th": {}}},
        "surface": {"model": "given", "options": {"given": {}}},
        "frontal_melt": {"model": "routing", "options": {"routing": {}}},
        "bed_deformation": {"model": "none", "options": {"none": {}}},
        "hydrology": {"model": "routing", "options": {"routing": {}}},
    }
    with pytest.raises(ValueError, match="time.end is required"):
        PismConfig.model_validate(data)
