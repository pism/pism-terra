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
Tests for the Greenland paleo run scripts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pism_terra.config import load_config
from pism_terra.greenland.paleo.run import run_paleo, staged_file_flags
from pism_terra.ismip7.greenland.run import _base_run_dict
from pism_terra.workflow import parse_cdl_options

CONFIG = Path(__file__).resolve().parents[1] / "pism_terra" / "config" / "greenland_paleo.toml"
TEMPLATE = Path(__file__).resolve().parents[1] / "pism_terra" / "templates" / "debug-ismip7.j2"

ROW = {
    "boot_file": "/in/boot.nc",
    "regrid_file": "/in/regrid.nc",
    "grid_file": "/in/grid.nc",
    "heatflux_file": "/in/heatflux.nc",
    "climate_file": "/in/climate.nc",
    "ocean_file": "/in/ocean.nc",
    "delta_T_file": "/in/pism_dT.nc",
    "delta_SL_file": "/in/pism_dSL.nc",
    "ocean_delta_T_file": "/in/pism_ocean_dT.nc",
}


def _flags(script: Path) -> dict[str, str]:
    """
    Read the PISM flags of a rendered script.

    Parameters
    ----------
    script : pathlib.Path
        Rendered job script.

    Returns
    -------
    dict of str to str
        Value per flag, without the leading dash.
    """
    # The first flag shares its line with ``mpirun ... pism``.
    flags = dict(re.findall(r"\s-([A-Za-z_][\w.]*)[ \t]+(\S+)", script.read_text("utf-8")))
    return flags


def test_glacial_cycle_climate_and_tracers(tmp_path):
    """
    The run spans the glacial cycle with paleo climate, sea level and age tracking.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    script = run_paleo(CONFIG, TEMPLATE, path=tmp_path, uq=staged_file_flags(ROW))
    flags = _flags(script)

    assert flags["time.start"] == "-125000"
    assert flags["time.end"] == "0"
    assert flags["time.calendar"] == "365_day"
    assert flags["atmosphere.models"] == "given,delta_T,precip_scaling,elevation_change"
    assert flags["atmosphere.given.file"] == "/in/climate.nc"
    assert flags["atmosphere.delta_T.file"] == "/in/pism_dT.nc"
    assert flags["atmosphere.precip_scaling.file"] == "/in/pism_dT.nc"
    assert flags["atmosphere.elevation_change.file"] == "/in/boot.nc"
    assert flags["surface.models"] == "pdd"
    assert flags["sea_level.models"] == "constant,delta_sl"
    assert flags["ocean.delta_sl.file"] == "/in/pism_dSL.nc"
    assert flags["ocean.models"] == "th,delta_T"
    assert flags["ocean.th.file"] == "/in/ocean.nc"
    assert flags["ocean.delta_T.file"] == "/in/pism_ocean_dT.nc"
    assert flags["age.enabled"] == "yes"
    assert flags["isochrones.deposition_times"] == "200"
    assert flags["input.file"] == "/in/boot.nc"
    assert flags["input.regrid.file"] == "/in/regrid.nc"
    assert "age" not in flags["input.regrid.vars"].split(",")
    assert flags["energy.bedrock_thermal.file"] == "/in/heatflux.nc"
    assert "none" not in flags.values()


def test_output_files_are_named_for_the_span(tmp_path):
    """
    State, spatial, scalar and snapshot files carry resolution and years.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    script = run_paleo(
        CONFIG,
        TEMPLATE,
        path=tmp_path,
        config_cli={"resolution": "18000m", "start": "-125000", "end": "-124900"},
        uq=staged_file_flags(ROW),
        sample="paleo-3",
    )
    flags = _flags(script)

    stem = "g18000m_id_paleo-3_-125000_-124900"
    assert script.name == f"submit_{stem}.sh"
    assert flags["grid.dx"] == "18000m"
    assert flags["time.end"] == "-124900"
    assert flags["output.file"].endswith(f"output/state/state_{stem}.nc")
    assert flags["output.spatial.file"].endswith(f"output/spatial/spatial_{stem}.nc")
    assert flags["output.scalar.file"].endswith(f"output/scalar/scalar_{stem}.nc")
    # No snapshot time falls inside these 100 years, and PISM refuses that.
    assert not [flag for flag in flags if flag.startswith("output.snapshot")]


def test_snapshots_are_limited_to_the_run(tmp_path):
    """
    Only the snapshot times inside the run's span are requested.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    script = run_paleo(CONFIG, TEMPLATE, path=tmp_path, config_cli={"end": "-60000"}, uq=staged_file_flags(ROW))
    flags = _flags(script)

    assert flags["output.snapshot.times"] == "-120000,-100000,-75000"
    assert flags["output.snapshot.size"] == "big_2d"
    assert Path(flags["output.snapshot.file"]).parent == (tmp_path / "output" / "snapshot").resolve()


def test_ensemble_member_overrides_and_model_choice(tmp_path):
    """
    A sampled flag replaces the config's, and a model choice swaps the option table.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    overrides = {"surface.pdd.refreeze": 0.3, "ocean.model": "th", "not.in.config": 1}
    overrides.update(staged_file_flags(ROW))

    flags = _flags(run_paleo(CONFIG, TEMPLATE, path=tmp_path, uq=overrides, sample="paleo-0"))

    assert flags["surface.pdd.refreeze"] == "0.3"
    assert flags["ocean.models"] == "th"
    assert "ocean.delta_T.file" not in flags
    assert "not.in.config" not in flags
    assert "ocean.model" not in flags


def test_shipped_config_uses_known_pism_options():
    """
    Every flag of the shipped config is a parameter of the packaged PISM config.
    """
    cdl = Path.home() / "pism" / "src" / "pism_config.cdl"
    if not cdl.exists():
        pytest.skip("needs a PISM source tree")
    run = _base_run_dict(load_config(CONFIG))
    assert not sorted(k for k in run if k not in parse_cdl_options(cdl))
