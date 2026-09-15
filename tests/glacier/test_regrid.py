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
Tests for regridding a glacier run from a spin-up state.

A long single spin-up seeds an ensemble on another grid: the ensemble keeps
bootstrapping from the boot file and regrids the thermal state from the
spin-up. The regrid belongs to whichever leg bootstraps, and never to a leg
that restarts from a state file.
"""

from pathlib import Path

import pytest

from pism_terra.glacier.run import (
    DEFAULT_REGRID_VARS,
    _render_forward_run,
    _resolve_regrid_file,
)

CONFIG = Path("pism_terra/config/s4f_carra2_maffezzoli.toml")
TEMPLATE = Path("pism_terra/templates/debug.j2")
RGI_ID = "RGI2000-v7.0-C-01-04374"

CLIMATE_FILE = "/in/carra2_RGI.nc"
BOOT_FILE = "/in/boot.nc"
STATE_FILE = "/out/state_g500m_RGI_id_0_0001-01-01_0501-01-01.nc"


def render(tmp_path, config=CONFIG, regrid_file=None) -> str:
    """
    Render one forward run script and return its text.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    config : pathlib.Path, optional
        Config TOML to render from.
    regrid_file : str or None, optional
        Spin-up state to regrid from, as resolved by the CLI.

    Returns
    -------
    str
        The rendered submission script.
    """
    _render_forward_run(
        RGI_ID,
        config,
        TEMPLATE,
        None,
        path=tmp_path,
        config_cli={"resolution": "1000m", "regrid_file": regrid_file},
        uq={
            "input.file": BOOT_FILE,
            "grid.file": "/in/grid.nc",
            "atmosphere.given.file": CLIMATE_FILE,
            "surface.debm_simple.std_dev.file": CLIMATE_FILE,
            "surface.debm_simple.albedo_input.file": CLIMATE_FILE,
            "surface.force_to_thickness.file": BOOT_FILE,
        },
        sample=0,
        init_climate_file=None,
    )
    return next((tmp_path / RGI_ID / "run_scripts").glob("*.sh")).read_text(encoding="utf-8")


def legs(script: str) -> list[str]:
    """
    Split a rendered script into its PISM invocations.

    Parameters
    ----------
    script : str
        Rendered submission script.

    Returns
    -------
    list of str
        One entry per ``pism`` call, in order.
    """
    return list(script.split("mpirun -np  8 pism ")[1:])


def strip_config(tmp_path, *, drop_init: bool, regrid_vars: str | None = None) -> Path:
    """
    Copy the config, optionally without its init keys and with a regrid list.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    drop_init : bool
        Remove ``init_*`` from ``[campaign]`` so the run has a single leg.
    regrid_vars : str or None, optional
        Value for ``input.regrid.vars`` in ``['input']``.

    Returns
    -------
    pathlib.Path
        The rewritten config.
    """
    lines = []
    for line in CONFIG.read_text(encoding="utf-8").splitlines():
        if drop_init and line.startswith(("init_climate", "init_start", "init_end", "init_surface_model")):
            continue
        lines.append(line)
        if line.strip() == "['input']" and regrid_vars is not None:
            lines.append(f"'input.regrid.vars' = \"{regrid_vars}\"")
    out = tmp_path / "config.toml"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def test_without_a_regrid_file_nothing_is_regridded(tmp_path):
    """
    Drop a declared ``input.regrid.vars`` too, so placeholders never reach PISM.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    config = strip_config(tmp_path, drop_init=True, regrid_vars=DEFAULT_REGRID_VARS)
    (main,) = legs(render(tmp_path, config=config))

    assert "-input.bootstrap yes" in main
    assert "-input.regrid" not in main


def test_a_single_leg_bootstraps_and_regrids(tmp_path):
    """
    Keep the boot file for the geometry and pull the thermal state from the spin-up.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    config = strip_config(tmp_path, drop_init=True)
    (main,) = legs(render(tmp_path, config=config, regrid_file=STATE_FILE))

    assert "-input.bootstrap yes" in main
    assert f"-input.file {BOOT_FILE}" in main
    assert f"-input.regrid.file {STATE_FILE}" in main
    assert f"-input.regrid.vars {DEFAULT_REGRID_VARS}" in main


def test_only_the_init_leg_regrids(tmp_path):
    """
    The init leg bootstraps with the regrid; the main leg restarts from its state.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    init, main = legs(render(tmp_path, regrid_file=STATE_FILE))

    assert "-input.bootstrap yes" in init
    assert f"-input.regrid.file {STATE_FILE}" in init
    assert "-input.bootstrap" not in main
    assert "-input.regrid" not in main
    assert "state_g1000m" in main


def test_the_config_can_pin_the_regridded_fields(tmp_path):
    """
    ``input.regrid.vars`` in ``['input']`` wins over the default list.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    config = strip_config(tmp_path, drop_init=True, regrid_vars="thk,enthalpy")
    (main,) = legs(render(tmp_path, config=config, regrid_file=STATE_FILE))

    assert "-input.regrid.vars thk,enthalpy" in main
    assert DEFAULT_REGRID_VARS not in main


@pytest.mark.parametrize("spec", [None, "none", "None", " NONE "])
def test_no_spec_means_no_regrid(tmp_path, spec):
    """
    ``None`` and the ``"none"`` placeholder both mean bootstrap only.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    spec : str or None
        The CLI/config value under test.
    """
    assert _resolve_regrid_file(spec, tmp_path) is None


def test_a_missing_state_file_fails_before_staging(tmp_path):
    """
    Point at the expected location rather than letting PISM discover the gap.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    with pytest.raises(FileNotFoundError, match="output/state"):
        _resolve_regrid_file(tmp_path / "missing.nc", tmp_path)


def test_an_existing_state_file_resolves(tmp_path):
    """
    A relative local path comes back absolute.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    state = tmp_path / "state.nc"
    state.touch()

    assert _resolve_regrid_file(str(state), tmp_path) == state.resolve()
