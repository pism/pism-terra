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
Tests for the ISMIP7 Greenland run-script renderers.

Covers the chained inverse workflow (init/prior -> pismi inversion ->
forward leg(s) with the inverted tauc), the forward workflow's optional
init leg, and the refactor-invariants of the forward renderer. The
renderers never open the data files they reference, so the tests render
into ``tmp_path`` with the shipped debug templates and assert on the
generated script text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pism_terra.ismip7.greenland.run import _render_forward_run, _render_inverse_run

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "pism_terra" / "config"
TEMPLATE_DIR = REPO / "pism_terra" / "templates"

FREE_HY = CONFIG_DIR / "ismip7_greenland_2007_historical_free_hy.toml"
C003 = CONFIG_DIR / "ismip7_greenland_c003.toml"


def _render_inverse(tmp_path: Path, config_file: Path, **kwargs) -> str:
    """
    Render an inverse script into ``tmp_path`` and return the script text.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory (pytest fixture).
    config_file : pathlib.Path
        PISM configuration TOML.
    **kwargs
        Forwarded to :func:`_render_inverse_run`.

    Returns
    -------
    str
        Content of the generated ``run_scripts/submit_*.sh``.
    """
    _render_inverse_run(
        config_file,
        TEMPLATE_DIR / "debug-ismip7-inverse.j2",
        None,
        path=tmp_path,
        **kwargs,
    )
    (script,) = (tmp_path / "run_scripts").glob("submit_*.sh")
    return script.read_text()


def _legs(script: str) -> list[str]:
    """
    Split a rendered debug script into one chunk per mpirun invocation.

    Parameters
    ----------
    script : str
        Rendered script text.

    Returns
    -------
    list of str
        One entry per ``mpirun`` line, starting with the executable name.
    """
    chunks = script.split("mpirun -np")[1:]
    return [re.sub(r"^\s*\d+\s+", "", c) for c in chunks]


def _c003_with_init(tmp_path: Path) -> Path:
    """
    Write a copy of the C003 config with init bounds added to [campaign].

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory to write the modified TOML to.

    Returns
    -------
    pathlib.Path
        Path to the modified config.
    """
    text = C003.read_text()
    text = text.replace(
        "[campaign]",
        '[campaign]\n\ninit_start = "1985-01-01"\ninit_end = "1986-01-01"',
        1,
    )
    cfg = tmp_path / "c003_init.toml"
    cfg.write_text(text)
    return cfg


def _search(pattern: str, text: str) -> str:
    """
    Return the first capture group of ``pattern`` in ``text``.

    Parameters
    ----------
    pattern : str
        Regular expression with one capture group.
    text : str
        Text to search.

    Returns
    -------
    str
        The captured group.
    """
    match = re.search(pattern, text)
    assert match is not None, pattern
    return match.group(1)


def test_inverse_chained_script_single_pathway(tmp_path):
    """
    A non-counter config renders init -> pismi -> single forward leg.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    script = _render_inverse(tmp_path, FREE_HY)
    legs = _legs(script)
    assert [leg.split()[0] for leg in legs] == ["pism", "pismi", "pism"]
    init, inv, fwd = legs

    # Init leg: bootstrap over campaign.init_start..init_end with the staged regrid.
    assert "-time.start 2006-01-01" in init
    assert "-time.end 2007-01-01" in init
    assert "-input.bootstrap yes" in init
    assert "-input.regrid.vars litho_temp,enthalpy,age,tillwat" in init
    assert "-basal_yield_stress.model mohr_coulomb" in init

    # Inversion leg: restarts from the init state, writes into output/inverse/.
    init_state = _search(r"-output\.file (\S+state_\S+2006-01-01_2007-01-01\.nc)", init)
    assert f"-input.file {init_state}" in inv
    inv_file = _search(r"-o (\S+/inverse/inv_\S+2006-01-01_2007-01-01\.nc)", inv)
    assert "mohr_coulomb" in inv

    # Forward leg: restarts from the init state (no bootstrap), regrids the
    # inverted tauc, and uses the constant yield-stress model.
    assert f"-input.file {init_state}" in fwd
    assert "-input.bootstrap" not in fwd
    assert f"-input.regrid.file {inv_file}" in fwd
    assert "-input.regrid.vars tauc" in fwd
    assert "-basal_yield_stress.model constant" in fwd
    assert "mohr_coulomb" not in fwd
    assert "-time.start 2007-01-01" in fwd
    assert "-time.end 2015-01-01" in fwd


def test_inverse_missing_init_bounds_raises(tmp_path):
    """
    Missing campaign.init_start/init_end aborts with a clear message.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    text = "\n".join(
        line for line in FREE_HY.read_text().splitlines() if not line.startswith(("init_start", "init_end"))
    )
    cfg = tmp_path / "no_init.toml"
    cfg.write_text(text)
    with pytest.raises(SystemExit, match="init_start"):
        _render_inverse(tmp_path, cfg)


def test_inverse_counter_adds_projection_leg(tmp_path):
    """
    A counter-driven config appends the projection continuation leg.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    cfg = _c003_with_init(tmp_path)
    script = _render_inverse(
        tmp_path,
        cfg,
        sample="MRI-ESM2-0",
        proj_overrides={"atmosphere.given.file": "proj_climate.nc"},
    )
    legs = _legs(script)
    assert [leg.split()[0] for leg in legs] == ["pism", "pismi", "pism", "pism"]
    init, _, hist, proj = legs

    assert "-time.start 1985-01-01" in init
    assert "-time.end 1986-01-01" in init

    # Historical leg stops at the ISMIP7 split; projection restarts from its state.
    assert "-time.end 2015-01-01" in hist
    hist_state = _search(r"-output\.file (\S+state_\S+1985-01-01_2015-01-01\.nc)", hist)
    assert f"-input.file {hist_state}" in proj
    assert "-time.start 2015-01-01" in proj
    assert "-input.bootstrap" not in proj
    assert "-input.regrid.file" not in proj

    # Both tauc legs drop the mohr_coulomb options; proj forcing hits leg 4 only.
    for leg in (hist, proj):
        assert "-basal_yield_stress.model constant" in leg
        assert "mohr_coulomb" not in leg
    assert "-atmosphere.given.file proj_climate.nc" in proj
    assert "-atmosphere.given.file proj_climate.nc" not in hist

    # ISMIP7 product post-processing is emitted for the projection product leg.
    assert "ismip7-compliance-checker" in script
    assert "postprocess_ismip7_scalar.sh" in script


def test_inverse_cli_start_applies_to_forward_leg_only(tmp_path):
    """
    --start/--end shift the forward legs; the init leg keeps its bounds.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    script = _render_inverse(tmp_path, FREE_HY, config_cli={"start": "2008-01-01"})
    init, _, fwd = _legs(script)
    assert "-time.start 2006-01-01" in init
    assert "-time.end 2007-01-01" in init
    assert "-time.start 2008-01-01" in fwd


def test_inverse_uq_routing(tmp_path):
    """
    UQ keys land only on the legs whose dicts own them.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    script = _render_inverse(
        tmp_path,
        FREE_HY,
        uq={
            "inverse.file": "obs.nc",
            "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_min": 5.0,
        },
    )
    init, inv, fwd = _legs(script)
    assert "-inverse.file obs.nc" in inv
    assert "inverse.file" not in init
    assert "inverse.file" not in fwd
    # mohr_coulomb overrides reach the legs that use the model (1/2) and are
    # purged from the tauc-driven forward leg.
    assert "-basal_yield_stress.mohr_coulomb.topg_to_phi.phi_min 5.0" in init
    assert "-basal_yield_stress.mohr_coulomb.topg_to_phi.phi_min 5.0" in inv
    assert "mohr_coulomb" not in fwd


def _render_forward(tmp_path: Path, config_file: Path, **kwargs) -> str:
    """
    Render a forward script into ``tmp_path`` and return the script text.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory (pytest fixture).
    config_file : pathlib.Path
        PISM configuration TOML.
    **kwargs
        Forwarded to :func:`_render_forward_run`.

    Returns
    -------
    str
        Content of the generated ``run_scripts/submit_*.sh``.
    """
    _render_forward_run(
        config_file,
        TEMPLATE_DIR / "debug-ismip7.j2",
        None,
        path=tmp_path,
        **kwargs,
    )
    (script,) = (tmp_path / "run_scripts").glob("submit_*.sh")
    return script.read_text()


def test_forward_render_invariants_without_init(tmp_path):
    """
    Refactor guard: without init bounds the forward output shape is unchanged.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    text = "\n".join(
        line for line in FREE_HY.read_text().splitlines() if not line.startswith(("init_start", "init_end"))
    )
    cfg = tmp_path / "no_init.toml"
    cfg.write_text(text)
    script = _render_forward(tmp_path, cfg)
    legs = _legs(script)
    assert [leg.split()[0] for leg in legs] == ["pism"]
    (leg,) = legs
    assert "-input.bootstrap yes" in leg
    assert "-time.start 2007-01-01" in leg
    assert "-time.end 2015-01-01" in leg
    assert "-basal_yield_stress.model mohr_coulomb" in leg
    assert '-run_info.experiment "historical"' in leg


def test_forward_init_leg_single_pathway(tmp_path):
    """
    With init bounds the forward run renders init -> forward restart.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    script = _render_forward(tmp_path, FREE_HY)
    legs = _legs(script)
    assert [leg.split()[0] for leg in legs] == ["pism", "pism"]
    init, fwd = legs

    # Init leg: bootstrap over campaign.init_start..init_end with the staged regrid.
    assert "-time.start 2006-01-01" in init
    assert "-time.end 2007-01-01" in init
    assert "-input.bootstrap yes" in init
    assert "-input.regrid.vars litho_temp,enthalpy,age,tillwat" in init

    # Forward leg: restarts from the init state — no bootstrap, no regrid,
    # and (unlike the inverse chain) the basal model is untouched.
    init_state = _search(r"-output\.file (\S+state_\S+2006-01-01_2007-01-01\.nc)", init)
    assert f"-input.file {init_state}" in fwd
    assert "-input.bootstrap" not in fwd
    assert "-input.regrid" not in fwd
    assert "-basal_yield_stress.model mohr_coulomb" in fwd
    assert "-time.start 2007-01-01" in fwd
    assert "-time.end 2015-01-01" in fwd


def test_forward_init_leg_counter(tmp_path):
    """
    A counter-driven config with init bounds renders init -> hist -> proj.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    cfg = _c003_with_init(tmp_path)
    script = _render_forward(
        tmp_path,
        cfg,
        sample="MRI-ESM2-0",
        proj_overrides={"atmosphere.given.file": "proj_climate.nc"},
    )
    legs = _legs(script)
    assert [leg.split()[0] for leg in legs] == ["pism", "pism", "pism"]
    init, hist, proj = legs

    assert "-time.start 1985-01-01" in init
    assert "-time.end 1986-01-01" in init
    assert "-input.bootstrap yes" in init

    # Historical leg restarts from the init state and stops at the ISMIP7 split.
    init_state = _search(r"-output\.file (\S+state_\S+1985-01-01_1986-01-01\.nc)", init)
    assert f"-input.file {init_state}" in hist
    assert "-input.bootstrap" not in hist
    assert "-input.regrid" not in hist
    assert "-time.end 2015-01-01" in hist

    # Projection leg restarts from the historical state with proj forcing.
    hist_state = _search(r"-output\.file (\S+state_\S+1985-01-01_2015-01-01\.nc)", hist)
    assert f"-input.file {hist_state}" in proj
    assert "-time.start 2015-01-01" in proj
    assert "-atmosphere.given.file proj_climate.nc" in proj
    assert "-atmosphere.given.file proj_climate.nc" not in hist
