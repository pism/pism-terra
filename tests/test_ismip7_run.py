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

import pandas as pd
import pytest
import toml

from pism_terra.config import load_config
from pism_terra.inversion import (
    forward_leg_from_inversion,
    inversion_uses_hardav,
    inverted_variables,
)
from pism_terra.ismip7.greenland.run import (
    MEMBER_ID_COLUMNS,
    MEMBERS_CSV,
    _render_forward_run,
    _render_inverse_run,
    is_ismip7_run,
    ismip7_identity,
    record_member,
)
from pism_terra.ismip7.naming import ISMIP7Names, member_ids, split_sample_id

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "pism_terra" / "config"
TEMPLATE_DIR = REPO / "pism_terra" / "templates"

FREE_HY = CONFIG_DIR / "ismip7_greenland_2007_historical_free.toml"
C003 = CONFIG_DIR / "ismip7_greenland_c003.toml"
C009 = CONFIG_DIR / "ismip7_greenland_c009.toml"
C004 = CONFIG_DIR / "ismip7_greenland_c004.toml"
C005 = CONFIG_DIR / "ismip7_greenland_c005.toml"
OUTLINE = REPO / "pism_terra" / "data" / "mouginot_basins_w_shelves.gpkg"
C011 = CONFIG_DIR / "ismip7_greenland_c011.toml"


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
        TEMPLATE_DIR / "debug-ismip7.j2",
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
    Write a copy of the C003 config with the init leg re-timed to 1985-1986.

    The counter configs ship with ``init_start``/``init_end`` (1980-1985);
    the counter tests use a one-year leg starting at the historical epoch.

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
    text = text.replace('init_start = "1980-01-01"', 'init_start = "1985-01-01"', 1)
    text = text.replace('init_end = "1985-01-01"', 'init_end = "1986-01-01"', 1)
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


def test_inverse_alternating_regrids_hardav(tmp_path):
    """
    An alternating tauc/hardav inversion feeds ``hardav`` to the forward leg.

    With ``inverse.design.variable`` naming a pair the forward leg must regrid
    both inverted fields and switch the Blatter solver to the prescribed
    hardness.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory (pytest fixture).
    """
    text = FREE_HY.read_text().replace(
        "[inverse]\n", "[inverse]\n'inverse.design.variable' = \"tauc_hardav\"\n'inverse.alternating_cycles' = 2\n", 1
    )
    cfg = tmp_path / "alternating.toml"
    cfg.write_text(text)

    script = _render_inverse(tmp_path, cfg)
    init, inv, fwd = _legs(script)

    assert "-inverse.alternating_cycles 2" in inv
    assert "-input.regrid.vars tauc,hardav" in fwd
    assert "-stress_balance.averaged_hardness.enabled yes" in fwd
    # The init leg is unaffected.
    assert "averaged_hardness" not in init
    assert "-input.regrid.vars litho_temp,enthalpy,age,tillwat" in init


def test_inverted_variables_follow_the_design_variable():
    """
    The inverted fields come from ``inverse.design.variable`` alone, as in ``pismi``.

    A single variable with a positive cycle count is still a single inversion:
    the forward leg must not ask for a field the inversion never wrote.
    """
    assert inverted_variables({}) == ("tauc",)
    assert inverted_variables({"inverse.design.variable": "tauc", "inverse.alternating_cycles": 1}) == ("tauc",)
    assert inverted_variables({"inverse.design.variable": "hardav", "inverse.alternating_cycles": 1}) == ("hardav",)
    assert inverted_variables({"inverse.design.variable": "tauc_hardav"}) == ("tauc", "hardav")
    assert inverted_variables({"inverse.design.variable": "hardav_tauc"}) == ("hardav", "tauc")
    assert inverted_variables({"inv_design": "hardav"}) == ("hardav",)
    assert inverted_variables({"inverse.design.variable": "tauc", "inv_design": "hardav"}) == ("tauc",)
    with pytest.raises(ValueError, match="inverse.design.variable"):
        inverted_variables({"inverse.design.variable": "usurf"})

    assert not inversion_uses_hardav({"inverse.alternating_cycles": 3})
    assert inversion_uses_hardav({"inverse.design.variable": "hardav_tauc"})


@pytest.mark.parametrize(
    ("design", "regrid", "hardness"),
    [
        ("tauc", "tauc", False),
        ("hardav", "hardav", True),
        ("tauc_hardav", "tauc,hardav", True),
        ("hardav_tauc", "hardav,tauc", True),
    ],
)
def test_forward_leg_regrids_exactly_the_inverted_fields(design, regrid, hardness):
    """
    The forward leg holds fixed what the inversion wrote, and only that.

    Parameters
    ----------
    design : str
        ``inverse.design.variable`` of the inversion.
    regrid : str
        Expected ``input.regrid.vars``.
    hardness : bool
        Whether the Blatter solver is switched to the prescribed hardness.
    """
    run = {"basal_yield_stress.model": "mohr_coulomb", "basal_yield_stress.mohr_coulomb.till_phi_default": 30}
    forward_leg_from_inversion(run, {"inverse.design.variable": design, "inverse.alternating_cycles": 1}, "inv.nc")
    assert run["input.regrid.file"] == "inv.nc"
    assert run["input.regrid.vars"] == regrid
    assert run["basal_yield_stress.model"] == "constant"
    assert not any(k.startswith("basal_yield_stress.mohr_coulomb.") for k in run)
    assert ("stress_balance.averaged_hardness.enabled" in run) == hardness


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


def _render_forward(tmp_path: Path, config_file: Path, outline_file: Path | None = None, **kwargs) -> str:
    """
    Render a forward script into ``tmp_path`` and return the script text.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory (pytest fixture).
    config_file : pathlib.Path
        PISM configuration TOML.
    outline_file : pathlib.Path or None, optional
        Outlines for the post-processing steps; without them those steps are
        not emitted at all.
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
        outline_file,
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


def test_forward_init_leg_surface_model(tmp_path):
    """
    ``campaign.init_surface_model`` swaps the surface model on the init leg only.

    The init leg runs the named ``[surface.options.*]`` table (here
    ``ismip7_forcing`` with force-to-thickness), with its file placeholders
    filled from the staged/UQ overrides; the forward leg keeps the main
    surface model untouched.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    text = FREE_HY.read_text().replace(
        'init_start = "2006-01-01"',
        'init_surface_model = "ismip7_forcing"\ninit_start = "2006-01-01"',
        1,
    )
    text += (
        "\n[surface.options.ismip7_forcing]\n\n"
        "'surface.models' = \"ismip7\"\n"
        "'surface.ismip7.file' = \"none\"\n"
        "'surface.ismip7.gradient.file' = \"none\"\n"
        "'surface.ismip7.reference.file' = \"none\"\n"
        "'surface.force_to_thickness.file' = \"none\"\n"
        "'surface.force_to_thickness.alpha' = 0.95\n"
    )
    cfg = tmp_path / "free_init_surface.toml"
    cfg.write_text(text)

    script = _render_forward(
        tmp_path,
        cfg,
        uq={
            "surface.ismip7.file": "climate_hist.nc",
            "surface.force_to_thickness.file": "boot.nc",
        },
    )
    init, fwd = _legs(script)

    # Init leg: the swapped model's options, placeholders filled from overrides.
    assert "-surface.force_to_thickness.file boot.nc" in init
    assert "-surface.force_to_thickness.alpha 0.95" in init
    assert "-surface.ismip7.file climate_hist.nc" in init
    assert "-time.start 2006-01-01" in init
    assert "-time.end 2007-01-01" in init

    # Forward leg: main surface model untouched, restarts from the init state.
    assert "force_to_thickness" not in fwd
    assert "-surface.ismip7.file climate_hist.nc" in fwd
    init_state = _search(r"-output\.file (\S+state_\S+2006-01-01_2007-01-01\.nc)", init)
    assert f"-input.file {init_state}" in fwd


def test_forward_c011_ocx(tmp_path):
    """
    C011 (OCX) renders init -> one continuous historical leg, no 2015 split.

    The init leg spins up 1980..1990 with ``ismip7_forcing``; the forward leg
    then runs the config's whole ``time.start``..``time.end`` (1990..2025) in
    one PISM invocation, because the reanalysis forcing is a single unbroken
    record. That leg is the ISMIP7 product, named with ``experiment_id``
    ``"OCX"``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    script = _render_forward(
        tmp_path,
        C011,
        sample="OCX",
        uq={
            "surface.ismip7.file": "ocx_climate.nc",
            "surface.force_to_thickness.file": "boot.nc",
        },
    )
    legs = _legs(script)
    assert [leg.split()[0] for leg in legs] == ["pism", "pism"]
    init, fwd = legs

    # Init leg: 1980..1990 with the swapped ismip7_forcing surface model.
    assert "-time.start 1980-01-01" in init
    assert "-time.end 1990-01-01" in init
    assert "-surface.force_to_thickness.file boot.nc" in init
    assert "-surface.ismip7.file ocx_climate.nc" in init

    # Forward leg: restarts from the init state and spans the full config range.
    init_state = _search(r"-output\.file (\S+state_\S+1980-01-01_1990-01-01\.nc)", init)
    assert f"-input.file {init_state}" in fwd
    assert "-time.start 1990-01-01" in fwd
    assert "-time.end 2025-01-01" in fwd
    assert "-time.end 2015-01-01" not in fwd
    assert '-run_info.experiment "OCX"' in fwd
    assert "-surface.ismip7.file ocx_climate.nc" in fwd
    assert "force_to_thickness" not in fwd

    # The one leg is the ISMIP7 product: submission names, spanning 1990-2024.
    assert "/CORE/C011/" in fwd
    assert "_OCX_C011_1990-2024.nc" in fwd


def test_forward_c011_ocx_postprocesses_the_submission_fluxes(tmp_path):
    """
    The OCX single leg still gets the per-basin flux integration.

    Routing C011 through the single-leg path must not cost it the ISMIP7
    post-processing that the two-leg counters get.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    script = _render_forward(tmp_path, C011, OUTLINE, sample="OCX")
    assert _postprocess_commands(script) == ["pism-ismip7-postprocess-flux"]
    flux = next(line for line in script.splitlines() if line.startswith("pism-ismip7-postprocess-flux"))
    assert flux.split()[1].endswith("/CORE/C011")


def test_forward_c009_ctrl(tmp_path):
    """
    C009 (CTRL2015) renders init -> hist -> ctrl product leg ending 2300.

    The control run is set up like any other projection, so the only things
    that distinguish it from C007 are the ``ctrl`` experiment id on the product
    leg and the ctrl forcing files wired into it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    script = _render_forward(
        tmp_path,
        C009,
        sample="CESM2-WACCM",
        proj_overrides={"surface.ismip7.file": "ctrl_climate.nc"},
    )
    _init, hist, proj = _legs(script)

    assert "-time.end 2015-01-01" in hist
    assert '-run_info.experiment "historical"' in hist
    assert "-surface.ismip7.file ctrl_climate.nc" not in hist

    # Control leg: 2015..2300 on the ctrl forcing, submitted as experiment "ctrl".
    hist_state = _search(r"-output\.file (\S+state_\S+_2015-01-01\.nc)", hist)
    assert f"-input.file {hist_state}" in proj
    assert "-time.start 2015-01-01" in proj
    assert "-time.end 2300-01-01" in proj
    assert '-run_info.experiment "ctrl"' in proj
    assert "-surface.ismip7.file ctrl_climate.nc" in proj
    # The ISMIP7 product leg writes into the C009 submission tree.
    assert "GrIS/UAF/PISM/CORE/C009" in proj


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


def test_init_leg_carries_the_bed_deformation_model(tmp_path):
    """
    The init leg runs the same bed deformation model as the legs restarting from it.

    C003 selects Lingle-Clark. On a restart that model reads its displacement
    fields from the input file instead of bootstrapping them, so an init
    state written with the no-op model cannot be continued.

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
    init, _, hist, proj = _legs(script)

    for leg in (init, hist, proj):
        assert "-bed_deformation.model lc" in leg


def test_forward_script_is_named_after_the_counter(tmp_path):
    """
    A counter-driven forward run is named by its Core-experiment counter.

    The counter is what identifies the experiment: it names the config that
    configured the run and the ``CORE/<counter>/`` tree the submission goes
    into, whereas the (GCM, experiment_id) pair it resolves to has to be
    looked up. The init leg keeps the GCM-only tag, since every counter
    sharing a forcing GCM restarts from the same init state.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    cfg = _c003_with_init(tmp_path)
    script_text = _render_forward(
        tmp_path,
        cfg,
        sample="MRI-ESM2-0",
        proj_overrides={"atmosphere.given.file": "proj_climate.nc"},
    )
    (script,) = (tmp_path / "run_scripts").glob("submit_*.sh")
    assert script.name.endswith("_id_C003.sh"), script.name

    init, hist, proj = _legs(script_text)
    # The shared init leg is still keyed on the GCM alone.
    assert "id_MRI-ESM2-0_1985-01-01_1986-01-01" in init
    # The forward legs' flat outputs carry the counter.
    for leg in (hist, proj):
        assert "id_C003_" in leg
        assert "id_MRI-ESM2-0_ssp370" not in leg


def test_forward_script_without_a_counter_keeps_the_pathway_name(tmp_path):
    """
    A single-pathway run has no counter, so it is still named by the pathway.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    _render_forward(tmp_path, FREE_HY, sample=0)
    (script,) = (tmp_path / "run_scripts").glob("submit_*.sh")
    assert script.name.startswith("submit_g")
    assert "_C0" not in script.name


def test_counters_do_not_share_an_init_state(tmp_path):
    """
    Two counters sharing a forcing GCM write their legs to separate trees.

    Every counter runs its own init leg, and that leg's state is named for
    the GCM rather than the counter — so C002 and C004, both MRI-ESM2-0,
    would write one path at once when submitted together. The per-leg
    directories therefore hang off ``output/<counter>/``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    states = {}
    for counter in ("C002", "C004"):
        # C004 is MRI-ESM2-0; re-stamp it as C002 to get the pair that
        # collides, with the init leg shortened to one year.
        text = C004.read_text()
        text = text.replace('init_start = "1980-01-01"', 'init_start = "1985-01-01"', 1)
        text = text.replace('init_end = "1985-01-01"', 'init_end = "1986-01-01"', 1)
        text = text.replace("'run_info.counter' = \"C004\"", f"'run_info.counter' = \"{counter}\"", 1)
        cfg_dir = tmp_path / counter
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg = cfg_dir / "config.toml"
        cfg.write_text(text)
        out = cfg_dir / "run"
        _render_forward(out, cfg, sample="MRI-ESM2-0")
        (script,) = (out / "run_scripts").glob("submit_*.sh")
        init = _legs(script.read_text())[0]
        states[counter] = _search(r"-output\.file (\S+state_\S+1985-01-01_1986-01-01\.nc)", init)

    # Same file name — that is the collision — but under different counters.
    assert Path(states["C002"]).name == Path(states["C004"]).name
    assert Path(states["C002"]).parent != Path(states["C004"]).parent
    assert Path(states["C002"]).parent.parent.name == "C002"
    assert Path(states["C004"]).parent.parent.name == "C004"


def test_run_without_a_counter_keeps_the_flat_output_tree(tmp_path):
    """
    A run with no counter writes to ``output/state`` as before.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    script_text = _render_forward(tmp_path, FREE_HY, sample=0)
    state = _search(r"-output\.file (\S+state_\S+\.nc)", script_text)
    assert Path(state).parent.name == "state"
    assert Path(state).parent.parent.name == "output"


def _single_leg_with_ismip7_naming(tmp_path: Path) -> Path:
    """
    Write a config that is single-leg but still uses ISMIP7 naming.

    Dropping the counter makes the run single-leg; ``output.ISMIP`` stays on,
    which is what gives it an ISMIP7 submission tree. Without a counter the
    config has to state ``time.end`` and ``run_info.experiment`` itself.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory to write the config into.

    Returns
    -------
    pathlib.Path
        Path to the config.
    """
    text = C005.read_text()
    text = re.sub(r"'run_info\.counter'\s*=\s*\"C005\"\n", "", text, count=1)
    text = text.replace("[time]\n", "[time]\n'time.end' = '2100-01-01'\n", 1)
    text = text.replace("'run_info.domain'", "'run_info.experiment' = 'ssp126'\n'run_info.domain'", 1)
    cfg = tmp_path / "single_leg_ismip7.toml"
    cfg.write_text(text)
    return cfg


def _postprocess_commands(script_text: str) -> list[str]:
    """
    List the post-processing commands a rendered script runs.

    Parameters
    ----------
    script_text : str
        The rendered run script.

    Returns
    -------
    list of str
        Command names, in the order they appear.
    """
    wanted = ("pism-postprocess-scalar", "pism-ismip7-postprocess-flux")
    return [line.split()[0] for line in script_text.splitlines() if line.startswith(wanted)]


def test_counter_run_postprocesses_the_submission_fluxes(tmp_path):
    """
    A counter-driven run integrates its submission fluxes over the basins.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    script = _render_forward(tmp_path, C005, OUTLINE, sample="CESM2-WACCM")
    assert _postprocess_commands(script) == ["pism-ismip7-postprocess-flux"]
    # Out of the submission tree, whose contents are checked for conformance.
    flux = next(line for line in script.splitlines() if line.startswith("pism-ismip7-postprocess-flux"))
    source, destination = flux.split()[1], flux.split()[2]
    assert source.endswith("/CORE/C005")
    assert destination.endswith("/output/basins")
    assert "--dim-name region --total-name GIS_GIS" in script


def test_single_leg_ismip7_run_postprocesses_only_the_fluxes(tmp_path):
    """
    An ISMIP7-named single leg gets the flux step, not the scalar one.

    ``pism-postprocess-scalar`` opens its input with ``xr.open_dataset``, and
    the ISMIP7 tree's ``output.spatial.file`` is a ``{var}`` pattern PISM
    expands into one file per variable — there is no single file at that path
    to open. The per-basin numbers come from ``pism-ismip7-postprocess-flux``
    over the submission directory instead.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    cfg = _single_leg_with_ismip7_naming(tmp_path)
    script = _render_forward(tmp_path / "run", cfg, OUTLINE, sample="CESM2-WACCM")
    assert _postprocess_commands(script) == ["pism-ismip7-postprocess-flux"]


def test_single_leg_flat_run_postprocesses_the_spatial_file(tmp_path):
    """
    A flat single leg gets the scalar step and no flux step.

    With ISMIP7 naming off the spatial output is a single combined file, so
    ``pism-postprocess-scalar`` can open it, and there is no submission tree
    for the flux step to walk.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    cfg_path = _single_leg_with_ismip7_naming(tmp_path)
    cfg_path.write_text(cfg_path.read_text().replace("'output.ISMIP' = \"yes\"", "'output.ISMIP' = \"no\""))
    script = _render_forward(tmp_path / "run", cfg_path, OUTLINE, sample="CESM2-WACCM")
    assert _postprocess_commands(script) == ["pism-postprocess-scalar"]
    (command,) = [line for line in script.splitlines() if line.startswith("pism-postprocess-scalar")]
    # ISMIP7 conventions: the region dimension and the GIS_GIS total match the
    # observed mass-balance products, and the file is region_<tag>.nc.
    assert "--dim-name region --total-name GIS_GIS" in command
    assert "/region_" in command.split()[2]


def test_run_without_outlines_postprocesses_nothing(tmp_path):
    """
    Both post-processing steps need outlines to reduce over.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    script = _render_forward(tmp_path, C005, sample="CESM2-WACCM")
    assert _postprocess_commands(script) == []


def test_split_sample_id_separates_forcing_from_draw():
    """
    An ensemble member id carries two different things.

    ``"CESM2-WACCM_uq_3"`` names both the forcing (the ``ESM_id`` field) and
    the parameter draw (which picks ``ISM_member_id``). Passing the whole
    string through put a non-ESM in the ESM slot and produced an
    11-field stem where the convention has 9.
    """
    assert split_sample_id("CESM2-WACCM_uq_3") == ("CESM2-WACCM", 3)
    assert split_sample_id("MRI-ESM2-0_uq_0") == ("MRI-ESM2-0", 0)
    # A plain id has no draw.
    assert split_sample_id("CESM2-WACCM") == ("CESM2-WACCM", None)
    # Not a draw index: leave the id alone rather than truncating it.
    assert split_sample_id("model_uq_beta") == ("model_uq_beta", None)


def test_ppe_member_ids_follow_the_draw_not_the_gcm():
    """
    Every member of a PPE matrix gets its own filename.

    ``ISM_member_id`` identifies the ice sheet configuration, so a given
    parameter draw keeps one ``mNNN`` across every forcing it is run under,
    and two draws never share a name. Deriving the index from the GCM's
    position instead collapsed a 24-run matrix onto 6 names, silently
    overwriting every draw but the last.
    """
    gcms = ["CESM2-WACCM", "MRI-ESM2-0"]

    def stem(sample: str, experiment: str) -> str:
        """
        Name one member the way the renderer does.

        Parameters
        ----------
        sample : str
            Composite ensemble sample id.
        experiment : str
            ISMIP7 experiment id.

        Returns
        -------
        str
            The filename stem.
        """
        esm_id, draw = split_sample_id(sample)
        index = draw if draw is not None else (gcms.index(esm_id) if esm_id in gcms else 0)
        set_counter, ism_member, forcing_member = member_ids("PPE", index)
        return ISMIP7Names(
            "GrIS",
            "UAF",
            "PISM",
            ism_member,
            esm_id,
            forcing_member,
            experiment,
            "PPE",
            set_counter,
            "2015-2100",
        ).stem()

    experiments = ("ssp126", "ssp370", "ssp585")
    stems = [stem(f"{gcm}_uq_{draw}", ssp) for draw in range(4) for gcm in gcms for ssp in experiments]

    assert len(stems) == 24
    assert len(set(stems)) == 24, "ensemble members are overwriting one another"
    assert all(len(s.split("_")) == 9 for s in stems), "the stem must keep the convention's 9 fields"

    # One draw, two forcings: same ice sheet configuration, so same mNNN.
    assert "_m003_CESM2-WACCM_" in stem("CESM2-WACCM_uq_2", "ssp585")
    assert "_m003_MRI-ESM2-0_" in stem("MRI-ESM2-0_uq_2", "ssp585")
    # Two draws under one forcing: different configurations, different mNNN.
    assert "_m001_" in stem("CESM2-WACCM_uq_0", "ssp585")


def _ppe_config(tmp_path: Path, two_leg: bool) -> Path:
    """
    Write a PPE config: a CORE experiment with the counter taken away.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory to write the config into.
    two_leg : bool
        Value of ``campaign.two_leg``.

    Returns
    -------
    pathlib.Path
        Path to the config.
    """
    raw = toml.loads(C005.read_text())
    del raw["run_info"]["run_info.counter"]
    raw["run_info"]["run_info.set"] = "PPE"
    raw["run_info"]["run_info.experiment"] = "ssp585"
    raw["campaign"]["pathway"] = "ssp585"
    raw["campaign"]["two_leg"] = two_leg
    raw["time"]["time.end"] = "2300-01-01"
    path = tmp_path / f"ppe_{two_leg}.toml"
    path.write_text(toml.dumps(raw))
    return path


def test_two_leg_splits_a_counterless_run_at_2015(tmp_path: Path):
    """
    A PPE runs the protocol's two legs even with no CORE counter.

    The two-leg split used to be gated on ``run_info.counter``, which only the
    CORE set has. Without it a PPE rendered one invocation spanning
    1985..2300 under the projection pathway -- handing a 2015-2300 forcing
    file to a run that starts in 1985.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    script = _render_forward(tmp_path, _ppe_config(tmp_path, two_leg=True), sample="CESM2-WACCM_uq_3")
    legs = [leg for leg in _legs(script) if leg.split()[0] == "pism"]
    assert len(legs) == 3, "expected init, historical and projection"
    _init, hist, proj = legs

    assert "-time.start 1985-01-01" in hist and "-time.end 2015-01-01" in hist
    assert '-run_info.experiment "historical"' in hist
    assert "-time.start 2015-01-01" in proj and "-time.end 2300-01-01" in proj
    assert '-run_info.experiment "ssp585"' in proj

    # Both legs are submission products for a PPE, named off the UQ draw.
    assert "_m004_CESM2-WACCM_f001_historical_P004_1985-2014.nc" in hist
    assert "_m004_CESM2-WACCM_f001_ssp585_P004_2015-2299.nc" in proj


def test_without_two_leg_a_counterless_run_stays_single(tmp_path: Path):
    """
    The flag is opt-in, so existing counterless configs are untouched.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    script = _render_forward(tmp_path, _ppe_config(tmp_path, two_leg=False), sample="CESM2-WACCM_uq_3")
    legs = [leg for leg in _legs(script) if leg.split()[0] == "pism"]
    assert len(legs) == 2, "expected init and one forward leg"
    assert "-time.end 2300-01-01" in legs[-1]
    assert "-time.end 2015-01-01" not in legs[-1]


def test_a_core_counter_still_decides_for_itself(tmp_path: Path):
    """
    ``two_leg`` does not override a counter that says it has no split.

    OCX (C011) is counter-driven and single-leg by design; setting the flag
    must not reintroduce a 2015 split it does not want.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    raw = toml.loads(C011.read_text())
    raw["campaign"]["two_leg"] = True
    path = tmp_path / "c011_two_leg.toml"
    path.write_text(toml.dumps(raw))

    script = _render_forward(tmp_path, path, sample="OCX")
    legs = [leg for leg in _legs(script) if leg.split()[0] == "pism"]
    assert len(legs) == 2, "C011 stays init + one continuous leg"
    assert "-time.end 2025-01-01" in legs[-1]


def test_set_counter_is_per_run_and_member_id_is_per_draw(tmp_path: Path):
    """
    A whole PPE matrix lands in distinct submission directories.

    ``set_counter`` "increments with each model run in a set" while
    ``ISM_member_id`` identifies the parameter set, so one draw run under two
    ESMs gets one ``mNNN`` and two ``Pnnn``. Tying them together sent every
    scenario of a draw to one directory, where their identical historical
    legs overwrote one another.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    configs = {
        "ssp126": CONFIG_DIR / "ismip7_greenland_ppe_ssp126.toml",
        "ssp370": CONFIG_DIR / "ismip7_greenland_ppe_ssp370.toml",
        "ssp585": CONFIG_DIR / "ismip7_greenland_ppe_ssp585.toml",
    }
    seen: dict[str, str] = {}
    members: dict[tuple[str, int], set[str]] = {}
    for scenario, config in configs.items():
        run_index = 0
        for gcm in ("CESM2-WACCM", "MRI-ESM2-0"):
            for draw in range(3):
                script = _render_forward(
                    tmp_path / f"{scenario}_{gcm}_{draw}",
                    config,
                    sample=f"{gcm}_uq_{draw}",
                    run_index=run_index,
                )
                found = _search(r"-output\.spatial\.file (\S*/PPE/\S+historical\S+\.nc)", script)
                key = found.split("/output/")[-1]
                assert key not in seen, f"{scenario}/{gcm}/draw{draw} collides with {seen.get(key)}"
                seen[key] = f"{scenario}/{gcm}/draw{draw}"
                members.setdefault((scenario, draw), set()).add(
                    _search(r"/PPE/\S+/\{var\}_GrIS_UAF_PISM_(m\d{3})_", script)
                )
                run_index += 1

    assert len(seen) == 18, "expected one directory per run"
    # One draw keeps one member id whichever ESM it ran under.
    for (_scenario, draw), ids in members.items():
        assert ids == {f"m{draw + 1:03d}"}, f"draw {draw} got {sorted(ids)}"


def test_set_counter_start_keeps_the_scenarios_apart(tmp_path: Path):
    """
    Each scenario config numbers from its own base.

    A set spans several invocations -- one per scenario -- and each numbers
    its runs from the start, so without a per-config base they would all
    begin at P001.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    for scenario, expected in (("ssp126", "P001"), ("ssp370", "P101"), ("ssp585", "P201")):
        script = _render_forward(
            tmp_path / scenario,
            CONFIG_DIR / f"ismip7_greenland_ppe_{scenario}.toml",
            sample="CESM2-WACCM_uq_0",
            run_index=0,
        )
        assert f"/PPE/{expected}/" in script
        # The member id is the draw's, not the counter's.
        assert "_m001_CESM2-WACCM_" in script


def test_member_table_accumulates_across_scenario_invocations(tmp_path: Path):
    """
    One table describes the whole set, not the last invocation.

    The protocol's ``set_counter`` "links to entries in a spreadsheet
    specifying parameter and modelling choices for this particular
    experiment". A set is submitted as one invocation per scenario, so the
    table has to survive the next one.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    draws = {d: {"basal_resistance.pseudo_plastic.q": 0.70 + d / 100} for d in range(3)}
    for scenario in ("ssp126", "ssp370", "ssp585"):
        cfg = load_config(CONFIG_DIR / f"ismip7_greenland_ppe_{scenario}.toml")
        run_index = 0
        for gcm in ("CESM2-WACCM", "MRI-ESM2-0"):
            for draw in range(3):
                record_member(tmp_path, ismip7_identity(cfg, f"{gcm}_uq_{draw}", run_index), draws[draw])
                run_index += 1

    table = pd.read_csv(tmp_path / MEMBERS_CSV)
    assert len(table) == 18
    assert table["set_counter"].nunique() == 18
    assert set(table["experiment_id"]) == {"ssp126", "ssp370", "ssp585"}
    # The identifying columns come first, then the parameters.
    assert list(table.columns)[: len(MEMBER_ID_COLUMNS)] == list(MEMBER_ID_COLUMNS)

    # One draw keeps one member id and one parameter set across the matrix.
    for draw in range(3):
        rows = table[table["uq_draw"] == draw]
        assert len(rows) == 6
        assert set(rows["ism_member_id"]) == {f"m{draw + 1:03d}"}
        assert set(rows["basal_resistance.pseudo_plastic.q"]) == {0.70 + draw / 100}


def test_member_table_replaces_a_rerendered_run(tmp_path: Path):
    """
    Re-rendering a scenario updates its rows instead of duplicating them.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    cfg = load_config(CONFIG_DIR / "ismip7_greenland_ppe_ssp126.toml")
    identity = ismip7_identity(cfg, "CESM2-WACCM_uq_0", 0)
    record_member(tmp_path, identity, {"basal_resistance.pseudo_plastic.q": 0.70})
    record_member(tmp_path, identity, {"basal_resistance.pseudo_plastic.q": 0.88})

    table = pd.read_csv(tmp_path / MEMBERS_CSV)
    assert len(table) == 1
    assert table["basal_resistance.pseudo_plastic.q"].iloc[0] == 0.88


def test_a_core_run_records_its_protocol_counter(tmp_path: Path):
    """
    A Core experiment is keyed by its own counter, not a derived one.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    cfg = load_config(C005)
    identity = ismip7_identity(cfg, "CESM2-WACCM", None)
    assert identity["set_counter"] == "C005"
    assert identity["ism_member_id"] == "m001"
    assert identity["forcing_member_id"] == "f001"
    assert identity["uq_draw"] is None

    record_member(tmp_path, identity, {})
    table = pd.read_csv(tmp_path / MEMBERS_CSV)
    assert table["set_counter"].tolist() == ["C005"]


def test_a_run_that_is_not_an_ismip7_product_records_no_member():
    """
    A config can drive PISM through this module without being a submission.

    The plain OCX config sets ``output.ISMIP = "no"`` and declares no
    ``run_info.set``, so there are no member ids to resolve. Recording the
    member unconditionally made that config fail outright, after its run
    script had already been written.
    """
    plain = load_config(CONFIG_DIR / "ismip7_greenland_ocx.toml")
    assert not is_ismip7_run(plain)
    # Which is why the guard is needed: resolving an identity without a set
    # is an error, not a default.
    with pytest.raises(ValueError, match="set_id must be one of"):
        ismip7_identity(plain, "OCX", 0)

    for name in ("ismip7_greenland_c005", "ismip7_greenland_ppe_ssp126"):
        assert is_ismip7_run(load_config(CONFIG_DIR / f"{name}.toml")), name


def test_is_ismip7_run_needs_both_the_flag_and_a_set(tmp_path: Path):
    """
    Either half missing means there is nothing to name or record.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    raw = toml.loads((CONFIG_DIR / "ismip7_greenland_ppe_ssp126.toml").read_text())

    def written(mutate) -> Path:
        """
        Write a variant config.

        Parameters
        ----------
        mutate : callable
            Applied to the parsed TOML before writing.

        Returns
        -------
        pathlib.Path
            The written config.
        """
        data = toml.loads(toml.dumps(raw))
        mutate(data)
        path = tmp_path / f"variant_{abs(hash(str(data)))}.toml"
        path.write_text(toml.dumps(data))
        return path

    assert is_ismip7_run(load_config(written(lambda d: None)))
    assert not is_ismip7_run(load_config(written(lambda d: d["reporting"].update({"output.ISMIP": "no"}))))
    assert not is_ismip7_run(load_config(written(lambda d: d["run_info"].pop("run_info.set"))))


def _other_hydrology_model(config_file: Path) -> tuple[str, str]:
    """
    Return the config's hydrology model and one other model it has a table for.

    Parameters
    ----------
    config_file : pathlib.Path
        PISM configuration TOML.

    Returns
    -------
    tuple of str
        ``(selected, other)`` model names.
    """
    hydrology = load_config(config_file).hydrology
    other = next(m for m in hydrology.options if m != hydrology.model)
    return hydrology.model, other


def test_forward_uq_row_swaps_the_hydrology_option_table(tmp_path):
    """
    ``hydrology.model`` in a UQ row selects that model's whole option table.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    selected, other = _other_hydrology_model(FREE_HY)
    script = _render_forward(tmp_path, FREE_HY, uq={"hydrology.model": other}, sample=0)
    for leg in _legs(script):
        assert f"-hydrology.model {other}" in leg
        assert f"-hydrology.model {selected}" not in leg
    # The previous model's own option must not linger.
    if selected == "null":
        assert "null_diffuse_till_water" not in script
    else:
        assert "surface_input_from_runoff" not in script


def test_inverse_uq_row_swaps_the_hydrology_option_table(tmp_path):
    """
    The inverse chain's init and forward legs follow a UQ-selected hydrology model.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    selected, other = _other_hydrology_model(FREE_HY)
    script = _render_inverse(tmp_path, FREE_HY, uq={"hydrology.model": other}, sample=0)
    init, _inv, fwd = _legs(script)
    for leg in (init, fwd):
        assert f"-hydrology.model {other}" in leg
        assert f"-hydrology.model {selected}" not in leg


def test_uq_row_naming_an_unknown_model_is_rejected(tmp_path):
    """
    A model with no ``[hydrology.options.*]`` table fails loudly.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    with pytest.raises(ValueError, match="hydrology.model = 'distributed'"):
        _render_forward(tmp_path, FREE_HY, uq={"hydrology.model": "distributed"}, sample=0)


def _with_profile(tmp_path: Path, config_file: Path, enabled: bool = True) -> Path:
    """
    Copy a config with ``campaign.profile`` set one way, whatever it ships with.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest scratch directory.
    config_file : pathlib.Path
        Config to copy.
    enabled : bool, optional
        ``True`` adds ``profile = true``; ``False`` leaves the key out.

    Returns
    -------
    pathlib.Path
        The modified copy.
    """
    text = "\n".join(line for line in config_file.read_text().splitlines() if not line.startswith("profile")) + "\n"
    if enabled:
        assert text.count("[campaign]\n") == 1
        text = text.replace("[campaign]\n", "[campaign]\n\nprofile = true\n", 1)
    copy = tmp_path / ("profile_on.toml" if enabled else "profile_off.toml")
    copy.write_text(text)
    return copy


def _assert_profiled(script: str) -> None:
    """
    Every ``pism`` leg carries ``-profile`` named after its state file; ``pismi`` none.

    Parameters
    ----------
    script : str
        Rendered submission script.
    """
    profiles = []
    for leg in _legs(script):
        profile = re.search(r"-profile (\S+)", leg)
        if leg.split()[0] != "pism":
            assert profile is None
            continue
        assert profile is not None, leg
        state = Path(_search(r"-output\.file (\S+)", leg))
        tag = state.stem.removeprefix("state_")
        assert profile.group(1) == str(state.parent.parent / "profile" / f"profile_{tag}.py")
        profiles.append(profile.group(1))
    assert len(profiles) == len(set(profiles)) >= 2


def test_forward_legs_profile_into_their_own_files(tmp_path):
    """
    ``campaign.profile`` gives the init and forward legs separate ``-profile`` files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    _assert_profiled(_render_forward(tmp_path, _with_profile(tmp_path, FREE_HY)))
    assert "-profile" not in _render_forward(tmp_path / "off", _with_profile(tmp_path, FREE_HY, enabled=False))


def test_inverse_legs_profile_into_their_own_files(tmp_path):
    """
    In the inverse chain the ``pism`` legs profile and the ``pismi`` leg does not.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    _assert_profiled(_render_inverse(tmp_path, _with_profile(tmp_path, FREE_HY)))


def test_cli_uploads_the_output_tree_when_a_bucket_is_given(tmp_path, monkeypatch):
    """
    ``--bucket``/``--bucket-prefix`` sync the whole output path to S3, like the glacier runner.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    monkeypatch : pytest.MonkeyPatch
        Fixture to stub staging, rendering and the upload.
    """
    from pism_terra.ismip7.greenland import run as ismip7  # pylint: disable=import-outside-toplevel

    files = {
        c: f"/in/{c}.nc"
        for c in (
            "boot_file",
            "regrid_file",
            "retreat_file",
            "grid_file",
            "heatflux_file",
            "climate_hist_file",
            "climate_gradient_hist_file",
            "ocean_hist_file",
            "climate_proj_file",
            "climate_gradient_proj_file",
            "ocean_proj_file",
        )
    }
    calls: dict[str, object] = {}
    monkeypatch.setattr(ismip7, "stage", lambda *a, **k: pd.DataFrame([files]))
    monkeypatch.setattr(ismip7, "prepare_observations", lambda *a, **k: None)
    monkeypatch.setattr(ismip7, "_render_forward_run", lambda *a, **k: None)
    monkeypatch.setattr(ismip7, "record_member", lambda *a, **k: None)
    monkeypatch.setattr(
        ismip7, "local_to_s3", lambda src, bucket, prefix: calls.update(upload=(Path(src), bucket, prefix))
    )

    argv = [
        "pism-ismip7-greenland-run-forward",
        "--output-path",
        str(tmp_path),
        str(FREE_HY),
        str(TEMPLATE_DIR / "debug-ismip7.j2"),
    ]
    monkeypatch.setattr("sys.argv", argv)
    ismip7._run(kind="forward")  # pylint: disable=protected-access
    assert "upload" not in calls
    # Project files are snapshotted under the output path, as the glacier runner does.
    assert (tmp_path / "config" / FREE_HY.name).read_text() == FREE_HY.read_text()
    assert (tmp_path / "templates" / "debug-ismip7.j2").exists()

    monkeypatch.setattr(
        "sys.argv", argv + ["--bucket", "pism-cloud-data", "--bucket-prefix", "ismip7/test_ensemble/abc"]
    )
    ismip7._run(kind="forward")  # pylint: disable=protected-access
    assert calls["upload"] == (tmp_path, "pism-cloud-data", "ismip7/test_ensemble/abc")


def test_script_creates_its_output_directories(tmp_path):
    """
    The rendered script makes every output directory the generator made, so a copy staged through S3 works.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary output directory.
    """
    script = _render_forward(tmp_path, C003, sample="CESM2-WACCM")
    (mkdir,) = [line for line in script.splitlines() if line.startswith("mkdir -p ")]
    made = set(mkdir.split()[2:])
    root = tmp_path.resolve()
    for sub in ("output", "output/C003/state", "output/C003/scalar", "output/C003/spatial", "logs"):
        assert str(root / sub) in made, sub
    submission = [d for d in made if d.startswith(str(root / "output" / "GrIS"))]
    assert any(d.endswith("CORE/C003") for d in submission)
    # Every directory the script writes a file into is made before the first leg.
    for output in re.findall(r"-output\.(?:file|scalar\.file) (\S+)", script):
        assert str(Path(output).parent) in made, output
    assert script.index("mkdir -p ") < script.index("mpirun")
