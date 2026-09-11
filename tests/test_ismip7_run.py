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

from pism_terra.inversion import inversion_uses_hardav
from pism_terra.ismip7.greenland.run import _render_forward_run, _render_inverse_run

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

    With ``inverse.alternating_cycles > 0`` the forward leg must regrid both
    inverted fields and switch the Blatter solver to the prescribed hardness.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory (pytest fixture).
    """
    text = FREE_HY.read_text().replace("[inverse]\n", "[inverse]\n'inverse.alternating_cycles' = 2\n", 1)
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


def test_inversion_uses_hardav():
    """``inversion_uses_hardav`` recognises alternation and hardness inversions."""
    assert not inversion_uses_hardav({})
    assert not inversion_uses_hardav({"inverse.alternating_cycles": 0, "inv_design": "tauc"})
    assert inversion_uses_hardav({"inverse.alternating_cycles": 3})
    assert inversion_uses_hardav({"inverse.alternating_cycles": "1"})
    assert inversion_uses_hardav({"inv_design": "hardav"})
    assert inversion_uses_hardav({"inverse.design.variable": "hardav"})
    assert not inversion_uses_hardav({"inverse.design.variable": "tauc", "inv_design": "hardav"})
    assert not inversion_uses_hardav({"inverse.alternating_cycles": "not-a-number"})


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
