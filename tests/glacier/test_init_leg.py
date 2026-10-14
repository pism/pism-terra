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
Tests for the optional init leg in glacier run scripts.

A campaign can spin up on a climatology and then continue on transient
forcing. That means two PISM calls whose only differences are the time bounds,
the forcing file, and where the second one gets its initial state.
"""

from pathlib import Path

import pytest

from pism_terra.config import load_config
from pism_terra.glacier.run import _render_forward_run

CONFIG = Path("tests/config/s4f_pytests.toml")
TEMPLATE = Path("pism_terra/templates/debug.j2")
RGI_ID = "RGI2000-v7.0-C-01-04374"

# Read the spans off the config: they are a modelling choice that changes as
# the campaign is tuned, so pinning literals here would make the tests brittle.
_CFG = load_config(CONFIG)
INIT_START, INIT_END = _CFG.campaign.init_start, _CFG.campaign.init_end
MAIN_START, MAIN_END = _CFG.time.time_start, _CFG.time.time_end
INIT_TAG = f"g1000m_{RGI_ID}_id_0_{INIT_START}_{INIT_END}"
MAIN_TAG = f"g1000m_{RGI_ID}_id_0_{MAIN_START}_{MAIN_END}"

CLIMATE_FILE = "/in/carra2_RGI.nc"
INIT_CLIMATE_FILE = "/in/carra2_monthly_mean_RGI.nc"
BOOT_FILE = "/in/boot.nc"


def render(tmp_path, config=CONFIG, init_climate_file=INIT_CLIMATE_FILE, uq_extra=None, **overrides) -> str:
    """
    Render one forward run script and return its text.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    config : pathlib.Path, optional
        Config TOML to render from.
    init_climate_file : str or None, optional
        Forcing staged for the init leg.
    uq_extra : dict or None, optional
        Staged files on top of the ones every render passes.
    **overrides
        Command-line overrides, e.g. ``init_start="0001-01-01"``.

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
        config_cli={"resolution": "1000m", **overrides},
        uq={
            "input.file": BOOT_FILE,
            "grid.file": "/in/grid.nc",
            "atmosphere.given.file": CLIMATE_FILE,
            "surface.debm_simple.std_dev.file": CLIMATE_FILE,
            "surface.debm_simple.albedo_input.file": CLIMATE_FILE,
            "surface.force_to_thickness.file": BOOT_FILE,
            **(uq_extra or {}),
        },
        sample=0,
        init_climate_file=init_climate_file,
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


@pytest.fixture(name="no_init_config")
def fixture_no_init_config(tmp_path):
    """
    Copy the config with its init keys removed.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.

    Returns
    -------
    pathlib.Path
        The stripped config.
    """
    stripped = "\n".join(
        line
        for line in CONFIG.read_text(encoding="utf-8").splitlines()
        if not line.startswith(("init_climate", "init_start", "init_end"))
    )
    out = tmp_path / "no_init.toml"
    out.write_text(stripped, encoding="utf-8")
    return out


def test_init_bounds_render_two_legs(tmp_path):
    """
    Emit the init call before the run of interest.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    init, main = legs(render(tmp_path))

    assert f"-time.start {INIT_START}" in init and f"-time.end {INIT_END}" in init
    assert f"-time.start {MAIN_START}" in main and f"-time.end {MAIN_END}" in main


def test_only_the_init_leg_bootstraps(tmp_path):
    """
    Restart the main leg from the init state instead of bootstrapping twice.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    init, main = legs(render(tmp_path))

    assert "-input.bootstrap yes" in init
    assert f"-input.file {BOOT_FILE}" in init
    assert "-input.bootstrap" not in main
    assert f"state_{INIT_TAG}.nc" in main


def test_init_leg_uses_the_init_climate(tmp_path):
    """
    Force the init leg with the climatology and the main leg with the transient.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    init, main = legs(render(tmp_path))

    assert f"-atmosphere.given.file {INIT_CLIMATE_FILE}" in init
    assert f"-surface.debm_simple.std_dev.file {INIT_CLIMATE_FILE}" in init
    assert f"-atmosphere.given.file {CLIMATE_FILE}" in main
    assert f"-surface.debm_simple.std_dev.file {CLIMATE_FILE}" in main


def test_init_leg_keeps_the_main_climate_without_an_init_file(tmp_path):
    """
    Leave the forcing alone when no init climate was staged.

    ``init_start``/``init_end`` alone still buy a spin-up; only
    ``init_climate`` changes which forcing it runs on.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    init, _ = legs(render(tmp_path, init_climate_file=None))

    assert f"-atmosphere.given.file {CLIMATE_FILE}" in init


def test_init_outputs_are_named_for_the_init_span(tmp_path):
    """
    Keep the init products from overwriting the main ones.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    init, main = legs(render(tmp_path))

    assert INIT_TAG != MAIN_TAG
    for kind in ("state", "scalar", "spatial"):
        assert f"{kind}_{INIT_TAG}.nc" in init
        assert f"{kind}_{MAIN_TAG}.nc" in main


def test_no_init_bounds_renders_one_leg(tmp_path, no_init_config):
    """
    Leave configs without init bounds exactly as they were.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    no_init_config : pathlib.Path
        The config with its init keys removed.
    """
    script = render(tmp_path, config=no_init_config)

    assert len(legs(script)) == 1
    assert "-input.bootstrap yes" in script
    assert f"-time.end {INIT_END}" not in script


def surface_models(script: str) -> list[str]:
    """
    Pull the ``surface.models`` value out of each PISM call.

    Parameters
    ----------
    script : str
        Rendered submission script.

    Returns
    -------
    list of str
        One entry per leg, in order.
    """
    return [
        next(line.split("-surface.models ")[1].split()[0] for line in leg.splitlines() if "-surface.models " in line)
        for leg in legs(script)
    ]


def test_init_leg_can_use_its_own_surface_model(tmp_path):
    """
    Swap the surface model for the spin-up only.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    init, main = surface_models(render(tmp_path))

    init_model = _CFG.campaign.init_surface_model
    assert init_model is not None, "the packaged config is expected to set init_surface_model"
    assert init == _CFG.surface.options[init_model]["surface.models"]
    assert main == _CFG.surface.selected()["surface.models"]
    assert init != main


def test_swapping_the_surface_model_keeps_the_staged_files(tmp_path):
    """
    Keep resolved paths rather than reverting them to the "none" placeholder.

    The init model's option table declares its forcing files as ``"none"``,
    the value staging fills in. Laying that table down verbatim would unset
    files the main leg had already resolved.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    init, _ = legs(render(tmp_path))

    # Carried over from the main leg, untouched by the swap.
    assert f"-surface.force_to_thickness.file {BOOT_FILE}" in init
    # Still re-pointed at the init climatology by the init_climate swap.
    assert f"-surface.debm_simple.std_dev.file {INIT_CLIMATE_FILE}" in init
    assert "-surface.debm_simple.std_dev.file none" not in init


def test_a_file_only_the_init_surface_model_reads_is_staged(tmp_path):
    """
    Fill in the force-to-thickness file when only the init leg has forcing.

    The staged files are filtered against the main leg's options. With a main
    leg on ``pdd`` and an init leg on ``pdd,forcing`` that dropped the boot
    file, and the init leg ran with ``-surface.force_to_thickness.file none``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    config = Path("tests/config/test_init_surface_model.toml")
    cfg = load_config(config)
    assert "surface.force_to_thickness.file" not in cfg.surface.selected()
    init_model = cfg.campaign.init_surface_model
    assert init_model is not None
    assert "surface.force_to_thickness.file" in cfg.surface.options[init_model]

    init, main = legs(render(tmp_path, config=config))

    assert f"-surface.force_to_thickness.file {BOOT_FILE}" in init
    assert "-surface.force_to_thickness.file none" not in init
    assert "-surface.force_to_thickness" not in main


def test_the_init_leg_can_write_spatial_fields_less_often(tmp_path):
    """
    Set ``output.spatial.times`` of the init leg alone from ``campaign.init_spatial_times``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    main_times = _CFG.reporting["output.spatial.times"]
    assert main_times != 10
    config = tmp_path / "sparse_init.toml"
    config.write_text(
        CONFIG.read_text(encoding="utf-8").replace("[campaign]\n", "[campaign]\ninit_spatial_times = 10\n", 1),
        encoding="utf-8",
    )
    assert load_config(config).campaign.init_spatial_times == 10

    init, main = legs(render(tmp_path, config=config))

    assert "-output.spatial.times 10 " in init
    assert f"-output.spatial.times {main_times} " in main


def test_without_init_spatial_times_both_legs_write_alike(tmp_path):
    """
    Keep the config's ``output.spatial.times`` in the init leg when the key is unset.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    assert _CFG.campaign.init_spatial_times is None
    main_times = _CFG.reporting["output.spatial.times"]

    init, main = legs(render(tmp_path))

    assert f"-output.spatial.times {main_times} " in init
    assert f"-output.spatial.times {main_times} " in main


def test_the_lapse_rate_correction_reads_the_init_climate(tmp_path):
    """
    Point ``atmosphere.elevation_change.file`` at the climate each leg runs on.

    The correction reads the surface of the forcing it corrects. The init leg
    used to keep the transient forcing there while ``atmosphere.given.file``
    read the climatology.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    config = Path("tests/config/test_init_surface_model.toml")
    assert "elevation_change" in load_config(config).atmosphere.selected()["atmosphere.models"]

    init, main = legs(render(tmp_path, config=config, uq_extra={"atmosphere.elevation_change.file": CLIMATE_FILE}))

    assert f"-atmosphere.elevation_change.file {INIT_CLIMATE_FILE}" in init
    assert f"-atmosphere.given.file {INIT_CLIMATE_FILE}" in init
    assert f"-atmosphere.elevation_change.file {CLIMATE_FILE}" in main
    assert f"-atmosphere.given.file {CLIMATE_FILE}" in main


def test_an_unknown_init_surface_model_is_rejected(tmp_path):
    """
    Name the available surface models rather than failing inside PISM.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    broken = "\n".join(
        'init_surface_model = "does_not_exist"' if line.startswith("init_surface_model") else line
        for line in CONFIG.read_text(encoding="utf-8").splitlines()
    )
    config = tmp_path / "broken.toml"
    config.write_text(broken, encoding="utf-8")

    with pytest.raises(ValueError, match="names no \\[surface.options"):
        render(tmp_path, config=config)


def test_without_an_init_surface_model_both_legs_agree(tmp_path):
    """
    Leave the surface model alone when the campaign does not override it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    stripped = "\n".join(
        line for line in CONFIG.read_text(encoding="utf-8").splitlines() if not line.startswith("init_surface_model")
    )
    config = tmp_path / "no_init_surface.toml"
    config.write_text(stripped, encoding="utf-8")

    init, main = surface_models(render(tmp_path, config=config))

    assert init == main == _CFG.surface.selected()["surface.models"]


def test_the_run_toml_is_no_longer_written(tmp_path):
    """
    Emit the post-processing commands directly instead of a run TOML.

    `pism-glacier-postprocess` read that TOML; the run script now calls
    `pism-postprocess-scalar` with the outline files positionally, so the
    TOML has no reader left.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    script = render(tmp_path)

    assert not (tmp_path / RGI_ID / "output" / "post_processing").exists()
    # The dh extraction (``pism-glacier-postprocess-dh``) is a different tool
    # that legitimately appears in the script; only the legacy TOML-reading
    # command must be gone.
    assert "pism-glacier-postprocess " not in script


def test_the_command_line_moves_the_init_leg(tmp_path):
    """
    Let ``--init-start`` and ``--init-end`` replace the config's init span.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    init, main = legs(render(tmp_path, init_start="0005-01-01", init_end="0105-01-01"))

    assert "-time.start 0005-01-01" in init and "-time.end 0105-01-01" in init
    # The init products are named for the span that was run, and the main leg restarts from them.
    tag = f"g1000m_{RGI_ID}_id_0_0005-01-01_0105-01-01"
    assert f"state_{tag}.nc" in init and "-input.file" in main and f"state_{tag}.nc" in main
    # The main leg keeps the config's own span.
    assert f"-time.start {MAIN_START}" in main and f"-time.end {MAIN_END}" in main


def test_one_bound_from_the_command_line_keeps_the_other_from_the_config(tmp_path):
    """
    Override one end of the init span and take the other from ``[campaign]``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    init, _ = legs(render(tmp_path, init_end="0777-01-01"))

    assert f"-time.start {INIT_START}" in init and "-time.end 0777-01-01" in init


def test_the_command_line_adds_an_init_leg_to_a_config_without_one(tmp_path, no_init_config):
    """
    Render an init leg from the command line alone, and refuse half a span.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    no_init_config : pathlib.Path
        The config with its init keys removed.
    """
    script = render(tmp_path / "both", config=no_init_config, init_start="0001-01-01", init_end="0011-01-01")
    init, main = legs(script)
    assert "-time.start 0001-01-01" in init and "-time.end 0011-01-01" in init
    assert "-input.bootstrap yes" in init and "-input.bootstrap yes" not in main

    # One bound with nothing to pair it with would silently render no init leg.
    with pytest.raises(SystemExit, match="--init-start needs --init-end"):
        render(tmp_path / "half", config=no_init_config, init_start="0001-01-01")


def test_the_init_leg_runs_on_a_365_day_calendar_from_year_one(tmp_path):
    """
    Give the init leg the time axis of the climatology it cycles, and leave the main leg's alone.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    init, main = legs(render(tmp_path))

    assert "-time.calendar 365_day" in init
    assert "-time.reference_date 0001-01-01" in init
    assert f"-time.start {INIT_START}" in init
    # The main leg keeps what the config's [time] section says.
    calendar, reference_date = _CFG.time.as_params()["time.calendar"], _CFG.time.as_params()["time.reference_date"]
    assert (calendar, reference_date) != ("365_day", "0001-01-01")
    assert f"-time.calendar {calendar}" in main and f"-time.reference_date {reference_date}" in main
    assert "365_day" not in main
