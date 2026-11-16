"""
Tests for pointing a glacier run at the staged debris file.

The debris config carries ``debris.transport.input.file = "none"`` as a
placeholder; the run fills it with ``debris_{rgi_id}.nc`` like the other
staged files, and refuses to leave it in place.
"""

from pathlib import Path

import pytest

from pism_terra.glacier.run import DEBRIS_SURFACE_FILE_OPTIONS, _render_forward_run

CONFIG = Path("tests/config/test_debris.toml")
TEMPLATE = Path("pism_terra/templates/debug.j2")
RGI_ID = "RGI2000-v7.0-C-01-04374"
DEBRIS_FILE = f"/in/debris_{RGI_ID}.nc"


def render(tmp_path: Path, debris_file: str | None, config: Path = CONFIG) -> str:
    """
    Render one forward run script with the staged files and return its text.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    debris_file : str or None
        Staged debris file, or ``None`` when none was staged.
    config : pathlib.Path, default :data:`CONFIG`
        Run config.

    Returns
    -------
    str
        The rendered submission script.
    """
    uq = {
        "input.file": "/in/boot.nc",
        "grid.file": "/in/grid.nc",
        "atmosphere.given.file": "/in/era5.nc",
        "energy.bedrock_thermal.file": "/in/bheatflux.nc",
    }
    if debris_file is not None:
        uq["debris.transport.input.file"] = debris_file
        # As the run generator does: offered to both, taken where the config has the placeholder.
        for option in DEBRIS_SURFACE_FILE_OPTIONS:
            uq[option] = debris_file
    _render_forward_run(
        RGI_ID,
        config,
        TEMPLATE,
        None,
        path=tmp_path,
        config_cli={"resolution": "200m"},
        uq=uq,
        sample=0,
        init_climate_file=None,
    )
    return next((tmp_path / RGI_ID / "run_scripts").glob("*.sh")).read_text(encoding="utf-8")


def test_every_leg_reads_the_staged_debris_file(tmp_path):
    """
    The placeholder is replaced by the debris file in the init and the main leg.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    script = render(tmp_path, DEBRIS_FILE)
    assert "-debris.models transport" in script
    assert script.count(f"-debris.transport.input.file {DEBRIS_FILE}") == script.count("-debris.models transport")
    assert "-debris.transport.input.file none" not in script


def test_every_leg_scales_ice_melt_with_the_staged_debris_file(tmp_path):
    """
    The melt-scaling placeholder is replaced by the debris file in every leg.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    script = render(tmp_path, DEBRIS_FILE)
    legs = script.count("-surface.models pdd")
    assert legs >= 1
    assert script.count(f"-surface.debris.ice_melt_scaling.file {DEBRIS_FILE}") == legs
    assert "-surface.debris.ice_melt_scaling.file none" not in script
    # PISM refuses both debris inputs at once; the empty thickness option is left at its default.
    assert "-surface.debris.thickness.file" not in script


def test_a_debris_run_without_a_staged_file_is_refused(tmp_path):
    """
    Leaving the placeholder would make PISM open a file called 'none'.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    with pytest.raises(ValueError, match="campaign.debris"):
        render(tmp_path, None)


def test_the_main_leg_carries_the_debris_state_of_the_init_leg(tmp_path):
    """
    Regrid the debris state from the init state when the debris model runs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    script = render(tmp_path, DEBRIS_FILE)
    main = script.split(" pism ")[-1]

    assert "-input.bootstrap yes" in main
    assert "-input.regrid.vars litho_temp,enthalpy,age,tillwat,debris_thickness,englacial_debris_concentration " in main


def thickness_config(tmp_path: Path) -> Path:
    """
    Write the test config with the debris thickness, not the melt factor, as the placeholder.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the config.

    Returns
    -------
    pathlib.Path
        The config.
    """
    text = CONFIG.read_text(encoding="utf-8")
    text = text.replace("'surface.debris.ice_melt_scaling.file' = \"none\"\n", "")
    text = text.replace("'surface.debris.thickness.file' = \"\"", "'surface.debris.thickness.file' = \"none\"")
    config = tmp_path / "test_debris_thickness.toml"
    config.write_text(text, encoding="utf-8")
    return config


def test_every_leg_reads_the_debris_thickness_from_the_staged_file(tmp_path):
    """
    A config with the thickness placeholder gets the debris file there, and no melt factor file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    script = render(tmp_path / "out", DEBRIS_FILE, thickness_config(tmp_path))
    legs = script.count("-surface.models pdd")
    assert legs >= 1
    assert script.count(f"-surface.debris.thickness.file {DEBRIS_FILE}") == legs
    assert "-surface.debris.thickness.file none" not in script
    assert "-surface.debris.ice_melt_scaling.file" not in script


def test_a_thickness_run_without_a_staged_file_is_refused(tmp_path):
    """
    The thickness placeholder left in place stops run generation too.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    with pytest.raises(ValueError, match="surface.debris.thickness.file"):
        render(tmp_path / "out", None, thickness_config(tmp_path))
