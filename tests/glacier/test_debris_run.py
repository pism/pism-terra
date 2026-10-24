"""
Tests for pointing a glacier run at the staged debris file.

The debris config carries ``debris.transport.input.file = "none"`` as a
placeholder; the run fills it with ``debris_{rgi_id}.nc`` like the other
staged files, and refuses to leave it in place.
"""

from pathlib import Path

import pytest

from pism_terra.glacier.run import _render_forward_run

CONFIG = Path("tests/config/test_debris.toml")
TEMPLATE = Path("pism_terra/templates/debug.j2")
RGI_ID = "RGI2000-v7.0-C-01-04374"
DEBRIS_FILE = f"/in/debris_{RGI_ID}.nc"


def render(tmp_path: Path, debris_file: str | None) -> str:
    """
    Render one forward run script with the staged files and return its text.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    debris_file : str or None
        Staged debris file, or ``None`` when none was staged.

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
    _render_forward_run(
        RGI_ID,
        CONFIG,
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
