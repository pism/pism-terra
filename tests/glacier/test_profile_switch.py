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
Tests for ``campaign.profile`` in the glacier run scripts.

With the switch on, every ``pism`` leg gets ``-profile`` pointed at a file
named after its own state file, so legs and ensemble members never share
one; ``pismi`` has no such option and is left alone.
"""

import re
from pathlib import Path

import pytest

from pism_terra.glacier.run import _render_forward_run, _render_inverse_run

CONFIG = Path("pism_terra/config/s4f_carra2_maffezzoli.toml")
TEMPLATE = Path("pism_terra/templates/debug.j2")
RGI_ID = "RGI2000-v7.0-C-01-04374"


def render(tmp_path: Path, renderer, *, profile: bool) -> str:
    """
    Render one run script from the config, with or without the switch.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    renderer : callable
        ``_render_forward_run`` or ``_render_inverse_run``.
    profile : bool
        Whether to set ``campaign.profile``.

    Returns
    -------
    str
        The rendered submission script.
    """
    # Start from a config without the key, whatever the shipped default is.
    text = (
        "\n".join(line for line in CONFIG.read_text(encoding="utf-8").splitlines() if not line.startswith("profile"))
        + "\n"
    )
    if profile:
        assert text.count("[campaign]\n") == 1
        text = text.replace("[campaign]\n", "[campaign]\n\nprofile = true\n", 1)
    config = tmp_path / "config.toml"
    config.write_text(text, encoding="utf-8")
    renderer(
        RGI_ID,
        config,
        TEMPLATE,
        None,
        path=tmp_path,
        config_cli={"resolution": "1000m"},
        uq={"input.file": "/in/boot.nc", "grid.file": "/in/grid.nc"},
        sample=0,
    )
    return next((tmp_path / RGI_ID / "run_scripts").glob("*.sh")).read_text(encoding="utf-8")


def legs(script: str) -> list[tuple[str, str]]:
    """
    Split a rendered script into ``(executable, flags)`` per PISM call.

    Parameters
    ----------
    script : str
        Rendered submission script.

    Returns
    -------
    list of tuple of str
        Executable name and its flag string, in order.
    """
    chunks = re.split(r"mpirun -np\s+\d+ ", script)[1:]
    return [(chunk.split(maxsplit=1)[0], chunk) for chunk in chunks]


def expected_profile(leg: str) -> str:
    """
    The profile path a leg should carry, derived from its state file.

    Parameters
    ----------
    leg : str
        One PISM call's flags.

    Returns
    -------
    str
        ``<output>/profile/profile_<tag>.py`` for the leg's ``state_<tag>.nc``.
    """
    match = re.search(r"-output\.file (\S+)", leg)
    assert match is not None, leg
    state = Path(match.group(1))
    tag = state.stem.removeprefix("state_")
    return str(state.parent.parent / "profile" / f"profile_{tag}.py")


@pytest.mark.parametrize("renderer", [_render_forward_run, _render_inverse_run], ids=["forward", "inverse"])
def test_every_pism_leg_profiles_into_its_own_file(tmp_path: Path, renderer) -> None:
    """
    Init and main legs get ``-profile`` named after their state file; pismi does not.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest scratch directory.
    renderer : callable
        The run renderer under test.
    """
    calls = legs(render(tmp_path, renderer, profile=True))
    assert [exe for exe, _ in calls].count("pism") == 2
    profiles = []
    for exe, leg in calls:
        match = re.search(r"-profile (\S+)", leg)
        if exe == "pism":
            assert match is not None, leg
            assert match.group(1) == expected_profile(leg)
            profiles.append(match.group(1))
        else:
            assert match is None
    assert len(set(profiles)) == 2
    assert all(Path(p).parent.is_dir() for p in profiles)


def test_switch_is_off_by_default(tmp_path: Path) -> None:
    """
    Without ``campaign.profile`` (removed from the config) no leg carries ``-profile``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest scratch directory.
    """
    assert "-profile" not in render(tmp_path, _render_forward_run, profile=False)
