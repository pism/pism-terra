# Copyright (C) 2025 Andy Aschwanden
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
Test glacier.run functions.
"""

from pathlib import Path

import pytest

from pism_terra.glacier.run import (
    DH_END,
    DH_START,
    _dh_command,
    _nullable_string,
    _postprocess_commands,
)


@pytest.mark.parametrize(
    "argument_string,expected",
    [
        ("None", None),
        ("none", None),
        (" NONE ", None),
        ("foobar", "foobar"),
    ],
)
def test_nullable_string(argument_string, expected) -> None:
    """
    Pytest for the glacier.run._nullable_string function.

    Parameters
    ----------
    argument_string : str
        String to be tested.
    expected : str | None
        Expected value to be returned.
    """
    assert _nullable_string(argument_string) == expected


OUTLINE_C = "/in/rgi_RGI2000-v7.0-C-01-04374-C.gpkg"
OUTLINE_G = "/in/rgi_RGI2000-v7.0-C-01-04374-G.gpkg"
SPATIAL = Path("/out/spatial/spatial_g500m_RGI2000-v7.0-C-01-04374_id_0_0001-01-01_0501-01-01.nc")


def test_both_outlines_are_reduced_into_separate_files():
    """
    Reduce against the complex and the per-glacier outlines, without collision.

    Both read the same spatial file, so letting the tool derive the output
    name would have the second run overwrite the first.
    """
    commands = _postprocess_commands(SPATIAL, Path("/out"), OUTLINE_C, OUTLINE_G, {}).splitlines()

    assert len(commands) == 2
    assert all(c.startswith("pism-postprocess-scalar ") for c in commands)
    assert OUTLINE_C in commands[0] and OUTLINE_G in commands[1]

    stem = "g500m_RGI2000-v7.0-C-01-04374_id_0_0001-01-01_0501-01-01.nc"
    assert f"/out/processed_scalar/scalar_C_{stem}" in commands[0]
    assert f"/out/processed_scalar/scalar_G_{stem}" in commands[1]


def test_postprocess_workers_are_clamped():
    """
    Keep a wide PISM decomposition from becoming that many Dask workers.
    """
    wide = _postprocess_commands(SPATIAL, Path("/out"), OUTLINE_C, OUTLINE_G, {"ntasks": 120})
    narrow = _postprocess_commands(SPATIAL, Path("/out"), OUTLINE_C, OUTLINE_G, {"ntasks": 2})
    unset = _postprocess_commands(SPATIAL, Path("/out"), OUTLINE_C, OUTLINE_G, {})

    assert wide.splitlines()[0].endswith("--ntasks 8")
    assert narrow.splitlines()[0].endswith("--ntasks 2")
    assert "--ntasks" not in unset


def test_no_outline_means_no_command():
    """
    Emit nothing when the run was staged without outlines to reduce over.
    """
    assert _postprocess_commands(SPATIAL, Path("/out"), "none", "none", {}) == ""
    only_c = _postprocess_commands(SPATIAL, Path("/out"), OUTLINE_C, "none", {})
    assert len(only_c.splitlines()) == 1 and OUTLINE_C in only_c


def test_dh_command_extracts_the_hugonnet_interval():
    """
    One dh call over 2000-2020, all variables, into ``output/dh/``.

    Unlike the scalar reductions the extraction needs no outline, so the
    command carries only the interval, the spatial file, and the output path.
    """
    command = _dh_command(SPATIAL, Path("/out"), "RGI2000-v7.0-C-01-04374", "id_0")

    assert command.startswith("pism-glacier-postprocess-dh ")
    assert f"--start {DH_START} --end {DH_END}" in command
    assert "--vars" not in command
    assert str(SPATIAL.resolve()) in command
    assert command.endswith(f"/out/dh/dh_RGI2000-v7.0-C-01-04374_id_0_{DH_START}_{DH_END}.nc")


def test_dh_filenames_stay_apart_across_ensemble_members():
    """
    The sample/uq tag is part of the dh filename.

    All members of an ensemble share ``output/dh/``; without the tag the
    members would overwrite one another's file.
    """
    member_0 = _dh_command(SPATIAL, Path("/out"), "RGI2000-v7.0-C-01-04374", "id_0_uq_0")
    member_1 = _dh_command(SPATIAL, Path("/out"), "RGI2000-v7.0-C-01-04374", "id_0_uq_1")

    assert member_0.endswith(f"/out/dh/dh_RGI2000-v7.0-C-01-04374_id_0_uq_0_{DH_START}_{DH_END}.nc")
    assert member_0.rsplit(" ", 1)[-1] != member_1.rsplit(" ", 1)[-1]
