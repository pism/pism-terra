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
Tests for placing a glacier's staged observations beside the run's output.

Staging writes ``obs_{rgi_id}.nc`` to the input directory (possibly a shared
``data_path``) and caches the dh raster in the staging directory. These pin
that both are copied into ``<rgi_id>/output/observations``, the dh raster only
when the campaign asks for dh, and that a raster that was never fetched does
not stop the run from being generated.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from pism_terra.glacier import stage

RGI_ID = "RGI2000-v7.0-C-01-09429"


@pytest.fixture(name="staged")
def fixture_staged(tmp_path: Path) -> tuple[Path, Path, Path]:
    """
    Stage an obs file and a dh raster in a shared data path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.

    Returns
    -------
    tuple of pathlib.Path
        The input directory, the staging directory and the glacier's run
        directory, which lies outside the data path.
    """
    input_path = tmp_path / "data" / RGI_ID / "input"
    staging_path = tmp_path / "data" / RGI_ID / "staging"
    input_path.mkdir(parents=True)
    staging_path.mkdir(parents=True)
    (input_path / f"obs_{RGI_ID}.nc").write_bytes(b"obs")
    (staging_path / f"{RGI_ID}_dh.tif").write_bytes(b"dh")
    return input_path, staging_path, tmp_path / "experiment" / RGI_ID


@pytest.mark.parametrize("config", [{}, {"dh": "none"}, {"dh": None}])
def test_without_dh_only_the_obs_file_is_placed(staged, config):
    """
    Place the obs file alone when the campaign does not ask for dh.

    Parameters
    ----------
    staged : tuple of pathlib.Path
        Input, staging and run directories from the fixture.
    config : dict
        Campaign config without a dh dataset.
    """
    input_path, staging_path, path = staged

    placed = stage.place_observations(config, RGI_ID, input_path, staging_path, path)

    destination = path / "output" / "observations"
    assert placed == [destination / f"obs_{RGI_ID}.nc"]
    assert sorted(p.name for p in destination.iterdir()) == [f"obs_{RGI_ID}.nc"]


def test_with_dh_the_raster_is_placed_too(staged):
    """
    Place the obs file and the dh raster, as copies of the staged files.

    Parameters
    ----------
    staged : tuple of pathlib.Path
        Input, staging and run directories from the fixture.
    """
    input_path, staging_path, path = staged

    placed = stage.place_observations({"dh": "hugonnet"}, RGI_ID, input_path, staging_path, path)

    destination = path / "output" / "observations"
    assert placed == [destination / f"obs_{RGI_ID}.nc", destination / f"{RGI_ID}_dh.tif"]
    assert (destination / f"obs_{RGI_ID}.nc").read_bytes() == b"obs"
    assert (destination / f"{RGI_ID}_dh.tif").read_bytes() == b"dh"
    assert not (destination / f"obs_{RGI_ID}.nc").is_symlink()


def test_a_missing_dh_raster_is_skipped_with_a_warning(staged, caplog):
    """
    Place the obs file when the complex has no prepared dh raster.

    ``fetch_dh_raster`` leaves no file for complexes outside the observed
    tiles; generating the run must not fail over it.

    Parameters
    ----------
    staged : tuple of pathlib.Path
        Input, staging and run directories from the fixture.
    caplog : pytest.LogCaptureFixture
        Captures the warning.
    """
    input_path, staging_path, path = staged
    (staging_path / f"{RGI_ID}_dh.tif").unlink()

    with caplog.at_level(logging.WARNING):
        placed = stage.place_observations({"dh": "hugonnet"}, RGI_ID, input_path, staging_path, path)

    assert placed == [path / "output" / "observations" / f"obs_{RGI_ID}.nc"]
    assert f"{RGI_ID}_dh.tif" in caplog.text


def test_placing_again_refreshes_the_copy(staged):
    """
    Overwrite an earlier copy when the obs file is staged again.

    Parameters
    ----------
    staged : tuple of pathlib.Path
        Input, staging and run directories from the fixture.
    """
    input_path, staging_path, path = staged
    stage.place_observations({}, RGI_ID, input_path, staging_path, path)
    (input_path / f"obs_{RGI_ID}.nc").write_bytes(b"restaged")

    stage.place_observations({}, RGI_ID, input_path, staging_path, path)

    assert (path / "output" / "observations" / f"obs_{RGI_ID}.nc").read_bytes() == b"restaged"
