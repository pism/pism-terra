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
Tests for staging the Greenland paleo inputs.
"""

from __future__ import annotations

from pathlib import Path

from pism_terra.config import load_config
from pism_terra.greenland.paleo import stage as stage_module
from pism_terra.greenland.paleo.stage import PALEO_FILES, SHARED_FILES, stage

CONFIG_DIR = Path(__file__).resolve().parents[1] / "pism_terra" / "config"


def _offline(monkeypatch, downloads):
    """
    Replace the bucket and the file checks.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to replace the S3 and validation calls.
    downloads : list of str
        Receives the URI of every download.
    """
    # pylint: disable=import-outside-toplevel
    from pism_terra.ismip7.greenland import stage as ismip7_stage

    def download(uri, dest):
        """
        Record a download and leave an empty file.

        Parameters
        ----------
        uri : str
            Requested object.
        dest : pathlib.Path
            Local target.
        """
        downloads.append(uri)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).touch()

    monkeypatch.setattr(stage_module, "download_from_s3", download)
    monkeypatch.setattr(stage_module, "check_xr_fully", lambda path: True)
    monkeypatch.setattr(stage_module, "check_xr_lazy", lambda path, verbose=True: True)
    monkeypatch.setattr(ismip7_stage, "s3_key_exists", lambda bucket, key: True)


def test_shared_and_paleo_files_come_from_their_own_prefix(tmp_path, monkeypatch):
    """
    ISMIP7 files are fetched from the shared prefix, paleo files from the campaign's.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to replace the bucket and the file checks.
    """
    downloads: list[str] = []
    _offline(monkeypatch, downloads)
    config = load_config(CONFIG_DIR / "greenland_paleo.toml").campaign.as_params()

    df = stage(config, path=tmp_path)

    shared = "s3://pism-cloud-data/ismip7/greenland/input/v3"
    paleo = "s3://pism-cloud-data/paleo/greenland/input/v1"
    assert f"{shared}/grids/pism_bedmachine_greenland_grid.nc" in downloads
    assert f"{shared}/boot_1985_g450m_GreenlandObsISMIP7-v1.3.nc" in downloads
    assert f"{shared}/g900m_1985_id_BAYES-MEDIAN_HYBRID.nc" in downloads
    assert f"{paleo}/paleo_greenland_climate_OCX_YMM_1960_1989.nc" in downloads
    assert f"{paleo}/pism_dSL.nc" in downloads
    assert len(downloads) == 1 + len(SHARED_FILES) + len(PALEO_FILES)

    assert len(df) == 1
    row = df.iloc[0]
    for key in ("grid_file", *SHARED_FILES, *PALEO_FILES):
        assert Path(row[key]).is_absolute()
        assert Path(row[key]).is_relative_to(tmp_path.resolve() / "input")


def test_staged_files_are_not_downloaded_again(tmp_path, monkeypatch):
    """
    A second staging into the same data path fetches nothing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to replace the bucket and the file checks.
    """
    downloads: list[str] = []
    _offline(monkeypatch, downloads)
    config = load_config(CONFIG_DIR / "greenland_paleo.toml").campaign.as_params()
    stage(config, path=tmp_path / "a", data_path=tmp_path / "shared")
    downloads.clear()

    stage(config, path=tmp_path / "b", data_path=tmp_path / "shared")

    assert not downloads
