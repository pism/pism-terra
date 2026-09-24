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
Tests for the ISMIP7 Greenland forcing-filename resolution.

Forcing filenames carry a per-GCM version (from the prepare setup TOML)
that is independent of the campaign ``version`` selecting the S3
subdirectory. A campaign config pins those versions per GCM in
``[campaign.forcing_versions]`` and staging uses them verbatim; only a GCM
with nothing pinned is discovered from the files present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pism_terra.config import CampaignConfig, load_config, version_tag
from pism_terra.ismip7.greenland.stage import (
    explicit_forcing_version,
    resolve_forcing_name,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "pism_terra" / "config"


def test_resolves_the_per_gcm_version():
    """
    An MRI file tagged v2 is found even under a v3 campaign.
    """
    candidates = {
        "ismip7_greenland_climate_historical_MRI-ESM2-0_v2_1900_2014.nc",
        "ismip7_greenland_climate_historical_CESM2-WACCM_v3_1900_2014.nc",
    }

    name = resolve_forcing_name(candidates, "climate", "historical", "MRI-ESM2-0", 1900, 2014, "v3")

    assert name == "ismip7_greenland_climate_historical_MRI-ESM2-0_v2_1900_2014.nc"


def test_newest_version_wins():
    """
    When several versions of one file exist, the highest number is picked.
    """
    candidates = {
        "ismip7_greenland_ocean_ssp126_MRI-ESM2-0_v2_2015_2300.nc",
        "ismip7_greenland_ocean_ssp126_MRI-ESM2-0_v4_2015_2300.nc",
        "ismip7_greenland_ocean_ssp126_MRI-ESM2-0_v3_2015_2300.nc",
    }

    name = resolve_forcing_name(candidates, "ocean", "ssp126", "MRI-ESM2-0", 2015, 2300, "v1")

    assert name == "ismip7_greenland_ocean_ssp126_MRI-ESM2-0_v4_2015_2300.nc"


def test_falls_back_to_the_campaign_version():
    """
    With no match, the conventional campaign-version name is returned.

    The later download then fails with the expected name in the message
    instead of a silent skip.
    """
    name = resolve_forcing_name(set(), "climate", "ssp585", "MRI-ESM2-0", 2015, 2300, "v3")

    assert name == "ismip7_greenland_climate_ssp585_MRI-ESM2-0_v3_2015_2300.nc"


def test_climate_does_not_match_climate_gradient():
    """
    The ``climate`` pattern must not swallow ``climate_gradient`` files.
    """
    candidates = {"ismip7_greenland_climate_gradient_historical_MRI-ESM2-0_v2_1900_2014.nc"}

    name = resolve_forcing_name(candidates, "climate", "historical", "MRI-ESM2-0", 1900, 2014, "v3")

    assert name == "ismip7_greenland_climate_historical_MRI-ESM2-0_v3_1900_2014.nc"
    grad = resolve_forcing_name(candidates, "climate_gradient", "historical", "MRI-ESM2-0", 1900, 2014, "v3")
    assert grad == "ismip7_greenland_climate_gradient_historical_MRI-ESM2-0_v2_1900_2014.nc"


def test_year_range_is_exact():
    """
    A file for a different epoch span is not accepted.
    """
    candidates = {"ismip7_greenland_climate_historical_MRI-ESM2-0_v2_1978_2014.nc"}

    name = resolve_forcing_name(candidates, "climate", "historical", "MRI-ESM2-0", 1900, 2014, "v3")

    assert name == "ismip7_greenland_climate_historical_MRI-ESM2-0_v3_1900_2014.nc"


def test_pinned_version_is_used_verbatim():
    """
    A pinned version wins over newer files that are still lying around.

    CESM2-WACCM's ocean product was re-published as v2 while the superseded
    v3 file stayed on S3; discovery would pick v3, the pin must not.
    """
    candidates = {
        "ismip7_greenland_ocean_historical_CESM2-WACCM_v2_1900_2014.nc",
        "ismip7_greenland_ocean_historical_CESM2-WACCM_v3_1900_2014.nc",
    }

    name = resolve_forcing_name(candidates, "ocean", "historical", "CESM2-WACCM", 1900, 2014, "v3", version="v2")

    assert name == "ismip7_greenland_ocean_historical_CESM2-WACCM_v2_1900_2014.nc"


def test_pinned_version_needs_no_candidates():
    """
    With a pin the listing is irrelevant, even when it is empty.
    """
    name = resolve_forcing_name(set(), "climate_gradient", "ssp585", "MRI-ESM2-0", 2015, 2300, "v3", version="v2")

    assert name == "ismip7_greenland_climate_gradient_ssp585_MRI-ESM2-0_v2_2015_2300.nc"


def test_explicit_forcing_version_precedence():
    """
    The per-GCM table beats the campaign-wide keys; the gradient shares ``climate``.
    """
    config = {
        "version": "v3",
        "climate_version": "v9",
        "ocean_version": "v9",
        "forcing_versions": {"CESM2-WACCM": {"climate": "v3", "ocean": "v2"}},
    }

    assert explicit_forcing_version(config, "CESM2-WACCM", "climate") == "v3"
    assert explicit_forcing_version(config, "CESM2-WACCM", "climate_gradient") == "v3"
    assert explicit_forcing_version(config, "CESM2-WACCM", "ocean") == "v2"
    # A GCM without a table entry falls back to the campaign-wide keys.
    assert explicit_forcing_version(config, "MRI-ESM2-0", "ocean") == "v9"
    # Nothing pinned at all: the caller has to discover the file.
    assert explicit_forcing_version({"version": "v3"}, "MRI-ESM2-0", "ocean") is None


def test_version_tag_normalisation():
    """
    Integers, digit strings and ``v<n>`` tags all render as ``v<n>``.
    """
    assert version_tag(2) == "v2"
    assert version_tag("2") == "v2"
    assert version_tag(" V2 ") == "v2"
    with pytest.raises(ValueError):
        version_tag("latest")
    with pytest.raises(ValueError):
        version_tag(True)


def test_campaign_config_pins_forcing_versions():
    """
    A shipped non-counter config carries the table and it is normalised.
    """
    cfg = load_config(CONFIG_DIR / "ismip7_greenland_2007_historical_free.toml")

    assert cfg.campaign.forcing_versions == {"CESM2-WACCM": {"climate": "v3", "ocean": "v2"}}
    params = cfg.campaign.as_params()
    assert explicit_forcing_version(params, "CESM2-WACCM", "ocean") == "v2"
    assert explicit_forcing_version(params, "CESM2-WACCM", "climate") == "v3"


def test_campaign_config_rejects_unknown_forcing():
    """
    A typo in the forcing product name is caught at load time.
    """
    with pytest.raises(ValueError, match="unknown forcing"):
        CampaignConfig(forcing_versions={"CESM2-WACCM": {"climat": 3}})


def test_counter_config_pins_versions_from_the_spec():
    """
    Counter configs need no table: the counter fills the campaign-wide keys.
    """
    cfg = load_config(CONFIG_DIR / "ismip7_greenland_c001.toml")
    params = cfg.campaign.as_params()

    assert cfg.campaign.forcing_versions is None
    assert explicit_forcing_version(params, "CESM2-WACCM", "climate") == "v3"
    assert explicit_forcing_version(params, "CESM2-WACCM", "ocean") == "v2"


def test_dh_observations_are_staged_when_the_config_names_them(tmp_path, monkeypatch):
    """
    The observed thickness-change files come down with the run inputs.

    PISM never reads them -- they are what the run is compared against
    afterwards -- but they have to be on the machine that does the comparing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to record the S3 keys instead of fetching them.
    """
    # pylint: disable=import-outside-toplevel
    from pism_terra.ismip7.greenland import stage as stage_module

    requested: list[str] = []

    def _record(uri, dest, **_kwargs):
        """
        Note the key instead of fetching it.

        Parameters
        ----------
        uri : str
            S3 URI requested.
        dest : str or pathlib.Path
            Where it would have been written.
        **_kwargs : dict
            Ignored.

        Returns
        -------
        pathlib.Path
            ``dest``.
        """
        requested.append(uri)
        return Path(dest)

    monkeypatch.setattr(stage_module, "download_from_s3", _record)

    cfg = load_config(CONFIG_DIR / "ismip7_greenland_c011.toml")
    assert cfg.campaign.dh_files, "the config should name the dh files"
    try:
        stage_module.stage(
            cfg.campaign.as_params(),
            path=tmp_path,
            force_overwrite=False,
            data_path=None,
            include_projection=False,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        # The staged files are stubs, so validation downstream fails; the
        # key list is already complete by then.
        pass

    staged = [uri.rsplit("/", 1)[-1] for uri in requested]
    for name in cfg.campaign.dh_files:
        assert name in staged, f"{name} was not staged"


def test_a_config_without_dh_files_stages_as_before(tmp_path, monkeypatch):
    """
    The field is optional, so omitting it changes nothing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to record the S3 keys instead of fetching them.
    """
    # pylint: disable=import-outside-toplevel
    from pism_terra.ismip7.greenland import stage as stage_module

    requested: list[str] = []

    def _record(uri, dest, **_kwargs):
        """
        Note the key instead of fetching it.

        Parameters
        ----------
        uri : str
            S3 URI requested.
        dest : str or pathlib.Path
            Where it would have been written.
        **_kwargs : dict
            Ignored.

        Returns
        -------
        pathlib.Path
            ``dest``.
        """
        requested.append(uri)
        return Path(dest)

    monkeypatch.setattr(stage_module, "download_from_s3", _record)

    cfg = load_config(CONFIG_DIR / "ismip7_greenland_c011.toml")
    config = cfg.campaign.as_params()
    config.pop("dh_files", None)
    try:
        stage_module.stage(config, path=tmp_path, force_overwrite=False, data_path=None, include_projection=False)
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    assert not [uri for uri in requested if "/dh_" in uri]


def test_dataset_version_cli_overrides_the_campaign_directory(monkeypatch, tmp_path):
    """
    Let ``--dataset-version`` pick the staged-input directory the config pins.

    ``campaign.version`` names the S3 subdirectory the inputs are fetched
    from (``<prefix>/<version>/``), so overriding it is how one config is
    pointed at a re-published set of inputs without editing the TOML.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to drive the CLI and capture what staging is handed.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    import pism_terra.ismip7.greenland.stage as stage_mod  # pylint: disable=import-outside-toplevel

    seen: dict = {}

    def fake_stage(config, **kwargs):  # pylint: disable=unused-argument
        """
        Record the campaign dict staging was handed, without touching S3.

        Parameters
        ----------
        config : dict
            Campaign parameters.
        **kwargs : dict
            Ignored.

        Returns
        -------
        pandas.DataFrame
            An empty manifest.
        """
        seen.update(config)
        import pandas as pd  # pylint: disable=import-outside-toplevel

        # main() writes its manifest beside the staged inputs; real staging
        # creates that directory on the way.
        (tmp_path / "input").mkdir(parents=True, exist_ok=True)
        return pd.DataFrame({"file": []})

    monkeypatch.setattr(stage_mod, "stage", fake_stage)
    monkeypatch.setattr(stage_mod, "prepare_observations", lambda *a, **k: {})
    config_file = str(Path("pism_terra/config/ismip7_greenland_c011.toml").resolve())

    monkeypatch.setattr(
        "sys.argv",
        ["pism-ismip7-greenland-stage", "--output-path", str(tmp_path), "--no-observations", config_file],
    )
    stage_mod.main()
    pinned = seen["version"]

    seen.clear()
    monkeypatch.setattr(
        "sys.argv",
        [
            "pism-ismip7-greenland-stage",
            "--output-path",
            str(tmp_path),
            "--no-observations",
            "--dataset-version",
            "v9",
            config_file,
        ],
    )
    stage_mod.main()

    assert pinned != "v9", "the config must not already pin v9, or this proves nothing"
    assert seen["version"] == "v9"
    # The override touches the directory and nothing else.
    assert seen["prefix"] == "ismip7/greenland/input"
    assert seen["boot_file"] == "boot_1985_g450m_GreenlandObsISMIP7-v1.3.nc"


def test_dataset_version_cli_defaults_to_the_config(monkeypatch, tmp_path):
    """
    Leave the config's version alone when the flag is not given.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to drive the CLI and capture what staging is handed.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    import pism_terra.ismip7.greenland.stage as stage_mod  # pylint: disable=import-outside-toplevel

    seen: dict = {}

    def fake_stage(config, **kwargs):  # pylint: disable=unused-argument
        """
        Record the campaign dict staging was handed, without touching S3.

        Parameters
        ----------
        config : dict
            Campaign parameters.
        **kwargs : dict
            Ignored.

        Returns
        -------
        pandas.DataFrame
            An empty manifest.
        """
        seen.update(config)
        import pandas as pd  # pylint: disable=import-outside-toplevel

        # main() writes its manifest beside the staged inputs; real staging
        # creates that directory on the way.
        (tmp_path / "input").mkdir(parents=True, exist_ok=True)
        return pd.DataFrame({"file": []})

    monkeypatch.setattr(stage_mod, "stage", fake_stage)
    config_file = str(Path("pism_terra/config/ismip7_greenland_c011.toml").resolve())
    monkeypatch.setattr(
        "sys.argv",
        ["pism-ismip7-greenland-stage", "--output-path", str(tmp_path), "--no-observations", config_file],
    )
    stage_mod.main()

    expected = load_config(config_file).campaign.version
    assert seen["version"] == expected


def test_dh_observations_are_placed_beside_the_output(tmp_path):
    """
    The staged dh files are copied to output/observations, where the comparison tools look.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    # pylint: disable=import-outside-toplevel
    from pism_terra.ismip7.greenland.stage import place_dh_observations

    input_dir = tmp_path / "shared_input"
    input_dir.mkdir()
    (input_dir / "dh_smith_g5000m_ICESat1-ICESat2-2021.nc").write_bytes(b"dh")
    run_dir = tmp_path / "run"
    placed = place_dh_observations({"dh_files": ["dh_smith_g5000m_ICESat1-ICESat2-2021.nc"]}, input_dir, run_dir)
    assert placed == [run_dir / "output" / "observations" / "dh_smith_g5000m_ICESat1-ICESat2-2021.nc"]
    assert placed[0].read_bytes() == b"dh"
    assert not place_dh_observations({}, input_dir, run_dir)
