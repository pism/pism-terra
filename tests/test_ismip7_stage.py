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
