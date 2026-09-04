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
Tests for :func:`pism_terra.ismip7.greenland.stage.resolve_forcing_name`.

Forcing filenames carry a per-GCM version (from the prepare setup TOML)
that is independent of the campaign ``version`` selecting the S3
subdirectory; staging must discover the actual name instead of assuming.
"""

from __future__ import annotations

from pism_terra.ismip7.greenland.stage import resolve_forcing_name


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
