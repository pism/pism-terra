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
Tests for picking the ITS_LIVE velocity tile.

The per-region COGs overlap generously in polar stereographic, so a domain
over Ellesmere Island sits inside both the Arctic Canada and the Greenland
footprints. The RGI ID settles it; geometry alone cannot.
"""

import pytest
from shapely.geometry import box

from pism_terra.glacier import observations
from pism_terra.glacier.observations import (
    region_code_from_bounds,
    region_code_from_rgi_id,
)


@pytest.fixture(name="fake_footprints")
def fixture_fake_footprints(monkeypatch):
    """
    Replace the S3 COG header probes with two overlapping footprints.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to swap the network-bound lookup.

    Returns
    -------
    dict
        The footprints keyed by region code, for assertions.
    """
    # Both cover the same square; only 01 reaches further north.
    footprints = {
        "03": box(0, 0, 100, 100),
        "05": box(0, 0, 100, 100),
        "01": box(0, 0, 100, 200),
    }
    monkeypatch.setattr(observations, "_ITS_LIVE_REGION_CODES", tuple(footprints))
    monkeypatch.setattr(
        observations,
        "_its_live_region_footprint",
        lambda code: ("EPSG:3413", footprints[code]),
    )
    return footprints


@pytest.mark.parametrize(
    "rgi_id,expected",
    [
        ("RGI2000-v7.0-C-03-01124", "03"),
        ("RGI2000-v7.0-G-01-05525", "01"),
        # High Mountain Asia is one mosaic; 13/15/16 fold into 14.
        ("RGI2000-v7.0-C-13-00001", "14"),
        ("RGI2000-v7.0-C-15-00001", "14"),
        # No region ITS_LIVE publishes, or no region at all.
        ("RGI2000-v7.0-C-99-00001", None),
        ("S4F_AK", None),
        (None, None),
    ],
)
def test_region_code_from_rgi_id(rgi_id, expected):
    """
    Read the o1 region out of an RGI identifier.

    Parameters
    ----------
    rgi_id : str or None
        Identifier to parse.
    expected : str or None
        Region code the identifier should map to.
    """
    assert region_code_from_rgi_id(rgi_id) == expected


def test_rgi_id_breaks_the_tie_between_overlapping_footprints(fake_footprints):
    """
    Prefer the region the glacier belongs to when footprints overlap.

    Parameters
    ----------
    fake_footprints : dict
        Stubbed COG footprints.
    """
    _ = fake_footprints
    bounds = (10, 10, 20, 20)  # inside 03, 05 and 01 alike

    assert region_code_from_bounds(bounds, "EPSG:3413", rgi_id="RGI2000-v7.0-C-05-00001") == "05"
    assert region_code_from_bounds(bounds, "EPSG:3413", rgi_id="RGI2000-v7.0-C-03-00001") == "03"


def test_a_domain_overhanging_a_tile_edge_still_resolves(fake_footprints, caplog):
    """
    Accept partial coverage, which a buffered domain routinely has.

    The glacier can sit well inside the tile while the staged domain — glacier
    plus buffer plus geographic pad — pokes over the edge. Requiring full
    containment rejected those runs outright.

    Parameters
    ----------
    fake_footprints : dict
        Stubbed COG footprints.
    caplog : pytest.LogCaptureFixture
        Captures the shortfall warning.
    """
    _ = fake_footprints
    # Half the box lies north of region 03's edge at y = 100.
    bounds = (10, 50, 20, 150)

    with caplog.at_level("WARNING"):
        assert region_code_from_bounds(bounds, "EPSG:3413", rgi_id="RGI2000-v7.0-C-03-00001") == "03"

    assert "50.0%" in caplog.text


def test_without_an_rgi_id_the_best_overlap_wins(fake_footprints, caplog):
    """
    Fall back to footprint geometry, preferring the most coverage.

    Parameters
    ----------
    fake_footprints : dict
        Stubbed COG footprints.
    caplog : pytest.LogCaptureFixture
        Captures the shortfall warning.
    """
    _ = fake_footprints
    bounds = (10, 50, 20, 150)

    with caplog.at_level("WARNING"):
        # Only 01 reaches y = 200, so it covers the box entirely.
        assert region_code_from_bounds(bounds, "EPSG:3413") == "01"

    assert "fully contains" not in caplog.text


def test_a_domain_outside_every_tile_is_rejected(fake_footprints):
    """
    Still fail when nothing overlaps, rather than picking an arbitrary tile.

    Parameters
    ----------
    fake_footprints : dict
        Stubbed COG footprints.
    """
    _ = fake_footprints

    with pytest.raises(ValueError, match="outside published coverage"):
        region_code_from_bounds((500, 500, 600, 600), "EPSG:3413")
