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
Tests for what :func:`pism_terra.glacier.rgi.prepare_rgi` writes.

The outline GeoPackages are the largest thing a project keeps around, and
most of what RGI ships in them is never read: two dozen attributes, and a Z
coordinate that is zero everywhere. These pin that the written files carry
only what staging uses -- and still everything it uses.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import pytest
import shapely
from shapely.geometry import Polygon

from pism_terra.glacier import rgi as rgi_mod
from pism_terra.vector import glaciers_in_complex

#: A few of the attributes RGI ships that nothing downstream reads.
UNUSED = {"cenlon": 0.0, "cenlat": 60.0, "glims_id": "G1", "glac_name": "A Glacier", "zmed_m": 1500.0}


def _square(x: float) -> Polygon:
    """
    A unit square with RGI's all-zero Z coordinate.

    Parameters
    ----------
    x : float
        Western edge.

    Returns
    -------
    shapely.geometry.Polygon
        The 3-D square.
    """
    return Polygon([(x, 0, 0), (x + 1, 0, 0), (x + 1, 1, 0), (x, 1, 0)])


def _region(outline_type: str) -> gpd.GeoDataFrame:
    """
    Stand in for one downloaded RGI region.

    Parameters
    ----------
    outline_type : str
        ``"C"`` for one complex spanning two squares, ``"G"`` for the two glaciers in it.

    Returns
    -------
    geopandas.GeoDataFrame
        Outlines with RGI's attributes and the ``crs`` column ``prepare_rgi_region`` adds.
    """
    if outline_type == "C":
        ids, geometry = ["RGI2000-v7.0-C-01-00001"], [shapely.union_all([_square(0), _square(1)])]
    else:
        ids, geometry = ["RGI2000-v7.0-G-01-00001", "RGI2000-v7.0-G-01-00002"], [_square(0), _square(1)]
    frame = pd.DataFrame(
        {"rgi_id": ids, "o1region": "01", "o2region": "01-05", "utm_zone": 6, "area_km2": 2.5, "crs": "EPSG:32606"}
    ).assign(**UNUSED)
    return gpd.GeoDataFrame(frame, geometry=shapely.force_3d(geometry), crs="EPSG:4326")


@pytest.fixture(name="written")
def fixture_written(tmp_path: Path, monkeypatch) -> dict[str, gpd.GeoDataFrame]:
    """
    Run ``prepare_rgi`` on a stand-in region, with an aggregate, and read back what it wrote.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture replacing the regional download.

    Returns
    -------
    dict
        The complexes (``"c"``) and glaciers (``"g"``) as read from the GeoPackages.
    """
    monkeypatch.setattr(rgi_mod, "prepare_rgi_region", lambda region, outline_type, **kwargs: _region(outline_type))
    files = rgi_mod.prepare_rgi(
        pd.DataFrame({"region": ["01_alaska"]}),
        output_path=tmp_path,
        glacier_groups={"S4F_AK": pd.DataFrame({"rgi_id": ["RGI2000-v7.0-G-01-00002"]})},
        extract_path=tmp_path / "staging",
    )
    return {
        "c": pyogrio.read_dataframe(files["rgi_complexes"], use_arrow=False),
        "g": pyogrio.read_dataframe(files["rgi_glaciers"], use_arrow=False),
    }


def test_only_the_columns_that_are_used_are_written(written: dict[str, gpd.GeoDataFrame]):
    """
    The files carry the kept RGI attributes and the complex links, and nothing else.

    Parameters
    ----------
    written : dict
        The GeoPackages ``prepare_rgi`` wrote.
    """
    kept = ["rgi_id", "o1region", "o2region", "utm_zone", "area_km2", "crs"]
    assert list(written["c"].columns) == [*kept, "geometry"]
    assert list(written["g"].columns) == [*kept, "rgi_id_c", "rgi_id_c_aggregate", "geometry"]


def test_the_all_zero_z_coordinate_is_dropped(written: dict[str, gpd.GeoDataFrame]):
    """
    The outlines are written 2-D, with the same footprint.

    Parameters
    ----------
    written : dict
        The GeoPackages ``prepare_rgi`` wrote.
    """
    for outlines in written.values():
        assert not shapely.has_z(outlines.geometry.values).any()
        assert outlines.crs == "EPSG:4326"
    assert written["g"].geometry.iloc[0].equals(shapely.force_2d(_square(0)))


def test_glaciers_still_resolve_to_their_complex(written: dict[str, gpd.GeoDataFrame]):
    """
    Membership -- by parent complex and by aggregate -- survives the slimming.

    Parameters
    ----------
    written : dict
        The GeoPackages ``prepare_rgi`` wrote.
    """
    assert set(written["c"]["rgi_id"]) == {"RGI2000-v7.0-C-01-00001", "S4F_AK"}
    assert glaciers_in_complex("RGI2000-v7.0-C-01-00001", written["g"]) == [
        "RGI2000-v7.0-G-01-00001",
        "RGI2000-v7.0-G-01-00002",
    ]
    assert glaciers_in_complex("S4F_AK", written["g"]) == ["RGI2000-v7.0-G-01-00002"]
