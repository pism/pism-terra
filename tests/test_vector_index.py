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
Tests for :func:`pism_terra.vector.index_geopackage`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import geopandas as gpd
import pyogrio
import pytest
from shapely.geometry import box

from pism_terra.vector import index_geopackage


def _indexes(path: Path, layer: str) -> dict[str, list[str]]:
    """
    Index name to indexed columns for one layer.

    Parameters
    ----------
    path : pathlib.Path
        The GeoPackage.
    layer : str
        Layer (table) name.

    Returns
    -------
    dict
        Index name to the columns it covers.
    """
    with sqlite3.connect(str(path)) as con:
        names = [row[1] for row in con.execute(f'PRAGMA index_list("{layer}")')]
        return {n: [c[2] for c in con.execute(f'PRAGMA index_info("{n}")')] for n in names}


@pytest.fixture(name="gpkg")
def fixture_gpkg(tmp_path: Path) -> Path:
    """
    A GeoPackage with an outline layer and a layer lacking ``rgi_id``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    pathlib.Path
        The file.
    """
    path = tmp_path / "rgi_c.gpkg"
    ids = [f"RGI2000-v7.0-C-01-{i:05d}" for i in range(50)]
    outlines = gpd.GeoDataFrame({"rgi_id": ids}, geometry=[box(i, 0, i + 0.5, 0.5) for i in range(50)], crs="EPSG:4326")
    outlines.to_file(path, layer="rgi_c", driver="GPKG")
    other = gpd.GeoDataFrame({"name": ["a"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    other.to_file(path, layer="basins", driver="GPKG", mode="a")
    return path


def test_indexes_rgi_id_in_layers_that_have_it(gpkg: Path):
    """
    Create one index per feature layer with the column, and skip the rest.

    Parameters
    ----------
    gpkg : pathlib.Path
        Fixture GeoPackage.
    """
    assert "rgi_id" not in {c for cols in _indexes(gpkg, "rgi_c").values() for c in cols}
    created = index_geopackage(gpkg)
    assert created == ["rgi_c_rgi_id_idx"]
    assert _indexes(gpkg, "rgi_c")["rgi_c_rgi_id_idx"] == ["rgi_id"]
    assert not any("rgi_id" in cols for cols in _indexes(gpkg, "basins").values())


def test_indexing_is_idempotent_and_keeps_the_file_readable(gpkg: Path):
    """
    Run twice without error, and leave GDAL able to read and filter the file.

    The whole point is that a pushed-down ``where`` on the column still works
    -- now via the index -- so that is what is checked, through pyogrio.

    Parameters
    ----------
    gpkg : pathlib.Path
        Fixture GeoPackage.
    """
    assert index_geopackage(gpkg) == index_geopackage(gpkg)
    layers = pyogrio.list_layers(gpkg)[:, 0].tolist()
    assert set(layers) == {"rgi_c", "basins"}
    hit = pyogrio.read_dataframe(gpkg, layer="rgi_c", where="rgi_id = 'RGI2000-v7.0-C-01-00007'", use_arrow=False)
    assert len(hit) == 1 and hit["rgi_id"].iloc[0] == "RGI2000-v7.0-C-01-00007"
    assert len(pyogrio.read_dataframe(gpkg, layer="rgi_c", use_arrow=False)) == 50
    # And the planner actually uses it.
    with sqlite3.connect(str(gpkg)) as con:
        plan = " ".join(row[3] for row in con.execute("EXPLAIN QUERY PLAN SELECT * FROM rgi_c WHERE rgi_id = 'x'"))
    assert "rgi_c_rgi_id_idx" in plan


def test_a_file_that_is_not_a_geopackage_is_skipped(tmp_path: Path):
    """
    Leave anything that is not a GeoPackage alone, without raising.

    ``prepare`` runs the indexing over whatever outline files it has; a
    placeholder or a shapefile must not take the whole preparation down.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    stray = tmp_path / "not_a_gpkg.gpkg"
    stray.write_text("placeholder")
    assert not index_geopackage(stray)
