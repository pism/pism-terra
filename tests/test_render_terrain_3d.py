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
Tests for the outlines of :mod:`pism_terra.glacier.render_terrain_3d`.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import shapely

from pism_terra.glacier.render_terrain_3d import (
    GridSampler,
    drape_outlines,
    load_outlines,
    outline_cells,
)

GRID_CRS = "EPSG:32606"


def write_outlines(path, polygons) -> None:
    """
    Write polygons given in the grid's CRS to a GeoPackage in EPSG:4326.

    Parameters
    ----------
    path : pathlib.Path
        File to write.
    polygons : list of shapely.Polygon
        Polygons in :data:`GRID_CRS`.
    """
    gpd.GeoDataFrame(geometry=polygons, crs=GRID_CRS).to_crs("EPSG:4326").to_file(path, driver="GPKG")


def test_load_outlines_reprojects_clips_and_keeps_holes(tmp_path):
    """
    Check that outlines come back in the grid's CRS, cut at its edge, holes included.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest per-test temporary directory.
    """
    x = 500_000.0 + 100.0 * np.arange(101)  # 10 km
    y = 7_000_000.0 + 100.0 * np.arange(81)[::-1]  # 8 km, north first
    inside = shapely.Polygon(
        shapely.box(x[10], y[-10], x[60], y[10]).exterior.coords,
        holes=[shapely.box(x[30], y[40], x[40], y[30]).exterior.coords],
    )
    straddling = shapely.box(x[80], y[60], x[-1] + 5000.0, y[20])
    far_away = shapely.box(x[-1] + 50_000.0, y[60], x[-1] + 60_000.0, y[20])
    path = tmp_path / "outlines.gpkg"
    write_outlines(path, [inside, straddling, far_away])

    lines = load_outlines(path, GRID_CRS, x, y)

    # The outer ring and the hole of the first, the part on the grid of the second.
    assert len(lines) == 3
    points = np.concatenate(lines)
    assert points[:, 0].min() >= x.min() - 1e-6 and points[:, 0].max() <= x.max() + 1e-6
    assert points[:, 1].min() >= y.min() - 1e-6 and points[:, 1].max() <= y.max() + 1e-6
    lengths = sorted(shapely.LineString(line).length for line in lines)
    np.testing.assert_allclose(lengths[0], inside.interiors[0].length, rtol=1e-3)
    np.testing.assert_allclose(lengths[-1], inside.exterior.length, rtol=1e-3)
    # Dense enough to follow the terrain: a vertex at least every grid spacing.
    for line in lines:
        assert np.hypot(*np.diff(line, axis=0).T).max() <= 100.0 + 1e-6


def test_load_outlines_without_any_on_the_grid(tmp_path):
    """
    A file with nothing on the grid gives no lines rather than an error.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest per-test temporary directory.
    """
    x = 500_000.0 + 100.0 * np.arange(11)
    y = 7_000_000.0 + 100.0 * np.arange(11)
    path = tmp_path / "outlines.gpkg"
    write_outlines(path, [shapely.box(600_000.0, 7_100_000.0, 601_000.0, 7_101_000.0)])

    assert not load_outlines(path, GRID_CRS, x, y)


def test_outlines_are_draped_on_the_surface():
    """
    The line mesh keeps every line apart and takes its heights from the surface.
    """
    x = np.arange(5.0)
    y = np.arange(4.0)
    height = np.add.outer(10.0 * y, x)  # z = 10 y + x
    lines = [np.array([[0.0, 0.0], [1.0, 1.0], [2.5, 1.0]]), np.array([[4.0, 3.0], [3.0, 2.0]])]

    xy, cells = outline_cells(lines)
    mesh = drape_outlines(xy, cells, height, GridSampler(x, y))

    assert mesh.n_lines == 2
    np.testing.assert_array_equal(cells, [3, 0, 1, 2, 2, 3, 4])
    np.testing.assert_allclose(mesh.points[:, 2], [0.0, 11.0, 12.5, 34.0, 23.0])
