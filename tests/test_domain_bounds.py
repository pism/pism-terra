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
Tests for :func:`pism_terra.domain.get_bounds_from_geometry`.

The box sets the model domain, so it must not move: buffering only the parts
on the rim of an outline has to give exactly the box that buffering all of
it gives, however many parts lie inside.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import shapely

from pism_terra.domain import get_bounds_from_geometry


def _blob(rng: np.random.Generator, x: float, y: float) -> shapely.Polygon:
    """
    An irregular polygon, so the buffer's arcs fall at arbitrary angles.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of the vertex radii.
    x : float
        Centre easting, m.
    y : float
        Centre northing, m.

    Returns
    -------
    shapely.Polygon
        A star-shaped polygon a few km across.
    """
    angles = np.sort(rng.uniform(0, 2 * np.pi, 12))
    radii = rng.uniform(500, 3000, 12)
    return shapely.Polygon(np.column_stack([x + radii * np.cos(angles), y + radii * np.sin(angles)]))


def _outline(seed: int, n_parts: int) -> gpd.GeoSeries:
    """
    A one-row outline made of scattered parts, most of them away from the rim.

    Parameters
    ----------
    seed : int
        Seed of the layout.
    n_parts : int
        Number of polygons.

    Returns
    -------
    geopandas.GeoSeries
        The multi-part outline, in a projected CRS.
    """
    rng = np.random.default_rng(seed)
    parts = [_blob(rng, *rng.uniform(-200_000, 200_000, 2)) for _ in range(n_parts)]
    return gpd.GeoSeries([shapely.MultiPolygon(parts)], crs="EPSG:5936")


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("buffer_dist", [2_000.0, 5_000.0])
def test_the_box_is_that_of_the_whole_buffer(seed: int, buffer_dist: float):
    """
    Buffering the rim parts alone gives the box of the buffered outline.

    Parameters
    ----------
    seed : int
        Seed of the outline's layout.
    buffer_dist : float
        Buffer distance, m.
    """
    outline = _outline(seed, n_parts=60)
    min_x, min_y, max_x, max_y = outline.buffer(buffer_dist).total_bounds

    x_bnds, y_bnds = get_bounds_from_geometry(outline, buffer_dist=buffer_dist, dx=1.0)

    assert x_bnds == [np.ceil(min_x), np.floor(max_x)]
    assert y_bnds == [np.ceil(min_y), np.floor(max_y)]


def test_the_box_is_snapped_inward_to_the_grid_spacing():
    """
    The bounds are multiples of ``dx`` inside the buffered outline's box.
    """
    outline = gpd.GeoSeries([shapely.box(100.0, 200.0, 10_300.0, 20_400.0)], crs="EPSG:5936")

    x_bnds, y_bnds = get_bounds_from_geometry(outline, buffer_dist=2_000.0, dx=1_000.0)

    assert (x_bnds, y_bnds) == ([-1_000.0, 12_000.0], [-1_000.0, 22_000.0])


def test_inner_parts_are_not_buffered(monkeypatch):
    """
    Only the parts that can touch the box are buffered -- the point of the exercise.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture wrapping ``GeoSeries.buffer`` to count what it is given.
    """
    buffered: list[int] = []
    buffer = gpd.GeoSeries.buffer

    def counting_buffer(self, *args, **kwargs):
        """
        Buffer as usual, noting how many geometries were passed in.

        Parameters
        ----------
        self : geopandas.GeoSeries
            The geometries being buffered.
        *args : Any
            Passed on to ``GeoSeries.buffer``.
        **kwargs : Any
            Passed on to ``GeoSeries.buffer``.

        Returns
        -------
        geopandas.GeoSeries
            The buffered geometries.
        """
        buffered.append(len(self))
        return buffer(self, *args, **kwargs)

    monkeypatch.setattr(gpd.GeoSeries, "buffer", counting_buffer)

    get_bounds_from_geometry(_outline(0, n_parts=500), buffer_dist=2_000.0, dx=1_000.0)

    assert len(buffered) == 1 and buffered[0] <= 8
