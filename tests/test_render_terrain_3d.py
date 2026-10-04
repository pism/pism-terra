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
import pandas as pd
import pytest
import shapely
import xarray as xr

# The renderer is a development tool: PyVista is in environment-dev.yml only,
# so the test environment may not have it.
pytest.importorskip("pyvista")

# pylint: disable=wrong-import-position
from pism_terra.glacier.render_terrain_3d import (  # noqa: E402
    GridSampler,
    current_outlines,
    drape_outlines,
    frame_time,
    load_outline_features,
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


def test_dated_fronts_load_as_lines_with_their_dates_and_glaciers(tmp_path):
    """
    Check that lines load as they are, each with its feature's date and glacier.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest per-test temporary directory.
    """
    x = 500_000.0 + 100.0 * np.arange(101)
    y = 7_000_000.0 + 100.0 * np.arange(81)
    fronts = gpd.GeoDataFrame(
        {
            "GlacierID": [1, 1, 2],
            "Date": pd.to_datetime(["1980-07-01", "1990-07-01", "1985-07-01"]),
        },
        geometry=[
            shapely.LineString([(x[10], y[10]), (x[10], y[50])]),
            shapely.LineString([(x[20], y[10]), (x[20], y[50])]),
            shapely.LineString([(x[60], y[10]), (x[90], y[10])]),
        ],
        crs=GRID_CRS,
    ).to_crs("EPSG:4326")
    path = tmp_path / "fronts.gpkg"
    fronts.to_file(path, driver="GPKG")

    features = load_outline_features(path, GRID_CRS, x, y, time="auto", group="auto")

    assert (features["time_column"], features["group_column"]) == ("Date", "GlacierID")
    assert len(features["lines"]) == 3  # the lines themselves, not their endpoints
    np.testing.assert_array_equal(features["groups"], [1, 1, 2])
    assert str(features["dates"][1])[:10] == "1990-07-01"
    # Static outlines read no dates.
    assert load_outline_features(path, GRID_CRS, x, y, time="none")["dates"] is None
    assert len(load_outlines(path, GRID_CRS, x, y)) == 3


def test_each_glacier_shows_its_latest_front_observed_by_then():
    """
    A glacier shows nothing before its first front and keeps its latest one after.
    """
    dates = np.array(["1980-07-01", "1990-07-01", "1990-07-01", "1985-07-01", "NaT"], dtype="datetime64[ns]")
    groups = np.array([1, 1, 1, 2, 2])  # glacier 1's 1990 front comes in two parts

    def at(day: str, **kwargs) -> list[bool]:
        """
        Select the fronts on screen on one day.

        Parameters
        ----------
        day : str
            ISO date.
        **kwargs
            Passed to :func:`current_outlines`.

        Returns
        -------
        list of bool
            The selection.
        """
        return current_outlines(dates, groups, np.datetime64(day, "ns"), **kwargs).tolist()

    assert at("1975-01-01") == [False] * 5
    assert at("1982-01-01") == [True, False, False, False, False]
    assert at("1987-01-01") == [True, False, False, True, False]
    assert at("2000-01-01") == [False, True, True, True, False]
    # A front older than the maximum age is no longer drawn.
    assert at("2000-01-01", max_age=np.timedelta64(3650, "D")) == [False, True, True, False, False]
    # Without groups the whole file is one sequence.
    assert current_outlines(dates, None, np.datetime64("1987-01-01", "ns")).tolist() == [
        False,
        False,
        False,
        True,
        False,
    ]


def test_frame_time_reads_datetimes_and_cftime():
    """
    Model times become datetime64 whether decoded by numpy or cftime.
    """
    cftime = pytest.importorskip("cftime")
    numpy_times = xr.Dataset(coords={"time": pd.to_datetime(["1985-06-16"])})
    cf_times = xr.Dataset(coords={"time": [cftime.DatetimeNoLeap(1985, 6, 16)]})
    plain = xr.Dataset(coords={"time": [3.5]})

    assert frame_time(numpy_times, 0) == np.datetime64("1985-06-16", "ns")
    assert frame_time(cf_times, 0) == np.datetime64("1985-06-16", "ns")
    assert frame_time(plain, 0) is None
    assert frame_time(xr.Dataset(), 0) is None
