# Copyright (C) 2025 Andy Aschwanden
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
Test DEM function.
"""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from numpy.testing import assert_array_almost_equal
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import box

from pism_terra.dem import (
    get_glacier_by_rgi_id,
    raster_overlaps_glacier,
)


@pytest.fixture(name="rgi")
def fixture_read_rgi() -> gpd.GeoDataFrame:
    """
    Fixture to read test RGI.

    Returns
    -------
    gpd.GeoDataFrame
        RGI test.
    """
    return gpd.read_file("tests/rgi_test.gpkg")


def test_get_glacier_by_rgi_id(rgi: gpd.GeoDataFrame):
    """
    Pytest for the `get_glacier_by_rgi_id` function.

    This test checks that the glacier returned by `get_glacier_by_rgi_id` for a known
    RGI ID has the expected central coordinates (longitude and latitude).

    Parameters
    ----------
    rgi : geopandas.GeoDataFrame
        The RGI dataset as a GeoDataFrame containing glacier geometries and attributes,
        including the 'rgi_id', 'cenlon', and 'cenlat' columns.

    Raises
    ------
    AssertionError
        If the coordinates of the returned glacier do not match the expected values.

    Notes
    -----
    This test uses a fixed glacier ID and expected coordinates, so it will only pass
    if the RGI dataset is version 7 and includes glacier ID "RGI2000-v7.0-C-01-16098".
    """

    m_id = "RGI2000-v7.0-C-01-16098"
    glacier = get_glacier_by_rgi_id(rgi, m_id)
    center_true = np.array([-129.73625986215418, 56.197765000000004])
    center = np.array([glacier["cenlon"], glacier["cenlat"]])
    assert_array_almost_equal(center, center_true)


@pytest.fixture(name="in_memory_raster")
def in_memory_raster():
    """
    Create an in-memory single-band raster dataset for testing.

    This fixture generates a small 10x10 raster using `rasterio.MemoryFile` with:
    - 1-meter resolution
    - Upper-left origin at (0, 10)
    - CRS: EPSG:32633 (UTM zone 33N)
    - Data filled with constant value 1

    The raster is written with a basic GeoTIFF profile suitable for unit tests that
    require spatially referenced raster data without writing to disk.

    Returns
    -------
    rasterio.io.MemoryFile
        A `MemoryFile` object containing the test raster. Use `.open()` to get a dataset.

    Examples
    --------
    def test_something(in_memory_raster):
        with in_memory_raster.open() as dataset:
            assert dataset.read(1).shape == (10, 10)
    """
    width = height = 10
    transform = from_origin(0, 10, 1, 1)  # (left, top, xres, yres)
    data = np.ones((1, height, width), dtype="uint8")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32633",
        "transform": transform,
    }

    memfile = MemoryFile()
    with memfile.open(**profile) as dataset:
        dataset.write(data)

    return memfile


@pytest.fixture(name="dataset")
def in_memory_raster_da():
    """
    Create and return an open in-memory single-band raster dataset for testing.

    This fixture generates a small 10x10 raster with:
    - Constant value 1
    - 1-meter resolution
    - Top-left origin at (0, 10)
    - CRS: EPSG:32633 (UTM zone 33N)

    The raster is created in memory using `rasterio.MemoryFile` and returned as
    an open dataset ready for testing.

    Yields
    ------
    rasterio.io.DatasetReader
        An open rasterio dataset backed by an in-memory file. Automatically closed
        after the test completes.

    Examples
    --------
    def test_read_data(in_memory_raster):
        assert in_memory_raster.read(1).shape == (10, 10)
        assert in_memory_raster.crs.to_epsg() == 32633
    """
    width = height = 10
    transform = from_origin(0, 10, 1, 1)
    data = np.ones((1, height, width), dtype="uint8")
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32633",
        "transform": transform,
    }

    memfile = MemoryFile()
    dataset = memfile.open(**profile)
    dataset.write(data)

    yield dataset

    dataset.close()
    memfile.close()


def test_raster_overlaps_true(in_memory_raster: MemoryFile):
    """
    Test that `raster_overlaps_glacier` correctly detects an overlapping glacier.

    This test creates a rectangular glacier polygon that lies fully within the
    bounds of the in-memory raster. It verifies that `raster_overlaps_glacier`
    returns `True` when the raster and glacier geometry intersect.

    Parameters
    ----------
    in_memory_raster : rasterio.io.MemoryFile
        A pytest fixture providing a 10x10 in-memory raster with CRS EPSG:32633
        and top-left corner at (0, 10), 1-meter resolution.

    Raises
    ------
    AssertionError
        If the function fails to detect overlap when expected.
    """
    with in_memory_raster.open() as dataset:
        # Create a polygon that overlaps the raster
        glacier_poly = box(2, 2, 5, 5)
        glacier = gpd.GeoSeries([glacier_poly], crs="EPSG:32633")

        assert raster_overlaps_glacier(dataset, glacier, crs="EPSG:32633")


def test_raster_overlaps_true_da(dataset: rasterio.DatasetBase):
    """
    Test that `raster_overlaps_glacier` correctly detects an overlapping glacier.

    This test creates a rectangular glacier polygon that lies fully within the
    bounds of the in-memory raster. It verifies that `raster_overlaps_glacier`
    returns `True` when the raster and glacier geometry intersect.

    Parameters
    ----------
    dataset : rasterio.DatasetBase
        A pytest fixture providing a 10x10 in-memory raster with CRS EPSG:32633
        and top-left corner at (0, 10), 1-meter resolution.

    Raises
    ------
    AssertionError
        If the function fails to detect overlap when expected.
    """

    # Create a polygon that overlaps the raster

    glacier_poly = box(2, 2, 5, 5)
    glacier = gpd.GeoSeries([glacier_poly], crs="EPSG:32633")

    assert raster_overlaps_glacier(dataset, glacier, crs="EPSG:32633")


def test_raster_overlaps_false(in_memory_raster: MemoryFile):
    """
    Test that `raster_overlaps_glacier` correctly detects an overlapping glacier.

    This test creates a rectangular glacier polygon that lies fully within the
    bounds of the in-memory raster. It verifies that `raster_overlaps_glacier`
    returns `False` when the raster and glacier geometry do not intersect.

    Parameters
    ----------
    in_memory_raster : rasterio.io.MemoryFile
        A pytest fixture providing a 10x10 in-memory raster with CRS EPSG:32633
        and top-left corner at (0, 10), 1-meter resolution.

    Raises
    ------
    AssertionError
        If the function fails to detect overlap when expected.
    """
    with in_memory_raster.open() as dataset:
        # Create a polygon far outside raster extent
        glacier_poly = box(1000, 1000, 1010, 1010)
        glacier = gpd.GeoSeries([glacier_poly], crs="EPSG:32633")

        assert not raster_overlaps_glacier(dataset, glacier, crs="EPSG:32633")


def test_raster_overlaps_false_da(dataset: rasterio.DatasetBase):
    """
    Test that `raster_overlaps_glacier` correctly detects an overlapping glacier.

    This test creates a rectangular glacier polygon that lies fully within the
    bounds of the in-memory raster. It verifies that `raster_overlaps_glacier`
    returns `False` when the raster and glacier geometry do not intersect.

    Parameters
    ----------
    dataset : rasterio.DatasetBase
        A pytest fixture providing a 10x10 in-memory raster with CRS EPSG:32633
        and top-left corner at (0, 10), 1-meter resolution.

    Raises
    ------
    AssertionError
        If the function fails to detect overlap when expected.
    """
    # Create a polygon far outside raster extent
    glacier_poly = box(1000, 1000, 1010, 1010)
    glacier = gpd.GeoSeries([glacier_poly], crs="EPSG:32633")

    assert not raster_overlaps_glacier(dataset, glacier, crs="EPSG:32633")
