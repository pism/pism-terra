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
Test DEM functions.
"""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import xarray as xr
from numpy.testing import assert_array_almost_equal
from rasterio.io import MemoryFile
from shapely.geometry import box

from pism_terra.dem import (
    boot_file_from_rgi_id,
    get_glacier_from_rgi_id,
)
from pism_terra.raster import raster_overlaps_glacier


@pytest.mark.integration
def test_boot_file_from_rgi_id(rgi, tmp_path):
    """
    Test the `glacier_dem_from_rgi_id` function for successful dataset creation.

    This test ensures that the glacier DEM and associated variables can be successfully
    generated from a given RGI ID and RGI dataset. It verifies that the returned object
    is a valid `xarray.Dataset` and implicitly tests core functionality like DEM stitching,
    reprojection, and ice thickness interpolation.

    Parameters
    ----------
    rgi : geopandas.GeoDataFrame
        Pre-loaded RGI dataset fixture or input, typically provided via a pytest fixture.
    tmp_path : pathlib.Path
        Temporary path fixture provided by PyTest.

    Notes
    -----
    - This test is intended to be run inside a test suite (e.g., with pytest).
    - The selected RGI ID ("RGI2000-v7.0-C-01-10853") should correspond to a valid glacier
      present in the provided RGI dataset.
    """
    path = tmp_path / "test_boot_file_from_rgi_id"
    path.mkdir(parents=True, exist_ok=True)

    rgi_id = "RGI2000-v7.0-C-01-10853"
    resolution = 100.0
    ds = boot_file_from_rgi_id(rgi_id, rgi, resolution=resolution, path=path)
    assert isinstance(ds, xr.Dataset)
    assert "surface" in ds
    assert "thickness" in ds
    assert "bed" in ds
    assert abs(ds.x[0] - ds.x[1]) == resolution


def test_get_glacier_from_rgi_id(rgi: gpd.GeoDataFrame):
    """
    Pytest for the `get_glacier_from_rgi_id` function.

    This test checks that the glacier returned by `get_glacier_from_rgi_id` for a known
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
    glacier = get_glacier_from_rgi_id(rgi, m_id).iloc[0]
    center_true = np.array([-129.73625986215418, 56.197765000000004])
    center = np.array([glacier["cenlon"], glacier["cenlat"]])
    assert_array_almost_equal(center, center_true)


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

        assert raster_overlaps_glacier(dataset, glacier)


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

    assert raster_overlaps_glacier(dataset, glacier)


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

        assert not raster_overlaps_glacier(dataset, glacier)


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

    assert not raster_overlaps_glacier(dataset, glacier)
