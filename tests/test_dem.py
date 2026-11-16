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
import xarray as xr
from numpy.testing import assert_array_almost_equal
from rasterio.io import DatasetReader, MemoryFile
from shapely.geometry import box

from pism_terra.glacier.dem import boot_file_from_grid
from pism_terra.raster import raster_overlaps_glacier
from pism_terra.vector import get_glacier_from_rgi_id


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


def test_raster_overlaps_true_da(dataset: DatasetReader):
    """
    Test that `raster_overlaps_glacier` correctly detects an overlapping glacier.

    This test creates a rectangular glacier polygon that lies fully within the
    bounds of the in-memory raster. It verifies that `raster_overlaps_glacier`
    returns `True` when the raster and glacier geometry intersect.

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
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


def test_raster_overlaps_false_da(dataset: DatasetReader):
    """
    Test that `raster_overlaps_glacier` correctly detects an overlapping glacier.

    This test creates a rectangular glacier polygon that lies fully within the
    bounds of the in-memory raster. It verifies that `raster_overlaps_glacier`
    returns `False` when the raster and glacier geometry do not intersect.

    Parameters
    ----------
    dataset : rasterio.io.DatasetReader
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


def test_boot_file_from_grid_rejects_empty_variables():
    """An empty ``variables`` selection is rejected before any DEM work is done."""
    with pytest.raises(ValueError, match="at least one data variable"):
        boot_file_from_grid(
            xr.Dataset(),
            "RGI2000-v7.0-C-01-00000",
            [],
            dem_dataset="glo_30",
            ice_thickness_dataset="maffezzoli",
            bathymetry_dataset="none",
            velocity_dataset="none",
            forcing_mask="none",
            variables=[],
        )
