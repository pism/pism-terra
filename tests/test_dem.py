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
from numpy.testing import assert_array_almost_equal

from pism_terra.dem import get_glacier_by_rgi_id


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
