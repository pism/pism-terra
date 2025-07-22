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
Test staging functions.
"""

import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
import xarray as xr

from pism_terra.stage import stage_glacier


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


@pytest.mark.parametrize("rgi_id", ["RGI2000-v7.0-C-01-10853"])
def test_stage_glacier(rgi_id, rgi):
    """
    Test that `stage_glacier` creates a valid NetCDF file with expected glacier variables.

    This test verifies that the `stage_glacier` function successfully generates
    a NetCDF file containing surface elevation, ice thickness, and bedrock topography
    from a given RGI glacier ID. It checks for correct file creation, valid xarray content,
    and the presence of required variables and dimensions.

    Parameters
    ----------
    rgi_id : str
        The RGI ID of the glacier to stage. This value is parameterized by pytest.
    rgi : geopandas.GeoDataFrame
        The full or subsetted RGI dataset fixture used to extract glacier geometry.

    Returns
    -------
    None
        Asserts are used to validate correctness. The test will raise an AssertionError
        if the output is invalid or incomplete.

    Notes
    -----
    - The test uses a temporary directory to avoid writing files to disk permanently.
    - The output NetCDF file is inspected for expected variables: 'surface', 'thickness',
      'bed', and 'land_ice_area_fraction_retreat'.
    - This test assumes the glacier specified by `rgi_id` exists in the provided RGI input.
    """

    resolution = 100.0
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(stage_glacier(rgi_id, rgi=rgi, path=tmpdir, resolution=resolution))

        # Check that the file exists
        assert output_path.exists()
        assert output_path.suffix == ".nc"

        # Open the NetCDF and validate contents
        ds = xr.open_dataset(output_path)
        assert "surface" in ds
        assert "thickness" in ds
        assert "bed" in ds
        assert "land_ice_area_fraction_retreat" in ds

        # Optional: check dimensions and CRS metadata
        assert "x" in ds.dims
        assert "y" in ds.dims
        assert ds["surface"].attrs.get("standard_name") == "bedrock_altitude"
        # assert abs(ds.x[0] - ds.x[1]) == resolution

        ds.close()
