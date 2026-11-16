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
Tests for the regional grids of the ISMIP7 Greenland preparation.
"""

import logging

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from pism_terra.ismip7.greenland.prepare import (
    REGIONAL_GRIDS,
    prepare_regional_grids,
    read_axes,
    regional_bounds,
)

CRS = "EPSG:3413"
# The 150 m grid of GreenlandObsISMIP7-v1.3.nc (cell centres).
OBS_X = np.arange(-652925.0, 879625.0 + 1, 150.0)
OBS_Y = np.arange(-3384425.0, -632675.0 + 1, 150.0)
# Least common multiple of the default resolutions (m).
LCM = 216_000.0


def test_default_resolutions_have_lcm_216km():
    """
    The default resolutions are those of the ice-sheet-wide domain.
    """
    resolutions = REGIONAL_GRIDS["base_resolution"] * np.array(REGIONAL_GRIDS["multipliers"])
    assert np.lcm.reduce(resolutions) == LCM


def test_regional_bounds_is_centred_and_tiled():
    """
    The domain keeps the centre of the window and is a multiple of the LCM wide.
    """
    # 300 km x 100 km: two tiles in x, one in y.
    geometry = box(-200_000.0, -2_300_000.0, 100_000.0, -2_200_000.0)
    x_bnds, y_bnds = regional_bounds(geometry, OBS_X, OBS_Y)

    assert x_bnds[1] - x_bnds[0] == 2 * LCM
    assert y_bnds[1] - y_bnds[0] == LCM
    # The window is the buffered box snapped to the nearest cell centres.
    assert np.isclose(sum(x_bnds) / 2, -50_000.0, atol=150.0)
    assert np.isclose(sum(y_bnds) / 2, -2_250_000.0, atol=150.0)
    # The buffered outline lies inside.
    assert x_bnds[0] <= -203_000.0 and x_bnds[1] >= 103_000.0
    assert y_bnds[0] <= -2_303_000.0 and y_bnds[1] >= -2_197_000.0
    # Every resolution divides the domain.
    for multiplier in REGIONAL_GRIDS["multipliers"]:
        resolution = REGIONAL_GRIDS["base_resolution"] * multiplier
        assert (x_bnds[1] - x_bnds[0]) % resolution == 0
        assert (y_bnds[1] - y_bnds[0]) % resolution == 0


def test_regional_bounds_follows_resolutions():
    """
    Other resolutions give another tile size.
    """
    geometry = box(0.0, -2_000_000.0, 10_000.0, -1_990_000.0)
    x_bnds, y_bnds = regional_bounds(geometry, OBS_X, OBS_Y, buffer=1000.0, base_resolution=150, multipliers=[1, 2, 4])
    # The window is 81 cells (12.15 km) wide, grown to the next multiple of the 600 m LCM.
    assert x_bnds[1] - x_bnds[0] == 12_600.0
    assert y_bnds[1] - y_bnds[0] == 12_600.0


def test_regional_bounds_rejects_geometry_off_the_grid():
    """
    A geometry outside the reference grid has no window.
    """
    geometry = box(2_000_000.0, 0.0, 2_010_000.0, 10_000.0)
    with pytest.raises(ValueError, match="fewer than two cells"):
        regional_bounds(geometry, OBS_X, OBS_Y)


def test_prepare_regional_grids_writes_one_file_per_outline(tmp_path, caplog):
    """
    Write single-cell grid files named after outlines in any CRS.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    caplog : pytest.LogCaptureFixture
        Captures the warnings.
    """
    outlines = gpd.GeoDataFrame(
        {"NAME": ["GLACIER_A", "GLACIER B/2"]},
        geometry=[
            box(-200_000.0, -2_300_000.0, -150_000.0, -2_250_000.0),
            box(300_000.0, -1_500_000.0, 350_000.0, -1_450_000.0),
        ],
        crs=CRS,
    ).to_crs("EPSG:4326")

    with caplog.at_level(logging.WARNING):
        grid_files = prepare_regional_grids(
            outlines,
            OBS_X,
            OBS_Y,
            tmp_path / "grids",
            x_bnds=[-750650.0, 977350.0],
            y_bnds=[-3412550.0, -604550.0],
        )

    assert list(grid_files) == ["GLACIER_A", "GLACIER B/2"]
    assert grid_files["GLACIER_A"] == tmp_path / "grids" / "pism_GLACIER_A_grid.nc"
    assert grid_files["GLACIER B/2"] == tmp_path / "grids" / "pism_GLACIER_B_2_grid.nc"
    assert not caplog.records

    with xr.open_dataset(grid_files["GLACIER_A"], decode_coords="all") as ds:
        assert ds.attrs["domain"] == "GLACIER_A"
        assert ds.sizes["x"] == 1 and ds.sizes["y"] == 1
        x_bnds = ds["x_bnds"].values.ravel()
        y_bnds = ds["y_bnds"].values.ravel()
        assert ds.rio.crs.to_epsg() == 3413
    assert x_bnds[1] - x_bnds[0] == LCM
    assert y_bnds[1] - y_bnds[0] == LCM
    assert x_bnds[0] <= -203_000.0 and x_bnds[1] >= -147_000.0
    assert y_bnds[0] <= -2_303_000.0 and y_bnds[1] >= -2_247_000.0


def test_prepare_regional_grids_warns_beyond_the_ice_sheet_domain(tmp_path, caplog):
    """
    A domain the ice-sheet-wide inputs do not cover is reported.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    caplog : pytest.LogCaptureFixture
        Captures the warnings.
    """
    outlines = gpd.GeoDataFrame(
        {"NAME": ["EDGE"]}, geometry=[box(-650_000.0, -700_000.0, -640_000.0, -690_000.0)], crs=CRS
    )
    with caplog.at_level(logging.WARNING):
        prepare_regional_grids(
            outlines, OBS_X, OBS_Y, tmp_path, x_bnds=[-750650.0, 977350.0], y_bnds=[-3412550.0, -604550.0]
        )
    assert any("EDGE" in record.getMessage() and "beyond" in record.getMessage() for record in caplog.records)


def test_read_axes_sorts_both_axes(tmp_path):
    """
    The axes come back ascending, whatever their order in the file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    path = tmp_path / "obs.nc"
    xr.Dataset(
        coords={"x": np.array([0, 150, 300], dtype="int32"), "y": np.array([450, 300, 150], dtype="int32")}
    ).to_netcdf(path, engine="h5netcdf")
    x, y = read_axes(path)
    np.testing.assert_array_equal(x, [0.0, 150.0, 300.0])
    np.testing.assert_array_equal(y, [150.0, 300.0, 450.0])
