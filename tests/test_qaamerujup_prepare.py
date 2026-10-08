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
Tests for the Qaamerujup domain preparation.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # pylint: disable=unused-import
import xarray as xr
from shapely.geometry import box

from pism_terra.ismip7.greenland.forcing import inverse_observations
from pism_terra.ismip7.qaamerujup.prepare import (
    EXCLUDED_BED,
    RHO_ICE,
    RHO_SEA_WATER,
    boot_geometry,
    domain_bounds,
    inside,
    merge_surface,
    product_names,
    read_outline,
)

CRS = "EPSG:3413"


def _grid(nx: int = 6, ny: int = 4, dx: float = 100.0, value: float = 0.0) -> xr.DataArray:
    """
    Build a small EPSG:3413 grid with descending y, like a GeoTIFF.

    Parameters
    ----------
    nx, ny : int
        Number of cells.
    dx : float
        Cell size (m).
    value : float
        Fill value.

    Returns
    -------
    xarray.DataArray
        The grid.
    """
    x = -220000.0 + dx * (np.arange(nx) + 0.5)
    y = -2045000.0 - dx * (np.arange(ny) + 0.5)
    return xr.DataArray(np.full((ny, nx), value), dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs(CRS)


def test_product_names_match_the_campaign_config() -> None:
    """
    The setup names its files as qaamerujup_century.toml expects them.
    """
    assert product_names({"name": "qaamerujup", "year": 1931}) == {
        "grid_file": "pism_qaamerujup_grid.nc",
        "boot_file": "boot_1931_qaamerujup.nc",
        "obs_file": "obs_1931_qaamerujup.nc",
    }


def test_read_outline_mends_a_geographic_label_on_projected_coordinates(tmp_path: Path) -> None:
    """
    An outline labelled EPSG:4326 whose coordinates are metres is read in the domain CRS.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory.
    """
    path = tmp_path / "domain.gpkg"
    gpd.GeoDataFrame(geometry=[box(-226000, -2050400, -215000, -2044000)], crs="EPSG:4326").to_file(path)
    outline = read_outline(path, CRS)
    assert outline.crs == CRS
    np.testing.assert_allclose(outline.total_bounds, [-226000, -2050400, -215000, -2044000])


def test_read_outline_reprojects_a_genuinely_geographic_outline(tmp_path: Path) -> None:
    """
    A correctly labelled geographic outline is reprojected, not relabelled.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory.
    """
    path = tmp_path / "lonlat.gpkg"
    gpd.GeoDataFrame(geometry=[box(-52.0, 71.0, -51.9, 71.1)], crs="EPSG:4326").to_file(path)
    xmin, _, _, ymax = read_outline(path, CRS).total_bounds
    assert xmin < -100000 and ymax < -1900000  # metres in polar stereographic, not degrees


@pytest.mark.parametrize("multipliers", [[1, 2, 3, 4], [1, 2, 4, 5, 10]])
def test_domain_bounds_cover_the_outline_and_tile_every_resolution(multipliers: list[int]) -> None:
    """
    The domain contains the outline and is a whole number of cells at every resolution.

    Parameters
    ----------
    multipliers : list of int
        Resolution multipliers of the base resolution.
    """
    outline = gpd.GeoDataFrame(geometry=[box(-226000, -2050400, -215000, -2044000)], crs=CRS)
    x_bnds, y_bnds = domain_bounds(outline, 32, multipliers)
    assert x_bnds[0] <= -226000 and x_bnds[1] >= -215000
    assert y_bnds[0] <= -2050400 and y_bnds[1] >= -2044000
    for dx in (32 * m for m in multipliers):
        for lo, hi in (x_bnds, y_bnds):
            assert (hi - lo) / dx == pytest.approx(round((hi - lo) / dx))


def test_boot_geometry_takes_thickness_from_the_reconstruction_only_inside_the_glacier() -> None:
    """
    Inside the glacier the thickness is surface minus bed; outside it follows the mask.
    """
    template = _grid(nx=5, ny=1)
    surface = template.copy(data=[[600.0, 600.0, 600.0, 30.0, 300.0]])
    bed = template.copy(data=[[100.0, 100.0, 550.0, -400.0, -100.0]])
    # glacier | grounded ice | land | floating ice | excluded
    mask = template.copy(data=[[1, 2, 1, 3, 0]])
    obs = xr.Dataset({"bed": bed, "mask": mask})
    glacier = template.copy(data=[[True, False, False, False, False]])
    buffer = template.copy(data=[[True, True, False, False, False]])
    excluded = template.copy(data=[[False, False, False, False, True]])

    boot = boot_geometry(surface, obs, glacier=glacier, buffer=buffer, excluded=excluded)

    alpha = 1.0 - RHO_ICE / RHO_SEA_WATER
    np.testing.assert_allclose(boot["thickness"].values[0], [500.0, 500.0, 0.0, 30.0 / alpha, 0.0], rtol=1e-6)
    assert boot["bed"].values[0, -1] == EXCLUDED_BED
    assert boot["surface"].values[0, -1] == 0.0
    np.testing.assert_array_equal(boot["ftt_mask"].values[0], [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(boot["land_ice_area_fraction_retreat"].values[0], [1, 1, 1, 1, 0])


def test_floating_thickness_is_capped_by_the_water_depth() -> None:
    """
    Freeboard over shallow water cannot make ice thicker than the water can float.
    """
    template = _grid(nx=1, ny=1)
    no = template.copy(data=[[False]])
    obs = xr.Dataset({"bed": template.copy(data=[[-10.0]]), "mask": template.copy(data=[[3]])})
    boot = boot_geometry(template.copy(data=[[50.0]]), obs, glacier=no, buffer=no, excluded=no)
    assert float(boot["thickness"].squeeze()) == pytest.approx(10.0 * RHO_SEA_WATER / RHO_ICE)


def test_merge_surface_uses_the_reconstruction_inside_and_fills_its_gaps() -> None:
    """
    The reconstruction replaces the reference inside the outline; uncovered cells are filled smoothly.
    """
    reference = _grid(nx=8, ny=8, value=100.0)
    reconstruction = _grid(nx=8, ny=8, value=300.0)
    reconstruction[:, 4] = np.nan  # a strip the reconstruction does not cover
    xmin, ymin, xmax, ymax = reference.rio.bounds()
    buffered = gpd.GeoDataFrame(geometry=[box(xmin + 200, ymin + 200, xmax - 200, ymax - 200)], crs=CRS)

    surface = merge_surface(reference, reconstruction, buffered)

    in_buffer = inside(reference, buffered)
    assert not bool(surface.isnull().any())
    assert float(surface.where(~in_buffer).max()) == 100.0
    filled = surface.isel(x=4).where(in_buffer.isel(x=4), drop=True)
    assert float(filled.min()) >= 100.0 and float(filled.max()) <= 300.0


def test_inverse_observations_free_tauc_only_on_observed_grounded_ice() -> None:
    """
    Grounded, observed ice is inverted for; floating ice and ice-free cells are fixed.
    """
    template = _grid(nx=3, ny=1)
    vx = template.copy(data=[[10.0, np.nan, 5.0]])
    vy = template.copy(data=[[0.0, np.nan, 5.0]])
    bed = template.copy(data=[[100.0, 100.0, -500.0]])
    thickness = template.copy(data=[[200.0, 0.0, 100.0]])
    ice = template.copy(data=[[True, False, True]])
    basins = xr.zeros_like(template, dtype="int8")

    vel = inverse_observations(vx, vy, bed, thickness, ice, basins)

    np.testing.assert_array_equal(vel["zeta_fixed_mask"].values[0], [0, 1, 1])
    np.testing.assert_array_equal(vel["vel_misfit_weight"].values[0], [1, 0, 0])
    np.testing.assert_allclose(vel["u_observed"].values[0], [10.0, 0.0, 5.0])
    assert "mapping" in vel
