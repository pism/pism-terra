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
Tests for the general point sampler in :mod:`pism_terra.grids`.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import Transformer
from shapely.geometry import Point

from pism_terra.grids import load_grid, sample_points

CRS = "EPSG:3413"
X = np.arange(-100_000.0, -100_000.0 + 10 * 1000.0, 1000.0)
Y = np.arange(-2_000_000.0, -2_000_000.0 + 8 * 1000.0, 1000.0)


def grid() -> xr.Dataset:
    """
    A small projected dataset shaped like a CARRA2 forcing store.

    ``field`` is ``i + 100 j`` so a nearest and a bilinear sample are easy to
    predict; ``layered`` carries an extra dimension; ``time_bounds`` has no
    horizontal dimensions; ``latitude`` is a 2-D grid coordinate.

    Returns
    -------
    xarray.Dataset
        The dataset, with ``spatial_ref`` carrying the CRS.
    """
    time = pd.date_range("2000-01-01", periods=3, freq="D")
    jj, ii = np.meshgrid(np.arange(Y.size), np.arange(X.size), indexing="ij")
    field = (ii + 100.0 * jj).astype(float)
    ds = xr.Dataset(
        {
            "field": (("time", "y", "x"), np.stack([field + t for t in range(time.size)]), {"units": "m"}),
            "layered": (("time", "level", "y", "x"), np.broadcast_to(field, (3, 2, Y.size, X.size)).copy()),
            "time_bounds": (("time", "nv"), np.stack([time, time + pd.Timedelta("1D")], axis=1)),
        },
        coords={"time": time, "level": [1000.0, 850.0], "x": X, "y": Y, "latitude": (("y", "x"), 70.0 + 0.01 * jj)},
    )
    ds = ds.rio.write_crs(CRS, grid_mapping_name="spatial_ref")
    ds["field"].attrs["grid_mapping"] = "spatial_ref"
    return ds


def sites(xy: list[tuple[float, float]], names: list[str] | None = None) -> gpd.GeoDataFrame:
    """
    Build points given in the grid's CRS, stored in EPSG:4326 like a site file.

    Parameters
    ----------
    xy : list of tuple of float
        Projected coordinates.
    names : list of str or None, optional
        Site labels for a ``site`` column; None leaves the frame unlabelled.

    Returns
    -------
    geopandas.GeoDataFrame
        The points.
    """
    to_lonlat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform([p[0] for p in xy], [p[1] for p in xy])
    frame = {"site": names} if names is not None else {}
    return gpd.GeoDataFrame(frame, geometry=[Point(a, b) for a, b in zip(lon, lat)], crs="EPSG:4326")


def test_load_grid_reads_a_description():
    """
    A shipped grid description is returned as text.
    """
    assert "gridtype" in load_grid("carra2")


def test_nearest_sample_carries_extra_dimensions_and_point_columns():
    """
    Every spatial variable is sampled at its cell, non-spatial ones pass through, and columns become coordinates.
    """
    ds = grid()
    points = sites([(X[2] + 100.0, Y[3] - 100.0), (X[7], Y[0])], ["A", "B"])
    points["elevation"] = [1500.0, 1200.0]
    out = sample_points(ds, points, variables=ds.data_vars)
    assert set(out.data_vars) == {"field", "layered", "time_bounds"}
    assert out["field"].dims == ("time", "site")
    assert out["layered"].dims == ("time", "level", "site")
    assert out["time_bounds"].dims == ("time", "nv")
    assert out["site"].values.tolist() == ["A", "B"]
    np.testing.assert_allclose(out["field"].sel(site="A").values, 302.0 + np.arange(3))
    np.testing.assert_allclose(out["field"].sel(site="B").values, 7.0 + np.arange(3))
    np.testing.assert_allclose(out["elevation"].values, [1500.0, 1200.0])
    np.testing.assert_allclose(out["x"].values, [X[2] + 100.0, X[7]])
    np.testing.assert_allclose(out["lon"].values, points.geometry.x.to_numpy())
    assert "latitude" not in out.coords
    assert out["field"].attrs["units"] == "m"
    assert out.attrs["sampling"] == "nearest"
    assert "3413" in out.attrs["crs"] or "Polar Stereographic" in out.attrs["crs"]


def test_default_variables_are_the_spatial_ones():
    """
    Without a list, the variables spanning both horizontal dimensions are sampled and the grid mapping is left out.
    """
    out = sample_points(grid(), sites([(X[1], Y[1])]))
    assert set(out.data_vars) == {"field", "layered"}
    assert out["site"].values.tolist() == [0]


def test_linear_interpolates_and_falls_back_to_nearest_at_gaps(caplog):
    """
    A site halfway between cells gets the mean of its neighbours; a NaN in the stencil yields the nearest cell.

    Parameters
    ----------
    caplog : pytest.LogCaptureFixture
        Captured log records.
    """
    ds = grid()
    ds["field"][:, 5, 5] = np.nan
    points = sites([(X[2] + 500.0, Y[3]), (X[5] + 400.0, Y[5] + 400.0)], ["mid", "gap"])
    out = sample_points(ds, points, variables=["field"], method="linear")
    np.testing.assert_allclose(out["field"].sel(site="mid").values, 302.5 + np.arange(3))
    # the nearest cell is the NaN itself, so there is nothing to fall back to
    assert out["field"].sel(site="gap").isnull().all()
    assert out.attrs["sampling"] == "linear"
    assert out["field"].attrs["units"] == "m"
    ds["field"][:, 5, 6] = np.nan
    gap = sites([(X[5] + 400.0, Y[4] + 400.0)], ["gap"])
    with caplog.at_level("INFO"):
        out = sample_points(ds, gap, variables=["field"], method="linear")
    np.testing.assert_allclose(out["field"].values.ravel(), 405.0 + np.arange(3))
    assert "using nearest" in caplog.text


def test_outside_points_are_dropped_and_none_inside_raises(caplog):
    """
    A site beyond the half-cell margin is dropped with a warning; with none left the call raises.

    Parameters
    ----------
    caplog : pytest.LogCaptureFixture
        Captured log records.
    """
    ds = grid()
    points = sites([(X[0], Y[0]), (X[-1] + 5000.0, Y[0])], ["in", "out"])
    out = sample_points(ds, points, variables=["field"])
    assert out["site"].values.tolist() == ["in"]
    assert "out" in caplog.text
    with pytest.raises(ValueError, match="inside"):
        sample_points(ds, sites([(X[-1] + 5000.0, Y[0])]), variables=["field"])


def test_descending_axis_missing_variable_and_crs_override(caplog):
    """
    A flipped axis is handled, a missing variable is skipped with a warning, and ``crs`` replaces the grid mapping.

    Parameters
    ----------
    caplog : pytest.LogCaptureFixture
        Captured log records.
    """
    ds = grid().sortby("y", ascending=False)
    out = sample_points(ds, sites([(X[2], Y[3])]), variables=["field", "nope"])
    np.testing.assert_allclose(out["field"].values.ravel(), 302.0 + np.arange(3))
    assert "nope" in caplog.text
    bare = ds.drop_vars("spatial_ref")
    with pytest.raises(ValueError):
        sample_points(bare, sites([(X[2], Y[3])]))
    out = sample_points(bare, sites([(X[2], Y[3])]), crs=CRS)
    assert out.attrs["crs"] == CRS


def test_bad_method_and_label():
    """
    An unknown method or a label that is not a column is refused up front.
    """
    with pytest.raises(ValueError, match="method"):
        sample_points(grid(), sites([(X[0], Y[0])]), method="cubic")
    with pytest.raises(ValueError, match="column"):
        sample_points(grid(), sites([(X[0], Y[0])]), label="name")
    points = sites([(X[0], Y[0])], ["A"]).rename(columns={"site": "name"})
    out = sample_points(grid(), points, label="name", dim="point")
    assert out["point"].values.tolist() == ["A"]
