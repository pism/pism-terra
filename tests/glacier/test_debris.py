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

# pylint: disable=too-many-positional-arguments

"""
Tests for the NSIDC HMA_DTE debris-thickness staging.

Everything runs offline: the Earthdata download is either bypassed
(link-selection and mosaic tests use synthetic GeoTIFFs) or monkeypatched
(the end-to-end test).
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from pyproj import Transformer
from shapely.geometry import box

from pism_terra.domain import create_domain
from pism_terra.glacier import debris
from pism_terra.workflow import check_xr_lazy

CRS = "EPSG:32606"


def _write_tif(path: Path, value: float, x0: float, x1: float, y0: float, y1: float, crs: str = CRS) -> Path:
    """
    Write a constant-valued GeoTIFF covering a rectangle.

    Parameters
    ----------
    path : pathlib.Path
        Output file.
    value : float
        Constant cell value.
    x0 : float
        Western edge in *crs* coordinates.
    x1 : float
        Eastern edge in *crs* coordinates.
    y0 : float
        Southern edge in *crs* coordinates.
    y1 : float
        Northern edge in *crs* coordinates.
    crs : str, optional
        CRS the raster is written in.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    x = np.arange(x0 + 5.0, x1, 10.0)
    y = np.arange(y1 - 5.0, y0, -10.0)
    da = xr.DataArray(
        np.full((y.size, x.size), value, dtype=np.float32),
        coords={"y": y, "x": x},
        dims=("y", "x"),
    ).rio.write_crs(crs)
    da.rio.to_raster(path)
    return path


@pytest.fixture(name="target_grid")
def fixture_target_grid() -> xr.Dataset:
    """
    Build a small 50 m target grid in UTM zone 6N.

    Returns
    -------
    xarray.Dataset
        A 20x20 grid spanning 1 km x 1 km.
    """
    return create_domain([500_000.0, 501_000.0], [6_700_000.0, 6_701_000.0], resolution=50.0, crs=CRS)


def test_select_debris_links_prefers_direct_estimates():
    """
    Ignore sidecars and prefer a direct estimate over its extrap fallback.
    """
    links = [
        "https://data.nsidc.org/HMA_DTE_1.00001_manifest.txt",
        "https://data.nsidc.org/HMA_DTE_1.00001_bins.csv",
        "https://data.nsidc.org/HMA_DTE_1.00001_hdts_m.tif.aux.xml",
        "https://data.nsidc.org/HMA_DTE_1.00001_hdts_m_extrap.tif",
        "https://data.nsidc.org/HMA_DTE_1.00001_hdts_m.tif",
        "https://data.nsidc.org/HMA_DTE_1.00001_meltfactor_extrap.tif",
    ]
    chosen = debris.select_debris_links(links)
    assert chosen == {
        "debris_thickness": "https://data.nsidc.org/HMA_DTE_1.00001_hdts_m.tif",
        "debris_melt_factor": "https://data.nsidc.org/HMA_DTE_1.00001_meltfactor_extrap.tif",
    }


def test_select_debris_links_empty_granule():
    """
    A granule with no tifs yields no variables.
    """
    assert not debris.select_debris_links(["https://data.nsidc.org/HMA_DTE_1.00001.xml"])


def test_assemble_debris_mosaics_and_fills(tmp_path, target_grid):
    """
    Disjoint per-glacier tifs in different CRSs mosaic onto the grid.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    target_grid : xarray.Dataset
        The toy UTM grid.
    """
    # Left strip in the grid's own CRS; right strip in the neighboring UTM
    # zone to exercise the per-file reprojection. (The right tif is built
    # from a bounds envelope, so a strongly rotated CRS would smear it into
    # the gap; the adjacent zone keeps the rotation to a few degrees.)
    left = _write_tif(tmp_path / "left.tif", 0.5, 500_000.0, 500_300.0, 6_700_000.0, 6_701_000.0)
    t = Transformer.from_crs(CRS, "EPSG:32607", always_xy=True)
    rx0, ry0, rx1, ry1 = t.transform_bounds(500_700.0, 6_700_000.0, 501_000.0, 6_701_000.0)
    right = _write_tif(tmp_path / "right.tif", 1.5, rx0, rx1, ry0, ry1, crs="EPSG:32607")

    geometries = [box(500_000.0, 6_700_000.0, 501_000.0, 6_701_000.0)]
    out = debris.assemble_debris({"debris_thickness": [left, right]}, target_grid, geometries)

    thickness = out["debris_thickness"]
    assert thickness.sel(x=500_150.0, y=6_700_500.0, method="nearest") == pytest.approx(0.5)
    assert thickness.sel(x=500_900.0, y=6_700_500.0, method="nearest") == pytest.approx(1.5)
    # The 300-700 m gap between the two strips is debris-free.
    assert thickness.sel(x=500_500.0, y=6_700_500.0, method="nearest") == pytest.approx(0.0)
    assert not thickness.isnull().any()

    # No melt-factor tifs: the field is the neutral constant 1 everywhere.
    assert (out["debris_melt_factor"] == 1.0).all()

    for name in ("debris_thickness", "debris_melt_factor"):
        assert "standard_name" not in out[name].attrs
        assert out[name].attrs["units"]


def test_assemble_debris_clips_to_outline(tmp_path, target_grid):
    """
    Fill cells outside the outline with the neutral constants.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    target_grid : xarray.Dataset
        The toy UTM grid.
    """
    tif = _write_tif(tmp_path / "full.tif", 2.0, 500_000.0, 501_000.0, 6_700_000.0, 6_701_000.0)
    geometries = [box(500_000.0, 6_700_000.0, 500_500.0, 6_701_000.0)]
    out = debris.assemble_debris({"debris_thickness": [tif]}, target_grid, geometries)
    thickness = out["debris_thickness"]
    assert thickness.sel(x=500_100.0, y=6_700_500.0, method="nearest") == pytest.approx(2.0)
    assert thickness.sel(x=500_900.0, y=6_700_500.0, method="nearest") == pytest.approx(0.0)


def test_debris_from_grid_end_to_end(tmp_path, target_grid, monkeypatch):
    """
    The staged file carries both variables and a single bounded time record.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    target_grid : xarray.Dataset
        The toy UTM grid.
    monkeypatch : pytest.MonkeyPatch
        Used to stub out the Earthdata download.
    """
    tif = _write_tif(tmp_path / "glacier.tif", 0.25, 500_000.0, 501_000.0, 6_700_000.0, 6_701_000.0)
    monkeypatch.setattr(debris, "download_debris_tifs", lambda *a, **k: {"debris_thickness": [tif]})

    geometries = [box(500_000.0, 6_700_000.0, 501_000.0, 6_701_000.0)]
    path = tmp_path / "debris_test.nc"
    debris.debris_from_grid(target_grid, geometries, rgi_id="test", path=path, staging_path=tmp_path)

    assert check_xr_lazy(path)
    # ``decode_coords="all"`` promotes ``spatial_ref`` to a coordinate so
    # rioxarray can resolve the CRS from the on-disk metadata alone.
    with xr.open_dataset(path, decode_coords="all") as ds:
        assert ds.sizes["time"] == 1
        # ``decode_coords="all"`` moves the CF ``bounds`` link into encoding.
        assert ds["time"].encoding.get("bounds") == "time_bnds"
        bnds = ds["time_bnds"].values[0]
        assert bnds[0] == np.datetime64("2000-01-01")
        assert bnds[1] == np.datetime64("2019-01-01")
        assert ds["debris_thickness"].isel(time=0).mean() == pytest.approx(0.25)
        assert (ds["debris_melt_factor"] == 1.0).all()
        assert ds.rio.crs is not None

    # A valid cache short-circuits: the download must not be reached again.
    def _boom(*_args, **_kwargs):
        """
        Fail the test if the download is attempted.

        Parameters
        ----------
        *_args : tuple
            Ignored.
        **_kwargs : dict
            Ignored.

        Raises
        ------
        AssertionError
            Always.
        """
        raise AssertionError("download_debris_tifs called despite a valid cache")

    monkeypatch.setattr(debris, "download_debris_tifs", _boom)
    cached = debris.debris_from_grid(target_grid, geometries, rgi_id="test", path=path, staging_path=tmp_path)
    assert cached.sizes["time"] == 1


def test_download_debris_tifs_rejects_unknown_dataset(tmp_path):
    """
    An unknown dataset name raises before any network access.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    with pytest.raises(NotImplementedError, match="rounce"):
        debris.download_debris_tifs((0.0, 0.0, 1.0, 1.0), tmp_path, dataset="nope")
