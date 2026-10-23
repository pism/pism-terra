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
Tests for the CARRA2 free-air versus 2 m temperature bias analysis.

The end-to-end case builds a store whose answer is known: the levels follow
one straight lapse line, so the fit at the terrain is exact, and the 2 m
temperature is set below it by a bias that is constant off glacier and
two-segment on glacier. The script has to give that bias back.
"""

import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401  pylint: disable=unused-import
import xarray as xr
from shapely.geometry import Point, box

from pism_terra.glacier import carra2_bias_correction as mod

CRS = "EPSG:3413"
NX, NY, DX = 40, 30, 2500.0
X = -100_000.0 + DX * np.arange(NX)
Y = -2_000_000.0 + DX * np.arange(NY)
#: Terrain rising along x from 200 m to 4100 m.
OROGRAPHY = np.broadcast_to(200.0 + 100.0 * np.arange(NX), (NY, NX))
#: Glacier: the upper half of the rows.
GLACIER_ROWS = slice(NY // 2, NY)
LEVELS = np.array([1000.0, 900.0, 800.0, 750.0, 500.0])
LEVEL_HEIGHTS = np.array([100.0, 1000.0, 2000.0, 2500.0, 5600.0])
LAPSE = -0.0065  # K per m
#: The bias the store is built with.
TRUE_BREAK, TRUE_AT_BREAK, TRUE_BELOW, TRUE_ABOVE, OFF_GLACIER = 2000.0, 1.5, -0.002, 0.0003, 1.5


def true_on_glacier(z: np.ndarray) -> np.ndarray:
    """
    Evaluate the on-glacier bias the store is built with.

    Parameters
    ----------
    z : numpy.ndarray
        Elevations, m.

    Returns
    -------
    numpy.ndarray
        Bias, K.
    """
    dz = np.asarray(z, dtype=float) - TRUE_BREAK
    return TRUE_AT_BREAK + TRUE_BELOW * np.minimum(dz, 0) + TRUE_ABOVE * np.maximum(dz, 0)


def synthetic_store(path: Path) -> Path:
    """
    Write a small store laid out like one from ``pism-s4f-carra2``.

    Parameters
    ----------
    path : pathlib.Path
        Store to write.

    Returns
    -------
    pathlib.Path
        The store.
    """
    time = pd.date_range("2005-06-01", periods=3, freq="D")
    heights = np.broadcast_to(LEVEL_HEIGHTS[None, :, None, None], (time.size, LEVELS.size, NY, NX))
    air_temp = 288.0 + LAPSE * heights
    t_ground = 288.0 + LAPSE * OROGRAPHY
    bias = np.full((NY, NX), OFF_GLACIER)
    bias[GLACIER_ROWS] = true_on_glacier(OROGRAPHY[GLACIER_ROWS])
    t2m = np.broadcast_to(t_ground - bias, (time.size, NY, NX))
    ds = xr.Dataset(
        {
            "air_temp": (("time", "pressure_level", "y", "x"), air_temp.astype("float32"), {"units": "K"}),
            "geopotential_height": (("time", "pressure_level", "y", "x"), heights.astype("float32"), {"units": "m"}),
            "air_temp_2m": (("time", "y", "x"), t2m.astype("float32"), {"units": "K"}),
            "orography": (("y", "x"), OROGRAPHY.astype("float32"), {"units": "m"}),
        },
        coords={"time": time, "pressure_level": LEVELS, "y": Y, "x": X},
    )
    ds = ds.rio.write_crs(CRS).rio.write_grid_mapping("spatial_ref")
    ds.to_zarr(path, mode="w", consolidated=True)
    return path


def glacier_outline() -> gpd.GeoDataFrame:
    """
    Build an outline around the glacier rows, in geographic coordinates like RGI.

    Returns
    -------
    geopandas.GeoDataFrame
        One polygon.
    """
    y0 = Y[GLACIER_ROWS.start] - DX / 2
    polygon = box(X[0] - DX / 2, y0, X[-1] + DX / 2, Y[-1] + DX / 2)
    return gpd.GeoDataFrame({"rgi_id": ["G1"]}, geometry=[polygon], crs=CRS).to_crs("EPSG:4326")


@pytest.mark.parametrize("below", [0, 1, None])
def test_fit_to_ground_matches_polyfit(below):
    """
    The vectorised fit is ordinary least squares through the chosen levels.

    Parameters
    ----------
    below : int or None
        Levels below the ground to use.
    """
    rng = np.random.default_rng(0)
    heights = np.array([80.0, 950.0, 1900.0, 2400.0, 5500.0])
    temps = 285.0 - 0.006 * heights + rng.normal(0, 0.5, heights.size)
    ground = 1500.0
    fit = mod.fit_to_ground(
        xr.DataArray(temps, dims="pressure_level"),
        xr.DataArray(heights, dims="pressure_level"),
        xr.DataArray(ground),
        below=below,
    )
    n_below = int((heights < ground).sum())
    keep = np.ones(heights.size, bool) if below is None else np.arange(heights.size) >= n_below - below
    slope, intercept = np.polyfit(heights[keep], temps[keep], 1)
    assert int(fit["n_levels"]) == keep.sum()
    assert float(fit["t_ground"]) == pytest.approx(intercept + slope * ground)
    assert float(fit["lapse_rate"]) == pytest.approx(1000 * slope)


def test_fit_to_ground_needs_two_levels():
    """
    A single usable level gives no fit, not a made-up one.
    """
    fit = mod.fit_to_ground(
        xr.DataArray([280.0, 270.0], dims="pressure_level"),
        xr.DataArray([100.0, 1500.0], dims="pressure_level"),
        xr.DataArray(1000.0),
        below=0,
    )
    assert int(fit["n_levels"]) == 1 and np.isnan(float(fit["t_ground"]))


def test_fit_piecewise_recovers_a_known_line():
    """
    Breakpoint, value there and both slopes come back from exact data.
    """
    z = np.arange(125.0, 4000.0, 250.0)
    fit = mod.fit_piecewise(z, true_on_glacier(z))
    assert fit is not None
    assert fit["z_break"] == pytest.approx(TRUE_BREAK, abs=15)
    assert fit["y_break"] == pytest.approx(TRUE_AT_BREAK, abs=0.05)
    assert fit["k_below"] == pytest.approx(TRUE_BELOW, rel=0.02)
    assert fit["k_above"] == pytest.approx(TRUE_ABOVE, rel=0.05)
    np.testing.assert_allclose(mod.piecewise_line(fit, z), true_on_glacier(z), atol=0.05)


def test_fit_piecewise_needs_enough_points():
    """
    Fewer than two segments' worth of points gives no fit.
    """
    assert mod.fit_piecewise([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 2.0, 1.0], min_points=3) is None


def test_bins_line_up_across_ranges():
    """
    Place bins at fixed multiples of the width, whatever range a store covers.
    """
    np.testing.assert_array_equal(mod.bin_centre(np.array([0.0, 249.9, 250.0, 1010.0]), 250.0), [125, 125, 375, 1125])


def test_read_vector_from_s3_fetches_shapefile_sidecars(tmp_path, monkeypatch):
    """
    A shapefile on S3 arrives with its sidecars, so GDAL can read it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Patch fixture.
    """
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    glacier_outline().to_file(bucket / "outlines.shp")

    class FakeS3:
        """
        Map ``s3://bucket/...`` onto the temporary bucket directory.

        Parameters
        ----------
        anon : bool
            Accepted like :class:`s3fs.S3FileSystem` does, and ignored.
        """

        def __init__(self, anon=True):  # pylint: disable=unused-argument
            pass

        @staticmethod
        def _local(uri):
            """
            Translate a URI to the local path.

            Parameters
            ----------
            uri : str
                ``s3://bucket/...``.

            Returns
            -------
            pathlib.Path
                The local file.
            """
            return bucket / uri.removeprefix("s3://bucket/")

        def exists(self, uri):
            """
            Tell whether the object exists.

            Parameters
            ----------
            uri : str
                Object URI.

            Returns
            -------
            bool
                True when it does.
            """
            return self._local(uri).exists()

        def glob(self, pattern):
            """
            List objects matching a pattern.

            Parameters
            ----------
            pattern : str
                URI pattern.

            Returns
            -------
            list of str
                Matching URIs, as s3fs returns them (without the scheme).
            """
            local = self._local(pattern)
            return [f"bucket/{p.name}" for p in local.parent.glob(local.name)]

        def get(self, source, destination):
            """
            Copy an object to a local file.

            Parameters
            ----------
            source : str
                Object URI or key.
            destination : str
                Local file.
            """
            shutil.copy(self._local(source.removeprefix("bucket/")), destination)

    monkeypatch.setattr(mod.s3fs, "S3FileSystem", FakeS3)
    frame = mod.read_vector("s3://bucket/outlines.shp")
    assert list(frame["rgi_id"]) == ["G1"] and frame.crs is not None
    with pytest.raises(FileNotFoundError):
        mod.read_vector("s3://bucket/missing.gpkg")


def test_glacier_mask_uses_cell_centres(tmp_path):
    """
    Exactly the rows inside the outline are on glacier.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    ds = mod.open_carra2(synthetic_store(tmp_path / "store.zarr"))
    mask = mod.glacier_mask(ds, glacier_outline())
    expected = np.zeros((NY, NX), bool)
    expected[GLACIER_ROWS] = True
    np.testing.assert_array_equal(mask.values, expected)


def test_open_carra2_refuses_an_incomplete_store(tmp_path):
    """
    A store without the 2 m temperature cannot be analysed.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    xr.Dataset({"air_temp": ("x", [1.0])}).to_zarr(tmp_path / "bad.zarr")
    with pytest.raises(ValueError, match="lacks"):
        mod.open_carra2(tmp_path / "bad.zarr")


def test_main_recovers_the_built_in_bias(tmp_path):
    """
    End to end: binned medians and fits give back the bias the store was built with.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    store = synthetic_store(tmp_path / "carra2_daily_test.zarr")
    outlines = tmp_path / "outlines.gpkg"
    glacier_outline().to_file(outlines)
    sites = tmp_path / "sites.gpkg"
    points = gpd.GeoDataFrame(
        {"site": ["low", "high"]}, geometry=[Point(X[2], Y[-2]), Point(X[-3], Y[-2])], crs=CRS
    ).to_crs("EPSG:4326")
    points.to_file(sites)
    out = tmp_path / "out"

    assert mod.main([str(store), "--outlines", str(outlines), "--sites", str(sites), "--output-path", str(out)]) == 0
    for name in ("bias_by_elevation.png", "bias_by_elevation_medians.csv", "bias_by_elevation_fits.csv"):
        assert (out / name).exists(), name
    for name in ("site_bias.csv", "site_bias_monthly.png", "site_bias_per_site.png"):
        assert (out / name).exists(), name

    medians = pd.read_csv(out / "bias_by_elevation_medians.csv")
    off = medians[medians["category"] == "off_glacier"]
    np.testing.assert_allclose(off["median"], OFF_GLACIER, atol=1e-4)
    on = medians[medians["category"] == "on_glacier"]
    # Each bin holds two terrain heights (100 m steps, 250 m bins), so its median
    # is the bias halfway between them; the cells are the 15 glacier rows times them.
    assert (on["cells"] % (NY - GLACIER_ROWS.start) == 0).all()

    fits = pd.read_csv(out / "bias_by_elevation_fits.csv", index_col="category")
    on_fit = fits.loc["on_glacier"]
    assert on_fit["z_break"] == pytest.approx(TRUE_BREAK, abs=150)
    assert on_fit["k_below_per_km"] == pytest.approx(1000 * TRUE_BELOW, rel=0.1)
    assert on_fit["k_above_per_km"] == pytest.approx(1000 * TRUE_ABOVE, rel=0.25)
    assert fits.loc["off_glacier", "rmse"] == pytest.approx(0.0, abs=1e-4)

    site = pd.read_csv(out / "site_bias.csv")
    assert set(site["site"]) == {"low", "high"}
    low = site[site["site"] == "low"]
    assert low["bias"].to_numpy() == pytest.approx(true_on_glacier(OROGRAPHY[0, 2]), abs=1e-3)


def test_main_selects_months(tmp_path):
    """
    ``--months`` keeps only those months; none left is an error, not an empty result.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    store = synthetic_store(tmp_path / "store.zarr")
    outlines = tmp_path / "outlines.gpkg"
    glacier_outline().to_file(outlines)
    assert (
        mod.main([str(store), "--outlines", str(outlines), "--months", "6", "--output-path", str(tmp_path / "jun")])
        == 0
    )
    with pytest.raises(ValueError, match="no days"):
        mod.main([str(store), "--outlines", str(outlines), "--months", "1", "--output-path", str(tmp_path / "jan")])
