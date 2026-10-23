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

The end-to-end cases build a store whose answer is known: the levels follow
one straight lapse line, so the fit at the terrain is exact, and the 2 m
temperature is set below it by a bias of exactly the fitted form -- linear in
elevation with one yearly cycle, the cycle changing with elevation on glacier
and not off it. The script has to give that bias back, from one store
spanning a leap year and the next, and from the same two years as two stores.
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
NX, NY, DX = 79, 30, 2500.0
X = -100_000.0 + DX * np.arange(NX)
Y = -2_000_000.0 + DX * np.arange(NY)
#: Terrain rising along x from 200 m to 4100 m in 50 m steps, so each 250 m
#: bin holds an odd number of heights and its median is one of them -- the
#: value a histogram reads off exactly, to within half its step.
OROGRAPHY = np.broadcast_to(200.0 + 50.0 * np.arange(NX), (NY, NX))
#: Glacier: the upper half of the rows.
GLACIER_ROWS = slice(NY // 2, NY)
LEVELS = np.array([1000.0, 900.0, 800.0, 750.0, 500.0])
LEVEL_HEIGHTS = np.array([100.0, 1000.0, 2000.0, 2500.0, 5600.0])
LAPSE = -0.0065  # K per m
#: The bias the store is built with, in the coefficients of mod.fit_seasonal.
TRUE_ON = {"mean": 6.0, "mean_per_km": -1.5, "cos": -2.0, "sin": -0.5, "cos_per_km": 0.6, "sin_per_km": 0.1}
TRUE_OFF = {"mean": 1.5, "mean_per_km": 0.0, "cos": 1.0, "sin": 0.0, "cos_per_km": 0.0, "sin_per_km": 0.0}


def fraction_of_year(time: pd.DatetimeIndex) -> np.ndarray:
    """
    Compute the fraction of the year at the middle of each day, independently of the module.

    Parameters
    ----------
    time : pandas.DatetimeIndex
        Days.

    Returns
    -------
    numpy.ndarray
        Fractions.
    """
    days = np.array([366.0 if t.is_leap_year else 365.0 for t in time])
    return (np.array([t.dayofyear for t in time]) - 0.5) / days


def true_bias(coef: dict[str, float], z: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    """
    Evaluate the bias the store is built with.

    Parameters
    ----------
    coef : dict
        :data:`TRUE_ON` or :data:`TRUE_OFF`.
    z : numpy.ndarray
        Elevations, m.
    fraction : numpy.ndarray
        Fractions of the year, broadcast against *z*.

    Returns
    -------
    numpy.ndarray
        Bias, K.
    """
    z_km = np.asarray(z) / 1000.0
    angle = 2 * np.pi * np.asarray(fraction)
    return (
        coef["mean"]
        + coef["mean_per_km"] * z_km
        + (coef["cos"] + coef["cos_per_km"] * z_km) * np.cos(angle)
        + (coef["sin"] + coef["sin_per_km"] * z_km) * np.sin(angle)
    )


def synthetic_store(path: Path, start: str = "2005-06-01", end: str = "2005-06-03") -> Path:
    """
    Write a small store laid out like one from ``pism-s4f-carra2``.

    Parameters
    ----------
    path : pathlib.Path
        Store to write.
    start, end : str
        First and last day, inclusive.

    Returns
    -------
    pathlib.Path
        The store.
    """
    time = pd.date_range(start, end, freq="D")
    heights = np.broadcast_to(LEVEL_HEIGHTS[None, :, None, None], (time.size, LEVELS.size, NY, NX))
    air_temp = 288.0 + LAPSE * heights
    t_ground = 288.0 + LAPSE * OROGRAPHY
    fraction = fraction_of_year(time)[:, None, None]
    bias = np.broadcast_to(true_bias(TRUE_OFF, OROGRAPHY, fraction), (time.size, NY, NX)).copy()
    bias[:, GLACIER_ROWS] = true_bias(TRUE_ON, OROGRAPHY[GLACIER_ROWS], fraction)
    t2m = t_ground[None] - bias
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


def test_year_fraction_counts_leap_years():
    """
    Every year runs from 0 to 1, a day counting at its middle, leap years included.
    """
    time = pd.to_datetime(["2004-01-01", "2004-12-31", "2005-01-01", "2005-07-02", "2005-12-31"])
    np.testing.assert_allclose(mod.year_fraction(time), [0.5 / 366, 365.5 / 366, 0.5 / 365, 182.5 / 365, 364.5 / 365])


def multi_year_sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample the on-glacier model over two years, the first a leap year.

    Returns
    -------
    z : numpy.ndarray
        Elevations, m.
    fraction : numpy.ndarray
        Fractions of the year.
    bias : numpy.ndarray
        Bias, K.
    """
    time = pd.date_range("2004-01-01", "2005-12-31", freq="D")
    z, fraction = np.meshgrid(np.arange(500.0, 4000.0, 250.0), fraction_of_year(time))
    return z.ravel(), fraction.ravel(), true_bias(TRUE_ON, z, fraction).ravel()


def test_fit_seasonal_recovers_the_model_over_several_years():
    """
    All six coefficients come back from exact data spanning two years.
    """
    fit = mod.fit_seasonal(*multi_year_sample())
    assert fit is not None
    for name, value in TRUE_ON.items():
        assert fit[name] == pytest.approx(value, abs=1e-9), name
    assert fit["rmse"] == pytest.approx(0.0, abs=1e-9) and fit["r2"] == pytest.approx(1.0)
    z, fraction, bias = multi_year_sample()
    np.testing.assert_allclose(mod.seasonal_curve(fit, z, fraction), bias, atol=1e-9)


def test_fit_seasonal_with_one_cycle_for_all_elevations():
    """
    Without the elevation-dependent season the per-km seasonal terms are zero.
    """
    z, fraction, _ = multi_year_sample()
    shared = {**TRUE_ON, "cos_per_km": 0.0, "sin_per_km": 0.0}
    fit = mod.fit_seasonal(z, fraction, true_bias(shared, z, fraction), elevation_dependent=False)
    assert fit is not None
    for name, value in shared.items():
        assert fit[name] == pytest.approx(value, abs=1e-9), name


def test_fit_seasonal_needs_spread_in_elevation_and_season():
    """
    A single elevation or a single day cannot pin down the model.
    """
    z, fraction, bias = multi_year_sample()
    assert mod.fit_seasonal(np.full_like(z, 1000.0), fraction, bias) is None
    assert mod.fit_seasonal(z, np.full_like(fraction, 0.5), bias) is None
    assert mod.fit_seasonal(z[:5], fraction[:5], bias[:5]) is None


def test_seasonal_cycle_gives_amplitude_and_peak_day():
    """
    A pure sine of amplitude 2 peaks a quarter into the year; elevation can shift it.
    """
    fit = {"mean": 0.0, "mean_per_km": 0.0, "cos": 0.0, "sin": 2.0, "cos_per_km": 0.0, "sin_per_km": 0.0}
    amplitude, day = mod.seasonal_cycle(fit, np.array([0.0, 3000.0]))
    np.testing.assert_allclose(amplitude, 2.0)
    np.testing.assert_allclose(day, 0.25 * 365 + 0.5)
    # cos turning on with height moves the peak toward 1 January.
    amplitude, day = mod.seasonal_cycle({**fit, "cos_per_km": 2.0 / 3.0}, np.array([3000.0]))
    np.testing.assert_allclose(amplitude, np.hypot(2.0, 2.0))
    np.testing.assert_allclose(day, 0.125 * 365 + 0.5)


def test_bins_line_up_across_ranges():
    """
    Place bins at fixed multiples of the width, whatever range a store covers.
    """
    np.testing.assert_array_equal(mod.bin_centre(np.array([0.0, 249.9, 250.0, 1010.0]), 250.0), [125, 125, 375, 1125])


def test_histogram_quantiles_match_numpy():
    """
    Read quantiles off a fine histogram and match numpy's to within a bin.
    """
    rng = np.random.default_rng(1)
    values = rng.gamma(2.0, 1.5, size=(3, 20_000)) - 1.0
    edges = mod.value_edges(0.05)
    counts = np.stack([np.histogram(v, bins=edges)[0] for v in values])
    got = mod.histogram_quantiles(counts, edges, (0.25, 0.5, 0.75))
    expected = np.quantile(values, (0.25, 0.5, 0.75), axis=1).T
    np.testing.assert_allclose(got, expected, atol=0.05)
    assert np.isnan(mod.histogram_quantiles(np.zeros((1, edges.size - 1)), edges, (0.5,))).all()


def test_histograms_do_not_depend_on_tiling(tmp_path):
    """
    Small odd tiles in parallel count exactly what one tile does; sea-level cells are left out.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    ds = mod.open_carra2(synthetic_store(tmp_path / "store.zarr"))
    on_glacier = mod.glacier_mask(ds, glacier_outline())
    whole = mod.bias_histograms(ds, on_glacier, tile=256, ntasks=1)
    tiled = mod.bias_histograms(ds, on_glacier, tile=7, ntasks=3, time_block=2)
    np.testing.assert_array_equal(whole["counts"].values, tiled["counts"].values)
    np.testing.assert_array_equal(whole["cells"].values, tiled["cells"].values)
    assert int(whole["cells"].sum()) == NX * NY and whole.attrs["n_sea"] == 0

    # Everything below 1000 m counted as sea: those cells and bins are gone.
    high = mod.bias_histograms(ds, on_glacier, min_elevation=1000.0)
    assert high.attrs["n_sea"] == int((OROGRAPHY < 1000.0).sum())
    assert high["centre"].min() == pytest.approx(1125.0)
    assert int(high["cells"].sum()) == int((OROGRAPHY >= 1000.0).sum())


def test_daily_medians_match_the_cells(tmp_path):
    """
    Each daily bin median is the median of that bin's cells that day, to half a histogram step.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    ds = mod.open_carra2(synthetic_store(tmp_path / "store.zarr"))
    on_glacier = mod.glacier_mask(ds, glacier_outline())
    daily = mod.daily_medians(mod.bias_histograms(ds, on_glacier), source="s")
    bias = mod.ground_bias(ds).load().values
    centre = mod.bin_centre(OROGRAPHY, mod.DEFAULT_BIN_WIDTH)
    for row in daily.sample(20, random_state=0).itertuples():
        day = int(np.flatnonzero(ds["time"].values == np.datetime64(row.time))[0])
        cells = (centre == row.centre) & (on_glacier.values == (row.category == "on_glacier"))
        assert row.cells == cells.sum()
        assert row.median == pytest.approx(np.median(bias[day][cells]), abs=mod.DEFAULT_HIST_STEP / 2 + 1e-6)


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


def check_fit(fits: pd.DataFrame) -> None:
    """
    Compare the script's fit with the model the store was built with.

    Bin medians sit at the middle terrain height of the bin, 25 m below most
    bin centres, so the intercepts carry that offset; medians read off the
    histograms are good to half a histogram step.

    Parameters
    ----------
    fits : pandas.DataFrame
        ``bias_seasonal_fit.csv``.
    """
    for category, truth in (("on_glacier", TRUE_ON), ("off_glacier", TRUE_OFF)):
        fit = fits.loc[category]
        for name in ("mean_per_km", "cos_per_km", "sin_per_km"):
            assert fit[name] == pytest.approx(truth[name], abs=0.05), (category, name)
        for name in ("mean", "cos", "sin"):
            assert fit[name] == pytest.approx(truth[name], abs=0.1), (category, name)
    assert fits.loc["off_glacier", "rmse"] < 0.03
    assert fits.loc["on_glacier", "rmse"] < 0.05


def test_main_recovers_the_built_in_bias(tmp_path):
    """
    End to end over a leap year and the next: the fit gives back the bias the store was built with.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    store = synthetic_store(tmp_path / "carra2_daily_test.zarr", "2004-01-01", "2005-12-31")
    outlines = tmp_path / "outlines.gpkg"
    glacier_outline().to_file(outlines)
    sites = tmp_path / "sites.gpkg"
    points = gpd.GeoDataFrame(
        {"site": ["low", "high"]}, geometry=[Point(X[2], Y[-2]), Point(X[-3], Y[-2])], crs=CRS
    ).to_crs("EPSG:4326")
    points.to_file(sites)
    out = tmp_path / "out"

    assert mod.main([str(store), "--outlines", str(outlines), "--sites", str(sites), "--output-path", str(out)]) == 0
    for name in (
        "bias_by_elevation.png",
        "bias_seasonal_fit.png",
        "bias_by_elevation_medians.csv",
        "bias_daily_medians.csv",
        "bias_seasonal_fit.csv",
        "bias_seasonal_by_elevation.csv",
        "site_bias.csv",
        "site_bias_monthly.png",
        "site_bias_per_site.png",
    ):
        assert (out / name).exists(), name

    check_fit(pd.read_csv(out / "bias_seasonal_fit.csv", index_col="category"))

    daily = pd.read_csv(out / "bias_daily_medians.csv")
    assert daily["time"].nunique() == 731

    cycle = pd.read_csv(out / "bias_seasonal_by_elevation.csv")
    off = cycle[cycle["category"] == "off_glacier"]
    np.testing.assert_allclose(off["amplitude"], 1.0, atol=0.03)
    np.testing.assert_allclose(off["annual_mean"], 1.5, atol=0.03)

    site = pd.read_csv(out / "site_bias.csv", parse_dates=["time"])
    low = site[site["site"] == "low"].sort_values("time")
    expected = true_bias(TRUE_ON, OROGRAPHY[0, 2], fraction_of_year(pd.DatetimeIndex(low["time"])))
    np.testing.assert_allclose(low["bias"], expected, atol=1e-3)


def test_years_as_separate_stores_give_the_same_fit(tmp_path):
    """
    The same two years as one store per year pool into the same seasonal fit.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    stores = [
        synthetic_store(tmp_path / "carra2_daily_2004.zarr", "2004-01-01", "2004-12-31"),
        synthetic_store(tmp_path / "carra2_daily_2005.zarr", "2005-01-01", "2005-12-31"),
    ]
    outlines = tmp_path / "outlines.gpkg"
    glacier_outline().to_file(outlines)
    out = tmp_path / "out"
    assert mod.main([*map(str, stores), "--outlines", str(outlines), "--output-path", str(out)]) == 0
    check_fit(pd.read_csv(out / "bias_seasonal_fit.csv", index_col="category"))
    assert set(pd.read_csv(out / "bias_daily_medians.csv")["source"]) == {"carra2_daily_2004", "carra2_daily_2005"}


def test_main_with_one_cycle_for_all_elevations(tmp_path):
    """
    ``--season constant`` fits no elevation dependence into the cycle.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    store = synthetic_store(tmp_path / "store.zarr", "2005-01-01", "2005-12-31")
    outlines = tmp_path / "outlines.gpkg"
    glacier_outline().to_file(outlines)
    out = tmp_path / "out"
    assert mod.main([str(store), "--outlines", str(outlines), "--season", "constant", "--output-path", str(out)]) == 0
    fits = pd.read_csv(out / "bias_seasonal_fit.csv", index_col="category")
    assert (fits[["cos_per_km", "sin_per_km"]] == 0).all().all()
    assert (fits["elevation_dependent"] == 0).all()


def test_main_selects_months(tmp_path):
    """
    ``--months`` keeps only those months; none left is an error, not an empty result.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    store = synthetic_store(tmp_path / "store.zarr", "2005-05-01", "2005-07-31")
    outlines = tmp_path / "outlines.gpkg"
    glacier_outline().to_file(outlines)
    out = tmp_path / "jun"
    assert mod.main([str(store), "--outlines", str(outlines), "--months", "6", "--output-path", str(out)]) == 0
    months = pd.to_datetime(pd.read_csv(out / "bias_daily_medians.csv")["time"]).dt.month
    assert set(months) == {6}
    with pytest.raises(ValueError, match="no days"):
        mod.main([str(store), "--outlines", str(outlines), "--months", "1", "--output-path", str(tmp_path / "jan")])
