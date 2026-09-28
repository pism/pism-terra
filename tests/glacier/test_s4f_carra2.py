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
Tests for :mod:`pism_terra.glacier.s4f_carra2`.

CDS is not contacted. The download is replaced by synthetic per-year files
laid out the way CDS actually delivers this dataset (checked against a real
file): ``y``/``x`` index dims, 2-D latitude/longitude, ECMWF short names,
the vertical dim under its GRIB name ``isobaricInhPa``, and a forecast
reference ``time`` with the ``valid_time`` beside it. The latitude/longitude
come from projecting a real block of the CARRA2 lattice, so recovering
``x``/``y`` exactly is a test of the projection round trip, not of the
fixture. The block is larger than the outline's box, so the local clip is
exercised too.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import box

from pism_terra.glacier import s4f_carra2 as mod
from pism_terra.glacier.climate import CARRA2_PROJ

CARRA2_CRS = CRS.from_proj4(CARRA2_PROJ)

#: A block of the CARRA2 lattice, by lattice index so the axes are exact.
#: 40 cells = 100 km a side; the outline below sits in its middle.
I0, J0, NX, NY = 1400, 1900, 40, 40
X_TRUE = mod.GRID_ORIGIN + mod.GRID_SPACING * np.arange(I0, I0 + NX)
Y_TRUE = mod.GRID_ORIGIN + mod.GRID_SPACING * np.arange(J0, J0 + NY)

#: The outline: a 10 km square at the block's centre, in CARRA2 metres.
OUTLINE_X = (X_TRUE[NX // 2] - 5_000, X_TRUE[NX // 2] + 5_000)
OUTLINE_Y = (Y_TRUE[NY // 2] - 5_000, Y_TRUE[NY // 2] + 5_000)


def lattice_latlon() -> tuple[np.ndarray, np.ndarray]:
    """
    Latitude/longitude of the fixture block, from the real projection.

    Returns
    -------
    tuple of numpy.ndarray
        2-D latitude and longitude, shape ``(NY, NX)``.
    """
    xx, yy = np.meshgrid(X_TRUE, Y_TRUE)
    inverse = Transformer.from_crs(CARRA2_CRS, "EPSG:4326", always_xy=True)
    lon, lat = inverse.transform(xx, yy)
    return lat, lon


def write_cds_file(path: Path, year: int, kind: str) -> Path:
    """
    Write one file the way CDS delivers a daily-means file.

    Parameters
    ----------
    path : pathlib.Path
        File to write.
    year : int
        Year of the stamps.
    kind : str
        ``"temperature"``, ``"snow"`` or ``"forecast"``.

    Returns
    -------
    pathlib.Path
        The file.
    """
    lat, lon = lattice_latlon()
    days = [2, 1, 3, 3, 4]  # out of order, with a duplicate: what has to be cleaned up
    valid = pd.to_datetime([f"{year}-01-{d:02d} 12:00" for d in days])
    if kind == "forecast":
        # Forecast-based: the reference time is the day before the day the
        # mean describes. Stamps must follow valid_time, not time.
        time = valid.normalize() - pd.Timedelta(days=1)
    else:
        time = valid.normalize()
    coords = {
        "time": ("time", time.values, {"standard_name": "forecast_reference_time"}),
        "valid_time": ("time", valid.values),
        "latitude": (("y", "x"), lat),
        "longitude": (("y", "x"), lon),
    }
    rng = np.random.default_rng(int(year))
    if kind == "temperature":
        levels = np.array([500.0, 1000.0, 750.0, 900.0, 800.0])  # unordered, GRIB name
        ds = xr.Dataset(
            {
                "t": (
                    ("time", "isobaricInhPa", "y", "x"),
                    (250.0 + rng.random((len(time), 5, NY, NX))).astype("float32"),
                    {"standard_name": "air_temperature", "units": "K", "GRIB_shortName": "t"},
                )
            },
            coords={**coords, "isobaricInhPa": ("isobaricInhPa", levels)},
        )
    elif kind == "snow":
        ds = xr.Dataset(
            {
                "sde": (
                    ("time", "y", "x"),
                    rng.random((len(time), NY, NX)).astype("float32"),
                    {"long_name": "Snow depth", "units": "m", "GRIB_shortName": "sde"},
                )
            },
            coords=coords,
        )
    else:
        shape = (len(time), NY, NX)
        ds = xr.Dataset(
            {
                "tp": (
                    ("time", "y", "x"),
                    rng.random(shape).astype("float32"),
                    {"units": "kg m**-2", "GRIB_shortName": "tp"},
                ),
                "ssr": (
                    ("time", "y", "x"),
                    rng.random(shape).astype("float32"),
                    {"standard_name": "surface_net_downward_shortwave_flux", "units": "J m**-2"},
                ),
                "ssrd": (
                    ("time", "y", "x"),
                    rng.random(shape).astype("float32"),
                    {"standard_name": "surface_downwelling_shortwave_flux_in_air", "units": "J m**-2"},
                ),
            },
            coords=coords,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


@pytest.fixture(name="fake_download")
def fixture_fake_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """
    Replace the CDS download with synthetic files and record the requests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Patch fixture.

    Returns
    -------
    dict
        Recorded requests, filled in as the fake runs.
    """
    seen: dict = {}

    def fake(
        dataset, request, file_path="tmp.nc", force_overwrite=False, max_workers=5
    ):  # pylint: disable=unused-argument
        """
        Stand in for :func:`pism_terra.download.carra_download_request`.

        Parameters
        ----------
        dataset : str
            CDS dataset.
        request : dict
            CDS request.
        file_path : Path or str, optional
            Cache file name; its stem names the request and its parent is the cache.
        force_overwrite : bool, optional
            Ignored.
        max_workers : int, optional
            Ignored.

        Returns
        -------
        list of pathlib.Path
            One file per requested year.
        """
        name = Path(file_path).stem
        seen[name] = {"dataset": dataset, "request": dict(request), "cache": Path(file_path).parent}
        return [write_cds_file(tmp_path / "cds" / f"{name}_{y}.nc", int(y), name) for y in request["year"]]

    monkeypatch.setattr(mod, "carra_download_request", fake)
    return seen


@pytest.fixture(name="outline")
def fixture_outline() -> gpd.GeoDataFrame:
    """
    The outline, as RGI7 ships it: geographic coordinates.

    Returns
    -------
    geopandas.GeoDataFrame
        One polygon with an ``rgi_id``.
    """
    square = gpd.GeoDataFrame(
        {"rgi_id": ["RGI2000-v7.0-C-01-12784"]},
        geometry=[box(OUTLINE_X[0], OUTLINE_Y[0], OUTLINE_X[1], OUTLINE_Y[1])],
        crs=CARRA2_CRS,
    )
    return square.to_crs("EPSG:4326")


@pytest.mark.parametrize(
    ("rgi_id", "expected"),
    [("RGI2000-v7.0-C-01-12784", ("C", "01")), ("RGI2000-v7.0-G-05-00007", ("G", "05"))],
)
def test_parse_rgi_id(rgi_id, expected):
    """
    Read outline type and region off the id.

    Parameters
    ----------
    rgi_id : str
        An RGI7 id.
    expected : tuple of str
        Outline type and region.
    """
    assert mod.parse_rgi_id(rgi_id) == expected
    assert expected[1] in mod.RGI7_REGIONS


def test_parse_rgi_id_rejects_other_ids():
    """
    Refuse an id that is not RGI7, rather than guessing a region.
    """
    with pytest.raises(ValueError, match="not an RGI7 id"):
        mod.parse_rgi_id("RGI60-01.12784")


def test_area_is_north_west_south_east_with_margin(outline):
    """
    Order the geographic box the way CDS wants it, padded outward.

    Kept for the record in the store's attributes even though CDS ignores it.

    Parameters
    ----------
    outline : geopandas.GeoDataFrame
        The outline.
    """
    west, south, east, north = outline.total_bounds
    n, w, s, e = mod.area_from_outline(outline, margin_km=25.0)
    assert n > north and s < south and w < west and e > east
    # 25 km is ~0.22 degrees of latitude, and longitude stretches by 1/cos(lat).
    assert n - north == pytest.approx(0.22, abs=0.02)
    assert (west - w) * np.cos(np.deg2rad(north)) == pytest.approx(0.22, abs=0.03)
    assert n <= 90.0


def test_bbox_is_the_outline_in_carra2_metres_with_margin(outline):
    """
    Give the local clip the outline's box in CARRA2 metres, padded.

    Parameters
    ----------
    outline : geopandas.GeoDataFrame
        The outline.
    """
    x_min, x_max, y_min, y_max = mod.bbox_from_outline(outline, margin_km=25.0)
    assert x_min == pytest.approx(OUTLINE_X[0] - 25_000, abs=1.0)
    assert x_max == pytest.approx(OUTLINE_X[1] + 25_000, abs=1.0)
    assert y_min == pytest.approx(OUTLINE_Y[0] - 25_000, abs=1.0)
    assert y_max == pytest.approx(OUTLINE_Y[1] + 25_000, abs=1.0)


def test_requests_use_the_pan_carra_keys_and_no_area():
    """
    Ask for the right things under the keys the pan-CARRA form actually has.

    The pressure levels go under ``level_location``; analysis- and
    forecast-based fields cannot share a request; and no ``area`` is sent,
    since CDS returns the whole domain regardless and a request without one
    is the same request for every glacier.
    """
    reqs = mod.build_requests([1990, 1991])
    assert set(reqs) == {"temperature", "snow", "forecast"}
    t = reqs["temperature"]
    assert t["level_type"] == "pressure_levels" and t["product_type"] == "analysis_based"
    assert t["level_location"] == ["1000", "900", "800", "750", "500"]
    assert "pressure_level" not in t
    assert reqs["snow"]["product_type"] == "analysis_based" and reqs["snow"]["variable"] == ["snow_depth"]
    assert reqs["forecast"]["product_type"] == "forecast_based"
    for r in reqs.values():
        assert r["time_aggregation"] == "daily" and r["year"] == ["1990", "1991"]
        assert r["data_format"] == "netcdf" and len(r["day"]) == 31
        assert "area" not in r


def test_projected_axes_recover_the_lattice_exactly():
    """
    Get CARRA2's own x/y back from the 2-D latitude/longitude CDS returns.

    Bit-exact after snapping, because the fixture's geographic coordinates
    were produced by the same projection. This is the check that ran clean
    on a real CDS file: 2869 cells from -3585000 m at 2500 m, to the metre.
    """
    lat, lon = lattice_latlon()
    x, y = mod.projected_axes(lat, lon)
    np.testing.assert_array_equal(x, X_TRUE)
    np.testing.assert_array_equal(y, Y_TRUE)


def test_projected_axes_reject_a_foreign_grid():
    """
    Refuse coordinates that are not on CARRA2's lattice.

    A regular lat/lon grid projects to a curved, non-rectilinear set of
    points; writing that under CARRA2's CRS would georeference it wrongly.
    """
    lon, lat = np.meshgrid(np.linspace(-141.5, -140.0, 8), np.linspace(60.0, 61.0, 6))
    with pytest.raises(ValueError, match="rectilinear|lattice"):
        mod.projected_axes(lat, lon)


def test_run_writes_a_clipped_store_on_the_carra2_grid(tmp_path: Path, fake_download, outline):
    """
    End to end with the download faked: clip, axes, levels, time, CRS.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    fake_download : dict
        Recorded requests.
    outline : geopandas.GeoDataFrame
        The outline.
    """
    rgi_file = tmp_path / "outline.gpkg"
    outline.to_file(rgi_file)
    store = mod.run("RGI2000-v7.0-C-01-12784", tmp_path / "out", rgi_file=rgi_file, start=1990, end=1991)
    assert store.name == "carra2_daily_RGI2000-v7.0-C-01-12784.zarr"
    assert fake_download["temperature"]["request"]["year"] == ["1990", "1991"]
    # The cache is shared, not per glacier: the files are the whole domain.
    assert fake_download["temperature"]["cache"] == tmp_path / "out" / "cds"

    # As a consumer must open it: decode_coords="all" is what lets rioxarray
    # follow the grid_mapping pointer to the stored spatial_ref.
    ds = xr.open_zarr(store, consolidated=True, decode_coords="all")
    assert set(ds.data_vars) >= {
        "air_temp",
        "snow_depth",
        "precipitation",
        "surface_net_solar_radiation",
        "surface_solar_radiation_downwards",
    }
    assert "time_bounds" in ds.variables and "time_bounds" not in ds.data_vars

    # Clipped to the outline's padded box: a strict, contiguous subset of the
    # lattice, still on CARRA2's own coordinates to the metre.
    x_min, x_max, y_min, y_max = mod.bbox_from_outline(outline, mod.DEFAULT_MARGIN_KM)
    assert 0 < ds.sizes["x"] < NX and 0 < ds.sizes["y"] < NY
    np.testing.assert_array_equal(ds["x"].values, X_TRUE[(X_TRUE >= x_min) & (X_TRUE <= x_max)])
    np.testing.assert_array_equal(ds["y"].values, Y_TRUE[(Y_TRUE >= y_min) & (Y_TRUE <= y_max)])
    assert ds["x"].attrs["standard_name"] == "projection_x_coordinate"
    assert ds["latitude"].dims == ("y", "x") and ds["latitude"].shape == (ds.sizes["y"], ds.sizes["x"])

    # Vertical axis: bottom up, CF direction stated.
    np.testing.assert_array_equal(ds["pressure_level"].values, [1000, 900, 800, 750, 500])
    assert ds["pressure_level"].attrs["positive"] == "down" and ds["pressure_level"].attrs["units"] == "hPa"
    assert ds["air_temp"].dims == ("time", "pressure_level", "y", "x")
    assert ds["snow_depth"].dims == ("time", "y", "x")

    # Time: daily at 00:00, duplicates dropped, sorted, two years concatenated.
    times = pd.DatetimeIndex(ds["time"].values)
    assert len(times) == 8 and times.is_monotonic_increasing and times.is_unique
    assert (times.values == times.values.astype("datetime64[D]").astype("datetime64[ns]")).all()
    assert times[0] == pd.Timestamp("1990-01-01") and times[-1] == pd.Timestamp("1991-01-04")
    assert (ds["time"].encoding.get("bounds") or ds["time"].attrs.get("bounds")) == "time_bounds"
    bounds = pd.DatetimeIndex(ds["time_bounds"].values[:, 1])
    assert (bounds - times == pd.Timedelta(days=1)).all()

    # CRS: CARRA2's, no reprojection.
    assert ds.rio.crs is not None and CRS(ds.rio.crs) == CARRA2_CRS


def test_forecast_fields_are_stamped_by_valid_time(
    tmp_path: Path, fake_download, outline
):  # pylint: disable=unused-argument
    """
    Stamp a forecast-based field by the day it describes, not the reference time.

    CDS carries the forecast reference time as ``time`` and the valid time
    beside it; for the forecast-based fields they differ by a day. The
    fixture makes them differ, and every field has to land on the same days.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    fake_download : dict
        Recorded requests.
    outline : geopandas.GeoDataFrame
        The outline.
    """
    rgi_file = tmp_path / "outline.gpkg"
    outline.to_file(rgi_file)
    store = mod.run("RGI2000-v7.0-C-01-12784", tmp_path / "out", rgi_file=rgi_file, start=1990, end=1990)
    ds = xr.open_zarr(store, consolidated=True)
    times = pd.DatetimeIndex(ds["time"].values)
    assert times[0] == pd.Timestamp("1990-01-01")
    # Had the forecast fields been stamped by their reference time, aligning
    # them with the analysis fields would have produced a 1989-12-31 record
    # with NaN temperature.
    assert not ds["precipitation"].isnull().all(("y", "x")).any()
    assert not ds["air_temp"].isnull().all(("pressure_level", "y", "x")).any()


def test_time_and_bounds_share_one_encoding(tmp_path: Path, fake_download, outline):  # pylint: disable=unused-argument
    """
    Write ``time`` and ``time_bounds`` on the same scale.

    Left to itself xarray encodes them independently, which CF forbids and
    which makes the bounds decode to a different epoch.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    fake_download : dict
        Recorded requests.
    outline : geopandas.GeoDataFrame
        The outline.
    """
    rgi_file = tmp_path / "outline.gpkg"
    outline.to_file(rgi_file)
    store = mod.run("RGI2000-v7.0-C-01-12784", tmp_path / "out", rgi_file=rgi_file, start=1990, end=1990)
    raw = xr.open_zarr(store, consolidated=True, decode_times=False)
    # xarray normalises the units string on write (drops the "00:00:00").
    assert raw["time"].attrs["units"].startswith("hours since 1850-01-01")
    assert raw["time"].dtype == np.int64
    assert raw["time"].values[0] == raw["time_bounds"].values[0, 0]


def test_a_bbox_outside_the_file_is_an_error():
    """
    Refuse to write an empty store when the clip selects nothing.
    """
    lat, lon = lattice_latlon()
    ds = xr.Dataset(
        {
            "t": (
                ("time", "isobaricInhPa", "y", "x"),
                np.zeros((1, 5, NY, NX), "float32"),
                {"standard_name": "air_temperature"},
            )
        },
        coords={
            "time": ("time", pd.to_datetime(["1990-01-01"]).values),
            "isobaricInhPa": ("isobaricInhPa", np.array([1000.0, 900.0, 800.0, 750.0, 500.0])),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
    )
    with pytest.raises(ValueError, match="selects no cells"):
        mod.normalize(ds, ["air_temp"], bbox=(0.0, 1.0, 0.0, 1.0))


def test_years_are_validated():
    """
    Refuse years outside the record or a reversed range.
    """
    with pytest.raises(ValueError, match="years must satisfy"):
        mod.run("RGI2000-v7.0-C-01-12784", "/nonexistent", start=1980, end=1990)
    with pytest.raises(ValueError, match="years must satisfy"):
        mod.run("RGI2000-v7.0-C-01-12784", "/nonexistent", start=1995, end=1990)


def test_cli_start_end_default_to_the_full_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    ``--start``/``--end`` narrow the years; without them every year is asked for.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Patch fixture.
    """
    calls: list[dict] = []

    def fake_run(rgi_id, output_path, **kw):  # pylint: disable=unused-argument
        """
        Record the keyword arguments instead of downloading.

        Parameters
        ----------
        rgi_id : str
            Ignored.
        output_path : Path
            Ignored.
        **kw : dict
            Recorded.

        Returns
        -------
        pathlib.Path
            A stand-in store path.
        """
        calls.append(kw)
        return tmp_path / "x.zarr"

    monkeypatch.setattr(mod, "run", fake_run)
    monkeypatch.setattr(mod, "setup_logging", lambda *_: None)
    assert mod.main(["--output-path", str(tmp_path), "RGI2000-v7.0-C-01-12784"]) == 0
    assert (calls[-1]["start"], calls[-1]["end"]) == (mod.FIRST_YEAR, mod.LAST_YEAR)
    assert (
        mod.main(["--output-path", str(tmp_path), "--start", "2005", "--end", "2005", "RGI2000-v7.0-C-01-12784"]) == 0
    )
    assert (calls[-1]["start"], calls[-1]["end"]) == (2005, 2005)
