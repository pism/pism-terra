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

#: Height of each pressure level over the fixture block, metres, as a
#: ``(level, y, x)`` plane per level in the GRIB file's (unordered) level
#: order, so a clipped, re-sorted block can be checked cell by cell.
#: The levels of each request in the (unordered) order a GRIB file has them.
GRIB_LEVELS = {"": [500.0, 1000.0, 750.0, 900.0, 800.0], "_extra": [600.0, 950.0, 700.0, 825.0, 925.0, 875.0, 850.0]}


def level_height(level: float) -> np.ndarray:
    """
    Height of one pressure level over the fixture block.

    A standard-atmosphere height plus a plane, so every level differs and a
    clipped, re-sorted block can be checked cell by cell.

    Parameters
    ----------
    level : float
        Pressure, hPa.

    Returns
    -------
    numpy.ndarray
        ``(NY, NX)`` heights, metres.
    """
    base = 44330.8 * (1.0 - (level / 1013.25) ** 0.190263)
    return base + 10.0 * np.arange(NY)[:, None] + 1.0 * np.arange(NX)[None, :]


#: Orography of the fixture block, metres: a plane, checked cell by cell.
OROGRAPHY = 800.0 + 10.0 * np.arange(NY)[:, None] + 1.0 * np.arange(NX)[None, :]


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
        ``"temperature"``, ``"surface_temperature"``, ``"snow"``,
        ``"forecast"``, ``"geopotential"`` or ``"orography"``; a
        ``"_extra"`` suffix on a profile gives the second level group.

    Returns
    -------
    pathlib.Path
        The file.
    """
    lat, lon = lattice_latlon()
    days = [1] if kind == "orography" else [2, 1, 3, 3, 4]  # out of order, with a duplicate, to be cleaned up
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
    suffix = "_extra" if kind.endswith("_extra") else ""
    kind = kind.removesuffix("_extra")
    levels = np.array(GRIB_LEVELS[suffix])  # unordered, under the GRIB name
    if kind == "temperature":
        ds = xr.Dataset(
            {
                "t": (
                    ("time", "isobaricInhPa", "y", "x"),
                    (250.0 + rng.random((len(time), levels.size, NY, NX))).astype("float32"),
                    {"standard_name": "air_temperature", "units": "K", "GRIB_shortName": "t"},
                )
            },
            coords={**coords, "isobaricInhPa": ("isobaricInhPa", levels)},
        )
    elif kind == "geopotential":
        heights = np.stack([level_height(p) for p in levels])
        geopotential = np.broadcast_to(heights * mod.STANDARD_GRAVITY, (len(time), levels.size, NY, NX))
        ds = xr.Dataset(
            {
                "z": (
                    ("time", "isobaricInhPa", "y", "x"),
                    geopotential.astype("float32"),
                    {"standard_name": "geopotential", "units": "m**2 s**-2", "GRIB_shortName": "z"},
                )
            },
            coords={**coords, "isobaricInhPa": ("isobaricInhPa", levels)},
        )
    elif kind == "orography":
        # One analysis time from reanalysis-pan-carra, as CDS really delivers
        # it: ``time``/``valid_time`` are scalars, so the field is purely
        # (y, x), and it is labelled geopotential_height, not surface_altitude.
        ds = xr.Dataset(
            {
                "orog": (
                    ("y", "x"),
                    OROGRAPHY.astype("float32"),
                    {
                        "long_name": "Orography",
                        "units": "m",
                        "standard_name": "geopotential_height",
                        "GRIB_shortName": "orog",
                    },
                )
            },
            coords={
                "time": time.values[0],
                "valid_time": valid.values[0],
                "step": np.timedelta64(0, "ns"),
                "surface": 0.0,
                "latitude": (("y", "x"), lat),
                "longitude": (("y", "x"), lon),
            },
        )
    elif kind == "surface_temperature":
        # 2 m temperature in the GRIB naming: variable t2m, short name 2t.
        ds = xr.Dataset(
            {
                "t2m": (
                    ("time", "y", "x"),
                    (260.0 + rng.random((len(time), NY, NX))).astype("float32"),
                    {"long_name": "2 metre temperature", "units": "K", "GRIB_shortName": "2t"},
                )
            },
            coords={**coords, "heightAboveGround": 2.0},
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
    assert set(reqs) == {
        "temperature",
        "temperature_extra",
        "surface_temperature",
        "snow",
        "forecast",
        "geopotential",
        "geopotential_extra",
        "orography",
    }
    # The first five levels keep their original request, so an existing cache
    # still matches; the rest come separately, and together they are the profile.
    for variable in ("temperature", "geopotential"):
        assert reqs[f"{variable}_extra"]["level_location"] == ["950", "925", "875", "850", "825", "700", "600"]
        assert reqs[f"{variable}_extra"]["variable"] == [variable]
        levels = reqs[variable]["level_location"] + reqs[f"{variable}_extra"]["level_location"]
        assert sorted(map(int, levels), reverse=True) == list(mod.PRESSURE_LEVELS)
    oro = reqs.pop("orography")
    assert oro["product_type"] == "analysis" and oro["variable"] == ["orography"]
    assert oro["level_type"] == "single_levels" and "time_aggregation" not in oro
    assert oro["year"] == ["1990"] and oro["month"] == ["01"] and oro["day"] == ["01"] and oro["time"] == ["00:00"]
    assert mod.REQUEST_DATASETS["orography"] == "reanalysis-pan-carra"
    geo = reqs["geopotential"]
    assert geo["product_type"] == "analysis_based" and geo["variable"] == ["geopotential"]
    assert geo["level_type"] == "pressure_levels" and geo["level_location"] == ["1000", "900", "800", "750", "500"]
    t = reqs["temperature"]
    assert t["level_type"] == "pressure_levels" and t["product_type"] == "analysis_based"
    assert t["level_location"] == ["1000", "900", "800", "750", "500"]
    assert "pressure_level" not in t
    assert reqs["snow"]["product_type"] == "analysis_based" and reqs["snow"]["variable"] == ["snow_depth"]
    t2m = reqs["surface_temperature"]
    assert t2m["level_type"] == "single_levels" and t2m["product_type"] == "analysis_based"
    assert t2m["variable"] == ["2m_temperature"] and "level_location" not in t2m
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
    assert fake_download["temperature"]["dataset"] == "reanalysis-pan-carra-means"
    assert fake_download["orography"]["dataset"] == "reanalysis-pan-carra"

    # As a consumer must open it: decode_coords="all" is what lets rioxarray
    # follow the grid_mapping pointer to the stored spatial_ref.
    ds = xr.open_zarr(store, consolidated=True, decode_coords="all")
    assert set(ds.data_vars) >= {
        "air_temp",
        "air_temp_2m",
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

    # Vertical axis: both level requests stacked, bottom up, CF direction stated.
    np.testing.assert_array_equal(ds["pressure_level"].values, mod.PRESSURE_LEVELS)
    assert not ds["air_temp"].isnull().any()
    assert ds["pressure_level"].attrs["positive"] == "down" and ds["pressure_level"].attrs["units"] == "hPa"
    assert ds["air_temp"].dims == ("time", "pressure_level", "y", "x")
    assert ds["snow_depth"].dims == ("time", "y", "x")
    # The 2 m temperature is single level, apart from the profile's air_temp.
    assert ds["air_temp_2m"].dims == ("time", "y", "x") and ds["air_temp_2m"].attrs["units"] == "K"
    assert not ds["air_temp_2m"].isnull().all(("y", "x")).any()

    # Geopotential becomes the height of each level, divided by g as the ERA5
    # path does, on the same axes as the temperature; the raw field is not kept.
    assert "geopotential" not in ds
    height = ds["geopotential_height"]
    assert height.dims == ("time", "pressure_level", "y", "x")
    assert height.attrs["standard_name"] == "geopotential_height" and height.attrs["units"] == "m"
    # The fixture's two unordered level groups come out as one profile, bottom up.
    expected = np.stack([level_height(p) for p in mod.PRESSURE_LEVELS])
    expected = expected[:, np.isin(Y_TRUE, ds["y"].values)][:, :, np.isin(X_TRUE, ds["x"].values)]
    np.testing.assert_allclose(height.isel(time=0).values, expected, rtol=1e-6)
    np.testing.assert_allclose(height.isel(time=-1).values, expected, rtol=1e-6)
    assert height.encoding.get("grid_mapping") == "spatial_ref" or height.attrs.get("grid_mapping")

    # The orography comes from the other dataset at one analysis time and is
    # stored static, with the attributes prepare_carra2 gives it.
    assert ds["orography"].dims == ("y", "x")
    assert ds["orography"].attrs["standard_name"] == "surface_altitude" and ds["orography"].attrs["units"] == "m"
    expected = OROGRAPHY[np.isin(Y_TRUE, ds["y"].values)][:, np.isin(X_TRUE, ds["x"].values)]
    np.testing.assert_allclose(ds["orography"].values, expected, rtol=1e-6)

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


def test_an_unknown_pressure_level_is_an_error():
    """
    Refuse a level outside :data:`PRESSURE_LEVELS` rather than store it unchecked.
    """
    lat, lon = lattice_latlon()
    ds = xr.Dataset(
        {
            "t": (
                ("time", "isobaricInhPa", "y", "x"),
                np.zeros((1, 2, NY, NX), "float32"),
                {"standard_name": "air_temperature"},
            )
        },
        coords={
            "time": ("time", pd.to_datetime(["1990-01-01"]).values),
            "isobaricInhPa": ("isobaricInhPa", np.array([1000.0, 650.0])),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
    )
    with pytest.raises(ValueError, match="not distinct levels"):
        mod.normalize(ds, ["air_temp"])


def test_stacking_refuses_a_level_requested_twice():
    """
    Two parts that both hold a level would silently duplicate it.
    """
    part = xr.Dataset(
        {"air_temp": (("pressure_level", "y", "x"), np.zeros((2, 1, 1)))},
        coords={"pressure_level": [1000.0, 900.0], "y": [0.0], "x": [0.0]},
    )
    with pytest.raises(ValueError, match="requested twice"):
        mod._stack_levels([part, part.assign_coords(pressure_level=[900.0, 800.0])])  # pylint: disable=protected-access


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


def test_parts_are_cut_to_the_grid_they_share(caplog):
    """
    A field delivered on a larger grid (CDS's snow depth) is cut to the others' cells.

    Parameters
    ----------
    caplog : pytest.LogCaptureFixture
        Captured log records.
    """
    x = mod.GRID_ORIGIN + mod.GRID_SPACING * np.arange(4)
    y = mod.GRID_ORIGIN + mod.GRID_SPACING * np.arange(7)
    small = xr.Dataset({"air_temp_2m": (("y", "x"), np.ones((5, 4)))}, coords={"y": y[:5], "x": x})
    large = xr.Dataset({"snow_depth": (("y", "x"), np.ones((7, 4)))}, coords={"y": y, "x": x})
    common = mod._on_common_grid({"surface_temperature": small, "snow": large})  # pylint: disable=protected-access
    np.testing.assert_array_equal(common["snow"]["y"], y[:5])
    assert xr.merge(list(common.values()), join="exact").sizes == {"y": 5, "x": 4}
    assert "snow: 7 x 4 cells cut to the 5 x 4" in caplog.text
    with pytest.raises(ValueError, match="share no grid cell"):
        mod._on_common_grid(  # pylint: disable=protected-access
            {"a": small, "b": large.assign_coords(x=x + 100 * mod.GRID_SPACING)}
        )


def test_outline_outside_domain_is_measured():
    """
    Report the share of an outline's area beyond the CARRA2 domain's edge.
    """
    edge = mod.GRID_ORIGIN + (mod.GRID_SIZE - 1) * mod.GRID_SPACING + mod.GRID_SPACING / 2

    def outline(x0: float, x1: float) -> gpd.GeoDataFrame:
        """
        Build a square outline spanning x0 to x1 in CARRA2 metres.

        Parameters
        ----------
        x0, x1 : float
            Its x extent.

        Returns
        -------
        geopandas.GeoDataFrame
            The outline, in geographic coordinates like RGI.
        """
        width = x1 - x0
        return gpd.GeoDataFrame(geometry=[box(x0, 0.0, x1, width)], crs=CARRA2_CRS).to_crs("EPSG:4326")

    assert mod.outline_outside_domain(outline(edge - 30_000, edge - 10_000)) == pytest.approx(0.0, abs=1e-3)
    assert mod.outline_outside_domain(outline(edge - 10_000, edge + 10_000)) == pytest.approx(0.5, abs=0.01)
    assert mod.outline_outside_domain(outline(edge + 10_000, edge + 30_000)) == pytest.approx(1.0, abs=1e-3)


def test_an_outline_outside_the_domain_stops_before_downloading(tmp_path: Path, fake_download):
    """
    A target CARRA2 does not cover is refused before hours in the CDS queue.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    fake_download : dict
        Recorded requests; must stay empty.
    """
    edge = mod.GRID_ORIGIN + (mod.GRID_SIZE - 1) * mod.GRID_SPACING + mod.GRID_SPACING / 2
    rgi_file = tmp_path / "outside.gpkg"
    gpd.GeoDataFrame(
        {"rgi_id": ["RGI2000-v7.0-C-01-99999"]},
        geometry=[box(edge + 10_000, 0.0, edge + 20_000, 10_000)],
        crs=CARRA2_CRS,
    ).to_crs("EPSG:4326").to_file(rgi_file)
    with pytest.raises(ValueError, match="outside the CARRA2 domain"):
        mod.run("RGI2000-v7.0-C-01-99999", tmp_path / "out", rgi_file=rgi_file, start=1990, end=1990)
    assert not fake_download


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
