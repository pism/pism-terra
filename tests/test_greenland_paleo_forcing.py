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
Tests for the Greenland paleo forcing: scalar series and monthly climatology.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import toml
import xarray as xr

from pism_terra.greenland.paleo.forcing import (
    climatology_filename,
    monthly_climatology,
    prepare_searise_series,
)
from pism_terra.ismip7.greenland.forcing import _forcing_tasks

CONFIG_DIR = Path(__file__).resolve().parents[1] / "pism_terra" / "config"


@pytest.fixture(name="searise_file")
def fixture_searise_file(tmp_path):
    """
    A SeaRISE-like file: ages in years before present, present first.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    pathlib.Path
        The file.
    """
    t_ages = np.arange(0.0, 501.0, 100.0)
    sl_ages = np.arange(0.0, 2001.0, 1000.0)
    ds = xr.Dataset(
        {
            "temp_time_series": (("oisotopestimes",), -t_ages / 100.0),
            "sealevel_time_series": (("sealeveltimes",), -sl_ages / 10.0),
        },
        coords={"oisotopestimes": t_ages, "sealeveltimes": sl_ages},
    )
    path = tmp_path / "Greenland_5km_v1.1.nc"
    ds.to_netcdf(path)
    return path


def test_series_run_forward_in_negative_years(searise_file, tmp_path):
    """
    Turn ages into negative years in increasing order, with matching values.

    Parameters
    ----------
    searise_file : pathlib.Path
        SeaRISE-like input file.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    files = prepare_searise_series(searise_file, tmp_path / "out")

    with xr.open_dataset(files["delta_T_file"], decode_times=False) as ds:
        np.testing.assert_allclose(ds["time"], [-500, -400, -300, -200, -100, 0])
        np.testing.assert_allclose(ds["delta_T"], [-5, -4, -3, -2, -1, 0])
        assert not np.signbit(ds["time"].values[-1])
        assert ds["time"].attrs["units"] == "common_years since 1-1-1"
        assert ds["time"].attrs["calendar"] == "365_day"
        assert ds["delta_T"].attrs["units"] == "kelvin"
    with xr.open_dataset(files["delta_SL_file"], decode_times=False) as ds:
        np.testing.assert_allclose(ds["time"], [-2000, -1000, 0])
        np.testing.assert_allclose(ds["delta_SL"], [-200, -100, 0])


def test_series_bounds_close_at_the_record(searise_file, tmp_path):
    """
    Each record ends its interval, and the first one is extrapolated.

    Parameters
    ----------
    searise_file : pathlib.Path
        SeaRISE-like input file.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    files = prepare_searise_series(searise_file, tmp_path / "out")

    with xr.open_dataset(files["delta_SL_file"], decode_times=False) as ds:
        assert ds["time"].attrs["bounds"] == "time_bnds"
        np.testing.assert_allclose(ds["time_bnds"], [[-3000, -2000], [-2000, -1000], [-1000, 0]])


def test_ocean_series_is_the_scaled_temperature(searise_file, tmp_path):
    """
    The ocean offsets are a fraction of the air-temperature anomaly.

    Parameters
    ----------
    searise_file : pathlib.Path
        SeaRISE-like input file.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    files = prepare_searise_series(searise_file, tmp_path / "out", ocean_scale=0.5)

    with xr.open_dataset(files["ocean_delta_T_file"], decode_times=False) as ds:
        np.testing.assert_allclose(ds["delta_T"], [-2.5, -2, -1.5, -1, -0.5, 0])


def test_series_start_drops_older_records(searise_file, tmp_path):
    """
    ``start`` trims the series to the run's time span.

    Parameters
    ----------
    searise_file : pathlib.Path
        SeaRISE-like input file.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    files = prepare_searise_series(searise_file, tmp_path / "out", start=-300)

    with xr.open_dataset(files["delta_T_file"], decode_times=False) as ds:
        np.testing.assert_allclose(ds["time"], [-300, -200, -100, 0])


def test_setup_builds_ocx_air_temperature_and_precipitation():
    """
    The shipped setup asks for the 1960-1989 OCX fields the PDD model needs.
    """
    config = toml.loads((CONFIG_DIR / "setup_greenland_paleo.toml").read_text("utf-8"))

    tasks = {task[2]: task for task in _forcing_tasks(config)}

    assert set(tasks) == {"climate", "ocean"}
    _, gcm, _, _, pathway, start_year, end_year, _, fields, _ = tasks["climate"]
    assert (gcm, pathway, start_year, end_year) == ("OCX", "historical", 1960, 1989)
    assert fields == ["tas", "pr"]
    assert config["ismip7_to_pism"]["tas"] == "air_temp"
    assert config["ismip7_to_pism"]["pr"] == "precipitation"
    assert climatology_filename("climate", gcm, start_year, end_year) == "paleo_greenland_climate_OCX_YMM_1960_1989.nc"


@pytest.mark.skipif(shutil.which("cdo") is None, reason="needs cdo")
def test_monthly_climatology_is_a_periodic_year(tmp_path):
    """
    Two years of monthly data reduce to 12 means on a 365-day axis.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    time = xr.date_range("1960-01-01", periods=24, freq="MS", calendar="noleap", use_cftime=True)
    values = np.concatenate([np.arange(12.0), np.arange(12.0) + 2.0])
    ds = xr.Dataset(
        {"air_temp": (("time", "y", "x"), values[:, None, None] * np.ones((24, 2, 3)), {"units": "kelvin"})},
        coords={"time": time, "y": [0.0, 1.0], "x": [0.0, 1.0, 2.0]},
    )
    monthly = tmp_path / "monthly.nc"
    ds.to_netcdf(monthly)

    out = monthly_climatology(monthly, tmp_path / "clim.nc")

    with xr.open_dataset(out, decode_times=False) as clim:
        assert clim.sizes["time"] == 12
        np.testing.assert_allclose(clim["air_temp"].isel(y=0, x=0), np.arange(12.0) + 1.0)
        assert clim["time"].attrs["calendar"] == "365_day"
        assert float(clim["time_bounds"][0, 0]) == 0.0
        assert float(clim["time_bounds"][-1, 1]) == 365.0


@pytest.mark.skipif(shutil.which("cdo") is None, reason="needs cdo")
def test_climatology_extrapolates_into_cells_without_data(tmp_path):
    """
    Fill cells the source left empty (zero precipitation) with their neighbour's climate.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    time = xr.date_range("1960-01-01", periods=12, freq="MS", calendar="noleap", use_cftime=True)
    air_temp = np.full((12, 2, 4), 270.0)
    precipitation = np.full((12, 2, 4), 2.0)
    # The filled part of the domain: a constant temperature and no precipitation.
    air_temp[:, :, 2:] = 260.0
    precipitation[:, :, 2:] = 0.0
    # cdo looks for neighbours on the sphere, so the grid needs its projection.
    mapping = {
        "grid_mapping_name": "polar_stereographic",
        "latitude_of_projection_origin": 90.0,
        "standard_parallel": 70.0,
        "straight_vertical_longitude_from_pole": -45.0,
        "false_easting": 0.0,
        "false_northing": 0.0,
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
    }
    x = ("x", np.arange(4) * 1000.0, {"units": "m", "standard_name": "projection_x_coordinate"})
    y = ("y", [-2000000.0, -1999000.0], {"units": "m", "standard_name": "projection_y_coordinate"})
    ds = xr.Dataset(
        {
            "air_temp": (("time", "y", "x"), air_temp, {"units": "kelvin", "grid_mapping": "mapping"}),
            "precipitation": (("time", "y", "x"), precipitation, {"units": "kg m-2 s-1", "grid_mapping": "mapping"}),
            "mapping": ((), 0, mapping),
        },
        coords={"time": time, "y": y, "x": x},
    )
    monthly = tmp_path / "monthly.nc"
    ds.to_netcdf(monthly)

    out = monthly_climatology(monthly, tmp_path / "clim.nc", extrapolate_from="precipitation")

    with xr.open_dataset(out, decode_times=False) as clim:
        np.testing.assert_allclose(clim["air_temp"], 270.0)
        np.testing.assert_allclose(clim["precipitation"], 2.0)
