"""
Tests for the KITP timeseries adjustment.

The adjustment trims spin-up, restamps the surviving years as 1..N and
subtracts each cumulative variable's value at year 1, so that runs starting at
different model times can be compared directly.
"""

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pism_terra.kitp.adjust_kitp_timeseries import (
    DEFAULT_VARIABLES,
    adjust_kitp_timeseries,
    write_outputs,
)


def monthly_run(first_year: int = 1, last_year: int = 320) -> xr.Dataset:
    """
    Build a monthly scalar timeseries shaped like PISM's output.

    Values increase by one per month so any trimming or restamping is visible
    in the numbers themselves.

    Parameters
    ----------
    first_year : int, default 1
        First model year.
    last_year : int, default 320
        Last model year (exclusive).

    Returns
    -------
    xarray.Dataset
        Dataset with ``ice_mass``, ``ice_mass_glacierized``, a flux variable
        and the ``time_bounds``/``nv`` pair PISM writes.
    """
    time = [cftime.DatetimeNoLeap(y, m, 1) for y in range(first_year, last_year) for m in range(1, 13)]
    n = len(time)
    values = np.arange(n, dtype="float64")
    ds = xr.Dataset(
        {
            "ice_mass": ("time", values),
            "ice_mass_glacierized": ("time", values * 2.0),
            "tendency_of_ice_mass": ("time", np.full(n, -7.0)),
            "time_bounds": (("time", "nv"), np.zeros((n, 2))),
        },
        coords={"time": time},
    )
    return ds


def test_trims_spinup_and_restamps_from_year_one():
    """
    Only the requested window survives, renumbered from year 1.
    """
    out = adjust_kitp_timeseries(monthly_run(), spinup_end_year=11, window_end_year=311)

    assert out.sizes["time"] == 301
    assert out["time"].values[0] == cftime.DatetimeNoLeap(1, 1, 1)
    assert out["time"].values[-1] == cftime.DatetimeNoLeap(301, 1, 1)
    # The bounds variable and its dimension cannot survive resampling.
    assert "time_bounds" not in out.variables
    assert "nv" not in out.dims


def test_normalises_only_the_requested_variables():
    """
    Cumulative variables start at zero; everything else is left alone.
    """
    raw = monthly_run()
    out = adjust_kitp_timeseries(raw, ["ice_mass"])

    assert float(out["ice_mass"].isel(time=0)) == 0.0
    # Differences are preserved — normalising is a shift, not a rescale.
    yearly = raw.sel(time=slice(cftime.DatetimeNoLeap(11, 1, 1), cftime.DatetimeNoLeap(311, 1, 1)))
    yearly = yearly["ice_mass"].resample(time="YS").mean()
    np.testing.assert_allclose(out["ice_mass"].values, yearly.values - yearly.values[0])
    # Not requested, so untouched.
    assert float(out["ice_mass_glacierized"].isel(time=0)) != 0.0
    np.testing.assert_allclose(out["tendency_of_ice_mass"].values, -7.0)


def test_default_variables_are_the_cumulative_ones():
    """
    Both mass variables are normalised when none are named.
    """
    out = adjust_kitp_timeseries(monthly_run())

    for var in DEFAULT_VARIABLES:
        assert float(out[var].isel(time=0)) == 0.0


def test_unknown_variable_is_rejected():
    """
    A misspelled variable fails up front, listing what is available.
    """
    with pytest.raises(ValueError, match=r"variables not in the input: \['nope'\]"):
        adjust_kitp_timeseries(monthly_run(), ["ice_mass", "nope"])


def test_empty_window_is_rejected():
    """
    A run shorter than the window fails rather than writing an empty file.
    """
    with pytest.raises(ValueError, match="no time steps between years"):
        adjust_kitp_timeseries(monthly_run(last_year=5))


def test_write_outputs_writes_both_forms(tmp_path):
    """
    The NetCDF and the CSV carry the same numbers.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    out = adjust_kitp_timeseries(monthly_run())
    netcdf = tmp_path / "nested" / "kitp.nc"
    csv = tmp_path / "nested" / "kitp.csv"

    df = write_outputs(out, netcdf, csv)

    assert netcdf.exists() and csv.exists()
    # No stray index column: the time coordinate is a named column.
    from_disk = pd.read_csv(csv)
    assert from_disk.columns[0] == "time"
    assert not any(col.startswith("Unnamed") for col in from_disk.columns)
    np.testing.assert_allclose(from_disk["ice_mass"].values, df["ice_mass"].values)

    with xr.open_dataset(netcdf, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)) as back:
        np.testing.assert_allclose(back["ice_mass"].values, out["ice_mass"].values)
        assert back.sizes["time"] == out.sizes["time"]
