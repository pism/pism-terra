"""
Tests for the KITP timeseries adjustment.

The adjustment trims spin-up, restamps the surviving years as 0..N-1 and
subtracts each cumulative variable's value at year 0, so that runs starting at
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
    select_glacier,
    trim_spinup,
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


def test_trims_spinup_and_restamps_from_year_zero():
    """
    Only the requested window survives, renumbered from year 0.
    """
    out = adjust_kitp_timeseries(monthly_run(), spinup_end_year=11, window_end_year=311)

    assert out.sizes["time"] == 301
    assert out["time"].values[0] == cftime.DatetimeNoLeap(0, 1, 1)
    assert out["time"].values[-1] == cftime.DatetimeNoLeap(300, 1, 1)
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
    The NetCDF carries the timeseries and the CSV its sea-level equivalent.

    The CSV holds ``sle`` alone, indexed by the position in the kept window,
    which is the model year since the window is stamped from year 0.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    out = adjust_kitp_timeseries(monthly_run())
    netcdf = tmp_path / "nested" / "kitp.nc"
    csv = tmp_path / "nested" / "kitp.csv"

    write_outputs(out, netcdf, csv)

    assert netcdf.exists() and csv.exists()
    from_disk = pd.read_csv(csv, index_col=0)
    assert list(from_disk.columns) == ["sle"]
    np.testing.assert_array_equal(from_disk.index.values, np.arange(out.sizes["time"]))
    expected = -out["ice_mass_glacierized"].values / 1e12 / 361.8
    np.testing.assert_allclose(from_disk["sle"].values, expected)
    # Normalised at year 0, so the sea-level contribution starts at zero.
    assert from_disk["sle"].iloc[0] == 0.0

    with xr.open_dataset(
        netcdf, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)  # pylint: disable=no-member
    ) as back:
        np.testing.assert_allclose(back["ice_mass"].values, out["ice_mass"].values)
        assert back.sizes["time"] == out.sizes["time"]


def test_trim_spinup_keeps_everything_after_the_spinup():
    """
    An open-ended window keeps runs of different lengths in full.
    """
    short = trim_spinup(monthly_run(last_year=150), window_end_year=None)
    long_run = trim_spinup(monthly_run(last_year=400), window_end_year=None)

    # 10 spin-up years dropped from each, the rest kept.
    assert short.sizes["time"] == 139
    assert long_run.sizes["time"] == 389
    for out in (short, long_run):
        assert out["time"].values[0] == cftime.DatetimeNoLeap(0, 1, 1)


def basin_run(names: tuple[str, ...] = ("NW", "GIS", "SE")) -> xr.Dataset:
    """
    Build a basin-wise timeseries: one :func:`monthly_run` per glacier.

    Each glacier's values are scaled by its position plus one, so the
    selected glacier is recognisable from the numbers.

    Parameters
    ----------
    names : tuple of str, default ("NW", "GIS", "SE")
        The ``glacier_id_name`` of each glacier, in order.

    Returns
    -------
    xarray.Dataset
        Dataset with a ``glacier_id`` axis labelled by ``glacier_id_name``,
        as the basin-wise scalar post-processing writes it.
    """
    run = monthly_run().drop_vars("time_bounds")
    ds = xr.concat([run * (k + 1) for k in range(len(names))], dim="glacier_id")
    return ds.assign_coords(glacier_id=np.arange(len(names)), glacier_id_name=("glacier_id", list(names)))


@pytest.mark.parametrize(("name", "factor"), [(None, 2), ("GIS", 2), ("SE", 3)])
def test_select_glacier_picks_by_name_with_gis_by_default(name, factor):
    """
    Keep the named glacier, or ``GIS`` when none is named, and drop the axis.

    Parameters
    ----------
    name : str or None
        Requested ``glacier_id_name``.
    factor : int
        Scale of the glacier expected to be kept.
    """
    ds = basin_run()

    out = select_glacier(ds, name)

    assert "glacier_id" not in out.dims
    np.testing.assert_allclose(out["ice_mass"].values, monthly_run()["ice_mass"].values * factor)


def test_select_glacier_rejects_an_unknown_name():
    """
    A misspelled name fails up front, listing the names that exist.
    """
    with pytest.raises(ValueError, match=r"no glacier_id_name 'XX' in the input; available: \['NW', 'GIS', 'SE'\]"):
        select_glacier(basin_run(), "XX")


def test_select_glacier_without_the_axis_returns_the_input(caplog):
    """
    A timeseries without ``glacier_id`` passes through; a named glacier is reported as ignored.

    Parameters
    ----------
    caplog : pytest.LogCaptureFixture
        Captures the warning.
    """
    ds = monthly_run()

    assert select_glacier(ds) is ds
    assert select_glacier(ds, "GIS") is ds
    assert "ignoring glacier_id_name 'GIS'" in caplog.text


def test_write_outputs_writes_the_selected_glacier_to_the_csv(tmp_path):
    """
    The CSV holds the chosen glacier and the NetCDF keeps all of them.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    out = trim_spinup(basin_run())
    netcdf, csv = tmp_path / "kitp.nc", tmp_path / "kitp.csv"

    write_outputs(out, netcdf, csv, glacier_id_name="SE")

    expected = -out["ice_mass_glacierized"].isel(glacier_id=2).values / 1e12 / 361.8
    np.testing.assert_allclose(pd.read_csv(csv)["sle"].values, expected)
    with xr.open_dataset(
        netcdf, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)  # pylint: disable=no-member
    ) as back:
        assert back.sizes["glacier_id"] == 3
