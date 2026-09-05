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
Tests for :mod:`pism_terra.processing`.

Covers:
- Hand-computable integrals for m^2/day on an irregular axis, kg/s, and W -> J.
- Monthly Gt/yr with NaN gaps under both the "left" and "trapezoid" rules.
- Explicit ``days_in_month`` widths, which include the final sample that ``diff`` drops.
- Unit detection: quantified input, ``attrs["units"]`` input, and rejection of non-rates.
- Independence from the time coordinate's datetime64 resolution.
- Identifier extraction in ``preprocess_netcdf`` for the glacier, ice sheet and
  GCM-forced output naming conventions.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pint_xarray  # pylint: disable=unused-import
import pytest
import xarray as xr

from pism_terra.processing import decode_pism_config, integrate_rate, preprocess_netcdf

# pint defines year as the Julian year; conversions out of Gt/yr use this.
JULIAN_YEAR_DAYS = 365.25


def make_rate(values, times, units):
    """
    Build a pint-quantified rate DataArray.

    Parameters
    ----------
    values : array_like
        Rate values.
    times : array_like
        Datetime-like coordinate labels.
    units : str
        Units to quantify with, e.g. ``"Gt/year"``.

    Returns
    -------
    xr.DataArray
        Quantified rate on a "time" dimension.
    """
    return xr.DataArray(
        np.asarray(values, dtype="float64"),
        coords={"time": pd.to_datetime(times)},
        dims="time",
        name="MB",
    ).pint.quantify(units)


@pytest.fixture(name="monthly_mb")
def fixture_monthly_mb():
    """
    Monthly 100 Gt/yr series spanning 1840-01 to 2024-06 with a two-month NaN gap.

    Returns
    -------
    xr.DataArray
        Quantified rate matching the shape of the Mankoff mass-balance record.
    """
    time = pd.date_range("1840-01-01", "2024-06-01", freq="MS")
    values = np.full(len(time), 100.0)
    values[1:3] = np.nan
    return make_rate(values, time, "Gt/year")


def test_irregular_spacing_area_rate():
    """Check m^2/day over 1-, 73- and 1-day gaps accumulates exactly, leap year included."""
    # 2020-01-02 -> 2020-03-15 is 30 + 29 + 14 = 73 days (2020 is a leap year).
    times = ["2020-01-01", "2020-01-02", "2020-03-15", "2020-03-16"]
    rate = make_rate([1.0, 1.0, 1.0, 1.0], times, "m^2 day^-1")

    result = integrate_rate(rate)

    assert str(result.pint.units) == "meter ** 2"
    np.testing.assert_allclose(result.pint.magnitude, [0.0, 1.0, 74.0, 75.0])


def test_mass_flux_seconds():
    """Check kg/s over one day gives 2 * 86400 kg."""
    rate = make_rate([2.0, 2.0], ["2020-01-01", "2020-01-02"], "kg/s")

    result = integrate_rate(rate)

    assert str(result.pint.units) == "kilogram"
    np.testing.assert_allclose(result.pint.magnitude, [0.0, 172800.0])


def test_derived_unit_watts_to_joules():
    """Check W integrates to J, where no explicit time unit appears in the unit string."""
    rate = make_rate([1000.0, 1000.0], ["2020-01-01", "2020-01-02"], "W")

    result = integrate_rate(rate, to="J")

    assert str(result.pint.units) == "joule"
    np.testing.assert_allclose(result.pint.magnitude, [0.0, 8.64e7])


def test_watts_without_target_unit_does_not_reduce():
    """Check W leaves an uncancelled ns * W when no target unit is given."""
    rate = make_rate([1000.0, 1000.0], ["2020-01-01", "2020-01-02"], "W")

    result = integrate_rate(rate)

    assert result.pint.units.dimensionality == {"[length]": 2, "[mass]": 1, "[time]": -2}
    np.testing.assert_allclose(result.pint.to("J").pint.magnitude, [0.0, 8.64e7])


def test_monthly_left_rule(monthly_mb):
    """
    Check the left rule zeroes the two intervals starting at NaN samples.

    Parameters
    ----------
    monthly_mb : xr.DataArray
        Monthly Gt/yr fixture with a two-month gap.
    """
    days = np.diff(monthly_mb["time"].values) / np.timedelta64(1, "D")
    # increments[i] = y[i] * dt[i]; y[1] and y[2] are NaN.
    expected = 100.0 * (days.sum() - days[1] - days[2]) / JULIAN_YEAR_DAYS

    result = integrate_rate(monthly_mb, method="left")

    assert result.sizes["time"] == monthly_mb.sizes["time"]
    assert result.pint.magnitude[0] == 0.0
    assert result.pint.magnitude[-1] == pytest.approx(expected)
    assert result.pint.magnitude[-1] == pytest.approx(18424.9, abs=0.1)


def test_monthly_trapezoid_rule(monthly_mb):
    """
    Check the trapezoid rule zeroes three intervals, since NaN poisons both neighbours.

    Parameters
    ----------
    monthly_mb : xr.DataArray
        Monthly Gt/yr fixture with a two-month gap.
    """
    days = np.diff(monthly_mb["time"].values) / np.timedelta64(1, "D")
    # increments[i] = 0.5 * (y[i] + y[i+1]) * dt[i]; NaN at i=1,2 kills increments 0,1,2.
    expected = 100.0 * (days.sum() - days[0] - days[1] - days[2]) / JULIAN_YEAR_DAYS

    result = integrate_rate(monthly_mb, method="trapezoid")

    assert result.pint.magnitude[-1] == pytest.approx(expected)
    assert result.pint.magnitude[-1] == pytest.approx(18416.4, abs=0.1)


def test_days_in_month_widths_include_final_sample(monthly_mb):
    """
    Check explicit ``days_in_month`` widths give n increments, not n - 1.

    Parameters
    ----------
    monthly_mb : xr.DataArray
        Monthly Gt/yr fixture with a two-month gap.
    """
    widths = monthly_mb["time"].dt.days_in_month
    lengths = widths.values.astype("float64")
    expected = 100.0 * (lengths.sum() - lengths[1] - lengths[2]) / JULIAN_YEAR_DAYS

    result = integrate_rate(monthly_mb, widths=widths)

    assert result.pint.magnitude[-1] == pytest.approx(expected)
    assert result.pint.magnitude[-1] == pytest.approx(18433.1, abs=0.1)
    # The last month contributes here but is dropped by the diff-based rules.
    assert result.pint.magnitude[-1] > integrate_rate(monthly_mb, method="left").pint.magnitude[-1]


def test_days_in_month_is_leap_aware():
    """Check February 1840 is weighted with 29 days, not 28."""
    time = pd.date_range("1840-02-01", periods=1, freq="MS")
    rate = make_rate([JULIAN_YEAR_DAYS], time, "Gt/year")

    result = integrate_rate(rate, widths=rate["time"].dt.days_in_month)

    assert result.pint.magnitude[0] == pytest.approx(29.0)


def test_quantify_from_attrs():
    """Check an unquantified array picks its units up from ``attrs``."""
    rate = xr.DataArray(
        [100.0, 100.0],
        coords={"time": pd.to_datetime(["2020-01-01", "2021-01-01"])},
        dims="time",
        attrs={"units": "Gt/year"},
    )

    result = integrate_rate(rate)

    # 2020 is a leap year: 366 days against pint's 365.25-day year.
    assert result.pint.magnitude[-1] == pytest.approx(100.0 * 366.0 / JULIAN_YEAR_DAYS)


def test_non_rate_raises():
    """Check a unit without a [time] dimension is rejected."""
    da = make_rate([1.0, 1.0], ["2020-01-01", "2020-01-02"], "Gt")

    with pytest.raises(ValueError, match=r"no \[time\] dimension"):
        integrate_rate(da)


def test_unknown_method_raises():
    """Check an unrecognized quadrature rule is rejected."""
    da = make_rate([1.0, 1.0], ["2020-01-01", "2020-01-02"], "Gt/year")

    with pytest.raises(ValueError, match="unknown method"):
        integrate_rate(da, method="simpson")


@pytest.mark.parametrize("resolution", ["ns", "us", "s"])
def test_independent_of_datetime_resolution(resolution):
    """
    Check the result does not depend on the time coordinate's datetime64 unit.

    Parameters
    ----------
    resolution : str
        NumPy datetime64 resolution code applied to the time coordinate.
    """
    times = pd.to_datetime(["2020-01-01", "2020-01-02"]).values.astype(f"datetime64[{resolution}]")
    rate = xr.DataArray([2.0, 2.0], coords={"time": times}, dims="time").pint.quantify("kg/s")

    result = integrate_rate(rate)

    np.testing.assert_allclose(result.pint.magnitude, [0.0, 172800.0])


def test_skipna_false_propagates_nan(monthly_mb):
    """
    Check NaN gaps propagate into the cumulative series when `skipna` is False.

    Parameters
    ----------
    monthly_mb : xr.DataArray
        Monthly Gt/yr fixture with a two-month gap.
    """
    result = integrate_rate(monthly_mb, skipna=False)

    assert np.isnan(result.pint.magnitude[-1])


@pytest.fixture(name="ensemble_mb")
def fixture_ensemble_mb():
    """
    Monthly Gt/yr rate on (time, uq_id, exp_id), one distinct constant per ensemble member.

    Returns
    -------
    xr.DataArray
        Quantified rate shaped like a scalar PISM ensemble.
    """
    time = pd.date_range("2000-01-01", periods=13, freq="MS")
    uq_id = np.arange(10)
    exp_id = ["c01"]
    values = np.empty((len(time), len(uq_id), len(exp_id)), dtype="float64")
    for i in range(len(uq_id)):
        values[:, i, 0] = 100.0 * (i + 1)
    return xr.DataArray(
        values,
        coords={"time": time, "uq_id": uq_id, "exp_id": exp_id},
        dims=("time", "uq_id", "exp_id"),
        name="tendency_of_ice_mass_due_to_surface_mass_flux",
    ).pint.quantify("Gt/year")


def test_nd_preserves_shape_and_coords(ensemble_mb):
    """
    Check an (time, uq_id, exp_id) input comes back with the same shape, dims and coords.

    Parameters
    ----------
    ensemble_mb : xr.DataArray
        Ensemble rate fixture.
    """
    result = integrate_rate(ensemble_mb)

    assert result.dims == ensemble_mb.dims
    assert result.shape == ensemble_mb.shape
    np.testing.assert_array_equal(result["uq_id"].values, ensemble_mb["uq_id"].values)
    np.testing.assert_array_equal(result["exp_id"].values, ensemble_mb["exp_id"].values)
    np.testing.assert_array_equal(result["time"].values, ensemble_mb["time"].values)
    assert result.name == "cumulative_tendency_of_ice_mass_due_to_surface_mass_flux"


@pytest.mark.parametrize("method", ["left", "trapezoid"])
def test_nd_matches_per_slice_1d(ensemble_mb, method):
    """
    Check every (uq_id, exp_id) slice equals the 1-D integral of that slice alone.

    Parameters
    ----------
    ensemble_mb : xr.DataArray
        Ensemble rate fixture.
    method : str
        Quadrature rule under test.
    """
    result = integrate_rate(ensemble_mb, method=method)

    for i in range(ensemble_mb.sizes["uq_id"]):
        for j in range(ensemble_mb.sizes["exp_id"]):
            expected = integrate_rate(ensemble_mb.isel(uq_id=i, exp_id=j), method=method)
            np.testing.assert_allclose(
                result.isel(uq_id=i, exp_id=j).pint.magnitude,
                expected.pint.magnitude,
            )


def test_nd_members_are_independent(ensemble_mb):
    """
    Check each member accumulates its own constant, analytically.

    Parameters
    ----------
    ensemble_mb : xr.DataArray
        Ensemble rate fixture.
    """
    days = np.diff(ensemble_mb["time"].values) / np.timedelta64(1, "D")
    result = integrate_rate(ensemble_mb, method="left")

    for i in range(ensemble_mb.sizes["uq_id"]):
        expected = 100.0 * (i + 1) * days.sum() / JULIAN_YEAR_DAYS
        assert result.isel(uq_id=i, exp_id=0).pint.magnitude[-1] == pytest.approx(expected)


@pytest.mark.parametrize(
    "order",
    [("time", "uq_id", "exp_id"), ("uq_id", "time", "exp_id"), ("uq_id", "exp_id", "time")],
)
def test_nd_dim_position_invariant(ensemble_mb, order):
    """
    Check the integration axis is located by name, not assumed to be leading.

    Parameters
    ----------
    ensemble_mb : xr.DataArray
        Ensemble rate fixture.
    order : tuple of str
        Dimension order the input is transposed into before integrating.
    """
    reference = integrate_rate(ensemble_mb)
    result = integrate_rate(ensemble_mb.transpose(*order))

    assert result.dims == order
    np.testing.assert_allclose(
        result.transpose(*ensemble_mb.dims).pint.magnitude,
        reference.pint.magnitude,
    )


def test_spatial_time_x_y():
    """Check a spatial (time, x, y) field integrates per grid cell."""
    time = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    values = np.arange(3 * 4 * 5, dtype="float64").reshape(3, 4, 5)
    field = xr.DataArray(
        values,
        coords={"time": time, "x": np.arange(4), "y": np.arange(5)},
        dims=("time", "x", "y"),
        name="smb",
    ).pint.quantify("m^2 day^-1")

    result = integrate_rate(field, method="left")

    assert result.dims == ("time", "x", "y")
    assert str(result.pint.units) == "meter ** 2"
    # One-day spacing and the left rule make this a plain cumulative sum of the first n-1 slabs.
    expected = np.concatenate([np.zeros((1, 4, 5)), np.cumsum(values[:-1], axis=0)], axis=0)
    np.testing.assert_allclose(result.pint.magnitude, expected)


def test_nd_nan_isolated_to_its_own_member(ensemble_mb):
    """
    Check a NaN in one member does not leak into the others.

    Parameters
    ----------
    ensemble_mb : xr.DataArray
        Ensemble rate fixture.
    """
    values = ensemble_mb.pint.dequantify()
    values[3, 2, 0] = np.nan
    poisoned = values.pint.quantify()
    days = np.diff(ensemble_mb["time"].values) / np.timedelta64(1, "D")

    result = integrate_rate(poisoned, method="left")

    # Member 2 loses exactly the interval starting at index 3; its neighbours are untouched.
    assert result.isel(uq_id=2, exp_id=0).pint.magnitude[-1] == pytest.approx(
        100.0 * 3 * (days.sum() - days[3]) / JULIAN_YEAR_DAYS
    )
    for i in (1, 3):
        assert result.isel(uq_id=i, exp_id=0).pint.magnitude[-1] == pytest.approx(
            100.0 * (i + 1) * days.sum() / JULIAN_YEAR_DAYS
        )


def test_nd_widths_broadcast(ensemble_mb):
    """
    Check 1-D `widths` broadcast across the non-time dimensions.

    Parameters
    ----------
    ensemble_mb : xr.DataArray
        Ensemble rate fixture.
    """
    widths = ensemble_mb["time"].dt.days_in_month
    lengths = widths.values.astype("float64")

    result = integrate_rate(ensemble_mb, widths=widths)

    assert result.shape == ensemble_mb.shape
    for i in range(ensemble_mb.sizes["uq_id"]):
        expected = 100.0 * (i + 1) * lengths.sum() / JULIAN_YEAR_DAYS
        assert result.isel(uq_id=i, exp_id=0).pint.magnitude[-1] == pytest.approx(expected)


def test_nd_widths_must_be_1d(ensemble_mb):
    """
    Check multi-dimensional `widths` are rejected rather than silently misbroadcast.

    Parameters
    ----------
    ensemble_mb : xr.DataArray
        Ensemble rate fixture.
    """
    bad = ensemble_mb["time"].dt.days_in_month.broadcast_like(ensemble_mb)

    with pytest.raises(ValueError, match="widths must be one-dimensional"):
        integrate_rate(ensemble_mb, widths=bad)


def test_missing_dim_raises(ensemble_mb):
    """
    Check integrating along an absent dimension is an error.

    Parameters
    ----------
    ensemble_mb : xr.DataArray
        Ensemble rate fixture.
    """
    with pytest.raises(ValueError, match="not a dimension of the input"):
        integrate_rate(ensemble_mb, dim="bogus")


def test_preserves_long_name(monthly_mb):
    """
    Check the ``long_name`` attribute and variable name are carried through.

    Parameters
    ----------
    monthly_mb : xr.DataArray
        Monthly Gt/yr fixture with a two-month gap.
    """
    monthly_mb.attrs["long_name"] = "Mass balance"

    result = integrate_rate(monthly_mb)

    assert result.name == "cumulative_MB"
    assert result.attrs["long_name"] == "Cumulative Mass balance"


# What PISM's ``Config::json()`` writes into the data of ``pism_config``: numbers
# as ``[value, units]`` pairs, flags as booleans, unset files as empty strings.
PISM_CONFIG_JSON = json.dumps(
    {
        "age.enabled": False,
        "geometry.front_retreat.prescribed.file": "",
        "grid.dx": [1200.0, "m"],
        "surface.pdd.factor_ice": [0.00879, "meter / (kelvin day)"],
        "stress_balance.model": "ssa+sia",
    }
)


def _write_run(tmp_path, name, command="", config_blob=None):
    """
    Write a minimal PISM-like output file and return its path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory to write into.
    name : str
        File name encoding the run identifiers.
    command : str, optional
        Value of the ``command`` attribute, by default "".
    config_blob : str or None, optional
        JSON text to store in the data of ``pism_config`` as a fixed-length
        character array, the way PISM does. By default the variable holds no
        JSON and only carries attributes.

    Returns
    -------
    pathlib.Path
        Path of the written file.
    """
    if config_blob is None:
        pism_config = xr.DataArray(0)
    else:
        chars = np.frombuffer(config_blob.encode().ljust(len(config_blob) + 8, b"\x00"), dtype="S1")
        pism_config = xr.DataArray(chars, dims=("cfg",))
    ds = xr.Dataset(
        {"thk": ("time", [1.0, 2.0]), "pism_config": pism_config},
        coords={"time": [0.0, 1.0]},
        attrs={"command": command},
    )
    ds["pism_config"].attrs = {"grid.dx": "1200", "grid.dx_doc": "ignored"}
    path = tmp_path / name
    ds.to_netcdf(path)
    return path


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "dh_RGI2000-v7.0-C-01-03383_id_0_uq_1_2000-01-01_2020-01-01.nc",
            {"rgi_id": "RGI2000-v7.0-C-01-03383", "uq_id": "1", "exp_id": "0"},
        ),
        (
            "spatial_GIS_g1200m_id_HIRHAM5-ERA5_YMM_1990_2019_uq_0_0001-01-01_0002-01-01.nc",
            {"uq_id": "0", "exp_id": "HIRHAM5-ERA5_YMM_1990_2019"},
        ),
        (
            "basin_g1200m_id_gcm_CESM2_exp_pdSST-futArcSIC_pdSST-pdSIC_0001-01-01_0301-01-01.nc",
            {"gcm_id": "CESM2", "exp_id": "gcm"},
        ),
    ],
)
def test_preprocess_ids_from_filename(tmp_path, name, expected):
    """
    Check the identifiers extracted from the three output-naming conventions.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    name : str
        Output file name to parse.
    expected : dict
        Identifier dimensions and values the file name encodes. Dimensions absent
        from it must not be added.
    """
    path = _write_run(tmp_path, name)

    ds = preprocess_netcdf(xr.open_dataset(path), process_config=False)

    assert {d: str(ds[d].values[0]) for d in ("rgi_id", "gcm_id", "uq_id", "exp_id") if d in ds.dims} == expected


def test_preprocess_rgi_id_falls_back_to_command(tmp_path):
    """
    Check the RGI identifier is taken from ``command`` when the file name lacks it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    path = _write_run(
        tmp_path,
        "spatial_g100m_id_0_uq_2_2000-01-01_2020-01-01.nc",
        command="pism -atmosphere.given.file /in/RGI2000-v7.0-C-01-03383/carra2.nc",
    )

    ds = preprocess_netcdf(xr.open_dataset(path), process_config=False)

    assert ds["rgi_id"].values.tolist() == ["RGI2000-v7.0-C-01-03383"]


def test_preprocess_config_matches_id_dims(tmp_path):
    """
    Check ``pism_config`` is stored over exactly the identifier dimensions added.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    path = _write_run(tmp_path, "dh_RGI2000-v7.0-C-01-03383_id_0_uq_1_2000-01-01_2020-01-01.nc")

    ds = preprocess_netcdf(xr.open_dataset(path))

    assert ds["pism_config"].dims == ("rgi_id", "uq_id", "exp_id")
    config = json.loads(ds["pism_config"].values.reshape(-1)[0])
    # ``_doc`` and friends are dropped; the retreat file is defaulted in.
    assert config == {"geometry.front_retreat.prescribed.file": "false", "grid.dx": "1200"}


def test_preprocess_unmatched_exp_regexp_raises(tmp_path):
    """
    Check an experiment pattern that matches nothing is reported, not asserted.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    path = _write_run(tmp_path, "spatial_no_identifiers_here.nc")

    with pytest.raises(ValueError, match="does not match"):
        preprocess_netcdf(xr.open_dataset(path), process_config=False)


def test_preprocess_config_prefers_json_blob(tmp_path):
    """
    Check the JSON blob in the data of ``pism_config`` wins over the attributes.

    The blob is normalised to the attribute convention: ``[value, units]`` pairs are
    unwrapped, flags become ``"true"``/``"false"`` and an empty retreat file is
    recorded as ``"false"``. The ``cfg`` character dimension must not survive.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    path = _write_run(
        tmp_path, "dh_RGI2000-v7.0-C-01-03383_id_0_uq_1_2000-01-01_2020-01-01.nc", config_blob=PISM_CONFIG_JSON
    )

    ds = preprocess_netcdf(xr.open_dataset(path))

    assert "cfg" not in ds.dims
    assert ds["pism_config"].dims == ("rgi_id", "uq_id", "exp_id")
    config = json.loads(ds["pism_config"].values.reshape(-1)[0])
    assert config == {
        "age.enabled": "false",
        "geometry.front_retreat.prescribed.file": "false",
        "grid.dx": 1200.0,
        "stress_balance.model": "ssa+sia",
        "surface.pdd.factor_ice": 0.00879,
    }


def test_preprocess_config_blob_survives_dask_and_raw_chars(tmp_path):
    """
    Check the blob is decoded from a dask-backed and from an undecoded character array.

    ``open_mfdataset`` hands ``preprocess`` dask-backed variables, and with
    ``decode_cf=False`` the ``|S1`` array is not collapsed into one byte string.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    path = _write_run(tmp_path, "spatial_g100m_id_0_uq_2_2000-01-01_2020-01-01.nc", config_blob=PISM_CONFIG_JSON)
    expected = {
        "age.enabled": "false",
        "geometry.front_retreat.prescribed.file": "",
        "grid.dx": 1200.0,
        "stress_balance.model": "ssa+sia",
        "surface.pdd.factor_ice": 0.00879,
    }

    with xr.open_dataset(path, chunks={}) as ds:
        assert decode_pism_config(ds["pism_config"]) == expected
    with xr.open_dataset(path, decode_cf=False) as ds:
        assert ds["pism_config"].dtype == np.dtype("S1") and ds["pism_config"].ndim >= 1
        assert decode_pism_config(ds["pism_config"]) == expected


def test_decode_pism_config_scalar_string_and_fallback():
    """
    Check a scalar string variable is decoded and attributes are used without a blob.

    A scalar string is the layout proposed for PISM once the per-parameter attributes
    are dropped; a variable whose data is not JSON falls back to the attributes with
    the documentation entries removed.
    """
    scalar = xr.DataArray(PISM_CONFIG_JSON, attrs={"grid.dx": "ignored when a blob is present"})
    assert decode_pism_config(scalar)["grid.dx"] == 1200.0

    attrs_only = xr.DataArray(np.int8(0), attrs={"grid.dx": "1200", "grid.dx_doc": "ignored", "grid.dx_type": "x"})
    assert decode_pism_config(attrs_only) == {"grid.dx": "1200"}

    not_json = xr.DataArray(np.array(b"garbage", dtype="S7"), attrs={"grid.dx": "1200"})
    assert decode_pism_config(not_json) == {"grid.dx": "1200"}
