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
Tests for the modelled winter/summer balances.

A balance year runs from one annual minimum to the next and splits at the
winter maximum, both dated per year in the USGS release. Integrating monthly
model output over exactly those intervals is what makes the seasonal
comparison meaningful, so the arithmetic and the interval bookkeeping are
pinned here.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pism_terra.glacier.usgs import integrate_rate, is_monthly, month_edges


def monthly_rate(value: float = 12.0, year: int = 2000, years: int = 1) -> xr.DataArray:
    """
    Build a monthly rate series stamped mid-month, as PISM writes it.

    Parameters
    ----------
    value : float, default 12.0
        Constant rate in Gt/yr.
    year : int, default 2000
        First calendar year.
    years : int, default 1
        Number of years.

    Returns
    -------
    xarray.DataArray
        Rate with ``month_start``/``month_end`` coordinates attached.
    """
    starts = pd.date_range(f"{year}-01-01", periods=12 * years, freq="MS")
    times = starts + (starts.shift(1, freq="MS") - starts) / 2
    rate = xr.DataArray(np.full(len(times), value), dims="time", coords={"time": times})
    month_start, month_end = month_edges(times.to_numpy())
    return rate.assign_coords(month_start=("time", month_start), month_end=("time", month_end))


@pytest.mark.parametrize(
    "start,end",
    [
        ("2000-01-01", "2001-01-01"),  # whole (leap) year
        ("2000-01-01", "2000-07-01"),  # whole months
        ("2000-01-01", "2000-01-16"),  # part of one month
        ("2000-04-18", "2000-09-22"),  # both ends mid-month, as measured dates are
    ],
)
def test_integration_is_exact_for_a_constant_rate(start, end):
    """
    Integrate a constant rate to rate times elapsed years.

    Parameters
    ----------
    start, end : str
        Interval bounds.
    """
    rate = monthly_rate(value=12.0)
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days

    got = integrate_rate(rate, pd.Timestamp(start), pd.Timestamp(end))

    assert got == pytest.approx(12.0 * days / 365.0)


def test_a_month_straddling_a_boundary_is_apportioned():
    """
    Split a month between the seasons it spans rather than snapping.

    Measured season dates fall mid-month far more often than not, so snapping
    to month boundaries would move up to half a month of melt between winter
    and summer.
    """
    rate = monthly_rate(value=12.0)
    split = pd.Timestamp("2000-06-15")

    winter = integrate_rate(rate, pd.Timestamp("2000-01-01"), split)
    summer = integrate_rate(rate, split, pd.Timestamp("2001-01-01"))
    whole = integrate_rate(rate, pd.Timestamp("2000-01-01"), pd.Timestamp("2001-01-01"))

    assert winter + summer == pytest.approx(whole)
    # 166 days of the year fall before the split.
    assert winter == pytest.approx(12.0 * 166 / 365.0)


def test_an_interval_outside_the_record_is_not_silently_truncated():
    """
    Return NaN rather than a sum over the part that happens to be covered.

    A partially covered season would otherwise plot as an anomalously small
    balance with nothing marking it as incomplete.
    """
    rate = monthly_rate(year=2000, years=1)

    assert np.isnan(integrate_rate(rate, pd.Timestamp("1999-06-01"), pd.Timestamp("2000-06-01")))
    assert np.isnan(integrate_rate(rate, pd.Timestamp("2000-06-01"), pd.Timestamp("2001-06-01")))


def test_a_varying_rate_weights_each_month_by_its_length():
    """
    Weight by days, not by month count.
    """
    rate = monthly_rate(value=0.0)
    values = np.arange(1.0, 13.0)
    rate = rate.copy(data=values)

    got = integrate_rate(rate, pd.Timestamp("2000-01-01"), pd.Timestamp("2001-01-01"))

    lengths = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
    assert got == pytest.approx(float((values * lengths).sum() / 365.0))


@pytest.mark.parametrize(
    "times,expected",
    [
        (pd.date_range("2000-01-16", periods=24, freq="MS"), True),
        (pd.date_range("2000-07-02", periods=10, freq="YS"), False),
        (pd.DatetimeIndex(["2000-01-16"]), False),
    ],
)
def test_monthly_detection(times, expected):
    """
    Tell monthly output from annual.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        Time axis to classify.
    expected : bool
        Whether it should count as monthly.
    """
    assert is_monthly(times.to_numpy()) is expected
