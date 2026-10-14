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
Tests for :func:`pism_terra.raster.add_monthly_time_bounds`.

Monthly means stamped on the first of the month carry their own interval:
it ends on the first of the next month. Building the bounds from that keeps
the last step, which pairing each stamp with the following one cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pism_terra.raster import add_monthly_time_bounds, add_time_bounds


def monthly(start: str, periods: int) -> xr.Dataset:
    """
    Build a dataset of monthly steps stamped on the first of the month.

    Parameters
    ----------
    start : str
        First stamp.
    periods : int
        Number of months.

    Returns
    -------
    xarray.Dataset
        One variable on ``time``.
    """
    time = pd.date_range(start, periods=periods, freq="MS")
    return xr.Dataset({"air_temp": ("time", np.arange(periods, dtype=float))}, coords={"time": time})


def test_each_month_ends_on_the_first_of_the_next():
    """
    December 2024 gets the bounds 2024-12-01 to 2025-01-01, and no step is dropped.
    """
    ds = add_monthly_time_bounds(monthly("2024-01-01", 12))

    bounds = pd.DataFrame(ds["time_bounds"].values)
    assert ds.sizes["time"] == 12
    assert ds["time"].attrs["bounds"] == "time_bounds"
    assert ds["time_bounds"].dims == ("time", "nv")
    assert (bounds[0] == ds["time"].values).all()
    assert bounds.iloc[-1].tolist() == [pd.Timestamp("2024-12-01"), pd.Timestamp("2025-01-01")]
    # A leap February, and intervals that tile the year without gaps.
    assert bounds.iloc[1].tolist() == [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")]
    assert (bounds[0].iloc[1:].values == bounds[1].iloc[:-1].values).all()


def test_it_keeps_the_step_the_pairing_helper_gives_up():
    """
    Pairing each stamp with the next one has no end for the last month and drops it.
    """
    ds = monthly("2024-01-01", 12)

    assert add_time_bounds(ds).sizes["time"] == 11
    assert add_monthly_time_bounds(ds).sizes["time"] == 12
    # A single month has bounds too.
    assert add_monthly_time_bounds(monthly("2024-12-01", 1))["time_bounds"].shape == (1, 2)


@pytest.mark.parametrize("stamp", ["2024-12-15", "2024-12-01T06:00"])
def test_a_stamp_that_is_not_the_first_of_a_month_is_refused(stamp: str):
    """
    Adding a month to any other stamp would not give the end of its interval.

    Parameters
    ----------
    stamp : str
        A time stamp inside a month, or on its first day but not at midnight.
    """
    stamps = [pd.Timestamp(value) for value in ("2024-10-01", "2024-11-01", stamp)]
    ds = monthly("2024-10-01", 3).assign_coords(time=stamps)

    with pytest.raises(ValueError, match="first-of-month"):
        add_monthly_time_bounds(ds)
