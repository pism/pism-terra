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
Tests for the CARRA2 monthly-mean climatology.

The climatology is a 12-step periodic forcing PISM cycles, so two things have
to hold: the averaging is over a fixed reference period (not "whatever is in
the store"), and the written file carries a time axis and a grid mapping PISM
can actually read.
"""

import netCDF4
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pism_terra.glacier.climate import (
    CARRA2_CLIMATOLOGY_YEARS,
    open_carra2_zarr,
    prepare_carra2_monthly_mean,
    stamp_monthly_climatology_axis,
)
from pism_terra.workflow import compressed_encoding, stamp_grid_mapping


def carra2_store(path, first_year: int = 1986, last_year: int = 2026):
    """
    Write a CARRA2-shaped Zarr store with a value we can predict exactly.

    Every time-varying field is ``month * 100 + (year - first_year)`` times a
    per-variable scale, so the mean over any set of years is known in closed
    form.

    Parameters
    ----------
    path : pathlib.Path
        Destination Zarr store.
    first_year : int, default 1986
        First year in the store.
    last_year : int, default 2026
        One past the last year in the store.

    Returns
    -------
    pathlib.Path
        The written store.
    """
    time = xr.date_range(f"{first_year}-01-01", periods=(last_year - first_year) * 12, freq="MS")
    stamps = pd.to_datetime(time)
    base = (np.array([t.month for t in stamps]) * 100 + (np.array([t.year for t in stamps]) - first_year)).astype(
        "float64"
    )

    def field(scale):
        """
        Broadcast the predictable signal over a small spatial grid.

        Parameters
        ----------
        scale : float
            Per-variable multiplier.

        Returns
        -------
        numpy.ndarray
            Array shaped ``(time, y, x)``.
        """
        return base[:, None, None] * scale * np.ones((1, 4, 5))

    ds = xr.Dataset(
        {
            "air_temp": (("time", "y", "x"), field(1.0)),
            "air_temp_sd": (("time", "y", "x"), field(0.01)),
            "precipitation": (("time", "y", "x"), field(0.1)),
            "orography": (("y", "x"), np.arange(20.0).reshape(4, 5)),
            "time_bnds": (("time", "bnds"), np.zeros((len(time), 2))),
        },
        coords={"time": time, "y": np.linspace(0, 3000, 4), "x": np.linspace(0, 4000, 5)},
    )
    ds = ds.rio.write_crs("EPSG:3413").rio.write_grid_mapping()
    ds.to_zarr(path, mode="w", consolidated=True)
    return path


@pytest.fixture(name="climatology")
def fixture_climatology(tmp_path):
    """
    Build the climatology from a synthetic store.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.

    Returns
    -------
    xarray.Dataset
        The written climatology, reopened.
    """
    source = carra2_store(tmp_path / "carra2.zarr")
    out = prepare_carra2_monthly_mean(source, tmp_path / "carra2_monthly_mean.zarr")
    return xr.open_zarr(out, consolidated=True)


def test_averages_over_the_fixed_reference_period(climatology):
    """
    Average 1990-2019, not whatever the store happens to hold.

    Parameters
    ----------
    climatology : xarray.Dataset
        Climatology built from the synthetic store.
    """
    years = list(CARRA2_CLIMATOLOGY_YEARS)
    assert (years[0], years[-1]) == (1990, 2019)
    # Source value is month*100 + (year - 1986), so the mean is known exactly.
    offset = np.mean(np.array(years) - 1986)

    assert climatology.sizes["time"] == 12
    for month in (1, 7, 12):
        assert float(climatology["air_temp"].sel(time=month).mean()) == pytest.approx(month * 100 + offset)
    assert climatology.attrs["climatology_period"] == "1990-2019"


def test_air_temp_sd_is_averaged_not_recomputed(climatology):
    """
    Keep ``air_temp_sd`` as within-month variability.

    The PDD scheme reads it as sub-monthly spread, so it is averaged like every
    other field rather than replaced by the interannual standard deviation
    (which would be far smaller and would understate melt).

    Parameters
    ----------
    climatology : xarray.Dataset
        Climatology built from the synthetic store.
    """
    offset = np.mean(np.array(list(CARRA2_CLIMATOLOGY_YEARS)) - 1986)

    assert float(climatology["air_temp_sd"].sel(time=7).mean()) == pytest.approx((7 * 100 + offset) * 0.01)


def test_time_invariant_fields_survive(climatology):
    """
    Leave orography two-dimensional and drop the stale source bounds.

    Parameters
    ----------
    climatology : xarray.Dataset
        Climatology built from the synthetic store.
    """
    assert climatology["orography"].dims == ("y", "x")
    assert "time_bnds" not in climatology.variables


def test_month_axis_is_unambiguous(climatology):
    """
    Label the intermediate by calendar month, 1 to 12.

    The CF climatological axis is stamped when a per-glacier file is written;
    keeping the intermediate as plain month numbers means it survives the
    per-group reprojection round trip without being reinterpreted.

    Parameters
    ----------
    climatology : xarray.Dataset
        Climatology built from the synthetic store.
    """
    assert list(climatology["time"].values) == list(range(1, 13))


def test_crs_is_recoverable(tmp_path, climatology):
    """
    Keep the store self-describing after the groupby.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    climatology : xarray.Dataset
        Climatology built from the synthetic store.
    """
    _ = climatology
    assert open_carra2_zarr(tmp_path / "carra2_monthly_mean.zarr").rio.crs is not None


def test_a_period_outside_the_store_is_rejected(tmp_path):
    """
    Fail loudly rather than writing an empty climatology.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    source = carra2_store(tmp_path / "short.zarr", first_year=2020, last_year=2024)

    with pytest.raises(ValueError, match="no time steps in 1990-2019"):
        prepare_carra2_monthly_mean(source, tmp_path / "out.zarr")


def test_cached_output_is_reused(tmp_path, climatology):
    """
    Skip the work when the store is already there.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    climatology : xarray.Dataset
        Climatology built from the synthetic store.
    """
    _ = climatology
    # A missing source would raise if the function did not short-circuit.
    again = prepare_carra2_monthly_mean(tmp_path / "does-not-exist.zarr", tmp_path / "carra2_monthly_mean.zarr")

    assert again == tmp_path / "carra2_monthly_mean.zarr"


def test_climatological_axis_matches_era5(climatology):
    """
    Place the 12 steps mid-month on a 365-day year spanning exactly one year.

    Parameters
    ----------
    climatology : xarray.Dataset
        Climatology built from the synthetic store.
    """
    stamped = stamp_monthly_climatology_axis(climatology.drop_vars("spatial_ref", errors="ignore"))

    assert stamped["time"].attrs["units"] == "days since 0001-01-01"
    assert stamped["time"].attrs["calendar"] == "365_day"
    assert stamped["time"].attrs["bounds"] == "time_bounds"
    np.testing.assert_allclose(stamped["time"].values[:2], [15.5, 45.0])
    bounds = stamped["time_bounds"].values
    assert (bounds[0][0], bounds[-1][1]) == (0.0, 365.0)
    # Bounds tile the year with no gaps.
    np.testing.assert_array_equal(bounds[1:, 0], bounds[:-1, 1])


def test_a_series_that_is_not_twelve_months_is_rejected(climatology):
    """
    Refuse to stamp a climatological axis on something that is not one.

    Parameters
    ----------
    climatology : xarray.Dataset
        Climatology built from the synthetic store.
    """
    with pytest.raises(ValueError, match="needs 12 time steps"):
        stamp_monthly_climatology_axis(climatology.isel(time=slice(0, 6)))


def test_compressed_encoding_keeps_the_grid_mapping(tmp_path):
    """
    Carry ``grid_mapping`` through into the written file.

    Handing ``to_netcdf`` an encoding dict replaces a variable's encoding, so
    asking for compression the obvious way drops the pointer to the CRS
    variable and PISM can no longer find the projection.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    ds = xr.Dataset(
        {"air_temp": (("y", "x"), np.ones((3, 4), "float32"))},
        coords={"y": np.arange(3.0), "x": np.arange(4.0)},
    )
    ds = stamp_grid_mapping(ds.rio.write_crs("EPSG:3338").rio.write_grid_mapping())

    naive, safe = tmp_path / "naive.nc", tmp_path / "safe.nc"
    ds.to_netcdf(naive, encoding={"air_temp": {"zlib": True, "complevel": 2}}, engine="h5netcdf")
    ds.to_netcdf(safe, encoding=compressed_encoding(ds), engine="h5netcdf")

    with netCDF4.Dataset(naive) as nc:  # pylint: disable=no-member
        assert not hasattr(nc.variables["air_temp"], "grid_mapping")
    with netCDF4.Dataset(safe) as nc:  # pylint: disable=no-member
        assert nc.variables["air_temp"].grid_mapping == "spatial_ref"
    assert xr.open_dataset(safe, decode_coords="all").rio.crs is not None
