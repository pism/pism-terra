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
Tests for band-wise Zarr writing.

The CARRA2 store keeps the whole record in one chunk along time while the
yearly batches it is built from are chunked per time step, so the store has
to be written a slab of rows at a time to keep the rechunk in memory.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pism_terra.glacier.climate import write_zarr_in_bands


def dataset(n_time: int = 6, ny: int = 10, nx: int = 7) -> xr.Dataset:
    """
    Build a small dataset chunked per time step, as the batches are.

    Parameters
    ----------
    n_time : int, default 6
        Time steps.
    ny : int, default 10
        Rows.
    nx : int, default 7
        Columns.

    Returns
    -------
    xarray.Dataset
        Two fields on ``(time, y, x)``, a static field on ``(y, x)``, time
        bounds and a scalar grid mapping.
    """
    rng = np.random.default_rng(0)
    time = pd.date_range("2000-01-01", periods=n_time, freq="MS")
    ds = xr.Dataset(
        {
            "a": (("time", "y", "x"), rng.random((n_time, ny, nx), dtype="float32")),
            "b": (("time", "y", "x"), rng.random((n_time, ny, nx), dtype="float32")),
            "orography": (("y", "x"), rng.random((ny, nx))),
            "time_bnds": (("time", "bnds"), np.stack([time, time + pd.Timedelta(days=1)], axis=1)),
            "crs": ((), np.int32(0)),
        },
        coords={"time": time, "y": np.arange(ny) * 10.0, "x": np.arange(nx) * 10.0},
    )
    return ds.chunk({"time": 1, "y": -1, "x": -1})


def test_bands_reproduce_the_dataset(tmp_path):
    """
    Slab writes give the same store as a single write, with the requested chunks.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    ds = dataset().chunk({"time": -1, "y": 4, "x": 4})
    store = write_zarr_in_bands(ds, tmp_path / "out.zarr", dim="y", band_size=4)
    out = xr.open_zarr(store)
    assert out["a"].encoding["chunks"] == (6, 4, 4)
    xr.testing.assert_identical(out.load(), ds.load())


def test_band_size_must_align_with_chunks(tmp_path):
    """
    A band that straddles store chunks is refused up front.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    ds = dataset().chunk({"time": -1, "y": 4, "x": 4})
    with pytest.raises(ValueError, match="multiple"):
        write_zarr_in_bands(ds, tmp_path / "out.zarr", dim="y", band_size=6)


def test_encoding_applies_to_static_variables(tmp_path):
    """
    Encoding passed for the eagerly written variables reaches the store.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    ds = dataset().chunk({"time": -1, "y": 5, "x": 7})
    encoding = {"time": {"dtype": "int64", "units": "hours since 1850-01-01 00:00:00"}}
    store = write_zarr_in_bands(ds, tmp_path / "out.zarr", encoding=encoding, dim="y", band_size=5)
    out = xr.open_zarr(store)
    assert out["time"].encoding["units"].startswith("hours since 1850-01-01")
    assert out["time"].encoding["dtype"] == np.dtype("int64")
    np.testing.assert_array_equal(out["time"].values, ds["time"].values)
