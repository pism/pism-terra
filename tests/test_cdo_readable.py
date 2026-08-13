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
Tests for :func:`pism_terra.workflow.make_cdo_readable`.

Covers the two things that stop CDO opening the per-region scalar files —
a leading region dimension and a string region coordinate — plus the no-op
paths for datasets that need neither.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from pism_terra.workflow import make_cdo_readable


def labelled(dim: str = "basin", labels=("CE", "CW", "GIS"), n_time: int = 4) -> xr.Dataset:
    """
    Build a dataset shaped like the postprocess output: (region, time).

    Parameters
    ----------
    dim : str, default "basin"
        Name of the region dimension.
    labels : tuple of str
        Region labels used as the coordinate.
    n_time : int, default 4
        Length of the time dimension.

    Returns
    -------
    xarray.Dataset
        Dataset with the region dimension leading, as ``expand_dims`` leaves it.
    """
    data = np.arange(len(labels) * n_time, dtype="float64").reshape(len(labels), n_time)
    return xr.Dataset(
        {"ice_mass": ((dim, "time"), data), "scalar_only": ((), np.float64(1.0))},
        coords={dim: list(labels), "time": np.arange(n_time, dtype="float64")},
    )


def test_moves_time_first_and_indexes_labels():
    """Both fixes applied: time leads, and the labels become an integer index."""
    ds = labelled()

    out = make_cdo_readable(ds, "basin")

    assert out["ice_mass"].dims == ("time", "basin")
    assert out["basin"].dtype == np.int32
    assert out["basin"].values.tolist() == [0, 1, 2]
    assert out["basin_name"].values.tolist() == ["CE", "CW", "GIS"]
    # Values follow their label, they are not reshuffled.
    np.testing.assert_array_equal(
        out.set_index(basin="basin_name")["ice_mass"].sel(basin="GIS").values,
        ds["ice_mass"].sel(basin="GIS").values,
    )
    # Variables without the region dim are left alone.
    assert out["scalar_only"].dims == ()


def test_custom_dim_and_name_var():
    """The region dimension is a parameter — glacier output uses ``RGIid``."""
    ds = labelled(dim="RGIid", labels=("RGI2000-v7.0-C-01-04374", "RGI2000-v7.0-C-01-09429"))

    out = make_cdo_readable(ds, "RGIid", name_var="rgi_id")

    assert out["ice_mass"].dims == ("time", "RGIid")
    assert out["rgi_id"].values.tolist() == list(ds["RGIid"].values)
    assert "'rgi_id'" in out["RGIid"].attrs["description"]


def test_numeric_labels_are_left_alone():
    """An already-numeric region coordinate only gets the transpose."""
    ds = labelled().assign_coords(basin=[10, 20, 30])

    out = make_cdo_readable(ds, "basin")

    assert out["ice_mass"].dims == ("time", "basin")
    assert out["basin"].values.tolist() == [10, 20, 30]
    assert "basin_name" not in out.coords


def test_no_time_dim_is_a_no_op_for_the_transpose():
    """A dataset without time still gets its labels indexed, and does not raise."""
    ds = labelled().isel(time=0, drop=True)

    out = make_cdo_readable(ds, "basin")

    assert out["ice_mass"].dims == ("basin",)
    assert out["basin_name"].values.tolist() == ["CE", "CW", "GIS"]


def test_missing_dim_is_a_no_op():
    """Calling with a dimension the dataset does not have changes nothing."""
    ds = labelled()

    out = make_cdo_readable(ds, "not_a_dim")

    assert out["ice_mass"].dims == ("time", "basin")
    assert out["basin"].values.tolist() == ["CE", "CW", "GIS"]
