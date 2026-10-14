"""
Tests for the KITP analysis helpers.

The scalar post-processing has written its region dimension three different
ways over time, and the analysis has to read all of them: ``basin`` with
string labels, the current ``glacier_id`` integer index with labels in
``glacier_id_name``, and raw PISM scalar files with no region dimension at all.
"""

import numpy as np
import pytest
import xarray as xr

from pism_terra.kitp.analyze import REGION_DIM, with_region_labels


def scalar_dataset(dim: str | None = "glacier_id", labels=("GIS_CE", "GIS"), indexed: bool = True) -> xr.Dataset:
    """
    Build a per-region scalar dataset in one of the historical layouts.

    Parameters
    ----------
    dim : str or None, default "glacier_id"
        Region dimension name. ``None`` omits the region dimension entirely,
        as raw PISM scalar output does.
    labels : tuple of str
        Region labels.
    indexed : bool, default True
        When True the region coordinate is a positional integer index and the
        labels live in ``<dim>_name``, as post-processing writes it. When
        False the labels are the coordinate, as older files have them.

    Returns
    -------
    xarray.Dataset
        Dataset with an ``ice_mass`` variable over ``(time, region)``.
    """
    n_time = 3
    if dim is None:
        return xr.Dataset(
            {"ice_mass": ("time", np.arange(n_time, dtype="float64"))},
            coords={"time": np.arange(n_time, dtype="float64")},
        )

    data = np.arange(n_time * len(labels), dtype="float64").reshape(n_time, len(labels))
    ds = xr.Dataset(
        {"ice_mass": (("time", dim), data)},
        coords={"time": np.arange(n_time, dtype="float64"), dim: list(labels)},
    )
    if indexed:
        ds = ds.assign_coords({f"{dim}_name": (dim, list(labels))})
        ds = ds.assign_coords({dim: np.arange(len(labels), dtype="int32")})
    return ds


def test_indexed_labels_become_selectable_by_name():
    """
    The positional index is swapped back for the labels it stands in for.
    """
    ds = scalar_dataset()

    out = with_region_labels(ds)

    assert out[REGION_DIM].values.tolist() == ["GIS_CE", "GIS"]
    # The companion name coordinate is consumed, not left as a duplicate.
    assert "glacier_id_name" not in out.coords
    np.testing.assert_array_equal(
        out.sel({REGION_DIM: "GIS"})["ice_mass"].values,
        ds["ice_mass"].isel(glacier_id=1).values,
    )


@pytest.mark.parametrize("dim", ["basin", "RGIid"])
def test_older_dimension_names_are_renamed(dim):
    """
    Load files predating the unified post-processing.

    Parameters
    ----------
    dim : str
        Region dimension name used by the older layout.
    """
    ds = scalar_dataset(dim=dim, indexed=False)

    out = with_region_labels(ds)

    assert REGION_DIM in out.dims
    assert dim not in out.dims
    assert out[REGION_DIM].values.tolist() == ["GIS_CE", "GIS"]


def test_dataset_without_a_region_dimension_gets_one():
    """
    Raw PISM scalar output is whole-domain, so it becomes a single region.
    """
    ds = scalar_dataset(dim=None)

    out = with_region_labels(ds)

    assert out[REGION_DIM].values.tolist() == ["GIS"]
    assert out["ice_mass"].dims == (REGION_DIM, "time")
    np.testing.assert_array_equal(out.sel({REGION_DIM: "GIS"})["ice_mass"].values, ds["ice_mass"].values)


def test_default_region_name_is_configurable():
    """
    The stand-in label for a region-less dataset can be chosen.
    """
    out = with_region_labels(scalar_dataset(dim=None), default="RGI2000-v7.0-C-01-04374")

    assert out[REGION_DIM].values.tolist() == ["RGI2000-v7.0-C-01-04374"]
