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
Tests for :mod:`pism_terra.postprocess_spatial`.

Reuses the synthetic PISM-like Greenland datasets and the real Mouginot
outlines from ``test_postprocess_scalar``.

Covers:
- ``_bbox_slices`` spanning exactly the ``True`` extent, empty-mask error.
- ``extract_basin`` against the ``rio.clip`` oracle for both crop modes,
  integer dtype preservation, and 3D (z-carrying) variables.
- ``process_file_spatial`` end-to-end for all three write methods, their
  cross-method equivalence, and the ``total`` union file.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # pylint: disable=unused-import
import xarray as xr
from test_postprocess_scalar import synthetic_greenland

from pism_terra.postprocess_scalar import basin_masks, resolve_column
from pism_terra.postprocess_spatial import (
    _bbox_slices,
    _write_zarr,
    extract_basin,
    process_file_spatial,
)


@pytest.fixture(name="outlinefile")
def fixture_outlinefile() -> Path:
    """
    Path to the Mouginot basin outlines shipped with the package.

    Returns
    -------
    Path
        Location of ``mouginot_basins_w_shelves.gpkg``.
    """
    return Path(str(files("pism_terra.data").joinpath("mouginot_basins_w_shelves.gpkg")))


@pytest.fixture(name="basins")
def fixture_basins(outlinefile: Path) -> gpd.GeoDataFrame:
    """
    Mouginot basin outlines.

    Parameters
    ----------
    outlinefile : Path
        Path to the outline GeoPackage.

    Returns
    -------
    geopandas.GeoDataFrame
        The seven Greenland drainage basins, in file order.
    """
    return gpd.read_file(outlinefile)


def test_bbox_slices_span_the_true_extent():
    """
    The window covers every True cell and nothing beyond the extent.
    """
    mask = np.zeros((10, 12), dtype=bool)
    mask[3, 4] = mask[7, 9] = True
    da = xr.DataArray(mask, dims=("y", "x")).chunk({"y": 4, "x": 5})

    y_slice, x_slice = _bbox_slices(da)

    assert (y_slice, x_slice) == (slice(3, 8), slice(4, 10))


def test_bbox_slices_empty_mask_raises():
    """
    An empty mask is an error the caller turns into a skip.
    """
    da = xr.DataArray(np.zeros((4, 4), dtype=bool), dims=("y", "x"))

    with pytest.raises(ValueError, match="no cells"):
        _bbox_slices(da)


def test_extract_basin_full_grid_matches_rio_clip(basins):
    """
    ``crop=False`` equals ``rio.clip(drop=False)`` cell-for-cell.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = synthetic_greenland(basins, n_time=2, time_chunks={"thk": 1, "ice_mass": 1, "mask": 1})
    masks = basin_masks(ds, basins)

    for (name, mask), (_, row) in zip(masks, basins.iterrows()):
        sub = extract_basin(ds[["thk"]], mask, crop=False)
        oracle = ds[["thk"]].rio.clip([row.geometry], basins.crs, drop=False)
        np.testing.assert_array_equal(sub["thk"].values, oracle["thk"].values, err_msg=name)


def test_extract_basin_cropped_matches_rio_clip_on_shared_window(basins):
    """
    ``crop=True`` equals ``rio.clip(drop=True)`` on the aligned coords.

    The mask-derived window is a subset of rio's polygon-bounds window and
    keeps every inside cell, so selecting the cropped coords out of the rio
    result must reproduce our output exactly.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = synthetic_greenland(basins, n_time=2, time_chunks={"thk": 1, "ice_mass": 1, "mask": 1})
    masks = basin_masks(ds, basins)

    for (name, mask), (_, row) in zip(masks, basins.iterrows()):
        sub = extract_basin(ds[["thk"]], mask, crop=True)
        oracle = ds[["thk"]].rio.clip([row.geometry], basins.crs, drop=True)
        # Our window is a subset of rio's bounds-based window ...
        assert set(np.asarray(sub["y"])) <= set(np.asarray(oracle["y"])), name
        assert set(np.asarray(sub["x"])) <= set(np.asarray(oracle["x"])), name
        # ... every inside (non-NaN) cell is retained ...
        assert int(oracle["thk"].notnull().sum()) == int(sub["thk"].notnull().sum()), name
        # ... and on the shared window the values agree.
        np.testing.assert_array_equal(sub["thk"].values, oracle["thk"].sel(y=sub["y"], x=sub["x"]).values, err_msg=name)


def test_extract_basin_preserves_integer_dtypes(basins):
    """
    Integer variables are cropped but never NaN-masked or promoted.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = synthetic_greenland(basins, n_time=2, time_chunks={"thk": 1, "mask": 1})
    name, mask = basin_masks(ds, basins)[0]

    sub = extract_basin(ds, mask, crop=True)

    assert sub["mask"].dtype == ds["mask"].dtype, name
    # Values inside the bbox are untouched, including outside-polygon cells.
    y_slice, x_slice = _bbox_slices(mask)
    np.testing.assert_array_equal(sub["mask"].values, ds["mask"].isel(y=y_slice, x=x_slice).values)
    # Float vars in the same dataset are NaN outside the polygon.
    cropped_mask = mask.isel(y=y_slice, x=x_slice).compute()
    assert bool(sub["thk"].isel(time=0).where(~cropped_mask).isnull().all().compute())
    # Variable order is preserved.
    assert list(sub.data_vars) == list(ds.data_vars)


def test_extract_basin_masks_3d_variables(basins):
    """
    A (time, y, x, z) variable is masked by broadcasting, dims intact.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = synthetic_greenland(basins, n_time=2, z_levels=5, time_chunks={"enthalpy": 1})
    name, mask = basin_masks(ds, basins)[0]

    sub = extract_basin(ds[["enthalpy"]], mask, crop=True)

    assert sub["enthalpy"].dims == ("time", "y", "x", "z")
    y_slice, x_slice = _bbox_slices(mask)
    cropped_mask = np.asarray(mask.isel(y=y_slice, x=x_slice))
    values = sub["enthalpy"].isel(time=1, z=3).values
    expected = ds["enthalpy"].isel(time=1, z=3, y=y_slice, x=x_slice).values
    np.testing.assert_array_equal(values[cropped_mask], expected[cropped_mask], err_msg=name)
    assert np.isnan(values[~cropped_mask]).all()


@pytest.mark.integration
@pytest.mark.parametrize("method", ["netcdf", "zarr", "shards"])
def test_process_file_spatial_end_to_end(tmp_path, basins, outlinefile, method):
    """
    One file per basin, values matching the mask oracle, for every method.

    Runs against a real (in-process) Dask cluster, which is what makes this
    opt-in rather than part of the default suite.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest per-test temporary directory.
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    outlinefile : pathlib.Path
        Path to the outline GeoPackage handed to ``process_file_spatial``.
    method : str
        Write strategy under test.
    """
    from dask.distributed import Client  # pylint: disable=import-outside-toplevel

    ds = synthetic_greenland(basins, n_time=3, z_levels=4)
    infile = tmp_path / "spatial_g20000m_test.nc"
    ds.to_netcdf(infile, engine="h5netcdf")

    with Client(processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None) as client:
        written = process_file_spatial(
            infile,
            tmp_path / "out",
            outlinefile,
            client,
            method=method,
            time_batch=2,
            scratch=tmp_path / "scratch",
        )

    names = basins[resolve_column(basins)].tolist()
    assert [p.name for p in written] == [f"spatial_{n}_g20000m_test.nc" for n in names]
    masks = dict(basin_masks(ds, basins))

    for name, path in zip(names, written):
        out = xr.open_dataset(path, decode_times=False, decode_timedelta=False)
        try:
            mask = np.asarray(masks[name])
            ys = np.flatnonzero(mask.any(axis=1))
            xs = np.flatnonzero(mask.any(axis=0))
            window = np.s_[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]
            cropped_mask = mask[window]

            # Floats: NaN outside the polygon, source values inside.
            thk = out["thk"].values
            src = ds["thk"].values[(np.s_[:],) + window]
            np.testing.assert_array_equal(thk[:, cropped_mask], src[:, cropped_mask])
            assert np.isnan(thk[:, ~cropped_mask]).all()
            # 3D variable survives with source values inside the polygon.
            np.testing.assert_array_equal(
                out["enthalpy"].values[:, cropped_mask, :],
                ds["enthalpy"].values[(np.s_[:],) + window + (np.s_[:],)][:, cropped_mask, :],
            )
            # Integers: dtype and values preserved across the whole bbox.
            assert out["mask"].dtype == ds["mask"].dtype
            np.testing.assert_array_equal(out["mask"].values, ds["mask"].values[(np.s_[:],) + window])
            # Georeference, carried non-spatial var, basin attr, unlimited time.
            assert "spatial_ref" in out.variables
            # The grid-mapping link must survive the encoding reset, or the
            # output is not machine-readably georeferenced.
            assert out.rio.crs is not None
            assert "pism_config" in out
            assert out.attrs["basin"] == name
            assert "time" in (out.encoding.get("unlimited_dims") or set())
        finally:
            out.close()


@pytest.mark.integration
def test_process_file_spatial_methods_agree(tmp_path, basins, outlinefile):
    """
    All three write methods produce value-identical files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest per-test temporary directory.
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    outlinefile : pathlib.Path
        Path to the outline GeoPackage handed to ``process_file_spatial``.
    """
    from dask.distributed import Client  # pylint: disable=import-outside-toplevel

    ds = synthetic_greenland(basins, n_time=3)
    infile = tmp_path / "spatial_agree.nc"
    ds.to_netcdf(infile, engine="h5netcdf")

    outputs = {}
    with Client(processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None) as client:
        for method in ("netcdf", "zarr", "shards"):
            outputs[method] = process_file_spatial(
                infile,
                tmp_path / method,
                outlinefile,
                client,
                method=method,
                time_batch=2,
                scratch=tmp_path / f"scratch_{method}",
            )

    for ref_path, zarr_path, shards_path in zip(outputs["netcdf"], outputs["zarr"], outputs["shards"]):
        ref = xr.open_dataset(ref_path, decode_times=False, decode_timedelta=False)
        for other_path in (zarr_path, shards_path):
            other = xr.open_dataset(other_path, decode_times=False, decode_timedelta=False)
            xr.testing.assert_allclose(ref, other)
            other.close()
        ref.close()


@pytest.mark.integration
def test_process_file_spatial_total_and_crop_off(tmp_path, basins, outlinefile):
    """
    ``total=True`` appends a full-grid GIS union file; default does not.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest per-test temporary directory.
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    outlinefile : pathlib.Path
        Path to the outline GeoPackage handed to ``process_file_spatial``.
    """
    from dask.distributed import Client  # pylint: disable=import-outside-toplevel

    ds = synthetic_greenland(basins, n_time=2)
    infile = tmp_path / "spatial_total.nc"
    ds.to_netcdf(infile, engine="h5netcdf")

    with Client(processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None) as client:
        default = process_file_spatial(infile, tmp_path / "d", outlinefile, client)
        with_total = process_file_spatial(infile, tmp_path / "t", outlinefile, client, total=True)

    assert len(default) == len(basins)
    assert len(with_total) == len(basins) + 1
    assert with_total[-1].name == "spatial_GIS_total.nc"

    gis = xr.open_dataset(with_total[-1], decode_times=False, decode_timedelta=False)
    try:
        # Never cropped: the union file keeps the full input grid.
        assert gis.sizes["y"] == ds.sizes["y"] and gis.sizes["x"] == ds.sizes["x"]
        union = np.zeros((ds.sizes["y"], ds.sizes["x"]), dtype=bool)
        for _, mask in basin_masks(ds, basins):
            union |= np.asarray(mask)
        assert bool(np.isnan(gis["thk"].values[:, ~union]).all())
    finally:
        gis.close()


def test_process_file_spatial_rejects_unknown_vars_and_method(tmp_path, basins, outlinefile):
    """
    Unknown ``variables`` and ``method`` raise with helpful messages.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest per-test temporary directory.
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    outlinefile : pathlib.Path
        Path to the outline GeoPackage handed to ``process_file_spatial``.
    """
    ds = synthetic_greenland(basins, n_time=1)
    infile = tmp_path / "spatial_bad.nc"
    ds.to_netcdf(infile, engine="h5netcdf")

    with pytest.raises(ValueError, match="unknown method"):
        process_file_spatial(infile, tmp_path, outlinefile, client=None, method="hdf")
    with pytest.raises(ValueError, match=r"unknown variables \['nope'\].*available"):
        process_file_spatial(infile, tmp_path, outlinefile, client=None, variables=["nope"])


@pytest.mark.integration
def test_process_file_spatial_vars_filter(tmp_path, basins, outlinefile):
    """
    ``variables`` restricts the output to the requested fields.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest per-test temporary directory.
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    outlinefile : pathlib.Path
        Path to the outline GeoPackage handed to ``process_file_spatial``.
    """
    from dask.distributed import Client  # pylint: disable=import-outside-toplevel

    ds = synthetic_greenland(basins, n_time=2)
    infile = tmp_path / "spatial_vars.nc"
    ds.to_netcdf(infile, engine="h5netcdf")

    with Client(processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None) as client:
        written = process_file_spatial(infile, tmp_path / "v", outlinefile, client, variables=["thk"])

    out = xr.open_dataset(written[0], decode_times=False, decode_timedelta=False)
    try:
        # Requested var plus the carried non-spatial pism_config; nothing else.
        assert set(out.data_vars) == {"thk", "pism_config"}
    finally:
        out.close()


def test_write_zarr_handles_irregular_chunks(tmp_path, basins):
    """
    The Zarr path survives the irregular chunks a bbox crop produces.

    Slicing chunk boundaries can leave a first chunk smaller than a later
    one — e.g. ``(23, 32)`` along x — which ``to_zarr`` rejects outright
    ("Final chunk of Zarr array must be the same size or smaller than the
    first"). ``_write_zarr`` must rechunk such dimensions instead of dying;
    this reproduced the exact failure the 2 GB benchmark hit.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest per-test temporary directory.
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = synthetic_greenland(basins, n_time=2).chunk({"time": 1, "y": 32, "x": 32})
    # A crop that starts mid-chunk: first x chunk becomes 23 < 32 == later chunks.
    sub = ds[["thk", "ice_mass"]].isel(x=slice(9, None), y=slice(5, None))
    assert sub["thk"].chunks[2][0] < sub["thk"].chunks[2][1]  # premise: irregular

    path = tmp_path / "irregular.nc"
    _write_zarr(sub, path, tmp_path)

    out = xr.open_dataset(path, decode_times=False, decode_timedelta=False)
    try:
        np.testing.assert_array_equal(out["thk"].values, sub["thk"].values)
        np.testing.assert_array_equal(out["ice_mass"].values, sub["ice_mass"].values)
    finally:
        out.close()
