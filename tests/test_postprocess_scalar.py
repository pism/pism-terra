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
Tests for :mod:`pism_terra.postprocess_scalar`.

Uses synthetic PISM-like fields on a coarse EPSG:3413 Greenland grid together
with the real basin outlines shipped in ``pism_terra/data``, so the
rasterization is exercised against the geometry actually used in production.

Covers:
- ``_dim_chunks`` reading y/x chunking off a dataset whose ``time`` chunking is
  inconsistent across variables — the case that made ``Dataset.chunksizes``
  raise and broke ``basin_masks``.
- ``basin_masks`` returning one mask per outline, in row order, chunked to match
  the dataset, and agreeing cell-for-cell with ``rio.clip(..., drop=False)``.
- ``process_file`` per-basin sums against hand-computed NumPy sums, including
  the derived ``ice_mass_glacierized``, the appended ``GIS`` total, and integer
  variables surviving the round trip as integers.
"""

from __future__ import annotations

import pickle
from importlib.resources import files
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # pylint: disable=unused-import
import xarray as xr
from pyproj import CRS as ProjCRS

from pism_terra.postprocess_scalar import (
    _dim_chunks,
    basin_masks,
    dataset_crs,
    process_file,
    resolve_column,
    resolve_outfile,
)

# Coarse enough that the whole grid is a few hundred kB, fine enough that every
# Mouginot basin still claims a healthy number of cells.
GRID_RESOLUTION = 20_000.0
CRS = "EPSG:3413"


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


def greenland_grid(basins: gpd.GeoDataFrame, resolution: float = GRID_RESOLUTION):
    """
    Build cell-center x/y axes covering the basins, PISM style.

    The axes descend in y and ascend in x, matching what PISM writes, so the
    affine transform ``rioxarray`` derives has the usual negative y stride.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Outlines whose bounds the grid must cover.
    resolution : float, default 20000.0
        Grid spacing in metres.

    Returns
    -------
    tuple of numpy.ndarray
        ``(y, x)`` cell-center coordinates.
    """
    x_min, y_min, x_max, y_max = basins.total_bounds
    x = np.arange(x_min - resolution, x_max + resolution, resolution) + resolution / 2
    y = np.arange(y_min - resolution, y_max + resolution, resolution) + resolution / 2
    return y[::-1], x


def synthetic_greenland(
    basins: gpd.GeoDataFrame,
    n_time: int = 4,
    time_chunks: dict[str, int] | None = None,
    resolution: float = GRID_RESOLUTION,
    *,
    z_levels: int | None = None,
    ice_free_thickness: float | None = None,
) -> xr.Dataset:
    """
    Synthetic PISM-like output on a coarse Greenland grid.

    Values are deterministic (seeded) and deliberately span dtypes — float32,
    float64 and an integer mask — because it is the per-dtype chunk sizing of
    ``chunks="auto"`` that produces inconsistent ``time`` chunking on real
    files.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Outlines used to size the grid.
    n_time : int, default 4
        Length of the time dimension.
    time_chunks : dict of str to int, optional
        Per-variable ``time`` chunk size. Variables named here are Dask-backed
        with that chunking; when ``None`` the dataset stays NumPy-backed.
    resolution : float, default 20000.0
        Grid spacing in metres.
    z_levels : int or None, optional
        When set, add a 3D ``enthalpy`` variable with dims
        ``(time, y, x, z)`` and this many vertical levels, as real PISM
        spatial files carry. ``None`` (default) keeps the dataset 2D-only,
        so existing tests are unaffected.
    ice_free_thickness : float or None, optional
        When set, record it as ``output.ice_free_thickness_standard`` on the
        ``pism_config`` variable, as PISM does. ``None`` (default) omits the
        parameter, which exercises the post-processing fallback.

    Returns
    -------
    xarray.Dataset
        Dataset with ``thk``, ``ice_mass``, ``mask`` and a non-spatial
        ``pism_config``, CRS and spatial dims already set.
    """
    y, x = greenland_grid(basins, resolution=resolution)
    shape = (n_time, y.size, x.size)
    rng = np.random.default_rng(42)

    data = {
        "thk": (rng.uniform(0.0, 3000.0, shape)).astype(np.float32),
        "ice_mass": (rng.uniform(0.0, 1e12, shape)).astype(np.float64),
        "mask": rng.integers(0, 4, shape).astype(np.int32),
    }

    ds = xr.Dataset(
        {name: (("time", "y", "x"), values, {"units": "1"}) for name, values in data.items()},
        coords={"time": np.arange(n_time, dtype="float64"), "y": y, "x": x},
    )
    if z_levels is not None:
        ds["enthalpy"] = xr.DataArray(
            rng.uniform(0.0, 1e5, (*shape, z_levels)).astype(np.float32),
            dims=("time", "y", "x", "z"),
            coords={"z": np.linspace(0.0, 4000.0, z_levels)},
            attrs={"units": "J kg-1"},
        )
    # A non-spatial, time-less variable: process_file carries these through to
    # the output untouched.
    config_attrs: dict = {"note": "synthetic"}
    if ice_free_thickness is not None:
        config_attrs["output.ice_free_thickness_standard"] = np.float64(ice_free_thickness)
    ds["pism_config"] = xr.DataArray(np.int8(0), attrs=config_attrs)
    # PISM's own CF grid-mapping variable, which process_file drops up front.
    # ``write_crs`` below adds rioxarray's ``spatial_ref`` alongside it, so the
    # written file carries both flavours — as real regridded input files do.
    ds["mapping"] = xr.DataArray(np.int8(0), attrs={"grid_mapping_name": "polar_stereographic"})

    if time_chunks is not None:
        for name, chunk in time_chunks.items():
            ds[name] = ds[name].chunk({"time": chunk, "y": 32, "x": 32})

    return ds.rio.write_crs(CRS).rio.set_spatial_dims(x_dim="x", y_dim="y")


def test_dim_chunks_reads_spatial_chunking_despite_inconsistent_time(basins):
    """
    Inconsistent ``time`` chunks must not hide the y/x chunking.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = synthetic_greenland(basins, time_chunks={"thk": 1, "ice_mass": 3, "mask": 2})

    # Guard the premise: this is exactly the failure the helper works around.
    with pytest.raises(ValueError, match="inconsistent chunks along dimension time"):
        _ = ds.chunksizes

    y_size, x_size = ds.sizes["y"], ds.sizes["x"]
    assert _dim_chunks(ds, "y", y_size) == ds["thk"].chunksizes["y"]
    assert _dim_chunks(ds, "x", x_size) == ds["thk"].chunksizes["x"]


def test_dim_chunks_falls_back_without_dask(basins):
    """
    A NumPy-backed dataset has no chunking, so the fallback is returned.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = synthetic_greenland(basins)

    assert _dim_chunks(ds, "y", ds.sizes["y"]) == ds.sizes["y"]
    assert _dim_chunks(ds, "x", ds.sizes["x"]) == ds.sizes["x"]


def test_dim_chunks_takes_the_majority_chunking(basins):
    """
    With y/x chunked two ways, the chunking used by most variables wins.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = synthetic_greenland(basins, time_chunks={"thk": 1, "ice_mass": 1, "mask": 1})
    ds["mask"] = ds["mask"].chunk({"time": 1, "y": 16, "x": 16})

    assert _dim_chunks(ds, "y", ds.sizes["y"]) == ds["thk"].chunksizes["y"]


def test_basin_masks_survives_inconsistent_time_chunks(basins):
    """
    The regression: ``basin_masks`` used to raise on such a dataset.

    Reading the y/x chunking via ``Dataset.chunksizes`` blew up before a single
    polygon was rasterized.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = synthetic_greenland(basins, time_chunks={"thk": 1, "ice_mass": 3, "mask": 2})

    masks = basin_masks(ds, basins)

    assert [name for name, _ in masks] == basins[resolve_column(basins)].tolist()
    for _, mask in masks:
        assert mask.dims == ("y", "x")
        assert mask.shape == (ds.sizes["y"], ds.sizes["x"])
        # Chunked to match the dataset, so ``ds.where(mask)`` stays blockwise.
        assert mask.chunksizes["y"] == ds["thk"].chunksizes["y"]
        assert mask.chunksizes["x"] == ds["thk"].chunksizes["x"]
        assert bool(mask.any().compute())


def test_basin_masks_match_rio_clip(basins):
    """
    Each mask selects the same cells as ``rio.clip(..., drop=False)``.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = synthetic_greenland(basins, n_time=1)
    masks = dict(basin_masks(ds, basins))

    for _, row in basins.iterrows():
        clipped = ds[["thk"]].rio.clip([row.geometry], basins.crs, drop=False)
        expected = clipped["thk"].isel(time=0).notnull().values
        np.testing.assert_array_equal(np.asarray(masks[row[resolve_column(basins)]]), expected)


@pytest.mark.integration
def test_basin_masks_scatter_keeps_graph_small(basins):
    """
    Passing a client scatters the rasters instead of embedding them.

    ``from_array`` puts the raster bytes straight into the task graph, so on a
    real grid the graph the client ships to the scheduler grows to tens of MB
    and Dask warns "Sending large graph". Scattering leaves only future
    references behind, and must not change a single cell.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    from dask.distributed import Client  # pylint: disable=import-outside-toplevel

    def graph_bytes(masks):
        """
        Serialized size of the task graphs behind ``masks``.

        Parameters
        ----------
        masks : list of (str, xarray.DataArray)
            Output of :func:`basin_masks`.

        Returns
        -------
        int
            Total pickled graph size in bytes.
        """
        return sum(len(pickle.dumps(dict(mask.data.__dask_graph__()))) for _, mask in masks)

    # 5 km with production-sized chunks: fine enough that the rasters dominate
    # an embedded graph, coarse enough to stay a fast test.
    ds = synthetic_greenland(basins, n_time=1, resolution=5_000.0).chunk({"time": 1, "y": 256, "x": 256})

    with Client(processes=False, n_workers=1, dashboard_address=None) as client:
        embedded = basin_masks(ds, basins)
        scattered = basin_masks(ds, basins, client=client)

        raw_bytes = sum(np.asarray(mask).nbytes for _, mask in embedded)
        # Embedding carries the rasters themselves; scattering carries future
        # references plus one small task per chunk. Thresholds are loose on
        # purpose — the mechanism is the point, not a serialization size.
        assert graph_bytes(embedded) > raw_bytes
        assert graph_bytes(scattered) < graph_bytes(embedded) / 10

        # And not a single cell changes.
        assert [name for name, _ in scattered] == [name for name, _ in embedded]
        for (_, want), (_, got) in zip(embedded, scattered):
            assert got.chunksizes == want.chunksizes
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.integration
def test_process_file_basin_sums(tmp_path, basins, outlinefile):
    """
    End-to-end reduction: per-basin sums match NumPy, totals and dtypes hold.

    Runs against a real (in-process) Dask cluster, which is what makes this
    opt-in rather than part of the default suite.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest per-test temporary directory, holding the input and output netCDFs.
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    outlinefile : pathlib.Path
        Path to the outline GeoPackage handed to ``process_file``.
    """
    from dask.distributed import Client  # pylint: disable=import-outside-toplevel

    ds = synthetic_greenland(basins, n_time=3)
    infile = tmp_path / "spatial.nc"
    outfile = tmp_path / "basin.nc"
    ds.to_netcdf(infile, engine="h5netcdf")

    with Client(processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None) as client:
        process_file(infile, outfile, outlinefile, client, total_name="GIS")

    scalar = xr.open_dataset(outfile)
    try:
        names = basins[resolve_column(basins)].tolist()
        # ``glacier_id`` is a positional index for CDO's benefit; the labels live
        # in ``glacier_id_name``. Restore label indexing for the assertions below.
        assert scalar["glacier_id"].dtype == np.int32
        assert scalar["glacier_id_name"].values.tolist() == names + ["GIS"]
        scalar = scalar.set_index(glacier_id="glacier_id_name")

        masks = {name: np.asarray(mask) for name, mask in basin_masks(ds, basins)}
        for name in names:
            mask = masks[name]
            for var in ("thk", "ice_mass", "mask"):
                expected = ds[var].values[:, mask].sum(axis=1)
                np.testing.assert_allclose(
                    scalar[var].sel(glacier_id=name).values,
                    expected,
                    rtol=1e-6,
                    err_msg=f"{var} over basin {name}",
                )

            # Derived in process_file, not present in the input. This dataset
            # carries no ``output.ice_free_thickness_standard``, so the 10 m
            # fallback applies.
            glacierized = np.where(ds["thk"].values > 10, ds["ice_mass"].values, np.nan)
            np.testing.assert_allclose(
                scalar["ice_mass_glacierized"].sel(glacier_id=name).values,
                np.nansum(glacierized[:, mask], axis=1),
                rtol=1e-6,
            )

        # GIS is the sum over the basins, not an independent reduction.
        np.testing.assert_allclose(
            scalar["ice_mass"].sel(glacier_id="GIS").values,
            scalar["ice_mass"].sel(glacier_id=names).sum(dim="glacier_id").values,
            rtol=1e-6,
        )

        # ``where`` promotes integers to float; process_file restores the dtype.
        assert np.issubdtype(scalar["mask"].dtype, np.integer)
        # Non-spatial, time-less variables ride along.
        assert "pism_config" in scalar
        # Grid-mapping variables are dropped, not carried into the scalar output.
        assert "mapping" not in scalar.variables
        assert "spatial_ref" not in scalar.data_vars
        # And no dangling spatial dimension survived the reduction.
        assert set(scalar["ice_mass"].dims) == {"glacier_id", "time"}
        # Time must lead: CDO reads the first dimension as the record dimension
        # and skips every variable that does not put time there.
        for var in ("thk", "ice_mass", "mask", "ice_mass_glacierized"):
            assert scalar[var].dims == ("time", "glacier_id"), f"{var} has {scalar[var].dims}"
    finally:
        scalar.close()


def test_glacierized_mass_uses_the_configured_ice_free_thickness(tmp_path, basins, outlinefile):
    """
    ``ice_mass_glacierized`` follows the run's own reporting threshold.

    The threshold lives in ``pism_config`` as
    ``output.ice_free_thickness_standard``; a run that overrides PISM's 10 m
    default must be summarised with the value it actually used.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    outlinefile : pathlib.Path
        Path to the outline GeoPackage handed to ``process_file``.
    """
    from dask.distributed import Client  # pylint: disable=import-outside-toplevel

    threshold = 750.0
    ds = synthetic_greenland(basins, n_time=2, ice_free_thickness=threshold)
    infile = tmp_path / "spatial.nc"
    outfile = tmp_path / "basin.nc"
    ds.to_netcdf(infile, engine="h5netcdf")

    with Client(processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None) as client:
        process_file(infile, outfile, outlinefile, client)

    scalar = xr.open_dataset(outfile).set_index(glacier_id="glacier_id_name")
    try:
        masks = {name: np.asarray(mask) for name, mask in basin_masks(ds, basins)}
        for name in basins[resolve_column(basins)].tolist():
            mask = masks[name]
            glacierized = np.where(ds["thk"].values > threshold, ds["ice_mass"].values, np.nan)
            np.testing.assert_allclose(
                scalar["ice_mass_glacierized"].sel(glacier_id=name).values,
                np.nansum(glacierized[:, mask], axis=1),
                rtol=1e-6,
                err_msg=f"basin {name}",
            )
            # The 10 m default would have kept almost every cell, so the two
            # thresholds are distinguishable rather than coincidentally equal.
            at_default = np.where(ds["thk"].values > 10, ds["ice_mass"].values, np.nan)
            assert np.nansum(at_default[:, mask]) > np.nansum(glacierized[:, mask])
    finally:
        scalar.close()


def test_resolve_outfile_names_a_file_inside_a_directory(tmp_path):
    """
    A directory destination is named after the input, ``spatial_`` → ``basin_``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    outdir = tmp_path / "basins"
    outdir.mkdir()
    infile = tmp_path / "spatial" / "spatial_g1200m_id_HIRHAM5_1990_2019.nc"

    assert resolve_outfile(infile, outdir).name == "basin_g1200m_id_HIRHAM5_1990_2019.nc"
    # A trailing separator means "directory" even before it exists.
    assert resolve_outfile(infile, f"{tmp_path}/new_dir/").name == "basin_g1200m_id_HIRHAM5_1990_2019.nc"
    assert (tmp_path / "new_dir").is_dir()
    # Inputs that do not follow the naming convention are still handled.
    assert resolve_outfile(tmp_path / "odd.nc", outdir).name == "basin_odd.nc"


def test_resolve_outfile_passes_file_paths_through(tmp_path):
    """
    An explicit file name is used as given, and its parent is created.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    target = tmp_path / "deep" / "nested" / "basin.nc"

    assert resolve_outfile(tmp_path / "spatial_x.nc", target) == target
    assert target.parent.is_dir()


def test_resolve_outfile_reports_unwritable_destinations(tmp_path):
    """
    An unwritable destination fails before the reduction, not after it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    try:
        with pytest.raises(PermissionError, match="cannot write to"):
            resolve_outfile(tmp_path / "spatial_x.nc", locked)
    finally:
        locked.chmod(0o700)


def test_process_file_accepts_a_directory_destination(tmp_path, basins, outlinefile):
    """
    Pointing the CLI at an output directory writes the file, not an error.

    Passing a directory used to run the entire reduction and only then die
    in ``to_netcdf`` with ``PermissionError: ... /output/basins``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    outlinefile : pathlib.Path
        Path to the outline GeoPackage handed to ``process_file``.
    """
    from dask.distributed import Client  # pylint: disable=import-outside-toplevel

    ds = synthetic_greenland(basins, n_time=1)
    infile = tmp_path / "spatial_g1200m_test.nc"
    outdir = tmp_path / "basins"
    outdir.mkdir()
    ds.to_netcdf(infile, engine="h5netcdf")

    with Client(processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None) as client:
        process_file(infile, outdir, outlinefile, client)

    (written,) = outdir.glob("*.nc")
    assert written.name == "basin_g1200m_test.nc"
    scalar = xr.open_dataset(written)
    try:
        assert "ice_mass" in scalar.data_vars
    finally:
        scalar.close()


def test_dataset_crs_is_read_from_the_grid_mapping(tmp_path, basins):
    """
    The projection comes from the file, so campaigns need not declare it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    infile = tmp_path / "spatial.nc"
    synthetic_greenland(basins, n_time=1).to_netcdf(infile, engine="h5netcdf")

    with xr.open_dataset(infile) as ds:
        detected = dataset_crs(ds)
        assert ProjCRS.from_user_input(detected) == ProjCRS.from_user_input(CRS)
        # An explicit CRS always wins over what the file says.
        assert dataset_crs(ds, "EPSG:32607") == "EPSG:32607"

    # A file with no usable grid mapping must say so rather than guess.
    bare = xr.Dataset({"thk": (("y", "x"), np.zeros((2, 2)))}, coords={"y": [1.0, 0.0], "x": [0.0, 1.0]})
    with pytest.raises(ValueError, match="no grid-mapping variable"):
        dataset_crs(bare)


def test_process_file_reprojects_outlines(tmp_path, basins, outlinefile):
    """
    Bring outlines stored in another CRS onto the model grid.

    RGI outlines ship in EPSG:4326 while the grid is projected; rasterizing
    without reprojecting would put every region outside the domain and sum
    nothing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    outlinefile : pathlib.Path
        Path to the outline GeoPackage in the dataset's own CRS.
    """
    from dask.distributed import Client  # pylint: disable=import-outside-toplevel

    ds = synthetic_greenland(basins, n_time=1)
    infile = tmp_path / "spatial.nc"
    ds.to_netcdf(infile, engine="h5netcdf")

    lonlat_outlines = tmp_path / "outlines_4326.gpkg"
    basins.to_crs("EPSG:4326").to_file(lonlat_outlines)

    with Client(processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None) as client:
        process_file(infile, tmp_path / "native.nc", outlinefile, client)
        process_file(infile, tmp_path / "lonlat.nc", lonlat_outlines, client)

    native = xr.open_dataset(tmp_path / "native.nc")
    lonlat = xr.open_dataset(tmp_path / "lonlat.nc")
    try:
        # Reprojection is not bit-exact, but must land on the same cells.
        np.testing.assert_allclose(lonlat["ice_mass"].values, native["ice_mass"].values, rtol=1e-6)
        assert float(native["ice_mass"].sum()) > 0.0
    finally:
        native.close()
        lonlat.close()


def test_process_file_dim_name_and_total_are_configurable(tmp_path, basins, outlinefile):
    """
    ``dim_name`` names the region dimension; the total row is opt-in.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    outlinefile : pathlib.Path
        Path to the outline GeoPackage handed to ``process_file``.
    """
    from dask.distributed import Client  # pylint: disable=import-outside-toplevel

    ds = synthetic_greenland(basins, n_time=1)
    infile = tmp_path / "spatial.nc"
    ds.to_netcdf(infile, engine="h5netcdf")
    n_regions = len(basins)

    with Client(processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None) as client:
        process_file(infile, tmp_path / "default.nc", outlinefile, client)
        process_file(infile, tmp_path / "basin.nc", outlinefile, client, dim_name="basin", total_name="GIS")

    default = xr.open_dataset(tmp_path / "default.nc")
    named = xr.open_dataset(tmp_path / "basin.nc")
    try:
        # Neutral default, no total appended.
        assert "glacier_id" in default.dims
        assert "glacier_id_name" in default.coords
        assert default.sizes["glacier_id"] == n_regions

        # Greenland campaigns keep their own dimension name and GIS total.
        assert "basin" in named.dims
        assert named.sizes["basin"] == n_regions + 1
        assert named["basin_name"].values.tolist()[-1] == "GIS"
    finally:
        default.close()
        named.close()


def test_process_file_reports_outline_area(tmp_path, basins, outlinefile):
    """
    Each region carries its outline area, independent of the grid.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    outlinefile : pathlib.Path
        Path to the outline GeoPackage handed to ``process_file``.
    """
    from dask.distributed import Client  # pylint: disable=import-outside-toplevel

    ds = synthetic_greenland(basins, n_time=1)
    infile = tmp_path / "spatial.nc"
    outfile = tmp_path / "basin.nc"
    ds.to_netcdf(infile, engine="h5netcdf")

    with Client(processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None) as client:
        process_file(infile, outfile, outlinefile, client)

    scalar = xr.open_dataset(outfile)
    try:
        assert scalar["area"].attrs["units"] == "m^2"
        np.testing.assert_allclose(
            scalar["area"].values,
            basins.geometry.area.values,
            rtol=1e-9,
        )
    finally:
        scalar.close()


def test_resolve_column_prefers_glacier_id_then_rgi_id(basins):
    """
    Outline files may name the label column in any of three ways.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    assert resolve_column(basins.rename(columns={"glacier_id": "rgi_id"})) == "rgi_id"
    assert resolve_column(basins.assign(rgi_id=basins[resolve_column(basins)])) == "glacier_id"
    assert resolve_column(basins[["SUBREGION1", "geometry"]]) == "SUBREGION1"

    with pytest.raises(ValueError, match="has none of"):
        resolve_column(basins[["geometry"]])
    with pytest.raises(ValueError, match="no column 'nope'"):
        resolve_column(basins, "nope")


def test_basin_masks_rejects_empty_and_mislabeled_outlines(basins):
    """
    Empty outlines and missing name columns fail with clear messages.

    An empty GeoDataFrame used to surface as ``client.scatter([])`` dying
    deep inside distributed with "not enough values to unpack" — the
    signature of a truncated outline file copied to the cluster.

    Parameters
    ----------
    basins : geopandas.GeoDataFrame
        Mouginot basin outlines fixture.
    """
    ds = synthetic_greenland(basins, n_time=1)

    with pytest.raises(ValueError, match="no features"):
        basin_masks(ds, basins.iloc[0:0])

    with pytest.raises(ValueError, match=r"no column 'nope'.*SUBREGION1"):
        basin_masks(ds, basins, column="nope")
