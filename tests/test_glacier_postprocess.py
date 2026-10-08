"""
Tests for the per-glacier scalar post-processing.

Covers the derived ``ice_mass_glacierized``, which counts only cells thicker
than the run's own ``output.ice_free_thickness_standard``.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

from pism_terra.glacier.postprocess import process_file

OUTLINES = Path(__file__).parent / "rgi_test.gpkg"


@pytest.fixture(name="outlines")
def fixture_outlines():
    """
    A single RGI outline, projected to a metric CRS.

    Returns
    -------
    geopandas.GeoDataFrame
        One glacier complex in UTM, so the synthetic grid can be built in
        metres around it.
    """
    outlines = gpd.read_file(OUTLINES).iloc[:1]
    return outlines.to_crs(outlines.estimate_utm_crs())


def synthetic_glacier(outlines, n_time=2, ice_free_thickness=None):
    """
    Synthetic PISM-like output covering the outlines.

    Parameters
    ----------
    outlines : geopandas.GeoDataFrame
        Outlines the grid must cover, in a metric CRS.
    n_time : int, default 2
        Length of the time dimension.
    ice_free_thickness : float or None, optional
        When set, recorded as ``output.ice_free_thickness_standard`` on the
        ``pism_config`` variable, as PISM does.

    Returns
    -------
    xarray.Dataset
        Dataset with ``thk``, ``ice_mass`` and a CF grid mapping.
    """
    minx, miny, maxx, maxy = outlines.total_bounds
    pad = 500.0
    x = np.arange(minx - pad, maxx + pad, 250.0)
    y = np.arange(miny - pad, maxy + pad, 250.0)
    shape = (n_time, y.size, x.size)
    rng = np.random.default_rng(0)

    ds = xr.Dataset(
        {
            "thk": (("time", "y", "x"), rng.uniform(0.0, 3000.0, shape).astype(np.float32), {"units": "m"}),
            "ice_mass": (("time", "y", "x"), rng.uniform(0.0, 1e12, shape), {"units": "kg"}),
        },
        coords={"time": np.arange(n_time, dtype="float64"), "y": y, "x": x},
    )

    config_attrs: dict = {"note": "synthetic"}
    if ice_free_thickness is not None:
        config_attrs["output.ice_free_thickness_standard"] = np.float64(ice_free_thickness)
    ds["pism_config"] = xr.DataArray(np.int8(0), attrs=config_attrs)

    ds = ds.rio.write_crs(outlines.crs).rio.set_spatial_dims(x_dim="x", y_dim="y")
    # process_file reads the CRS off the CF grid-mapping variable by name.
    ds["spatial_ref"].attrs.setdefault("crs_wkt", outlines.crs.to_wkt())
    return ds


def clipped_glacierized_sum(ds, outlines, threshold):
    """
    Independently derive the expected per-outline glacierized mass sum.

    Parameters
    ----------
    ds : xarray.Dataset
        The synthetic spatial input.
    outlines : geopandas.GeoDataFrame
        Outlines in the dataset's CRS.
    threshold : float
        Ice-free thickness standard in metres.

    Returns
    -------
    numpy.ndarray
        Summed glacierized mass per time step, for the first outline.
    """
    glacierized = ds["ice_mass"].where(ds["thk"] > threshold)
    glacierized = glacierized.rio.write_crs(ds.rio.crs).rio.set_spatial_dims(x_dim="x", y_dim="y")
    clipped = glacierized.rio.clip([outlines.iloc[0].geometry], drop=False)
    return clipped.sum(dim=["y", "x"]).values


def run_process_file(tmp_path, ds):
    """
    Write ``ds`` into the layout process_file expects and post-process it.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    ds : xarray.Dataset
        Synthetic spatial output to write.

    Returns
    -------
    xarray.Dataset
        The per-outline scalar result.
    """
    from dask.distributed import Client  # pylint: disable=import-outside-toplevel

    spatial_dir = tmp_path / "output" / "spatial"
    spatial_dir.mkdir(parents=True)
    infile = spatial_dir / "spatial.nc"
    ds.to_netcdf(infile, engine="netcdf4")

    with Client(processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None) as client:
        process_file(infile, OUTLINES, "C", client)

    (scalar_file,) = (tmp_path / "output" / "processed_scalar").glob("fldsum_C_*.nc")
    return xr.open_dataset(scalar_file)


@pytest.mark.parametrize("threshold", [None, 750.0])
def test_glacierized_mass_honors_the_ice_free_thickness(tmp_path, outlines, threshold):
    """
    Only cells above the reporting threshold contribute to glacierized mass.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    outlines : geopandas.GeoDataFrame
        Single-outline fixture.
    threshold : float or None
        Value written to ``pism_config``; ``None`` exercises the 10 m
        fallback.
    """
    ds = synthetic_glacier(outlines, ice_free_thickness=threshold)
    scalar = run_process_file(tmp_path, ds)

    try:
        assert "ice_mass_glacierized" in scalar.data_vars

        # ``make_cdo_readable`` turns RGIid into a positional index; restore
        # label selection to pick the outline the grid was built around.
        by_name = scalar.set_index(RGIid="RGIid_name")
        got = by_name["ice_mass_glacierized"].sel(RGIid=outlines.iloc[0]["rgi_id"]).values

        effective = 10.0 if threshold is None else threshold
        expected = clipped_glacierized_sum(ds, outlines, effective)
        np.testing.assert_allclose(got, expected, rtol=1e-6)

        # The two thresholds must give different answers, otherwise this test
        # would pass even if pism_config were ignored.
        other = clipped_glacierized_sum(ds, outlines, 750.0 if threshold is None else 10.0)
        assert not np.allclose(expected, other)
    finally:
        scalar.close()


def test_glacierized_mass_is_skipped_without_thickness(tmp_path, outlines):
    """
    A reduced variable set does not gain a bogus glacierized diagnostic.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    outlines : geopandas.GeoDataFrame
        Single-outline fixture.
    """
    ds = synthetic_glacier(outlines).drop_vars("thk")
    scalar = run_process_file(tmp_path, ds)

    try:
        assert "ice_mass" in scalar.data_vars
        assert "ice_mass_glacierized" not in scalar.data_vars
    finally:
        scalar.close()
