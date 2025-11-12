"""PyTest test configuration."""

# pylint: disable=missing-docstring

import geopandas as gpd
import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin


def pytest_addoption(parser):  # numpydoc ignore=GL08
    parser.addoption("--skip-integration", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):  # numpydoc ignore=GL08
    if config.getoption("--skip-integration"):
        skip_slow = pytest.mark.skip(reason="--skip-integration option provided")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(name="rgi")
def fixture_read_rgi() -> gpd.GeoDataFrame:
    """
    Fixture to read test RGI.

    Returns
    -------
    gpd.GeoDataFrame
        RGI test.
    """
    return gpd.read_file("tests/rgi_test.gpkg")


@pytest.fixture(name="in_memory_raster")
def in_memory_raster():
    """
    Create an in-memory single-band raster dataset for testing.

    This fixture generates a small 10x10 raster using `rasterio.MemoryFile` with:
    - 1-meter resolution
    - Upper-left origin at (0, 10)
    - CRS: EPSG:32633 (UTM zone 33N)
    - Data filled with constant value 1

    The raster is written with a basic GeoTIFF profile suitable for unit tests that
    require spatially referenced raster data without writing to disk.

    Returns
    -------
    rasterio.io.MemoryFile
        A `MemoryFile` object containing the test raster. Use `.open()` to get a dataset.

    Examples
    --------
    def test_something(in_memory_raster):
        with in_memory_raster.open() as dataset:
            assert dataset.read(1).shape == (10, 10)
    """
    width = height = 10
    transform = from_origin(0, 10, 1, 1)  # (left, top, xres, yres)
    data = np.ones((1, height, width), dtype="uint8")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32633",
        "transform": transform,
    }

    memfile = MemoryFile()
    with memfile.open(**profile) as dataset:
        dataset.write(data)

    return memfile


@pytest.fixture(name="dataset")
def in_memory_raster_da():
    """
    Create and return an open in-memory single-band raster dataset for testing.

    This fixture generates a small 10x10 raster with:
    - Constant value 1
    - 1-meter resolution
    - Top-left origin at (0, 10)
    - CRS: EPSG:32633 (UTM zone 33N)

    The raster is created in memory using `rasterio.MemoryFile` and returned as
    an open dataset ready for testing.

    Yields
    ------
    rasterio.io.DatasetReader
        An open rasterio dataset backed by an in-memory file. Automatically closed
        after the test completes.

    Examples
    --------
    def test_read_data(in_memory_raster):
        assert in_memory_raster.read(1).shape == (10, 10)
        assert in_memory_raster.crs.to_epsg() == 32633
    """
    width = height = 10
    transform = from_origin(0, 10, 1, 1)
    data = np.ones((1, height, width), dtype="uint8")
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32633",
        "transform": transform,
    }

    memfile = MemoryFile()
    dataset = memfile.open(**profile)
    dataset.write(data)

    yield dataset

    dataset.close()
    memfile.close()
