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
Tests for the Cloud Optimized GeoTIFF writers in :mod:`pism_terra.raster`.

GDAL 3.13 gave the COG driver a ``Create()`` entry point that loses band
descriptions and turns NaN nodata into 0 when rasterio writes through it, so
these check the CreateCopy-based helpers preserve both on whatever GDAL is
installed, and that the staging GeoTIFF never survives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from pism_terra.raster import cog_writer, write_cog

CRS = "EPSG:32606"


def _bands(height: int = 80, width: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a dh band with a NaN strip and a constant error band.

    Parameters
    ----------
    height : int, optional
        Rows, by default 80.
    width : int, optional
        Columns, by default 80.

    Returns
    -------
    tuple of numpy.ndarray
        ``(dh, err)`` float32 arrays.
    """
    dh = np.full((height, width), 2.0, dtype=np.float32)
    dh[:20] = np.nan
    err = np.full((height, width), 5.0, dtype=np.float32)
    return dh, err


def _check_cog(path: Path) -> None:
    """
    Assert ``path`` is a COG with intact descriptions, nodata and values.

    Parameters
    ----------
    path : pathlib.Path
        The written file.
    """
    with rasterio.open(path) as src:
        assert src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
        assert src.descriptions == ("dh", "err_dh")
        assert np.isnan(src.nodata)
        dh = src.read(1, masked=True)
        err = src.read(2, masked=True)
    assert dh.mask.sum() == 20 * 80
    np.testing.assert_array_equal(dh.compressed(), 2.0)
    np.testing.assert_array_equal(err.compressed(), 5.0)


def test_cog_writer_preserves_descriptions_and_nan_nodata(tmp_path: Path):
    """
    Check bands written through the staging dataset come out as a proper COG.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    out = tmp_path / "nested" / "dh.tif"
    profile = {
        "driver": "COG",  # ignored: the staging file is a GTiff
        "dtype": "float32",
        "count": 2,
        "height": 80,
        "width": 80,
        "crs": CRS,
        "transform": from_origin(0.0, 8000.0, 100.0, 100.0),
        "nodata": np.nan,
    }

    with cog_writer(out, profile, compress="DEFLATE", predictor=3, blocksize=512, BIGTIFF="YES") as dst:
        for idx, (band, name) in enumerate(zip(_bands(), ("dh", "err_dh")), start=1):
            dst.write(band, idx)
            dst.set_band_description(idx, name)

    _check_cog(out)
    assert sorted(p.name for p in out.parent.iterdir()) == ["dh.tif"]


def test_cog_writer_removes_staging_file_on_error(tmp_path: Path):
    """
    Check a failure inside the block leaves neither the COG nor the staging GeoTIFF.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    out = tmp_path / "dh.tif"
    profile = {
        "dtype": "float32",
        "count": 1,
        "height": 8,
        "width": 8,
        "crs": CRS,
        "transform": from_origin(0, 800, 100, 100),
    }

    with pytest.raises(RuntimeError, match="boom"):
        with cog_writer(out, profile):
            raise RuntimeError("boom")

    assert not list(tmp_path.iterdir())


def test_write_cog_from_dataarray(tmp_path: Path):
    """
    Check a rioxarray DataArray round-trips with its nodata, name and values.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    dh, err = _bands()
    da = xr.DataArray(
        np.stack([dh, err]),
        dims=("band", "y", "x"),
        coords={
            "band": [1, 2],
            "y": 8000.0 - 100.0 * (np.arange(80) + 0.5),
            "x": 100.0 * (np.arange(80) + 0.5),
        },
        attrs={"long_name": ("dh", "err_dh")},
    )
    da = da.rio.write_crs(CRS).rio.write_nodata(np.nan)
    out = tmp_path / "dh.tif"

    written = write_cog(da, out, compress="ZSTD", predictor=3, overview_resampling="cubic", num_threads="ALL_CPUS")

    assert written == out
    _check_cog(out)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dh.tif"]
