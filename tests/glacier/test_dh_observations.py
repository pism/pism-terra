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
Tests for the Hugonnet dh observations in :mod:`pism_terra.glacier.observations`.

Everything runs offline: synthetic UTM tiles stand in for the Hugonnet
archive, and the staged raster is placed where :func:`add_dh_observations`
would cache the S3 download.

Covers:
- ``dh_tile_index`` footprint parsing (all four hemispheres) and err pairing.
- ``build_dh_raster`` mosaicking into a two-band COG, and the no-overlap skip.
- ``dh_from_tif`` alignment, outline clipping, and CF attributes.
- ``add_dh_observations`` merging into an obs file, idempotence, and the
  unknown-dataset error.
- ``ensure_dh_tiles`` extraction of one period from the nested archive.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import rioxarray  # pylint: disable=unused-import
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

from pism_terra.domain import create_domain
from pism_terra.glacier.observations import (
    add_dh_observations,
    build_dh_raster,
    dh_from_tif,
    dh_tile_index,
    ensure_dh_tiles,
)
from pism_terra.glacier.s4f import write_dh_cogs

CRS = "EPSG:32607"
PERIOD = "2000-01-01_2020-01-01"
# Inside tile N60W139 (UTM zone 7N).
GEOM_UTM = box(615_000.0, 6_665_000.0, 625_000.0, 6_675_000.0)


def write_tile(path: Path, value: float, bounds=(610_000.0, 6_660_000.0, 630_000.0, 6_680_000.0)) -> Path:
    """
    Write a constant-valued 100 m UTM tile like the Hugonnet tifs.

    Parameters
    ----------
    path : pathlib.Path
        Target GeoTIFF path, parents created.
    value : float
        Constant cell value.
    bounds : tuple of float, optional
        ``(minx, miny, maxx, maxy)`` extent in EPSG:32607.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    minx, miny, maxx, maxy = bounds
    width, height = int((maxx - minx) / 100), int((maxy - miny) / 100)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": width,
        "height": height,
        "crs": CRS,
        "transform": from_origin(minx, maxy, 100.0, 100.0),
        "nodata": np.nan,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(np.full((height, width), value, dtype=np.float32), 1)
    return path


@pytest.fixture(name="tiles_dir")
def fixture_tiles_dir(tmp_path: Path) -> Path:
    """
    A one-tile Hugonnet layout: dh plus its err companion.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.

    Returns
    -------
    pathlib.Path
        The period directory holding a ``07N`` zone with one tile pair.
    """
    zone = tmp_path / PERIOD / "07N"
    write_tile(zone / f"N60W139_dh_{PERIOD}.tif", 2.0)
    write_tile(zone / f"N60W139_err_dh_{PERIOD}.tif", 5.0)
    return tmp_path / PERIOD


@pytest.fixture(name="grid")
def fixture_grid() -> xr.Dataset:
    """
    A 200 m target grid covering the synthetic outline.

    Returns
    -------
    xarray.Dataset
        Grid with CRS EPSG:32607.
    """
    return create_domain([612_000.0, 628_000.0], [6_662_000.0, 6_678_000.0], resolution=200.0, crs=CRS)


def test_dh_tile_index_parses_footprints_and_pairs_errors(tmp_path: Path):
    """
    Parse footprints from the south-west-corner names and pair err tiles.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    zone = tmp_path / "07N"
    zone.mkdir()
    for name in (f"N60W139_dh_{PERIOD}.tif", f"N60W139_err_dh_{PERIOD}.tif", f"S69E155_dh_{PERIOD}.tif"):
        (zone / name).touch()

    index = dh_tile_index(tmp_path)

    assert len(index) == 2
    by_name = {dh.name: (err, footprint) for dh, err, footprint in index}
    err, footprint = by_name[f"N60W139_dh_{PERIOD}.tif"]
    assert err is not None and footprint.bounds == (-139.0, 60.0, -138.0, 61.0)
    err, footprint = by_name[f"S69E155_dh_{PERIOD}.tif"]
    assert err is None and footprint.bounds == (155.0, -69.0, 156.0, -68.0)


def test_build_dh_raster_writes_two_band_cog(tmp_path: Path, tiles_dir: Path):
    """
    The mosaic carries dh in band 1 and the error in band 2, in the dst CRS.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    tiles_dir : pathlib.Path
        Synthetic tile layout fixture.
    """
    geometry_4326 = gpd.GeoSeries([GEOM_UTM], crs=CRS).to_crs("EPSG:4326").iloc[0]
    out = tmp_path / "out" / "RGI2000-v7.0-C-01" / "RGI2000-v7.0-C-01-99999_dh.tif"

    written = build_dh_raster(
        "RGI2000-v7.0-C-01-99999", geometry_4326, CRS, dh_tile_index(tiles_dir), out, resolution=100.0
    )

    assert written == out
    with rasterio.open(out) as src:
        assert src.count == 2
        assert src.crs == rasterio.CRS.from_user_input(CRS)
        assert src.descriptions == ("dh", "err_dh")
        dh, err = src.read(1, masked=True), src.read(2, masked=True)
    np.testing.assert_allclose(dh.compressed(), 2.0)
    np.testing.assert_allclose(err.compressed(), 5.0)
    assert dh.count() > 0


def test_build_dh_raster_skips_complexes_without_tiles(tmp_path: Path, tiles_dir: Path):
    """
    An outline far from every tile yields no file, not an error.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    tiles_dir : pathlib.Path
        Synthetic tile layout fixture.
    """
    faraway = box(10.0, 46.0, 10.5, 46.5)  # the Alps, in EPSG:4326

    written = build_dh_raster("X", faraway, CRS, dh_tile_index(tiles_dir), tmp_path / "x_dh.tif")

    assert written is None
    assert not (tmp_path / "x_dh.tif").exists()


def test_dh_from_tif_aligns_and_clips(tmp_path: Path, tiles_dir: Path, grid: xr.Dataset):
    """
    Align the fields to the target grid: NaN outside the outline, CF attrs set.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    tiles_dir : pathlib.Path
        Synthetic tile layout fixture.
    grid : xarray.Dataset
        Target grid fixture.
    """
    geometry_4326 = gpd.GeoSeries([GEOM_UTM], crs=CRS).to_crs("EPSG:4326").iloc[0]
    tif = build_dh_raster("X", geometry_4326, CRS, dh_tile_index(tiles_dir), tmp_path / "x_dh.tif")
    assert tif is not None

    ds = dh_from_tif(tif, grid, [GEOM_UTM])

    assert set(ds.data_vars) == {"dh", "dh_err"}
    assert ds["dh"].sizes == {"y": grid.sizes["y"], "x": grid.sizes["x"]}
    inside = ds["dh"].sel(x=620_000.0, y=6_670_000.0, method="nearest")
    outside = ds["dh"].sel(x=613_000.0, y=6_663_000.0, method="nearest")
    np.testing.assert_allclose(float(inside), 2.0)
    assert np.isnan(float(outside))
    assert ds["dh"].attrs["units"] == "m"
    assert "Hugonnet" in ds["dh_err"].attrs["source"]


def test_add_dh_observations_merges_into_obs_file(tmp_path: Path, tiles_dir: Path, grid: xr.Dataset):
    """
    Merge dh and dh_err into the obs file; existing variables survive.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    tiles_dir : pathlib.Path
        Synthetic tile layout fixture.
    grid : xarray.Dataset
        Target grid fixture.
    """
    rgi_id = "RGI2000-v7.0-C-01-99999"
    geometry_4326 = gpd.GeoSeries([GEOM_UTM], crs=CRS).to_crs("EPSG:4326").iloc[0]
    staging = tmp_path / "staging"
    staging.mkdir()
    # Pre-place the raster where the S3 download would cache it.
    build_dh_raster(rgi_id, geometry_4326, CRS, dh_tile_index(tiles_dir), staging / f"{rgi_id}_dh.tif")

    obs_file = tmp_path / f"obs_{rgi_id}.nc"
    obs = grid.copy()
    obs["u_observed"] = xr.ones_like(grid["x"] * grid["y"], dtype="float32")
    obs.to_netcdf(obs_file)

    result = add_dh_observations(obs_file, grid, [GEOM_UTM], rgi_id, staging_path=staging)

    assert {"dh", "dh_err"} <= set(result.data_vars)
    with xr.open_dataset(obs_file) as written:
        assert {"dh", "dh_err", "u_observed"} <= set(written.data_vars)
        np.testing.assert_allclose(
            float(written["dh"].sel(x=620_000.0, y=6_670_000.0, method="nearest")), 2.0, rtol=1e-6
        )
        assert written["dh"].attrs.get("grid_mapping") == "spatial_ref"
        mtime = obs_file.stat().st_mtime_ns

    # A second call finds dh present and leaves the file untouched.
    again = add_dh_observations(obs_file, grid, [GEOM_UTM], rgi_id, staging_path=staging)
    assert "dh" in again.data_vars
    assert obs_file.stat().st_mtime_ns == mtime


def test_add_dh_observations_survives_a_missing_raster(tmp_path: Path, grid: xr.Dataset, monkeypatch):
    """
    A complex without a prepared raster leaves the obs file untouched.

    Complexes outside the observed tiles have nothing on S3; staging must
    warn and continue instead of dying on the download.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    grid : xarray.Dataset
        Target grid fixture.
    monkeypatch : pytest.MonkeyPatch
        Used to make the S3 download fail.
    """
    import pism_terra.glacier.observations as obs_mod  # pylint: disable=import-outside-toplevel

    def _boom(s3_uri, dest):
        """
        Fail like a download of a missing S3 object.

        Parameters
        ----------
        s3_uri : str
            The URI whose download fails.
        dest : pathlib.Path
            Ignored.
        """
        raise FileNotFoundError(s3_uri)

    monkeypatch.setattr(obs_mod, "download_from_s3", _boom)
    obs_file = tmp_path / "obs_X.nc"
    grid.to_netcdf(obs_file)

    result = add_dh_observations(obs_file, grid, [GEOM_UTM], "X", staging_path=tmp_path)

    assert "dh" not in result.data_vars
    with xr.open_dataset(obs_file) as written:
        assert "dh" not in written.data_vars


def test_add_dh_observations_rejects_unknown_dataset(tmp_path: Path, grid: xr.Dataset):
    """
    An unknown dh dataset fails fast, before touching the obs file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    grid : xarray.Dataset
        Target grid fixture.
    """
    with pytest.raises(NotImplementedError, match="hugonnet"):
        add_dh_observations(tmp_path / "obs.nc", grid, [GEOM_UTM], "X", dataset="frobnicate")


def test_write_dh_cogs_places_planning_cogs(tmp_path: Path, tiles_dir: Path, grid: xr.Dataset):
    """
    Write dh and dh_err COGs beside the boot COGs; skip without a dh key.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    tiles_dir : pathlib.Path
        Synthetic tile layout fixture.
    grid : xarray.Dataset
        Target grid fixture.
    """
    rgi_id = "RGI2000-v7.0-C-01-99999"
    geometry_4326 = gpd.GeoSeries([GEOM_UTM], crs=CRS).to_crs("EPSG:4326").iloc[0]
    staging = tmp_path / "staging"
    staging.mkdir()
    # Pre-place the raster where fetch_dh_raster caches the S3 download.
    build_dh_raster(rgi_id, geometry_4326, CRS, dh_tile_index(tiles_dir), staging / f"{rgi_id}_dh.tif")
    out = tmp_path / "input"
    out.mkdir()
    config = {"dh": "hugonnet", "bucket": "b", "prefix": "p"}

    written = write_dh_cogs(config, rgi_id, grid, [GEOM_UTM], out, staging)

    assert set(written) == {f"{rgi_id}_dh", f"{rgi_id}_dh_err"}
    with rasterio.open(written[f"{rgi_id}_dh"]) as src:
        assert src.crs == rasterio.CRS.from_user_input(CRS)
        assert (src.width, src.height) == (grid.sizes["x"], grid.sizes["y"])
        dh = src.read(1, masked=True)
    np.testing.assert_allclose(dh.compressed(), 2.0, rtol=1e-6)

    assert not write_dh_cogs({"dh": "none"}, rgi_id, grid, [GEOM_UTM], out, staging)
    assert not write_dh_cogs({}, rgi_id, grid, [GEOM_UTM], out, staging)


def test_ensure_dh_tiles_extracts_one_period(tmp_path: Path):
    """
    The requested period's inner zip is extracted; a missing period raises.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    tile = write_tile(tmp_path / "src" / PERIOD / "07N" / f"N60W139_dh_{PERIOD}.tif", 1.0)
    inner = tmp_path / f"{PERIOD}.zip"
    with zipfile.ZipFile(inner, "w") as z:
        z.write(tile, f"{PERIOD}/07N/{tile.name}")
    extract_path = tmp_path / "extract"
    extract_path.mkdir()
    # The archive is already in place, so no S3 download happens.
    with zipfile.ZipFile(extract_path / "mb_rgi7.zip", "w") as z:
        z.write(inner, f"mb_rgi7/{PERIOD}.zip")

    tiles_dir = ensure_dh_tiles(extract_path, "s3://bucket/hugonnet/mb_rgi7.zip")

    assert tiles_dir == extract_path / PERIOD
    assert (tiles_dir / "07N" / f"N60W139_dh_{PERIOD}.tif").exists()

    with pytest.raises(KeyError, match="2015-01-01_2020-01-01"):
        ensure_dh_tiles(extract_path, "s3://bucket/hugonnet/mb_rgi7.zip", start="2015-01-01")
