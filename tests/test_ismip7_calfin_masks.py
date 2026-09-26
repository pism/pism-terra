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
Tests for the per-resolution CalFin front-retreat masks.

PISM's prescribed front retreat zeroes the ice of every margin cell whose
mask value is below 1. A 0/1 mask interpolated from another grid carries a
fraction on every margin cell, so the mask must be built on the run's own
grid and be exactly 0 or 1 there. These tests pin the grid alignment, the
binary values, the filename token a run resolves, and staging's choice.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from pism_terra.config import load_config
from pism_terra.domain import create_domain
from pism_terra.ismip7.greenland import stage as stage_module
from pism_terra.ismip7.greenland.forcing import (
    CALFIN_RESOLUTIONS,
    calfin_filename,
    prepare_calfin,
)
from pism_terra.ismip7.greenland.stage import (
    resolution_in_meters,
    retreat_file_for_resolution,
)
from pism_terra.raster import rasterize_retreat_masks

CONFIG_DIR = Path(__file__).resolve().parents[1] / "pism_terra" / "config"

#: The ISMIP7 Greenland domain of setup_ismip7_greenland.toml.
GRIS_X_BNDS = [-750650.0, 977350.0]
GRIS_Y_BNDS = [-3412550.0, -604550.0]


def test_resolution_in_meters():
    """
    Every spelling a config or CLI uses lands on whole meters.
    """
    assert resolution_in_meters(1800) == 1800
    assert resolution_in_meters("1800m") == 1800
    assert resolution_in_meters("1800 m") == 1800
    assert resolution_in_meters("1.8km") == 1800
    with pytest.raises(ValueError):
        resolution_in_meters("1800")


def test_retreat_file_for_resolution_swaps_the_grid_token():
    """
    The configured 450 m name resolves to the file of the run's grid.
    """
    name = "pism_g450m_frontretreat_calfin_1972_2019_MS.nc"
    assert retreat_file_for_resolution(name, "1800m") == "pism_g1800m_frontretreat_calfin_1972_2019_MS.nc"
    assert retreat_file_for_resolution(name, 900) == "pism_g900m_frontretreat_calfin_1972_2019_MS.nc"
    assert retreat_file_for_resolution(name, None) == name
    assert retreat_file_for_resolution("retreat_mask.nc", "1800m") == "retreat_mask.nc"
    assert retreat_file_for_resolution(name, 1800) == calfin_filename(1800)


def test_every_published_resolution_tiles_the_greenland_domain():
    """
    Each published resolution is a whole number of cells across the ISMIP7 domain.

    Otherwise PISM would build a different grid from the same bounds and
    interpolate the mask after all.
    """
    for res in CALFIN_RESOLUTIONS:
        for lo, hi in (GRIS_X_BNDS, GRIS_Y_BNDS):
            assert (hi - lo) % res == 0, f"{res} m does not tile {lo}..{hi}"


def test_prepare_calfin_rejects_a_resolution_that_does_not_tile_the_domain(tmp_path):
    """
    A resolution the domain is not a whole number of cells at is refused up front.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    with pytest.raises(ValueError, match="not a whole number of cells"):
        prepare_calfin(tmp_path, [7000], GRIS_X_BNDS, GRIS_Y_BNDS)


def test_masks_sit_on_the_pism_grid_and_are_binary(tmp_path):
    """
    One date rasterized at two resolutions lands on create_domain's cells, as 0/1.

    The reference outline is a 9 km square; the retreated area is its western
    half, so cells with centres west of the front are 0 and the rest 1.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    x_bnds, y_bnds = [0.0, 9000.0], [-9000.0, 0.0]
    outline = gpd.GeoSeries([box(0, -9000, 9000, 0)], crs="EPSG:3413")
    retreated = gpd.GeoDataFrame(
        {"Date": [pd.Timestamp("1980-01-15")]}, geometry=[box(0, -9000, 4500, 0)], crs="EPSG:3413"
    )

    files = rasterize_retreat_masks(
        tmp_path, pd.Timestamp("1980-01-15"), retreated, outline, x_bnds, y_bnds, [900, 1800]
    )

    assert sorted(files) == [900, 1800]
    for res, path in files.items():
        assert path.name == f"frontretreat_g{res}m_1980-1-15.nc"
        with xr.open_dataset(path) as ds:
            grid = create_domain(x_bnds, y_bnds, res)
            np.testing.assert_array_equal(ds["x"].values, grid["x"].values)
            np.testing.assert_array_equal(ds["y"].values, grid["y"].values)
            mask = ds["land_ice_area_fraction_retreat"]
            assert mask.dims == ("time", "y", "x")
            assert mask.sizes["time"] == 1
            values = mask.isel(time=0).values
            assert set(np.unique(values)) <= {0.0, 1.0}
            assert (values[:, ds["x"].values < 4500] == 0).all()
            assert (values[:, ds["x"].values > 4500] == 1).all()
            assert mask.attrs["grid_mapping"] == "spatial_ref"
            # PISM reads the projection off crs_wkt, or the global proj attribute.
            assert "crs_wkt" in ds["spatial_ref"].attrs
            assert ds.attrs["proj"] == "EPSG:3413"


def test_the_1800m_grid_matches_what_pism_builds():
    """
    The domain builder reproduces PISM's 1800 m grid from the ISMIP7 bounds.

    The first cell centres are those PISM wrote into the 1800 m state files
    of the OCX runs (``-grid.file`` + ``-grid.dx 1800m``), so a mask on this
    grid needs no interpolation.
    """
    grid = create_domain(GRIS_X_BNDS, GRIS_Y_BNDS, 1800)
    assert grid.sizes["x"] == 960 and grid.sizes["y"] == 1560
    assert float(grid["x"][0]) == -749750.0
    assert float(grid["y"][0]) == -3411650.0


def _stage_recording(monkeypatch, tmp_path, resolution, present):
    """
    Run stage() against stubs and return the filenames it asked S3 for.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the S3 calls.
    tmp_path : pathlib.Path
        Staging directory.
    resolution : str
        Resolution handed to stage().
    present : callable
        ``present(key) -> bool``: which keys the bucket is said to hold.

    Returns
    -------
    list of str
        Basenames of the requested S3 URIs.
    """
    requested: list[str] = []

    def _record(uri, dest, **_kwargs):
        """
        Note the key instead of fetching it.

        Parameters
        ----------
        uri : str
            S3 URI requested.
        dest : str or pathlib.Path
            Where it would have been written.
        **_kwargs : dict
            Ignored.

        Returns
        -------
        pathlib.Path
            ``dest``.
        """
        requested.append(uri)
        return Path(dest)

    monkeypatch.setattr(stage_module, "download_from_s3", _record)
    monkeypatch.setattr(stage_module, "s3_key_exists", lambda _bucket, key: present(key))
    cfg = load_config(CONFIG_DIR / "ismip7_greenland_c011.toml")
    try:
        stage_module.stage(
            cfg.campaign.as_params(),
            path=tmp_path,
            include_projection=False,
            resolution=resolution,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        # The staged files are stubs, so validation downstream fails; the
        # key list is already complete by then.
        pass
    return [uri.rsplit("/", 1)[-1] for uri in requested]


def test_stage_picks_the_mask_of_the_run_resolution(tmp_path, monkeypatch, capsys):
    """
    With the 1800 m mask in the bucket, staging fetches it instead of the configured 450 m one.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Replaces the S3 calls.
    capsys : pytest.CaptureFixture
        Captures the staging report.
    """
    wanted = "pism_g1800m_frontretreat_calfin_1972_2019_MS.nc"
    staged = _stage_recording(monkeypatch, tmp_path, "1800m", lambda key: key.endswith(wanted))
    assert wanted in staged
    assert "pism_g450m_frontretreat_calfin_1972_2019_MS.nc" not in staged
    assert "WARNING: no front-retreat mask" not in capsys.readouterr().out


def test_stage_warns_and_falls_back_when_the_mask_is_missing(tmp_path, monkeypatch, capsys):
    """
    Without a mask for the run's grid, staging says so loudly and keeps the configured file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Replaces the S3 calls.
    capsys : pytest.CaptureFixture
        Captures the staging report.
    """
    staged = _stage_recording(monkeypatch, tmp_path, "2700m", lambda _key: False)
    assert "pism_g450m_frontretreat_calfin_1972_2019_MS.nc" in staged
    assert "pism_g2700m_frontretreat_calfin_1972_2019_MS.nc" not in staged
    out = capsys.readouterr().out
    assert "WARNING: no front-retreat mask for the 2700 m grid" in out
    assert "pism-ismip7-greenland-prepare --include calfin" in out
