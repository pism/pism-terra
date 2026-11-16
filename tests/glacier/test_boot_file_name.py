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
Tests for the forcing mask in the boot file's name.

The boot file carries ``ftt_mask``, and campaigns that share an input
directory do not all use the same ``campaign.forcing_mask``. With one
``bootfile_<rgi_id>.nc`` for all of them, staging a second campaign rewrote the
file the first one's run scripts point at. These pin that each mask has its own
file, and that staging one leaves the others alone.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pytest
import xarray as xr
from shapely.geometry import box

from pism_terra.glacier import stage
from pism_terra.glacier.stage import boot_file_name, stage_glacier

RGI_ID = "RGI2000-v7.0-C-01-14094"
CRS = "EPSG:32606"
CONFIG = {
    "dem": "glo_30",
    "ice_thickness": "frank",
    "velocity": "none",
    "bathymetry": "none",
    "heatflux": "lucazeau",
    "climate": "carra2",
    "bucket": "pism-cloud-data",
    "prefix": "glacier/input",
}
#: What the stand-in boot dataset holds in ``ftt_mask`` for each mask.
MASK_VALUE = {"none": 0, "glacier": 2, "all": 1}


@pytest.mark.parametrize(
    "forcing_mask, expected",
    [
        ("all", f"bootfile_forcing_mask_all_{RGI_ID}.nc"),
        ("glacier", f"bootfile_forcing_mask_glacier_{RGI_ID}.nc"),
        ("none", f"bootfile_forcing_mask_none_{RGI_ID}.nc"),
        (None, f"bootfile_forcing_mask_none_{RGI_ID}.nc"),
    ],
)
def test_the_name_carries_the_forcing_mask(forcing_mask: str | None, expected: str):
    """
    Name the three masks apart, and an unset mask as ``none``.

    Parameters
    ----------
    forcing_mask : str or None
        The campaign's forcing mask.
    expected : str
        The boot file name it should give.
    """
    assert boot_file_name(RGI_ID, forcing_mask) == expected


def test_an_unknown_mask_is_refused():
    """Refuse a misspelt mask instead of staging it as an all-zero one."""
    with pytest.raises(ValueError, match="glaciers.*none, glacier, all"):
        boot_file_name(RGI_ID, "glaciers")


@pytest.fixture(name="staged")
def fixture_staged(tmp_path: Path, monkeypatch):
    """
    Stage a stand-in glacier without any download.

    Everything that fetches data is replaced; the file names, the writing of
    the boot file and the returned index are the real ``stage_glacier``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the downloads.

    Returns
    -------
    callable
        ``staged(**campaign)`` stages into one shared input directory with
        the given campaign keys on top of :data:`CONFIG` and returns the
        index ``stage_glacier`` hands the run generators.
    """
    outline = gpd.GeoSeries([box(450_000, 6_800_000, 453_000, 6_803_000)], crs=CRS).to_crs("EPSG:4326")
    row = {"o1region": "01", "o2region": "01-05", "utm_zone": 6, "area_km2": 9.0, "crs": CRS}
    complexes = gpd.GeoDataFrame(pd.DataFrame([{"rgi_id": RGI_ID, **row}]), geometry=outline.values, crs="EPSG:4326")
    glaciers = gpd.GeoDataFrame(
        pd.DataFrame([{"rgi_id": "RGI2000-v7.0-G-01-00001", **row, "rgi_id_c": RGI_ID, "rgi_id_c_aggregate": None}]),
        geometry=outline.values,
        crs="EPSG:4326",
    )
    pyogrio.write_dataframe(complexes, tmp_path / "c.gpkg")
    pyogrio.write_dataframe(glaciers, tmp_path / "g.gpkg")

    def fake_boot(grid_ds, rgi_id, geometries, **kwargs):  # pylint: disable=unused-argument
        """
        Stand in for the DEM and thickness build.

        Parameters
        ----------
        grid_ds : xarray.Dataset
            The target grid.
        rgi_id : str
            Glacier identifier.
        geometries : geopandas.GeoSeries
            Glacier outline.
        **kwargs
            The remaining arguments of ``boot_file_from_grid``.

        Returns
        -------
        xarray.Dataset
            An ``ftt_mask`` filled with the value of :data:`MASK_VALUE`.
        """
        shape = (grid_ds.sizes["y"], grid_ds.sizes["x"])
        mask = xr.DataArray(
            np.full(shape, MASK_VALUE[kwargs["forcing_mask"]], dtype="uint8"),
            coords={"y": grid_ds["y"].values, "x": grid_ds["x"].values},
            dims=("y", "x"),
            name="ftt_mask",
        )
        return mask.to_dataset().rio.write_crs(CRS)

    def fake_file(*args, path, **kwargs):  # pylint: disable=unused-argument
        """
        Stand in for a builder that writes ``path``.

        Parameters
        ----------
        *args
            Positional arguments of the builder.
        path : pathlib.Path
            The file the builder would write.
        **kwargs
            Keyword arguments of the builder.

        Returns
        -------
        xarray.Dataset
            An empty dataset.
        """
        Path(path).touch()
        return xr.Dataset()

    def fake_climate(name, grid_ds, *, rgi_id, path, **kwargs):  # pylint: disable=unused-argument
        """
        Stand in for the climate builder.

        Parameters
        ----------
        name : str
            Climate name.
        grid_ds : xarray.Dataset
            The target grid.
        rgi_id : str
            Glacier identifier.
        path : pathlib.Path
            Input directory.
        **kwargs
            The remaining arguments of ``_build_climate``.

        Returns
        -------
        list of pathlib.Path
            One forcing file name.
        """
        return [Path(path) / f"{name}_{rgi_id}.nc"]

    monkeypatch.setattr(stage, "staged_rgi_outlines", lambda config, cache: (tmp_path / "c.gpkg", tmp_path / "g.gpkg"))
    monkeypatch.setattr(stage, "boot_file_from_grid", fake_boot)
    monkeypatch.setattr(stage, "heatflux_from_grid", fake_file)
    monkeypatch.setattr(stage, "glacier_velocities_from_grid", fake_file)
    monkeypatch.setattr(stage, "check_xr_fully", lambda path: True)
    monkeypatch.setattr(stage, "_build_climate", fake_climate)

    def run(**campaign) -> pd.DataFrame:
        """
        Stage the glacier for one campaign.

        Parameters
        ----------
        **campaign
            Campaign keys on top of :data:`CONFIG`.

        Returns
        -------
        pandas.DataFrame
            The index ``stage_glacier`` returns.
        """
        return stage_glacier(
            {**CONFIG, **campaign},
            RGI_ID,
            path=tmp_path / "input",
            staging_path=tmp_path / "staging",
            resolution=1000.0,
        )

    return run


def _mask_value(boot_file: Path | str) -> int:
    """
    Read the value the stand-in boot file holds.

    Parameters
    ----------
    boot_file : pathlib.Path or str
        A boot file written by the ``staged`` fixture.

    Returns
    -------
    int
        The one value ``ftt_mask`` is filled with.
    """
    with xr.open_dataset(boot_file) as ds:
        values = np.unique(ds["ftt_mask"].values)
    assert values.size == 1
    return int(values[0])


def test_each_mask_is_staged_into_its_own_boot_file(staged):
    """
    Keep the boot file of one mask when another is staged next to it.

    Parameters
    ----------
    staged : callable
        Stages the stand-in glacier for a campaign.
    """
    boot_files = {mask: Path(staged(forcing_mask=mask)["boot_file"].iloc[0]) for mask in ("all", "glacier", "none")}

    assert {mask: p.name for mask, p in boot_files.items()} == {
        mask: f"bootfile_forcing_mask_{mask}_{RGI_ID}.nc" for mask in boot_files
    }
    assert len({p.parent for p in boot_files.values()}) == 1
    # All three were staged in turn into the same directory: each still
    # holds its own mask.
    assert {mask: _mask_value(p) for mask, p in boot_files.items()} == MASK_VALUE


def test_an_unset_mask_is_staged_as_none(staged):
    """
    Stage a campaign without ``forcing_mask`` into the ``none`` boot file.

    Parameters
    ----------
    staged : callable
        Stages the stand-in glacier for a campaign.
    """
    boot_file = Path(staged()["boot_file"].iloc[0])

    assert boot_file.name == f"bootfile_forcing_mask_none_{RGI_ID}.nc"
    assert _mask_value(boot_file) == MASK_VALUE["none"]


def test_an_unknown_mask_stops_staging_before_the_dem(staged, tmp_path: Path):
    """
    Fail on a misspelt mask without writing a boot file.

    Parameters
    ----------
    staged : callable
        Stages the stand-in glacier for a campaign.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    with pytest.raises(ValueError, match="forcing_mask"):
        staged(forcing_mask="glaciers")
    assert not list((tmp_path / "input").glob("bootfile_*"))
