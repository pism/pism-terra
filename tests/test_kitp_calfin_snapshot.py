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
Tests for the single-record CalFin masks of the KITP runs.

KITP runs on a 365_day axis from year 1, so the mask holds one observed
front (a summer month of 2007) in a single record bounded by the run.
"""

from __future__ import annotations

from pathlib import Path

import cftime
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from pism_terra.config import load_config
from pism_terra.ismip7.greenland import forcing as forcing_module
from pism_terra.ismip7.greenland.forcing import (
    calfin_snapshot_filename,
    prepare_calfin_snapshot,
)
from pism_terra.ismip7.greenland.stage import retreat_file_for_resolution

CONFIG_DIR = Path(__file__).resolve().parents[1] / "pism_terra" / "config"

X_BNDS = [0.0, 36000.0]
Y_BNDS = [0.0, 18000.0]


def _fake_calfin(_tmp_path, crs="EPSG:3413"):
    """
    Two fronts, one in 2007 and one in 2010, inside a box outline.

    Parameters
    ----------
    _tmp_path : Path
        Unused download directory.
    crs : str
        CRS of the returned frames.

    Returns
    -------
    tuple
        ``(calfin, outline)`` as :func:`forcing_module._load_calfin` returns them.
    """
    calfin = gpd.GeoDataFrame(
        {"Date": pd.DatetimeIndex(["2007-07-12", "2010-08-01"])},
        geometry=[box(0, 0, 9000, 18000), box(9000, 0, 18000, 18000)],
        crs=crs,
    ).set_index("Date")
    outline = gpd.GeoSeries([box(X_BNDS[0], Y_BNDS[0], X_BNDS[1], Y_BNDS[1])], crs=crs)
    return calfin, outline


def test_kitp_config_names_the_snapshot_file():
    """
    The KITP campaign's retreat file is what the prepare step writes.
    """
    cfg = load_config(CONFIG_DIR / "kitp_greenland.toml")
    name = cfg.campaign.as_params()["retreat_file"]
    assert name == calfin_snapshot_filename(450, 2007, "0001-01-01", "1001-01-01")
    assert retreat_file_for_resolution(name, "1200m") == calfin_snapshot_filename(
        1200, 2007, "0001-01-01", "1001-01-01"
    )


def test_snapshot_is_one_record_bounded_by_the_run(tmp_path, monkeypatch):
    """
    One record at year 1, bounds to year 1001 on a 365_day axis, with only the 2007 front cut out.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Replaces the CALFIN download.
    """
    monkeypatch.setattr(forcing_module, "_load_calfin", _fake_calfin)
    files = prepare_calfin_snapshot(tmp_path / "input", [1500, 3000], X_BNDS, Y_BNDS, date="2007-07")
    assert sorted(files) == [1500, 3000]
    for res, path in files.items():
        assert path.name == calfin_snapshot_filename(res, 2007, "0001-01-01", "1001-01-01")
        with xr.open_dataset(path, decode_times=False) as ds:
            assert ds.sizes["time"] == 1
            assert ds["time"].attrs["calendar"] == "365_day"
            assert ds["time"].attrs["units"] == "days since 0001-01-01"
            np.testing.assert_array_equal(ds["time_bnds"].values, [[0.0, 365000.0]])
            mask = ds["land_ice_area_fraction_retreat"].isel(time=0).values
            x = ds["x"].values
            first = cftime.num2date(ds["time"].values[0], ds["time"].attrs["units"], ds["time"].attrs["calendar"])
        assert set(np.unique(mask)) <= {0.0, 1.0}
        # The 2007 front (x < 9 km) is open water; the 2010 one is not yet.
        assert (mask[:, x < 9000] == 0).all()
        assert (mask[:, x > 9000] == 1).all()
        assert (first.year, first.month, first.day) == (1, 1, 1)
