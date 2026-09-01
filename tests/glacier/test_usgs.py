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
Tests for the pieces the USGS benchmark tools share.

Reading the release, placing its glaciers in RGI outlines, and the time
bookkeeping that turns PISM's reporting intervals into calendar dates.
"""

from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pism_terra.glacier import usgs

GLACIER_A = "RGI2000-v7.0-G-01-00001"
GLACIER_B = "RGI2000-v7.0-G-01-00002"


def test_sites_keep_only_stakes(usgs_release):
    """
    Weather stations are dropped and the points carry the glacier name.

    Parameters
    ----------
    usgs_release : dict of str to Path
        The synthetic release directories.
    """
    sites = usgs.load_sites(usgs_release["sites"])
    assert sites["Type"].eq("Glaciological").all()
    assert sites["Glacier"].tolist() == ["Foo", "Foo", "Foo", "Bar", "Baz"]
    assert sites.crs.to_epsg() == 4326


def test_match_takes_the_modal_outline(usgs_release, usgs_rgi):
    """
    Two of Foo's three stakes sit in A, so A wins; the count is reported.

    Parameters
    ----------
    usgs_release : dict of str to Path
        The synthetic release directories.
    usgs_rgi : geopandas.GeoDataFrame
        The two synthetic outlines.
    """
    matches = usgs.match_rgi_ids(usgs.load_sites(usgs_release["sites"]), usgs_rgi).set_index("glacier")
    assert matches.loc["Foo", "rgi_id"] == GLACIER_A
    assert matches.loc["Foo", "n_sites"] == 3
    assert matches.loc["Foo", "n_matched"] == 2
    assert matches.loc["Foo", "area_km2"] == 10.0


def test_match_falls_back_to_nearest(usgs_release, usgs_rgi):
    """
    Fall back to the nearer outline when no stake lies inside any, within a distance cap.

    Bar's stake is a kilometre north of outline B; Baz's is more than a
    hundred kilometres from either outline and stays unmatched.

    Parameters
    ----------
    usgs_release : dict of str to Path
        The synthetic release directories.
    usgs_rgi : geopandas.GeoDataFrame
        The two synthetic outlines.
    """
    matches = usgs.match_rgi_ids(usgs.load_sites(usgs_release["sites"]), usgs_rgi).set_index("glacier")
    assert matches.loc["Bar", "rgi_id"] == GLACIER_B
    assert matches.loc["Bar", "n_matched"] == 0
    assert pd.isna(matches.loc["Baz", "rgi_id"])
    assert pd.isna(matches.loc["Baz", "area_km2"])


def test_glacier_wide_area_is_carried_forward(usgs_release):
    """
    The year without an area-altitude distribution keeps the last known area.

    Parameters
    ----------
    usgs_release : dict of str to Path
        The synthetic release directories.
    """
    obs = usgs.load_glacier_wide(usgs_release["data"], "Foo")
    assert obs is not None
    assert obs["time"].values.tolist() == [2000, 2001, 2002]
    np.testing.assert_allclose(obs["area"].values, [10.0, 8.0, 8.0])
    assert obs["Ba"].attrs["units"] == "m year^-1"
    assert "Ba_unc" in obs
    assert np.isnan(obs["Bw"].values[1])


def test_glacier_wide_missing_for_point_only_glacier(usgs_release):
    """
    A glacier without a glacier-wide solution file yields None.

    Parameters
    ----------
    usgs_release : dict of str to Path
        The synthetic release directories.
    """
    assert usgs.load_glacier_wide(usgs_release["data"], "Bar") is None


def test_mean_years_handles_bounds_and_new_year_stamps():
    """
    Interval starts come from ``time_bounds``; a 1 January stamp closes the previous year.
    """
    times = pd.to_datetime(["2001-01-01", "2002-01-01"])
    ds = xr.Dataset(coords={"time": ("time", times)})
    assert usgs.mean_years(ds).tolist() == [2000, 2001]

    bounds = np.array([pd.to_datetime(["1990-01-01", "1991-01-01"]), pd.to_datetime(["1991-01-01", "1992-01-01"])])
    ds = ds.assign(time_bounds=(("time", "nv"), bounds))
    ds["time"].attrs["bounds"] = "time_bounds"
    assert usgs.mean_years(ds).tolist() == [1990, 1991]


def test_interval_edges_prefer_decoded_bounds():
    """
    Decoded ``time_bounds`` give the exact interval; without them the calendar month is assumed.
    """
    stamps = [cftime.DatetimeGregorian(2000, 1, 16, 12), cftime.DatetimeGregorian(2000, 2, 15)]
    bounds = np.array(
        [
            [cftime.DatetimeGregorian(2000, 1, 1), cftime.DatetimeGregorian(2000, 2, 1)],
            [cftime.DatetimeGregorian(2000, 2, 1), cftime.DatetimeGregorian(2000, 3, 1)],
        ]
    )
    ds = xr.Dataset(
        {"time_bounds": (("time", "nv"), bounds)}, coords={"time": ("time", stamps, {"bounds": "time_bounds"})}
    )
    starts, ends = usgs.interval_edges(ds)
    assert list(starts) == [pd.Timestamp("2000-01-01"), pd.Timestamp("2000-02-01")]
    assert list(ends) == [pd.Timestamp("2000-02-01"), pd.Timestamp("2000-03-01")]

    # Undecoded (numeric) bounds are no use; fall back to the stamps.
    raw = ds.assign(time_bounds=(("time", "nv"), np.zeros((2, 2))))
    starts_raw, ends_raw = usgs.interval_edges(raw)
    assert list(starts_raw) == list(starts) and list(ends_raw) == list(ends)
    assert usgs.is_monthly(np.array(stamps))


def test_run_label_keeps_ensemble_members_apart():
    """
    ``id_<n>`` and ``uq_<m>`` tokens both make it into the label.
    """
    root = Path("/runs")
    base = root / "exp" / "output" / "spatial"
    assert usgs.run_label(base / "spatial_g100m_X_id_0_1986-01-01_2025-01-01.nc", root) == "exp id_0"
    assert usgs.run_label(base / "spatial_g100m_X_id_0_uq_3_1986-01-01_2025-01-01.nc", root) == "exp id_0 uq_3"
    assert usgs.run_label(base / "other.nc", root) == "exp/other.nc"
    assert usgs.run_label(base / "other.nc", None) == "other.nc"


def test_score_on_plain_arrays():
    """
    Score plain arrays like aligned DataArrays; drop NaNs and withhold r below three points.
    """
    full = usgs.score(np.array([1.0, 2.0, 3.0, np.nan]), np.array([1.5, 2.5, 3.5, 9.0]))
    assert full["n"] == 3 and full["r"] == pytest.approx(1.0) and full["bias"] == pytest.approx(0.5)
    short = usgs.score(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
    assert short["n"] == 2 and np.isnan(short["r"]) and short["mae"] == pytest.approx(0.5)
    assert usgs.score(np.array([np.nan]), np.array([1.0]))["n"] == 0
