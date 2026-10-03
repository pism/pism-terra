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

# pylint: disable=protected-access

"""
Tests for the USGS benchmark-glacier download and the CDS request cache.

No network access.
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pism_terra import download as dl


def test_download_uses_resolved_urls(tmp_path, monkeypatch):
    """
    Fetch archives by name from the item's file list and extract them next to it.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the downloads.
    """
    fetched: list[str] = []

    def fake_urls(item_id=dl.SCIENCEBASE_ITEM):  # pylint: disable=unused-argument
        """
        Stand in for the ScienceBase lookup.

        Parameters
        ----------
        item_id : str
            Ignored.

        Returns
        -------
        dict of str to str
            File-name to URL mapping.
        """
        return {
            dl.USGS_DATA_ARCHIVE: "https://example.org/data.zip",
            dl.USGS_SITES_ARCHIVE: "https://example.org/sites.zip",
        }

    def fake_download(url, dest=None, force_overwrite=False, verbose=True):  # pylint: disable=unused-argument
        """
        Record the URL and write an empty zip.

        Parameters
        ----------
        url : str
            Source URL.
        dest : Path
            Destination.
        force_overwrite : bool
            Ignored.
        verbose : bool
            Ignored.

        Returns
        -------
        Path
            The destination.
        """
        fetched.append(url)
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("placeholder.txt", "")
        return Path(dest)

    monkeypatch.setattr(dl, "sciencebase_file_urls", fake_urls)
    monkeypatch.setattr(dl, "download_archive", fake_download)

    paths = dl.download_usgs_benchmark(tmp_path)
    assert fetched == ["https://example.org/data.zip", "https://example.org/sites.zip"]
    assert (paths["data"] / "placeholder.txt").exists()
    assert paths["sites"] == tmp_path / "Glacier_Mass_Balance_Sites"

    dl.download_usgs_benchmark(tmp_path)
    assert len(fetched) == 2


def test_empty_cache_is_not_reused(tmp_path):
    """
    Reject a cached NetCDF that holds no data variables.

    A CDS download in which every year failed leaves ``xr.merge([])`` behind:
    a valid, empty dataset. It opens cleanly and carries whatever request key
    it was stamped with, so without this check it would be reused forever.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    """
    empty = tmp_path / "empty.nc"
    ds = xr.merge([])
    ds.attrs["cds_request_key"] = "abc"
    ds.to_netcdf(empty)
    # It passes every other gate, which is exactly why it needs its own.
    assert dl._cache_key_matches(empty, "abc")
    assert not dl._has_variables(empty)

    real = tmp_path / "real.nc"
    xr.Dataset(
        {"t2m": (("latitude", "longitude"), np.zeros((3, 4)))},
        coords={"latitude": np.arange(3.0), "longitude": np.arange(4.0)},
    ).to_netcdf(real)
    assert dl._has_variables(real)

    assert not dl._has_variables(tmp_path / "absent.nc")


def test_download_request_refuses_an_empty_result(tmp_path, monkeypatch):
    """
    Raise, and cache nothing, when every year of a CDS request fails.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the CDS download.
    """

    def no_downloads(*args, **kwargs):  # pylint: disable=unused-argument
        """
        Stand in for a CDS download whose every year failed.

        Parameters
        ----------
        *args : tuple
            Ignored.
        **kwargs : dict
            Ignored.

        Returns
        -------
        list
            No files, as ``_cds_download_years`` returns when it has logged
            and omitted every year.
        """
        return []

    monkeypatch.setattr(dl, "_cds_download_years", no_downloads)
    monkeypatch.setattr(dl, "_DatastoresClient", lambda *a, **k: object())

    out = tmp_path / "era5.nc"
    with pytest.raises(RuntimeError, match="no data downloaded"):
        dl.download_request(area=(62.0, -145.0, 61.0, -141.0), year=[2000, 2001], file_path=out)
    # Nothing cached, so the next run retries instead of reusing a husk.
    assert not out.exists()


def _part(tmp_path: Path, name: str, var: str, lat_noise: float) -> Path:
    """
    Write one CDS-style per-variable part with a 2-D latitude field.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory to write into.
    name : str
        File name.
    var : str
        Data variable name.
    lat_noise : float
        Offset added to the latitude field (float32 rounding stand-in).

    Returns
    -------
    pathlib.Path
        The written file.
    """
    y, x = np.arange(3.0), np.arange(4.0)
    lat = np.float32(60.0 + y[:, None] * 0.01 + x[None, :] * 0.001 + lat_noise)
    ds = xr.Dataset(
        {var: (("time", "y", "x"), np.random.default_rng(1).random((2, 3, 4)))},
        coords={"time": pd.date_range("1986-01-01", periods=2), "y": y, "x": x, "latitude": (("y", "x"), lat)},
    )
    path = tmp_path / name
    ds.to_netcdf(path)
    return path


def test_float_noise_in_shared_coordinates_does_not_block_the_merge(tmp_path):
    """
    Merge parts whose latitude differs only by float noise, but still fail on a real conflict.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    a = _part(tmp_path, "data_0.nc", "ssrd", 0.0)
    b = _part(tmp_path, "data_1.nc", "ssr", 1e-6)
    out = dl._merge_year_parts([a, b], tmp_path / "year.nc")  # pylint: disable=protected-access
    with xr.open_dataset(out) as merged:
        assert {"ssrd", "ssr"} <= set(merged.data_vars)
        assert merged.latitude.dtype.kind == "f"
    c = _part(tmp_path, "data_2.nc", "t2m", 0.5)  # half a degree is not noise
    with pytest.raises(xr.MergeError):
        dl._merge_year_parts([a, c], tmp_path / "bad.nc")  # pylint: disable=protected-access


def test_a_delivered_zip_is_finished_without_a_new_request(tmp_path):
    """
    Extract and merge a cached ``_cds_<year>.zip`` instead of resubmitting the year.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    dataset, request, year = "reanalysis-test", {"variable": ["ssrd", "ssr"]}, "1986"
    nc_path = dl._cds_year_cache_path(dataset, request, year, tmp_path)  # pylint: disable=protected-access
    parts = [_part(tmp_path, "data_0.nc", "ssrd", 0.0), _part(tmp_path, "data_1.nc", "ssr", 1e-6)]
    with zipfile.ZipFile(tmp_path / f"_cds_{year}.zip", "w") as zf:
        for p in parts:
            zf.write(p, p.name)

    class NoSubmit:
        """A client that must not be asked for anything."""

        def submit(self, *args, **kwargs):
            """
            Fail the test if called.

            Parameters
            ----------
            *args : tuple
                Ignored.
            **kwargs : dict
                Ignored.
            """
            raise AssertionError("the year should have been finished from the cached zip")

    files = dl._cds_download_years(
        NoSubmit(), dataset, request, [year], tmp_path, verbose=False
    )  # pylint: disable=protected-access
    assert files == [nc_path]
    with xr.open_dataset(nc_path) as merged:
        assert {"ssrd", "ssr"} <= set(merged.data_vars)
