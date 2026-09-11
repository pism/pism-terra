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
