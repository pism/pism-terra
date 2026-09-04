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
Tests for the USGS benchmark-glacier download. No network access.
"""

import zipfile
from pathlib import Path

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
