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
Tests for the shared RGI outline cache.

The complex and glacier GeoPackages describe the whole project, not one
glacier, and the S4F pair is 309 MB. Staging a region's worth of glaciers used
to fetch and store them once per glacier.
"""

from pathlib import Path

import pytest

from pism_terra.glacier import stage
from pism_terra.glacier.stage import staged_rgi_outlines

CONFIG = {
    "bucket": "pism-cloud-data",
    "prefix": "glacier/input",
    "project_directory": "s4f",
    "rgi_complex_file": "s4f_c.gpkg",
    "rgi_glacier_file": "s4f_g.gpkg",
}
GLACIERS = ("RGI2000-v7.0-C-03-01124", "RGI2000-v7.0-C-03-01236", "RGI2000-v7.0-C-03-01263")


@pytest.fixture(name="downloads")
def fixture_downloads(monkeypatch):
    """
    Record every S3 fetch instead of performing one.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the download.

    Returns
    -------
    list of str
        The URIs fetched, appended to as the test runs.
    """
    fetched: list[str] = []

    def fake_download(uri, dest):
        """
        Stand in for a real download.

        Parameters
        ----------
        uri : str
            Source URI.
        dest : str or pathlib.Path
            Destination path.

        Returns
        -------
        pathlib.Path
            The written file.
        """
        fetched.append(uri)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"gpkg")
        return dest

    monkeypatch.setattr(stage, "download_from_s3", fake_download)
    return fetched


def test_one_fetch_serves_every_glacier(downloads, tmp_path):
    """
    Fetch the project outlines once, however many glaciers are staged.

    Parameters
    ----------
    downloads : list of str
        Recorded fetches.
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    for _ in GLACIERS:
        complex_file, glacier_file = staged_rgi_outlines(CONFIG, tmp_path)

    assert len(downloads) == 2
    assert complex_file == tmp_path / "s4f_c.gpkg"
    assert glacier_file == tmp_path / "s4f_g.gpkg"
    assert sorted(p.name for p in tmp_path.glob("*.gpkg")) == ["s4f_c.gpkg", "s4f_g.gpkg"]


def test_a_per_glacier_directory_refetches(downloads, tmp_path):
    """
    Pin the behaviour a shared cache avoids.

    Passing each glacier's own staging directory is still supported — it is
    the default when no cache path is given — but costs one fetch per glacier.

    Parameters
    ----------
    downloads : list of str
        Recorded fetches.
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    for rgi_id in GLACIERS:
        staged_rgi_outlines(CONFIG, tmp_path / rgi_id / "staging")

    assert len(downloads) == 2 * len(GLACIERS)


def test_the_uri_follows_the_project_prefix(downloads, tmp_path):
    """
    Fetch from the project subtree, not the shared root.

    Parameters
    ----------
    downloads : list of str
        Recorded fetches.
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    staged_rgi_outlines(CONFIG, tmp_path)

    assert downloads == [
        "s3://pism-cloud-data/glacier/input/s4f/rgi/s4f_c.gpkg",
        "s3://pism-cloud-data/glacier/input/s4f/rgi/s4f_g.gpkg",
    ]


def test_a_campaign_without_a_project_directory_uses_the_bare_prefix(downloads, tmp_path):
    """
    Leave ISMIP7/KITP-style campaigns addressing ``{prefix}/rgi`` unchanged.

    Parameters
    ----------
    downloads : list of str
        Recorded fetches.
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    staged_rgi_outlines({**CONFIG, "project_directory": None}, tmp_path)

    assert downloads[0] == "s3://pism-cloud-data/glacier/input/rgi/s4f_c.gpkg"
