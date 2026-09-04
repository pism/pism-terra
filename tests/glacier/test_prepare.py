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
Tests for the unified glacier prepare command.

The layout is the contract: products that depend on a project's CRS overrides
have to land under ``input/<project>``, and the global ones beside it, or two
projects silently overwrite each other's files.
"""

import pandas as pd
import pytest

from pism_terra.aws import project_prefix
from pism_terra.glacier import prepare as prepare_mod
from pism_terra.glacier.prepare import (
    PREPARE_DATASETS,
    prepare,
    prepare_paths,
    read_glacier_groups,
)

SETUP_TOML = """
[staging]
project_directory = "s4f"

[regions]
1 = {name = "alaska", crs = "EPSG:5936"}
3 = {name = "arctic_canada_north", crs = "EPSG:3413"}
"""


@pytest.fixture(name="setup_file")
def fixture_setup_file(tmp_path):
    """
    Write a minimal S4F-style setup TOML.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.

    Returns
    -------
    pathlib.Path
        Path to the written TOML.
    """
    path = tmp_path / "setup.toml"
    path.write_text(SETUP_TOML, encoding="utf-8")
    return path


def test_project_specific_products_live_under_the_project(tmp_path):
    """
    Put the CRS-dependent products under the project subdirectory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    paths = prepare_paths(tmp_path, "s4f")
    project = tmp_path / "input" / "s4f"

    for key in ("rgi", "ice_thickness", "ice_thickness_frank", "ice_thickness_maffezzoli", "project_climate"):
        assert project in paths[key].parents or paths[key] == project


def test_shared_products_live_beside_the_project(tmp_path):
    """
    Keep the global products out of any project subdirectory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    paths = prepare_paths(tmp_path, "s4f")

    assert paths["gebco"] == tmp_path / "input" / "gebco"
    assert paths["heatflux"] == tmp_path / "input" / "heatflux"
    # SNAP and the merged CARRA2 store are global; only the per-group CARRA2
    # files depend on a project's CRS.
    assert paths["climate"] == tmp_path / "input" / "climate"
    assert paths["staging_snap"] == tmp_path / "staging" / "snap"
    # Staging is shared too: the raw downloads do not depend on the project.
    assert paths["staging"] == tmp_path / "staging"
    assert paths["staging_carra2"] == tmp_path / "staging" / "carra2"


def test_two_projects_do_not_collide(tmp_path):
    """
    Give each project its own ice-thickness tree.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    rgi_paths = prepare_paths(tmp_path, "rgi")
    s4f_paths = prepare_paths(tmp_path, "s4f")

    assert rgi_paths["ice_thickness_frank"] != s4f_paths["ice_thickness_frank"]
    # ...but only one copy of the global data.
    assert rgi_paths["gebco"] == s4f_paths["gebco"]
    assert rgi_paths["staging"] == s4f_paths["staging"]


def test_prepare_paths_creates_nothing(tmp_path):
    """
    Build paths without touching the filesystem.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    prepare_paths(tmp_path, "s4f")

    assert not (tmp_path / "input").exists()


def test_glacier_files_become_named_groups(tmp_path):
    """
    Turn ``S4F_target_AK_RGI_id.csv`` into the group ``S4F_AK``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    csv = tmp_path / "S4F_target_AK_RGI_id.csv"
    csv.write_text("rgi_id\nRGI2000-v7.0-G-01-00001\nRGI2000-v7.0-G-03-00002\n", encoding="utf-8")

    groups, glaciers = read_glacier_groups([csv])

    assert list(groups) == ["S4F_AK"]
    assert glaciers is not None
    assert sorted(glaciers["o1regions"]) == ["01", "03"]


def test_unconventional_filename_falls_back_to_the_stem(tmp_path):
    """
    Name a group after the file when it does not match the convention.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    csv = tmp_path / "my_glaciers.csv"
    csv.write_text("rgi_id\nRGI2000-v7.0-G-01-00001\n", encoding="utf-8")

    groups, _ = read_glacier_groups([csv])

    assert list(groups) == ["my_glaciers"]


def test_no_glacier_files_means_no_groups():
    """
    Cover whole regions when no glacier list is given.
    """
    groups, glaciers = read_glacier_groups([])

    assert not groups
    assert glaciers is None


def test_missing_project_directory_is_rejected(tmp_path):
    """
    Refuse a setup TOML that does not say where its data belongs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    setup = tmp_path / "setup.toml"
    setup.write_text('[regions]\n1 = {name = "alaska"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="project_directory"):
        prepare([str(setup), str(tmp_path / "out")])


def test_prepare_writes_the_project_layout(monkeypatch, tmp_path, setup_file):
    """
    Run the command end to end with the download step stubbed out.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the network-bound RGI step.
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    setup_file : pathlib.Path
        Minimal setup TOML from the fixture.
    """
    seen = {}

    def fake_prepare_rgi(regions, **kwargs):
        """
        Record the call and write empty GeoPackages.

        Parameters
        ----------
        regions : pandas.DataFrame
            Regions table built from the setup TOML.
        **kwargs
            Everything the real function takes.

        Returns
        -------
        dict
            The two output paths, as the real function returns.
        """
        seen["regions"] = regions
        seen["kwargs"] = kwargs
        out = kwargs["output_path"]
        files = {
            "rgi_complexes": out / f"{kwargs['name_prefix']}_c.gpkg",
            "rgi_glaciers": out / f"{kwargs['name_prefix']}_g.gpkg",
        }
        for path in files.values():
            path.touch()
        return files

    monkeypatch.setattr(prepare_mod, "prepare_rgi", fake_prepare_rgi)
    out_path = tmp_path / "glacier_input"

    rgi_files = prepare(["--include", "rgi", str(setup_file), str(out_path)])

    assert rgi_files["rgi_complexes"] == out_path / "input" / "s4f" / "rgi" / "s4f_c.gpkg"
    assert rgi_files["rgi_glaciers"].exists()
    assert seen["kwargs"]["name_prefix"] == "s4f"
    # No glacier CSVs, so the run covers whole regions.
    assert seen["kwargs"]["glaciers"] is None
    assert seen["kwargs"]["glacier_groups"] is None
    assert sorted(seen["regions"]["region"]) == ["01_alaska", "03_arctic_canada_north"]
    # Nothing else was selected, so no other product directory exists.
    assert not (out_path / "input" / "gebco").exists()


def test_glacier_files_scope_the_regions(monkeypatch, tmp_path, setup_file):
    """
    Drop regions no listed glacier falls in.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the network-bound RGI step.
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    setup_file : pathlib.Path
        Minimal setup TOML from the fixture.
    """
    seen = {}

    def fake_prepare_rgi(regions, **kwargs):
        """
        Record the call and write empty GeoPackages.

        Parameters
        ----------
        regions : pandas.DataFrame
            Regions table built from the setup TOML.
        **kwargs
            Everything the real function takes.

        Returns
        -------
        dict
            The two output paths, as the real function returns.
        """
        seen["regions"] = regions
        seen["kwargs"] = kwargs
        out = kwargs["output_path"]
        files = {
            "rgi_complexes": out / f"{kwargs['name_prefix']}_c.gpkg",
            "rgi_glaciers": out / f"{kwargs['name_prefix']}_g.gpkg",
        }
        for path in files.values():
            path.touch()
        return files

    monkeypatch.setattr(prepare_mod, "prepare_rgi", fake_prepare_rgi)
    csv = tmp_path / "S4F_target_AK_RGI_id.csv"
    csv.write_text("rgi_id\nRGI2000-v7.0-G-01-00001\n", encoding="utf-8")

    prepare(["--include", "rgi", str(setup_file), str(tmp_path / "out"), str(csv)])

    assert list(seen["regions"]["region"]) == ["01_alaska"]
    assert list(seen["kwargs"]["glacier_groups"]) == ["S4F_AK"]
    assert isinstance(seen["kwargs"]["glaciers"], pd.DataFrame)


def test_dataset_list_is_the_execution_order():
    """
    Keep the selector list in step with what the body actually runs.

    The order is the order the datasets are prepared in, and it is what
    ``--include`` advertises. GlacierMIP4 was dropped; its archive is only
    mirrored into staging and nothing publishes it.
    """
    assert PREPARE_DATASETS == [
        "rgi",
        "ice_thickness_frank",
        "ice_thickness_maffezzoli",
        "dh_hugonnet",
        "gebco",
        "heatflux_lucazeau",
        "snap",
        "carra2",
    ]


def test_dropped_datasets_are_rejected(tmp_path, setup_file):
    """
    Fail loudly on a dataset the command no longer prepares.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    setup_file : pathlib.Path
        Minimal setup TOML from the fixture.
    """
    with pytest.raises(SystemExit):
        prepare(["--include", "glaciermip4", str(setup_file), str(tmp_path / "out")])


@pytest.mark.parametrize(
    "prefix,project,expected",
    [
        ("glacier/input", "s4f", "glacier/input/s4f"),
        ("glacier/input", "rgi", "glacier/input/rgi"),
        # Campaigns whose input tree is not split by project are unaffected.
        ("kitp/input", None, "kitp/input"),
        ("kitp/input", "", "kitp/input"),
        ("glacier/input/", "/s4f/", "glacier/input/s4f"),
        (None, "s4f", "s4f"),
    ],
)
def test_project_prefix(prefix, project, expected):
    """
    Join the shared prefix with the project subdirectory.

    Parameters
    ----------
    prefix : str or None
        Shared S3 key prefix.
    project : str or None
        Project subdirectory.
    expected : str
        Expected joined prefix.
    """
    assert project_prefix(prefix, project) == expected
