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
Tests for the files the documentation tells readers to use.

The tutorials and feature pages name configs, UQ files and templates under
``pism_terra/``. Clearing out old configs once left eight pages pointing at
files that were gone. The configs the docs need are kept as
``pism_terra/config/docs_*.toml``; these pin that every path the docs name
exists, and that every docs config still renders a run script.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pism_terra.glacier.run import _render_forward_run, _render_inverse_run

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "source"
#: A path into the package as the docs write it, e.g. ``pism_terra/config/x.toml``.
PACKAGE_PATH = re.compile(r"pism_terra/[\w/.-]+\.(?:toml|j2|gpkg|csv|cdl|nc)\b")
DOCS_CONFIGS = sorted((ROOT / "pism_terra" / "config").glob("docs_*.toml"))
RGI_ID = "RGI2000-v7.0-C-01-04374"
#: What staging would hand the run generators.
STAGED = {
    "input.file": "/in/boot.nc",
    "grid.file": "/in/grid.nc",
    "atmosphere.given.file": "/in/climate.nc",
    "atmosphere.elevation_change.file": "/in/climate.nc",
    "energy.bedrock_thermal.file": "/in/bheatflux.nc",
    "surface.pdd.std_dev.file": "/in/climate.nc",
    "surface.force_to_thickness.file": "/in/boot.nc",
    "inverse.file": "/in/obs.nc",
}


def _docs_pages() -> list[Path]:
    """
    List the pages of the documentation source.

    Returns
    -------
    list of pathlib.Path
        Markdown and reStructuredText pages, without build output, generated
        galleries or editor backups.
    """
    return [
        page
        for page in sorted(DOCS.rglob("*"))
        if page.suffix in (".md", ".rst")
        and not {"_build", "auto_examples"} & set(page.relative_to(DOCS).parts)
        and not page.name.startswith((".", "#"))
    ]


def test_every_package_file_the_docs_name_exists():
    """Name only configs, UQ files and templates that are in the package."""
    named = {
        (page.relative_to(DOCS).as_posix(), path)
        for page in _docs_pages()
        for path in PACKAGE_PATH.findall(page.read_text(encoding="utf-8"))
    }
    assert named, "no package paths found in the docs; has the pattern gone stale?"
    missing = sorted(f"{page}: {path}" for page, path in named if not (ROOT / path).exists())
    assert not missing, "\n".join(missing)


def test_there_are_docs_configs():
    """Find the docs configs this module renders."""
    assert DOCS_CONFIGS


@pytest.mark.parametrize("config", DOCS_CONFIGS, ids=[config.stem for config in DOCS_CONFIGS])
def test_every_docs_config_renders_a_run(config: Path, tmp_path: Path):
    """
    Render a run script from each docs config, with the inputs staging would provide.

    Parameters
    ----------
    config : pathlib.Path
        A ``docs_*.toml`` config.
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    inverse = "inverse" in config.stem
    render = _render_inverse_run if inverse else _render_forward_run
    render(
        RGI_ID,
        config,
        ROOT / "pism_terra" / "templates" / "debug.j2",
        None,
        path=tmp_path,
        config_cli={"resolution": "500m"},
        uq=STAGED,
        sample=0,
        init_climate_file="/in/climate_monthly_mean.nc",
    )
    script = next((tmp_path / RGI_ID / "run_scripts").glob("*.sh")).read_text(encoding="utf-8")
    assert "-input.file /in/boot.nc" in script
    if inverse:
        assert re.search(r"(?:^|\s)pismi ", script)
