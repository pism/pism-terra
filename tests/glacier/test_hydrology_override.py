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
Tests for the hydrology section and for UQ rows that select a model.

The glacier runner builds its PISM flags from the selected option table of
each model section. A ``[hydrology]`` table must reach the script, and a UQ
row that picks ``hydrology.model`` must swap the whole option table, not just
the one flag.
"""

from pathlib import Path

import pytest

from pism_terra.glacier.run import _render_forward_run

CONFIG = Path("pism_terra/config/s4f_perf.toml")
TEMPLATE = Path("pism_terra/templates/debug.j2")
RGI_ID = "RGI2000-v7.0-C-01-04374"


def render(tmp_path: Path, **uq) -> str:
    """
    Render one forward run script and return its text.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    **uq
        Extra ensemble overrides merged into the row.

    Returns
    -------
    str
        The rendered submission script.
    """
    _render_forward_run(
        RGI_ID,
        CONFIG,
        TEMPLATE,
        None,
        path=tmp_path,
        config_cli={"resolution": "1000m"},
        uq={"input.file": "/in/boot.nc", "grid.file": "/in/grid.nc", **uq},
        sample=0,
    )
    return next((tmp_path / RGI_ID / "run_scripts").glob("*.sh")).read_text(encoding="utf-8")


def test_hydrology_section_reaches_the_script(tmp_path: Path) -> None:
    """The config's selected hydrology model and its options are emitted.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    script = render(tmp_path)
    assert "-hydrology.model null" in script
    assert "-hydrology.null_diffuse_till_water no" in script
    assert "surface_input_from_runoff" not in script


def test_uq_row_swaps_the_hydrology_option_table(tmp_path: Path) -> None:
    """``hydrology.model`` in a UQ row selects that model's whole option table.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    script = render(tmp_path, **{"hydrology.model": "routing"})
    assert "-hydrology.model routing" in script
    assert "-hydrology.surface_input_from_runoff yes" in script
    assert "null_diffuse_till_water" not in script


def test_unknown_model_in_uq_row_is_rejected(tmp_path: Path) -> None:
    """A model with no ``[hydrology.options.*]`` table fails loudly.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    """
    with pytest.raises(ValueError, match="hydrology.model = 'distributed'"):
        render(tmp_path, **{"hydrology.model": "distributed"})
