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
Tests for :func:`pism_terra.workflow.check_template_legs`.

The runners fill different template slots — the glacier ones a single
``run_str``, the ISMIP7 ones ``run_hist_str``/``run_proj_str`` — and Jinja
drops a variable the template never mentions. Rendering the wrong pairing
produced a job script with its init leg and nothing else, discovered only
after the job had been queued and come back empty.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pism_terra.workflow import check_template_legs

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "pism_terra" / "templates"


def test_a_glacier_template_is_refused_for_an_ismip7_run():
    """
    The pairing that silently lost the forward leg is now an error.
    """
    params = {"run_init_str": "pism -init", "run_hist_str": "pism -hist", "run_proj_str": ""}
    with pytest.raises(SystemExit, match="would drop run_hist_str"):
        check_template_legs(TEMPLATE_DIR / "chinook-apptainer-async.j2", params)


def test_the_matching_ismip7_template_passes():
    """
    The template written for this runner declares the slots it is handed.
    """
    params = {"run_init_str": "pism -init", "run_hist_str": "pism -hist", "run_proj_str": "pism -proj"}
    check_template_legs(TEMPLATE_DIR / "chinook-apptainer-ismip7-async.j2", params)


def test_an_empty_leg_is_not_required():
    """
    A template may omit a leg this run does not have.

    A single-leg run leaves ``run_proj_str`` empty, and the glacier templates
    have no such slot; that is not a mismatch.
    """
    params = {"run_init_str": "pism -init", "run_str": "pism -fwd", "run_proj_str": ""}
    check_template_legs(TEMPLATE_DIR / "chinook-apptainer-async.j2", params)


def test_whitespace_only_counts_as_empty(tmp_path: Path):
    """
    A slot holding only blanks contributes no run, so it is not required.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    template = tmp_path / "bare.j2"
    template.write_text("#!/bin/bash\n{{ run_init_str }}\n", encoding="utf-8")
    check_template_legs(template, {"run_init_str": "pism -init", "run_str": "   \n  "})


def test_the_message_names_what_the_template_does_offer(tmp_path: Path):
    """
    Say which slots the template has, so the fix is obvious.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    template = tmp_path / "none.j2"
    template.write_text("#!/bin/bash\necho nothing\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no run slots"):
        check_template_legs(template, {"run_str": "pism -fwd"})
