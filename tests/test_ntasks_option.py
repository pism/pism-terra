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
Tests for the shared ``--ntasks`` option.

The tools spelled the number of parallel workers five different ways. They
all take ``--ntasks`` now; these pin that, and that the old spellings still
work without being advertised.
"""

from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from importlib.metadata import entry_points

import pytest

from pism_terra.ntasks_option import add_ntasks_argument

#: Console scripts that work in parallel, and the spelling each used before.
TOOLS = {
    "pism-glacier-sensitivity-indices": "--n-jobs",
    "pism-sensitivity-indices": "--n-jobs",
    "pism-s4f-report": "--n-jobs",
    "pism-glacier-usgs-benchmark-glaciers": "--n-jobs",
    "pism-glacier-usgs-benchmark-stakes": "--n-jobs",
    "pism-ismip7-greenland-mass-balance": "--n-workers",
    "pism-validate": "--workers",
    "pism-s4f-carra2": "--max-workers",
}


def parser(**kwargs) -> ArgumentParser:
    """
    Build a parser with the option under test.

    Parameters
    ----------
    **kwargs
        Passed to :func:`add_ntasks_argument`.

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """
    result = ArgumentParser(prog="tool", formatter_class=ArgumentDefaultsHelpFormatter)
    add_ntasks_argument(result, **kwargs)
    return result


def test_ntasks_and_the_old_spelling_set_the_same_value():
    """
    Either spelling sets the attribute the tool reads, and the default holds without them.
    """
    tool = parser(default=4, help_text="Worker processes.", dest="n_jobs", deprecated=("--n-jobs",))

    assert tool.parse_args([]).n_jobs == 4
    assert tool.parse_args(["--ntasks", "8"]).n_jobs == 8
    assert tool.parse_args(["--n-jobs", "2"]).n_jobs == 2
    # A default of None, for "all CPUs", survives the hidden alias.
    assert parser(default=None, help_text="x", deprecated=("--n-jobs",)).parse_args([]).ntasks is None


def test_only_ntasks_is_advertised():
    """
    The help shows ``--ntasks`` with its default and none of the old spellings.
    """
    text = parser(default=4, help_text="Worker processes.", dest="n_jobs", deprecated=("--n-jobs", "-j")).format_help()

    # Shown the same way in every tool, whatever attribute the value is stored under.
    assert "--ntasks NTASKS" in text and "Worker processes. (default: 4)" in text
    assert "--n-jobs" not in text and "-j" not in text.replace("--ntasks", "")


@pytest.mark.parametrize("script", TOOLS)
def test_every_parallel_tool_takes_ntasks(script: str, capsys):
    """
    Each console script lists ``--ntasks`` in its help and not its earlier spelling.

    Parameters
    ----------
    script : str
        Console script name.
    capsys : pytest.CaptureFixture
        Captures the help text.
    """
    (entry,) = [ep for ep in entry_points(group="console_scripts") if ep.name == script]
    with pytest.raises(SystemExit) as stop:
        entry.load()(["--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out

    assert "--ntasks" in text
    assert TOOLS[script] not in text
