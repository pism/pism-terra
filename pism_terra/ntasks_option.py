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
The ``--ntasks`` option shared by the command-line tools.

Every tool that does work in parallel takes the number of workers as
``--ntasks``. Some of them used to spell it ``--n-jobs``, ``--n-workers``,
``--workers``, ``--max-workers`` or ``--jobs``; those spellings are still
accepted, so existing scripts keep working, but are left out of the help.
"""

from __future__ import annotations

from argparse import SUPPRESS, ArgumentParser
from collections.abc import Sequence


def add_ntasks_argument(
    parser: ArgumentParser,
    *,
    default: int | None,
    help_text: str,
    dest: str = "ntasks",
    deprecated: Sequence[str] = (),
) -> None:
    """
    Add a ``--ntasks`` option, and the earlier spellings of it as hidden aliases.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.
    default : int or None
        Value when the option is not given.
    help_text : str
        Help text, saying what the workers are (processes, threads, requests).
    dest : str, default "ntasks"
        Attribute the value is stored under, so a tool can keep the name its
        code already uses.
    deprecated : sequence of str, optional
        Earlier option strings, e.g. ``("--n-jobs",)``. They set the same
        attribute and are not shown in the help.
    """
    parser.add_argument("--ntasks", dest=dest, type=int, default=default, metavar="NTASKS", help=help_text)
    for flag in deprecated:
        parser.add_argument(flag, dest=dest, type=int, default=SUPPRESS, help=SUPPRESS)
