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
Shared helpers for the ``--include`` dataset selector used by prepare CLIs.
"""

from __future__ import annotations

import logging
from argparse import ArgumentParser
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def add_include_argument(parser: ArgumentParser, available: Sequence[str]) -> None:
    """
    Add a ``--include`` option that selects which datasets to process.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.
    available : sequence of str
        Dataset names this command can process, listed in the help text.
    """
    parser.add_argument(
        "--include",
        default=None,
        metavar="DATASET[,DATASET...]",
        help=(
            "Comma-separated list of datasets to process; if omitted, all are "
            f"processed. Available: {', '.join(available)}."
        ),
    )


def select_datasets(include: str | None, available: Sequence[str]) -> list[str]:
    """
    Resolve ``--include`` to an ordered subset of ``available`` datasets.

    Parameters
    ----------
    include : str or None
        Raw ``--include`` value (comma-separated), or ``None`` for "all".
    available : sequence of str
        Canonical dataset names; the returned list preserves this order.

    Returns
    -------
    list of str
        Selected dataset names.

    Raises
    ------
    SystemExit
        If ``include`` names a dataset not in ``available``.
    """
    if not include:
        return list(available)
    requested = {s.strip() for s in include.split(",") if s.strip()}
    unknown = sorted(requested - set(available))
    if unknown:
        raise SystemExit(f"Unknown dataset(s) in --include: {', '.join(unknown)}. Available: {', '.join(available)}.")
    selected = [d for d in available if d in requested]
    logger.info("Processing datasets: %s", ", ".join(selected))
    return selected
