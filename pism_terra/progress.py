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
Progress reporting for the analysis command-line tools.

Two kinds of waiting happen in the calibration tools: a Python loop over
glaciers, regions or bootstrap draws, and a Dask graph that reads and
reduces an ensemble. The first gets a :mod:`tqdm` bar
(:func:`progress_bar`); the second gets whichever Dask progress display
fits the scheduler in use (:func:`compute`): the distributed dashboard-style
bar when a :class:`dask.distributed.Client` is active in the process, and
the local :class:`dask.diagnostics.ProgressBar` otherwise. Both are quiet
when the output is not a terminal, so a job's log file is not filled with
carriage returns.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

import dask
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

T = TypeVar("T")

#: Seconds a local Dask compute must last before its bar is drawn, so a
#: string of sub-second reductions does not fill the terminal with bars.
BAR_MINIMUM_SECONDS = 1.0


def show_progress() -> bool:
    """
    Whether progress displays should be drawn at all.

    Returns
    -------
    bool
        ``True`` when standard error is a terminal; a log file or a batch
        job gets the plain log lines instead.
    """
    try:
        return sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def progress_bar(items: Iterable[T], desc: str | None = None, total: int | None = None, **kwargs: Any) -> Iterator[T]:
    """
    Iterate with a :mod:`tqdm` bar on a terminal, silently otherwise.

    Parameters
    ----------
    items : iterable
        What to loop over.
    desc : str or None, optional
        Label of the bar.
    total : int or None, optional
        Number of items when ``items`` has no length.
    **kwargs : Any
        Further :class:`tqdm.tqdm` options.

    Returns
    -------
    iterator
        The items, one by one.
    """
    return tqdm(items, desc=desc, total=total, disable=not show_progress(), leave=False, **kwargs)


def distributed_client():
    """
    The active :class:`dask.distributed.Client`, if the process has one.

    Returns
    -------
    dask.distributed.Client or None
        The client, or ``None`` when Dask runs on its local scheduler (or
        ``distributed`` is not installed).
    """
    try:
        # pylint: disable-next=import-outside-toplevel
        from dask.distributed import get_client
    except ImportError:
        return None
    try:
        return get_client()
    except ValueError:
        return None


def compute(*objects: Any, desc: str | None = None) -> tuple:
    """
    Compute Dask-backed objects with a progress display fitting the scheduler.

    With a distributed client the graph is persisted on the cluster and
    :func:`dask.distributed.progress` follows the tasks; on the local
    scheduler :class:`dask.diagnostics.ProgressBar` does, once the compute
    has run for :data:`BAR_MINIMUM_SECONDS`. Objects that are not
    Dask-backed pass through :func:`dask.compute` unchanged, and when none
    of them is lazy nothing is announced or drawn: there is nothing to wait
    for.

    Parameters
    ----------
    *objects : Any
        Lazy xarray objects, Dask collections or plain values.
    desc : str or None, optional
        One line printed before the bar, so a run with several computes
        says which one it is at.

    Returns
    -------
    tuple
        The computed objects, in order, as :func:`dask.compute` returns them.
    """
    if not show_progress() or not any(dask.is_dask_collection(o) for o in objects):
        return dask.compute(*objects)
    if desc:
        print(desc, file=sys.stderr, flush=True)
    client = distributed_client()
    if client is not None:
        # pylint: disable-next=import-outside-toplevel
        from dask.distributed import progress

        persisted = client.persist(objects)
        progress(persisted)
        print(file=sys.stderr)  # the distributed bar leaves the cursor on its line
        return dask.compute(*persisted)
    with ProgressBar(minimum=BAR_MINIMUM_SECONDS, out=sys.stderr):
        return dask.compute(*objects)


__all__ = ("compute", "distributed_client", "progress_bar", "show_progress")
