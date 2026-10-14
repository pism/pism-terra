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
Tests for the live run log of the PISM-Cloud apps.

The apps are notebooks, with no module to import, so the code under test is
lifted out of their cells. Following a log used to download all of it from
S3 every 15 s for as long as the page was open; these pin that a refresh now
reads only what the log gained, and that following stops once the job ended.
"""

from __future__ import annotations

import ast
import collections
import datetime
import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
APPS = ("pism_cloud_progress_app", "pism_cloud_ismip7_app", "pism_cloud_glacier_app")
LIFTED = ("LogFollower", "_published_ago", "_ago", "_render_log", "_schedule_log")


def app_code(app: str) -> str:
    """
    Join the code cells of an app.

    Parameters
    ----------
    app : str
        Notebook name without its suffix.

    Returns
    -------
    str
        The notebook's code.
    """
    cells = json.loads((NOTEBOOKS / f"{app}.ipynb").read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")


def segment(code: str, node: ast.AST) -> str:
    """
    Return the source text of one definition or assignment.

    Parameters
    ----------
    code : str
        The code the node was parsed from.
    node : ast.AST
        A top-level node of it.

    Returns
    -------
    str
        The node's source.
    """
    source = ast.get_source_segment(code, node)
    assert source is not None
    return source


def lifted(app: str) -> dict[str, str]:
    """
    Cut the named classes and functions out of an app, as source text.

    Parameters
    ----------
    app : str
        Notebook name without its suffix.

    Returns
    -------
    dict of str to str
        Source of each definition in :data:`LIFTED`.
    """
    code = app_code(app)
    nodes = {node.name: node for node in ast.parse(code).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    return {name: segment(code, nodes[name]) for name in LIFTED}


@pytest.fixture(name="follower_class", params=APPS)
def fixture_follower_class(request):
    """
    Build each app's ``LogFollower``, with the patterns and helpers it reads from its notebook.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Carries the app name.

    Returns
    -------
    dict
        The namespace the class and its helpers were defined in.
    """
    code = app_code(request.param)
    namespace: dict = {"collections": collections, "datetime": datetime, "re": re}
    for name in ("TIME_OPT_RE", "MODEL_DATE_RE", "STALE_AFTER"):
        (assignment,) = [
            node
            for node in ast.parse(code).body
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name
        ]
        exec(segment(code, assignment), namespace)  # pylint: disable=exec-used
    source = lifted(request.param)
    for name in ("LogFollower", "_published_ago", "_ago"):
        exec(source[name], namespace)  # pylint: disable=exec-used
    return namespace


class FakeS3:
    """
    Serve one growing object and record every byte range asked for.

    Parameters
    ----------
    content : bytes
        The object as it is now.
    """

    def __init__(self, content: bytes = b""):
        """
        Set up the stand-in.

        Parameters
        ----------
        content : bytes
            The object as it is now.
        """
        self.content = content
        self.ranges: list[tuple[int, int]] = []

    def cat_file(self, key, start=None, end=None):  # pylint: disable=unused-argument
        """
        Return a byte range of the object.

        Parameters
        ----------
        key : str
            Object key, ignored.
        start : int
            First byte.
        end : int
            One past the last byte.

        Returns
        -------
        bytes
            The requested range.
        """
        assert start is not None and end is not None, "the whole object was asked for"
        self.ranges.append((start, end))
        return self.content[start:end]


HEAD = "+ mpirun -np 24 pism -time.end 1987-01-01 -time.reference_date 1850-01-01 -time.start 1986-01-01\n"


def step(date: str, hours: float = 0.0) -> str:
    """
    Write a line the way PISM echoes a model date.

    Parameters
    ----------
    date : str
        Model date.
    hours : float, optional
        Hours into that day.

    Returns
    -------
    str
        The log line.
    """
    return f"S {date} {hours:.3f}h: dt = 12.000 h\n"


def test_the_follower_is_the_same_in_every_app():
    """
    The three notebooks cannot import shared code, so they carry copies: keep them identical.
    """
    sources = [lifted(app) for app in APPS]
    for name in ("LogFollower", "_published_ago", "_ago", "_schedule_log"):
        assert sources[0][name] == sources[1][name] == sources[2][name], name


@pytest.mark.parametrize("app", APPS)
def test_showing_the_log_never_asks_for_all_of_it(app: str):
    """
    The log view reads through the follower, not with a whole-object ``cat_file``.

    Parameters
    ----------
    app : str
        Notebook name without its suffix.
    """
    source = lifted(app)
    assert "cat_file" not in source["_render_log"]
    assert "follower.update(" in source["_render_log"]
    assert "start=self.offset" in source["LogFollower"]


def test_a_refresh_reads_only_what_the_log_gained(follower_class):
    """
    Read the whole log once, then only the appended bytes, and nothing when it has not grown.

    Parameters
    ----------
    follower_class : dict
        Namespace holding the app's ``LogFollower``.
    """
    first = HEAD + step("1986-01-01") + step("1986-04-02")
    s3 = FakeS3(first.encode())
    follower = follower_class["LogFollower"](tail_lines=2)

    follower.update(s3, "key", len(s3.content))
    assert s3.ranges == [(0, len(first))]
    assert follower.n_lines == 3
    assert follower.lines() == [step("1986-01-01").rstrip(), step("1986-04-02").rstrip()]

    follower.update(s3, "key", len(s3.content))
    assert len(s3.ranges) == 1

    s3.content += step("1986-07-02").encode()
    follower.update(s3, "key", len(s3.content))
    assert s3.ranges[-1] == (len(first), len(s3.content))
    assert follower.n_lines == 4
    assert follower.lines()[-1] == step("1986-07-02").rstrip()


def test_progress_comes_from_the_leg_bounds_and_the_latest_model_date(follower_class):
    """
    Carry the bounds seen at the start of the log through later, partial reads.

    Parameters
    ----------
    follower_class : dict
        Namespace holding the app's ``LogFollower``.
    """
    s3 = FakeS3(HEAD.encode())
    follower = follower_class["LogFollower"]()
    follower.update(s3, "key", len(s3.content))
    assert follower.progress() is None  # no model date yet

    s3.content += step("1986-07-02", 12.0).encode()
    follower.update(s3, "key", len(s3.content))
    fraction, current, start, end = follower.progress()
    assert (start, end) == (datetime.datetime(1986, 1, 1), datetime.datetime(1987, 1, 1))
    assert current == datetime.datetime(1986, 7, 2, 12)
    assert fraction == pytest.approx(182.5 / 365)


def test_a_line_cut_by_a_refresh_is_completed_by_the_next(follower_class):
    """
    Show a half-written last line, and count and parse it once it is whole.

    Parameters
    ----------
    follower_class : dict
        Namespace holding the app's ``LogFollower``.
    """
    whole = HEAD + step("1986-10-01")
    cut = len(HEAD) + 8
    s3 = FakeS3(whole.encode()[:cut])
    follower = follower_class["LogFollower"]()

    follower.update(s3, "key", len(s3.content))
    assert follower.n_lines == 1 and follower.lines()[-1] == step("1986-10-01")[:8]
    assert follower.progress() is None

    s3.content = whole.encode()
    follower.update(s3, "key", len(s3.content))
    assert follower.n_lines == 2 and follower.lines()[-1] == step("1986-10-01").rstrip()
    assert follower.progress()[1] == datetime.datetime(1986, 10, 1)


def test_a_log_that_starts_over_is_read_from_the_top(follower_class):
    """
    A rerun publishes a shorter log under the same key: forget the old one.

    Parameters
    ----------
    follower_class : dict
        Namespace holding the app's ``LogFollower``.
    """
    s3 = FakeS3((HEAD + step("1986-10-01") + step("1986-11-01")).encode())
    follower = follower_class["LogFollower"]()
    follower.update(s3, "key", len(s3.content))

    s3.content = (HEAD + step("1986-02-01")).encode()
    follower.update(s3, "key", len(s3.content))

    assert s3.ranges[-1] == (0, len(s3.content))
    assert follower.n_lines == 2
    assert follower.progress()[1] == datetime.datetime(1986, 2, 1)


def test_following_stops_for_a_log_no_longer_published(follower_class):
    """
    Tell a job that has ended by the age of its log, and stop scheduling refreshes for it.

    The executor republishes the log every 30 s while the job runs, so the
    time since the last publish says whether anyone is still writing it.

    Parameters
    ----------
    follower_class : dict
        Namespace holding the app's helpers.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    ago = follower_class["_published_ago"]
    assert ago(None) is None
    assert ago(now - datetime.timedelta(seconds=40)) == pytest.approx(40, abs=5)
    # A naive stamp is taken as UTC, which is what S3 reports.
    assert ago((now - datetime.timedelta(minutes=10)).replace(tzinfo=None)) == pytest.approx(600, abs=5)
    assert 2 * 30 < follower_class["STALE_AFTER"] < ago(now - datetime.timedelta(minutes=10))
    assert [follower_class["_ago"](s) for s in (45, 720, 12600)] == ["45 s", "12 min", "3.5 h"]

    schedule = lifted(APPS[0])["_schedule_log"]
    stop, timer = schedule.index("age > STALE_AFTER"), schedule.index("threading.Timer(")
    assert stop < schedule.index("follow_w.value = False") < schedule.index("return", stop) < timer
