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
Tests for the live run log in :mod:`pism_terra.glacier.execute`.

The log is what the notebook shows people who have no AWS credentials, so
what matters is that it is written *while* the script runs rather than at the
end, that stdout still carries everything for the CloudWatch stream, that a
failing script still publishes, and that a broken S3 never takes the
simulation down with it.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from pism_terra.glacier.execute import ProgressPublisher, execute


def write_script(path: Path, body: str) -> Path:
    """
    Write an executable shell script.

    Parameters
    ----------
    path : pathlib.Path
        Where to write it.
    body : str
        Script body, without the shebang.

    Returns
    -------
    pathlib.Path
        The script.
    """
    path.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_execute_tees_to_the_log_and_to_stdout(tmp_path: Path, capfd):
    """
    Everything the script prints reaches both the log and stdout.

    Whoever has AWS credentials keeps the complete CloudWatch stream; the
    log is an addition, not a diversion.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    capfd : pytest.CaptureFixture
        File-descriptor capture (the output comes from a child process).
    """
    script = write_script(tmp_path / "run.sh", "echo hello-from-pism")
    log = tmp_path / "logs" / "progress.log"

    execute(script, log_file=log)

    assert "hello-from-pism" in log.read_text(encoding="utf-8")
    assert "hello-from-pism" in capfd.readouterr().out


def test_execute_writes_the_log_while_the_script_is_still_running(tmp_path: Path, monkeypatch):
    """
    The log grows during the run, not only at the end.

    A log that only appears on exit is the thing we already had.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to record each publish.
    """
    script = write_script(tmp_path / "run.sh", "echo first-leg\nsleep 2\necho second-leg")
    log = tmp_path / "logs" / "progress.log"
    seen: list[str] = []

    publisher = ProgressPublisher(log, "bucket", "key", interval=0.2)
    monkeypatch.setattr(
        publisher, "publish", lambda: seen.append(log.read_text(encoding="utf-8") if log.exists() else "")
    )

    with publisher:
        execute(script, log_file=log)

    mid = [s for s in seen if "first-leg" in s and "second-leg" not in s]
    assert mid, f"log never held only the first leg; saw {seen}"


def test_execute_raises_on_failure_but_the_log_survives(tmp_path: Path, monkeypatch):
    """
    A failing script still leaves its output behind.

    A failed run is exactly when someone wants to read the log, and the
    publisher's final write happens on the way out of the block.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to record each publish.
    """
    script = write_script(tmp_path / "run.sh", "echo got-this-far\nexit 3")
    log = tmp_path / "logs" / "progress.log"
    published: list[str] = []

    publisher = ProgressPublisher(log, "bucket", "key", interval=60.0)
    monkeypatch.setattr(publisher, "publish", lambda: published.append(log.read_text(encoding="utf-8")))

    with pytest.raises(subprocess.CalledProcessError):
        with publisher:
            execute(script, log_file=log)

    assert "got-this-far" in log.read_text(encoding="utf-8")
    assert published and "got-this-far" in published[-1]


def test_a_broken_s3_does_not_stop_the_run(tmp_path: Path, capfd):
    """
    Publishing is best-effort.

    The publisher runs alongside the simulation; an S3 outage must cost the
    live view and nothing else.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    capfd : pytest.CaptureFixture
        File-descriptor capture.
    """
    log = tmp_path / "logs" / "progress.log"
    log.parent.mkdir(parents=True)
    log.write_text("some output\n", encoding="utf-8")

    publisher = ProgressPublisher(log, "bucket", "key", interval=0.1)

    def _boom(**_kwargs):
        """
        Stand in for an S3 that is refusing writes.

        Parameters
        ----------
        **_kwargs : dict
            Put_object arguments, ignored.
        """
        raise RuntimeError("no such bucket")

    publisher._s3.put_object = _boom  # pylint: disable=protected-access

    with publisher:
        time.sleep(0.3)

    assert "no such bucket" in capfd.readouterr().err


def test_a_missing_log_file_is_not_an_error(tmp_path: Path):
    """
    Publishing before the first line is written does nothing quietly.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    publisher = ProgressPublisher(tmp_path / "never-written.log", "bucket", "key")

    def _unexpected(**_kwargs):
        """
        Fail the test if an upload is attempted at all.

        Parameters
        ----------
        **_kwargs : dict
            Put_object arguments, ignored.
        """
        raise AssertionError("should not upload a file that does not exist")

    publisher._s3.put_object = _unexpected  # pylint: disable=protected-access
    publisher.publish()
