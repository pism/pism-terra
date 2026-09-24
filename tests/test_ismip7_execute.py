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
Tests for :mod:`pism_terra.ismip7.greenland.execute`.

The cloud runs an ISMIP7 prepare job's script from ``<cwd>/data``; these pin
where the job tree is mirrored to, that the run's output is teed into the
log the PISM-Cloud notebook shows live, and the S3 key it is published at.
"""

from __future__ import annotations

from pathlib import Path

from pism_terra.ismip7.greenland import execute as ex

#: The script URI of a PISM_EXECUTE job, as the API hands it to the container.
URI = "s3://pism-cloud-data/test_ismip7/test_ensemble/341d2b28-8d8e-4d38-aa5b-8a10b9ab9ff4/run_scripts/submit_g900m_id_C001.sh"


def test_main_publishes_the_run_log_under_the_job_prefix(tmp_path: Path, monkeypatch):
    """
    The job tree is mirrored into data/, the script's output teed into data/logs/progress.log and published there.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory, used as the container's working directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture replacing the S3 transfers and the publisher.
    """
    calls: list[tuple] = []
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "data" / "run_scripts" / "submit_g900m_id_C001.sh"

    def fake_download(bucket, prefix, dest):
        """
        Stand in for the S3 mirror: record the call and write the run script.

        Parameters
        ----------
        bucket : str
            Source bucket.
        prefix : str
            Source prefix.
        dest : pathlib.Path
            Local destination.
        """
        calls.append(("down", bucket, prefix, Path(dest)))
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("echo leg one\necho leg two\n")

    class FakePublisher:
        """
        Record the key the log would be published to instead of talking to S3.

        Parameters
        ----------
        log_file : pathlib.Path
            The log being published.
        bucket : str
            Destination bucket.
        key : str
            Destination key.
        interval : float, optional
            Publish interval, unused.
        """

        def __init__(self, log_file, bucket, key, interval=30.0):  # pylint: disable=unused-argument
            """
            Record the publish target.

            Parameters
            ----------
            log_file : pathlib.Path
                The log being published.
            bucket : str
                Destination bucket.
            key : str
                Destination key.
            interval : float, optional
                Publish interval, unused.
            """
            calls.append(("publish", Path(log_file), bucket, key))

        def __enter__(self):
            """
            Enter the context.

            Returns
            -------
            FakePublisher
                This instance.
            """
            return self

        def __exit__(self, *exc):
            """
            Leave the context without publishing.

            Parameters
            ----------
            *exc : Any
                Exception details, ignored.
            """
            return None

    monkeypatch.setattr(ex, "s3_to_local", fake_download)
    monkeypatch.setattr(ex, "ProgressPublisher", FakePublisher)
    monkeypatch.setattr(ex, "local_to_s3", lambda src, bucket, prefix: calls.append(("up", Path(src), bucket, prefix)))

    ex.main([URI, "--bucket", "pism-cloud-data", "--bucket-prefix", "test_ismip7/test_ensemble/0f1ffaf5/"])

    data = tmp_path / "data"
    assert calls == [
        ("down", "pism-cloud-data", "test_ismip7/test_ensemble/341d2b28-8d8e-4d38-aa5b-8a10b9ab9ff4", data),
        (
            "publish",
            data / "logs" / "progress.log",
            "pism-cloud-data",
            "test_ismip7/test_ensemble/0f1ffaf5/logs/progress.log",
        ),
        ("up", data, "pism-cloud-data", "test_ismip7/test_ensemble/0f1ffaf5/"),
    ]
    # The run's output went into the log the notebook reads, not only to stdout.
    assert "leg one" in (data / "logs" / "progress.log").read_text()
    assert "leg two" in (data / "logs" / "progress.log").read_text()


def test_main_without_a_bucket_still_keeps_a_log(tmp_path: Path, monkeypatch):
    """
    A local run tees its output into logs/progress.log and publishes nothing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory, used as the working directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture replacing the publisher, which must not be used.
    """
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "data" / "run_scripts" / "submit.sh"
    script.parent.mkdir(parents=True)
    script.write_text("echo hello\n")
    monkeypatch.setattr(ex, "ProgressPublisher", None)
    ex.main([str(script)])
    assert "hello" in (tmp_path / "data" / "logs" / "progress.log").read_text()
