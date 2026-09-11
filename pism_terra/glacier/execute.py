"""Execute the pism-run scripts."""

import shutil
import subprocess
import sys
import threading
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from types import TracebackType
from typing import Tuple

import boto3
from botocore.config import Config

from pism_terra.aws import local_to_s3, s3_to_local

#: Seconds between publishes of the growing run log.
PROGRESS_INTERVAL = 30.0

#: Name of the run log, under ``<RGI dir>/logs/`` both locally and on S3.
PROGRESS_NAME = "progress.log"


def find_first_and_execute(work_dir: Path = Path.cwd()):
    """
    Execute the first pism-run script found in work_dir.

    Parameters
    ----------
    work_dir : Path
        Directory to search inside for pism-run scripts.
    """
    run_scripts = list(work_dir.glob("**/run_scripts/*.sh"))

    if len(run_scripts) > 1:
        warnings.warn(f"More than one run script found! Only executing the first:\n{run_scripts}")

    execute(run_scripts[0])


class ProgressPublisher:
    """
    Copy a growing local log file to S3 while the run is still going.

    The job's own CloudWatch stream already holds everything the run prints,
    but reading it needs AWS credentials, which the people submitting through
    the notebook do not have. The same text published into the job's S3
    prefix is readable the way the run scripts themselves already are, so the
    app can show progress with nothing but an Earthdata login.

    Publishing is best-effort: a failed upload is reported and the run
    continues. Losing the live view is not a reason to lose the simulation.

    Parameters
    ----------
    log_file : pathlib.Path
        Local file being appended to.
    bucket : str
        Destination bucket.
    key : str
        Destination key.
    interval : float, optional
        Seconds between publishes.
    """

    def __init__(self, log_file: Path, bucket: str, key: str, interval: float = PROGRESS_INTERVAL):
        """
        Initialize the ProgressPublisher class.

        Parameters
        ----------
        log_file : pathlib.Path
            Local file being appended to.
        bucket : str
            Destination bucket.
        key : str
            Destination key.
        interval : float, optional
            Seconds between publishes.
        """
        self.log_file = Path(log_file)
        self.bucket = bucket
        self.key = key
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._s3 = boto3.client("s3", config=Config(retries={"max_attempts": 3}))

    def publish(self) -> None:
        """
        Upload the log file as it currently stands.

        Errors are printed rather than raised — this runs alongside the
        simulation and must never be the thing that stops it.
        """
        try:
            body = self.log_file.read_bytes()
        except OSError:
            return
        try:
            self._s3.put_object(
                Bucket=self.bucket,
                Key=self.key,
                Body=body,
                ContentType="text/plain; charset=utf-8",
                # Not ``file_type=product``: this is a log, and everything
                # tagged as a product is offered to the user as a result.
                Tagging="file_type=log",
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"progress publish to s3://{self.bucket}/{self.key} failed: {exc}", file=sys.stderr)

    def _loop(self) -> None:
        """
        Publish every ``interval`` seconds until asked to stop.
        """
        while not self._stop.wait(self.interval):
            self.publish()

    def __enter__(self) -> "ProgressPublisher":
        """
        Start the background publisher.

        Returns
        -------
        ProgressPublisher
            This instance.
        """
        print(f"Publishing progress to s3://{self.bucket}/{self.key} every {self.interval:.0f}s")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Stop the publisher and publish once more.

        The final publish runs whether or not the run succeeded: a failed
        run is exactly when someone wants to read the log.

        Parameters
        ----------
        exc_type : type of BaseException or None
            Exception class, if the block raised.
        exc : BaseException or None
            Exception instance, if the block raised.
        traceback : types.TracebackType or None
            Traceback, if the block raised.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval)
        self.publish()


def execute(script: Path, log_file: Path | None = None):
    """
    Execute a script.

    Parameters
    ----------
    script : Path
        Path to a script to execute.
    log_file : pathlib.Path or None, optional
        When given, the script's combined output is written here as well as
        to this process' stdout, so it can be published while the run is
        still going. Without it the output goes straight to stdout, as before.

    Raises
    ------
    subprocess.CalledProcessError
        If the script exits non-zero.
    RuntimeError
        If the child's output pipe cannot be opened.
    """
    print("Executing script: ", script)
    command = f"bash -ex {script.resolve()}"
    if log_file is not None and shutil.which("stdbuf"):
        # Line-buffer the legs so the published log tracks the run instead of
        # arriving one pipe buffer at a time. Only an improvement, not a
        # guarantee: a program that buffers on its own still buffers.
        command = f"stdbuf -oL -eL {command}"

    if log_file is None:
        subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr, shell=True, check=True)
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen(  # pylint: disable=consider-using-with
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        stream = proc.stdout
        if stream is None:  # pragma: no cover - stdout=PIPE always gives one
            raise RuntimeError("could not capture the run script's output")
        with log_file.open("w", encoding="utf-8") as handle:
            for line in stream:
                # Everything still reaches stdout, so the CloudWatch stream
                # stays complete for whoever does have AWS credentials.
                sys.stdout.write(line)
                sys.stdout.flush()
                handle.write(line)
                handle.flush()

    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, command)


def ensure_pism_terra_structure(script_uri: str) -> Tuple[str | None, str, Path]:
    """
    Ensure that the expected PISM-TERRA structure exists around a run script.

    Parameters
    ----------
    script_uri : str
        URI or local path to a PISM-TERRA run script.

    Returns
    -------
    str | None
        The S3 bucket inferred from the script_uri to stage files from.
    str
        The S3 prefix inferred from the script_uri to stage files from.
    Path
        The local path to the PISM-TERRA run script.
    """
    script = Path(script_uri)

    staging_bucket = None
    staging_prefix = "."  # No-prefix value that would be computed below
    if script_uri.startswith("s3://"):
        # pylint: disable=E1101
        staging_bucket = str(script.parents[-3].relative_to(script.parents[-2]))
        staging_prefix = str(script.parents[2].relative_to(script.parents[-3]))
        script = script.relative_to(script.parents[2])

    if (script.parents[0].name != "run_scripts") or not script.parents[1].name.startswith("RGI"):
        raise ValueError(
            f"{script} should be inside a PISM-TERRA generate directory of the form ``RGI*/runs_scripts/``"
        )

    rgi_dir = script.parents[1]

    (rgi_dir / "input").mkdir(parents=True, exist_ok=True)
    (rgi_dir / "logs").mkdir(parents=True, exist_ok=True)
    (rgi_dir / "output" / "inverse").mkdir(parents=True, exist_ok=True)
    (rgi_dir / "output" / "post_processing").mkdir(parents=True, exist_ok=True)
    (rgi_dir / "output" / "scalar").mkdir(parents=True, exist_ok=True)
    (rgi_dir / "output" / "spatial").mkdir(parents=True, exist_ok=True)
    (rgi_dir / "output" / "state").mkdir(parents=True, exist_ok=True)

    return staging_bucket, staging_prefix, script


def main():
    """CLI Enterypoint to execute a PISM-TERRA run script."""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Execute a PISM-TERRA run script."

    output_bucket = parser.add_argument_group(
        title="AWS S3 Bucket and prefix to upload the local working directory to at the end of processing."
    )
    output_bucket.add_argument(
        "--bucket",
    )
    output_bucket.add_argument(
        "--bucket-prefix",
        default="",
    )

    parser.add_argument(
        "RUN_SCRIPT",
        help="S3 URL or local path to the PISM run script to execute. If an S3 URI is provided, "
        "execute assumes a structure like `s3://{some-bucket}/{some-prefix}/RGI*/runs_scripts/*.sh`"
        "and files under `s3://{some-bucket}/{some-prefix}/` will be downloaded to the local work directory.",
        type=str,
    )

    args = parser.parse_args()

    work_dir = Path.cwd()

    staging_bucket, staging_prefix, local_run_script = ensure_pism_terra_structure(args.RUN_SCRIPT)
    if staging_bucket and staging_prefix:

        s3_to_local(staging_bucket, staging_prefix if staging_prefix != "." else "", work_dir)

    # The log lives where the final sync would put it anyway, so publishing it
    # live and uploading it at the end write the same key rather than two.
    rgi_dir = local_run_script.parents[1]
    log_file = work_dir / rgi_dir / "logs" / PROGRESS_NAME

    if args.bucket:
        key = "/".join(part for part in (args.bucket_prefix.strip("/"), rgi_dir.name, "logs", PROGRESS_NAME) if part)
        with ProgressPublisher(log_file, args.bucket, key):
            execute(local_run_script, log_file=log_file)
    else:
        execute(local_run_script, log_file=log_file)

    if args.bucket:
        local_to_s3(work_dir, args.bucket, args.bucket_prefix)
