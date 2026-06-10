"""Execute the pism-run scripts."""

import subprocess
import sys
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Tuple

from pism_terra.aws import local_to_s3, s3_to_local


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


def execute(script: Path):
    """
    Execute a script.

    Parameters
    ----------
    script : Path
        Path to a script to execute.
    """
    print("Executing script: ", script)
    subprocess.run(
        f"bash -ex {script.resolve()}",
        stdout=sys.stdout,
        stderr=sys.stderr,
        shell=True,
        check=True,
    )


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

    execute(local_run_script)

    if args.bucket:
        local_to_s3(work_dir, args.bucket, args.bucket_prefix)
