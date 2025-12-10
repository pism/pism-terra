"""Execute the pism-run scripts."""

import subprocess
import sys
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from pism_cloud.aws import local_to_s3, s3_to_local


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


def ensure_pism_terra_structure(script: Path):
    """
    Ensure that the expected PISM-TERRA structure exists around a run script.

    Parameters
    ----------
    script : Path
        Path to a PISM-TERRA run script.
    """
    if not script.exists():
        raise ValueError(f"{script} does not exist")

    if (script.parents[0].name != "run_scripts") or not script.parents[1].name.startswith("RGI"):
        raise ValueError(
            f"{script} should be inside a PISM-TERRA generate directory of the form ``RGI*/runs_scripts/``"
        )

    rgi_dir = script.parents[1]

    (rgi_dir / "input").mkdir(parents=True, exist_ok=True)
    (rgi_dir / "logs").mkdir(parents=True, exist_ok=True)
    (rgi_dir / "output" / "post_processing").mkdir(parents=True, exist_ok=True)
    (rgi_dir / "output" / "spatial").mkdir(parents=True, exist_ok=True)
    (rgi_dir / "output" / "state").mkdir(parents=True, exist_ok=True)


def main():
    """CLI Enterypoint to execute a PISM-TERRA run script."""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Execute a PISM-TERRA run script."
    parser.add_argument(
        "--bucket",
        help="AWS S3 Bucket to sync with the local working directory.",
    )
    parser.add_argument(
        "--bucket-prefix",
        help="AWS prefix to sync with the local working directory.",
        default="",
    )
    parser.add_argument(
        "--job-id",
        help="PISM_TERRA_PREP_ENSEMBLE to stage glacier run files from.",
    )
    parser.add_argument(
        "RUN_SCRIPT",
        help="Path to the PISM run script to execute. If you've provided ``--bucket`` and ``--bucket-prefix``, "
        "this path will need to be relative to ``f's3://{bucket}/{bucket_prefix}/'``.",
        type=Path,
    )

    args = parser.parse_args()

    work_dir = Path.cwd()

    if args.bucket:
        # FIXME: pism-terra produces hard-coded absolute paths, so things _must_ end up in ${HOME}/data  # pylint: disable=W0511
        work_dir /= "data"
        s3_to_local(args.bucket, args.job_id, work_dir)

    run_script = args.work_dir / args.RUN_SCRIPT
    ensure_pism_terra_structure(run_script)

    execute(run_script)

    if args.bucket:
        local_to_s3(work_dir, args.bucket, args.bucket_prefix)
