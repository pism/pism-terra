"""
PISM-TERRA entrypoint dispatcher for PISM-Cloud.
"""

import argparse
import os
import sys
import warnings
from importlib.metadata import entry_points
from pathlib import Path


def main():
    """
    PISM-TERRA entrypoint dispatcher for PISM-Cloud.
    """
    parser = argparse.ArgumentParser(prefix_chars="+", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "++process",
        choices=[
            "pism-glacier-stage",
            "pism-glacier-run",
            "pism-glacier-run-ensemble",
            "pism-glacier-execute",
            "pism-glacier-postprocess",
            "combine-crameri-colormaps",
        ],
        default="pism-glacier-stage",
        help="Select the console_script entrypoint to use",  # as specified in `pyproject.toml`
    )

    args, unknowns = parser.parse_known_args()

    # Hand off credentials to ecmwf-datastores-client. It accepts either env
    # vars (ECMWF_DATASTORES_URL / ECMWF_DATASTORES_KEY) or ~/.ecmwfdatastoresrc.
    # We accept either CDS_API_* (legacy) or ECMWF_DATASTORES_* and write the
    # config file when the rc file isn't already present.
    cds_url = os.environ.get("ECMWF_DATASTORES_URL") or os.environ.get("CDS_API_URL")
    cds_key = os.environ.get("ECMWF_DATASTORES_KEY") or os.environ.get("CDS_API_KEY")
    if (cds_rc_file := Path.home() / ".ecmwfdatastoresrc").exists():
        if cds_url or cds_key:
            warnings.warn(
                "ECMWF Data Stores credentials provided in both environment variables "
                "and the `~/.ecmwfdatastoresrc` file. Preferring file."
            )
    elif cds_url and cds_key:
        cds_rc_file.write_text(f"url: {cds_url}\nkey: {cds_key}\n")

    eps = entry_points(group="console_scripts")
    (process_entry_point,) = {process for process in eps if process.name == args.process}

    sys.argv = [args.process, *unknowns]
    sys.exit(process_entry_point.load()())


if __name__ == "__main__":
    main()
