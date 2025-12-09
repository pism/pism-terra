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
            "pism-glacier-postprocess",
            "combine-crameri-colormaps",
        ],
        default="pism-glacier-stage",
        help="Select the console_script entrypoint to use",  # as specified in `pyproject.toml`
    )

    args, unknowns = parser.parse_known_args()

    cds_api_url = os.environ.get("CDS_API_URL")
    cds_api_key = os.environ.get("CDS_API_KEY")
    if (cds_api_file := Path.home() / ".cdsapirc").exists():
        warnings.warn(
            "CDS API credentials provided in both environment variables and the `~/.cdsapirc` file. Preferring file."
        )
    else:
        cds_api_file.write_text(f"url: {cds_api_url}\nkey: {cds_api_key}\n")

    eps = entry_points(group="console_scripts")
    (process_entry_point,) = {process for process in eps if process.name == args.process}

    sys.argv = [args.process, *unknowns]
    sys.exit(process_entry_point.load()())


if __name__ == "__main__":
    main()
