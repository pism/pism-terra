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
    # Derive the allowed processes from the package's actual console_scripts
    # entries so this dispatcher stays in sync when scripts are renamed/added
    # in pyproject.toml.
    eps = entry_points(group="console_scripts")
    pism_processes = sorted(
        {ep.name for ep in eps if ep.name.startswith("pism-") or ep.name == "combine-crameri-colormaps"}
    )

    parser = argparse.ArgumentParser(prefix_chars="+", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "++process",
        choices=pism_processes,
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

    (process_entry_point,) = {process for process in eps if process.name == args.process}

    sys.argv = [args.process, *unknowns]
    sys.exit(process_entry_point.load()())


if __name__ == "__main__":
    main()
