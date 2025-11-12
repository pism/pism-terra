"""
pism-terra entrypoint dispatcher for PISM-Cloud
"""

import argparse
import sys
from importlib.metadata import entry_points


def main():
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

    eps = entry_points(group="console_scripts")
    (process_entry_point,) = {process for process in eps if process.name == args.process}

    sys.argv = [args.process, *unknowns]
    sys.exit(process_entry_point.load()())


if __name__ == "__main__":
    main()
