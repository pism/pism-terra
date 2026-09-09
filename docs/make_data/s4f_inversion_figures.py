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
Regenerate the figures on the S4F inverse-modelling page.

The page shows three penalty sweeps — inverting for ``tauc``, for ``hardav``,
and for both by alternation — as L-curves and as maps. Rather than reduce them
to a checked-in fixture the way the Greenland page does, this runs
``pism-inverse-lcurve`` and ``pism-inverse-plot`` exactly as the page
documents them and drops the PNGs into ``docs/source/_static/s4f/``, so the
figures are literally the output of the commands a reader would type.

Re-run it whenever the sweeps advance; the ensembles are long-running and a
member contributes nothing until it has written its inversion diagnostics::

    python docs/make_data/s4f_inversion_figures.py --root /mnt/storstrommen/pism/terra

Only the experiments present under ``--root`` are regenerated, so it is safe
to run while some sweeps are still queued. The page writes the sweeps to bare
``inverse_*_penalty`` directories; ``--prefix`` bridges that to the dated ones
an actual campaign uses.
"""

from __future__ import annotations

import subprocess
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

# Output directory of each sweep as the page names it, and the design
# variables to map for it. The alternating run solves for both in turn, so it
# gets a figure per phase.
EXPERIMENTS = {
    "tauc": ("inverse_tauc_penalty", "tauc"),
    "hardav": ("inverse_hardav_penalty", "hardav"),
    "alt": ("inverse_alt_penalty", "tauc,hardav"),
}

DEFAULT_ROOT = Path("/mnt/storstrommen/pism/terra")
# Campaign directories are dated on disk; the page writes the bare names, so
# bridge the two here rather than dating every path in the documentation.
DEFAULT_PREFIX = "2026_09_s4f_"
DEFAULT_RGI_ID = "RGI2000-v7.0-C-01-04374"
DEFAULT_RESOLUTION = "200m"


def member_files(root: Path, directory: str, rgi_id: str, resolution: str) -> list[Path]:
    """
    List one sweep's inversion output files.

    Parameters
    ----------
    root : pathlib.Path
        Directory holding the campaign output directories.
    directory : str
        The sweep's output directory name.
    rgi_id : str
        RGI7 complex id of the glacier.
    resolution : str
        Grid resolution tag embedded in the file names, e.g. ``"200m"``.

    Returns
    -------
    list of pathlib.Path
        The members, sorted; empty when the sweep has not started.
    """
    inverse = root / directory / rgi_id / "output" / "inverse"
    return sorted(inverse.glob(f"inv_g{resolution}_{rgi_id}_id_0_uq_*.nc"))


def run(command: list[str]) -> bool:
    """
    Run one figure command, reporting rather than raising on failure.

    A sweep with too few finished members is a normal state here, not an
    error worth stopping the whole regeneration for.

    Parameters
    ----------
    command : list of str
        Command and arguments.

    Returns
    -------
    bool
        True when the command succeeded.
    """
    print(f"$ {' '.join(command[:4])} ... ({len(command) - 4} files)")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  failed: {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else result.returncode}")
        return False
    return True


def main() -> int:
    """
    Regenerate every figure that has data behind it.

    Returns
    -------
    int
        0 when at least one figure was written, 1 when none could be.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Regenerate the L-curves and maps on the S4F inverse-modelling page."
    parser.add_argument(
        "--root", help="Directory holding the sweep output directories.", type=Path, default=DEFAULT_ROOT
    )
    parser.add_argument(
        "--prefix",
        help="Prepended to each sweep's directory name, for dated campaign directories.",
        type=str,
        default=DEFAULT_PREFIX,
    )
    parser.add_argument(
        "--output-dir",
        help="Where the figures are written; the page reads them from here.",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "source" / "_static" / "s4f",
    )
    parser.add_argument("--rgi-id", help="RGI7 complex id.", type=str, default=DEFAULT_RGI_ID)
    parser.add_argument(
        "--resolution", help="Grid resolution tag in the file names.", type=str, default=DEFAULT_RESOLUTION
    )
    parser.add_argument("--dpi", help="Figure resolution; the page does not need print quality.", type=int, default=150)
    options = parser.parse_args()

    options.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, (directory, designs) in EXPERIMENTS.items():
        directory = f"{options.prefix}{directory}"
        files = member_files(options.root, directory, options.rgi_id, options.resolution)
        if not files:
            print(f"{name}: no members under {options.root / directory}, skipped")
            continue
        print(f"{name}: {len(files)} members")
        paths = [str(f) for f in files]
        run(
            ["pism-inverse-lcurve", "--dpi", str(options.dpi), "-o", str(options.output_dir / f"{name}_lcurve.png")]
            + paths
        )
        run(
            [
                "pism-inverse-plot",
                "--design-variable",
                designs,
                "--dpi",
                str(options.dpi),
                "-o",
                str(options.output_dir / f"{name}_maps.png"),
            ]
            + paths
        )
        figures = sorted(options.output_dir.glob(f"{name}_*.png"))
        for figure in figures:
            print(f"  {figure.name}")
        written += len(figures)

    # The tools drop a table beside each figure and a log beside that; neither
    # belongs in the documentation's static tree.
    for byproduct in list(options.output_dir.glob("*.csv")) + list(options.output_dir.glob("*.log")):
        byproduct.unlink()

    if not written:
        print("no figures written; check --root", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
