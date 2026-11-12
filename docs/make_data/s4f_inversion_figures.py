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

The page shows one project that samples the penalty weight and the inversion
strategy together — ``tauc``, ``hardav``, ``tauc_hardav`` and ``hardav_tauc``
— as L-curves and as maps. Rather than reduce it to a checked-in fixture the
way the Greenland page does, this runs ``pism-inverse-lcurve`` and
``pism-inverse-plot`` exactly as the page documents them and drops the PNGs
into ``docs/source/_static/s4f/``, so the figures are literally the output of
the commands a reader would type.

Re-run it whenever the project advances; the members are long-running and one
contributes nothing until it has written its inversion diagnostics::

    python docs/make_data/s4f_inversion_figures.py --root /mnt/storstrommen/pism/terra

Members that have not finished are skipped by the tools, so it is safe to run
while some are still queued.
"""

from __future__ import annotations

import subprocess
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

DEFAULT_ROOT = Path("/mnt/storstrommen/pism/terra")
DEFAULT_PROJECT = "2026_10_inverse_lcurve"
DEFAULT_RGI_ID = "RGI2000-v7.0-C-01-04374"
DEFAULT_RESOLUTION = "200m"


def member_files(project: Path, rgi_id: str, resolution: str) -> list[Path]:
    """
    List the project's inversion output files.

    Parameters
    ----------
    project : pathlib.Path
        Project directory, as given to ``pism-glacier-run-inverse --output-path``.
    rgi_id : str
        RGI7 complex id of the glacier.
    resolution : str
        Grid resolution tag embedded in the file names, e.g. ``"200m"``.

    Returns
    -------
    list of pathlib.Path
        The members, sorted; empty when the project has not started.
    """
    inverse = project / rgi_id / "output" / "inverse"
    return sorted(inverse.glob(f"inv_g{resolution}_{rgi_id}_id_0_uq_*.nc"))


def run(command: list[str], n_files: int) -> bool:
    """
    Run one figure command, reporting rather than raising on failure.

    A project with too few finished members is a normal state here, not an
    error worth stopping the whole regeneration for.

    Parameters
    ----------
    command : list of str
        Command and arguments.
    n_files : int
        Number of member files at the end of ``command``, left out of the echo.

    Returns
    -------
    bool
        True when the command succeeded.
    """
    print(f"$ {' '.join(command[:-n_files])} ... ({n_files} files)")
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
    parser.add_argument("--root", help="Directory holding the project directory.", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--project", help="Project directory under --root.", type=str, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--output-path",
        help="Where the figures are written; the page reads them from here.",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "source" / "_static" / "s4f",
    )
    parser.add_argument("--rgi-id", help="RGI7 complex id.", type=str, default=DEFAULT_RGI_ID)
    parser.add_argument(
        "--resolution", help="Grid resolution tag in the file names.", type=str, default=DEFAULT_RESOLUTION
    )
    parser.add_argument("--dpi", help="Figure resolution; the page does not need print quality.", type=int, default=300)
    options = parser.parse_args()

    project = options.root / options.project
    files = member_files(project, options.rgi_id, options.resolution)
    if not files:
        print(f"no members under {project}", file=sys.stderr)
        return 1
    print(f"{project}: {len(files)} members")
    options.output_path.mkdir(parents=True, exist_ok=True)
    paths = [str(f) for f in files]
    dpi = ["--dpi", str(options.dpi)]
    run(["pism-inverse-lcurve", *dpi, "-o", str(options.output_path / "lcurve.png")] + paths, len(paths))
    run(["pism-inverse-plot", *dpi, "-o", str(options.output_path / "maps.png")] + paths, len(paths))

    figures = sorted(options.output_path.glob("lcurve*.png")) + sorted(options.output_path.glob("maps_*.png"))
    for figure in figures:
        print(f"  {figure.name}")

    # The tools drop a table beside each figure and a log beside that; neither
    # belongs in the documentation's static tree.
    for byproduct in list(options.output_path.glob("*.csv")) + list(options.output_path.glob("*.log")):
        byproduct.unlink()

    if not figures:
        print("no figures written; check --root and --project", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
