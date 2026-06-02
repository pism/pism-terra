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
Bulk-validate NetCDF files under a directory using :func:`check_xr_lazy`.
"""

from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

from tqdm.auto import tqdm

from pism_terra.workflow import check_xr_lazy


def validate_directory(directory: Path | str, workers: int = 8, pattern: str = "*.nc") -> list[Path]:
    """
    Validate every NetCDF under *directory* in parallel.

    Walks *directory* recursively for files matching *pattern*, then runs
    :func:`pism_terra.workflow.check_xr_lazy` on each via a thread pool.
    Only invalid files are printed; a tqdm progress bar tracks completion.

    Parameters
    ----------
    directory : pathlib.Path or str
        Root directory to search. Searched recursively via ``Path.rglob``.
    workers : int, default 8
        Number of worker threads. ``check_xr_lazy`` is mostly I/O so threads
        overlap latency well; raise this on fast storage if you have many
        files.
    pattern : str, default ``"*.nc"``
        Glob pattern matched against file names during the recursive walk.

    Returns
    -------
    list of pathlib.Path
        Paths that failed validation (i.e. ``check_xr_lazy`` returned False).
        Empty when everything passes.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    files = sorted(root.rglob(pattern))
    if not files:
        print(f"No files matching {pattern!r} under {root}")
        return []

    invalid: list[Path] = []
    # Processes (not threads): HDF5 isn't reliably thread-safe across all
    # builds (Chinook's HPC build segfaults under concurrent opens), so
    # each worker gets its own interpreter and HDF5 state.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_path = {executor.submit(check_xr_lazy, p, verbose=False): p for p in files}
        for future in tqdm(
            as_completed(future_to_path),
            total=len(future_to_path),
            desc=f"Validating {root}",
            unit="file",
        ):
            p = future_to_path[future]
            try:
                ok = future.result()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # check_xr_lazy catches its own exceptions, so this is defensive.
                ok = False
                print(f"{p.resolve()} is not valid ✗ ({type(exc).__name__}: {exc})")
            if not ok:
                invalid.append(p)
                # check_xr_lazy(verbose=False) is silent, so surface the failure here.
                print(f"{p.resolve()} is not valid ✗")

    return invalid


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the bulk NetCDF validator.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments (without the program name). When ``None``
        (default), :data:`sys.argv` is used.

    Returns
    -------
    int
        Process exit code: ``0`` when every file is valid (or no files were
        found), ``1`` when at least one file failed validation.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Recursively validate NetCDF files under a directory."
    parser.add_argument(
        "--workers",
        help="Number of parallel worker threads.",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--pattern",
        help="Glob pattern for file names (matched recursively).",
        type=str,
        default="*.nc",
    )
    parser.add_argument(
        "DIRECTORY",
        help="Directory to walk recursively.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    invalid = validate_directory(args.DIRECTORY, workers=args.workers, pattern=args.pattern)

    if invalid:
        print(f"\n{len(invalid)} file(s) failed validation:")
        for p in invalid:
            print(f"  {p.resolve()}")
        return 1
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(main())
