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

import xarray as xr
from tqdm.auto import tqdm

from pism_terra.workflow import check_dataset_lazy


def _validate_one(path_str: str) -> tuple[bool, str]:
    """
    Open a dataset, run the sampled health check, and surface any failure reason.

    Mirrors :func:`pism_terra.workflow.check_xr_lazy` but returns the failure
    reason instead of printing it, so the parent process can show it both
    inline and in the final summary.

    Parameters
    ----------
    path_str : str
        Absolute path to the NetCDF / Zarr dataset (str rather than ``Path``
        keeps worker process pickling cheap).

    Returns
    -------
    tuple of (bool, str)
        ``(True, "")`` on success. ``(False, "ExceptionType: message")`` if
        the dataset cannot be opened or fails the lazy checks.
    """
    try:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        delta_coder = xr.coders.CFTimedeltaCoder()
        ds = xr.open_dataset(path_str, decode_times=time_coder, decode_timedelta=delta_coder)
        check_dataset_lazy(ds)
    except FileNotFoundError as exc:
        return False, f"FileNotFoundError: {exc}"
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def validate_directory(
    directory: Path | str,
    workers: int = 8,
    pattern: str = "*.nc",
) -> list[tuple[Path, str]]:
    """
    Validate every NetCDF under *directory* in parallel.

    Walks *directory* recursively for files matching *pattern*, runs the
    sampled health check on each in a worker process pool, and prints both
    inline per-file failures and an end-of-run summary that includes the
    failure reason (exception type + message) for each invalid file.

    Parameters
    ----------
    directory : pathlib.Path or str
        Root directory to search. Searched recursively via ``Path.rglob``.
    workers : int, default 8
        Number of parallel worker processes. HDF5 is not reliably
        thread-safe across builds, so each worker gets its own interpreter
        and HDF5 state.
    pattern : str, default ``"*.nc"``
        Glob pattern matched against file names during the recursive walk.

    Returns
    -------
    list of tuple of (pathlib.Path, str)
        One entry per failed file: ``(path, "ExceptionType: message")``.
        Empty when everything passes. The previous version returned only
        the paths; the reason string is added so callers can present it in
        their own reports without re-opening the file.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    files = sorted(root.rglob(pattern))
    if not files:
        print(f"No files matching {pattern!r} under {root}")
        return []

    invalid: list[tuple[Path, str]] = []
    # Processes (not threads): HDF5 isn't reliably thread-safe across all
    # builds (Chinook's HPC build segfaults under concurrent opens), so
    # each worker gets its own interpreter and HDF5 state.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_path = {executor.submit(_validate_one, str(p)): p for p in files}
        for future in tqdm(
            as_completed(future_to_path),
            total=len(future_to_path),
            desc=f"Validating {root}",
            unit="file",
        ):
            p = future_to_path[future]
            try:
                ok, reason = future.result()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # _validate_one catches its own exceptions, so this is defensive.
                ok = False
                reason = f"{type(exc).__name__}: {exc}"
            if not ok:
                invalid.append((p, reason))
                # Surface inline so the user sees failures as the bar advances.
                print(f"{p.resolve()} is not valid ✗ ({reason})")

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
        for p, reason in invalid:
            print(f"  {p.resolve()}")
            print(f"    → {reason}")
        return 1
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(main())
