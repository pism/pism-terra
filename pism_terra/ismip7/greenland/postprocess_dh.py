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
Thickness change of an ISMIP7 Greenland run, to compare against observations.

The ISMIP7 counterpart of ``pism-glacier-postprocess-dh``, differing in two
ways that follow from what is being compared.

**Thickness, not surface elevation.** The observed field is Smith et al.
(2020) ``dhdt``, a thickness-change rate in metres of ice equivalent. Its
model counterpart is ``lithk``, not ``orog`` — the glacier tool differences
``usurf`` because Hugonnet et al. (2021) measures surface elevation from
DEMs, which is a different quantity.

**One file per variable.** A counter-driven run writes its spatial output
into the submission tree with a file per variable, so the input is the
experiment directory rather than a single combined file. A run with
``output.ISMIP = "no"`` writes one flat spatial file instead, and that is
accepted directly.

Two modes, and either lines up with the observations. Giving ``--end``
produces the single 2003-2019 interval the Smith product is: one record.
Omitting it produces change since ``--start`` at every model step; the
comparison aligns on time, so it takes the 2019 record out of that series
and the rest is there for diagnostics. Both need the run to have a step at
the observed date -- a historical run stopping in 2014 overlaps Smith
nowhere, and the comparison says so rather than inventing an interval.
"""

from __future__ import annotations

import logging
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Sequence

import rioxarray  # noqa: F401  pylint: disable=unused-import
import xarray as xr
from xarray.coders import CFDatetimeCoder

from pism_terra.ismip7.postprocess_flux import find_flux_files, submission_crs
from pism_terra.log import setup_logging
from pism_terra.postprocess_dh import _nearest_time_index, process_file_dh
from pism_terra.postprocess_spatial import DROP_VARS, _encoding

xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)

#: What the observed dH/dt measures: ice thickness, in metres ice equivalent.
#: ``orog`` is the surface elevation and is not what those products report.
DEFAULT_VARIABLES = ("lithk",)

#: Start of the observed record: the Smith et al. (2020) rate covers
#: 2003-2019, so the cumulative reference has one correct value and is a
#: default rather than something to be supplied every time. Kept in step with
#: ``DH_SMITH_START`` in :mod:`pism_terra.ismip7.greenland.forcing`, which is
#: deliberately not imported -- that module pulls in cdo, s3fs and geopandas,
#: which is a lot to load for one string.
DEFAULT_START = "2003-01-01"


def compute_cumulative_dh(ds: xr.Dataset, start: str, variables: Sequence[str] | None = None) -> xr.Dataset:
    """
    Change since ``start`` at every model step from there on.

    The observed products are cumulative from the beginning of their record,
    so a run has to be reduced the same way before the two can be subtracted.
    ``time`` stays the step's own date and ``time_bnds`` runs from the start
    of the record to it, matching :func:`pism_terra.postprocess_dh.compute_dh`
    for the single-interval case.

    Parameters
    ----------
    ds : xarray.Dataset
        Spatial dataset with a decoded ``time`` coordinate.
    start : str
        ISO date the record starts at; the nearest model step is used.
    variables : sequence of str or None, optional
        Variables to difference. ``None`` keeps all of them.

    Returns
    -------
    xarray.Dataset
        One record per model step at or after ``start``, plus ``time_bnds``.
    """
    i_start = _nearest_time_index(ds, start)
    t_start = ds["time"].values[i_start]
    logger.info("Cumulative from step %d (%s)", i_start, t_start)

    timeless = [v for v in ds.data_vars if "time" not in ds[v].dims]
    timed = ds.drop_vars(timeless)
    if variables is not None:
        missing = sorted(set(variables) - set(timed.data_vars))
        if missing:
            raise ValueError(f"{', '.join(missing)} not in {sorted(timed.data_vars)}")
        timed = timed[list(variables)]

    reference = timed.isel(time=i_start).drop_vars("time")
    dh = timed.isel(time=slice(i_start, None)) - reference
    for name in dh.data_vars:
        dh[name].attrs = dict(timed[name].attrs)
        dh[name].attrs["comment"] = f"change since {t_start} (nearest model time step)"
    for name in timeless:
        dh[name] = ds[name]

    starts = xr.full_like(dh["time"], t_start)
    dh["time_bnds"] = xr.concat([starts, dh["time"]], dim="bnds").transpose("time", "bnds")
    dh["time"].attrs["bounds"] = "time_bnds"
    dh.attrs["dh_start_time"] = str(t_start)
    return dh


def process_file_cumulative(
    infile: Path | str,
    outfile: Path | str,
    start: str,
    variables: Sequence[str] | None = None,
    crs: str | None = None,
) -> Path:
    """
    Write the cumulative change of one spatial file.

    Parameters
    ----------
    infile : Path or str
        Spatial NetCDF to reduce.
    outfile : Path or str
        Destination file; parent directories are created.
    start : str
        ISO date the record starts at.
    variables : sequence of str or None, optional
        Variables to difference.
    crs : str or None, optional
        CRS to stamp on the output.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    infile, outfile = Path(infile), Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with xr.open_dataset(
        infile,
        decode_times=CFDatetimeCoder(use_cftime=False),
        decode_timedelta=False,
        chunks="auto",
        engine="h5netcdf",
        drop_variables=DROP_VARS,
    ) as ds:
        # Captured before _encoding() clears every variable's encoding: without
        # it xarray picks units for ``time`` and ``time_bnds`` independently,
        # which CF forbids and which it warns about on write.
        time_enc = {k: v for k, v in ds["time"].encoding.items() if k in ("units", "calendar", "dtype")}

        dh = compute_cumulative_dh(ds, start, variables)
        if crs is not None:
            dh = dh.rio.write_crs(crs).rio.set_spatial_dims(x_dim="x", y_dim="y")

        encoding = _encoding(dh)
        if time_enc:
            encoding.setdefault("time", {}).update(time_enc)
            encoding["time_bnds"] = dict(time_enc)

        logger.info("Writing %s", outfile)
        dh.to_netcdf(outfile, encoding=encoding, engine="h5netcdf")
    logger.info("%s in %.0fs", outfile.name, time.time() - started)
    return outfile


def source_files(source: Path | str, variables: Sequence[str]) -> dict[str, Path]:
    """
    Resolve the input to one spatial file per variable.

    A counter-driven run writes the submission tree, a file per variable; a
    run with ISMIP7 naming off writes one flat spatial file holding all of
    them. Both are accepted, so the same command serves C011 and a plain OCX
    run.

    Parameters
    ----------
    source : Path or str
        Experiment directory, or a single spatial NetCDF.
    variables : sequence of str
        Variables wanted.

    Returns
    -------
    dict of str to pathlib.Path
        Variable name to the file holding it. A flat file maps every
        requested variable to itself.

    Raises
    ------
    FileNotFoundError
        If a directory holds no file for any requested variable.
    """
    source = Path(source)
    if source.is_file():
        return {variable: source for variable in variables}
    # The same per-variable matcher the flux post-processing uses: it keys on
    # the leading token, so ``lithk`` does not also match ``dlithkdt``.
    found = find_flux_files(source, set(variables))
    if not found:
        raise FileNotFoundError(f"no file for {', '.join(variables)} in {source}")
    return found


def postprocess_dh(
    source: Path | str,
    output_dir: Path | str,
    start: str,
    *,
    end: str | None = None,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    crs: str | None = None,
) -> list[Path]:
    """
    Reduce a run to thickness change and write it beside the run.

    Parameters
    ----------
    source : Path or str
        Experiment directory of the submission tree, or a flat spatial file.
    output_dir : Path or str
        Directory the ``dh_*`` files go in.
    start : str
        ISO date of the start of the interval.
    end : str or None, optional
        ISO date of the end. ``None`` gives the cumulative series from
        ``start`` at every model step instead of a single interval.
    variables : sequence of str, optional
        Variables to difference; thickness by default.
    crs : str or None, optional
        CRS of the input, when its grid mapping does not say.

    Returns
    -------
    list of pathlib.Path
        The written files, one per variable.
    """
    output_dir = Path(output_dir)
    written = []
    for variable, infile in sorted(source_files(source, variables).items()):
        outfile = output_dir / f"dh_{variable}_{infile.stem}.nc"
        # Submission files state their projection as ``proj_params`` rather
        # than the ``crs_wkt`` the shared helper wants, so resolve it here and
        # hand the answer down.
        with xr.open_dataset(infile, decode_times=False, engine="h5netcdf") as probe:
            resolved = submission_crs(probe, crs)
        if end is None:
            written.append(process_file_cumulative(infile, outfile, start, [variable], crs=resolved))
        else:
            written.append(process_file_dh(infile, outfile, start, end, [variable], resolved))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """
    Console entry point.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments without the program name.

    Returns
    -------
    int
        Exit code, ``0`` on success.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Thickness change of an ISMIP7 Greenland run, for comparison with observed dH/dt."
    parser.add_argument(
        "--start",
        help="ISO date the interval starts at; the nearest model step is used.",
        default=DEFAULT_START,
    )
    parser.add_argument(
        "--end",
        help="ISO date the interval ends at, for a single record. Omit for the cumulative series from "
        "--start at every model step; the observed record aligns with either.",
        default=None,
    )
    parser.add_argument(
        "--variables",
        help="Comma-separated variables to difference. The observed dH/dt is thickness, hence lithk.",
        default=",".join(DEFAULT_VARIABLES),
    )
    parser.add_argument("--crs", help="CRS of the input (default: read from its grid mapping).", default=None)
    parser.add_argument("SOURCE", nargs=1, help="Experiment directory of the submission tree, or a spatial file.")
    parser.add_argument("OUTPUT_DIR", nargs=1, help="Directory to write the dh files into.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = Path(args.OUTPUT_DIR[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "postprocess_dh.log")

    written = postprocess_dh(
        args.SOURCE[0],
        output_dir,
        args.start,
        end=args.end,
        variables=[v.strip() for v in args.variables.split(",") if v.strip()],
        crs=args.crs,
    )
    for path in written:
        logger.info("%s", path)
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(main())
