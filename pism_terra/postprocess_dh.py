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

# pylint: disable=too-many-positional-arguments,too-many-locals,too-many-arguments

"""
Change between two dates of PISM spatial output.

The temporal counterpart of :mod:`pism_terra.postprocess_spatial`: instead of
splitting a spatial file into per-basin files, this module picks the two time
steps nearest a start and an end date and writes their difference
(``end - start``) for the requested variables — e.g. the surface elevation or
ice mass change of an RGI7 complex over 2000–2020, ready to compare against
observed elevation-change products such as Hugonnet et al. (2021).

Input handling is shared with the spatial module (same lazy chunked open,
grid-mapping cleanup, CRS resolution, and write encoding), so the output is a
georeferenced NetCDF on the input grid. Only two time slices are ever
computed, so no Dask cluster is needed; the default threaded scheduler
streams the difference chunk by chunk.

The output keeps the input variable names and carries a single ``time`` step
(the selected end date) with ``time_bnds`` spanning the selected interval.
Variables without a ``time`` dimension are copied through unchanged.
"""

import datetime
import logging
import time
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  pylint: disable=unused-import
import xarray as xr
from xarray.coders import CFDatetimeCoder

from pism_terra.log import setup_logging
from pism_terra.postprocess_scalar import dataset_crs
from pism_terra.postprocess_spatial import DROP_VARS, _encoding, _unlimited

xr.set_options(keep_attrs=True)
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)

logger = logging.getLogger(__name__)


def _nearest_time_index(ds: xr.Dataset, when: str) -> int:
    """
    Index of the time step nearest an ISO date, calendar-aware.

    ``sel(time=..., method="nearest")`` with a plain string is unreliable on
    non-standard calendars, so the target is built as the same date type the
    index carries — a :class:`cftime` subclass for PISM's ``365_day``/
    ``standard`` calendars, ``datetime64`` otherwise — before asking the
    index for its nearest position.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with a decoded ``time`` coordinate.
    when : str
        ISO 8601 date or datetime, e.g. ``"2000-01-01"``.

    Returns
    -------
    int
        Position of the nearest time step.
    """
    index = ds.indexes["time"]
    parsed = datetime.datetime.fromisoformat(when)
    if isinstance(index, xr.CFTimeIndex):
        date_type = type(index[0])
        target = date_type(parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute, parsed.second)
    else:
        target = np.datetime64(parsed)
    return int(index.get_indexer([target], method="nearest")[0])


def compute_dh(ds: xr.Dataset, start: str, end: str, variables: list[str] | None = None) -> xr.Dataset:
    """
    Lazy ``end - start`` difference of the spatial variables of ``ds``.

    The two time steps nearest ``start`` and ``end`` are selected; asking for
    dates outside the simulated period therefore clamps to the first/last
    step (the selection is logged, and identical steps are an error). The
    difference keeps each variable's name, dtype and attributes; a
    ``comment`` attribute records the differenced interval. Variables
    without a ``time`` dimension (e.g. a time-less ``topg``) are copied
    through unchanged.

    Parameters
    ----------
    ds : xarray.Dataset
        Spatial dataset with a decoded ``time`` coordinate; every data
        variable must carry ``y`` and ``x``.
    start : str
        ISO date of the start of the interval.
    end : str
        ISO date of the end of the interval.
    variables : list of str or None, optional
        Variables to difference. ``None`` (default) keeps all of them.
        Unknown names raise ``ValueError`` listing what is available.

    Returns
    -------
    xarray.Dataset
        Lazy dataset with one ``time`` step (the selected end date) and
        ``time_bnds`` spanning the selected interval.

    Raises
    ------
    ValueError
        On unknown ``variables`` or when start and end select the same time
        step.
    """
    if variables is not None:
        unknown = sorted(set(variables) - set(ds.data_vars))
        if unknown:
            raise ValueError(f"unknown variables {unknown}; available: {sorted(ds.data_vars)}")
        ds = ds[variables]

    i_start = _nearest_time_index(ds, start)
    i_end = _nearest_time_index(ds, end)
    if i_start == i_end:
        raise ValueError(
            f"start={start} and end={end} both select time step {i_start} "
            f"({ds['time'].values[i_start]}); the interval lies outside or between the model's time steps"
        )
    t_start = ds["time"].values[i_start]
    t_end = ds["time"].values[i_end]
    logger.info("Differencing time steps %d (%s) and %d (%s)", i_end, t_end, i_start, t_start)

    timeless = [v for v in ds.data_vars if "time" not in ds[v].dims]
    timed = ds.drop_vars(timeless)
    # Drop the scalar time coords before subtracting: they differ between
    # the two slices, and xarray would silently drop the coordinate anyway.
    dh = timed.isel(time=i_end).drop_vars("time") - timed.isel(time=i_start).drop_vars("time")
    dh = dh.expand_dims(time=[t_end])
    for name in dh.data_vars:
        dh[name].attrs["comment"] = f"difference {end} minus {start} (nearest model time steps)"
    for name in timeless:
        dh[name] = ds[name]
    dh["time_bnds"] = xr.DataArray([[t_start, t_end]], dims=("time", "bnds"))
    dh["time"].attrs["bounds"] = "time_bnds"
    dh.attrs["dh_start_time"] = str(t_start)
    dh.attrs["dh_end_time"] = str(t_end)
    return dh[[*ds.data_vars, "time_bnds"]]


def process_file_dh(
    infile: str | Path,
    outfile: str | Path,
    start: str,
    end: str,
    variables: list[str] | None = None,
    crs: str | None = None,
) -> Path:
    """
    Write the ``end - start`` difference of ``infile`` to ``outfile``.

    Shares the input handling of
    :func:`pism_terra.postprocess_spatial.process_file_spatial` — lazy
    chunked open, grid-mapping cleanup, CRS resolution, compressed write
    encoding — but decodes times (needed to select by date) and reduces the
    file to a single differenced time step instead of splitting it into
    basins. Non-spatial time-less variables (e.g. ``pism_config``) are
    carried through, like the spatial module.

    Parameters
    ----------
    infile : str or Path
        Path to the NetCDF file to difference. Must contain x/y spatial
        dimensions and a decodable ``time`` coordinate.
    outfile : str or Path
        Path of the output NetCDF file, overwritten if present.
    start : str
        ISO date of the start of the interval.
    end : str
        ISO date of the end of the interval.
    variables : list of str or None, optional
        Spatial variables to difference. ``None`` (default) keeps all of
        them. Unknown names raise ``ValueError`` listing what is available.
    crs : str or None, optional
        CRS written to the output. ``None`` (default) reads it from the
        file; see :func:`~pism_terra.postprocess_scalar.dataset_crs`.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    infile = Path(infile)
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    tic = time.time()

    ds = xr.open_dataset(
        infile,
        decode_timedelta=False,
        # ``use_cftime=True``: PISM runs on non-standard calendars (e.g.
        # ``365_day``), so keep every file on one time type instead of mixing
        # datetime64 and cftime depending on the calendar.
        decode_times=CFDatetimeCoder(use_cftime=True),
        chunks="auto",
        engine="h5netcdf",
    )
    try:
        # Read the projection off the grid mapping before DROP_VARS removes it.
        dst_crs = dataset_crs(ds, crs)
        ds = ds.drop_vars(DROP_VARS, errors="ignore")

        # Split off variables that lack spatial dims; the time-less ones (e.g.
        # ``pism_config``) are carried into the output, like the spatial module.
        non_spatial_vars = [var for var in ds.data_vars if "x" not in ds[var].dims or "y" not in ds[var].dims]
        extra = ds[[v for v in non_spatial_vars if "time" not in ds[v].dims]].compute()
        ds = ds.drop_vars(non_spatial_vars)

        # Preserve the input's time units/calendar for the output's time axis.
        time_enc = {k: v for k, v in ds["time"].encoding.items() if k in ("units", "calendar", "dtype")}

        dh = compute_dh(ds, start, end, variables=variables)
        for var in extra.data_vars:
            dh[var] = extra[var]
        dh = dh.rio.write_crs(dst_crs).rio.set_spatial_dims(x_dim="x", y_dim="y")

        encoding = _encoding(dh)
        if time_enc:
            encoding.setdefault("time", {}).update(time_enc)
            encoding["time_bnds"] = dict(time_enc)

        outfile.unlink(missing_ok=True)
        logger.info("Writing %s", outfile)
        dh.to_netcdf(outfile, engine="h5netcdf", encoding=encoding, unlimited_dims=_unlimited(dh))
    finally:
        ds.close()

    logger.info("Time elapsed for %s: %.0fs", infile.name, time.time() - tic)
    return outfile


def main():
    """
    Run main script.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = (
        "Extract the change (end minus start) of spatial variables between two dates "
        "of a PISM spatial output file, e.g. the 2000-2020 surface elevation change "
        "of an RGI7 complex."
    )
    parser.add_argument(
        "--start",
        help="ISO date of the start of the interval; the nearest model time step is used.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--end",
        help="ISO date of the end of the interval; the nearest model time step is used.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--vars",
        help="Comma-separated spatial variables to difference (default: all).",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--crs",
        help="CRS of the input file (default: read from its grid mapping).",
        type=str,
        default=None,
    )
    parser.add_argument(
        "INFILE",
        help="input file.",
        nargs=1,
    )
    parser.add_argument(
        "OUTFILE",
        help="output file.",
        nargs=1,
    )

    options, _ = parser.parse_known_args()
    outfile = Path(options.OUTFILE[0]).resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(outfile.parent / "postprocess_dh.log")

    process_file_dh(
        options.INFILE[0],
        outfile,
        start=options.start,
        end=options.end,
        variables=options.vars.split(",") if options.vars else None,
        crs=options.crs,
    )


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
