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
Put a KITP scalar timeseries on a comparable footing.

A KITP run spends its first years settling after initialisation, and its time
axis starts wherever the run did. Both make runs awkward to compare with each
other. This module trims the spin-up, restamps what is left as years 0..N-1,
and subtracts each cumulative variable's value at year 0 so every series starts at
zero.

The result is written as both NetCDF and CSV — the same numbers in the shape
each downstream tool wants.
"""

import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import cftime
import pandas as pd
import xarray as xr

from pism_terra.processing import normalize_timeseries

logger = logging.getLogger(__name__)

# Variables normalised to zero at the reference year. Cumulative quantities
# only: normalising a flux would subtract a constant from a rate.
DEFAULT_VARIABLES = ["ice_mass", "ice_mass_glacierized"]

# The window kept from the raw run. The first ten years are discarded as
# spin-up, and 300 years are kept after that.
SPINUP_END_YEAR = 11
WINDOW_END_YEAR = 311

# The glacier written to the CSV when the input has a ``glacier_id`` axis
# (basin-wise scalar output) and none is named: the whole ice sheet.
DEFAULT_GLACIER_ID_NAME = "GIS"


def adjust_kitp_timeseries(
    ds: xr.Dataset,
    variables: list[str] | None = None,
    spinup_end_year: int = SPINUP_END_YEAR,
    window_end_year: int = WINDOW_END_YEAR,
) -> xr.Dataset:
    """
    Trim spin-up, restamp the time axis from year 0, and normalise to its start.

    Parameters
    ----------
    ds : xarray.Dataset
        Scalar timeseries as written by PISM or by the scalar post-processing.
        A ``time_bounds`` variable and its ``nv`` dimension are dropped if
        present, since they would not survive the resampling.
    variables : list of str or None, optional
        Cumulative variables to normalise to zero at the first kept year.
        Defaults to :data:`DEFAULT_VARIABLES`.
    spinup_end_year : int, optional
        First year kept; everything before it is treated as spin-up.
    window_end_year : int, optional
        Last year kept (exclusive of the following year's samples).

    Returns
    -------
    xarray.Dataset
        Yearly means over the kept window, with ``time`` running from year 0
        and the requested variables starting at zero.

    Raises
    ------
    ValueError
        If the requested window selects no time steps, or if any requested
        variable is not in the dataset.
    """
    variables = list(DEFAULT_VARIABLES if variables is None else variables)
    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        raise ValueError(f"variables not in the input: {missing}; available: {sorted(ds.data_vars)}")

    ds = trim_spinup(ds, spinup_end_year=spinup_end_year, window_end_year=window_end_year)

    # The reference is the first kept year, which trim_spinup stamps year 0.
    ds = normalize_timeseries(ds, variables, cftime.DatetimeNoLeap(0, 1, 1))
    logger.info("Normalised %s to zero at year 0", ", ".join(variables))
    return ds


def trim_spinup(
    ds: xr.Dataset,
    spinup_end_year: int = SPINUP_END_YEAR,
    window_end_year: int | None = WINDOW_END_YEAR,
) -> xr.Dataset:
    """
    Drop the spin-up years, average to years, and restamp the rest from year 0.

    A run's first years are still settling after initialisation, and its time
    axis starts wherever the run did. Discarding the former and renumbering
    what is left puts runs of different vintages on a common axis.

    Parameters
    ----------
    ds : xarray.Dataset
        Scalar timeseries as written by PISM or by the scalar post-processing.
        A ``time_bounds`` variable and its ``nv`` dimension are dropped if
        present, since they would not survive the resampling.
    spinup_end_year : int, optional
        First year kept; everything before it is treated as spin-up.
    window_end_year : int or None, optional
        Last year kept. ``None`` keeps everything after the spin-up, which is
        what you want when the runs differ in length.

    Returns
    -------
    xarray.Dataset
        Yearly means over the kept window, with ``time`` running from year 0.

    Raises
    ------
    ValueError
        If the requested window selects no time steps.
    """
    ds = ds.drop_vars(["time_bounds"], errors="ignore").drop_dims("nv", errors="ignore")
    end = None if window_end_year is None else cftime.DatetimeNoLeap(window_end_year, 1, 1)
    ds = ds.sel({"time": slice(cftime.DatetimeNoLeap(spinup_end_year, 1, 1), end)})
    if ds.sizes.get("time", 0) == 0:
        raise ValueError(
            f"no time steps between years {spinup_end_year} and {window_end_year}; "
            "check the run length and the spin-up / window options"
        )

    ds = ds.resample(time="YS").mean()
    ds["time"] = [cftime.DatetimeNoLeap(year, 1, 1) for year in range(0, len(ds.time))]
    logger.info("Dropped the first %d years; kept %d, restamped from year 0", spinup_end_year - 1, ds.sizes["time"])
    return ds


def select_glacier(ds: xr.Dataset, glacier_id_name: str | None = None) -> xr.Dataset:
    """
    Pick one glacier out of a timeseries with a ``glacier_id`` axis.

    Basin-wise scalar output carries one series per glacier along
    ``glacier_id``, labelled by the ``glacier_id_name`` coordinate. A
    timeseries without that axis is returned unchanged.

    Parameters
    ----------
    ds : xarray.Dataset
        Timeseries, with or without a ``glacier_id`` axis.
    glacier_id_name : str or None, optional
        Name of the glacier to keep. Defaults to
        :data:`DEFAULT_GLACIER_ID_NAME` when the axis exists.

    Returns
    -------
    xarray.Dataset
        The chosen glacier's series without the ``glacier_id`` axis, or
        ``ds`` itself when it has none.

    Raises
    ------
    ValueError
        If ``ds`` has a ``glacier_id`` axis but no ``glacier_id_name``
        coordinate, or if no glacier has the requested name.
    """
    if "glacier_id" not in ds.dims:
        if glacier_id_name is not None:
            logger.warning("No glacier_id axis in the input; ignoring glacier_id_name %r", glacier_id_name)
        return ds
    if "glacier_id_name" not in ds.variables:
        raise ValueError("the input has a glacier_id axis but no glacier_id_name to select by")

    name = DEFAULT_GLACIER_ID_NAME if glacier_id_name is None else glacier_id_name
    names = [str(n) for n in ds["glacier_id_name"].values]
    if name not in names:
        raise ValueError(f"no glacier_id_name {name!r} in the input; available: {names}")
    logger.info("Selecting glacier_id_name %s", name)
    return ds.isel(glacier_id=names.index(name))


def write_outputs(
    ds: xr.Dataset,
    outfile: str | Path,
    outcsv: str | Path,
    glacier_id_name: str | None = None,
) -> pd.DataFrame:
    """
    Write the adjusted timeseries as NetCDF and as a flat CSV.

    Parameters
    ----------
    ds : xarray.Dataset
        Adjusted timeseries from :func:`adjust_kitp_timeseries`.
    outfile : str or pathlib.Path
        Destination NetCDF file, with every glacier. Parent directories are
        created.
    outcsv : str or pathlib.Path
        Destination CSV file. Parent directories are created.
    glacier_id_name : str or None, optional
        Glacier written to the CSV when ``ds`` has a ``glacier_id`` axis; see
        :func:`select_glacier`.

    Returns
    -------
    pandas.DataFrame
        The frame that was written, with the index reset to columns.
    """
    outfile, outcsv = Path(outfile), Path(outcsv)
    for path in (outfile, outcsv):
        path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing %s", outfile)
    ds.to_netcdf(outfile)

    ds = select_glacier(ds, glacier_id_name)
    ds["sle"] = -ds["ice_mass_glacierized"] / 1e12 / 361.8
    df = ds.to_dataframe().reset_index()
    df = df["sle"]
    logger.info("Writing %s", outcsv)
    df.to_csv(outcsv, index=True)
    return df


def main():
    """
    Run main script.
    """

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Trim spin-up from a KITP scalar timeseries and normalise it to its first year."
    parser.add_argument(
        "--variables",
        help="Comma-separated cumulative variables to normalise to zero at the first kept year.",
        type=str,
        default=",".join(DEFAULT_VARIABLES),
    )
    parser.add_argument(
        "--spinup-end-year",
        help="First year kept; everything before it is discarded as spin-up.",
        type=int,
        default=SPINUP_END_YEAR,
    )
    parser.add_argument(
        "--window-end-year",
        help="Last year kept.",
        type=int,
        default=WINDOW_END_YEAR,
    )
    parser.add_argument(
        "--glacier-id-name",
        help="Glacier written to the CSV when the input has a glacier_id axis, by its glacier_id_name; "
        f"{DEFAULT_GLACIER_ID_NAME} when unset. The NetCDF keeps every glacier.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "INFILE",
        help="input scalar timeseries (NetCDF).",
        nargs=1,
    )
    parser.add_argument(
        "OUTFILE",
        help="output NetCDF file.",
        nargs=1,
    )
    parser.add_argument(
        "OUTCSV",
        help="output CSV file.",
        nargs=1,
    )

    options, _ = parser.parse_known_args()
    variables = [v.strip() for v in options.variables.split(",") if v.strip()]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)  # pylint: disable=no-member
    ds = xr.open_dataset(options.INFILE[0], decode_times=time_coder)
    # A char-array config blob has no place in a dataframe.
    ds = ds.drop_vars("pism_config", errors="ignore")

    adjusted = adjust_kitp_timeseries(
        ds,
        variables,
        spinup_end_year=options.spinup_end_year,
        window_end_year=options.window_end_year,
    )
    write_outputs(adjusted, options.OUTFILE[0], options.OUTCSV[0], glacier_id_name=options.glacier_id_name)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
