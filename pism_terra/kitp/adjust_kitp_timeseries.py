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
other. This module trims the spin-up, restamps what is left as years 1..N, and
subtracts each cumulative variable's value at year 1 so every series starts at
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


def adjust_kitp_timeseries(
    ds: xr.Dataset,
    variables: list[str] | None = None,
    spinup_end_year: int = SPINUP_END_YEAR,
    window_end_year: int = WINDOW_END_YEAR,
) -> xr.Dataset:
    """
    Trim spin-up, restamp the time axis from year 1, and normalise to its start.

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
        Yearly means over the kept window, with ``time`` running from year 1
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

    ds = ds.drop_vars(["time_bounds"], errors="ignore").drop_dims("nv", errors="ignore")
    ds = ds.sel(
        {
            "time": slice(
                cftime.DatetimeNoLeap(spinup_end_year, 1, 1),
                cftime.DatetimeNoLeap(window_end_year, 1, 1),
            )
        }
    )
    if ds.sizes.get("time", 0) == 0:
        raise ValueError(
            f"no time steps between years {spinup_end_year} and {window_end_year}; "
            "check the run length and the --spinup-end-year / --window-end-year options"
        )

    ds = ds.resample(time="YS").mean()
    ds["time"] = [cftime.DatetimeNoLeap(year, 1, 1) for year in range(1, len(ds.time) + 1)]
    logger.info("Kept %d years, restamped as years 1-%d", ds.sizes["time"], ds.sizes["time"])

    ds = normalize_timeseries(ds, variables, cftime.DatetimeNoLeap(1, 1, 1))
    logger.info("Normalised %s to zero at year 1", ", ".join(variables))
    return ds


def write_outputs(ds: xr.Dataset, outfile: str | Path, outcsv: str | Path) -> pd.DataFrame:
    """
    Write the adjusted timeseries as NetCDF and as a flat CSV.

    Parameters
    ----------
    ds : xarray.Dataset
        Adjusted timeseries from :func:`adjust_kitp_timeseries`.
    outfile : str or pathlib.Path
        Destination NetCDF file. Parent directories are created.
    outcsv : str or pathlib.Path
        Destination CSV file. Parent directories are created.

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

    df = ds.to_dataframe().reset_index()
    logger.info("Writing %s", outcsv)
    df.to_csv(outcsv, index=False)
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

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(options.INFILE[0], decode_times=time_coder)
    # A char-array config blob has no place in a dataframe.
    ds = ds.drop_vars("pism_config", errors="ignore")

    adjusted = adjust_kitp_timeseries(
        ds,
        variables,
        spinup_end_year=options.spinup_end_year,
        window_end_year=options.window_end_year,
    )
    write_outputs(adjusted, options.OUTFILE[0], options.OUTCSV[0])


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
