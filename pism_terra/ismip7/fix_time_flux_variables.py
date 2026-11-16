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
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Re-stamp ISMIP7 flux time series onto the middle of their averaging interval.

PISM reports yearly flux diagnostics (``tendacabf``, ``tendlicalvf``, …) at the
*end* of the year they average over, so the value covering 2015 is written at
2016-01-01. ISMIP7 expects the timestamp to sit inside that interval, so each
time is moved back to July 1 of the preceding year, leaving ``time_bounds``
(and therefore the interval itself) untouched.

Console entry point ``pism-ismip7-fix-time-flux-variables``, called per flux
variable by ``pism_terra/data/postprocess_ismip7_scalar.sh`` after the variable
has been split out of the scalar file.
"""

from argparse import ArgumentParser
from typing import Sequence

import cftime
import netCDF4


def main(argv: Sequence[str] | None = None) -> int:
    """
    Move each time stamp to July 1 of the previous year, in place.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments (without the program name). If ``None``
        (default), uses ``sys.argv``.

    Returns
    -------
    int
        Exit code (``0`` on success).
    """
    parser = ArgumentParser(description="Re-stamp ISMIP7 flux times onto the middle of their averaging interval.")
    parser.add_argument("FILE", help="NetCDF file with a single ISMIP7 flux variable (modified in place).")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # replace times with July 1 of the previous year
    with netCDF4.Dataset(args.FILE, "a") as ds:
        calendar = ds.variables["time"].calendar
        units = ds.variables["time"].units

        time = ds.variables["time"][:]
        for j, value in enumerate(time):
            year = cftime.num2date(value, units, calendar).year
            fixed_date = cftime.datetime(year - 1, 7, 1, calendar=calendar)
            time[j] = cftime.date2num(fixed_date, units, calendar)
        ds.variables["time"][:] = time
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
