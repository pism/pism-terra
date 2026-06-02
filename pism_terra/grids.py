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

"""CDO grid description files for common projections."""

from pathlib import Path

_GRIDS_DIR = Path(__file__).parent / "grids"


def load_grid(name: str) -> str:
    """
    Load a CDO grid description by name.

    Parameters
    ----------
    name : str
        Grid name (e.g., ``"ismip6"``, ``"carra2"``, ``"hirham"``, ``"bedmachine"``).
        The ``.txt`` extension is appended automatically.

    Returns
    -------
    str
        Contents of the grid description file.
    """
    return (_GRIDS_DIR / f"{name}.txt").read_text(encoding="utf-8")
