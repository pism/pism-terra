# Copyright (C) 2025 Andy Aschwanden
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

# mypy: disable-error-code="call-overload"

"""
Vector Functions.
"""

from pathlib import Path

import geopandas as gpd


def get_glacier_from_rgi_id(rgi: gpd.GeoDataFrame | str | Path, rgi_id: str) -> gpd.GeoDataFrame:
    """
    Return the row in the GeoDataFrame matching the given RGI ID.

    Parameters
    ----------
    rgi : geopandas.GeoDataFrame
        GeoDataFrame containing glacier data.
    rgi_id : str
        RGI identifier to look up.

    Returns
    -------
    geopandas.GeoSeries
        The matching row.
    """
    if isinstance(rgi, str | Path):
        rgi = gpd.read_file(rgi)

    glacier = rgi[rgi["rgi_id"] == rgi_id]
    return glacier
