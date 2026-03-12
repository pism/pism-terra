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


def glaciers_in_complex(rgi_c_id: str, rgi_g: gpd.GeoDataFrame) -> list:
    """
    Return the glacier outline IDs that belong to a given complex outline.

    Parameters
    ----------
    rgi_c_id : str
        The complex outline identifier (e.g. ``"RGI2000-v7.0-C-01-09429"``).
    rgi_g : geopandas.GeoDataFrame
        Glacier outline dataframe with an ``rgi_id_c`` column mapping each
        glacier to its parent complex.

    Returns
    -------
    list
        List of ``rgi_id`` strings whose ``rgi_id_c`` matches *rgi_c_id*.
    """
    return rgi_g.loc[rgi_g["rgi_id_c"] == rgi_c_id, "rgi_id"].tolist()


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


def dissolve(ds, date, crs: str = "EPSG:3413"):
    """
    Dissolve geometries.

    Parameters
    ----------
    ds : geopandas.GeoDataFrame
        The GeoDataFrame containing geometries to dissolve.
    date : pd.Timestamp
        The date associated with the geometries.
    crs : str, optional
        Coordinate reference system, by default "EPSG:3413".

    Returns
    -------
    geopandas.GeoDataFrame
        The dissolved GeoDataFrame with the date set as the index.
    """
    ds = gpd.GeoDataFrame(ds, crs=crs)
    geom_valid = ds.geometry.make_valid()
    ds.geometry = geom_valid
    ds = ds.dissolve()
    ds["Date"] = date
    ds = ds.set_index("Date")
    return ds


def aggregate(n, df):
    """
    Aggregate geometries.

    Parameters
    ----------
    n : int
        The number of geometries to aggregate.
    df : geopandas.GeoDataFrame
        The GeoDataFrame containing geometries to aggregate.

    Returns
    -------
    geopandas.GeoDataFrame
        The aggregated GeoDataFrame.
    """
    if n == 0:
        return df.iloc[[n]]
    else:
        geom = df.iloc[range(n)].unary_union
        merged_df = df.iloc[[n]]
        merged_df.iloc[0].geometry = geom
        return merged_df
