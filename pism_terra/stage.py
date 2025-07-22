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

"""
Staging.
"""

from pathlib import Path

from pism_terra.dem import glacier_dem_from_rgi_id


def stage_glacier(
    rgi_id: str, rgi: str | Path = "rgi/rgi.gpkg", path: str | Path = "boot_files", resolution: float = 50.0
) -> str:
    """
    Generate and save a glacier DEM and related variables to a NetCDF file.

    This function stages a glacier dataset for use in modeling or analysis.
    It retrieves a glacier DEM, ice thickness, and bed topography based on a given
    RGI ID, and saves the resulting dataset as a NetCDF file in a specified directory.

    Parameters
    ----------
    rgi_id : str
        The RGI ID of the glacier to stage (e.g., "RGI2000-v7.0-C-06-00014").
    rgi : str or Path, optional
        Path to the RGI file (GeoPackage or shapefile). Default is "rgi/rgi.gpkg".
    path : str or Path, optional
        Directory where the staged NetCDF file will be saved. Created if it doesn't exist.
        Default is "boot_file".
    resolution : float, optional
        Resolution (in meters) for the target grid. Passed to the DEM generation function.
        Default is 50.0.

    Returns
    -------
    str
        Path to file.

    See Also
    --------
    glacier_dem_from_rgi_id : Generates the glacier dataset with DEM, thickness, and bed.

    Notes
    -----
    - Output dataset includes variables: `surface`, `thickness`, `bed`, and
      `land_ice_area_fraction_retreat`.
    - The staging directory is created if it doesn't exist.
    """

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    filename = path / Path(f"{rgi_id}_g{int(resolution)}m.nc")
    ds = glacier_dem_from_rgi_id(rgi_id, rgi)
    ds.to_netcdf(filename)

    return str(filename)
