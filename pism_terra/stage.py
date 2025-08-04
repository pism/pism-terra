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

# pylint: disable=unused-import

"""
Staging.
"""
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import cf_xarray
import geopandas as gpd
import pandas as pd
import rioxarray
import xarray as xr
from dask.distributed import Client
from shapely.geometry import Polygon

import pism_terra.interpolation
from pism_terra.climate import era5_reanalysis_from_rgi_id
from pism_terra.dem import add_malaspina_bed, glacier_dem_from_rgi_id
from pism_terra.domain import create_grid
from pism_terra.observations import glacier_velocities_from_rgi_id
from pism_terra.raster import apply_perimeter_band
from pism_terra.vector import get_glacier_from_rgi_id


def stage_glacier(
    rgi_id: str, rgi: str | Path = "rgi/rgi.gpkg", path: str | Path = "input_files", resolution: float = 50.0
) -> dict:
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

    print("=" * 80)
    print(f"Stage Glacier {rgi_id}")
    print("-" * 80)
    print("")

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    v_filename = path / Path(f"obs_velocities_g{int(resolution)}m_{rgi_id}.nc")
    v = glacier_velocities_from_rgi_id(rgi_id, rgi, buffer_distance=5000.0)
    v.to_netcdf(v_filename)

    boot_filename = path / Path(f"bootfile_g{int(resolution)}m_{rgi_id}.nc")
    dem = glacier_dem_from_rgi_id(rgi_id, rgi, buffer_distance=5000.0)
    crs = dem.rio.crs

    tillwat = xr.zeros_like(dem["surface"])
    tillwat.name = "tillwat"
    del tillwat.attrs["standard_name"]
    tillwat.attrs.update({"units": "m"})

    dem["tillwat"] = tillwat.where(v["v"].rio.reproject_match(dem).fillna(0) < 100.0, 2)

    glacier = get_glacier_from_rgi_id(rgi, rgi_id)
    glacier_filename = path / Path(f"rgi_{rgi_id}.gpkg")
    glacier.to_file(glacier_filename)

    grid = create_grid(glacier, dem, crs=crs, buffer_distance=2500.0)
    bounds = [
        grid["x_bnds"].values[0][0],
        grid["y_bnds"].values[0][0],
        grid["x_bnds"].values[0][1],
        grid["y_bnds"].values[0][1],
    ]

    for v in ["bed", "thickness", "surface"]:
        dem[v] = apply_perimeter_band(dem[v], bounds=bounds)
    dem["thickness"] = dem["thickness"].where(dem["thickness"] > 0.0, 0.0)
    dem["surface"] = dem["surface"].where(dem["thickness"] > 0.0, 0.0)
    if rgi_id == "RGI2000-v7.0-C-01-09429-A":
        dem = add_malaspina_bed(dem, target_crs=crs)
    dem.rio.write_crs(crs, inplace=True)
    dem.to_netcdf(boot_filename)

    grid.attrs.update({"domain": rgi_id})
    grid_filename = path / Path(f"grid_g{int(resolution)}m_{rgi_id}.nc")
    grid.to_netcdf(grid_filename, engine="h5netcdf")

    x_point_list = [grid.x_bnds[0][0], grid.x_bnds[0][0], grid.x_bnds[0][1], grid.x_bnds[0][1], grid.x_bnds[0][0]]
    y_point_list = [grid.y_bnds[0][0], grid.y_bnds[0][1], grid.y_bnds[0][1], grid.y_bnds[0][0], grid.y_bnds[0][0]]
    polygon_geom = Polygon(zip(x_point_list, y_point_list))
    polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom])
    polygon_filename = path / Path(f"domain_{rgi_id}.gpkg")
    polygon.to_file(polygon_filename)

    climate_filename = path / Path(f"era5_wgs84_{rgi_id}.nc")
    climate = era5_reanalysis_from_rgi_id(
        rgi_id, rgi, buffer_distance=0.2, dataset="reanalysis-era5-land-monthly-means"
    )
    print(f"Saving {climate_filename}")
    climate.to_netcdf(climate_filename)

    # climate_projected_filename = path / Path(f"era5_{rgi_id}.nc")
    # climate_projected = climate[["air_temp", "precipitation"]].rio.reproject_match(dem.thin({"x": 4, "y": 4}))

    # client = Client()
    # print(f"Open client in browser: {client.dashboard_link}")
    # start = time.time()
    # for v in ["precipitation", "air_temp"]:
    #     climate_projected[v] = climate_projected[v].utils.fillna(client=client)
    # end = time.time()
    # time_elapsed = end - start
    # print(f"Time elapsed {time_elapsed:.0f}s")

    # climate_projected["time_bounds"] = climate["time_bounds"]
    # print(f"Saving {climate_projected_filename}")
    # climate_projected.to_netcdf(climate_projected_filename)

    return {
        "boot_file": boot_filename.absolute(),
        "historical_climate_file": climate_filename.absolute(),
        "grid_file": grid_filename.absolute(),
    }


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Stage RGI Glacier."
    parser.add_argument(
        "--rgi_file",
        help="""Path to RGI file. Default="data/rgi/rgi.gpkg".""",
        type=str,
        default="data/rgi/rgi.gpkg",
    )
    parser.add_argument(
        "--output_path",
        help="""Path to save all files. Default="data".""",
        type=str,
        default="data",
    )
    parser.add_argument(
        "RGI_ID",
        help="""RGI ID.""",
        nargs=1,
    )

    options, unknown = parser.parse_known_args()
    path = options.output_path
    rgi = options.rgi_file
    rgi_id = options.RGI_ID[0]

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    glacier_path = path / Path(rgi_id)
    glacier_path.mkdir(parents=True, exist_ok=True)

    input_path = glacier_path / Path("input")
    input_path.mkdir(parents=True, exist_ok=True)

    glacier_dict = {"rgi_id": rgi_id}
    glacier_dict.update(stage_glacier(rgi_id, rgi, path=input_path))
    glacier_df = pd.DataFrame.from_dict([glacier_dict])
    glacier_df.to_csv(input_path / Path(f"{rgi_id}.csv"))
