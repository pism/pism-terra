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

# pylint: disable=unused-import,unused-variable,broad-exception-caught

"""
Staging.
"""
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import cf_xarray
import geopandas as gpd
import pandas as pd
import rioxarray
import xarray as xr
from prefect import flow
from pyfiglet import Figlet
from shapely.geometry import Polygon

from pism_terra.climate import era5, pmip4
from pism_terra.dem import boot_file_from_rgi_id
from pism_terra.domain import create_grid
from pism_terra.raster import apply_perimeter_band
from pism_terra.vector import get_glacier_from_rgi_id
from pism_terra.workflow import check_dataset, check_xr

CLIMATE: Mapping[str, Callable] = {"pmip4": pmip4, "era5": era5}


def stage_glacier(
    config: dict,
    rgi_id: str,
    rgi: gpd.GeoDataFrame | str | Path = "rgi/rgi.gpkg",
    path: str | Path = "input_files",
    resolution: float = 50.0,
) -> pd.DataFrame:
    """
    Stage glacier inputs (boot, grid, outline, climate) and return a file index.

    For the glacier identified by ``rgi_id``, this function:
    (1) reads/loads the glacier geometry from an RGI GeoDataFrame or file,
    (2) generates a boot (DEM-derived) dataset and target model grid,
    (3) applies small perimeter masks/cleanups,
    (4) writes boot and grid NetCDFs plus an outline/domain file,
    (5) fetches and writes climate forcing using the configured climate builder,
    and (6) returns a tidy table (one row per climate file) with absolute paths.

    Parameters
    ----------
    config : dict
        Configuration mapping. Must contain at least:
        - ``"dem"`` : str
            Name of the DEM source to use in ``boot_file_from_rgi_id``.
        - ``"climate"`` : str
            Key for the climate builder in ``CLIMATE`` (e.g., ``"pmip4"``).
    rgi_id : str
        Glacier identifier (e.g., ``"RGI2000-v7.0-C-06-00014"``).
    rgi : geopandas.GeoDataFrame or str or pathlib.Path, default ``"rgi/rgi.gpkg"``
        Either an in-memory RGI GeoDataFrame, or path to a GeoPackage/shape
        readable by :func:`geopandas.read_file`.
    path : str or pathlib.Path, default ``"input_files"``
        Output directory. Created if missing. Files are written here.
    resolution : float, default ``50.0``
        Target grid resolution (meters) used to name outputs and guide grid
        generation where applicable.

    Returns
    -------
    pandas.DataFrame
        Table with one row per produced **climate** file and columns:
        ``rgi_id``, ``outline`` (gpkg), ``boot_file`` (nc),
        ``grid_file`` (nc), ``climate_file`` (nc). All paths are absolute.

    Raises
    ------
    KeyError
        If required keys (e.g., ``"dem"``, ``"climate"``) are missing in ``config``.
    FileNotFoundError
        If the provided RGI path does not exist.
    ValueError
        If ``rgi_id`` is not found in the provided RGI layer.
    Exception
        Propagated from helper functions (e.g., I/O or projection errors).

    See Also
    --------
    boot_file_from_rgi_id :
        Builds the boot (DEM/thickness/bed) dataset around the glacier.
    create_grid :
        Creates the target model grid and bounds.
    CLIMATE :
        Mapping from climate name (e.g., ``"pmip4"``) to a function that
        generates climate NetCDF(s) for the glacier/bounds.

    Notes
    -----
    - Applies a perimeter band via :func:`apply_perimeter_band` to clean edges.
    - Ensures bed is below surface and thickness is non-negative.
    - The returned DataFrame is convenient for downstream workflow fan-out.

    Examples
    --------
    >>> cfg = {"dem": "cop30", "climate": "pmip4"}
    >>> df = stage_glacier(cfg, "RGI2000-v7.0-C-06-00014", path="inputs", resolution=100)
    >>> df[["boot_file", "grid_file", "climate_file"]].head()
    """
    # Banner
    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    print("=" * 80)
    print(banner)
    print("=" * 80)
    print(f"Stage Glacier {rgi_id}")
    print("-" * 80)
    print("")

    # Outputs dir
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Load RGI (accept GeoDataFrame or file path)
    if isinstance(rgi, (str, Path)):
        rgi = gpd.read_file(rgi)

    glacier = get_glacier_from_rgi_id(rgi, rgi_id)
    if glacier.empty:
        raise ValueError(f"RGI ID not found: {rgi_id}")

    glacier_filename = path / f"rgi_{rgi_id}.gpkg"
    glacier_series = glacier.iloc[0]
    crs = glacier_series["epsg"]
    glacier.to_file(glacier_filename)

    # Output filenames
    boot_filename = path / f"bootfile_g{int(resolution)}m_{rgi_id}.nc"
    grid_filename = path / f"grid_g{int(resolution)}m_{rgi_id}.nc"

    # Build boot dataset (DEM/thickness/bed)
    boot_ds = boot_file_from_rgi_id(rgi_id, rgi, buffer_distance=5000.0, dem_name=config["dem"])

    # Grid & bounds
    grid_ds = create_grid(glacier, boot_ds, crs=crs, buffer_distance=2500.0)
    bounds = [
        grid_ds["x_bnds"].values[0][0],
        grid_ds["y_bnds"].values[0][0],
        grid_ds["x_bnds"].values[0][1],
        grid_ds["y_bnds"].values[0][1],
    ]

    # Edge cleanup and simple physical constraints
    for v in ["bed", "thickness", "surface"]:
        boot_ds[v] = apply_perimeter_band(boot_ds[v], bounds=bounds)
    boot_ds["thickness"] = boot_ds["thickness"].where(boot_ds["thickness"] > 0.0, 0.0)
    boot_ds["bed"] = boot_ds["bed"].where(boot_ds["surface"] > 0.0, -1000.0)
    boot_ds.rio.write_crs(crs, inplace=True)
    boot_ds.to_netcdf(boot_filename)

    grid_ds.attrs.update({"domain": rgi_id})
    grid_ds.to_netcdf(grid_filename, engine="h5netcdf")

    # Save domain extent polygon as a GPKG
    x_point_list = [
        grid_ds.x_bnds[0][0],
        grid_ds.x_bnds[0][0],
        grid_ds.x_bnds[0][1],
        grid_ds.x_bnds[0][1],
        grid_ds.x_bnds[0][0],
    ]
    y_point_list = [
        grid_ds.y_bnds[0][0],
        grid_ds.y_bnds[0][1],
        grid_ds.y_bnds[0][1],
        grid_ds.y_bnds[0][0],
        grid_ds.y_bnds[0][0],
    ]
    domain_bounds_geom = Polygon(zip(x_point_list, y_point_list))
    domain_bounds = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[domain_bounds_geom])
    domain_bounds_filename = path / f"domain_{rgi_id}.gpkg"
    domain_bounds.to_file(domain_bounds_filename)

    # Climate forcing
    climate_from_rgi = CLIMATE[config["climate"]]
    responses = climate_from_rgi(rgi_id=rgi_id, rgi=rgi, path=path)  # list[Path]
    # Normalize to list[Path]
    if isinstance(responses, (str, Path)):
        responses = [Path(responses)]
    else:
        responses = [Path(p) for p in responses]

    # Build file index (one row per climate file)
    files_dict = {
        "rgi_id": rgi_id,
        "outline": glacier_filename.absolute(),
        "boot_file": boot_filename.absolute(),
        "grid_file": grid_filename.absolute(),
    }
    dfs: list[pd.DataFrame] = []
    for fpath in responses:
        row = {**files_dict, "climate_file": Path(fpath).absolute()}
        dfs.append(pd.DataFrame.from_dict([row]))

    df = pd.concat(dfs).reset_index(drop=True)
    return df


def main():
    """
    Run main script.
    """

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

    glacier_df = stage_glacier(rgi_id, rgi, path=input_path)
    glacier_df.to_csv(input_path / Path(f"{rgi_id}.csv"))


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
