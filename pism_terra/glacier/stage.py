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

# pylint: disable=unused-import,unused-variable,broad-exception-caught,too-many-positional-arguments

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
from pyfiglet import Figlet
from shapely.geometry import Polygon

from pism_terra.climate import create_offset_file, era5, pmip4, snap
from pism_terra.config import load_config
from pism_terra.dem import boot_file_from_rgi_id
from pism_terra.domain import create_grid
from pism_terra.raster import apply_perimeter_band
from pism_terra.vector import get_glacier_from_rgi_id
from pism_terra.workflow import check_dataset, check_xr

xr.set_options(keep_attrs=True)

CLIMATE: Mapping[str, Callable] = {"pmip4": pmip4, "era5": era5, "snap": snap}


def stage_glacier(
    config: dict,
    rgi_id: str,
    rgi: gpd.GeoDataFrame | str | Path,
    path: str | Path = "input_files",
    resolution: float = 50.0,
    force_overwrite: bool = False,
) -> pd.DataFrame:
    """
    Stage glacier inputs (boot, grid, outline, climate) and return a file index.

    For the glacier identified by ``rgi_id``, this function:
    (1) loads the glacier geometry (GeoDataFrame or GPKG),
    (2) builds a DEM/thickness/bed “boot” dataset,
    (3) creates a target model grid and derives simple perimeter masks,
    (4) writes the boot and grid NetCDF files and the glacier outline/domain bounds as GPKG,
    (5) generates climate forcing files using the configured climate builder,
    and (6) returns a tidy table (one row per **climate** file) with absolute paths.

    Parameters
    ----------
    config : dict
        Configuration mapping. Must contain at least:
        - ``"dem"`` : str
            DEM source passed to :func:`boot_file_from_rgi_id`.
        - ``"climate"`` : str
            Key in :data:`CLIMATE` (e.g., ``"pmip4"``) selecting the climate builder.
    rgi_id : str
        Glacier identifier (e.g., ``"RGI2000-v7.0-C-06-00014"``).
    rgi : geopandas.GeoDataFrame or str or pathlib.Path, default ``"rgi/rgi.gpkg"``
        In-memory RGI GeoDataFrame or a path to a GeoPackage/shape readable by
        :func:`geopandas.read_file`.
    path : str or pathlib.Path, default ``"input_files"``
        Output directory. Created if missing. All staged artifacts are written here.
    resolution : float, default ``50.0``
        Target grid resolution (meters), used both for grid construction and in
        output filenames.
    force_overwrite : bool, default ``False``
        If ``True``, downstream helpers may regenerate intermediate/final artifacts
        even if cache files exist (e.g., passed to :func:`boot_file_from_rgi_id`
        and to the selected climate builder via :data:`CLIMATE`).

    Returns
    -------
    pandas.DataFrame
        One row per produced **climate** file, with absolute-path columns:
        ``rgi_id``, ``outline`` (GPKG), ``boot_file`` (NetCDF),
        ``grid_file`` (NetCDF), ``climate_file`` (NetCDF).

    Raises
    ------
    KeyError
        If required keys (e.g., ``"dem"``, ``"climate"``) are missing in ``config``.
    FileNotFoundError
        If an RGI path is provided and does not exist.
    ValueError
        If ``rgi_id`` is not found in the RGI layer or geometry/CRS is invalid.
    Exception
        Propagated errors from DEM/thickness preparation, reprojection, or I/O.

    See Also
    --------
    boot_file_from_rgi_id
        Builds the boot (DEM, thickness, bed, masks) dataset around the glacier.
    create_grid
        Creates the target model grid and bounds.
    CLIMATE
        Mapping from climate name (e.g., ``"pmip4"``) to a function that generates
        climate NetCDF file(s) for the glacier domain.

    Notes
    -----
    - Applies :func:`apply_perimeter_band` to clean DEM edges.
    - Enforces simple constraints (non-negative thickness; bed below surface).
    - Writes two vector layers:
        - Glacier outline: ``rgi_{rgi_id}.gpkg`` (same CRS as RGI entry).
        - Domain bounds polygon: ``domain_{rgi_id}.gpkg``.
    - The returned DataFrame is convenient for downstream orchestration/fan-out.
    """

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

    glacier_file = path / f"rgi_{rgi_id}.gpkg"
    glacier_series = glacier.iloc[0]
    crs = glacier_series["epsg"]
    glacier.to_file(glacier_file)

    # Output filenames
    boot_file = path / f"bootfile_g{int(resolution)}m_{rgi_id}.nc"
    grid_file = path / f"grid_g{int(resolution)}m_{rgi_id}.nc"

    # Build boot dataset (DEM/thickness/bed)
    boot_ds = boot_file_from_rgi_id(
        rgi_id, rgi, buffer_distance=5000.0, dem_name=config["dem"], path=path, force_overwrite=force_overwrite
    )

    # Grid & bounds
    grid_ds = create_grid(glacier, boot_ds, crs=crs, buffer_distance=2500.0)
    bounds = [
        grid_ds["x_bnds"].values[0][0],
        grid_ds["y_bnds"].values[0][0],
        grid_ds["x_bnds"].values[0][1],
        grid_ds["y_bnds"].values[0][1],
    ]

    # Edge cleanup and simple physical constraints
    for v in ["bed"]:
        boot_ds[v] = apply_perimeter_band(boot_ds[v], bounds=bounds)
    for v in ["surface"]:
        boot_ds[v] = apply_perimeter_band(boot_ds[v], bounds=bounds, value=0.0)
    boot_ds["thickness"] = boot_ds["thickness"].where(boot_ds["thickness"] > 0.0, 0.0)
    boot_ds.rio.write_crs(crs, inplace=True)
    if hasattr(boot_ds["spatial_ref"], "GeoTransform"):
        del boot_ds["spatial_ref"].attrs["GeoTransform"]
    for name in ("x", "y", "thickness", "bed", "surface", "tillwat", "ftt_mask", "land_ice_area_fraction_retreat"):
        if name in boot_ds:
            boot_ds[name].encoding.update({"_FillValue": None})

    print("")
    print("Saving bootfile")
    print("-" * 80)
    print(boot_file.resolve())
    boot_ds.to_netcdf(boot_file)

    grid_ds.attrs.update({"domain": rgi_id})
    grid_ds.to_netcdf(grid_file, engine="h5netcdf")

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
    domain_bounds_file = path / f"domain_{rgi_id}.gpkg"
    domain_bounds.to_file(domain_bounds_file)

    scalar_offset_file = path / Path(f"scalar_offset_{rgi_id}_id_0.nc")
    create_offset_file(scalar_offset_file, delta_T=0.0, frac_P=0.0)

    # Climate forcing
    climate_from_rgi = CLIMATE[config["climate"]]
    responses = climate_from_rgi(rgi_id=rgi_id, rgi=rgi, path=path, force_overwrite=force_overwrite)  # list[Path]
    # Normalize to list[Path]
    if isinstance(responses, (str, Path)):
        responses = [Path(responses)]
    else:
        responses = [Path(p) for p in responses]

    # Build file index (one row per climate file)
    files_dict = {
        "rgi_id": rgi_id,
        "outline": glacier_file.resolve(),
        "boot_file": boot_file.resolve(),
        "grid_file": grid_file.resolve(),
        "scalar_offset_file": scalar_offset_file.resolve(),
    }
    dfs: list[pd.DataFrame] = []
    for fpath in responses:
        row = {**files_dict, "climate_file": Path(fpath).resolve()}
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
        "--output-path",
        help="Path to save all files.",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--force-overwrite",
        help="Force downloading all files.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "RGI_ID",
        help="RGI ID.",
        nargs=1,
    )
    parser.add_argument(
        "RGI_FILE",
        help="RGI.",
        nargs=1,
    )
    parser.add_argument(
        "CONFIG_FILE",
        help="CONFIG TOML.",
        nargs=1,
    )

    options, unknown = parser.parse_known_args()
    path = options.output_path
    config_file = options.CONFIG_FILE[0]
    force_overwrite = options.force_overwrite
    rgi_file = options.RGI_FILE[0]
    rgi_id = options.RGI_ID[0]

    cfg = load_config(config_file)
    config = cfg.campaign.as_params()

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    glacier_path = path / Path(rgi_id)
    glacier_path.mkdir(parents=True, exist_ok=True)

    input_path = glacier_path / Path("input")
    input_path.mkdir(parents=True, exist_ok=True)
    glacier_df = stage_glacier(config, rgi_id, rgi_file, path=glacier_path, force_overwrite=force_overwrite)
    glacier_df.to_csv(input_path / Path(f"{rgi_id}.csv"))


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
