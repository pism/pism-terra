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
S4F Mission Planning.

Methods to assist with Snow4Flow Mission Planning.
"""

import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import cf_xarray
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from pyfiglet import Figlet
from shapely.geometry import Point, Polygon, box

from pism_terra.aws import download_from_s3, local_to_s3
from pism_terra.config import load_config
from pism_terra.domain import create_domain, get_bounds_from_geometry
from pism_terra.glacier.climate import (
    create_offset_file,
    create_step_file,
    era5,
    pmip4,
    snap,
)
from pism_terra.glacier.dem import boot_file_from_grid
from pism_terra.raster import apply_perimeter_band
from pism_terra.vector import (
    get_glacier_from_rgi_id,
    grid_cells_from_dataset,
    grid_points_from_dataset,
)
from pism_terra.workflow import check_dataset_fully, check_xr_fully, check_xr_lazy

xr.set_options(keep_attrs=True)

CLIMATE: Mapping[str, Callable] = {"pmip4": pmip4, "era5": era5, "snap": snap, "abrupt": snap}
MODIFIER: Mapping[str, Callable] = {
    "pmip4": create_offset_file,
    "era5": create_offset_file,
    "snap": create_offset_file,
    "abrupt": create_step_file,
}


def s4f_glacier(
    config: dict,
    rgi_id: str,
    path: str | Path = "input_files",
    staging_path: str | Path | None = None,
    resolution: float = 100.0,
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
    path : str or pathlib.Path, default ``"input_files"``
        Final output directory. Created if missing. Holds only the
        Cloud Optimized GeoTIFF outputs (one per spatial variable in the boot
        dataset).
    staging_path : str or pathlib.Path or None, optional
        Working directory for intermediate files (RGI table cache, glacier
        outline GPKG, DEM tifs, ice-thickness/velocity intermediates). Created
        if missing. If ``None`` (default), falls back to ``path`` (legacy
        behavior — everything in one directory).
    resolution : float, default ``100.0``
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
    print("=" * 120)
    print(banner)
    print("=" * 120)
    print(f"S4F Planning Glacier {rgi_id}")
    print("-" * 120)
    print("")

    # Output dirs: `path` holds only COGs; `staging_path` holds intermediates.
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    staging_path = Path(staging_path) if staging_path is not None else path
    staging_path.mkdir(parents=True, exist_ok=True)

    print("RGI Database")
    rgi_s3_uri = f"""s3://{config["bucket"]}/{config["prefix"]}/rgi/{config["rgi_file"]}"""
    rgi_local = staging_path / config["rgi_file"]
    if not rgi_local.exists():
        print(f"Downloading {rgi_s3_uri} -> {rgi_local}")
        download_from_s3(rgi_s3_uri, rgi_local)
    else:
        print(f"Using cached {rgi_local}")
    rgi = gpd.read_file(rgi_local)

    glacier = get_glacier_from_rgi_id(rgi, rgi_id)
    if glacier.empty:
        raise ValueError(f"RGI ID not found: {rgi_id}")

    glacier_file = staging_path / f"rgi_{rgi_id}.fgb"
    dst_crs = glacier["epsg"].values[0]
    glacier_projected = glacier.to_crs(dst_crs)
    glacier.to_file(glacier_file)

    x_bnds, y_bnds = get_bounds_from_geometry(glacier_projected.geometry, buffer_dist=2_000.0, dx=1_000.0)
    grid_ds = create_domain(x_bnds, y_bnds, resolution=resolution, crs=dst_crs)

    # Build boot dataset (DEM/thickness/bed) — caches go to staging
    boot_ds = boot_file_from_grid(
        grid_ds,
        rgi_id,
        glacier_projected.geometry,
        dem_dataset=config["dem"],
        ice_thickness_dataset=config["ice_thickness"],
        velocity_dataset=config["velocity"],
        path=staging_path,
        force_overwrite=force_overwrite,
        bucket=config["bucket"],
        prefix=config["prefix"],
    )

    print("")
    print("Saving Cloud Optimized GeoTIFFs")
    print("-" * 120)
    for var in boot_ds.data_vars:
        da = boot_ds[var]
        if not {"x", "y"}.issubset(set(da.dims)):
            continue
        cog_path = path / f"{rgi_id}_{var}.tif"
        out = da.astype("uint8") if da.dtype == bool else da
        out.rio.to_raster(cog_path, driver="COG", compress="DEFLATE")
        print(cog_path)
        if var == "surface":
            cog_clipped_path = path / f"{rgi_id}_{var}_clipped.tif"
            out_clipped = out.rio.clip(glacier_projected.geometry, drop=False)
            out_clipped.rio.to_raster(cog_clipped_path, driver="COG", compress="DEFLATE")
            print(cog_clipped_path)
        if var == "bed":
            nc_path = path / f"{rgi_id}_{var}.nc"
            out.to_netcdf(nc_path)


def main():
    """
    Run main script.
    """

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Stage RGI Glacier."
    parser.add_argument("--bucket", help="AWS S3 Bucket to upload output files to")
    parser.add_argument(
        "--bucket-prefix",
        help="AWS prefix (location in bucket) to add to product files",
        default="",
    )
    parser.add_argument(
        "--output-path",
        help="Path to save all files.",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--force-overwrite",
        help="Force downloading all files.",
        action="store_true",
        default=False,
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

    path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_file)
    config = cfg.campaign.as_params()

    print("RGI Database")
    rgi_s3_uri = f"""s3://{config["bucket"]}/{config["prefix"]}/rgi/{config["rgi_file"]}"""
    rgi_local = path / config["rgi_file"]
    if not rgi_local.exists():
        print(f"Downloading {rgi_s3_uri} -> {rgi_local}")
        download_from_s3(rgi_s3_uri, rgi_local)
    else:
        print(f"Using cached {rgi_local}")
    rgi = gpd.read_file(rgi_local)

    rgi_cloud = path / config["rgi_file"].replace("gpkg", "fgb")
    print(rgi_cloud)
    rgi.to_file(rgi_cloud)

    for rgi_id in rgi.rgi_id:
        glacier_path = path / Path(rgi_id)
        glacier_path.mkdir(parents=True, exist_ok=True)

        input_path = glacier_path / Path("input")
        input_path.mkdir(parents=True, exist_ok=True)
        staging_path = glacier_path / Path("staging")
        staging_path.mkdir(parents=True, exist_ok=True)
        s4f_glacier(
            config,
            rgi_id,
            path=input_path,
            staging_path=staging_path,
            force_overwrite=force_overwrite,
        )

    print(config)
    bucket = config["bucket"]
    prefix = config["prefix"]
    print("Now run")
    print(f"""aws s3 sync {path} s3://{bucket}/{prefix}/planning --exclude "*/staging/*" """)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
