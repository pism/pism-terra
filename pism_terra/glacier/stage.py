# Copyright (C) 2025-26 Andy Aschwanden
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

import shutil
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
import pyogrio
import rioxarray
import xarray as xr
from pyfiglet import Figlet
from shapely.geometry import Point, Polygon, box

from pism_terra.aws import download_from_s3, local_to_s3
from pism_terra.config import load_config
from pism_terra.domain import create_domain, get_bounds_from_geometry
from pism_terra.glacier.climate import (
    carra2,
    create_offset_file,
    create_step_file,
    era5,
    era5_mean,
    era5_monthly_mean,
    snap,
)
from pism_terra.glacier.dem import boot_file_from_grid
from pism_terra.glacier.observations import glacier_velocities_from_grid
from pism_terra.heat_flow import heat_flow_from_grid
from pism_terra.raster import apply_perimeter_band
from pism_terra.vector import get_glacier_from_rgi_id, glaciers_in_complex
from pism_terra.workflow import (
    check_dataset_fully,
    check_xr_fully,
    check_xr_lazy,
    drop_geotransform_attr,
)

xr.set_options(keep_attrs=True)

CLIMATE: Mapping[str, Callable] = {
    "carra2": carra2,
    "era5": era5,
    "era5-mean": era5_mean,
    "era5-monthly-mean": era5_monthly_mean,
    "snap": snap,
}
MODIFIER: Mapping[str, Callable] = {
    "era5": create_offset_file,
    "snap": create_offset_file,
}


def stage_glacier(
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
            DEM source passed to :func:`boot_file_from_grid`.
        - ``"climate"`` : str
            Key in :data:`CLIMATE` (e.g., ``"pmip4"``) selecting the climate builder.
    rgi_id : str
        Glacier identifier (e.g., ``"RGI2000-v7.0-C-06-00014"``).
    path : str or pathlib.Path, default ``"input_files"``
        Final output directory. Created if missing. Holds the artifacts that
        downstream tooling consumes: glacier outline GPKG, boot NetCDF,
        grid NetCDF, and climate forcing NetCDF.
    staging_path : str or pathlib.Path or None, optional
        Working directory for intermediate files (RGI table cache, DEM tifs,
        ice-thickness/velocity intermediates, ERA5/PMIP4 raw downloads,
        debug GPKGs, domain-bounds polygon). Created if missing. If ``None``
        (default), falls back to ``path`` (legacy behavior — everything in
        one directory).
    resolution : float, default ``100.0``
        Target grid resolution (meters), used both for grid construction and in
        output filenames.
    force_overwrite : bool, default ``False``
        If ``True``, downstream helpers may regenerate intermediate/final artifacts
        even if cache files exist (e.g., passed to :func:`boot_file_from_grid`
        and to the selected climate builder via :data:`CLIMATE`).

    Returns
    -------
    pandas.DataFrame
        One row per produced **climate** file, with absolute-path columns:
        ``rgi_id``, ``outline_file`` (GPKG), ``boot_file`` (NetCDF),
        ``grid_file`` (NetCDF), ``climate_file`` (NetCDF), and ``sample`` (int).

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
    boot_file_from_grid
        Builds the boot (DEM, thickness, bed, masks) dataset around the glacier.
    create_domain
        Creates the target model grid and bounds.
    CLIMATE
        Mapping from climate name (e.g., ``"pmip4"``) to a function that generates
        climate NetCDF file(s) for the glacier domain.
    """

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    print("=" * 120)
    print(banner)
    print("=" * 120)
    print(f"Stage Glacier {rgi_id}")
    print("-" * 120)
    print("")

    # Output dirs: `path` holds final artifacts; `staging_path` holds intermediates.
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    staging_path = Path(staging_path) if staging_path is not None else path
    staging_path.mkdir(parents=True, exist_ok=True)

    print("RGI Database")
    rgi_glacier_s3_uri = f"""s3://{config["bucket"]}/{config["prefix"]}/rgi/{config["rgi_glacier_file"]}"""
    rgi_glacier_local = staging_path / config["rgi_glacier_file"]
    if not rgi_glacier_local.exists():
        print(f"Downloading {rgi_glacier_s3_uri} -> {rgi_glacier_local}")
        download_from_s3(rgi_glacier_s3_uri, rgi_glacier_local)
    else:
        print(f"Using cached {rgi_glacier_local}")

    rgi_complex_s3_uri = f"""s3://{config["bucket"]}/{config["prefix"]}/rgi/{config["rgi_complex_file"]}"""
    rgi_complex_local = staging_path / config["rgi_complex_file"]
    if not rgi_complex_local.exists():
        print(f"Downloading {rgi_complex_s3_uri} -> {rgi_complex_local}")
        download_from_s3(rgi_complex_s3_uri, rgi_complex_local)
    else:
        print(f"Using cached {rgi_complex_local}")

    # NOTE: gpd.read_file/to_file (via pyogrio's geopandas wrapper) corrupts the
    # heap on some envs and crashes the next libgdal allocation (e.g. inside
    # dem_stitcher). Calling pyogrio directly avoids the trigger.
    rgi_complex = pyogrio.read_dataframe(rgi_complex_local, use_arrow=False)
    glacier = get_glacier_from_rgi_id(rgi_complex, rgi_id)
    if glacier.empty:
        raise ValueError(f"RGI ID not found: {rgi_id}")

    glacier_complex_file = path / f"rgi_{rgi_id}-C.gpkg"
    dst_crs = glacier["crs"].values[0]
    glacier_projected = glacier.to_crs(dst_crs)
    pyogrio.write_dataframe(glacier, glacier_complex_file)

    # Extract the individual glacier outlines that make up this complex and
    # write them to the "-G" file. Membership is given by the glacier-level
    # "rgi_id_c" column (handled by glaciers_in_complex, incl. aggregates).
    glacier_file = path / f"rgi_{rgi_id}-G.gpkg"
    rgi_glacier = pyogrio.read_dataframe(rgi_glacier_local, use_arrow=False)
    glacier_ids = glaciers_in_complex(rgi_id, rgi_glacier)
    glaciers = rgi_glacier[rgi_glacier["rgi_id"].isin(glacier_ids)]
    if glaciers.empty:
        print(f"Warning: no glacier outlines found for complex {rgi_id}")
    pyogrio.write_dataframe(glaciers, glacier_file)

    x_bnds, y_bnds = get_bounds_from_geometry(glacier_projected.geometry, buffer_dist=5_000.0, dx=1_000.0)
    grid_ds = create_domain(x_bnds, y_bnds, resolution=resolution, crs=dst_crs)

    # Output filenames
    boot_file = path / f"bootfile_{rgi_id}.nc"
    grid_file = path / f"grid_{rgi_id}.nc"
    obs_file = path / f"obs_{rgi_id}.nc"
    bheatflux_file = path / f"bheatflux_{rgi_id}.nc"

    # Build boot dataset (DEM/thickness/bed) — caches go to staging
    boot_ds = boot_file_from_grid(
        grid_ds,
        rgi_id,
        glacier_projected.geometry,
        dem_dataset=config["dem"],
        ice_thickness_dataset=config["ice_thickness"],
        velocity_dataset=config["velocity"],
        bathymetry_dataset=config["bathymetry"],
        forcing_mask=config["forcing_mask"],
        path=staging_path,
        force_overwrite=force_overwrite,
        bucket=config["bucket"],
        prefix=config["prefix"],
    )

    print("")
    print("Saving bootfile")
    print("-" * 120)
    boot_file.unlink(missing_ok=True)
    # rioxarray writes a GeoTransform on spatial_ref consistent with ascending y
    # (positive dy); GDAL then prefers that over the y coordinate variable and
    # displays the raster upside-down in QGIS. Drop it so GDAL falls back to
    # deriving the transform from the y coordinate variable (top-down).
    drop_geotransform_attr(boot_ds)
    boot_ds.to_netcdf(boot_file, engine="h5netcdf")
    check_xr_lazy(boot_file)

    heat_flow_ds = heat_flow_from_grid(
        grid_ds,
        dataset=config["heatflux"],
        path=bheatflux_file,
        bucket=config["bucket"],
        prefix=config["prefix"],
        force_overwrite=force_overwrite,
    )
    check_xr_fully(bheatflux_file)

    grid_ds.attrs.update({"domain": rgi_id})
    grid_file.unlink(missing_ok=True)
    drop_geotransform_attr(grid_ds)
    grid_ds.to_netcdf(grid_file, engine="h5netcdf")
    check_xr_fully(grid_file)

    _ = glacier_velocities_from_grid(grid_ds, glacier_projected.geometry, path=obs_file)
    check_xr_fully(obs_file)

    # Save domain extent polygon as a GPKG (intermediate, used for sanity checks)
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
    domain_bounds = gpd.GeoDataFrame(index=[0], crs=dst_crs, geometry=[domain_bounds_geom])
    domain_bounds_file = staging_path / f"domain_{rgi_id}.gpkg"
    pyogrio.write_dataframe(domain_bounds, domain_bounds_file)

    clim_mod = config["climate"]
    # Climate forcing — built into staging, then final outputs moved to `path`
    climate_from_rgi = CLIMATE[config["climate"]]
    responses = climate_from_rgi(
        grid_ds,
        rgi_id=rgi_id,
        years=config["years"],
        path=staging_path,
        bucket=config["bucket"],
        prefix=config["prefix"],
        force_overwrite=force_overwrite,
    )  # list[Path]
    # Normalize to list[Path]
    if isinstance(responses, (str, Path)):
        responses = [Path(responses)]
    else:
        responses = [Path(p) for p in responses]
    if staging_path.resolve() != path.resolve():
        # Copy (not move) the final climate file(s) into the input dir, leaving
        # the builder's output in staging so its own cache guard short-circuits
        # on reruns. Moving deletes the staging copy, so the climate would be
        # regenerated/re-downloaded every run even though it already exists in
        # the input dir. (Staging is excluded from the S3 upload.)
        copied: list[Path] = []
        for src in responses:
            dst = path / src.name
            if src.resolve() != dst.resolve():
                dst.unlink(missing_ok=True)
                shutil.copy2(str(src), str(dst))
            copied.append(dst)
        responses = copied

    # Build file index (one row per climate file)
    files_dict = {
        "rgi_id": rgi_id,
        "outline_file": glacier_file.resolve(),
        "boot_file": boot_file.resolve(),
        "grid_file": grid_file.resolve(),
        "heatflux_file": bheatflux_file.resolve(),
        "obs_file": obs_file.resolve(),
    }
    dfs: list[pd.DataFrame] = []
    for idx, fpath in enumerate(responses):
        row = {**files_dict, "climate_file": Path(fpath).resolve(), "sample": idx}
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
        default=Path("."),
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
        "CONFIG_FILE",
        help="CONFIG TOML.",
        nargs=1,
    )

    options, unknown = parser.parse_known_args()
    path = options.output_path
    config_file = options.CONFIG_FILE[0]
    force_overwrite = options.force_overwrite
    rgi_id = options.RGI_ID[0]

    cfg = load_config(config_file)
    # Cover every calendar year touched by the simulation. PISM's time.end is
    # exclusive (e.g. "2025-01-01" means stop at midnight Jan 1, so 2025
    # itself is not simulated), hence the - 1 when end is exactly Jan 1.
    start = pd.Timestamp(cfg.time.time_start)
    end = pd.Timestamp(cfg.time.time_end)
    last_year = end.year - 1 if (end.month == 1 and end.day == 1) else end.year
    years = list(range(start.year, last_year + 1))
    config = cfg.campaign.as_params()
    config["years"] = years

    path.mkdir(parents=True, exist_ok=True)
    glacier_path = path / Path(rgi_id)
    glacier_path.mkdir(parents=True, exist_ok=True)

    input_path = glacier_path / Path("input")
    input_path.mkdir(parents=True, exist_ok=True)
    staging_path = glacier_path / Path("staging")
    staging_path.mkdir(parents=True, exist_ok=True)
    glacier_df = stage_glacier(
        config,
        rgi_id,
        path=input_path,
        staging_path=staging_path,
        force_overwrite=force_overwrite,
    )
    glacier_df.to_csv(input_path / Path(f"{rgi_id}.csv"))

    if options.bucket:
        prefix = f"{options.bucket}/{rgi_id}" if options.bucket_prefix else rgi_id
        local_to_s3(glacier_path, bucket=options.bucket, prefix=prefix)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
