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

# pylint: disable=too-many-positional-arguments,unused-import,broad-exception-caught
"""
Prepare RGI and S4F data sets.
"""

import logging
import os
import re
import shutil
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed as cf_as_completed
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import cf_xarray
import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr  # pylint: disable=unused-import
import toml
import xarray as xr
import xarray_regrid.methods.conservative  # pylint: disable=unused-import
from cdo import Cdo
from dask.diagnostics import ProgressBar
from dask.distributed import Client, as_completed
from pyfiglet import Figlet
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from tqdm.auto import tqdm

from pism_terra.download import (
    download_archive,
    download_file,
    download_gebco,
    extract_archive,
)
from pism_terra.glacier.climate import (
    convert_many_tifs_concurrent,
    prepare_carra2,
    prepare_carra2_for_group,
    prepare_glaciermip4,
    prepare_snap,
)
from pism_terra.glacier.ice_thickness import (
    prepare_ice_thickness_frank,
    prepare_ice_thickness_maffezzoli,
)
from pism_terra.glacier.rgi import prepare_rgi
from pism_terra.log import setup_logging
from pism_terra.vector import glaciers_in_complex
from pism_terra.workflow import check_xr_lazy

xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)


def s4f(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Prepare S4F glacier input data sets.

    This function is the programmatic entry point. It parses command-line style
    arguments, creates the target grid, downloads and processes observation data,
    and prepares climate/ocean forcing files for PISM simulations.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments **excluding** the program name (i.e., like
        ``sys.argv[1:]``). If ``None`` (default), arguments are taken from the
        current process' ``sys.argv[1:]``. Passing ``argv=[]`` is recommended
        when calling from a Jupyter notebook to avoid ipykernel arguments.

    Returns
    -------
    dict[str, Any]
        Mapping returned by :func:`prepare_rgi` (e.g. ``"rgi_complexes"``
        and ``"rgi_glaciers"`` paths).
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--force-overwrite",
        help="Force downloading all files.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ntasks",
        help="Parallel tasks.",
        type=int,
        default=8,
    )
    parser.add_argument("CONFIG_FILE", nargs=1)
    parser.add_argument("OUTPUT_PATH", nargs=1)
    parser.add_argument("GLACIER_FILES", nargs="*")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_file = args.CONFIG_FILE[0]
    glacier_files = args.GLACIER_FILES
    force_overwrite = args.force_overwrite
    ntasks = args.ntasks
    output_path = Path(args.OUTPUT_PATH[0])
    output_path.mkdir(parents=True, exist_ok=True)

    setup_logging(output_path / "prepare.log")

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    logger.info("=" * 120)
    logger.info("\n%s", banner)
    logger.info("=" * 120)
    logger.info("Preparing S4F data")
    logger.info("-" * 120)

    config = toml.loads(Path(config_file).read_text("utf-8"))
    regions = pd.DataFrame.from_dict(config["regions"], orient="index")
    regions["region"] = regions.index.astype(str).str.zfill(2) + "_" + regions["name"]

    glacier_groups: dict[str, pd.DataFrame] = {}  # aggregated-name -> CSV rows
    if len(glacier_files) > 0:
        for glacier_file in glacier_files:
            p = Path(glacier_file)
            # "S4F_target_AK_RGI_id.csv" -> "S4F_AK"; fall back to the file stem.
            m = re.match(r"^(?P<prefix>.+?)_target_(?P<region>.+?)_RGI_id$", p.stem)
            name = f"{m['prefix']}_{m['region']}" if m else p.stem
            glacier_groups[name] = pd.read_csv(p)

        glaciers = pd.concat(glacier_groups.values(), ignore_index=True)
        glaciers["o1regions"] = glaciers["rgi_id"].str.extract(r"-G-(\d{2})-")
        o1regions = glaciers["o1regions"].unique().astype(int).astype(str)
        regions = regions[regions.index.isin(o1regions)]
    else:
        glaciers = None
    # Final outputs land under glacier_path; everything intermediate lives in
    # staging_path so the user can `rm -rf staging` after a clean run without
    # losing anything that downstream tools need.
    glacier_path = output_path / Path("glacier")
    glacier_path.mkdir(parents=True, exist_ok=True)
    staging_path = output_path / Path("staging")
    staging_path.mkdir(parents=True, exist_ok=True)

    # --- RGI ---
    rgi_path = glacier_path / Path("rgi")
    rgi_path.mkdir(parents=True, exist_ok=True)
    rgi_staging = staging_path / Path("rgi")
    rgi_staging.mkdir(parents=True, exist_ok=True)

    rgi_files = prepare_rgi(
        regions,
        glaciers=glaciers,
        glacier_groups=glacier_groups or None,
        output_path=rgi_path,
        extract_path=rgi_staging,
        force_overwrite=force_overwrite,
        ntasks=ntasks,
    )

    complexes = gpd.read_file(rgi_files["rgi_complexes"])
    glaciers = gpd.read_file(rgi_files["rgi_glaciers"])

    # https://springernature.figshare.com/ndownloader/articles/29940932/versions/1
    # https://springernature.figshare.com/ndownloader/files/57288257

    # --- Ice thickness ---
    ice_thickness_path = glacier_path / Path("ice_thickness")
    ice_thickness_path.mkdir(parents=True, exist_ok=True)
    ice_thickness_staging = staging_path / Path("ice_thickness")
    ice_thickness_staging.mkdir(parents=True, exist_ok=True)

    frank_path = ice_thickness_path / Path("frank")
    frank_path.mkdir(parents=True, exist_ok=True)

    prepare_ice_thickness_frank(
        regions.index,
        complexes=complexes,
        glaciers=glaciers,
        output_path=frank_path,
        extract_path=ice_thickness_staging,
        rgi_extract_path=rgi_staging,
        force_overwrite=force_overwrite,
        ntasks=ntasks,
    )

    maffezzoli_path = ice_thickness_path / Path("maffezzoli")
    maffezzoli_path.mkdir(parents=True, exist_ok=True)

    prepare_ice_thickness_maffezzoli(
        regions.index,
        complexes=complexes,
        glaciers=glaciers,
        output_path=maffezzoli_path,
        extract_path=ice_thickness_staging,
        force_overwrite=force_overwrite,
        ntasks=ntasks,
    )

    # --- GEBCO ---
    # Source NetCDF download lands in staging; only the COG goes to glacier/.
    gebco_path = glacier_path / Path("gebco")
    gebco_path.mkdir(parents=True, exist_ok=True)
    gebco_staging = staging_path / Path("gebco")
    gebco_staging.mkdir(parents=True, exist_ok=True)
    gebco_nc = download_gebco(target_dir=gebco_staging)
    cog_gebco_p = gebco_path / Path("bathymetry.tif")

    # Use xr.open_dataset (CF-aware) so the lat/lon coords become a real
    # geotransform; rxr.open_rasterio treats netCDF as a generic raster and
    # loses the georeferencing.
    ds = xr.open_dataset(gebco_nc, chunks={"lat": 1024, "lon": 1024})
    da = ds["elevation"].rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326")
    predictor = 3 if np.issubdtype(da.dtype, np.floating) else 2
    da.rio.to_raster(
        cog_gebco_p,
        driver="COG",
        compress="DEFLATE",
        predictor=predictor,
        blocksize=512,
        bigtiff="YES",
        overview_resampling="AVERAGE",
        num_threads="ALL_CPUS",
    )

    # --- Climate (CARRA2) ---
    # Run the download/merge under staging, then move only the merged product
    # into glacier/climate. Year-by-year CDS intermediates stay in staging.
    climate_path = glacier_path / Path("climate")
    climate_path.mkdir(parents=True, exist_ok=True)

    glaciermip4_staging = staging_path / Path("glaciermip4")
    glaciermip4_staging.mkdir(parents=True, exist_ok=True)
    prepare_glaciermip4(glaciermip4_staging)

    carra2_staging = staging_path / Path("carra2")
    carra2_staging.mkdir(parents=True, exist_ok=True)

    carra2_staging_file = prepare_carra2(carra2_staging)
    carra2_final = climate_path / Path(carra2_staging_file.name)
    if carra2_staging_file.is_dir():
        # Zarr store — copytree
        if carra2_final.exists():
            shutil.rmtree(carra2_final)
        shutil.copytree(carra2_staging_file, carra2_final)
    else:
        # NetCDF or other single-file output
        shutil.copy2(carra2_staging_file, carra2_final)

    if glacier_groups:
        # For each S4F group, pre-reproject CARRA2 to that group's CRS at
        # CARRA2's native ~2.5 km resolution. Uploaded as
        # ``carra2_<group>.nc`` so ``stage.carra2()`` can fetch a single
        # small file per glacier instead of streaming the full Zarr and
        # reprojecting every time.
        for group_name in glacier_groups:
            row = complexes.loc[complexes["rgi_id"] == group_name]
            if row.empty:
                logger.warning("Aggregate complex %s not found in rgi_c.gpkg; skipping CARRA2 prep", group_name)
                continue
            group_crs = row["crs"].iloc[0]
            if not isinstance(group_crs, str) or not group_crs:
                logger.warning("Aggregate complex %s has no CRS; skipping CARRA2 prep", group_name)
                continue
            group_geom = row.geometry.iloc[0]
            group_out = climate_path / f"carra2_{group_name}.nc"
            logger.info("Preparing CARRA2 for group %s (%s) -> %s", group_name, group_crs, group_out)
            prepare_carra2_for_group(
                carra2_zarr=carra2_final,
                dst_crs=group_crs,
                geometry=group_geom,
                geometry_crs=str(complexes.crs),
                output_file=group_out,
                force_overwrite=force_overwrite,
            )

    return rgi_files


def rgi(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Prepare RGI glacier input data sets.

    This function is the programmatic entry point. It parses command-line style
    arguments, creates the target grid, downloads and processes observation data,
    and prepares climate/ocean forcing files for PISM simulations.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments **excluding** the program name (i.e., like
        ``sys.argv[1:]``). If ``None`` (default), arguments are taken from the
        current process' ``sys.argv[1:]``. Passing ``argv=[]`` is recommended
        when calling from a Jupyter notebook to avoid ipykernel arguments.

    Returns
    -------
    dict[str, Any]
        Mapping returned by :func:`prepare_rgi` (e.g. ``"rgi_complexes"``
        and ``"rgi_glaciers"`` paths).
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--force-overwrite",
        help="Force downloading all files.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ntasks",
        help="Parallel tasks.",
        type=int,
        default=8,
    )
    parser.add_argument("CONFIG_FILE", nargs=1)
    parser.add_argument("OUTPUT_PATH", nargs=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_file = args.CONFIG_FILE[0]
    force_overwrite = args.force_overwrite
    ntasks = args.ntasks
    output_path = Path(args.OUTPUT_PATH[0])
    output_path.mkdir(parents=True, exist_ok=True)

    setup_logging(output_path / "prepare.log")

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    logger.info("=" * 120)
    logger.info("\n%s", banner)
    logger.info("=" * 120)
    logger.info("Preparing S4F data")
    logger.info("-" * 120)

    config = toml.loads(Path(config_file).read_text("utf-8"))
    regions = pd.DataFrame.from_dict(config["regions"], orient="index")
    regions["region"] = regions.index.astype(str).str.zfill(2) + "_" + regions["name"]

    # Final outputs land under glacier_path; everything intermediate lives in
    # staging_path so the user can `rm -rf staging` after a clean run without
    # losing anything that downstream tools need.
    glacier_path = output_path / Path("glacier")
    glacier_path.mkdir(parents=True, exist_ok=True)
    staging_path = output_path / Path("staging")
    staging_path.mkdir(parents=True, exist_ok=True)

    # --- RGI ---
    rgi_path = glacier_path / Path("rgi")
    rgi_path.mkdir(parents=True, exist_ok=True)
    rgi_staging = staging_path / Path("rgi")
    rgi_staging.mkdir(parents=True, exist_ok=True)

    rgi_files = prepare_rgi(
        regions,
        glaciers=None,
        glacier_groups=None,
        output_path=rgi_path,
        extract_path=rgi_staging,
        force_overwrite=force_overwrite,
        ntasks=ntasks,
    )

    complexes = gpd.read_file(rgi_files["rgi_complexes"])
    glaciers = gpd.read_file(rgi_files["rgi_glaciers"])

    # --- Ice thickness ---
    ice_thickness_path = glacier_path / Path("ice_thickness")
    ice_thickness_path.mkdir(parents=True, exist_ok=True)
    ice_thickness_staging = staging_path / Path("ice_thickness")
    ice_thickness_staging.mkdir(parents=True, exist_ok=True)

    frank_path = ice_thickness_path / Path("frank")
    frank_path.mkdir(parents=True, exist_ok=True)

    prepare_ice_thickness_frank(
        regions.index,
        complexes=complexes,
        glaciers=glaciers,
        output_path=frank_path,
        extract_path=ice_thickness_staging,
        rgi_extract_path=rgi_staging,
        force_overwrite=force_overwrite,
        ntasks=ntasks,
    )

    maffezzoli_path = ice_thickness_path / Path("maffezzoli")
    maffezzoli_path.mkdir(parents=True, exist_ok=True)

    prepare_ice_thickness_maffezzoli(
        regions.index,
        complexes=complexes,
        glaciers=glaciers,
        output_path=maffezzoli_path,
        extract_path=ice_thickness_staging,
        force_overwrite=force_overwrite,
        ntasks=ntasks,
    )

    # --- GEBCO ---
    # Source NetCDF download lands in staging; only the COG goes to glacier/.
    gebco_path = glacier_path / Path("gebco")
    gebco_path.mkdir(parents=True, exist_ok=True)
    gebco_staging = staging_path / Path("gebco")
    gebco_staging.mkdir(parents=True, exist_ok=True)
    gebco_nc = download_gebco(target_dir=gebco_staging)
    cog_gebco_p = gebco_path / Path("bathymetry.tif")

    # Use xr.open_dataset (CF-aware) so the lat/lon coords become a real
    # geotransform; rxr.open_rasterio treats netCDF as a generic raster and
    # loses the georeferencing.
    ds = xr.open_dataset(gebco_nc, chunks={"lat": 1024, "lon": 1024})
    da = ds["elevation"].rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326")
    predictor = 3 if np.issubdtype(da.dtype, np.floating) else 2
    da.rio.to_raster(
        cog_gebco_p,
        driver="COG",
        compress="DEFLATE",
        predictor=predictor,
        blocksize=512,
        bigtiff="YES",
        overview_resampling="AVERAGE",
        num_threads="ALL_CPUS",
    )

    return rgi_files


def cli(argv: Sequence[str] | None = None) -> int:
    """
    Console entry point.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments (without the program name). If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    _ = rgi(argv=argv)
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(cli())
