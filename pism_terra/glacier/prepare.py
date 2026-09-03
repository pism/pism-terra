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
Prepare glacier input data sets.

One command covers every project. The setup TOML's ``[staging]
project_directory`` names the subdirectory under ``input/`` that receives the
products depending on that project's ``[regions]`` CRS overrides; the global
data sets are written once, beside it, and shared.
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
    prepare_carra2_monthly_mean,
    prepare_snap,
)
from pism_terra.glacier.ice_thickness import (
    prepare_ice_thickness_frank,
    prepare_ice_thickness_maffezzoli,
)
from pism_terra.glacier.observations import prepare_dh_hugonnet
from pism_terra.glacier.rgi import prepare_rgi
from pism_terra.heatflux import prepare_heatflux_lucazeau
from pism_terra.log import setup_logging
from pism_terra.prepare_select import add_include_argument, select_datasets
from pism_terra.vector import glaciers_in_complex
from pism_terra.workflow import check_xr_lazy

xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)

# Datasets ``prepare`` can process, in execution order. Used for the
# ``--include`` selector and its help text.
PREPARE_DATASETS = [
    "rgi",
    "ice_thickness_frank",
    "ice_thickness_maffezzoli",
    "dh_hugonnet",
    "gebco",
    "heatflux_lucazeau",
    "snap",
    "carra2",
]


def prepare_paths(output_path: Path | str, project_directory: str) -> dict[str, Path]:
    """
    Build the output directory layout for one project.

    Final artifacts land under ``<output>/input``, which is the tree that gets
    synced to S3; everything intermediate lives in ``<output>/staging`` so it
    can be deleted after a clean run without losing anything downstream tools
    need.

    ``input`` is split in two. Products whose content depends on the project's
    ``[regions]`` CRS overrides — the RGI outlines, the ice-thickness rasters
    and the per-group climate files — go under ``input/<project_directory>``,
    so an S4F run and an RGI run cannot overwrite each other. The global
    products (GEBCO, heat flux, SNAP, the merged CARRA2 store) sit directly
    under ``input`` and are shared by every project. Staging is shared too: the
    raw downloads are identical whatever the project.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Root directory passed on the command line.
    project_directory : str
        Project subdirectory name, e.g. ``"rgi"`` or ``"s4f"``, taken from the
        setup TOML's ``[staging] project_directory``.

    Returns
    -------
    dict[str, pathlib.Path]
        Directories keyed by role. No directory is created; call
        :func:`ensure_dir` on the ones you actually write to.
    """
    output_path = Path(output_path)
    input_path = output_path / "input"
    project_path = input_path / project_directory
    staging_path = output_path / "staging"

    return {
        "output": output_path,
        "input": input_path,
        "project": project_path,
        # Project-specific (CRS-dependent).
        "rgi": project_path / "rgi",
        "ice_thickness": project_path / "ice_thickness",
        "ice_thickness_frank": project_path / "ice_thickness" / "frank",
        "ice_thickness_maffezzoli": project_path / "ice_thickness" / "maffezzoli",
        "dh_hugonnet": project_path / "dh" / "hugonnet",
        "project_climate": project_path / "climate",
        # Shared across projects.
        "gebco": input_path / "gebco",
        "heatflux": input_path / "heatflux",
        "climate": input_path / "climate",
        # Intermediates, shared, never uploaded.
        "staging": staging_path,
        "staging_rgi": staging_path / "rgi",
        "staging_ice_thickness": staging_path / "ice_thickness",
        "staging_dh": staging_path / "dh",
        "staging_gebco": staging_path / "gebco",
        "staging_heatflux": staging_path / "heatflux",
        "staging_snap": staging_path / "snap",
        "staging_carra2": staging_path / "carra2",
    }


def ensure_dir(path: Path) -> Path:
    """
    Create a directory (and its parents) and return it.

    Parameters
    ----------
    path : pathlib.Path
        Directory to create.

    Returns
    -------
    pathlib.Path
        The same path, now guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_glacier_groups(glacier_files: Sequence[str | Path]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """
    Read the per-group glacier ID lists that scope a project to a study area.

    Each CSV names one aggregate: ``S4F_target_AK_RGI_id.csv`` becomes the group
    ``S4F_AK``. Files that do not match that convention fall back to their stem.

    Parameters
    ----------
    glacier_files : sequence of str or pathlib.Path
        CSV files with an ``rgi_id`` column. May be empty, in which case the
        project covers every glacier in the configured regions.

    Returns
    -------
    glacier_groups : dict[str, pandas.DataFrame]
        Mapping of aggregate name to its rows. Empty when no files were given.
    glaciers : pandas.DataFrame or None
        All rows concatenated, with an ``o1regions`` column derived from the
        RGI IDs. ``None`` when no files were given.
    """
    glacier_groups: dict[str, pd.DataFrame] = {}
    if not glacier_files:
        return glacier_groups, None

    for glacier_file in glacier_files:
        p = Path(glacier_file)
        # "S4F_target_AK_RGI_id.csv" -> "S4F_AK"; fall back to the file stem.
        m = re.match(r"^(?P<prefix>.+?)_target_(?P<region>.+?)_RGI_id$", p.stem)
        name = f"{m['prefix']}_{m['region']}" if m else p.stem
        glacier_groups[name] = pd.read_csv(p)

    glaciers = pd.concat(glacier_groups.values(), ignore_index=True)
    glaciers["o1regions"] = glaciers["rgi_id"].str.extract(r"-G-(\d{2})-")
    return glacier_groups, glaciers


def prepare(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Prepare glacier input data sets for one project.

    This function is the programmatic entry point. It parses command-line style
    arguments, downloads and processes observation data, and prepares
    climate/ocean forcing files for PISM simulations.

    The setup TOML's ``[staging] project_directory`` decides where the
    CRS-dependent products land (see :func:`prepare_paths`). Passing one or more
    glacier-ID CSVs additionally scopes the run to those glaciers, adds an
    aggregate "complex" per CSV, and pre-reprojects CARRA2 for each aggregate —
    which is what an S4F run needs and a whole-region RGI run does not.

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

    Raises
    ------
    ValueError
        If the setup TOML does not declare ``[staging] project_directory``.
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
    add_include_argument(parser, PREPARE_DATASETS)
    parser.add_argument("CONFIG_FILE", nargs=1)
    parser.add_argument("OUTPUT_PATH", nargs=1)
    parser.add_argument("GLACIER_FILES", nargs="*")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_file = args.CONFIG_FILE[0]
    glacier_files = args.GLACIER_FILES
    force_overwrite = args.force_overwrite
    ntasks = args.ntasks
    output_path = ensure_dir(Path(args.OUTPUT_PATH[0]))

    setup_logging(output_path / "prepare.log")

    selected = select_datasets(args.include, PREPARE_DATASETS)

    config = toml.loads(Path(config_file).read_text("utf-8"))
    project_directory = config.get("staging", {}).get("project_directory")
    if not isinstance(project_directory, str) or not project_directory:
        raise ValueError(
            f'{config_file} must declare a project directory, e.g.\n\n[staging]\nproject_directory = "s4f"\n\n'
            "It names the subdirectory under `input/` holding the products that depend on this "
            "project's [regions] CRS overrides."
        )

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    logger.info("=" * 120)
    logger.info("\n%s", banner)
    logger.info("=" * 120)
    logger.info("Preparing %s data", project_directory)
    logger.info("-" * 120)

    regions = pd.DataFrame.from_dict(config["regions"], orient="index")
    regions["region"] = regions.index.astype(str).str.zfill(2) + "_" + regions["name"]

    glacier_groups, glaciers_csv = read_glacier_groups(glacier_files)
    if glaciers_csv is not None:
        # Only the O1 regions the listed glaciers actually fall in are worth
        # downloading.
        o1regions = glaciers_csv["o1regions"].unique().astype(int).astype(str)
        regions = regions[regions.index.isin(o1regions)]

    paths = prepare_paths(output_path, project_directory)
    ensure_dir(paths["input"])
    ensure_dir(paths["staging"])

    # --- RGI ---
    # The RGI outlines are a dependency of ice thickness (and the CARRA2 group
    # reprojection); paths are always resolved, ``prepare_rgi`` only runs when
    # selected, otherwise the cached gpkg files from a previous run are reused.
    rgi_path = ensure_dir(paths["rgi"])
    rgi_staging = ensure_dir(paths["staging_rgi"])

    rgi_files: dict[str, Any] = {
        "rgi_complexes": rgi_path / f"{project_directory}_c.gpkg",
        "rgi_glaciers": rgi_path / f"{project_directory}_g.gpkg",
    }
    if "rgi" in selected:
        rgi_files = prepare_rgi(
            regions,
            glaciers=glaciers_csv,
            glacier_groups=glacier_groups or None,
            output_path=rgi_path,
            extract_path=rgi_staging,
            force_overwrite=force_overwrite,
            ntasks=ntasks,
            name_prefix=project_directory,
        )

    # Load the RGI outlines once if any consumer needs them.
    need_outlines = bool({"ice_thickness_frank", "ice_thickness_maffezzoli", "dh_hugonnet"} & set(selected)) or (
        "carra2" in selected and bool(glacier_groups)
    )
    complexes = gpd.read_file(rgi_files["rgi_complexes"]) if need_outlines else None
    glaciers = gpd.read_file(rgi_files["rgi_glaciers"]) if need_outlines else None

    # --- Ice thickness ---
    if {"ice_thickness_frank", "ice_thickness_maffezzoli"} & set(selected):
        ensure_dir(paths["ice_thickness"])
        ice_thickness_staging = ensure_dir(paths["staging_ice_thickness"])

        if "ice_thickness_frank" in selected:
            prepare_ice_thickness_frank(
                regions.index,
                complexes=complexes,
                glaciers=glaciers,
                output_path=ensure_dir(paths["ice_thickness_frank"]),
                extract_path=ice_thickness_staging,
                rgi_extract_path=rgi_staging,
                force_overwrite=force_overwrite,
                ntasks=ntasks,
            )

        if "ice_thickness_maffezzoli" in selected:
            prepare_ice_thickness_maffezzoli(
                regions.index,
                complexes=complexes,
                glaciers=glaciers,
                output_path=ensure_dir(paths["ice_thickness_maffezzoli"]),
                extract_path=ice_thickness_staging,
                force_overwrite=force_overwrite,
                ntasks=ntasks,
            )

    # --- Observed elevation change (Hugonnet 2021) ---
    if "dh_hugonnet" in selected:
        assert complexes is not None  # loaded above (need_outlines)
        prepare_dh_hugonnet(
            complexes,
            output_path=ensure_dir(paths["dh_hugonnet"]),
            extract_path=ensure_dir(paths["staging_dh"]),
            ntasks=ntasks,
            force_overwrite=force_overwrite,
        )

    # --- GEBCO ---
    if "gebco" in selected:
        # Source NetCDF download lands in staging; only the COG goes to input/.
        gebco_path = ensure_dir(paths["gebco"])
        gebco_nc = download_gebco(target_dir=ensure_dir(paths["staging_gebco"]))
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

    # --- Heat flow (Lucazeau 2019) ---
    if "heatflux_lucazeau" in selected:
        prepare_heatflux_lucazeau(
            output_path=ensure_dir(paths["heatflux"]),
            extract_path=ensure_dir(paths["staging_heatflux"]),
            force_overwrite=force_overwrite,
        )

    # --- Climate ---
    # SNAP and CARRA2 are both global products, so they land side by side in
    # the shared ``input/climate``; only the per-group CARRA2 files below
    # depend on a project's CRS.
    if "snap" in selected:
        # SNAP/CRU-TS40 monthly climatologies (built under staging, copied to
        # input/climate for upload; one file per 30-year window).
        climate_path = ensure_dir(paths["climate"])
        snap_staging = ensure_dir(paths["staging_snap"])
        for snap_file in prepare_snap(snap_staging, force_overwrite=force_overwrite):
            shutil.copy2(snap_file, climate_path / Path(snap_file).name)

    if "carra2" in selected:
        # Run the download/merge under staging, then move only the merged
        # product into input/climate. Year-by-year CDS intermediates stay
        # in staging.
        climate_path = ensure_dir(paths["climate"])
        carra2_staging = ensure_dir(paths["staging_carra2"])

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

        # Twelve-step monthly climatology over a fixed reference period. It is
        # in CARRA2's own CRS, so it is shared by every project like the store
        # it comes from; ``stage.carra2_monthly_mean()`` reads it.
        carra2_climatology = prepare_carra2_monthly_mean(
            carra2_zarr=carra2_final,
            output_zarr=climate_path / "carra2_monthly_mean.zarr",
            force_overwrite=force_overwrite,
        )

        if glacier_groups:
            # For each group, pre-reproject CARRA2 to that group's CRS at
            # CARRA2's native ~2.5 km resolution. Uploaded as
            # ``carra2_<group>.nc`` so ``stage.carra2()`` can fetch a single
            # small file per glacier instead of streaming the full Zarr and
            # reprojecting every time. The result depends on the group's
            # CRS, so it lives under the project directory. The climatology
            # gets the same treatment for ``stage.carra2_monthly_mean()``.
            assert complexes is not None  # loaded above (need_outlines)
            project_climate_path = ensure_dir(paths["project_climate"])
            for group_name in glacier_groups:
                row = complexes.loc[complexes["rgi_id"] == group_name]
                if row.empty:
                    logger.warning(
                        "Aggregate complex %s not found in %s; skipping CARRA2 prep",
                        group_name,
                        rgi_files["rgi_complexes"],
                    )
                    continue
                group_crs = row["crs"].iloc[0]
                if not isinstance(group_crs, str) or not group_crs:
                    logger.warning("Aggregate complex %s has no CRS; skipping CARRA2 prep", group_name)
                    continue
                group_geom = row.geometry.iloc[0]
                for source, stem in ((carra2_final, "carra2"), (carra2_climatology, "carra2_monthly_mean")):
                    group_out = project_climate_path / f"{stem}_{group_name}.nc"
                    logger.info("Preparing %s for group %s (%s) -> %s", stem, group_name, group_crs, group_out)
                    prepare_carra2_for_group(
                        carra2_zarr=source,
                        dst_crs=group_crs,
                        geometry=group_geom,
                        geometry_crs=str(complexes.crs),
                        output_file=group_out,
                        force_overwrite=force_overwrite,
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
    _ = prepare(argv=argv)
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(cli())
