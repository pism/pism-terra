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

# pylint: disable=too-many-positional-arguments,unused-import
"""
Prepare ISMIP7 Greenland data sets.
"""

import logging
import os
import re
import time
from argparse import ArgumentParser
from importlib.resources import files
from pathlib import Path
from typing import Any, Sequence

import cf_xarray
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # pylint: disable=unused-import
import toml
import xarray as xr
import xarray_regrid.methods.conservative  # pylint: disable=unused-import
from dask.distributed import Client, as_completed
from pyfiglet import Figlet
from tqdm.auto import tqdm

from pism_terra.domain import create_domain, get_bounds
from pism_terra.download import download_file
from pism_terra.ismip7.greenland.forcing import (
    CALFIN_RESOLUTIONS,
    add_basins_to_ocean_files,
    download_calfin,
    prepare_calfin,
    prepare_dh_observations,
    prepare_ismip7_forcing,
    prepare_observations,
)
from pism_terra.ismip7.greenland.stage import GRIDS_DIR
from pism_terra.log import setup_logging
from pism_terra.prepare_select import add_include_argument, select_datasets
from pism_terra.vector import dissolve
from pism_terra.workflow import check_xr_fully, check_xr_lazy

xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)

# Datasets the ISMIP7 Greenland prepare can process, in execution order.
ISMIP7_DATASETS = ["grid", "regional_grids", "observations", "dh", "forcings", "calfin"]

# Regional grids: one per glacier of the packaged outlines, with the resolutions
# of the ice-sheet-wide domain. The setup's ``[regional_grids]`` table overrides
# any of these.
REGIONAL_GRIDS: dict[str, Any] = {
    "outline_file": "mouginot_glaciers_w_shelves.gpkg",
    "name_column": "NAME",
    "buffer": 3000.0,
    "base_resolution": 150,
    "multipliers": [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 20, 24, 30, 32],
}

# Default observation NetCDF (Globus) with the boot / velocity / heat-flux inputs.
DEFAULT_OBS_URL = "https://g-ab4495.8c185.08cc.data.globus.org/ISMIP7/Observations/Greenland/GreenlandObsISMIP7-v1.3.nc"
# The same file in the PISM bucket, which can be read by byte ranges (anonymously).
OBS_MIRROR_URL = "s3://pism-cloud-data/obs/GreenlandObsISMIP7-v1.3.nc"

# Observed thickness change: the Smith et al. (2020) ICESat-1/ICESat-2 archive
# (University of Washington ResearchWorks), a zip of GeoTIFFs covering both ice
# sheets, of which only the Greenland rasters are used. Unlike the rates the
# observation NetCDF carries, these come with a per-cell RMSE.
_DH_BITSTREAM = "cc12195c-b71e-4e26-bf85-0978dd9ce933"
DEFAULT_DH_URL = f"https://digital.lib.washington.edu/researchworks/bitstreams/{_DH_BITSTREAM}/download"
DH_ARCHIVE_NAME = "ICESat1_ICESat2_mass_change_updated_2_2021.zip"


def read_axes(source: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the x and y cell centres of a NetCDF file.

    Parameters
    ----------
    source : str or pathlib.Path
        Path or URL (``https://`` or ``s3://``, read anonymously). A remote file
        is read by byte ranges, so only its header and the two axes are transferred.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x, y)``, both ascending.
    """
    options = {"anon": True} if str(source).startswith("s3://") else {}
    fs, path = fsspec.core.url_to_fs(str(source), **options)
    with fs.open(path, "rb", block_size=2**20) as handle:
        with xr.open_dataset(handle, engine="h5netcdf") as ds:
            return np.sort(ds["x"].values.astype(float)), np.sort(ds["y"].values.astype(float))


def regional_bounds(
    geometry: Any,
    x: np.ndarray,
    y: np.ndarray,
    buffer: float = REGIONAL_GRIDS["buffer"],
    base_resolution: int = REGIONAL_GRIDS["base_resolution"],
    multipliers: Sequence[int] = tuple(REGIONAL_GRIDS["multipliers"]),
) -> tuple[list[float], list[float]]:
    """
    Bounding box around a geometry that every resolution of the setup tiles.

    The bounding box of the buffered geometry is snapped to the nearest cells
    of the reference grid; the domain shares the centre of that window and is
    a whole multiple of the least common multiple of the resolutions wide
    (see :func:`pism_terra.domain.get_bounds`).

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        Outline, in the CRS of the reference grid.
    x, y : numpy.ndarray
        Cell centres of the reference grid, ascending.
    buffer : float, default 3000
        Margin (m) added around the outline.
    base_resolution : int, default 150
        Finest resolution (m).
    multipliers : sequence of int
        The resolutions are ``base_resolution * multipliers``.

    Returns
    -------
    tuple of list of float
        ``([x_min, x_max], [y_min, y_max])``.

    Raises
    ------
    ValueError
        If the buffered geometry covers fewer than two cells of the reference
        grid along an axis, as one lying outside the grid does.
    """
    minx, miny, maxx, maxy = geometry.buffer(buffer).bounds
    window = xr.Dataset(
        coords={
            "x": x[np.abs(x - minx).argmin() : np.abs(x - maxx).argmin() + 1],
            "y": y[np.abs(y - miny).argmin() : np.abs(y - maxy).argmin() + 1],
        }
    )
    if min(window.sizes["x"], window.sizes["y"]) < 2:
        raise ValueError("The buffered geometry covers fewer than two cells of the reference grid")
    x_bnds, y_bnds = get_bounds(window, base_resolution=base_resolution, multipliers=list(multipliers))
    return [float(v) for v in x_bnds], [float(v) for v in y_bnds]


def prepare_regional_grids(
    outlines: gpd.GeoDataFrame,
    x: np.ndarray,
    y: np.ndarray,
    output_path: Path | str,
    name_column: str = REGIONAL_GRIDS["name_column"],
    buffer: float = REGIONAL_GRIDS["buffer"],
    base_resolution: int = REGIONAL_GRIDS["base_resolution"],
    multipliers: Sequence[int] = tuple(REGIONAL_GRIDS["multipliers"]),
    crs: str = "EPSG:3413",
    x_bnds: Sequence[float] | None = None,
    y_bnds: Sequence[float] | None = None,
) -> dict[str, Path]:
    """
    Write one regional grid file per outline.

    Each file, ``pism_{name}_grid.nc``, holds the extent of the domain
    (:func:`regional_bounds`) as a single cell, like the grid file of a
    regional setup; the run sets the resolution.

    Parameters
    ----------
    outlines : geopandas.GeoDataFrame
        One row per domain.
    x, y : numpy.ndarray
        Cell centres of the reference grid (the observation dataset), ascending.
    output_path : str or pathlib.Path
        Directory the grid files are written to.
    name_column : str, default "NAME"
        Column naming the domains. Characters other than letters, digits,
        ``_``, ``.`` and ``-`` become ``_`` in the file name.
    buffer : float, default 3000
        Margin (m) added around each outline.
    base_resolution : int, default 150
        Finest resolution (m).
    multipliers : sequence of int
        The resolutions are ``base_resolution * multipliers``.
    crs : str, default "EPSG:3413"
        CRS of the reference grid and of the domains.
    x_bnds, y_bnds : sequence of float or None, optional
        Extent of the ice-sheet-wide domain. A regional domain that reaches
        beyond it is logged, since the ice-sheet-wide inputs do not cover it.

    Returns
    -------
    dict[str, pathlib.Path]
        The grid file of each domain, keyed by its name.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    grid_files: dict[str, Path] = {}
    for _, row in tqdm(outlines.to_crs(crs).iterrows(), total=len(outlines), desc="Regional grids"):
        name = str(row[name_column])
        rx_bnds, ry_bnds = regional_bounds(row.geometry, x, y, buffer, base_resolution, multipliers)
        if x_bnds is not None and y_bnds is not None:
            if rx_bnds[0] < x_bnds[0] or rx_bnds[1] > x_bnds[1] or ry_bnds[0] < y_bnds[0] or ry_bnds[1] > y_bnds[1]:
                logger.warning(
                    "%s: x %s, y %s reaches beyond the ice-sheet-wide domain x %s, y %s",
                    name,
                    rx_bnds,
                    ry_bnds,
                    list(x_bnds),
                    list(y_bnds),
                )
        grid = create_domain(rx_bnds, ry_bnds, crs=crs)
        grid.attrs.update({"domain": name})
        grid_file = output_path / f"pism_{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}_grid.nc"
        grid.to_netcdf(grid_file)
        grid_files[name] = grid_file
    return grid_files


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Prepare ISMIP7 Greenland input data sets.

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
        Results dictionary containing:

        - ``"config"`` : dict — parsed TOML configuration.
        - ``"grid_file"`` : Path — generated grid NetCDF.
        - ``"regional_grid_files"`` : dict — regional grid NetCDF per glacier name.
        - ``"boot_file"`` : Path — observation-derived boot NetCDF.
        - ``"heatflux_file"`` : Path — geothermal heat-flux NetCDF.
        - ``"dh_files"`` : dict — observed cumulative thickness change per source.
        - ``"forcing_files"`` : sequence of Path — climate/ocean forcing files.
        - ``"retreat_file"`` : Path — CALFIN front-retreat NetCDF on the setup's own grid.
        - ``"retreat_files"`` : dict — CALFIN front-retreat NetCDF per resolution (m).
        - ``"calfin_fronts"`` : Path — the dated CALFIN calving fronts (lines shapefile).
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--force-overwrite",
        help="Force downloading all files.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--data-path", help="Path to ISMIP7 data folder. If not None, use local folder instead of remote.", default=None
    )
    parser.add_argument(
        "--cache-path",
        help="Directory the source.coop forcing files are cached in (reused across runs; "
        "only files that changed upstream are re-downloaded). Defaults to <OUTPUT_PATH>/cloud_cache.",
        default=None,
    )
    add_include_argument(parser, ISMIP7_DATASETS)
    # Narrow the ``forcings`` step to one corner of the tree, for reruns that
    # only need to pick up a variable that has appeared upstream.
    parser.add_argument(
        "--gcm",
        metavar="GCM[,GCM...]",
        default=None,
        help="Only process these GCMs in the 'forcings' step; default is all of them.",
    )
    parser.add_argument(
        "--pathway",
        metavar="PATHWAY[,PATHWAY...]",
        default=None,
        help="Only process these pathways (e.g. 'ctrl') in the 'forcings' step; default is all of them.",
    )
    parser.add_argument(
        "--forcing",
        metavar="FORCING[,FORCING...]",
        default=None,
        help="Only process these forcings ('climate', 'ocean') in the 'forcings' step; default is both.",
    )
    parser.add_argument(
        "--calfin-resolutions",
        metavar="RES[,RES...]",
        default=",".join(str(r) for r in CALFIN_RESOLUTIONS),
        help="Grid resolutions (m) the CalFin retreat mask is built at in the 'calfin' step. A run stages the "
        "file matching its own grid (staging warns when that file is not in the bucket), so build every "
        "resolution anyone runs at. The setup's own resolution is always included.",
    )
    parser.add_argument("CONFIG_FILE", nargs=1)
    parser.add_argument("OUTPUT_PATH", nargs=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_file = args.CONFIG_FILE[0]
    force_overwrite = args.force_overwrite
    data_path = Path(args.data_path) if args.data_path else None
    output_path = Path(args.OUTPUT_PATH[0])
    output_path.mkdir(parents=True, exist_ok=True)
    # Everything we ship goes into ``input/`` and nothing else does, so the
    # upload is a plain 1:1 sync with no excludes:
    #   aws s3 sync <OUTPUT_PATH>/input s3://{bucket}/{prefix}/{version}
    # Intermediates (cloud_cache, staging, obs, calfin, prepare.log) stay
    # beside it under OUTPUT_PATH.
    input_path = output_path / "input"
    input_path.mkdir(parents=True, exist_ok=True)
    # Original per-year forcing files synced from source.coop land here and
    # survive across runs; a rerun re-downloads only what changed upstream.
    cache_path = Path(args.cache_path) if args.cache_path else output_path / "cloud_cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    # Intermediate scratch (cdo tmps, per-epoch hist/proj) goes here so
    # ``input/`` only carries the merged files we actually ship.
    # Matches the ``staging`` convention used by ``pism-glacier-stage``.
    staging_path = output_path / "staging"
    staging_path.mkdir(parents=True, exist_ok=True)

    setup_logging(output_path / "prepare.log")

    selected = select_datasets(args.include, ISMIP7_DATASETS)

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    logger.info("=" * 120)
    logger.info("\n%s", banner)
    logger.info("=" * 120)
    logger.info("Preparing ISMIP7 Greenland data")
    logger.info("-" * 120)

    config = toml.loads(Path(config_file).read_text("utf-8"))

    x_bnds = config["domain"]["x_bounds"]
    y_bnds = config["domain"]["y_bounds"]
    resolution_str = config["domain"]["resolution"]
    match = re.match(r"^([\d.]+)(.+)$", resolution_str)
    if match is None:
        raise ValueError(f"Cannot parse resolution string: {resolution_str!r}")
    resolution, _ = int(match.group(1)), match.group(2)

    # --- Grid (a dependency of the observations target grid) ---
    # Grid files go to ``input/grids``, where staging looks for the campaign's
    # ``grid_file``.
    grids_path = input_path / GRIDS_DIR
    grid_file = grids_path / Path("pism_bedmachine_greenland_grid.nc")
    grid_ds = None
    if {"grid", "observations"} & set(selected):
        logger.info("-" * 120)
        logger.info("Grid File")
        logger.info("-" * 120)
        grid_ds = create_domain(x_bnds, y_bnds, resolution)
        if "grid" in selected:
            grids_path.mkdir(parents=True, exist_ok=True)
            grid_ds.to_netcdf(grid_file)
            check_xr_fully(grid_file)

    # Observation NetCDF with the boot / velocity / heat-flux inputs, used by the
    # observations step, and whose grid the regional grids are cut from.
    obs_url: str | Path = DEFAULT_OBS_URL
    if data_path is not None:
        obs_url = (
            data_path / Path(config["ice_sheet"]) / Path("obs") / Path("mipkit") / Path("GreenlandObsISMIP7-v1.3.nc")
        )
    # When data_path is None, fall back to an obs-cache subdir under output_path
    # so prepare_observations always has a real directory to download into.
    obs_input_path = (
        data_path / Path("GrIS") / Path("obs") / Path("mipkit") if data_path is not None else output_path / Path("obs")
    )

    # --- Regional grids (one per glacier) ---
    regional_grid_files: dict[str, Path] = {}
    if "regional_grids" in selected:
        logger.info("-" * 120)
        logger.info("Regional Grid Files")
        logger.info("-" * 120)
        regional = {**REGIONAL_GRIDS, **config.get("regional_grids", {})}
        outline_file = Path(regional["outline_file"])
        if not outline_file.is_file():
            outline_file = Path(str(files("pism_terra.data") / regional["outline_file"]))
        # Only the axes are needed: use a local copy if there is one, and read
        # them off the bucket's copy otherwise rather than downloading the file.
        obs_local = obs_input_path / Path(str(obs_url)).name
        obs_x, obs_y = read_axes(obs_local if obs_local.is_file() else OBS_MIRROR_URL)
        regional_grid_files = prepare_regional_grids(
            gpd.read_file(outline_file),
            obs_x,
            obs_y,
            grids_path,
            name_column=regional["name_column"],
            buffer=float(regional["buffer"]),
            base_resolution=int(regional["base_resolution"]),
            multipliers=regional["multipliers"],
            crs=config["domain"].get("crs", "EPSG:3413"),
            x_bnds=x_bnds,
            y_bnds=y_bnds,
        )
        for regional_grid_file in regional_grid_files.values():
            check_xr_fully(regional_grid_file)
        logger.info(
            "%d regional grids from %s: %s",
            len(regional_grid_files),
            outline_file.name,
            grids_path,
        )

    # --- Observations (boot, heatflux, velocity) ---
    obs_files_1985: dict[str, Any] = {}
    obs_files_2007: dict[str, Any] = {}
    if "observations" in selected:
        logger.info("-" * 120)
        logger.info("Boot File")
        logger.info("-" * 120)
        surface_dem = "s3://pism-cloud-data/dem_reconstructions/bedmachine1980_GP_reconstruction_g600.nc"
        obs_files_1985 = prepare_observations(
            obs_url,
            obs_input_path,
            input_path,
            config,
            surface_dem=surface_dem,
            target_grid=grid_ds,
            force_overwrite=force_overwrite,
        )
        for v in obs_files_1985.values():
            check_xr_lazy(v)

        obs_files_2007 = prepare_observations(
            obs_url,
            obs_input_path,
            input_path,
            config,
            target_grid=grid_ds,
            force_overwrite=force_overwrite,
        )
        for v in obs_files_2007.values():
            check_xr_lazy(v)

    # --- Observed thickness change ---
    # The archive states dH/dt; a run reports thickness change, so the rate is
    # integrated here once rather than in every analysis. Left on the
    # archive's own 5 km grid, which is already the ISMIP7 projection.
    dh_files: dict = {}
    if "dh" in selected:
        logger.info("-" * 120)
        logger.info("Observed Thickness Change")
        logger.info("-" * 120)
        dh_archive = (
            data_path / Path(config["ice_sheet"]) / Path("obs") / Path("smith") / Path(DH_ARCHIVE_NAME)
            if data_path is not None
            else Path(download_file(DEFAULT_DH_URL, output_path / Path("obs") / Path(DH_ARCHIVE_NAME)))
        )
        dh_files = prepare_dh_observations(
            dh_archive,
            input_path,
            force_overwrite=force_overwrite,
        )
        for v in dh_files.values():
            check_xr_lazy(v)

    # --- Forcings ---
    forcing_files: list = []
    if "forcings" in selected:
        logger.info("-" * 120)
        logger.info("Forcings")
        logger.info("-" * 120)
        forcing_files = list(
            prepare_ismip7_forcing(
                cache_path,
                input_path,
                config,
                data_path=data_path,
                staging_path=staging_path,
                gcms=args.gcm,
                pathways=args.pathway,
                forcings=args.forcing,
            )
        )
        logger.info("Forcing files: %s", forcing_files)

    # --- CalFin glacier fronts ---
    retreat_files: dict[int, Path] = {}
    calfin_fronts: Path | None = None
    if "calfin" in selected:
        logger.info("-" * 120)
        logger.info("Calfin Glacier Fronts Files")
        logger.info("-" * 120)
        calfin_resolutions = sorted({int(r) for r in args.calfin_resolutions.split(",") if r.strip()} | {resolution})
        retreat_files = {
            int(res): Path(fn)
            for res, fn in prepare_calfin(
                input_path, calfin_resolutions, x_bnds=x_bnds, y_bnds=y_bnds, force_overwrite=force_overwrite
            ).items()
        }
        # The dated calving fronts, beside the polygons the masks come from,
        # for drawing them (render_terrain_3d --outlines animates them).
        calfin_fronts = download_calfin(output_path / Path("calfin"), "lines")
        logger.info("CALFIN polygons and fronts: %s", calfin_fronts.parent.resolve())

    # Every shipped product was written into ``input/`` directly, so the
    # upload is a plain 1:1 sync — no excludes, no duplicated copy on disk.
    logger.info("-" * 120)
    logger.info("Now run")
    logger.info(
        "aws s3 sync %s s3://%s/%s/%s",
        input_path.resolve(),
        config["bucket"],
        config["prefix"],
        config["version"],
    )
    logger.info("-" * 120)

    return {
        "config": config,
        "grid_file": grid_file,
        "regional_grid_files": regional_grid_files,
        "boot_file_1985": obs_files_1985.get("boot_file"),
        "boot_file_2007": obs_files_2007.get("boot_file"),
        "heatflux_file": obs_files_1985.get("heatflux_file") or obs_files_2007.get("heatflux_file"),
        "dh_files": dh_files,
        "forcing_files": forcing_files,
        "retreat_file": retreat_files.get(resolution),
        "retreat_files": retreat_files,
        "calfin_fronts": calfin_fronts,
        "obs_file_1985": obs_files_1985.get("obs_file"),
        "obs_file_2007": obs_files_2007.get("obs_file"),
    }


def add_basins(argv: Sequence[str] | None = None) -> int:
    """
    Backfill the GrIS basin mask onto existing ISMIP7 ocean forcing files.

    Console entry point (``pism-ismip7-greenland-add-basins``) that stamps the
    ``basins`` variable (from the packaged basin polygons) onto already-generated
    ocean forcing files without regenerating them. Each positional argument may be
    an ocean NetCDF or a directory (scanned for ``ismip7_greenland_ocean_*.nc``);
    only files whose name contains ``_ocean_`` are processed.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments (without the program name). If ``None`` (default),
        uses ``sys.argv``.

    Returns
    -------
    int
        Exit code: ``0`` on success, ``1`` if no ocean files were found.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = ArgumentParser(description="Add the GrIS basin mask to existing ISMIP7 ocean forcing files.")
    parser.add_argument(
        "OCEAN_FILES",
        nargs="+",
        help="Ocean forcing NetCDF file(s), or directories to scan for ismip7_greenland_ocean_*.nc.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    candidates: list[Path] = []
    for entry in args.OCEAN_FILES:
        p = Path(entry)
        if p.is_dir():
            candidates.extend(sorted(p.glob("ismip7_greenland_ocean_*.nc")))
        else:
            candidates.append(p)
    ocean_files = [p for p in candidates if "_ocean_" in p.name]

    if not ocean_files:
        logger.warning("No ocean forcing files found (need '_ocean_' in the filename).")
        return 1

    logger.info("Adding basin mask to %d ocean forcing file(s)", len(ocean_files))
    add_basins_to_ocean_files(ocean_files)
    return 0


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
    _ = main(argv=argv)
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(cli())
