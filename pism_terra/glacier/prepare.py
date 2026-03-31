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
Prepare RGI data sets.
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

from pism_terra.download import download_archive, download_file, extract_archive
from pism_terra.glacier.climate import convert_many_tifs_concurrent
from pism_terra.glacier.ice_thickness import prepare_ice_thickness_maffezzoli
from pism_terra.glacier.rgi import prepare_rgi
from pism_terra.vector import glaciers_in_complex
from pism_terra.workflow import check_xr_lazy

xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Prepare RGI input data sets.

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

        - ``"config"``: dict
          The parsed TOML configuration used for processing.
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

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.WARNING, format=log_format)
    for handler in logging.root.handlers:
        handler.setLevel(logging.WARNING)
    file_handler = logging.FileHandler(output_path / "prepare.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger("pism_terra").setLevel(logging.INFO)
    logging.getLogger("pism_terra").addHandler(file_handler)

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    logger.info("=" * 120)
    logger.info("\n%s", banner)
    logger.info("=" * 120)
    logger.info("Preparing RGI data")
    logger.info("-" * 120)

    config = toml.loads(Path(config_file).read_text("utf-8"))
    regions = pd.DataFrame.from_dict(config["regions"], orient="index", columns=["name"])
    regions["region"] = regions.index.astype(str).str.zfill(2) + "_" + regions["name"]

    glacier_path = output_path / Path("glacier")
    glacier_path.mkdir(parents=True, exist_ok=True)

    rgi_path = glacier_path / Path("rgi")
    rgi_path.mkdir(parents=True, exist_ok=True)

    rgi_files = prepare_rgi(regions["region"], output_path=rgi_path, force_overwrite=force_overwrite, ntasks=ntasks)

    complexes = gpd.read_file(rgi_files["rgi_complexes"])
    glaciers = gpd.read_file(rgi_files["rgi_glaciers"])

    ice_thickness_path = glacier_path / Path("ice_thickness")
    ice_thickness_path.mkdir(parents=True, exist_ok=True)

    maffezzoli_path = ice_thickness_path / Path("maffezzoli")
    maffezzoli_path.mkdir(parents=True, exist_ok=True)

    prepare_ice_thickness_maffezzoli(
        regions.index,
        complexes=complexes,
        glaciers=glaciers,
        output_path=maffezzoli_path,
        extract_path=output_path,
        force_overwrite=force_overwrite,
        ntasks=ntasks,
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
    _ = main(argv=argv)
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(cli())
