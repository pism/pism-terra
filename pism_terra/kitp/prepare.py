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
import shutil
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Sequence

import cf_xarray
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # pylint: disable=unused-import
import toml
import xarray as xr
import xarray_regrid.methods.conservative  # pylint: disable=unused-import
from cdo import Cdo
from dask.distributed import Client, as_completed
from pyfiglet import Figlet
from tqdm.auto import tqdm

from pism_terra.domain import create_domain
from pism_terra.ismip7.greenland.forcing import prepare_observations
from pism_terra.kitp.forcing import (
    baseline_with_anomalies,
    prepare_anomalies,
    prepare_carra2_climatology,
    prepare_hirham5_climatology,
)
from pism_terra.raster import create_ds
from pism_terra.vector import dissolve
from pism_terra.workflow import check_xr_fully, check_xr_lazy

xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Prepare KITP Greenland input data sets.

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
    parser.add_argument("--obs-path", default="data/obs")
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
    obs_path = Path(args.obs_path)
    output_path = Path(args.OUTPUT_PATH[0])
    output_path.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.WARNING, format=log_format)
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
    logger.info("Preparing ISMIP7 Greenland data")
    logger.info("-" * 120)

    config = toml.loads(Path(config_file).read_text("utf-8"))
    version = config["version"]

    logger.info("-" * 120)
    logger.info("Grid File")
    logger.info("-" * 120)

    data_path = output_path / Path(f"stage_{config["version"]}")
    data_path.mkdir(exist_ok=True)

    x_bnds = config["domain"]["x_bounds"]
    y_bnds = config["domain"]["y_bounds"]
    resolution_str = config["domain"]["resolution"]
    match = re.match(r"^([\d.]+)(.+)$", resolution_str)
    if match is None:
        raise ValueError(f"Cannot parse resolution string: {resolution_str!r}")
    resolution, _ = int(match.group(1)), match.group(2)

    grid_ds = create_domain(x_bnds, y_bnds, resolution)
    grid_file = data_path / Path("ismip7_greenland_grid.nc")
    encoding = {var: {"_FillValue": None} for var in list(grid_ds.data_vars) + list(grid_ds.coords)}
    grid_ds.to_netcdf(grid_file, encoding=encoding)
    check_xr_fully(grid_file)

    url = "https://g-ab4495.8c185.08cc.data.globus.org/ISMIP6/ISMIP7_Prep/Observations/Greenland/GreenlandObsISMIP7-v1.3.nc"
    logger.info("-" * 120)
    logger.info("Boot File")
    logger.info("-" * 120)
    obs_files = prepare_observations(
        url,
        obs_path,
        data_path,
        config,
        target_grid=grid_ds,
        force_overwrite=force_overwrite,
    )
    for v in obs_files.values():
        check_xr_lazy(v)

    logger.info("-" * 120)
    logger.info("Baseline Climatology")
    logger.info("-" * 120)
    baseline = config["baseline"]
    if baseline == "hirham5":
        start_year = config["climatology"][baseline]["start_year"]
        end_year = config["climatology"][baseline]["end_year"]
        baseline_file = prepare_hirham5_climatology(
            data_path,
            start_year=start_year,
            end_year=end_year,
            version=version,
            n_workers=ntasks,
            force_overwrite=force_overwrite,
        )
    elif baseline == "carra2":
        year = config["climatology"][baseline]["year"]
        baseline_file = prepare_carra2_climatology(
            data_path,
            year=year,
            version=version,
            n_workers=ntasks,
            force_overwrite=force_overwrite,
        )
    else:
        raise ValueError(f"Unknown baseline {baseline!r}. Supported: 'hirham5', 'carra2'.")

    logger.info("-" * 120)
    logger.info("Anomaly Forcing")
    logger.info("-" * 120)

    bucket = config["forcing"]["bucket"]
    prefix = config["forcing"]["prefix"]
    gcms = config["gcms"]
    present_day_forcings = config["forcing"]["present_day_forcings"]
    future_forcings = config["forcing"]["future_forcings"]

    forcing_files = prepare_anomalies(
        data_path,
        bucket=bucket,
        prefix=prefix,
        gcms=gcms,
        present_day_forcings=present_day_forcings,
        future_forcings=future_forcings,
        version=version,
        n_workers=ntasks,
        force_overwrite=force_overwrite,
    )

    combined_files = baseline_with_anomalies(baseline_file, forcing_files)

    input_files = [grid_file] + list(obs_files.values()) + [baseline_file] + forcing_files + combined_files

    s3_output_path = output_path / Path(config["prefix"]) / Path(config["version"])
    s3_output_path.mkdir(parents=True, exist_ok=True)
    logger.info("-" * 120)
    logger.info("Copying input files to %s", s3_output_path)
    logger.info("-" * 120)
    for f in input_files:
        dest = s3_output_path / Path(f).name
        shutil.copy2(f, dest)
        logger.info("  %s", dest)

    return {
        "config": config,
        "grid_file": grid_file,
        "boot_file": obs_files["boot_file"],
        "heatflux_file": obs_files["heatflux_file"],
        "baseline_file": baseline_file,
    }


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
