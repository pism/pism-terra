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

"""
Prepare Greenland paleo data sets.
"""

import logging
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Sequence

import toml
import xarray as xr
from pyfiglet import Figlet

from pism_terra.domain import create_domain
from pism_terra.download import download_file
from pism_terra.greenland.paleo.forcing import (
    prepare_ocx_climatology,
    prepare_searise_series,
)
from pism_terra.ismip7.greenland.forcing import prepare_observations
from pism_terra.ismip7.greenland.prepare import DEFAULT_OBS_URL
from pism_terra.ismip7.greenland.stage import GRIDS_DIR
from pism_terra.log import setup_logging
from pism_terra.prepare_select import select_datasets
from pism_terra.workflow import check_xr_fully, check_xr_lazy

xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)

# Datasets the paleo prepare can process, in execution order.
PALEO_DATASETS = ["grid", "observations", "searise", "climatology"]
# Processed when ``--include`` is not given: grid, boot and heat-flux files are
# shared with ISMIP7 and staged from its inputs, so they are rebuilt on request only.
DEFAULT_DATASETS = ["searise", "climatology"]


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Prepare Greenland paleo input data sets.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments **excluding** the program name. If ``None``
        (default), arguments are taken from ``sys.argv[1:]``.

    Returns
    -------
    dict[str, Any]
        Results dictionary containing:

        - ``"config"`` : dict — parsed TOML configuration.
        - ``"grid_file"`` : Path — grid NetCDF.
        - ``"boot_file"``, ``"heatflux_file"`` : Path or None — observation-derived files.
        - ``"delta_T_file"``, ``"delta_SL_file"``, ``"ocean_delta_T_file"`` : Path or None —
          scalar forcing series.
        - ``"climate_file"``, ``"ocean_file"`` : Path or None — monthly climatologies.
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--force-overwrite",
        help="Rebuild files that already exist.",
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
    parser.add_argument(
        "--include",
        default=None,
        metavar="DATASET[,DATASET...]",
        help=(
            f"Comma-separated list of datasets to process; default: {', '.join(DEFAULT_DATASETS)}. "
            f"Available: {', '.join(PALEO_DATASETS)}."
        ),
    )
    parser.add_argument(
        "--forcing",
        metavar="FORCING[,FORCING...]",
        default=None,
        help="Only process these forcings ('climate', 'ocean') in the 'climatology' step; default is both.",
    )
    parser.add_argument("CONFIG_FILE", nargs=1)
    parser.add_argument("OUTPUT_PATH", nargs=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    force_overwrite = args.force_overwrite
    data_path = Path(args.data_path) if args.data_path else None
    output_path = Path(args.OUTPUT_PATH[0])
    # As in the ISMIP7 prepare: everything shipped goes into ``input/``, which
    # syncs 1:1 to s3://{bucket}/{prefix}/{version}; intermediates stay beside it.
    input_path = output_path / "input"
    input_path.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_path) if args.cache_path else output_path / "cloud_cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    staging_path = output_path / "staging"
    staging_path.mkdir(parents=True, exist_ok=True)

    setup_logging(output_path / "prepare.log")

    selected = select_datasets(args.include, PALEO_DATASETS) if args.include else list(DEFAULT_DATASETS)

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    logger.info("=" * 120)
    logger.info("\n%s", banner)
    logger.info("=" * 120)
    logger.info("Preparing Greenland paleo data")
    logger.info("-" * 120)

    config = toml.loads(Path(args.CONFIG_FILE[0]).read_text("utf-8"))
    paleo = config["paleo"]

    # --- Grid and observations (shared with ISMIP7) ---
    grids_path = input_path / GRIDS_DIR
    grid_file = grids_path / Path("pism_bedmachine_greenland_grid.nc")
    obs_files: dict[str, Any] = {}
    if {"grid", "observations"} & set(selected):
        logger.info("-" * 120)
        logger.info("Grid File")
        logger.info("-" * 120)
        match = re.match(r"^([\d.]+)(.+)$", config["domain"]["resolution"])
        if match is None:
            raise ValueError(f"Cannot parse resolution string: {config['domain']['resolution']!r}")
        grid_ds = create_domain(config["domain"]["x_bounds"], config["domain"]["y_bounds"], int(match.group(1)))
        if "grid" in selected:
            grids_path.mkdir(parents=True, exist_ok=True)
            grid_ds.to_netcdf(grid_file)
            check_xr_fully(grid_file)
        if "observations" in selected:
            logger.info("-" * 120)
            logger.info("Boot File")
            logger.info("-" * 120)
            obs_url: str | Path = DEFAULT_OBS_URL
            obs_input_path = output_path / Path("obs")
            if data_path is not None:
                obs_input_path = data_path / Path("GrIS") / Path("obs") / Path("mipkit")
                obs_url = obs_input_path / Path("GreenlandObsISMIP7-v1.3.nc")
            obs_files = prepare_observations(
                obs_url,
                obs_input_path,
                input_path,
                config,
                surface_dem="s3://pism-cloud-data/dem_reconstructions/bedmachine1980_GP_reconstruction_g600.nc",
                target_grid=grid_ds,
                force_overwrite=force_overwrite,
            )
            for v in obs_files.values():
                check_xr_lazy(v)

    # --- Scalar forcing series ---
    series_files: dict[str, Path] = {}
    if "searise" in selected:
        logger.info("-" * 120)
        logger.info("SeaRISE temperature and sea-level series")
        logger.info("-" * 120)
        searise_url = paleo["searise_url"]
        searise_file = download_file(searise_url, cache_path / Path(searise_url).name)
        series_files = prepare_searise_series(
            searise_file,
            input_path,
            ocean_scale=float(paleo.get("ocean_delta_T_scale", 0.25)),
            start=paleo.get("start"),
        )
        for name, path in series_files.items():
            logger.info("%s: %s", name, path.resolve())

    # --- Base climate ---
    climatology_files: dict[str, Path] = {}
    if "climatology" in selected:
        logger.info("-" * 120)
        logger.info("Monthly climatologies")
        logger.info("-" * 120)
        climatology_files = prepare_ocx_climatology(
            config,
            cache_path,
            input_path,
            staging_path,
            data_path=data_path,
            forcings=args.forcing,
            force_overwrite=force_overwrite,
        )
        for path in climatology_files.values():
            check_xr_lazy(path)

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
        "boot_file": obs_files.get("boot_file"),
        "heatflux_file": obs_files.get("heatflux_file"),
        "delta_T_file": series_files.get("delta_T_file"),
        "delta_SL_file": series_files.get("delta_SL_file"),
        "ocean_delta_T_file": series_files.get("ocean_delta_T_file"),
        "climate_file": climatology_files.get("climate"),
        "ocean_file": climatology_files.get("ocean"),
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
