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

from pism_terra.aws import local_to_s3, s3_to_local
from pism_terra.config import load_config
from pism_terra.workflow import check_dataset_fully, check_xr_fully, check_xr_lazy

xr.set_options(keep_attrs=True)


def stage(
    config: dict,
    path: str | Path = "input_files",
    bucket: str = "pism-cloud-data",
    prefix: str = "ismip7_greenland_input",
    force_overwrite: bool = False,
) -> pd.DataFrame:
    """
    Stage ISMIP7 Greenland inputs and return a file index.

    Syncs pre-built input data from S3, validates each file, and returns
    a single-row DataFrame with absolute paths to all staged artifacts
    (boot, grid, heatflux, regrid, retreat, climate, and ocean files).

    Parameters
    ----------
    config : dict
        Configuration mapping. Must contain at least:
        - ``"grid_file"`` : str
            Path to the grid NetCDF file relative to the input directory.
        - ``"boot_file"`` : str
            Path to the boot NetCDF file relative to the input directory.
        - ``"heatflux_file"`` : str
            Path to the heatflux NetCDF file relative to the input directory.
        - ``"regrid_file"`` : str
            Path to the regrid NetCDF file relative to the input directory.
        - ``"retreat_file"`` : str
            Path to the retreat NetCDF file relative to the input directory.
        - ``"pathway"`` : str
            ISMIP7 pathway identifier.
        - ``"gcm"`` : str
            GCM model name.
        - ``"version"`` : str
            Dataset version.
        - ``"start_year"`` : int
            Start year of the forcing period.
        - ``"end_year"`` : int
            End year of the forcing period.
    path : str or pathlib.Path, default ``"input_files"``
        Output directory. Created if missing. All staged artifacts are written here.
    bucket : str, default ``"pism-cloud-data"``
        AWS S3 bucket name to sync ISMIP7 input data from.
    prefix : str, default ``"ismip7_greenland_input"``
        S3 key prefix (folder path within the bucket).
    force_overwrite : bool, default ``False``
        If ``True``, downstream helpers may regenerate intermediate/final artifacts
        even if cache files exist.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame with absolute-path columns including
        ``boot_file``, ``grid_file``, ``heatflux_file``, ``regrid_file``,
        ``retreat_file``, ``climate_file``, ``ocean_file``,
        ``surface_input_file``, and ``frontal_melt_file``.
    """

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    print("=" * 80)
    print(banner)
    print("=" * 80)
    print("Stage ISMIP7 Greenland")
    print("-" * 80)
    print("")

    # Outputs dir
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    input_path = path / Path("input")
    if force_overwrite:
        input_path.unlink(missing_ok=True)
    input_path.mkdir(parents=True, exist_ok=True)

    s3_to_local(bucket, prefix=prefix, dest=input_path)

    grid_file = input_path / Path(config["grid_file"])
    check_xr_fully(grid_file)

    boot_file = input_path / Path(config["boot_file"])
    check_xr_lazy(boot_file)

    heatflux_file = input_path / Path(config["heatflux_file"])
    check_xr_lazy(heatflux_file)

    regrid_file = input_path / Path(config["regrid_file"])
    check_xr_lazy(regrid_file)

    retreat_file = input_path / Path(config["retreat_file"])
    check_xr_lazy(retreat_file)

    pathway = config["pathway"]
    gcm = config["gcm"]
    version = config["version"]
    start_year = config["start_year"]
    end_year = config["end_year"]

    # Build file index (one row per climate file)
    files_dict = {
        "boot_file": boot_file.resolve(),
        "grid_file": grid_file.resolve(),
        "heatflux_file": heatflux_file.resolve(),
        "regrid_file": regrid_file.resolve(),
        "retreat_file": retreat_file.resolve(),
    }
    for forcing in ["climate", "ocean"]:
        forcing_file = input_path / Path(
            f"ismip7_greenland_{forcing}_{pathway}_{gcm}_v{version}_{start_year}_{end_year}.nc"
        )
        check_xr_lazy(forcing_file)
        files_dict[f"{forcing}_file"] = forcing_file.resolve()

    forcing = "climate"
    surface_input_file = input_path / Path(
        f"ismip7_greenland_{forcing}_{pathway}_{gcm}_v{version}_{start_year}_{end_year}.nc"
    )
    check_xr_lazy(surface_input_file)
    files_dict["surface_input_file"] = surface_input_file.resolve()

    forcing = "ocean"
    frontal_melt_file = input_path / Path(
        f"ismip7_greenland_{forcing}_{pathway}_{gcm}_v{version}_{start_year}_{end_year}.nc"
    )
    check_xr_lazy(frontal_melt_file)
    files_dict["frontal_melt_file"] = frontal_melt_file.resolve()

    dfs: list[pd.DataFrame] = []
    dfs.append(pd.DataFrame.from_dict([files_dict]))

    df = pd.concat(dfs).reset_index(drop=True)
    return df


def main():
    """
    Run main script.
    """

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Stage ISMIP7 Greenland."
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
        default=Path("data/ismip7_greenland"),
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

    cfg = load_config(config_file)
    config = cfg.campaign.as_params()

    path.mkdir(parents=True, exist_ok=True)

    is_df = stage(config, path=path, force_overwrite=force_overwrite)
    is_df.to_csv(path / Path("input") / Path("ismip7_greenland_files.csv"))

    if options.bucket:
        prefix = f"{options.bucket_prefix}/ismip7_greenland" if options.bucket_prefix else "ismip7_greenland"
        local_to_s3(path, bucket=options.bucket, prefix=prefix)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
