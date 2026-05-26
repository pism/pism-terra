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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import cf_xarray
import geopandas as gpd
import pandas as pd
import rioxarray
import xarray as xr
from pyfiglet import Figlet
from shapely.geometry import Polygon
from tqdm.auto import tqdm

from pism_terra.aws import local_to_s3, s3_to_local
from pism_terra.config import load_config
from pism_terra.workflow import check_dataset_fully, check_xr_fully, check_xr_lazy

xr.set_options(keep_attrs=True)


def stage(
    config: dict,
    bucket: str,
    prefix: str,
    output_path: str | Path,
    force_overwrite: bool = False,
) -> pd.DataFrame:
    """
    Stage KITP Greenland inputs and return a file index.

    Syncs pre-built input data from S3, validates each file, and returns
    a DataFrame with absolute paths to all staged artifacts (boot, grid,
    heatflux, regrid, ocean, outline, and climate files), one row per
    GCM/forcing combination plus a baseline-climatology row.

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
        - ``"ocean_file"`` : str
            Path to the ocean-forcing NetCDF file relative to the input directory.
        - ``"outline_file"`` : str
            Path to the basin outline file relative to the input directory.
        - ``"gcms"`` : dict[str, list[list[str]]]
            Mapping of GCM names to their forcing pairs ``[[future, present], ...]``.
        - ``"climatology"`` : str
            Baseline climatology stem used to build climate filenames.
        - ``"version"`` : str
            Dataset version.
    bucket : str
        AWS S3 bucket name to sync KITP input data from.
    prefix : str
        S3 key prefix (folder path within the bucket).
    output_path : str or pathlib.Path`
        Output directory. Created if missing. All staged artifacts are written here.
    force_overwrite : bool, default ``False``
        If ``True``, downstream helpers may regenerate intermediate/final artifacts
        even if cache files exist.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per GCM/forcing pair (plus a baseline-only
        row), with absolute-path columns ``boot_file``, ``grid_file``,
        ``heatflux_file``, ``ocean_file``, ``outline_file``, ``regrid_file``,
        and ``climate_file``, plus a ``sample`` identifier column.
    """

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    print("=" * 120)
    print(banner)
    print("=" * 120)
    print("Stage KITP Greenland")
    print("-" * 120)
    print("")

    # Outputs dir
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    input_path = output_path / Path(prefix)

    if force_overwrite:
        input_path.unlink(missing_ok=True)
    input_path.mkdir(parents=True, exist_ok=True)
    s3_to_local(bucket, prefix=prefix, dest=input_path)

    grid_file = input_path / Path(config["grid_file"])
    # Grid file gets a heavier check (full load) so leave it sequential.
    check_xr_fully(grid_file)

    boot_file = input_path / Path(config["boot_file"])
    heatflux_file = input_path / Path(config["heatflux_file"])
    regrid_file = input_path / Path(config["regrid_file"])
    ocean_file = input_path / Path(config["ocean_file"])
    outline_file = input_path / Path(config["outline_file"])

    # Validate the lazy-check inputs concurrently; only invalid files print.
    input_lazy_files = [boot_file, heatflux_file, regrid_file, ocean_file]
    # Processes (not threads): HDF5 isn't reliably thread-safe across all
    # builds (Chinook segfaults), so each worker gets its own interpreter
    # and HDF5 state.
    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_path = {executor.submit(check_xr_lazy, p, verbose=False): p for p in input_lazy_files}
        for future in tqdm(
            as_completed(future_to_path),
            total=len(future_to_path),
            desc="Checking input files",
            unit="file",
        ):
            p = future_to_path[future]
            if not future.result():
                print(f"{p.resolve()} is not valid ✗")

    # Build file index (one row per climate file)
    files_dict: dict[str, str | Path] = {
        "boot_file": boot_file.resolve(),
        "grid_file": grid_file.resolve(),
        "heatflux_file": heatflux_file.resolve(),
        "ocean_file": ocean_file.resolve(),
        "outline_file": outline_file.resolve(),
        "regrid_file": regrid_file.resolve(),
    }

    gcms = config["gcms"]
    climatology = config["climatology"]
    version = config["version"]

    # Build tasks from per-GCM forcing pairs [[future, present], ...]
    tasks = [(gcm, pair[1], pair[0]) for gcm, pairs in gcms.items() for pair in pairs]
    dfs: list[pd.DataFrame] = []

    # Enumerate every climate file we'll need (climatology baseline + one per
    # GCM/forcing combo) so the validation can run in parallel below.
    climate_specs: list[tuple[str, Path]] = [
        (climatology, input_path / Path(f"{climatology}_{version}.nc")),
    ]
    for gcm, pd_forcing, ff in tasks:
        climate_specs.append(
            (
                f"gcm_{gcm}_exp_{ff}_{pd_forcing}",
                input_path / Path(f"{climatology}_{gcm}_anomalies_{ff}_{pd_forcing}_{version}.nc"),
            )
        )

    # Validate climate files concurrently. ``check_xr_lazy`` is mostly I/O
    # (open netCDF, sample a window); a worker pool overlaps the latency.
    # Suppress its per-file ✓ chatter and only surface invalid files here.
    invalid_climate_files: list[Path] = []
    # Processes (not threads): HDF5 isn't reliably thread-safe across all
    # builds (Chinook segfaults), so each worker gets its own interpreter
    # and HDF5 state.
    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_path = {executor.submit(check_xr_lazy, p, verbose=False): p for _, p in climate_specs}
        for future in tqdm(
            as_completed(future_to_path),
            total=len(future_to_path),
            desc="Checking climate files",
            unit="file",
        ):
            p = future_to_path[future]
            if not future.result():
                invalid_climate_files.append(p)
                print(f"{p.resolve()} is not valid ✗")

    for sample, climate_file in climate_specs:
        row = dict(files_dict)
        row["climate_file"] = climate_file.resolve()
        row["sample"] = sample
        dfs.append(pd.DataFrame.from_dict([row]))

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
    config_file = options.CONFIG_FILE[0]
    force_overwrite = options.force_overwrite
    output_path = options.output_path
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_file)
    config = cfg.campaign.as_params()

    s3_bucket: str = config.pop("bucket", "pism-cloud-data")
    s3_prefix: str = config.pop("prefix", "kitp/input")
    version: str = config.pop("version", "v2")
    s3_path = f"""{s3_prefix}/{version}"""

    is_df = stage(config, s3_bucket, s3_path, output_path, force_overwrite=force_overwrite)
    is_df.to_csv(output_path / Path(s3_path) / Path("ismip7_greenland_files.csv"))

    if options.bucket:
        prefix = f"{options.bucket_prefix}/kitp_greenland" if options.bucket_prefix else "kitp_greenland"
        local_to_s3(output_path, bucket=options.bucket, prefix=prefix)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
