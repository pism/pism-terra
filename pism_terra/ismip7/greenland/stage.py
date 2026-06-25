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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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

from pism_terra.aws import download_from_s3, local_to_s3
from pism_terra.config import load_config
from pism_terra.workflow import check_dataset_fully, check_xr_fully, check_xr_lazy

xr.set_options(keep_attrs=True)


def stage(
    config: dict,
    path: str | Path = "input_files",
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
        - ``"pathway"`` : str
            ISMIP7 pathway identifier.
        - ``"gcms"`` : str or list[str]
            GCM model name(s).
        - ``"version"`` : str
            Dataset version.
        - ``"start_year"`` : int
            Start year of the forcing period.
        - ``"end_year"`` : int
            End year of the forcing period.
    path : str or pathlib.Path, default ``"input_files"``
        Output directory. Created if missing. All staged artifacts are written here.
    force_overwrite : bool, default ``False``
        If ``True``, downstream helpers may regenerate intermediate/final artifacts
        even if cache files exist.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per GCM and absolute-path columns including
        ``boot_file``, ``grid_file``, ``heatflux_file``, ``regrid_file``,
        ``outline_file``, ``climate_file``, ``ocean_file``,
        ``surface_input_file``, ``frontal_melt_file``, and ``sample``.
    """

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    print("=" * 120)
    print(banner)
    print("=" * 120)
    print("Stage ISMIP7 Greenland")
    print("-" * 120)
    print("")

    # Outputs dir
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    input_path = path / Path("input")
    if force_overwrite:
        input_path.unlink(missing_ok=True)
    input_path.mkdir(parents=True, exist_ok=True)

    bucket = config["bucket"]
    # ``prefix`` is a plain string in CampaignConfig; build the S3-side
    # prefix with f-string concatenation rather than Path division (Path /
    # str would TypeError on the leading str, and S3 keys aren't filesystem
    # paths anyway). Include ``version`` so the URL resolves to the layout
    # ``prepare`` writes, e.g. ``s3://…/ismip7/greenland/input/v2/…``.
    prefix = f"{config['prefix']}/{config['version']}"

    pathway = config["pathway"]
    gcms = config["gcms"]
    gcms = [gcms] if isinstance(gcms, str) else gcms
    version = config["version"]
    start_year = config["start_year"]
    end_year = config["end_year"]

    grid_file = input_path / Path(config["grid_file"])
    boot_file = input_path / Path(config["boot_file"])
    heatflux_file = input_path / Path(config["heatflux_file"])
    regrid_file = input_path / Path(config["regrid_file"])
    outline_file = input_path / Path(config["outline_file"])
    obs_file = input_path / Path(config["obs_file"])

    # Enumerate every S3 key we actually need so we don't bulk-sync the
    # whole prefix (which carries per-GCM forcings for GCMs we aren't
    # running, plus assorted bookkeeping files). ``required_files`` pairs
    # the rel-key under ``prefix`` with its target local path.
    required_files: list[tuple[str, Path]] = [
        (config["grid_file"], grid_file),
        (config["boot_file"], boot_file),
        (config["heatflux_file"], heatflux_file),
        (config["regrid_file"], regrid_file),
        (config["outline_file"], outline_file),
        (config["obs_file"], obs_file),
    ]
    # Only the final merged forcing files are published to S3 by
    # ``prepare`` — the per-epoch hist/proj outputs are now scratch and
    # disappear with the staging tempdir. The filename pattern below
    # matches that merged-file naming exactly (start_year is the
    # historical start, end_year the projection end). ``climate_gradient``
    # is the annual elevation-gradient companion of ``climate`` (see
    # ``prepare_ismip7_forcing``); validation downstream expects all three.
    for gcm in gcms:
        for forcing in ("climate", "climate_gradient", "ocean"):
            rel = f"ismip7_greenland_{forcing}_{pathway}_{gcm}_{version}.nc"
            required_files.append((rel, input_path / rel))

    # Skip files that already exist locally unless force_overwrite is set.
    # We intentionally do not re-validate cached files here; the explicit
    # validation passes below do that and surface failures.
    to_download = [
        (rel_key, local_path) for rel_key, local_path in required_files if force_overwrite or not local_path.exists()
    ]

    if to_download:
        # boto3 clients are safe for concurrent ``download_from_s3``; the
        # outer 4-way fan-out keeps the connection pool busy without
        # interleaving the per-file tqdm bars too aggressively.
        # Use a distinct name from the ``ProcessPoolExecutor`` blocks below
        # so mypy doesn't narrow the variable's type across reuse.
        with ThreadPoolExecutor(max_workers=4) as dl_executor:
            futures = {
                dl_executor.submit(
                    download_from_s3,
                    f"s3://{bucket}/{prefix}/{rel_key}",
                    local_path,
                ): rel_key
                for rel_key, local_path in to_download
            }
            for dl_future in as_completed(futures):
                try:
                    dl_future.result()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    print(f"Failed to download s3://{bucket}/{prefix}/{futures[dl_future]}: {exc}")

    # Grid file gets the heavier full-load check; leave it sequential.
    check_xr_fully(grid_file)

    # Validate the lazy-check inputs concurrently; only invalid files print.
    input_lazy_files = [boot_file, heatflux_file, regrid_file]
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
    files_dict = {
        "boot_file": boot_file.resolve(),
        "grid_file": grid_file.resolve(),
        "heatflux_file": heatflux_file.resolve(),
        "regrid_file": regrid_file.resolve(),
        "outline_file": outline_file.resolve(),
    }

    # Per-GCM climate/ocean forcing paths. ``surface_input_file`` aliases the
    # climate forcing and ``frontal_melt_file`` aliases the ocean forcing, so
    # we only need to validate the two distinct paths per GCM.
    def _forcing_path(forcing: str, gcm: str) -> Path:
        """
        Build the local path for one (forcing, gcm) merged NetCDF.

        Parameters
        ----------
        forcing : str
            ``"climate"`` or ``"ocean"``.
        gcm : str
            GCM name (e.g. ``"CESM2-WACCM"``).

        Returns
        -------
        pathlib.Path
            Absolute path to the merged forcing file under ``input_path``.
        """
        return input_path / Path(f"ismip7_greenland_{forcing}_{pathway}_{gcm}_{version}.nc")

    forcing_paths: dict[tuple[str, str], Path] = {}
    for gcm in gcms:
        for forcing in ("climate", "climate_gradient", "ocean"):
            forcing_paths[(gcm, forcing)] = _forcing_path(forcing, gcm)

    # Processes (not threads): HDF5 isn't reliably thread-safe across all
    # builds (Chinook segfaults), so each worker gets its own interpreter
    # and HDF5 state.
    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_path = {executor.submit(check_xr_lazy, p, verbose=False): p for p in forcing_paths.values()}
        for future in tqdm(
            as_completed(future_to_path),
            total=len(future_to_path),
            desc="Checking forcing files",
            unit="file",
        ):
            p = future_to_path[future]
            if not future.result():
                print(f"{p.resolve()} is not valid ✗")

    dfs: list[pd.DataFrame] = []
    for gcm in gcms:
        climate_file = forcing_paths[(gcm, "climate")]
        climate_gradient_file = forcing_paths[(gcm, "climate_gradient")]
        ocean_file = forcing_paths[(gcm, "ocean")]
        row = dict(files_dict)
        row["climate_file"] = climate_file.resolve()
        row["climate_gradient_file"] = climate_gradient_file.resolve()
        row["ocean_file"] = ocean_file.resolve()
        row["surface_input_file"] = climate_file.resolve()
        row["frontal_melt_file"] = ocean_file.resolve()
        row["sample"] = gcm
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
