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
Staging of Greenland paleo inputs.
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from pyfiglet import Figlet

from pism_terra.aws import download_from_s3, local_to_s3
from pism_terra.config import load_config
from pism_terra.ismip7.greenland.stage import select_grid_file
from pism_terra.workflow import check_xr_fully, check_xr_lazy

#: Campaign keys of the inputs borrowed from another campaign (``shared_prefix``).
SHARED_FILES = ("boot_file", "heatflux_file", "regrid_file")
#: Campaign keys of the inputs ``pism-greenland-paleo-prepare`` builds.
PALEO_FILES = ("climate_file", "ocean_file", "delta_T_file", "delta_SL_file", "ocean_delta_T_file")
#: Scalar series: too small for the windowed check the gridded files get.
SERIES_FILES = ("delta_T_file", "delta_SL_file", "ocean_delta_T_file")


def stage(
    config: dict,
    path: str | Path = "input_files",
    force_overwrite: bool = False,
    data_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Stage Greenland paleo inputs and return a file index.

    The grid, boot, heat-flux and initial-state files are the ISMIP7 ones and
    come from ``shared_prefix``; the base climate and the scalar series come
    from the campaign's own ``<prefix>/<version>``. Only the files the campaign
    names are downloaded, and only when missing locally.

    Parameters
    ----------
    config : dict
        Campaign parameters (``cfg.campaign.as_params()``). Must contain
        ``bucket``, ``prefix``, ``version``, ``shared_prefix``, ``grid_file``
        and every key of :data:`SHARED_FILES` and :data:`PALEO_FILES`.
    path : str or pathlib.Path, optional
        Output directory. Created if missing.
    force_overwrite : bool, optional
        Download files again even when they exist locally.
    data_path : str or pathlib.Path or None, optional
        Shared directory for the staged inputs; defaults to ``<path>/input``.

    Returns
    -------
    pandas.DataFrame
        One row with an absolute-path column per staged file (``grid_file``
        plus :data:`SHARED_FILES` and :data:`PALEO_FILES`) and a ``sample``
        column.
    """

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    print("=" * 120)
    print(banner)
    print("=" * 120)
    print("Stage Greenland paleo")
    print("-" * 120)
    print("")

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    input_path = Path(data_path) if data_path is not None else path / Path("input")
    input_path.mkdir(parents=True, exist_ok=True)

    bucket = config["bucket"]
    prefix = f"{config['prefix']}/{config['version']}"
    shared_prefix = str(config["shared_prefix"]).strip("/")

    grid_name = select_grid_file(config["grid_file"], input_path, bucket, shared_prefix)
    # (column, S3 prefix, key relative to it)
    required: list[tuple[str, str, str]] = [("grid_file", shared_prefix, grid_name)]
    required += [(key, shared_prefix, config[key]) for key in SHARED_FILES]
    required += [(key, prefix, config[key]) for key in PALEO_FILES]

    files = {key: input_path / Path(name) for key, _, name in required}
    to_download = [
        (f"s3://{bucket}/{key_prefix}/{name}", files[key])
        for key, key_prefix, name in required
        if force_overwrite or not files[key].exists()
    ]
    if to_download:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(download_from_s3, uri, local): uri for uri, local in to_download}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    print(f"Failed to download {futures[future]}: {exc}")

    check_xr_fully(files["grid_file"])
    for key, local in files.items():
        if key == "grid_file":
            continue
        valid = local.exists() if key in SERIES_FILES else check_xr_lazy(local, verbose=False)
        if not valid:
            print(f"{local.resolve()} is not valid ✗")

    row: dict[str, object] = {key: local.resolve() for key, local in files.items()}
    row["sample"] = "paleo"
    return pd.DataFrame.from_dict([row])


def main():
    """
    Run main script.
    """

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Stage Greenland paleo."
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
        default=Path("data/greenland_paleo"),
    )
    parser.add_argument(
        "--data-path",
        help="Shared directory for staged input data (reused across runs). Defaults to <output-path>/input.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help="Overrides campaign.version, the S3 subdirectory (<prefix>/<version>/) the paleo inputs are fetched from.",
    )
    parser.add_argument(
        "--force-overwrite",
        help="Force downloading all files.",
        action="store_true",
        default=False,
    )
    parser.add_argument("CONFIG_FILE", help="CONFIG TOML.", nargs=1)

    options = parser.parse_args()
    output_path = options.output_path
    data_path = options.data_path
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(options.CONFIG_FILE[0])
    if options.dataset_version is not None:
        cfg.campaign.version = options.dataset_version

    df = stage(cfg.campaign.as_params(), path=output_path, force_overwrite=options.force_overwrite, data_path=data_path)
    input_dir = Path(data_path) if data_path is not None else output_path / Path("input")
    df.to_csv(input_dir / Path("greenland_paleo_files.csv"))

    if options.bucket:
        prefix = f"{options.bucket_prefix}/greenland_paleo" if options.bucket_prefix else "greenland_paleo"
        local_to_s3(output_path, bucket=options.bucket, prefix=prefix)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
