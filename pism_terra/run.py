# Copyright (C) 2025 Andy Aschwanden
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
Staging.
"""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import pandas as pd
import xarray as xr


def merge_dicts(*dicts: dict) -> dict:
    """
    Merge multiple dictionaries into one.

    Parameters
    ----------
    *dicts : dict
        Dictionaries to merge.

    Returns
    -------
    dict
        A single dictionary containing all key-value pairs from the input dictionaries.
        If there are duplicate keys, the value from the last dictionary is used.
    """
    merged_dict = {}
    for d in dicts:
        merged_dict.update(d)
    return merged_dict


def sort_dict_by_key(d: dict) -> dict:
    """
    Sort a dictionary by its keys.

    Parameters
    ----------
    d : dict
        The dictionary to sort.

    Returns
    -------
    dict
        A new dictionary sorted by keys.
    """
    return {k: d[k] for k in sorted(d.keys())}


def dict2str(d: dict) -> str:
    """
    Convert a dictionary into a formatted string of key-value pairs.

    Parameters
    ----------
    d : dict
        The dictionary to convert.

    Returns
    -------
    str
        A string representation of the dictionary, where each key-value pair is
        formatted as `-key value` and pairs are separated by spaces.

    Examples
    --------
    >>> d = {"a": 1, "b": 2}
    >>> dict2str(d)
    '-a 1 -b 2'
    """
    return " ".join(f"-{k} {v}" for k, v in d.items())


def initialize_glacier(rgi_file: str | Path, path: str | Path = "result", resolution: str = "500m"):
    """
    Initialize configuration and directory structure for a glacier model run.

    This function reads glacier metadata from a CSV file and sets up a simulation
    directory with configuration attributes required to run a PISM (Parallel Ice
    Sheet Model) experiment. It modifies attributes of an existing template
    configuration file and prepares paths for output.

    Parameters
    ----------
    rgi_file : str or Path
        Path to a CSV file containing glacier-specific input metadata. Must include
        the columns: 'rgi_id', 'boot_file', 'grid_file', and 'historical_climate_file'.
    path : str or Path, optional
        Base output directory where the glacier subdirectory will be created.
        Default is "result".
    resolution : str, optional
        Horizontal grid resolution to assign to the model domain, e.g., "500m".
        This string is used in the output file name and the configuration.

    Notes
    -----
    - Assumes the `historical.nc` configuration template exists at "data/historical.nc".
    - The start and end times are hard-coded as 1980–2000.
    - Configuration attributes are updated in-memory and printed but not saved.
    - This function is a setup step and does not execute the actual model.
    """

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(rgi_file)
    rgi_id = df["rgi_id"].iloc[0]
    config = xr.open_dataarray("data/historical.nc")
    start = "1980-01-01"
    end = "2000-01-01"
    rgi_path = path / Path(rgi_id)
    rgi_path.mkdir(parents=True, exist_ok=True)

    blatter_dict = {
        "bp_ksp_monitor": "",
        "bp_ksp_view_singularvalues": "",
        "bp_snes_monitor_ratio": "",
        "bp_pc_type": "mg",
        "bp_mg_levels_ksp_type richardson": "",
        "bp_mg_levels_pc_type": "sor",
        "bp_mg_coarse_ksp_type": "preonly",
        "bp_mg_coarse_pc_type": "lu",
        "bp_pc_mg_levels": 3,
        "bp_pc_type mg": "",
        "bp_snes_ksp_ew": 1,
        "bp_snes_ksp_ew_version": 3,
    }
    config.attrs.update(blatter_dict)
    stress_balance_dict = {"stress_balance.model": "blatter"}
    config.attrs.update(stress_balance_dict)
    spatial_file = rgi_path / Path(f"spatial_g{resolution}_{rgi_id}_{start}_{end}.nc")
    config.attrs.update(
        {
            "input.file": df["boot_file"].iloc[0],
            "grid.file": df["grid_file"].iloc[0],
            "grid.dx": resolution,
            "grid.dy": resolution,
            "output.extra.file": spatial_file,
            "time.start": start,
            "time.end": end,
            "surface.models": "pdd,forcing",
            "surface.force_to_thickness.file": df["boot_file"].iloc[0],
            "atmosphere.given.file": df["historical_climate_file"].iloc[0],
        }
    )
    pism_config = config.attrs
    run_str = dict2str(sort_dict_by_key(pism_config))
    print(run_str)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Stage RGI Glacier."
    parser.add_argument(
        "--output_path",
        help="""Path to save all files. Default="data".""",
        type=str,
        default="result",
    )
    parser.add_argument(
        "--resolution",
        help="""Horizontal grid resolution. Default="500m".""",
        type=str,
        default="500m",
    )
    parser.add_argument(
        "RGI_FILE",
        help="""RGI CSV.""",
        nargs=1,
    )

    options, unknown = parser.parse_known_args()
    path = options.output_path
    rgi_file = options.RGI_FILE[0]
    resolution = options.resolution

    initialize_glacier(rgi_file, path=path, resolution=resolution)
