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

# pylint: disable=too-many-positional-arguments

"""
Running.
"""
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import pandas as pd
import toml
from jinja2 import Environment, FileSystemLoader


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
    '-a 1
     -b 2'
    """
    return " ".join(f"\\ \n  -{k} {v}" for k, v in d.items())


def run_glacier(
    rgi_file: str | Path,
    config_file: str | Path,
    template_file: Path | str,
    path: str | Path = "result",
    resolution: None | str = None,
    debug: bool = False,
):
    """
    Configure and generate a PISM model run script for a glacier.

    This function reads glacier metadata from a CSV file and simulation settings
    from a TOML configuration file, builds a full PISM command-line string,
    renders an HPC submission script from a Jinja2 template, and saves both to disk.

    Parameters
    ----------
    rgi_file : str or Path
        Path to a CSV file with glacier metadata. Required columns:
        'rgi_id', 'boot_file', 'grid_file', 'historical_climate_file'.
    config_file : str or Path
        Path to a TOML file containing the PISM run configuration, including time,
        energy model, stress balance model, grid, and reporting options.
    template_file : str or Path
        Path to a Jinja2 shell script template (e.g., SLURM submission script).
        The template is rendered using job parameters like partition, ntasks, and walltime.
    path : str or Path, optional
        Base directory for storing model outputs. A subdirectory with the glacier RGI ID
        will be created within this path. Default is "result".
    resolution : str or None, optional
        Horizontal resolution to assign to the model grid (e.g., "500m").
        If None, uses the value from the config file.
    debug : bool, optional
        If True, skip rendering the template and only print the constructed run command.
        Default is False.

    Returns
    -------
    None
        This function writes the run script and TOML configuration file to disk.
        It prints the rendered run script including the full PISM command.

    Notes
    -----
    - Assumes the config file contains nested dictionaries under keys like
      'run', 'grid', 'surface', 'energy', 'stress_balance', etc.
    - The Jinja2 template should contain placeholders for 'queue', 'ntasks', and 'walltime'.
    - The function does not execute the model; it only prepares the necessary files for submission.
    """

    df = pd.read_csv(rgi_file)
    rgi_id = df["rgi_id"].iloc[0]

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    glacier_path = path / Path(rgi_id)
    glacier_path.mkdir(parents=True, exist_ok=True)

    output_path = glacier_path / Path("output")
    output_path.mkdir(parents=True, exist_ok=True)

    config_toml = toml.load(config_file)
    config = json.loads(json.dumps(config_toml))
    prefix = f"""{config["run"]["mpi"]} {config["run"]["ntasks"]} {config["run"]["exec"]} """

    template_file = Path(template_file)
    env = Environment(loader=FileSystemLoader(template_file.parent))
    template = env.get_template(template_file.name)

    run = {}
    if resolution is None:
        resolution = config["grid"]["resolution"]

    start = config["time"]["time.start"]
    end = config["time"]["time.end"]

    run.update(config["atmosphere"])
    run.update(config["geometry"])
    run.update(config["ocean"])
    run.update(config["grid"])
    run.update(config["calving"])
    run.update(config["iceflow"])
    run.update(config["surface"])
    run.update(config["reporting"])
    run.update(config["time"])
    run.update(config["input"])

    stress_balance = config["stress_balance"]["stress_balance.model"]
    run.update(config["stress_balance"]["options"][stress_balance])
    energy = config["energy"]["model"]
    run.update(config["energy"]["options"][energy])
    spatial_file = output_path / Path(
        f"spatial_g{resolution}_{rgi_id}_energy_{energy}_stress_balance_{stress_balance}_{start}_{end}.nc"
    )
    state_file = output_path / Path(
        f"state_g{resolution}_{rgi_id}_energy_{energy}_stress_balance_{stress_balance}_{start}_{end}.nc"
    )
    run.update(
        {
            "input.file": df["boot_file"].iloc[0],
            "grid.file": df["grid_file"].iloc[0],
            "grid.dx": resolution,
            "grid.dy": resolution,
            "output.file": state_file.absolute(),
            "output.extra.file": spatial_file.absolute(),
            "surface.force_to_thickness.file": df["boot_file"].iloc[0],
            "atmosphere.given.file": df["historical_climate_file"].iloc[0],
        }
    )
    run_str = dict2str(sort_dict_by_key(run))
    run_str = prefix + run_str

    # Variables to substitute
    params = {
        "queue": config["run"]["queue"],
        "ntasks": config["run"]["ntasks"],
        "walltime": config["run"]["walltime"],
    }

    if debug:
        rendered_script = ""
    else:
        # Render the template
        rendered_script = template.render(params)

    # Append the run_str
    rendered_script += f"\n\n{run_str}\n"

    run_script_path = glacier_path / Path("run_scripts")
    run_script_path.mkdir(parents=True, exist_ok=True)

    run_script = run_script_path / Path(
        f"submit_g{resolution}_{rgi_id}_energy_{energy}_stress_balance_{stress_balance}_{start}_{end}.sh"
    )
    # Save or print the output
    run_script.write_text(rendered_script)

    input_path = glacier_path / Path("input")
    glacier_filename = input_path / Path(f"rgi_{rgi_id}.gpkg")

    run_toml = {
        "rgi_id": rgi_id,
        "outline": str(glacier_filename.absolute()),
        "output": {"spatial": str(spatial_file.absolute()), "state": str(state_file.absolute())},
        "config": run,
    }
    run_file = output_path / Path(
        f"g{resolution}_{rgi_id}_energy_{energy}_stress_balance_{stress_balance}_{start}_{end}.toml"
    )
    with open(run_file, "w", encoding="utf-8") as toml_file:
        toml.dump(run_toml, toml_file)
    print(rendered_script)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Stage RGI Glacier."
    parser.add_argument(
        "--output_path",
        help="""Base path to save all files data/rgi_id/output. Default="data".""",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--resolution",
        help="""Override horizontal grid resolution. Default is None.""",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--debug",
        help="""Debug or testing mode, do not write template, just the run command. Default is False.""",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "RGI_FILE",
        help="""RGI CSV.""",
        nargs=1,
    )
    parser.add_argument(
        "CONFIG_FILE",
        help="""CONFIG TOML.""",
        nargs=1,
    )
    parser.add_argument(
        "TEMPLATE_FILE",
        help="""TEMPLATE J2.""",
        nargs=1,
    )

    options, unknown = parser.parse_known_args()
    path = options.output_path
    rgi_file = options.RGI_FILE[0]
    config_file = options.CONFIG_FILE[0]
    template_file = options.TEMPLATE_FILE[0]
    resolution = options.resolution
    debug = options.debug

    run_glacier(rgi_file, config_file, template_file, path=path, resolution=resolution, debug=debug)
