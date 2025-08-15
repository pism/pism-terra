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
from __future__ import annotations

import json
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd
import toml
from jinja2 import Environment, FileSystemLoader


@dataclass
class RunOpts:
    """
    Options used to assemble batch/run parameters for template rendering.

    Attributes
    ----------
    queue : str or None
        Scheduler queue/partition name. If ``None``, the key is omitted
        from rendered parameters.
    ntasks : int or None
        Total number of MPI tasks (processes).
    walltime : str or None
        Wall clock time limit (e.g., ``"02:00:00"``). The accepted format
        depends on the scheduler.
    nodes : int or None
        Number of compute nodes to request.

    Notes
    -----
    This is a dataclass; the attributes also serve as constructor arguments
    with the same names and defaults. Validation (e.g., non-negative values
    or specific time formats) should be enforced upstream if needed.
    """

    queue: str | None = None
    ntasks: int | None = None
    walltime: str | None = None
    nodes: int | None = None

    def as_params(self) -> dict[str, Any]:
        """
        Convert options to a minimal parameter dictionary.

        Keys with "empty" values are dropped to keep templates clean.
        Boolean values (if present in the future) are always preserved,
        even when ``False``.

        Returns
        -------
        dict[str, Any]
            Dictionary containing only keys whose values are considered present.
            Values equal to ``None``, empty strings ``""``, empty lists ``[]``,
            or empty dicts ``{}`` are omitted.

        Examples
        --------
        >>> opts = RunOpts(queue="debug", ntasks=8)
        >>> opts.as_params()
        {'queue': 'debug', 'ntasks': 8}
        """
        return {k: v for k, v in asdict(self).items() if isinstance(v, bool) or v not in (None, "", [], {})}


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
    return " ".join(f" -{k} {v}" for k, v in d.items())


def run_glacier(
    rgi_file: str | Path,
    config_file: str | Path,
    template_file: Path | str,
    path: str | Path = "result",
    resolution: None | str = None,
    nodes: None | int = None,
    ntasks: None | int = None,
    queue: None | str = None,
    walltime: None | str = None,
    debug: bool = False,
):
    """
    Configure and generate a PISM job script for a single glacier.

    Reads glacier metadata from a CSV and model settings from a TOML file,
    assembles the full PISM command line, renders an HPC submission script
    from a Jinja2 template, and writes both the script and a resolved TOML
    (with output paths) to disk.

    Parameters
    ----------
    rgi_file : str or pathlib.Path
        Path to a CSV with glacier metadata. Required columns:
        ``'rgi_id'``, ``'boot_file'``, ``'grid_file'``, ``'historical_climate_file'``.
        If ``rgi_id == "RGI2000-v7.0-C-01-12784"``, column ``'cosipy_CCSM_file'`` is
        also expected.
    config_file : str or pathlib.Path
        Path to a TOML file containing the PISM run configuration. Expected
        top-level sections include ``run``, ``grid``, ``time``, ``surface``,
        ``energy``, ``stress_balance``, ``ocean``, ``geometry``, ``calving``,
        and ``reporting``. Keys under ``run`` should include ``mpi``,
        ``ntasks``, and ``exec``.
    template_file : str or pathlib.Path
        Path to a Jinja2 shell template (e.g., SLURM/LSF/PBS submit script).
        The template may reference job parameters such as ``queue``, ``ntasks``,
        ``nodes``, and ``walltime``.
    path : str or pathlib.Path, optional
        Base directory for all outputs. A subdirectory named after the
        glacier RGI ID is created inside this path. Default is ``"result"``.
    resolution : str or None, optional
        Horizontal grid spacing (e.g., ``"500m"``). If ``None``, the value
        from ``config['grid']['resolution']`` is used.
    nodes : int or None, optional
        Number of compute nodes to request. Overrides the value derived from
        ``config['run']`` when provided.
    ntasks : int or None, optional
        Total number of MPI ranks. Overrides ``config['run']['ntasks']`` when provided.
    queue : str or None, optional
        Scheduler queue/partition. Overrides ``config['run']['queue']`` when provided.
    walltime : str or None, optional
        Wall clock limit (scheduler format, e.g., ``"02:00:00"``). Overrides
        ``config['run']['walltime']`` when provided.
    debug : bool, optional
        If ``True``, skip template rendering (empty script body) but still
        print the assembled PISM command line. Default is ``False``.

    Notes
    -----
    The command prefix is constructed from ``config['run']['mpi']``,
    ``config['run']['ntasks']``, and ``config['run']['exec']``.
    Job parameters passed via function arguments take precedence over
    values in ``config['run']``. The template is rendered with only
    non-empty parameters (``queue``, ``ntasks``, ``nodes``, ``walltime``).

    Examples
    --------
    >>> from pathlib import Path
    >>> run_glacier(
    ...     rgi_file="glaciers.csv",
    ...     config_file="config.toml",
    ...     template_file="submit.slurm.j2",
    ...     path=Path("result"),
    ...     resolution="500m",
    ...     ntasks=64,
    ...     queue="normal",
    ...     walltime="04:00:00",
    ... )
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
            "atmosphere.elevation_change.file": df["historical_climate_file"].iloc[0],
        }
    )
    if rgi_id == "RGI2000-v7.0-C-01-12784":
        run.update({"surface.given.file": df["cosipy_CCSM_file"].iloc[0]})

    run_str = dict2str(sort_dict_by_key(run))

    base_kwargs = {f.name: config["run"].get(f.name) for f in fields(RunOpts)}
    opts = RunOpts(**base_kwargs)
    opts = RunOpts(**opts.as_params())  # optional: strip empties using your helper

    # override with CLI
    for k, v in {"queue": queue, "ntasks": ntasks, "walltime": walltime, "nodes": nodes}.items():
        if v is not None:
            setattr(opts, k, v)

    params = opts.as_params()
    rendered_script = "" if debug else template.render(params)

    prefix = f"""{config["run"]["mpi"]} {params["ntasks"]} {config["run"]["exec"]} """
    run_str = prefix + run_str
    # Append the run_str
    rendered_script += f"\n\n{run_str}\n"

    run_script_path = glacier_path / Path("run_scripts")
    run_script_path.mkdir(parents=True, exist_ok=True)

    run_script = run_script_path / Path(
        f"submit_g{resolution}_{rgi_id}_energy_{energy}_stress_balance_{stress_balance}_{start}_{end}.sh"
    )
    # Save or print the output
    run_script.write_text(rendered_script)

    output_glacier_filename = output_path / Path(f"rgi_{rgi_id}.gpkg")
    shutil.copy(rgi_file, output_glacier_filename)

    run_toml = {
        "rgi": {"rgi_id": rgi_id, "outline": str(output_glacier_filename.absolute())},
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
        "--queue",
        help="""Overrides queue in config file. Default=None.""",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ntasks",
        help="""Overrides ntatsks in config file. Default=None.""",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--nodes",
        help="""Overrides nodes in config file. Default=None.""",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--walltime",
        help="""Overrides walltime in config file. Default=None.""",
        type=str,
        default=None,
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
    queue = options.queue
    ntasks = options.ntasks
    nodes = options.nodes
    walltime = options.walltime

    run_glacier(
        rgi_file,
        config_file,
        template_file,
        path=path,
        resolution=resolution,
        nodes=nodes,
        ntasks=ntasks,
        queue=queue,
        walltime=walltime,
        debug=debug,
    )
