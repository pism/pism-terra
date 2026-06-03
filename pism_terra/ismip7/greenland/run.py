# Copyright (C) 2025, 2026 Andy Aschwanden
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

# pylint: disable=too-many-positional-arguments,broad-exception-caught,unused-variable

"""
Running.
"""

from __future__ import annotations

import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import toml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pyfiglet import Figlet

from pism_terra.config import JobConfig, load_config, load_uq
from pism_terra.ismip7.greenland.stage import stage
from pism_terra.sampling import create_samples
from pism_terra.workflow import (
    apply_choice_mapping,
    dict2str,
    filter_overrides_by_config,
    normalize_row,
    sort_dict_by_key,
    validate_pism_options,
)

# one Jinja environment for all renders
_JINJA = Environment(undefined=StrictUndefined, autoescape=False)


def run_greenland(
    config_file: str | Path,
    template_file: Path | str,
    outline_file: Path | str | None,
    path: str | Path = "result",
    config_cli: dict | None = None,
    debug: bool = False,
    *,
    uq: Mapping[str, object] | pd.Series | None = None,
    sample: int | None = None,
    pism_config_cdl: str | Path | None = None,
):
    """
    Configure and generate a PISM job script for a single glacier (ensemble-ready).

    Reads a TOML configuration, merges optional ensemble overrides (``uq``),
    renders a submission script from a Jinja2 template, and writes both the
    script and a companion TOML describing the resolved run parameters.
    Also emits a command-line string of PISM flags derived from the config and
    overrides.

    Parameters
    ----------
    config_file : str or pathlib.Path
        Path to the PISM configuration TOML (contains ``run``, ``grid``,
        ``time``, ``surface``, ``energy``, ``stress_balance``, etc.).
    template_file : str or pathlib.Path
        Path to a Jinja2 submission template (e.g., SLURM/LSF script). The
        context is populated from validated ``RunConfig`` and ``JobConfig``.
    outline_file : str or pathlib.Path
        Path to a geopandas file with the glacier outline.
    path : str or pathlib.Path, optional
        Base output directory. ``output/`` and ``run_scripts/`` subdirectories
        are created inside it. Default is ``"result"``.
    config_cli : dict or None, optional
        CLI-side overrides applied after reading the config. Recognized keys:
        ``"resolution"`` (e.g. ``"200m"``), ``"nodes"`` (int), ``"ntasks"``
        (int), ``"tasks"`` (int, MPI tasks per node), ``"queue"`` (str),
        ``"walltime"`` (``HH:MM:SS``), ``"stress_balance"`` (sub-model name
        swap, e.g. ``"sia"``), and ``"start"`` / ``"end"`` (``YYYY-MM-DD``
        time bounds). Any value of ``None`` falls back to the config file.
        Default is ``None`` (no overrides).
    debug : bool, optional
        If ``True``, skip rendering the template (leave it empty) but still
        append the constructed PISM command line to the output script.
        Default is ``False``.
    uq : Mapping[str, object] or pandas.Series or None, optional
        Ensemble overrides. Keys are **dotted PISM flags** (e.g.,
        ``"surface.pdd.factor_ice"``, ``"input.file"``). Values are inserted into
        the run dictionary and thus into the generated command line. If ``uq``
        contains a key ``"sample"``, it is used (when ``sample`` is not provided)
        to suffix output filenames and scripts.
    sample : int or None, optional
        Ensemble member identifier. If not provided, and ``uq`` has
        ``"sample"``, that value is used. The value changes the filename
        stem used for outputs (e.g., ``..._s0042``). If neither is provided,
        filenames use a descriptive ``surface/energy/stress_balance`` suffix.
    pism_config_cdl : str or Path or None, optional
        Path to a PISM CDL master config file. If provided, all run options
        are validated against it before generating the command line.

    Raises
    ------
    ValueError
        If configuration validation fails upstream (e.g., via Pydantic models),
        or if provided overrides are of incompatible types.

    Notes
    -----
    - The Jinja2 context is populated from validated ``RunConfig`` and
      ``JobConfig`` (config values) plus any CLI overrides provided here
      for ``ntasks``, ``nodes``, ``queue``, ``walltime``.
    - ``uq`` overrides are merged **after** reading the config; they can set or
      replace any dotted PISM flag (e.g., swapping input or forcing files).
    - The function attempts to open NetCDF inputs referenced by keys ending
      with ``.file`` (excluding ``output.*``) using ``xarray.open_dataset`` and
      prints a ✓/✗ check; it does not stop the run on failure.

    Examples
    --------
    Basic use with config and template:

    >>> run_greenland(
    ...     config_file="config/init_stampede3.toml",
    ...     template_file="templates/stampede3.j2",
    ...     outline_file=None,
    ...     path="result",
    ... )

    Ensemble member with overrides from a pandas row (e.g., Latin Hypercube):

    >>> row = df_samples.loc[17]  # contains dotted keys + 'sample'
    >>> run_greenland(
    ...     config_file="config/init_stampede3.toml",
    ...     template_file="templates/stampede3.j2",
    ...     outline_file=None,
    ...     uq=row,             # dotted PISM flags to override
    ...     sample=None,        # will be inferred from row['sample'] if present
    ...     ntasks=112,         # optional template/run override
    ... )
    """

    cfg = load_config(config_file)

    config_cli = config_cli or {}
    resolution = config_cli.get("resolution")
    if resolution:
        resolution = re.sub(r"\s+", "", resolution)

        # update GridConfig and force dx/dy to be derived from the new resolution
        cfg.grid.resolution = resolution
        cfg.grid.dx = None
        cfg.grid.dy = None

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    log_path = path / Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    output_path = path / Path("output")
    output_path.mkdir(parents=True, exist_ok=True)
    scalar_path = output_path / Path("scalar")
    scalar_path.mkdir(parents=True, exist_ok=True)
    spatial_path = output_path / Path("spatial")
    spatial_path.mkdir(parents=True, exist_ok=True)
    state_path = output_path / Path("state")
    state_path.mkdir(parents=True, exist_ok=True)

    run = {}
    for section in (
        "geometry",
        "calving",
        "iceflow",
        "reporting",
        "input",
        "time_stepping",
    ):
        run.update(getattr(cfg, section))
    run.update(cfg.atmosphere.selected())
    run.update(cfg.energy.selected())
    run.update(cfg.ocean.selected())
    run.update(cfg.frontal_melt.selected())
    run.update(cfg.grid.as_params())
    run.update(cfg.hydrology.selected())
    run.update(cfg.run_info.as_params())
    run.update(cfg.surface.selected())
    run.update(cfg.stress_balance.selected())
    run.update(cfg.time.as_params())

    template_file = Path(template_file)
    env = Environment(loader=FileSystemLoader(template_file.parent))
    template = env.get_template(template_file.name)

    start = cfg.model_dump(by_alias=True)["time"]["time.start"]
    end = cfg.model_dump(by_alias=True)["time"]["time.end"]
    writer = cfg.model_dump()["run"]["writer"] if (cfg.model_dump()["run"]["writer"] is not None) else ""

    if resolution is None:
        resolution = cfg.model_dump(by_alias=True)["grid"]["resolution"]
    # CLI override for the stress-balance model. Drop the previous model's
    # options from ``run`` first so leftover keys (e.g. blatter.*) don't
    # leak into e.g. a sia run.
    stress_balance = config_cli.get("stress_balance")
    if stress_balance is not None:
        for old_key in cfg.stress_balance.selected():
            run.pop(old_key, None)
        cfg.stress_balance.model = stress_balance
        run.update(cfg.stress_balance.selected())
    stress_balance = cfg.model_dump(by_alias=True)["stress_balance"]["model"]

    # CLI overrides for time bounds. ``cfg.time`` is a TimeConfig pydantic
    # model with field names ``time_start`` / ``time_end`` (aliased to the
    # dotted ``"time.start"`` / ``"time.end"``), so attribute assignment is
    # required. Drop the prior dotted entry from ``run`` and re-apply via
    # ``as_params()`` so the new value lands cleanly.
    _start = config_cli.get("start")
    _end = config_cli.get("end")
    if _start is not None:
        run.pop("time.start", None)
        cfg.time.time_start = _start
        run.update(cfg.time.as_params())
    if _end is not None:
        run.pop("time.end", None)
        cfg.time.time_end = _end
        run.update(cfg.time.as_params())

    energy = cfg.model_dump(by_alias=True)["energy"]["model"]
    surface = cfg.model_dump(by_alias=True)["surface"]["model"]

    if sample is None:
        name_options = f"surface_{surface}_energy_{energy}_stress_balance_{stress_balance}"
    else:
        name_options = f"id_{sample}"

    uq_clean = normalize_row(uq) if uq is not None else {}
    # Prefer explicit `sample` arg; else default from uq['sample']
    if sample is None and "sample" in uq_clean:
        try:
            sample = int(uq_clean["sample"])
        except Exception:
            pass

    # Remove 'sample' from flag overrides; drop any key not in the config-derived
    # run dict (e.g., surface.debm_simple.std_dev.file when surface.model == "pdd").
    overrides = {k: v for k, v in uq_clean.items() if k != "sample"}
    overrides, skipped = filter_overrides_by_config(overrides, run.keys())
    if skipped:
        print(f"Skipping uq overrides not in config: {skipped}")
    # Apply to runtime dict (these should be dotted PISM flags)
    run.update(overrides)

    scalar_file = scalar_path / Path(f"scalar_g{resolution}_{name_options}_{start}_{end}.nc")
    spatial_file = spatial_path / Path(f"spatial_g{resolution}_{name_options}_{start}_{end}.nc")
    state_file = state_path / Path(f"state_g{resolution}_{name_options}_{start}_{end}.nc")
    run.update(
        {
            "output.file": state_file.resolve(),
            "output.spatial.file": spatial_file.resolve(),
            "output.scalar.file": scalar_file.resolve(),
        }
    )

    if pism_config_cdl is not None:
        validate_pism_options(run, pism_config_cdl)

    run_str = dict2str(sort_dict_by_key(run))

    job_opts = JobConfig(**cfg.job.model_dump())

    params = {
        **job_opts.model_dump(exclude_none=True, by_alias=True),
    }

    job_kwargs = {
        k: v
        for k, v in {
            "nodes": config_cli.get("nodes"),
            "ntasks": config_cli.get("ntasks"),
            "queue": config_cli.get("queue"),
            "output_path": log_path.resolve(),
            "tasks": config_cli.get("tasks"),
            "walltime": config_cli.get("walltime"),
        }.items()
        if v is not None
    }
    if job_kwargs:
        params.update(JobConfig(**job_kwargs).as_params())

    outline_file = str(Path(outline_file).resolve()) if (outline_file is not None) else "none"
    run_toml = {
        "basin": {"basin": "Mouginot/Rignot", "outline": outline_file},
        "output": {
            "spatial": str(spatial_file.resolve()),
            "state": str(state_file.resolve()),
        },
        "config": run,
    }
    post_path = output_path / Path("post_processing")
    post_path.mkdir(parents=True, exist_ok=True)

    post_file = post_path / Path(f"g{resolution}_{name_options}_{start}_{end}.toml")
    with open(post_file, "w", encoding="utf-8") as toml_file:
        toml.dump(run_toml, toml_file)

    params.update({"run_str": run_str})
    rendered_script = "" if debug else template.render(params)

    run_script_path = path / Path("run_scripts")
    run_script_path.mkdir(parents=True, exist_ok=True)

    run_script = run_script_path / Path(f"submit_g{resolution}_{name_options}_{start}_{end}.sh")

    # Save or print the output
    run_script.write_text(rendered_script)

    print(f"\nJob script written to {run_script.resolve()}\n")


def run_single():
    """
    Run single glacier.
    """

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Run ISMIP7 Greenland."
    parser.add_argument(
        "--force-overwrite",
        help="Force downloading all files.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--output-path",
        help="Base path to save all files to. Files will be saved in `f'{out_path}/{RGI_ID}/output/'`.",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--queue",
        help="Overrides queue in config file.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ntasks",
        help="Numbers of cores.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--tasks",
        help="Cores per node.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--nodes",
        help="Overrides nodes in config file.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--walltime",
        help="Overrides walltime in config file.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--resolution",
        help="Override horizontal grid resolution.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--start",
        help="Override the time.start selection.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--end",
        help="Override the time.end selection.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--stress-balance",
        help="Override the [stress_balance].model selection (e.g. 'sia', 'blatter').",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--debug",
        help="Debug or testing mode, do not write template, just the run command.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--pism-config-cdl",
        help="Path to PISM CDL config file for option validation.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "CONFIG_FILE",
        help="CONFIG TOML.",
        nargs=1,
    )
    parser.add_argument(
        "TEMPLATE_FILE",
        help="TEMPLATE J2.",
        nargs=1,
    )

    options, _ = parser.parse_known_args()
    force_overwrite = options.force_overwrite
    path = options.output_path
    config_file = options.CONFIG_FILE[0]
    template_file = options.TEMPLATE_FILE[0]
    resolution = options.resolution
    debug = options.debug
    queue = options.queue
    ntasks = options.ntasks
    nodes = options.nodes
    tasks = options.tasks
    walltime = options.walltime
    stress_balance = options.stress_balance
    start_cli = options.start
    end_cli = options.end
    pism_config_cdl = options.pism_config_cdl

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    input_path = path / Path("input")
    input_path.mkdir(parents=True, exist_ok=True)
    output_path = path / Path("output")
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_file)
    campaign_config = cfg.campaign.as_params()

    bucket = campaign_config["bucket"]
    prefix = campaign_config["prefix"]

    df = stage(campaign_config, bucket=bucket, prefix=prefix, path=path, force_overwrite=force_overwrite)

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    print("=" * 120)
    print(banner)
    print("=" * 120)
    print("Generate Run for ISMIP7")
    print("-" * 120)
    for idx, row in df.iterrows():
        uq = {
            "input.file": row["boot_file"],
            "input.regrid.file": row["regrid_file"],
            "frontal_melt.routing.file": row["frontal_melt_file"],
            "geometry.front_retreat.prescribed.file": row["retreat_file"],
            "grid.file": row["grid_file"],
            "energy.bedrock_thermal.file": row["heatflux_file"],
            "atmosphere.given.file": row["climate_file"],
            "surface.given.file": row["climate_file"],
            "surface.ismip6.file": row["climate_file"],
            "surface.ismip6.reference_file": row["climate_file"],
            "hydrology.surface_input.file": row["surface_input_file"],
            "ocean.th.file": row["ocean_file"],
        }
        sample = int(row["sample"]) if "sample" in row else idx
        outline_file = row["outline_file"] if "outline_file" in row else None
        run_greenland(
            config_file,
            template_file,
            outline_file,
            path=path,
            config_cli={
                "resolution": resolution,
                "nodes": nodes,
                "ntasks": ntasks,
                "tasks": tasks,
                "queue": queue,
                "walltime": walltime,
                "stress_balance": stress_balance,
                "start": start_cli,
                "end": end_cli,
            },
            debug=debug,
            uq=uq,
            sample=sample,
            pism_config_cdl=pism_config_cdl,
        )


def run_ensemble():
    """
    Run single glacier.
    """

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Run RGI Glacier Ensemble."
    parser.add_argument(
        "--output-path",
        help="Base path to save all files to. Files will be saved in `f'{out_path}/{RGI_ID}/output/'`.",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--queue",
        help="Overrides queue in config file.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ntasks",
        help="Numbers of cores.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--tasks",
        help="Cores per node.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--nodes",
        help="Overrides nodes in config file.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--walltime",
        help="Overrides walltime in config file.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--resolution",
        help="Override horizontal grid resolution.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--start",
        help="Override the time.start selection.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--end",
        help="Override the time.end selection.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--posterior-file",
        help="CSV file posterior parameter distributions to sample from. Default=None.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--stress-balance",
        help="Override the [stress_balance].model selection (e.g. 'sia', 'blatter').",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--debug",
        help="Debug or testing mode, do not write template, just the run command.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--pism-config-cdl",
        help="Path to PISM CDL config file for option validation.",
        type=str,
        default=None,
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
    parser.add_argument(
        "TEMPLATE_FILE",
        help="TEMPLATE J2.",
        nargs=1,
    )
    parser.add_argument(
        "UQ_FILE",
        help="UQ TOML.",
        nargs=1,
    )

    options, _ = parser.parse_known_args()
    force_overwrite = options.force_overwrite
    path = options.output_path
    config_file = options.CONFIG_FILE[0]
    template_file = options.TEMPLATE_FILE[0]
    uq_file = options.UQ_FILE[0]
    resolution = options.resolution
    posterior_file = options.posterior_file
    debug = options.debug
    queue = options.queue
    ntasks = options.ntasks
    nodes = options.nodes
    tasks = options.tasks
    walltime = options.walltime
    stress_balance = options.stress_balance
    start_cli = options.start
    end_cli = options.end
    pism_config_cdl = options.pism_config_cdl

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    input_path = path / Path("input")
    input_path.mkdir(parents=True, exist_ok=True)
    output_path = path / Path("output")
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_file)
    campaign_config = cfg.campaign.as_params()

    bucket = campaign_config["bucket"]
    prefix = campaign_config["prefix"]

    df = stage(campaign_config, bucket=bucket, prefix=prefix, path=path, force_overwrite=force_overwrite)

    seed = 42
    rng = np.random.default_rng(seed=seed)
    uq = load_uq(uq_file)
    n_samples = uq.samples
    mapping = uq.mapping

    uq_df = create_samples(uq.to_flat(), n_samples=n_samples, seed=seed)
    if posterior_file is not None:
        posterior_df = pd.read_csv(posterior_file).drop(columns=["Unnamed: 0", "exp_id"], errors="ignore")
        choice_indices = rng.choice(range(len(posterior_df)), n_samples)
        posterior_sampled_df = posterior_df.iloc[choice_indices].reset_index(drop=True)
        duplicate_cols = list(set(uq_df.columns) & set(posterior_sampled_df.columns) - {"sample"})
        if duplicate_cols:
            print(f"WARNING: posterior overrides UQ for columns: {sorted(duplicate_cols)}")
            uq_df = uq_df.drop(columns=duplicate_cols)
        uq_df = pd.concat([uq_df, posterior_sampled_df], axis=1)

    uq_file = output_path / Path("uq.csv")
    uq_df.rename(columns={"sample": "id"}).to_csv(uq_file, index=False)

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    print("=" * 120)
    print(banner)
    print("=" * 120)
    print("Generate Ensemble Runs for Greenland")
    print("-" * 120)

    if uq.mapping:
        uq_df = apply_choice_mapping(uq_df, df, uq.mapping)

    merged_df = df.merge(uq_df, how="cross", suffixes=("_df", "_uq"))
    merged_df["sample"] = merged_df["sample_df"].astype(str) + "_uq_" + merged_df["sample_uq"].astype(int).astype(str)
    merged_df = merged_df.drop(columns=["sample_df", "sample_uq"])

    for _, row in merged_df.iterrows():
        row_uq = {
            "input.file": row["boot_file"],
            "input.regrid.file": row["regrid_file"],
            "frontal_melt.routing.file": row["frontal_melt_file"],
            "geometry.front_retreat.prescribed.file": row["retreat_file"],
            "grid.file": row["grid_file"],
            "energy.bedrock_thermal.file": row["heatflux_file"],
            "atmosphere.given.file": row["climate_file"],
            "surface.given.file": row["climate_file"],
            "surface.ismip6.file": row["climate_file"],
            "surface.ismip6.reference_file": row["climate_file"],
            "hydrology.surface_input.file": row["surface_input_file"],
            "ocean.th.file": row["ocean_file"],
        }
        row_uq.update(row.drop(labels=list(df.columns) + ["sample"]).to_dict())
        sample = row["sample"]
        outline_file = row["outline_file"] if "outline_file" in row else None
        run_greenland(
            config_file,
            template_file,
            outline_file,
            path=path,
            config_cli={
                "resolution": resolution,
                "nodes": nodes,
                "ntasks": ntasks,
                "tasks": tasks,
                "queue": queue,
                "walltime": walltime,
                "stress_balance": stress_balance,
                "start": start_cli,
                "end": end_cli,
            },
            debug=debug,
            uq=row_uq,
            sample=sample,
            pism_config_cdl=pism_config_cdl,
        )


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    run_single()
