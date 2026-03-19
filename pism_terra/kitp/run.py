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

from pism_terra.config import JobConfig, RunConfig, load_config, load_uq
from pism_terra.kitp.stage import stage
from pism_terra.sampling import create_samples
from pism_terra.workflow import (
    apply_choice_mapping,
    dict2str,
    merge_model,
    normalize_row,
    sort_dict_by_key,
)

# one Jinja environment for all renders
_JINJA = Environment(undefined=StrictUndefined, autoescape=False)


def run_kitp(
    config_file: str | Path,
    template_file: Path | str,
    outline_file: Path | str,
    path: str | Path = "result",
    resolution: None | str = None,
    nodes: None | int = None,
    ntasks: None | int = None,
    queue: None | str = None,
    walltime: None | str = None,
    debug: bool = False,
    *,
    uq: Mapping[str, object] | pd.Series | None = None,
    sample: int | None = None,
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
        Base output directory. A subfolder ``<path>/<rgi_id>`` is created with
        ``output/`` and ``run_scripts/`` subdirectories. Default is ``"result"``.
    resolution : str or None, optional
        Grid resolution (e.g., ``"200m"``). If ``None``, the value from
        ``[grid].resolution`` in the config is used.
    nodes : int or None, optional
        Node count override for the submission template. If ``None``, use config.
    ntasks : int or None, optional
        MPI task count override for the submission template/run options.
        If ``None``, use config.
    queue : str or None, optional
        Batch queue/partition override for the submission template. If ``None``,
        use config.
    walltime : str or None, optional
        Wall time override in ``HH:MM:SS``. If ``None``, use config.
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

    >>> run_glacier(
    ...     rgi_id="RGI2000-v7.0-C-01-04374",
    ...     config_file="config/init_stampede3.toml",
    ...     template_file="templates/stampede3.j2",
    ...     path="result",
    ... )

    Ensemble member with overrides from a pandas row (e.g., Latin Hypercube):

    >>> row = df_samples.loc[17]  # contains dotted keys + 'sample'
    >>> run_glacier(
    ...     rgi_id="RGI2000-v7.0-C-01-04374",
    ...     config_file="config/init_stampede3.toml",
    ...     template_file="templates/stampede3.j2",
    ...     uq=row,             # dotted PISM flags to override
    ...     sample=None,        # will be inferred from row['sample'] if present
    ...     ntasks=112,         # optional template/run override
    ... )
    """

    outline_file = Path(outline_file)
    cfg = load_config(config_file)

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
        "ocean",
        "calving",
        "iceflow",
        "reporting",
        "input",
        "time_stepping",
    ):
        run.update(getattr(cfg, section))
    run.update(cfg.atmosphere.selected())
    run.update(cfg.energy.selected())
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
    stress_balance = cfg.model_dump(by_alias=True)["stress_balance"]["model"]
    energy = cfg.model_dump(by_alias=True)["energy"]["model"]
    surface = cfg.model_dump(by_alias=True)["surface"]["model"]

    if sample is None:
        name_options = f"surface_{surface}_energy_{energy}_stress_balance_{stress_balance}"
    else:
        name_options = f"id_{sample}"
        # run.update({"output.experiment_id": sample})

    uq_clean = normalize_row(uq) if uq is not None else {}
    # Prefer explicit `sample` arg; else default from uq['sample']
    if sample is None and "sample" in uq_clean:
        try:
            sample = int(uq_clean["sample"])
        except Exception:
            pass

    # Remove 'sample' from flag overrides
    overrides = {k: v for k, v in uq_clean.items() if k != "sample"}
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

    run_str = dict2str(sort_dict_by_key(run)) + f" {writer}"

    run_opts = RunConfig(**cfg.run.model_dump())
    job_opts = JobConfig(**cfg.job.model_dump())

    params = {
        **run_opts.model_dump(exclude_none=True, by_alias=True),
        **job_opts.model_dump(exclude_none=True, by_alias=True),
    }

    # run_opts comes from your config; ntasks comes from CLI (or None)
    active_run_opts = merge_model(run_opts, ntasks=ntasks)

    # Use this ONE source to update params and to compute mpi_str
    run_params = active_run_opts.as_params()
    params.update(run_params)
    mpi_str = run_params["mpi"]  # guaranteed consistent with ntasks override

    job_kwargs = {
        k: v
        for k, v in {"queue": queue, "walltime": walltime, "nodes": nodes, "output_path": log_path.resolve()}.items()
        if v is not None
    }
    if job_kwargs:
        params.update(JobConfig(**job_kwargs).as_params())

    run_toml = {
        "basin": {"basin": "Mouginot/Rignot", "outline": str(outline_file.resolve())},
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

    prefix = f"{mpi_str} {cfg.run.executable} "
    postfix = f"pism-kitp-postprocess {post_file}"
    rendered_script = "" if debug else template.render(params)
    rendered_script += f"\n\n{prefix}{run_str}\n\n{postfix}"

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
    parser.description = "Run KITP Greenland."
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
        help="Overrides ntatsks in config file.",
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
        "--debug",
        help="Debug or testing mode, do not write template, just the run command.",
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
    walltime = options.walltime

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
    print(df)
    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    print("=" * 120)
    print(banner)
    print("=" * 120)
    print("Generate Run for KITP")
    print("-" * 120)
    for idx, row in df.iterrows():
        uq = {
            "input.file": row["boot_file"],
            "input.regrid.file": row["regrid_file"],
            "energy.bedrock_thermal.file": row["heatflux_file"],
            "grid.file": row["grid_file"],
            "atmosphere.given.file": row["climate_file"],
        }
        outline_file = row["outline_file"]
        run_kitp(
            config_file,
            template_file,
            outline_file,
            path=path,
            resolution=resolution,
            nodes=nodes,
            ntasks=ntasks,
            queue=queue,
            walltime=walltime,
            debug=debug,
            uq=uq,
            sample=row["sample"] if "sample" in row else idx,
        )


def run_ensemble():
    """
    Run KTIP UQ Ensemble.
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
        help="Overrides ntatsks in config file.",
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
        "--posterior-file",
        help="CSV file posterior parameter distributions to sample from. Default=None.",
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
    walltime = options.walltime

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
    outline_file = df["outline_file"].iloc[0]

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
            "energy.bedrock_thermal.file": row["heatflux_file"],
            "grid.file": row["grid_file"],
            "atmosphere.given.file": row["climate_file"],
        }
        row_uq.update(row.drop(labels=list(df.columns) + ["sample"]).to_dict())
        outline_file = row["outline_file"]
        sample = row["sample"]
        run_kitp(
            config_file,
            template_file,
            outline_file,
            path=path,
            resolution=resolution,
            nodes=nodes,
            ntasks=ntasks,
            queue=queue,
            walltime=walltime,
            debug=debug,
            uq=row_uq,
            sample=sample,
        )


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    run_single()
