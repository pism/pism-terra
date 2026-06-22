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
from pism_terra.sampling import generate_samples
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


def _render_forward_run(
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
    Configure and generate a PISM forward job script for ISMIP7 Greenland (ensemble-ready).

    Reads a TOML configuration, merges optional ensemble overrides (``uq``),
    renders a submission script from a Jinja2 template, and writes both the
    script and a companion TOML describing the resolved run parameters. Also
    emits a command-line string of PISM flags derived from the config and
    overrides.

    Parameters
    ----------
    config_file : str or pathlib.Path
        Path to the PISM configuration TOML (contains ``run``, ``grid``,
        ``time``, ``surface``, ``energy``, ``stress_balance``, etc.).
    template_file : str or pathlib.Path
        Path to a Jinja2 submission template (e.g., SLURM/LSF script). The
        context is populated from validated ``RunConfig`` and ``JobConfig``.
    outline_file : str or pathlib.Path or None
        Path to a geopandas file with the basin outline used by
        post-processing. Pass ``None`` to record it as the literal string
        ``"none"``.
    path : str or pathlib.Path, optional
        Base output directory. ``output/`` and ``run_scripts/`` subdirectories
        are created inside it. Default is ``"result"``.
    config_cli : dict or None, optional
        CLI-side overrides applied after reading the config. Recognized keys:
        ``"resolution"`` (e.g. ``"500m"``), ``"nodes"`` (int), ``"ntasks"``
        (int), ``"tasks"`` (int, MPI tasks per node), ``"queue"`` (str),
        ``"walltime"`` (``HH:MM:SS``), ``"stress_balance"`` (sub-model name
        swap, e.g. ``"sia"``), and ``"start"`` / ``"end"`` (``YYYY-MM-DD``
        time bounds). Any value of ``None`` falls back to the config file.
        Default is ``None`` (no overrides).
    debug : bool, optional
        If ``True``, skip rendering the template (leave it empty) but still
        write the resolved post-processing TOML. Default is ``False``.
    uq : Mapping[str, object] or pandas.Series or None, optional
        Ensemble overrides. Keys are **dotted PISM flags** (e.g.,
        ``"surface.pdd.factor_ice"``, ``"input.file"``). Values are inserted
        into the run dictionary and thus into the generated command line. If
        ``uq`` contains a key ``"sample"``, it is used (when ``sample`` is
        not provided) to suffix output filenames and scripts.
    sample : int or None, optional
        Ensemble member identifier. If not provided, and ``uq`` has
        ``"sample"``, that value is used. The value changes the filename
        stem used for outputs (e.g., ``..._id_0042``). If neither is
        provided, filenames use a descriptive
        ``surface/energy/stress_balance`` suffix.
    pism_config_cdl : str or Path or None, optional
        Path to a PISM CDL master config file. If provided, all run options
        are validated against it before generating the command line.

    Raises
    ------
    ValueError
        If configuration validation fails upstream (e.g., via Pydantic models),
        or if provided overrides are of incompatible types.
    """

    outline_file = str(Path(outline_file).resolve()) if (outline_file is not None) else "none"
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
    print(f"Postprocessing script written to {post_file.resolve()}\n")


def _render_inverse_run(
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
    Configure and generate a PISM inverse job script for ISMIP7 Greenland (ensemble-ready).

    Same interface as :func:`_render_forward_run` but also builds an ``inv``
    dict of ``inverse.*`` / ``stress_balance.*`` flags that the Jinja2
    template can render as a ``pismi`` command line via ``inv_str``. Output
    files mirror the forward layout with an additional ``inverse/``
    subdirectory under ``output/``.

    Parameters
    ----------
    config_file : str or pathlib.Path
        Path to the PISM configuration TOML (contains ``run``, ``grid``,
        ``time``, ``surface``, ``energy``, ``stress_balance``, ``inverse``).
    template_file : str or pathlib.Path
        Path to a Jinja2 submission template. The context includes both
        ``run_str`` (forward command line) and ``inv_str`` (inverse command
        line) so a single template can launch the prior + pismi pair.
    outline_file : str or pathlib.Path or None
        Path to a geopandas file with the basin outline used by
        post-processing. Pass ``None`` to record it as the literal string
        ``"none"``.
    path : str or pathlib.Path, optional
        Base output directory. ``output/`` (with an extra ``inverse/``
        subdirectory) and ``run_scripts/`` are created inside it. Default
        is ``"result"``.
    config_cli : dict or None, optional
        CLI-side overrides applied after reading the config. See
        :func:`_render_forward_run` for the recognized keys. Default is
        ``None`` (no overrides).
    debug : bool, optional
        If ``True``, skip rendering the template (leave it empty) but still
        write the resolved post-processing TOML. Default is ``False``.
    uq : Mapping[str, object] or pandas.Series or None, optional
        Ensemble overrides. Keys are dotted PISM flags belonging to either
        the forward (``run``) or inverse (``inv``) dict; each key is routed
        to the dict that owns it, so e.g. ``inverse.file`` propagates into
        ``inv_str``.
    sample : int or None, optional
        Ensemble member identifier. If not provided, and ``uq`` has
        ``"sample"``, that value is used. Changes the filename stem used
        for outputs (e.g., ``..._id_0042``).
    pism_config_cdl : str or pathlib.Path or None, optional
        Path to a PISM CDL master config file. If provided, all forward
        run options are validated against it before generating the
        command line.
    """

    outline_file = str(Path(outline_file).resolve()) if (outline_file is not None) else "none"
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
    inv_path = output_path / Path("inverse")
    inv_path.mkdir(parents=True, exist_ok=True)

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

    inv: dict = {}
    inv.update(getattr(cfg, "iceflow"))
    inv.update(getattr(cfg, "inverse"))

    # cfg.stress_balance.selected() carries everything the forward run needs
    # (model options + PETSc solver knobs like bp_* / inv_adj_*). The pismi
    # call only needs the ``stress_balance.*`` dotted options; the solver
    # flags are picked up by the prior pism call (and inherited from the
    # state file). Filter so inv_str stays minimal.
    inv.update({k: v for k, v in cfg.stress_balance.selected().items() if k.startswith("stress_balance.")})

    template_file = Path(template_file)
    env = Environment(loader=FileSystemLoader(template_file.parent))
    template = env.get_template(template_file.name)

    # CLI overrides for time bounds. ``cfg.time`` is a TimeConfig pydantic
    # model with field names ``time_start`` / ``time_end`` (aliased to the
    # dotted ``"time.start"`` / ``"time.end"``), so we set attributes, not
    # items. We drop the prior value from ``run`` first and re-apply via
    # ``as_params()`` so the dotted alias replaces cleanly.
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

    start = cfg.model_dump(by_alias=True)["time"]["time.start"]
    end = cfg.model_dump(by_alias=True)["time"]["time.end"]

    if resolution is None:
        resolution = cfg.model_dump(by_alias=True)["grid"]["resolution"]
    # CLI override for the stress-balance model.
    stress_balance = config_cli.get("stress_balance")
    if stress_balance is not None:
        for old_key in cfg.stress_balance.selected():
            run.pop(old_key, None)
        cfg.stress_balance.model = stress_balance
        run.update(cfg.stress_balance.selected())
    stress_balance = cfg.model_dump(by_alias=True)["stress_balance"]["model"]

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

    # Drop any uq key that isn't in either the ``run`` or ``inv`` dicts (e.g.
    # surface.debm_simple.std_dev.file when surface.model == "pdd"). ``inverse.*``
    # keys live in ``inv`` only, so filtering against ``run.keys()`` alone would
    # silently drop them.
    all_overrides = {k: v for k, v in uq_clean.items() if k != "sample"}
    run_overrides, _ = filter_overrides_by_config(all_overrides, run.keys())
    inv_overrides, _ = filter_overrides_by_config(all_overrides, inv.keys())
    skipped = [k for k in all_overrides if k not in run and k not in inv]
    if skipped:
        print(f"Skipping uq overrides not in config: {skipped}")
    run.update(run_overrides)
    inv.update(inv_overrides)

    scalar_file = scalar_path / Path(f"scalar_g{resolution}_{name_options}_{start}_{end}.nc")
    spatial_file = spatial_path / Path(f"spatial_g{resolution}_{name_options}_{start}_{end}.nc")
    state_file = state_path / Path(f"state_g{resolution}_{name_options}_{start}_{end}.nc")
    run.update(
        {
            "output.file": state_file.resolve(),
            "output.scalar.file": scalar_file.resolve(),
            "output.spatial.file": spatial_file.resolve(),
        }
    )

    if pism_config_cdl is not None:
        validate_pism_options(run, pism_config_cdl)

    run_str = dict2str(sort_dict_by_key(run))

    inv_file = inv_path / Path(f"inv_g{resolution}_{name_options}_{start}_{end}.nc")
    # Feed the forward run's state file into pismi as its input.
    inv.update({"input.file": state_file.resolve()})
    inv.update({"o": inv_file.resolve()})
    inv_str = dict2str(sort_dict_by_key(inv))

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
    params.update({"inv_str": inv_str})
    rendered_script = "" if debug else template.render(params)

    run_script_path = path / Path("run_scripts")
    run_script_path.mkdir(parents=True, exist_ok=True)

    run_script = run_script_path / Path(f"submit_g{resolution}_{name_options}_{start}_{end}.sh")

    run_script.write_text(rendered_script)

    print(f"\nJob script written to {run_script.resolve()}\n")
    print(f"Postprocessing script written to {post_file.resolve()}\n")


def _nullable_string(argument_string: str) -> str | None:
    """
    Treat the literal CLI argument ``"none"`` as Python ``None``.

    Lets job submission systems that can't omit arguments pass a sentinel
    string instead of dropping the flag. Mirrors
    :func:`pism_terra.glacier.run._nullable_string`.

    Parameters
    ----------
    argument_string : str
        Argument string to parse.

    Returns
    -------
    str or None
        ``None`` if the argument is the case-insensitive literal ``"none"``,
        otherwise the argument unchanged.
    """
    if argument_string.strip().lower() == "none":
        return None
    return argument_string


def _build_cli_parser(description: str, *, supports_execute: bool) -> ArgumentParser:
    """
    Build the argparse parser shared by ``run_forward`` and ``run_inverse``.

    ``UQ_FILE`` is exposed as an *optional* positional: omit it to render one
    job script (single mode), supply it to render an ensemble.

    Parameters
    ----------
    description : str
        Parser description shown in ``--help``.
    supports_execute : bool
        Whether to add the ``--execute`` flag. Currently a placeholder for
        symmetry with the glacier CLI; ISMIP7 templates are normally
        submitted via SLURM rather than executed in-process.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = description
    parser.add_argument(
        "--output-path",
        help="Base path to save all files to.",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--force-overwrite",
        help="Force downloading all files.",
        action="store_true",
        default=False,
    )
    parser.add_argument("--queue", type=str, default=None, help="Overrides queue in config file.")
    parser.add_argument("--ntasks", type=int, default=None, help="Numbers of cores.")
    parser.add_argument("--tasks", type=int, default=None, help="Cores per node.")
    parser.add_argument("--nodes", type=int, default=None, help="Overrides nodes in config file.")
    parser.add_argument("--walltime", type=str, default=None, help="Overrides walltime in config file.")
    parser.add_argument(
        "--resolution", type=_nullable_string, default=None, help="Override horizontal grid resolution."
    )
    parser.add_argument(
        "--stress-balance",
        type=_nullable_string,
        default=None,
        help="Override the [stress_balance].model selection (e.g. 'sia', 'blatter').",
    )
    parser.add_argument("--start", type=_nullable_string, default=None, help="Override the time.start selection.")
    parser.add_argument("--end", type=_nullable_string, default=None, help="Override the time.end selection.")
    parser.add_argument(
        "--posterior-file",
        type=_nullable_string,
        default=None,
        help="CSV file of posterior parameter distributions to sample from (ensemble mode only).",
    )
    if supports_execute:
        parser.add_argument(
            "--execute",
            action="store_true",
            help="Reserved for parity with the glacier CLI; currently a no-op for ISMIP7.",
        )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug or testing mode, do not write template, just the run command.",
    )
    parser.add_argument(
        "--pism-config-cdl",
        type=_nullable_string,
        default=None,
        help="Path to PISM CDL config file for option validation.",
    )
    parser.add_argument("CONFIG_FILE", help="CONFIG TOML.")
    parser.add_argument("TEMPLATE_FILE", help="TEMPLATE J2.")
    parser.add_argument(
        "UQ_FILE",
        nargs="?",
        default=None,
        type=_nullable_string,
        help="UQ TOML (optional). Supply to render an ensemble; omit for a single run.",
    )
    return parser


def _build_ensemble_df(
    df: pd.DataFrame,
    uq_file: Path,
    output_path: Path,
    posterior_file: str | Path | None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build the per-member DataFrame for an ensemble run.

    Samples the UQ specification, optionally folds in a posterior CSV, then
    cross-joins with the staged DataFrame ``df`` and assigns a composite
    ``sample`` ID per row.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`pism_terra.ismip7.greenland.stage.stage` (one row
        per staged forcing tuple).
    uq_file : Path
        Path to the UQ TOML.
    output_path : Path
        Directory under which the realised sample CSV is persisted.
    posterior_file : str or Path or None
        Optional CSV of posterior parameter draws to override / extend the
        UQ samples with.
    seed : int, default 42
        Seed for sampling (and posterior row choice).

    Returns
    -------
    pandas.DataFrame
        Per-ensemble-member DataFrame with all columns from ``df`` plus the
        sampled UQ columns and a composite string ``sample`` column.
    """
    rng = np.random.default_rng(seed=seed)
    uq = load_uq(uq_file)
    n_samples = uq.samples

    uq_df = generate_samples(uq.to_flat(), n_samples=n_samples, method=uq.method, seed=seed)

    if posterior_file is not None:
        posterior_df = pd.read_csv(posterior_file).drop(columns=["Unnamed: 0", "exp_id"], errors="ignore")
        choice_indices = rng.choice(range(len(posterior_df)), n_samples)
        posterior_sampled_df = posterior_df.iloc[choice_indices].reset_index(drop=True)
        duplicate_cols = list(set(uq_df.columns) & set(posterior_sampled_df.columns) - {"sample"})
        if duplicate_cols:
            print(f"WARNING: posterior overrides UQ for columns: {sorted(duplicate_cols)}")
            uq_df = uq_df.drop(columns=duplicate_cols)
        uq_df = pd.concat([uq_df, posterior_sampled_df], axis=1)

    uq_df.rename(columns={"sample": "uq"}).to_csv(output_path / "uq.csv", index=False)

    if uq.mapping:
        uq_df = apply_choice_mapping(uq_df, df, uq.mapping)

    merged_df = df.merge(uq_df, how="cross", suffixes=("_df", "_uq"))
    merged_df["sample"] = merged_df["sample_df"].astype(str) + "_uq_" + merged_df["sample_uq"].astype(int).astype(str)
    merged_df = merged_df.drop(columns=["sample_df", "sample_uq"])
    return merged_df


def _run(*, kind: str) -> None:
    """
    Shared CLI body for ISMIP7 Greenland forward and inverse runs.

    Parses arguments, stages inputs, optionally builds an ensemble, then
    renders one run script per member by calling ``_render_<kind>_run``.
    The two CLI entry points :func:`run_forward` and :func:`run_inverse`
    are one-line wrappers around this function.

    Parameters
    ----------
    kind : {"forward", "inverse"}
        Which run script template to render. Selects the per-row worker
        and decides whether to forward ``inverse.file`` to PISM.
    """
    if kind not in ("forward", "inverse"):
        raise ValueError(f"kind must be 'forward' or 'inverse', got {kind!r}")
    render = _render_forward_run if kind == "forward" else _render_inverse_run

    parser = _build_cli_parser(
        description=f"Stage ISMIP7 Greenland and render a {kind} run script (ensemble if UQ_FILE is given).",
        supports_execute=False,
    )
    options = parser.parse_args()
    force_overwrite = options.force_overwrite

    path = Path(options.output_path)
    path.mkdir(parents=True, exist_ok=True)
    input_path = path / Path("input")
    input_path.mkdir(parents=True, exist_ok=True)
    output_path = path / Path("output")
    output_path.mkdir(parents=True, exist_ok=True)

    config_file = options.CONFIG_FILE
    template_file = options.TEMPLATE_FILE
    uq_file = options.UQ_FILE
    pism_config_cdl = options.pism_config_cdl

    cfg = load_config(config_file)
    campaign_config = cfg.campaign.as_params()
    bucket = campaign_config["bucket"]
    prefix = campaign_config["prefix"]

    df = stage(campaign_config, bucket=bucket, prefix=prefix, path=path, force_overwrite=force_overwrite)

    if uq_file is not None:
        rows_df = _build_ensemble_df(df, uq_file, output_path, options.posterior_file)
        header = f"Generate Ensemble {kind.capitalize()} Runs for ISMIP7 Greenland"
    else:
        rows_df = df
        header = f"Generate {kind.capitalize()} Run for ISMIP7 Greenland"
    is_ensemble = uq_file is not None

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    print("=" * 120)
    print(banner)
    print("=" * 120)
    print(header)
    print("-" * 120)

    config_cli = {
        "resolution": options.resolution,
        "nodes": options.nodes,
        "ntasks": options.ntasks,
        "tasks": options.tasks,
        "queue": options.queue,
        "walltime": options.walltime,
        "stress_balance": options.stress_balance,
        "start": options.start,
        "end": options.end,
    }

    for idx, row in rows_df.iterrows():
        if is_ensemble:
            # Drop the staged columns and the composite sample id; whatever
            # remains is a row of UQ overrides to forward to PISM.
            uq_overrides = row.drop(labels=list(df.columns) + ["sample"]).to_dict()
        else:
            uq_overrides = {}

        # File paths from the staging table override UQ-supplied paths for the
        # same flag (matches the glacier behavior).
        uq_overrides.update(
            {
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
        )
        # Wire the inverse observation file only when the stage produced one
        # (campaign config can opt in via an ``obs_file`` key); otherwise
        # rely on whatever ``inverse.file`` the UQ supplied.
        if kind == "inverse" and "obs_file" in row and pd.notna(row["obs_file"]):
            uq_overrides["inverse.file"] = row["obs_file"]

        outline_file = row["outline_file"] if "outline_file" in row else None
        sample = row["sample"] if is_ensemble else (int(row["sample"]) if "sample" in row else idx)
        render(
            config_file,
            template_file,
            outline_file,
            path=path,
            config_cli=config_cli,
            debug=options.debug,
            uq=uq_overrides,
            sample=sample,
            pism_config_cdl=pism_config_cdl,
        )


def run_forward() -> None:
    """
    CLI entry point for ISMIP7 Greenland forward runs (single or ensemble).

    Behaves as a single run when no ``UQ_FILE`` positional is supplied, and
    as a UQ ensemble when one is. The argument schema and output layout are
    otherwise identical.
    """
    _run(kind="forward")


def run_inverse() -> None:
    """
    CLI entry point for ISMIP7 Greenland inverse runs (single or ensemble).

    Behaves as a single inverse run when no ``UQ_FILE`` positional is
    supplied, and as a UQ ensemble when one is. When the staged row
    includes an ``obs_file`` column (set by the campaign config), it is
    wired through as ``inverse.file``; otherwise the user can pass
    ``inverse.file`` via the UQ TOML.
    """
    _run(kind="inverse")


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    run_forward()
