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

from pism_terra.aws import local_to_s3
from pism_terra.config import JobConfig, load_config, load_uq
from pism_terra.download import file_localizer
from pism_terra.glacier.climate import create_offset_file
from pism_terra.glacier.execute import find_first_and_execute
from pism_terra.glacier.stage import stage_glacier
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


def _render_inverse_run(
    rgi_id: str,
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
    Configure and generate a PISM inverse job script for a single glacier (ensemble-ready).

    Reads a TOML configuration, merges optional ensemble overrides (``uq``),
    renders a submission script from a Jinja2 template, and writes both the
    script and a companion TOML describing the resolved run parameters.
    Also emits a command-line string of PISM flags derived from the config and
    overrides.

    Parameters
    ----------
    rgi_id : str
        Glacier identifier (e.g., ``"RGI2000-v7.0-C-01-04374"``). Used to build
        output directory and filenames.
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

    >>> run_forward(
    ...     rgi_id="RGI2000-v7.0-C-01-04374",
    ...     config_file="config/init_stampede3.toml",
    ...     template_file="templates/stampede3.j2",
    ...     path="result",
    ... )

    Ensemble member with overrides from a pandas row (e.g., Latin Hypercube):

    >>> row = df_samples.loc[17]  # contains dotted keys + 'sample'
    >>> run_forward(
    ...     rgi_id="RGI2000-v7.0-C-01-04374",
    ...     config_file="config/init_stampede3.toml",
    ...     template_file="templates/stampede3.j2",
    ...     uq=row,             # dotted PISM flags to override
    ...     sample=None,        # will be inferred from row['sample'] if present
    ...     ntasks=112,         # optional template/run override
    ... )
    """

    outline_file = str(Path(outline_file).resolve()) if (outline_file is not None) else "none"
    # Derive the complex (-C) and glacier (-G) outline paths (co-located in the
    # staging dir) so postprocessing can clip to either. See stage.py.
    if outline_file != "none":
        _outline_dir = Path(outline_file).parent
        outline_c_file = str((_outline_dir / f"rgi_{rgi_id}-C.gpkg").resolve())
        outline_g_file = str((_outline_dir / f"rgi_{rgi_id}-G.gpkg").resolve())
    else:
        outline_c_file = outline_g_file = "none"
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
    glacier_path = path / Path(rgi_id)
    glacier_path.mkdir(parents=True, exist_ok=True)
    log_path = glacier_path / Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    output_path = glacier_path / Path("output")
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
    run.update(cfg.stress_balance.selected())
    run.update(cfg.atmosphere.selected())
    run.update(cfg.ocean.selected())
    run.update(cfg.surface.selected())
    run.update(cfg.energy.selected())
    run.update(cfg.grid.as_params())
    run.update(cfg.run_info.as_params())
    run.update(cfg.time.as_params())
    # Forward solver knobs ([solver.forward]) drive the forward pism call.
    run.update(cfg.solver.get("forward", {}))

    inv = {}
    inv.update(getattr(cfg, "iceflow"))
    inv.update(getattr(cfg, "inverse"))
    # Inverse solver knobs ([solver.inverse]) drive the pismi call.
    inv.update(cfg.solver.get("inverse", {}))

    # cfg.stress_balance.selected() carries everything the forward run needs
    # (model options + PETSc solver knobs like bp_* / inv_adj_*). The pismi
    # call only needs the ``stress_balance.*`` dotted options; the solver
    # flags are picked up by the prior pism call (and inherited from the
    # state file). Filter so inv_str stays minimal.
    inv.update({k: v for k, v in cfg.stress_balance.selected().items() if k.startswith("stress_balance.")})

    # The energy block defines the ice rheology (``energy.model`` and the
    # ``flow_law``/ice-softness settings), which the Blatter forward and adjoint
    # solves in pismi depend on. It is pure physics (no PETSc solver knobs), so
    # forward it whole; without it the inverse run silently uses PISM's default
    # rheology instead of the configured one.
    inv.update(cfg.energy.selected())

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

    # Remove 'sample' from flag overrides; drop any key not in either the
    # ``run`` or ``inv`` dicts (e.g., surface.debm_simple.std_dev.file when
    # surface.model == "pdd"). ``inverse.*`` keys live in ``inv`` only, so
    # filtering against ``run.keys()`` alone would silently drop them — that
    # was the bug that kept ``inverse.file`` stuck at "none".
    all_overrides = {k: v for k, v in uq_clean.items() if k != "sample"}
    run_overrides, _ = filter_overrides_by_config(all_overrides, run.keys())
    inv_overrides, _ = filter_overrides_by_config(all_overrides, inv.keys())
    skipped = [k for k in all_overrides if k not in run and k not in inv]
    if skipped:
        print(f"Skipping uq overrides not in config: {skipped}")
    # Apply to both runtime dicts (these should be dotted PISM flags)
    run.update(run_overrides)
    inv.update(inv_overrides)

    scalar_file = scalar_path / Path(f"scalar_g{resolution}_{rgi_id}_{name_options}_{start}_{end}.nc")
    spatial_file = spatial_path / Path(f"spatial_g{resolution}_{rgi_id}_{name_options}_{start}_{end}.nc")
    state_file = state_path / Path(f"state_g{resolution}_{rgi_id}_{name_options}_{start}_{end}.nc")
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

    inv_file = inv_path / Path(f"inv_g{resolution}_{rgi_id}_{name_options}_{start}_{end}.nc")
    # Feed the forward run's state file into pismi as its input.
    inv.update({"input.file": state_file.resolve()})
    # inverse output file
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
        "rgi": {"rgi_id": rgi_id, "outline_c": outline_c_file, "outline_g": outline_g_file},
        "output": {
            "spatial": str(spatial_file.resolve()),
            "scalar.file": scalar_file.resolve(),
            "state": str(state_file.resolve()),
        },
        "config": run,
    }
    post_path = output_path / Path("post_processing")
    post_path.mkdir(parents=True, exist_ok=True)

    post_file = post_path / Path(f"g{resolution}_{rgi_id}_{name_options}_{start}_{end}.toml")
    with open(post_file, "w", encoding="utf-8") as toml_file:
        toml.dump(run_toml, toml_file)

    params.update({"run_str": run_str})
    params.update({"inv_str": inv_str})
    rendered_script = "" if debug else template.render(params)

    run_script_path = glacier_path / Path("run_scripts")
    run_script_path.mkdir(parents=True, exist_ok=True)

    run_script = run_script_path / Path(f"submit_g{resolution}_{rgi_id}_{name_options}_{start}_{end}.sh")

    # Save or print the output
    run_script.write_text(rendered_script)

    print(f"\nSLURM script written to {run_script.resolve()}\n")
    print(f"Postprocessing script written to {post_file.resolve()}\n")


def _render_forward_run(
    rgi_id: str,
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
    rgi_id : str
        Glacier identifier (e.g., ``"RGI2000-v7.0-C-01-04374"``). Used to build
        output directory and filenames.
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

    >>> run_forward(
    ...     rgi_id="RGI2000-v7.0-C-01-04374",
    ...     config_file="config/init_stampede3.toml",
    ...     template_file="templates/stampede3.j2",
    ...     path="result",
    ... )

    Ensemble member with overrides from a pandas row (e.g., Latin Hypercube):

    >>> row = df_samples.loc[17]  # contains dotted keys + 'sample'
    >>> run_forward(
    ...     rgi_id="RGI2000-v7.0-C-01-04374",
    ...     config_file="config/init_stampede3.toml",
    ...     template_file="templates/stampede3.j2",
    ...     uq=row,             # dotted PISM flags to override
    ...     sample=None,        # will be inferred from row['sample'] if present
    ...     ntasks=112,         # optional template/run override
    ... )
    """

    outline_file = str(Path(outline_file).resolve()) if (outline_file is not None) else "none"
    # Derive the complex (-C) and glacier (-G) outline paths (co-located in the
    # staging dir) so postprocessing can clip to either. See stage.py.
    if outline_file != "none":
        _outline_dir = Path(outline_file).parent
        outline_c_file = str((_outline_dir / f"rgi_{rgi_id}-C.gpkg").resolve())
        outline_g_file = str((_outline_dir / f"rgi_{rgi_id}-G.gpkg").resolve())
    else:
        outline_c_file = outline_g_file = "none"
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
    glacier_path = path / Path(rgi_id)
    glacier_path.mkdir(parents=True, exist_ok=True)
    log_path = glacier_path / Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    output_path = glacier_path / Path("output")
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
        "inverse",
        "time_stepping",
    ):
        run.update(getattr(cfg, section))
    run.update(cfg.stress_balance.selected())
    run.update(cfg.atmosphere.selected())
    run.update(cfg.ocean.selected())
    run.update(cfg.surface.selected())
    run.update(cfg.energy.selected())
    run.update(cfg.grid.as_params())
    run.update(cfg.run_info.as_params())
    run.update(cfg.time.as_params())
    # Forward solver knobs ([solver.forward]) drive the forward pism call.
    run.update(cfg.solver.get("forward", {}))

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

    scalar_file = scalar_path / Path(f"scalar_g{resolution}_{rgi_id}_{name_options}_{start}_{end}.nc")
    spatial_file = spatial_path / Path(f"spatial_g{resolution}_{rgi_id}_{name_options}_{start}_{end}.nc")
    state_file = state_path / Path(f"state_g{resolution}_{rgi_id}_{name_options}_{start}_{end}.nc")
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
        "rgi": {"rgi_id": rgi_id, "outline_c": outline_c_file, "outline_g": outline_g_file},
        "output": {
            "spatial": str(spatial_file.resolve()),
            "scalar.file": scalar_file.resolve(),
            "state": str(state_file.resolve()),
        },
        "config": run,
    }
    post_path = output_path / Path("post_processing")
    post_path.mkdir(parents=True, exist_ok=True)

    post_file = post_path / Path(f"g{resolution}_{rgi_id}_{name_options}_{start}_{end}.toml")
    with open(post_file, "w", encoding="utf-8") as toml_file:
        toml.dump(run_toml, toml_file)

    params.update({"run_str": run_str})
    rendered_script = "" if debug else template.render(params)

    run_script_path = glacier_path / Path("run_scripts")
    run_script_path.mkdir(parents=True, exist_ok=True)

    run_script = run_script_path / Path(f"submit_g{resolution}_{rgi_id}_{name_options}_{start}_{end}.sh")

    # Save or print the output
    run_script.write_text(rendered_script)

    print(f"\nSLURM script written to {run_script.resolve()}\n")
    print(f"Postprocessing script written to {post_file.resolve()}\n")


def _nullable_string(argument_string: str) -> str | None:
    """
    Handle null/None CLI parameters from HyP3.

    There's no way in AWS batch to selectively include parameters, so HyP3 needs
    special handling for optional (nullable) API parameters. In the docker run command,
    null arguments will appear as the string `None`. This argparse type ensures all `None`
    strings are accurately represented as `None` objects in Python.

    Parameters
    ----------
    argument_string : str
        Argument string to parse.

    Returns
    -------
    str | None
        The parsed argument string.
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
        Whether to add the ``--execute`` flag. Forward runs accept it (the
        first generated script is launched in-process); inverse runs may
        also accept it.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = description
    parser.add_argument("--bucket", help="AWS S3 Bucket to upload output files to")
    parser.add_argument(
        "--bucket-prefix",
        help="AWS prefix (location in bucket) to add to product files",
        default="",
    )
    parser.add_argument(
        "--output-path",
        help="Base path to save all files to. Files will be saved in `f'{out_path}/{RGI_ID}/output/'`.",
        type=str,
        default=".",
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
            help="Execute the first generated run script in-process. Ignored in ensemble mode or with --debug.",
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
    parser.add_argument("RGI_ID", help="RGI ID.")
    parser.add_argument("CONFIG_FILE", help="CONFIG TOML.")
    parser.add_argument("TEMPLATE_FILE", help="TEMPLATE J2.")
    parser.add_argument(
        "UQ_FILE",
        nargs="?",
        default=None,
        type=_nullable_string,
        help="UQ TOML (optional). Supply to render an ensemble; omit for a single-glacier run.",
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
    cross-joins with the staged glacier DataFrame ``df`` and assigns a
    composite ``sample`` ID per row.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`pism_terra.glacier.stage.stage_glacier` (one row
        per staged ``boot``/``grid``/``climate`` tuple).
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
    Shared CLI body for forward and inverse runs.

    Parses arguments, stages inputs, optionally builds an ensemble, then
    renders one run script per member by calling ``_render_<kind>_run``.
    The two CLI entry points :func:`run_forward` and :func:`run_inverse` are
    one-line wrappers around this function.

    Parameters
    ----------
    kind : {"forward", "inverse"}
        Which run script template to render. Selects the per-row worker
        and decides whether to include ``inverse.file`` in the UQ dict.
    """
    if kind not in ("forward", "inverse"):
        raise ValueError(f"kind must be 'forward' or 'inverse', got {kind!r}")
    render = _render_forward_run if kind == "forward" else _render_inverse_run

    parser = _build_cli_parser(
        description=f"Stage RGI Glacier and render a {kind} run script (ensemble if UQ_FILE is given).",
        supports_execute=True,
    )
    options = parser.parse_args()
    force_overwrite = options.force_overwrite

    path = Path(options.output_path)
    rgi_id = options.RGI_ID
    glacier_path = path / rgi_id

    input_path = glacier_path / "input"
    input_path.mkdir(parents=True, exist_ok=True)
    staging_path = glacier_path / "staging"
    staging_path.mkdir(parents=True, exist_ok=True)
    output_path = glacier_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    uq_path = output_path / "uq"
    uq_path.mkdir(parents=True, exist_ok=True)

    config_file = file_localizer(options.CONFIG_FILE, path / "config")
    pism_config_cdl = file_localizer(options.pism_config_cdl, path / "config") if options.pism_config_cdl else None
    template_file = file_localizer(options.TEMPLATE_FILE, path / "templates")
    uq_file = file_localizer(options.UQ_FILE, path / "uq") if options.UQ_FILE else None

    start_cli = options.start
    end_cli = options.end

    cfg = load_config(config_file)
    # ``years`` is derived from the *effective* run span: CLI overrides win
    # over the config's [time] section. Local Timestamps stay distinct from
    # the CLI string overrides (``start_cli`` / ``end_cli``) below.
    start_ts = pd.Timestamp(start_cli or cfg.time.time_start)
    end_ts = pd.Timestamp(end_cli or cfg.time.time_end)
    last_year = end_ts.year - 1 if (end_ts.month == 1 and end_ts.day == 1) else end_ts.year
    years = list(range(start_ts.year, last_year + 1))
    campaign_config = cfg.campaign.as_params()
    campaign_config["years"] = years

    df = stage_glacier(
        campaign_config,
        rgi_id,
        path=input_path,
        staging_path=staging_path,
        force_overwrite=force_overwrite,
    )

    if uq_file is not None:
        rows_df = _build_ensemble_df(df, uq_file, output_path, options.posterior_file)
        header = f"Generate Ensemble Runs for Glacier {rgi_id}"
    else:
        rows_df = df
        header = f"Generate Run for Glacier {rgi_id}"
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
        "start": start_cli,
        "end": end_cli,
    }

    for idx, row in rows_df.iterrows():
        delta_T = row["atmosphere.delta_T"] if "atmosphere.delta_T" in row else 0
        frac_P = row["atmosphere.frac_P"] if "atmosphere.frac_P" in row else 0
        scalar_offset_file = input_path / Path(f"scalar_offset_{rgi_id}_id_{idx}.nc")
        create_offset_file(scalar_offset_file, delta_T=delta_T, frac_P=frac_P)

        if is_ensemble:
            # Drop the staged-glacier columns and the composite sample id;
            # whatever remains is a row of UQ overrides to forward to PISM.
            uq_overrides = row.drop(labels=list(df.columns) + ["sample"]).to_dict()
        else:
            uq_overrides = {}

        uq_overrides.update(
            {
                "input.file": row["boot_file"],
                "grid.file": row["grid_file"],
                "atmosphere.delta_T.file": scalar_offset_file,
                "atmosphere.elevation_change.file": row["boot_file"],
                "atmosphere.precip_scaling.file": scalar_offset_file,
                "atmosphere.given.file": row["climate_file"],
                "energy.bedrock_thermal.file": row["heatflux_file"],
                "surface.debm_simple.albedo_input.file": row["climate_file"],
                "surface.debm_simple.std_dev.file": row["climate_file"],
                "surface.force_to_thickness.file": row["boot_file"],
                "surface.pdd.std_dev.file": row["climate_file"],
            }
        )
        if kind == "inverse":
            uq_overrides["inverse.file"] = row["obs_file"]

        outline_file = row["outline_file"] if "outline_file" in row else None
        sample = row["sample"] if is_ensemble else (int(row["sample"]) if "sample" in row else idx)
        render(
            rgi_id,
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

    uq_out_file = uq_path / "uq.csv"
    uq_overrides.to_csv()

    # ``--execute`` only fires the *first* generated script. Use it for the
    # single-glacier mode; ignore it for ensembles (would only run member 0).
    if not is_ensemble and getattr(options, "execute", False) and not options.debug:
        find_first_and_execute(path / rgi_id)

    if options.bucket:
        prefix = f"{options.bucket_prefix}/{rgi_id}" if options.bucket_prefix else rgi_id
        local_to_s3(glacier_path, bucket=options.bucket, prefix=prefix)


def run_forward() -> None:
    """
    CLI entry point for forward runs (single or ensemble).

    Behaves as a single-glacier run when no ``UQ_FILE`` positional is
    supplied, and as a UQ ensemble when one is. The argument schema and
    output layout are otherwise identical.
    """
    _run(kind="forward")


def run_inverse() -> None:
    """
    CLI entry point for inverse runs (single or ensemble).

    Behaves as a single-glacier inverse run when no ``UQ_FILE`` positional
    is supplied, and as a UQ ensemble when one is. The per-row UQ dict
    additionally maps ``inverse.file`` to the staged observation file.
    """
    _run(kind="inverse")


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    run_forward()
