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

# pylint: disable=too-many-positional-arguments

"""
Running Greenland paleo simulations.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from pyfiglet import Figlet

from pism_terra.aws import local_to_s3
from pism_terra.config import JobConfig, load_config
from pism_terra.download import file_localizer
from pism_terra.glacier.run import snapshot_project_file
from pism_terra.greenland.paleo.stage import stage
from pism_terra.ismip7.greenland.run import (
    _base_run_dict,
    _build_cli_parser,
    _build_ensemble_df,
    _make_output_paths,
    run_directories,
)
from pism_terra.workflow import (
    add_profile_option,
    add_provenance,
    check_template_legs,
    dict2str,
    filter_overrides_by_config,
    normalize_row,
    sort_dict_by_key,
    validate_pism_options,
)


def staged_file_flags(row: Mapping[str, object]) -> dict[str, object]:
    """
    Map one row of the staging table to the PISM flags that take its files.

    Parameters
    ----------
    row : Mapping
        Row of :func:`pism_terra.greenland.paleo.stage.stage`.

    Returns
    -------
    dict
        Dotted PISM flags. A flag only reaches the run when the config's
        selected option tables declare it.
    """
    return {
        "input.file": row["boot_file"],
        "input.regrid.file": row["regrid_file"],
        "grid.file": row["grid_file"],
        "energy.bedrock_thermal.file": row["heatflux_file"],
        "atmosphere.given.file": row["climate_file"],
        "atmosphere.delta_T.file": row["delta_T_file"],
        "atmosphere.precip_scaling.file": row["delta_T_file"],
        # The base climate is valid at the present-day surface.
        "atmosphere.elevation_change.file": row["boot_file"],
        "ocean.th.file": row["ocean_file"],
        "ocean.delta_T.file": row["ocean_delta_T_file"],
        "ocean.delta_sl.file": row["delta_SL_file"],
    }


def snapshot_times_within(times: str, start: str | None, end: str | None) -> str:
    """
    Keep the snapshot times that fall inside a run.

    Parameters
    ----------
    times : str
        ``output.snapshot.times``: a comma-separated list of years.
    start : str or None
        ``time.start`` of the run, in years.
    end : str or None
        ``time.end`` of the run, in years.

    Returns
    -------
    str
        The list without the times outside ``[start, end]``; ``times`` itself
        when it or the bounds are not plain years (a range, dates).
    """
    try:
        first, last = float(str(start)), float(str(end))
        years = [(float(t), t.strip()) for t in times.split(",") if t.strip()]
    except ValueError:
        return times
    return ",".join(text for year, text in years if first <= year <= last)


def run_paleo(
    config_file: str | Path,
    template_file: Path | str,
    path: str | Path = "result",
    config_cli: dict | None = None,
    debug: bool = False,
    *,
    uq: Mapping[str, object] | pd.Series | None = None,
    sample: int | str | None = None,
    pism_config_cdl: str | Path | None = None,
) -> Path:
    """
    Render the job script of one glacial-cycle run.

    A paleo run is a single PISM invocation from ``time.start`` to
    ``time.end`` that bootstraps from the boot file and regrids the thermal
    state from the initial-state file.

    Parameters
    ----------
    config_file : str or pathlib.Path
        Run configuration TOML.
    template_file : str or pathlib.Path
        Jinja2 submission template of the ISMIP7 family; the run fills its
        ``run_hist_str`` slot.
    path : str or pathlib.Path, optional
        Base output directory.
    config_cli : dict or None, optional
        CLI overrides: ``"resolution"``, ``"nodes"``, ``"ntasks"``,
        ``"tasks"``, ``"queue"``, ``"walltime"``, ``"stress_balance"``,
        ``"start"`` and ``"end"`` (years, negative before present).
    debug : bool, optional
        Skip rendering the template and write an empty script.
    uq : Mapping or pandas.Series or None, optional
        Dotted PISM flags overriding the config, e.g. one ensemble member
        and the staged file paths.
    sample : int or str or None, optional
        Member identifier used in the file names.
    pism_config_cdl : str or Path or None, optional
        PISM CDL master config to validate the flags against.

    Returns
    -------
    pathlib.Path
        The job script.
    """
    cfg = load_config(config_file)
    config_cli = config_cli or {}

    resolution = config_cli.get("resolution")
    if resolution:
        cfg.grid.resolution = re.sub(r"\s+", "", resolution)
        cfg.grid.dx = None
        cfg.grid.dy = None
    resolution = cfg.grid.resolution
    if config_cli.get("stress_balance"):
        cfg.stress_balance.model = config_cli["stress_balance"]
    if config_cli.get("start") is not None:
        cfg.time.time_start = str(config_cli["start"])
    if config_cli.get("end") is not None:
        cfg.time.time_end = str(config_cli["end"])
    start, end = cfg.time.time_start, cfg.time.time_end

    uq_clean = normalize_row(uq) if uq is not None else {}
    if sample is None:
        sample = uq_clean.get("sample")
    # An ensemble row may pick a model; the option table has to follow it.
    cfg.select_models(uq_clean)

    paths = _make_output_paths(path)
    run = _base_run_dict(cfg)

    overrides = {k: v for k, v in uq_clean.items() if k != "sample" and not k.endswith(".model")}
    overrides, skipped = filter_overrides_by_config(overrides, run.keys())
    if skipped:
        print(f"Skipping overrides not in config: {skipped}")
    run.update(overrides)

    if sample is None:
        name_options = (
            f"surface_{cfg.surface.model}_energy_{cfg.energy.model}_stress_balance_{cfg.stress_balance.model}"
        )
    else:
        name_options = f"id_{sample}"
    stem = f"g{resolution}_{name_options}_{start}_{end}"
    run.update(
        {
            "output.file": (paths["state"] / f"state_{stem}.nc").resolve(),
            "output.spatial.file": (paths["spatial"] / f"spatial_{stem}.nc").resolve(),
            "output.scalar.file": (paths["scalar"] / f"scalar_{stem}.nc").resolve(),
        }
    )
    # PISM aborts when no snapshot time falls inside the run.
    snapshot_times = snapshot_times_within(str(run.pop("output.snapshot.times", "")), start, end)
    if snapshot_times:
        snapshot_path = paths["output"] / "snapshot"
        snapshot_path.mkdir(parents=True, exist_ok=True)
        run["output.snapshot.times"] = snapshot_times
        run["output.snapshot.file"] = (snapshot_path / f"snapshot_{stem}").resolve()
    else:
        run.pop("output.snapshot.size", None)
    add_profile_option(run, cfg.campaign.profile)

    if pism_config_cdl is not None:
        validate_pism_options(run, pism_config_cdl)

    run_str = dict2str(sort_dict_by_key(run))

    params = JobConfig(**cfg.job.model_dump()).model_dump(exclude_none=True, by_alias=True)
    job_kwargs = {
        k: v
        for k, v in {
            "nodes": config_cli.get("nodes"),
            "ntasks": config_cli.get("ntasks"),
            "queue": config_cli.get("queue"),
            "output_path": paths["log"].resolve(),
            "tasks": config_cli.get("tasks"),
            "walltime": config_cli.get("walltime"),
        }.items()
        if v is not None
    }
    if job_kwargs:
        params.update(JobConfig(**job_kwargs).as_params())
    # The ISMIP7 templates split a run across named legs; the glacial cycle
    # is one leg and takes the historical slot.
    params.update(
        {
            "run_init_str": "",
            "inv_str": "",
            "run_hist_str": run_str,
            "run_proj_str": "",
            "post_process_str": "",
            "post_scalar_str": "",
            "ism_checker_str": "",
            "output_dirs": run_directories(paths["output"]),
        }
    )

    template_file = Path(template_file)
    check_template_legs(template_file, params)
    template = Environment(loader=FileSystemLoader(template_file.parent)).get_template(template_file.name)
    rendered_script = "" if debug else add_provenance(template.render(params))

    run_script_path = Path(path) / Path("run_scripts")
    run_script_path.mkdir(parents=True, exist_ok=True)
    run_script = run_script_path / Path(f"submit_{stem}.sh")
    run_script.write_text(rendered_script)
    print(f"\nJob script written to {run_script.resolve()}\n")
    return run_script


def main() -> None:
    """
    CLI entry point for Greenland paleo runs (single or ensemble).
    """
    parser = _build_cli_parser(
        description="Stage Greenland paleo inputs and render a glacial-cycle run script (ensemble if UQ_FILE is given).",
        supports_execute=False,
    )
    options = parser.parse_args()

    path = Path(options.output_path)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / Path("output")
    output_path.mkdir(parents=True, exist_ok=True)

    config_path, template_path, uq_path = path / "config", path / "templates", path / "uq"
    config_file = snapshot_project_file(file_localizer(options.CONFIG_FILE, config_path), config_path)
    pism_config_cdl = (
        snapshot_project_file(file_localizer(options.pism_config_cdl, config_path), config_path)
        if options.pism_config_cdl
        else None
    )
    template_file = snapshot_project_file(file_localizer(options.TEMPLATE_FILE, template_path), template_path)
    uq_file = snapshot_project_file(file_localizer(options.UQ_FILE, uq_path), uq_path) if options.UQ_FILE else None

    cfg = load_config(config_file)
    if options.dataset_version is not None:
        cfg.campaign.version = options.dataset_version

    df = stage(
        cfg.campaign.as_params(),
        path=path,
        force_overwrite=options.force_overwrite,
        data_path=options.data_path,
    )

    if uq_file is not None:
        rows_df = _build_ensemble_df(df, uq_file, output_path, options.posterior_file, samples=options.samples)
        header = "Generate Ensemble Runs for Greenland paleo"
    else:
        rows_df = df
        header = "Generate Run for Greenland paleo"

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

    for _, row in rows_df.iterrows():
        # Whatever is not a staged column is a sampled parameter; the staged
        # paths override a path the UQ file may carry for the same flag.
        overrides = row.drop(labels=list(df.columns)).to_dict() if uq_file is not None else {}
        overrides.update(staged_file_flags(row))
        run_paleo(
            config_file,
            template_file,
            path=path,
            config_cli=config_cli,
            debug=options.debug,
            uq=overrides,
            sample=row["sample"] if uq_file is not None else None,
            pism_config_cdl=pism_config_cdl,
        )

    if options.bucket:
        local_to_s3(path, bucket=options.bucket, prefix=options.bucket_prefix)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
