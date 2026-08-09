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
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pyfiglet import Figlet

from pism_terra.config import JobConfig, load_config, load_uq
from pism_terra.ismip7.experiments import resolve_counter
from pism_terra.ismip7.greenland.stage import stage
from pism_terra.ismip7.naming import ISMIP7Names, member_ids
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

# Upper bound on Dask workers for the per-basin post-processing step. The run's
# ``ntasks`` sizes the PISM *MPI* decomposition (40+ on Chinook); handing that
# straight to Dask spawns one worker process per task, and each nanny needs
# several file descriptors, so a large run dies with
# ``OSError: [Errno 24] Too many open files`` before any work starts. The
# post-processing is a per-basin clip + field sum, so a handful of workers is
# plenty regardless of how wide the PISM run was.
_POSTPROCESS_MAX_WORKERS = 8


def _postprocess_ntasks(config_cli: dict) -> str:
    """
    Build the ``--ntasks`` flag for the post-processing command.

    Clamps the run's MPI task count to :data:`_POSTPROCESS_MAX_WORKERS` so a
    wide PISM decomposition does not translate into an unusable number of Dask
    worker processes.

    Parameters
    ----------
    config_cli : dict
        CLI overrides; only ``"ntasks"`` is consulted.

    Returns
    -------
    str
        ``" --ntasks N"`` when a task count is set, else an empty string.
    """
    ntasks = config_cli.get("ntasks")
    if not ntasks:
        return ""
    return f" --ntasks {min(int(ntasks), _POSTPROCESS_MAX_WORKERS)}"


def _make_output_paths(path: str | Path, *, inverse: bool = False) -> dict[str, Path]:
    """
    Create the run's output directory tree and return the paths.

    Parameters
    ----------
    path : str or pathlib.Path
        Base output directory.
    inverse : bool, optional
        Also create the ``output/inverse`` subdirectory used for pismi
        products. Default is ``False``.

    Returns
    -------
    dict of str to pathlib.Path
        Keys ``"log"``, ``"output"``, ``"scalar"``, ``"spatial"``,
        ``"state"`` and (when ``inverse``) ``"inverse"``.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / Path("output")
    paths = {
        "log": path / Path("logs"),
        "output": output_path,
        "scalar": output_path / Path("scalar"),
        "spatial": output_path / Path("spatial"),
        "state": output_path / Path("state"),
    }
    if inverse:
        paths["inverse"] = output_path / Path("inverse")
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _base_run_dict(cfg, *, bed_deformation: bool = True) -> dict:
    """
    Merge the config sections shared by every forward PISM invocation.

    Parameters
    ----------
    cfg : PismConfig
        Loaded configuration.
    bed_deformation : bool, optional
        Include ``cfg.bed_deformation.selected()``. The inverse init/prior
        leg historically omits it; forward legs include it. Default is
        ``True``.

    Returns
    -------
    dict
        Dotted PISM flags for one forward run, including the
        ``[solver.forward]`` PETSc/blatter knobs.
    """
    run: dict = {}
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
    if bed_deformation:
        run.update(cfg.bed_deformation.selected())
    run.update(cfg.energy.selected())
    run.update(cfg.ocean.selected())
    run.update(cfg.frontal_melt.selected())
    run.update(cfg.grid.as_params())
    run.update(cfg.hydrology.selected())
    run.update(cfg.run_info.as_params())
    run.update(cfg.surface.selected())
    run.update(cfg.stress_balance.selected())
    run.update(cfg.time.as_params())
    # PETSc / blatter solver knobs from [solver.forward]. Matches what the
    # inverse runner does for the prior pism call; see also pism_terra.glacier.run.
    run.update(cfg.solver.get("forward", {}))
    return run


def _build_init_leg(
    cfg,
    *,
    init_start: str,
    init_end: str,
    resolution: str,
    name_options_init: str,
    overrides: Mapping[str, object],
    state_path: Path,
    scalar_path: Path,
    spatial_path: Path,
    pism_config_cdl: str | Path | None,
) -> tuple[str, Path, str]:
    """
    Build the short init/prior leg shared by the forward and inverse chains.

    A bootstrap run (staged regrid of ``litho_temp,enthalpy,age,tillwat``)
    spanning ``campaign.init_start``..``campaign.init_end`` with flat output
    names carrying the init range. The follow-on legs restart from the
    returned state file.

    Parameters
    ----------
    cfg : PismConfig
        Loaded configuration (CLI mutations such as ``--stress-balance``
        must already be applied).
    init_start, init_end : str
        Time bounds of the init leg as ``YYYY-MM-DD``.
    resolution : str
        Grid resolution tag (e.g. ``"900m"``), for filenames.
    name_options_init : str
        Filename stem chunk for the init products (``id_<sample>`` or the
        descriptive fallback; no experiment id — the init leg is never an
        ISMIP7 product).
    overrides : Mapping[str, object]
        UQ / staged-file overrides (dotted PISM flags). Filtered against
        the init dict's keys before being applied.
    state_path, scalar_path, spatial_path : pathlib.Path
        Output directories (see :func:`_make_output_paths`).
    pism_config_cdl : str or pathlib.Path or None
        Optional PISM CDL master config for option validation.

    Returns
    -------
    tuple of (str, pathlib.Path, str)
        ``(run_init_str, state_init, init_tag)`` — the rendered command
        line, the absolute state-file path the next leg restarts from, and
        the ``g<res>_<opts>_<start>_<end>`` filename tag (used e.g. for the
        inversion output).
    """
    run_init = _base_run_dict(cfg, bed_deformation=False)
    run_init.pop("time.start", None)
    run_init.pop("time.end", None)
    run_init.update({"time.start": init_start, "time.end": init_end})

    init_overrides, _ = filter_overrides_by_config(dict(overrides), run_init.keys())
    run_init.update(init_overrides)

    init_tag = f"g{resolution}_{name_options_init}_{init_start}_{init_end}"
    state_init = state_path / Path(f"state_{init_tag}.nc")
    run_init.update(
        {
            "output.file": state_init.resolve(),
            "output.scalar.file": (scalar_path / Path(f"scalar_{init_tag}.nc")).resolve(),
            "output.spatial.file": (spatial_path / Path(f"spatial_{init_tag}.nc")).resolve(),
        }
    )

    if pism_config_cdl is not None:
        validate_pism_options(run_init, pism_config_cdl)

    return dict2str(sort_dict_by_key(run_init)), state_init, init_tag


def _build_forward_legs(
    cfg,
    run_hist: dict,
    *,
    start: str,
    end: str,
    resolution: str,
    name_options: str,
    sample: int | str | None,
    output_path: Path,
    state_path: Path,
    spatial_path: Path,
    scalar_path: Path,
    outline_file: str,
    config_cli: dict,
    proj_overrides: Mapping[str, object] | None,
    pism_config_cdl: str | Path | None,
) -> dict[str, str]:
    """
    Build the forward leg command line(s) and post-processing strings.

    Owns the single-leg vs counter-driven two-leg (historical ->
    projection) split, ISMIP7 submission naming and product-leg gating,
    and the generation of the checker / scalar-splitter / per-basin
    post-processing commands. Shared by the forward renderer (where
    ``run_hist`` is the bootstrap run) and the inverse renderer (where
    ``run_hist`` restarts from the init leg's state with the inverted
    ``tauc`` regridded in).

    Parameters
    ----------
    cfg : PismConfig
        Loaded configuration (consulted for counter, pathway, run_info
        and campaign fields).
    run_hist : dict
        Fully-populated run dict for the first forward leg. Mutated in
        place: ``time.end``, ``run_info.experiment`` and the ``output.*``
        files are (re)set here.
    start, end : str
        Time bounds (``YYYY-MM-DD``) of the forward span. ``end`` is only
        used verbatim for single-leg runs; a counter-driven historical leg
        always stops at 2015-01-01 and the projection leg runs to ``end``.
    resolution : str
        Grid resolution tag (e.g. ``"900m"``), for filenames.
    name_options : str
        Filename stem chunk (``id_<sample>_<experiment>`` or the
        descriptive fallback).
    sample : int or str or None
        Ensemble member identifier (GCM name for ISMIP7 staging).
    output_path, state_path, spatial_path, scalar_path : pathlib.Path
        Output directories (see :func:`_make_output_paths`).
    outline_file : str
        Resolved basin outline path or the literal ``"none"``.
    config_cli : dict
        CLI overrides; consulted for the post-processing ``--ntasks`` clamp.
    proj_overrides : Mapping or None
        Projection-epoch file overrides applied to the projection leg only.
    pism_config_cdl : str or pathlib.Path or None
        Optional PISM CDL master config for option validation.

    Returns
    -------
    dict of str to str
        Template context: ``run_hist_str``, ``run_proj_str``,
        ``post_process_str``, ``ism_checker_str``, ``post_scalar_str``.
    """
    # A counter drives the full ISMIP7 two-leg protocol (historical -> 2015 ->
    # projection end). Without a counter the config describes a *single* forcing
    # leg named by ``campaign.pathway`` that must span the TOML
    # ``time.start``..``time.end`` verbatim and be emitted into the matching
    # template slot (``run_hist_str`` for a historical pathway, ``run_proj_str``
    # for a projection pathway).
    counter = cfg.run_info.counter
    pathway = (cfg.campaign.pathway or "").strip().lower()
    single_leg = not counter
    single_is_historical = pathway in ("", "historical")

    run_hist.pop("time.end", None)
    # Single-leg runs respect the config end; the counter-driven historical leg
    # always stops at the ISMIP7 historical/projection split (2015-01-01).
    run_hist.update({"time.end": end if single_leg else "2015-01-01"})
    # Match InfoConfig._quote()'s output shape so both hist and proj write
    # ``run_info.experiment`` the same way (see run_proj below).
    hist_experiment = pathway if (single_leg and not single_is_historical) else "historical"
    run_hist.update({"run_info.experiment": f'"{hist_experiment}"'})

    # ISMIP7 submission naming (conventions doc section 8): when output.ISMIP is
    # set, write the spatial/scalar outputs into the
    # <domain>/<source>/<ism>/<set>/<set_counter>/ tree with conforming names.
    # PISM expands the {var} placeholder, so the spatial output is already one
    # conforming file per variable. The scalar time series stays a single file
    # (its per-variable split is deferred to post-processing). The state/restart
    # file is not an ISMIP7 product, so it stays in state/.
    #
    # The forward call splits into an ``hist`` PISM invocation and a follow-on
    # ``proj`` invocation; each needs its own conforming filenames because the
    # ``experiment_id`` and ``time_range`` differ. Precompute the parts that
    # only depend on the ensemble member (gcm / ism member / set counter) and
    # then generate the two file triples via ``_output_files``.
    # ISMIP7 Core Experiment counter (e.g. "C003"), if this run is counter-driven.
    # It fixes the ISMIP7 ``set_counter`` and selects which of the two forward legs
    # is the submission product (the other leg gets flat filenames). ``None`` keeps
    # the legacy behavior: both legs use ISMIP7 names when ``output.ISMIP`` is set.
    product_leg: str | None = None
    if counter:
        product_leg = resolve_counter(counter).product_leg

    use_ismip = str(run_hist.get("output.ISMIP", "no")).strip().strip("\"'").lower() in ("yes", "true", "1")
    ismip7_ctx: dict | None = None
    if use_ismip:
        ri = cfg.run_info
        missing = [a for a in ("domain", "group", "ism", "set_id", "experiment") if not getattr(ri, a)]
        if missing:
            raise SystemExit(f"output.ISMIP requires run_info fields: {', '.join(f'run_info.{m}' for m in missing)}")
        gcms = cfg.campaign.as_params().get("gcms") or []
        esm_id = str(sample) if sample is not None else (gcms[0] if gcms else "none")
        member_index = gcms.index(esm_id) if esm_id in gcms else 0
        set_counter, ism_member, forcing_member = member_ids(str(ri.set_id), member_index)
        # A counter-driven run uses its protocol counter as the ISMIP7 set_counter
        # (member_ids still supplies the CORE m001/f001 member ids).
        if counter:
            set_counter = counter
        ismip7_ctx = {
            "domain_id": str(ri.domain),
            "source_id": str(ri.group),
            "ism_id": str(ri.ism),
            "ism_member_id": ism_member,
            "esm_id": esm_id,
            "forcing_member_id": forcing_member,
            "set_id": str(ri.set_id),
            "set_counter": set_counter,
        }

    def _output_files(
        experiment_id: str, start_str: str, end_str: str, *, ismip7: bool
    ) -> tuple[Path, Path, Path, Path]:
        """
        Build the (state, spatial, scalar, basin) file tuple for one PISM invocation.

        Uses ISMIP7-conforming names under ``<domain>/<source>/…`` when
        ``output.ISMIP`` is enabled; falls back to the flat
        ``g<res>_<opts>_<start>_<end>`` layout under the usual ``scalar/`` /
        ``spatial/`` / ``state/`` subdirectories otherwise. The state file
        always stays in ``state/`` (not an ISMIP7 product).

        Parameters
        ----------
        experiment_id : str
            ISMIP7 experiment identifier (e.g. ``"historical"`` or
            ``"ssp370"``). Only used when ``output.ISMIP`` is enabled; feeds
            into both the directory tree and the encoded filename stem.
        start_str : str
            Start of the simulated interval as ``YYYY-MM-DD``. Contributes
            the leading year of the ``time_range`` in the ISMIP7 stem, and
            appears verbatim in the flat-layout fallback filename.
        end_str : str
            End of the simulated interval as ``YYYY-MM-DD``. Contributes
            the trailing year of ``time_range`` (with a ``-1`` correction
            when the timestamp lands exactly on Jan 1 so the range reads
            inclusive on the source-year side).
        ismip7 : bool
            Whether *this* leg is the ISMIP7 submission product. When ``False``
            (or ``output.ISMIP`` is off) the flat ``spatial_``/``scalar_`` layout
            is used even if ``ismip7_ctx`` is populated, so the non-product leg of
            a counter-driven run does not land in the submission tree.

        Returns
        -------
        tuple of pathlib.Path
            ``(state, spatial, scalar, basin)`` — absolute paths for PISM's
            ``output.file``, ``output.spatial.file``, ``output.scalar.file``,
            and the per-basin scalar file written by the post-processing step.
            The ISMIP7-tree ``spatial`` carries a ``{var}`` placeholder that
            PISM fills in per variable (one conforming file per variable); the
            flat ``spatial`` is a single combined file so it can be fed to
            ``pism-ismip7-greenland-postprocess`` as one input.
        """
        tag = f"g{resolution}_{name_options}_{start_str}_{end_str}"
        state = state_path / Path(f"state_{tag}.nc")
        if ismip7_ctx is None or not ismip7:
            spatial = spatial_path / Path(f"spatial_g{resolution}_{name_options}_{start_str}_{end_str}.nc")
            scalar = scalar_path / Path(f"scalar_g{resolution}_{name_options}_{start_str}_{end_str}.nc")
            basin = scalar_path / Path(f"basin_g{resolution}_{name_options}_{start_str}_{end_str}.nc")
            return state, spatial, scalar, basin
        end_ts = pd.Timestamp(end_str)
        last_year = end_ts.year - 1 if (end_ts.month == 1 and end_ts.day == 1) else end_ts.year
        time_range = f"{pd.Timestamp(start_str).year}-{last_year}"
        names = ISMIP7Names(experiment_id=experiment_id, time_range=time_range, **ismip7_ctx)
        ismip7_dir = names.directory(output_path)
        ismip7_dir.mkdir(parents=True, exist_ok=True)
        spatial = ismip7_dir / names.filename("{var}")
        scalar = ismip7_dir / f"scalar_{names.stem()}.nc"
        basin = ismip7_dir / f"basin_{names.stem()}.nc"
        return state, spatial, scalar, basin

    # Which leg is the ISMIP7 submission product: for a counter-driven run only the
    # designated leg gets ISMIP7 names (the other is an internal continuation with
    # flat names); legacy runs (product_leg is None) keep both legs on ISMIP7 names.
    hist_ismip7 = product_leg in (None, "historical")
    proj_ismip7 = product_leg in (None, "projection")

    # Per-basin post-processing command, populated only for the single-leg run
    # (counter-driven product runs use the ISMIP7 scalar splitter instead).
    post_process_str = ""

    if single_leg:
        # --- Single-pathway run (no ISMIP7 counter) ---
        # Emit exactly one PISM invocation, spanning the config's
        # time.start..time.end, into the template slot that matches
        # campaign.pathway. ``run_hist`` already carries the common sections,
        # bootstrap, and (historical) forcing overrides applied above.
        experiment_id = "historical" if single_is_historical else pathway
        proj_experiment = experiment_id
        run_projection = False
        state_one, spatial_one, scalar_one, basin_one = _output_files(experiment_id, start, end, ismip7=True)
        run_one = run_hist
        run_one.update(
            {
                "output.file": state_one.resolve(),
                "output.spatial.file": spatial_one.resolve(),
                "output.scalar.file": scalar_one.resolve(),
            }
        )
        # A projection pathway swaps the historical forcing (applied via ``uq``)
        # for the projection-epoch files ``_run`` passes on ``proj_overrides``.
        if not single_is_historical and proj_overrides:
            proj_clean = {k: v for k, v in dict(proj_overrides).items() if k != "sample"}
            proj_clean, proj_skipped = filter_overrides_by_config(proj_clean, run_one.keys())
            if proj_skipped:
                print(f"Skipping proj overrides not in config: {proj_skipped}")
            run_one.update(proj_clean)
        if pism_config_cdl is not None:
            validate_pism_options(run_one, pism_config_cdl)
        one_str = dict2str(sort_dict_by_key(run_one))
        # Route to the matching slot; the other stays empty so the template omits it.
        run_hist_str = one_str if single_is_historical else ""
        run_proj_str = "" if single_is_historical else one_str
        # The single leg is the product; expose its scalar under the name the
        # post-processing block below expects (scalar_hist, gated by hist_ismip7).
        scalar_hist = scalar_one
        scalar_proj = None
        # Clip the (combined) spatial output to basins and write per-basin
        # scalar sums. Needs a real outline; skip if none was supplied.
        if outline_file != "none":
            _nt = _postprocess_ntasks(config_cli)
            post_process_str = (
                f"pism-ismip7-greenland-postprocess "
                f"{spatial_one.resolve()} {basin_one.resolve()} {outline_file}{_nt}"
            )
    else:
        # --- Counter-driven ISMIP7 two-leg run (historical -> projection) ---
        # Historical experiments (C001/C002) are historical-only: skip the projection
        # continuation (and its forcing) entirely (product_leg == "historical").
        run_projection = product_leg != "historical"
        state_hist, spatial_hist, scalar_hist, _ = _output_files("historical", start, "2015-01-01", ismip7=hist_ismip7)
        proj_experiment = str(cfg.run_info.experiment) if cfg.run_info.experiment else "none"

        run_hist.update(
            {
                "output.file": state_hist.resolve(),
                "output.spatial.file": spatial_hist.resolve(),
                "output.scalar.file": scalar_hist.resolve(),
            }
        )

        if pism_config_cdl is not None:
            validate_pism_options(run_hist, pism_config_cdl)

        run_hist_str = dict2str(sort_dict_by_key(run_hist))

        # Projection continuation leg. Skipped entirely for historical-only
        # experiments (C001/C002), whose projection forcing is neither staged nor run;
        # ``run_proj_str`` stays empty so the template omits the projection invocation.
        run_proj_str = ""
        scalar_proj = None
        if run_projection:
            # Projection end comes from the config (time.end), which the counter
            # resolver sets from the Core Experiment's proj_end_year (2100 or 2300).
            state_proj, spatial_proj, scalar_proj, _ = _output_files(
                proj_experiment, "2015-01-01", end, ismip7=proj_ismip7
            )

            run_proj = run_hist.copy()
            # Restore run_info.experiment to the projection value (run_hist has it
            # forced to "historical"); the rest of run_info survives the copy.
            # ``InfoConfig.as_params()`` deliberately drops the ISMIP7 naming-only
            # fields (domain / set / ism / experiment) so they don't leak into the
            # PISM command; we want ``run_info.experiment`` to survive, so bypass
            # the filter with a direct assignment, quoted the same way as_params()
            # would if it emitted the field.
            run_proj.update(cfg.run_info.as_params())
            if cfg.run_info.experiment:
                run_proj["run_info.experiment"] = f'"{cfg.run_info.experiment}"'
            run_proj.update({"input.file": state_hist.resolve()})
            run_proj.pop("time.start", None)
            run_proj.update({"time.start": "2015-01-01"})
            run_proj.pop("time.end", None)
            run_proj.update({"time.end": end})
            run_proj.pop("input.bootstrap", None)
            run_proj.pop("input.regrid.file", None)
            run_proj.pop("input.regrid.vars", None)
            run_proj.update(
                {
                    "output.file": state_proj.resolve(),
                    "output.spatial.file": spatial_proj.resolve(),
                    "output.scalar.file": scalar_proj.resolve(),
                }
            )
            # Projection-epoch file paths supplied by ``_run()`` (climate / ocean
            # / gradient) — filtered against ``run_proj`` so a mis-typed key from
            # the caller doesn't silently vanish.
            if proj_overrides:
                proj_clean = {k: v for k, v in dict(proj_overrides).items() if k != "sample"}
                proj_clean, proj_skipped = filter_overrides_by_config(proj_clean, run_proj.keys())
                if proj_skipped:
                    print(f"Skipping proj overrides not in config: {proj_skipped}")
                run_proj.update(proj_clean)
            run_proj_str = dict2str(sort_dict_by_key(run_proj))

    # Point the compliance checker at this run's actual ISMIP7 submission
    # directory (output/<domain>/<source>/<ism>/<set>/<set_counter>/) rather than a
    # hardcoded path. The directory depends only on the ismip7_ctx identity fields
    # (experiment_id/time_range don't affect it), and both forward legs write into
    # it. When ISMIP7 naming is off there is no submission tree, so leave it empty.
    if ismip7_ctx is not None:
        submission_dir = ISMIP7Names(experiment_id=proj_experiment, time_range="", **ismip7_ctx).directory(
            output_path.resolve()
        )
        ism_checker_str = f"ismip7-compliance-checker --source-path {submission_dir}/ --variable-list ismip7"
        # Split each ISMIP7 product scalar file into per-variable diagnostics via
        # the packaged post-processing script. The product leg's scalar is used:
        # scalar_hist for historical experiments (C001/C002), scalar_proj for the
        # projection experiments; a legacy non-counter run treats both legs as
        # products (hist_ismip7 and proj_ismip7 both True) and post-processes both.
        post_script = Path(__file__).resolve().parents[2] / "data" / "postprocess_ismip7_scalar.sh"
        post_scalars = []
        if hist_ismip7:
            post_scalars.append(scalar_hist)
        if run_projection and proj_ismip7 and scalar_proj is not None:
            post_scalars.append(scalar_proj)
        post_scalar_str = "\n".join(f"bash {post_script} {s.resolve()}" for s in post_scalars)
    else:
        ism_checker_str = ""
        post_scalar_str = ""

    return {
        "run_hist_str": run_hist_str,
        "run_proj_str": run_proj_str,
        "post_process_str": post_process_str,
        "ism_checker_str": ism_checker_str,
        "post_scalar_str": post_scalar_str,
    }


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
    proj_overrides: Mapping[str, object] | None = None,
):
    """
    Configure and generate a PISM forward job script for ISMIP7 Greenland (ensemble-ready).

    Reads a TOML configuration, merges optional ensemble overrides (``uq``),
    renders a submission script from a Jinja2 template, and writes both the
    script and a companion TOML describing the resolved run parameters. Also
    emits a command-line string of PISM flags derived from the config and
    overrides.

    When the campaign config defines ``init_start``/``init_end``, a short
    init/prior leg (``run_init_str``) is rendered first — the same leg 1 the
    inverse workflow uses — and the forward leg(s) restart from its state
    file instead of bootstrapping. Without those fields the forward leg
    bootstraps directly, as before.

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
    proj_overrides : Mapping[str, object] or None, optional
        Projection-only overrides applied to ``run_proj`` after it is
        copied from ``run_hist``. Same schema as ``uq`` (dotted PISM
        flags). Used to point ``atmosphere.given.file`` etc. at the
        projection-epoch forcing file while ``run_hist`` keeps the
        historical file. Default is ``None`` (no proj-only overrides).

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
    paths = _make_output_paths(path)
    log_path = paths["log"]
    output_path = paths["output"]
    scalar_path = paths["scalar"]
    spatial_path = paths["spatial"]
    state_path = paths["state"]

    run_hist = _base_run_dict(cfg)

    template_file = Path(template_file)
    env = Environment(loader=FileSystemLoader(template_file.parent))
    template = env.get_template(template_file.name)

    # CLI overrides for time bounds. ``cfg.time`` is a TimeConfig pydantic
    # model with field names ``time_start`` / ``time_end`` (aliased to the
    # dotted ``"time.start"`` / ``"time.end"``), so attribute assignment is
    # required. Drop the prior dotted entry from ``run_hist`` and re-apply via
    # ``as_params()`` so the new value lands cleanly.
    _start = config_cli.get("start")
    _end = config_cli.get("end")
    if _start is not None:
        run_hist.pop("time.start", None)
        cfg.time.time_start = _start
        run_hist.update(cfg.time.as_params())
    if _end is not None:
        run_hist.pop("time.end", None)
        cfg.time.time_end = _end
        run_hist.update(cfg.time.as_params())

    start = cfg.model_dump(by_alias=True)["time"]["time.start"]
    end = cfg.model_dump(by_alias=True)["time"]["time.end"]

    if resolution is None:
        resolution = cfg.model_dump(by_alias=True)["grid"]["resolution"]
    # CLI override for the stress-balance model. Drop the previous model's
    # options from ``run_hist`` first so leftover keys (e.g. blatter.*) don't
    # leak into e.g. a sia run.
    stress_balance = config_cli.get("stress_balance")
    if stress_balance is not None:
        for old_key in cfg.stress_balance.selected():
            run_hist.pop(old_key, None)
        cfg.stress_balance.model = stress_balance
        run_hist.update(cfg.stress_balance.selected())
    stress_balance = cfg.model_dump(by_alias=True)["stress_balance"]["model"]

    energy = cfg.model_dump(by_alias=True)["energy"]["model"]
    surface = cfg.model_dump(by_alias=True)["surface"]["model"]

    # ``InfoConfig.as_params()`` deliberately drops the ISMIP7 naming-only
    # fields (see ``_PISM_FIELDS`` in config.py) — grab ``experiment`` off
    # the pydantic model directly.
    experiment = cfg.run_info.experiment or "none"
    if sample is None:
        name_options = f"surface_{surface}_energy_{energy}_stress_balance_{stress_balance}"
    else:
        name_options = f"id_{sample}_{experiment}"

    uq_clean = normalize_row(uq) if uq is not None else {}
    # Prefer explicit `sample` arg; else default from uq['sample']
    if sample is None and "sample" in uq_clean:
        try:
            sample = int(uq_clean["sample"])
        except Exception:
            pass

    # Remove 'sample' from flag overrides; drop any key not in the config-derived
    # run_hist dict (e.g., surface.debm_simple.std_dev.file when surface.model == "pdd").
    overrides = {k: v for k, v in uq_clean.items() if k != "sample"}
    overrides, skipped = filter_overrides_by_config(overrides, run_hist.keys())
    if skipped:
        print(f"Skipping uq overrides not in config: {skipped}")
    # Apply to runtime dict (these should be dotted PISM flags)
    run_hist.update(overrides)

    # Optional init/prior leg: when the campaign config carries
    # init_start/init_end, render the short bootstrap leg first (mirroring
    # the inverse workflow's leg 1, minus the inversion) and restart the
    # forward leg(s) from its state instead of bootstrapping them directly.
    # Without the campaign fields the forward leg bootstraps as before.
    init_start = cfg.campaign.init_start
    init_end = cfg.campaign.init_end
    run_init_str = ""
    if init_start and init_end:
        name_options_init = f"id_{sample}" if sample is not None else name_options
        run_init_str, state_init, _ = _build_init_leg(
            cfg,
            init_start=init_start,
            init_end=init_end,
            resolution=resolution,
            name_options_init=name_options_init,
            overrides=overrides,
            state_path=state_path,
            scalar_path=scalar_path,
            spatial_path=spatial_path,
            pism_config_cdl=pism_config_cdl,
        )
        run_hist.update({"input.file": state_init.resolve()})
        run_hist.pop("input.bootstrap", None)
        run_hist.pop("input.regrid.file", None)
        run_hist.pop("input.regrid.vars", None)

    # Single-leg vs counter-driven two-leg split, ISMIP7 naming, and the
    # post-processing command strings all live in the shared helper.
    leg_params = _build_forward_legs(
        cfg,
        run_hist,
        start=start,
        end=end,
        resolution=resolution,
        name_options=name_options,
        sample=sample,
        output_path=output_path,
        state_path=state_path,
        spatial_path=spatial_path,
        scalar_path=scalar_path,
        outline_file=outline_file,
        config_cli=config_cli,
        proj_overrides=proj_overrides,
        pism_config_cdl=pism_config_cdl,
    )

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

    # Empty when the campaign config has no init bounds; the templates guard
    # the slot so the script shape is unchanged in that case.
    params.update({"run_init_str": run_init_str})
    params.update(leg_params)

    rendered_script = "" if debug else template.render(params)

    run_script_path = path / Path("run_scripts")
    run_script_path.mkdir(parents=True, exist_ok=True)

    run_script = run_script_path / Path(f"submit_g{resolution}_{name_options}.sh")

    # Save or print the output
    run_script.write_text(rendered_script)

    print(f"\nJob script written to {run_script.resolve()}\n")


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
    proj_overrides: Mapping[str, object] | None = None,
):
    """
    Configure and generate a chained PISM inverse job script for ISMIP7 Greenland.

    The generated script runs three (or four) legs:

    1. **Init/prior** (``run_init_str``): the bootstrap forward run, spanning
       ``campaign.init_start``..``campaign.init_end`` (required config
       fields), regridding ``litho_temp,enthalpy,age,tillwat`` from the
       staged regrid file.
    2. **Inversion** (``inv_str``): the pismi call, restarting from the init
       leg's state file and writing the inverted ``tauc`` to
       ``output/inverse/``.
    3. **Forward** (``run_hist_str``): a forward run restarting from the init
       leg's state (no bootstrap), regridding ``tauc`` from the inversion
       output, with ``basal_yield_stress.model = "constant"`` (the
       ``basal_yield_stress.mohr_coulomb.*`` options are dropped). For a
       config without an ISMIP7 counter this single leg spans the config's
       ``time.start``..``time.end``.
    4. **Projection** (``run_proj_str``): for counter-driven ISMIP7 configs
       only — the forward leg stops at 2015-01-01 and this continuation runs
       to ``time.end``, with the projection-epoch forcing from
       ``proj_overrides``; ISMIP7 submission naming, product-leg gating and
       post-processing match :func:`_render_forward_run` exactly.

    Parameters
    ----------
    config_file : str or pathlib.Path
        Path to the PISM configuration TOML (contains ``run``, ``grid``,
        ``time``, ``surface``, ``energy``, ``stress_balance``, ``inverse``;
        the ``[campaign]`` section must carry ``init_start``/``init_end``).
    template_file : str or pathlib.Path
        Path to a Jinja2 submission template. The context includes
        ``run_init_str`` (init pism call, also aliased as ``run_str`` for
        the generic inverse templates), ``inv_str`` (pismi call), and the
        forward-leg slots ``run_hist_str``/``run_proj_str`` plus the
        post-processing strings shared with the forward template.
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
        :func:`_render_forward_run` for the recognized keys. ``"start"`` /
        ``"end"`` apply to the *forward* legs only; the init leg is always
        bounded by ``campaign.init_start``/``init_end``. Default is
        ``None`` (no overrides).
    debug : bool, optional
        If ``True``, skip rendering the template (leave it empty) but still
        write the resolved post-processing TOML. Default is ``False``.
    uq : Mapping[str, object] or pandas.Series or None, optional
        Ensemble overrides. Keys are dotted PISM flags routed to whichever
        of the init (``run_init``), inversion (``inv``) or forward
        (``run_fwd``) dicts own them; e.g. ``inverse.file`` propagates into
        ``inv_str`` only, while historical forcing files reach both the
        init and forward legs.
    sample : int or None, optional
        Ensemble member identifier. If not provided, and ``uq`` has
        ``"sample"``, that value is used. Changes the filename stem used
        for outputs (e.g., ``..._id_0042``).
    pism_config_cdl : str or pathlib.Path or None, optional
        Path to a PISM CDL master config file. If provided, all forward
        run options are validated against it before generating the
        command line.
    proj_overrides : Mapping[str, object] or None, optional
        Projection-only overrides applied to the projection continuation
        leg (dotted PISM flags, same schema as in
        :func:`_render_forward_run`). Default is ``None``.

    Raises
    ------
    SystemExit
        If ``campaign.init_start`` / ``campaign.init_end`` are missing.
    """

    outline_file = str(Path(outline_file).resolve()) if (outline_file is not None) else "none"
    cfg = load_config(config_file)

    init_start = cfg.campaign.init_start
    init_end = cfg.campaign.init_end
    if not init_start or not init_end:
        raise SystemExit(
            "run-inverse requires [campaign] init_start and init_end (e.g. "
            'init_start = "2006-01-01", init_end = "2007-01-01") to bound the '
            "init/prior leg; add them to the campaign section of the config TOML."
        )

    config_cli = config_cli or {}
    resolution = config_cli.get("resolution")
    if resolution:
        resolution = re.sub(r"\s+", "", resolution)

        # update GridConfig and force dx/dy to be derived from the new resolution
        cfg.grid.resolution = resolution
        cfg.grid.dx = None
        cfg.grid.dy = None

    path = Path(path)
    paths = _make_output_paths(path, inverse=True)
    log_path = paths["log"]
    output_path = paths["output"]
    scalar_path = paths["scalar"]
    spatial_path = paths["spatial"]
    state_path = paths["state"]
    inv_path = paths["inverse"]

    # CLI override for the stress-balance model, applied to ``cfg`` *before*
    # any run dict is built so the init leg, the pismi call and the forward
    # legs all agree on the model (no stale-key cleanup needed).
    stress_balance = config_cli.get("stress_balance")
    if stress_balance is not None:
        cfg.stress_balance.model = stress_balance
    stress_balance = cfg.model_dump(by_alias=True)["stress_balance"]["model"]

    # CLI overrides for time bounds apply to the *forward* legs only; the
    # init leg is always bounded by campaign.init_start/init_end. ``cfg.time``
    # is a TimeConfig pydantic model with field names ``time_start`` /
    # ``time_end`` (aliased to the dotted keys), so set attributes.
    _start = config_cli.get("start")
    _end = config_cli.get("end")
    if _start is not None:
        cfg.time.time_start = _start
    if _end is not None:
        cfg.time.time_end = _end

    start = cfg.model_dump(by_alias=True)["time"]["time.start"]
    end = cfg.model_dump(by_alias=True)["time"]["time.end"]

    if resolution is None:
        resolution = cfg.model_dump(by_alias=True)["grid"]["resolution"]

    # Leg 2 (inversion): pismi flags.
    inv: dict = {}
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

    # pismi runs the (Blatter) stress balance during the inversion, so it needs the
    # same energy model and flow law as the forward prior. These live in
    # cfg.energy.selected() (e.g. energy.model and stress_balance.*.flow_law =
    # gpbld); without them pismi would silently fall back to PISM's default flow
    # law, making the inversion inconsistent with the forward runs.
    inv.update(cfg.energy.selected())

    # Legs 3/4 base (forward with inverted tauc), spanning time.start..time.end.
    run_fwd = _base_run_dict(cfg)

    template_file = Path(template_file)
    env = Environment(loader=FileSystemLoader(template_file.parent))
    template = env.get_template(template_file.name)

    energy = cfg.model_dump(by_alias=True)["energy"]["model"]
    surface = cfg.model_dump(by_alias=True)["surface"]["model"]

    uq_clean = normalize_row(uq) if uq is not None else {}
    # Prefer explicit `sample` arg; else default from uq['sample']
    if sample is None and "sample" in uq_clean:
        try:
            sample = int(uq_clean["sample"])
        except Exception:
            pass

    # ``InfoConfig.as_params()`` deliberately drops the ISMIP7 naming-only
    # fields — grab ``experiment`` off the pydantic model directly. Legs 1/2
    # use flat non-ISMIP7 names without the experiment (matching the historical
    # inverse behavior); the forward legs use the forward naming convention.
    experiment = cfg.run_info.experiment or "none"
    if sample is None:
        name_options_init = f"surface_{surface}_energy_{energy}_stress_balance_{stress_balance}"
        name_options_fwd = name_options_init
    else:
        name_options_init = f"id_{sample}"
        name_options_fwd = f"id_{sample}_{experiment}"

    # Route each uq key to whichever dict(s) own it: pismi call and/or the
    # forward legs (the init leg filters the same overrides internally, and
    # its keys are a subset of ``run_fwd``'s). A key is "skipped" only if
    # nobody knows it (e.g. surface.debm_simple.std_dev.file when
    # surface.model == "pdd").
    all_overrides = {k: v for k, v in uq_clean.items() if k != "sample"}
    inv_overrides, _ = filter_overrides_by_config(all_overrides, inv.keys())
    fwd_overrides, _ = filter_overrides_by_config(all_overrides, run_fwd.keys())
    skipped = [k for k in all_overrides if k not in inv and k not in run_fwd]
    if skipped:
        print(f"Skipping uq overrides not in config: {skipped}")
    inv.update(inv_overrides)
    run_fwd.update(fwd_overrides)

    # Leg 1 (init/prior): bootstrap + staged regrid over init_start..init_end.
    run_init_str, state_init, init_tag = _build_init_leg(
        cfg,
        init_start=init_start,
        init_end=init_end,
        resolution=resolution,
        name_options_init=name_options_init,
        overrides=all_overrides,
        state_path=state_path,
        scalar_path=scalar_path,
        spatial_path=spatial_path,
        pism_config_cdl=pism_config_cdl,
    )

    inv_file = inv_path / Path(f"inv_{init_tag}.nc")
    # Feed the init leg's state file into pismi as its input.
    inv.update({"input.file": state_init.resolve()})
    inv.update({"o": inv_file.resolve()})
    inv_str = dict2str(sort_dict_by_key(inv))

    # Leg-3 wiring, applied AFTER the uq overrides so it always wins: restart
    # from the init state (no bootstrap) and regrid the inverted tauc, driven
    # by the constant yield-stress model (the mohr_coulomb options only apply
    # to legs 1/2, which produced the tauc field being read back here).
    run_fwd.update({"input.file": state_init.resolve()})
    run_fwd.pop("input.bootstrap", None)
    run_fwd.update({"input.regrid.file": inv_file.resolve(), "input.regrid.vars": "tauc"})
    run_fwd["basal_yield_stress.model"] = "constant"
    for key in [k for k in run_fwd if k.startswith("basal_yield_stress.mohr_coulomb.")]:
        run_fwd.pop(key)

    leg_params = _build_forward_legs(
        cfg,
        run_fwd,
        start=start,
        end=end,
        resolution=resolution,
        name_options=name_options_fwd,
        sample=sample,
        output_path=output_path,
        state_path=state_path,
        spatial_path=spatial_path,
        scalar_path=scalar_path,
        outline_file=outline_file,
        config_cli=config_cli,
        proj_overrides=proj_overrides,
        pism_config_cdl=pism_config_cdl,
    )

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

    params.update({"run_init_str": run_init_str})
    # The generic (non-ISMIP7) inverse templates name the pism prior slot
    # ``run_str``; keep them usable as init+pismi-only scripts.
    params.update({"run_str": run_init_str})
    params.update({"inv_str": inv_str})
    # ``run_hist_str`` / ``run_proj_str`` and the post-processing strings now
    # carry the forward (tauc) legs, matching the forward template semantics.
    params.update(leg_params)

    rendered_script = "" if debug else template.render(params)

    run_script_path = path / Path("run_scripts")
    run_script_path.mkdir(parents=True, exist_ok=True)

    run_script = run_script_path / Path(f"submit_g{resolution}_{name_options_fwd}.sh")

    run_script.write_text(rendered_script)

    print(f"\nJob script written to {run_script.resolve()}\n")


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
        "--data-path",
        help="Shared directory for staged input data (reused across runs). " "Defaults to <output-path>/input.",
        type=str,
        default=None,
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
    # Input data location is handled by ``stage`` (``--data-path`` or <path>/input).
    data_path = options.data_path
    output_path = path / Path("output")
    output_path.mkdir(parents=True, exist_ok=True)

    config_file = options.CONFIG_FILE
    template_file = options.TEMPLATE_FILE
    uq_file = options.UQ_FILE
    pism_config_cdl = options.pism_config_cdl

    cfg = load_config(config_file)
    campaign_config = cfg.campaign.as_params()

    # Skip staging the (large) projection forcing when it isn't used.
    # Historical experiments (C001/C002, product_leg == "historical") are
    # historical-only and run no projection continuation. Both the forward and
    # the (chained) inverse workflows end in the same forward legs, so the
    # rules are identical for both kinds.
    counter = cfg.run_info.counter
    pathway = str(campaign_config.get("pathway") or "").strip().lower()
    if counter:
        # Counter-driven ISMIP7 protocol run: historical experiments (C001/C002)
        # are historical-only; the rest continue into a projection leg.
        include_projection = resolve_counter(counter).product_leg != "historical"
    else:
        # Single-pathway run: only stage projection forcing when the pathway is
        # itself a projection (a historical run needs no projection continuation).
        include_projection = pathway != "historical"
    df = stage(
        campaign_config,
        path=path,
        force_overwrite=force_overwrite,
        include_projection=include_projection,
        data_path=data_path,
    )

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
        # same flag (matches the glacier behavior). ``uq_overrides`` carries
        # the *historical*-epoch paths (which populate ``run_hist``); the
        # projection paths ride on ``proj_overrides`` and only touch
        # ``run_proj``.
        uq_overrides.update(
            {
                "input.file": row["boot_file"],
                "input.regrid.file": row["regrid_file"],
                "geometry.front_retreat.prescribed.file": row["retreat_file"],
                "grid.file": row["grid_file"],
                "energy.bedrock_thermal.file": row["heatflux_file"],
                "surface.ismip7.reference.file": row["boot_file"],
                "atmosphere.given.file": row["climate_hist_file"],
                "surface.given.file": row["climate_hist_file"],
                "surface.ismip7.file": row["climate_hist_file"],
                "surface.ismip7.gradient.file": row["climate_gradient_hist_file"],
                "ocean.pico.file": row["ocean_hist_file"],
                "ocean.picop.file": row["ocean_hist_file"],
                "ocean.th.file": row["ocean_hist_file"],
                "frontal_melt.routing.file": row["ocean_hist_file"],
            }
        )
        # Projection-epoch overrides only exist when the projection forcing was
        # staged. Historical(-only) runs — pathway == "historical" or the C001/
        # C002 experiments — skip it, so the ``*_proj_file`` columns are absent
        # from the row (see ``include_projection`` above /
        # stage(..., include_projection=...)).
        proj_overrides = None
        if include_projection:
            proj_overrides = {
                "atmosphere.given.file": row["climate_proj_file"],
                "geometry.front_retreat.prescribed.file": row["retreat_file"],
                "surface.given.file": row["climate_proj_file"],
                "surface.ismip7.file": row["climate_proj_file"],
                "surface.ismip7.gradient.file": row["climate_gradient_proj_file"],
                "ocean.pico.file": row["ocean_proj_file"],
                "ocean.picop.file": row["ocean_proj_file"],
                "ocean.th.file": row["ocean_proj_file"],
                "frontal_melt.routing.file": row["ocean_proj_file"],
            }
        # Wire the inverse observation file only when the stage produced one
        # (campaign config can opt in via an ``obs_file`` key); otherwise
        # rely on whatever ``inverse.file`` the UQ supplied.
        if kind == "inverse" and "obs_file" in row and pd.notna(row["obs_file"]):
            uq_overrides["inverse.file"] = row["obs_file"]

        outline_file = row["outline_file"] if "outline_file" in row else None
        # ISMIP7 staging uses the GCM name (a string like "CESM2-WACCM") as
        # the sample id, so don't try to coerce to int the way the glacier
        # CLI does.
        sample = row["sample"] if "sample" in row else idx
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
            proj_overrides=proj_overrides,
        )


def run_forward() -> None:
    """
    CLI entry point for ISMIP7 Greenland forward runs (single or ensemble).

    When the campaign config defines ``init_start``/``init_end``, a short
    init/prior leg runs first and the forward leg(s) restart from its state
    (see :func:`_render_forward_run`); otherwise the forward leg bootstraps
    directly.

    Behaves as a single run when no ``UQ_FILE`` positional is supplied, and
    as a UQ ensemble when one is. The argument schema and output layout are
    otherwise identical.
    """
    _run(kind="forward")


def run_inverse() -> None:
    """
    CLI entry point for ISMIP7 Greenland inverse runs (single or ensemble).

    Renders a chained script: init/prior run over
    ``campaign.init_start``..``init_end``, a pismi inversion, and the
    forward leg(s) restarting from the init state with the inverted
    ``tauc`` regridded in (see :func:`_render_inverse_run`).

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
