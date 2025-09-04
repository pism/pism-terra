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

import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import toml
import xarray as xr
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel

from pism_terra.config import JobConfig, RunConfig, load_config, load_uq
from pism_terra.sampling import create_samples

# one Jinja environment for all renders
_JINJA = Environment(undefined=StrictUndefined, autoescape=False)

T = TypeVar("T", bound=BaseModel)


def _merge_model(base_model: T, **overrides: Any) -> T:
    """
    Create a new Pydantic model instance by shallow-merging non-None overrides.

    This returns a **new** instance of the same model class as ``base_model``.
    It starts from ``base_model``'s data (via ``model_dump()``) and applies
    values from ``overrides`` where the value is not ``None``. Unknown fields
    or invalid values will raise a Pydantic ``ValidationError``.

    Parameters
    ----------
    base_model : T
        An existing Pydantic model instance to use as the base.
    **overrides : Any
        Field values to override on top of ``base_model``. Keys must be valid
        field names of the model. Any key whose value is ``None`` is ignored.

    Returns
    -------
    T
        A **new** instance of ``type(base_model)`` with overrides applied.

    Raises
    ------
    pydantic.ValidationError
        If any override fails validation or refers to an unknown field,
        depending on the model's configuration.
    TypeError
        If ``base_model`` is not a Pydantic ``BaseModel`` instance.

    Notes
    -----
    * This is a **shallow** merge. Nested models or containers are replaced,
      not deep-merged.
    * In Pydantic v2, an equivalent (and concise) pattern is::

        new_model = base_model.model_copy(
            update={k: v for k, v in overrides.items() if v is not None}
        )

      This function mirrors that behavior while making the "skip None" rule explicit.

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> class RunConfig(BaseModel):
    ...     ntasks: int | None = None
    ...     mpi: str = "srun -n {ntasks}"
    ...
    >>> rc = RunConfig(ntasks=16)
    >>> new = _merge_model(rc, ntasks=80, extra=None)  # 'extra' ignored; None skipped
    >>> new.ntasks
    80
    >>> type(new) is type(rc)
    True
    """
    if not isinstance(base_model, BaseModel):
        raise TypeError("base_model must be a Pydantic BaseModel instance")

    data = base_model.model_dump()
    data.update({k: v for k, v in overrides.items() if v is not None})
    return type(base_model)(**data)


def _to_python_scalar(v: Any) -> Any:
    """
    Convert NumPy/Pandas scalar types to built-in Python objects.

    Attempts to coerce common array/scalar dtypes (e.g., ``np.int64``,
    ``np.float32``) and pandas datetime-like scalars into plain Python types
    suitable for TOML/JSON serialization and Jinja rendering.

    Parameters
    ----------
    v : Any
        Value to coerce. Supports values deriving from ``numpy.generic`` as well
        as pandas objects exposing ``.to_pydatetime()`` or ``.to_pytimedelta()``.
        Other types are returned unchanged.

    Returns
    -------
    Any
        A built-in Python scalar (e.g., ``int``, ``float``, ``bool``,
        ``datetime.datetime``, ``datetime.timedelta``) when conversion applies;
        otherwise the original value.

    Notes
    -----
    - This function does **not** recurse into containers; it only converts the
      top-level value. Use it element-wise for sequences/mappings.
    - Any exception during the NumPy scalar probe is intentionally ignored to
      keep the conversion path robust across environments.

    Examples
    --------
    >>> import numpy as np
    >>> _to_python_scalar(np.int64(3))
    3
    >>> import pandas as pd
    >>> _to_python_scalar(pd.Timestamp("2020-01-01"))
    datetime.datetime(2020, 1, 1, 0, 0)
    """
    try:

        if isinstance(v, (np.generic,)):
            return v.item()
    except Exception:
        pass

    # Pandas Timestamp/Timedelta-like
    if hasattr(v, "to_pydatetime"):
        return v.to_pydatetime()
    if hasattr(v, "to_pytimedelta"):
        return v.to_pytimedelta()

    return v


def _normalize_row(row) -> dict:
    """
    Normalize a row-like object into a clean ``dict[str, Any]``.

    Converts values to plain Python scalars via :func:`_to_python_scalar`,
    casts keys to ``str``, and drops entries whose value is ``None``. Intended
    for preparing parameter dictionaries for TOML/JSON dumps and Jinja contexts.

    Parameters
    ----------
    row : mapping or pandas.Series
        A mapping-like object (e.g., ``dict``) or a ``pandas.Series`` obtained
        from ``DataFrame.iterrows()``. Keys are interpreted as parameter names.

    Returns
    -------
    dict
        Dictionary with string keys and Python-native scalar values. Any items
        with ``None`` values are omitted.

    Notes
    -----
    - This function does **not** deep-convert nested containers; nested dicts or
      lists are left as-is except for top-level value coercion.
    - If ``row`` is a pandas ``Series``, its ``.to_dict()`` is used first.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series({"a": pd.Timestamp("2021-01-01"), "b": None, "c": 1})
    >>> _normalize_row(s)
    {'a': datetime.datetime(2021, 1, 1, 0, 0), 'c': 1}
    """
    d = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        out[str(k)] = _to_python_scalar(v)
    return out


def run_glacier(
    rgi_id: str,
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

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    glacier_path = path / Path(rgi_id)
    glacier_path.mkdir(parents=True, exist_ok=True)
    output_path = glacier_path / Path("output")
    output_path.mkdir(parents=True, exist_ok=True)

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
    run.update(cfg.stress_balance.selected())
    run.update(cfg.atmosphere.selected())
    run.update(cfg.surface.selected())
    run.update(cfg.energy.selected())
    run.update(cfg.grid.as_params())
    run.update(cfg.run_info.as_params())
    run.update(cfg.time.as_params())

    template_file = Path(template_file)
    env = Environment(loader=FileSystemLoader(template_file.parent))
    template = env.get_template(template_file.name)

    start = cfg.model_dump(by_alias=True)["time"]["time.start"]
    end = cfg.model_dump(by_alias=True)["time"]["time.end"]

    if resolution is None:
        resolution = cfg.model_dump(by_alias=True)["grid"]["resolution"]
    stress_balance = cfg.model_dump(by_alias=True)["stress_balance"]["model"]
    energy = cfg.model_dump(by_alias=True)["energy"]["model"]
    surface = cfg.model_dump(by_alias=True)["surface"]["model"]

    if sample is None:
        name_options = f"surface_{surface}_energy_{energy}_stress_balance_{stress_balance}"
    else:
        name_options = f"id_{sample}"

    uq_clean = _normalize_row(uq) if uq is not None else {}
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

    spatial_file = output_path / Path(f"spatial_g{resolution}_{rgi_id}_{name_options}_{start}_{end}.nc")
    state_file = output_path / Path(f"state_g{resolution}_{rgi_id}_{name_options}_{start}_{end}.nc")
    run.update(
        {
            "output.file": state_file.absolute(),
            "output.extra.file": spatial_file.absolute(),
        }
    )

    print("Checking files")
    print("-" * 80)
    input_files = {k: v for k, v in run.items() if k.endswith(".file") and not k.startswith("output.")}
    for k, v in input_files.items():
        p = Path(v)
        try:
            xr.open_dataset(p)
            print(f"{k}: {v} is valid ✓")
        except FileNotFoundError as e:
            print(f"{k}: {v} is valid ✗")
    print("-" * 80)

    run_str = dict2str(sort_dict_by_key(run))

    run_opts = RunConfig(**cfg.run.model_dump())
    job_opts = JobConfig(**cfg.job.model_dump())

    params = {
        **run_opts.model_dump(exclude_none=True, by_alias=True),
        **job_opts.model_dump(exclude_none=True, by_alias=True),
    }

    # run_opts comes from your config; ntasks comes from CLI (or None)
    active_run_opts = _merge_model(run_opts, ntasks=ntasks)

    # Use this ONE source to update params and to compute mpi_str
    run_params = active_run_opts.as_params()
    params.update(run_params)
    mpi_str = run_params["mpi"]  # guaranteed consistent with ntasks override

    job_kwargs = {k: v for k, v in {"queue": queue, "walltime": walltime, "nodes": nodes}.items() if v is not None}
    if job_kwargs:
        params.update(JobConfig(**job_kwargs).as_params())
    prefix = f"{mpi_str} {cfg.run.executable} "

    rendered_script = "" if debug else template.render(params)
    rendered_script += f"\n\n{prefix}{run_str}\n"

    run_script_path = glacier_path / Path("run_scripts")
    run_script_path.mkdir(parents=True, exist_ok=True)

    run_script = run_script_path / Path(f"submit_g{resolution}_{rgi_id}_{name_options}_{start}_{end}.sh")

    # Save or print the output
    run_script.write_text(rendered_script)

    run_toml = {
        "rgi": {"rgi_id": rgi_id, "outline": str(outline_file.absolute())},
        "output": {
            "spatial": str(spatial_file.absolute()),
            "state": str(state_file.absolute()),
        },
        "config": run,
    }
    run_file = output_path / Path(f"g{resolution}_{rgi_id}_{name_options}_{start}_{end}.toml")
    with open(run_file, "w", encoding="utf-8") as toml_file:
        toml.dump(run_toml, toml_file)
    print(f"\nSLURM script written to {run_script.absolute()}\n")
    print(f"Postprocessing script written to {run_file.absolute()}\n")


def apply_choice_mapping(uq_df: pd.DataFrame, df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """
    Replace integer choices in `uq_df` with values from `df` using a per-flag mapping.

    Parameters
    ----------
    uq_df : pandas.DataFrame
        DataFrame produced by your sampler (has integer-coded columns like
        'surface.given.file', 'atmosphere.given.file', etc.).
    df : pandas.DataFrame
        Source DataFrame that contains the lookup columns (e.g., 'cosipy_file').
        Row order defines the integer choices: 0 -> first row, 1 -> second row, etc.
    mapping : dict[str, str]
        Mapping from dotted flag name in `uq_df` to column name in `df`,
        e.g. {"surface.given.file": "cosipy_file"}.

    Returns
    -------
    pandas.DataFrame
        A copy of `uq_df` with the specified columns mapped to their path strings.
    """
    out = uq_df.copy()

    if not isinstance(mapping, dict) or not mapping:
        return out

    for flag, df_col in mapping.items():
        if flag not in out.columns:
            # nothing to map for this flag; skip
            continue
        if df_col not in df.columns:
            raise KeyError(f"Mapping for '{flag}' points to missing df column '{df_col}'")

        # Build int-choice -> value mapping using the *row order* of df[df_col]
        # Using a Series preserves integer index 0..n-1 after reset_index(drop=True)
        choice_series = df[df_col].reset_index(drop=True)

        # Ensure the source choices are integer-coded
        out[flag] = pd.to_numeric(out[flag], errors="raise").astype("int64")

        # Series.map with a Series maps by index; perfect for 0..n-1 codes
        out[flag] = out[flag].map(choice_series)

        # Optional: fail fast if any choice was out of bounds
        if out[flag].isna().any():
            bad = out.loc[out[flag].isna(), flag].index.tolist()
            raise ValueError(
                f"Found out-of-range choice(s) for '{flag}' at rows {bad}; "
                f"valid choices are 0..{len(choice_series)-1}"
            )

    return out


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
    return """  \\\n""".join(f"  -{k} {v}" for k, v in d.items())


def run_single():
    """
    Run single glacier.
    """

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

    options, _ = parser.parse_known_args()
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

    df = pd.read_csv(rgi_file)
    rgi_id = df["rgi_id"].iloc[0]

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    glacier_path = path / Path(rgi_id)
    glacier_path.mkdir(parents=True, exist_ok=True)
    output_path = glacier_path / Path("output")
    output_path.mkdir(parents=True, exist_ok=True)

    input_outline_file = df["outline"].iloc[0]
    output_outline_file = output_path / Path(f"rgi_{rgi_id}.gpkg")
    shutil.copy(input_outline_file, output_outline_file)

    uq = {
        "input.file": df["boot_file"].iloc[0],
        "grid.file": df["grid_file"].iloc[0],
        "surface.force_to_thickness.file": df["boot_file"].iloc[0],
        "atmosphere.given.file": df["historical_climate_file"].iloc[0],
        "atmosphere.elevation_change.file": df["historical_climate_file"].iloc[0],
    }

    run_glacier(
        rgi_id,
        config_file,
        template_file,
        output_outline_file,
        path=path,
        resolution=resolution,
        nodes=nodes,
        ntasks=ntasks,
        queue=queue,
        walltime=walltime,
        debug=debug,
        uq=uq,
    )


def run_ensemble():
    """
    Run single glacier.
    """

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
    parser.add_argument(
        "UQ_FILE",
        help="""UQ TOML.""",
        nargs=1,
    )

    options, _ = parser.parse_known_args()
    path = options.output_path
    rgi_file = options.RGI_FILE[0]
    config_file = options.CONFIG_FILE[0]
    template_file = options.TEMPLATE_FILE[0]
    uq_file = options.UQ_FILE[0]
    resolution = options.resolution
    debug = options.debug
    queue = options.queue
    ntasks = options.ntasks
    nodes = options.nodes
    walltime = options.walltime

    df = pd.read_csv(rgi_file)
    rgi_id = df["rgi_id"].iloc[0]

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    glacier_path = path / Path(rgi_id)
    glacier_path.mkdir(parents=True, exist_ok=True)
    output_path = glacier_path / Path("output")
    output_path.mkdir(parents=True, exist_ok=True)

    input_outline_file = df["outline"].iloc[0]
    output_outline_file = output_path / Path(f"rgi_{rgi_id}.gpkg")
    shutil.copy(input_outline_file, output_outline_file)

    default = {
        "input.file": df["boot_file"].iloc[0],
        "grid.file": df["grid_file"].iloc[0],
        "surface.force_to_thickness.file": df["boot_file"].iloc[0],
        "atmosphere.given.file": df["historical_climate_file"].iloc[0],
        "atmosphere.elevation_change.file": df["historical_climate_file"].iloc[0],
    }

    uq = load_uq(uq_file)
    n_samples = uq.samples
    mapping = uq.mapping
    uq_df = create_samples(uq.to_flat(), n_samples=n_samples, seed=42)
    print("Ensemble")
    print("-" * 80)
    print(uq_df)
    if uq.mapping:
        uq_df = apply_choice_mapping(uq_df, df, uq.mapping)
    for idx, row in uq_df.iterrows():
        default.update(row)
        run_glacier(
            rgi_id,
            config_file,
            template_file,
            output_outline_file,
            path=path,
            resolution=resolution,
            nodes=nodes,
            ntasks=ntasks,
            queue=queue,
            walltime=walltime,
            debug=debug,
            uq=default,
            sample=int(row["sample"]) if "sample" in row else idx,
        )


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    run_single()
