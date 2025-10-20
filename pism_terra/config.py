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
Config.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar, Iterator

import scipy.stats as st
import toml
from jinja2 import Environment, StrictUndefined
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# one Jinja environment for all renders
_JINJA = Environment(undefined=StrictUndefined, autoescape=False)


def load_config(path: str | Path) -> PismConfig:
    """
    Load and validate a PISM configuration from a TOML file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the TOML configuration file.

    Returns
    -------
    PismConfig
        Parsed and validated configuration model.
    """
    data = toml.loads(Path(path).read_text("utf-8"))
    return PismConfig.model_validate(data)


def load_uq(path: str | Path) -> UQConfig:
    """
    Load and validate a UQ configuration from a TOML file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the TOML file describing the uncertainty specification.
        The file may contain a top-level ``samples`` (int), optional
        ``mapping`` (str), and a nested or already-dotted set of leaves
        defining distributions.

    Returns
    -------
    UQConfig
        Parsed and validated configuration model.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    toml.TomlDecodeError
        If the file content is not valid TOML.
    pydantic.ValidationError
        If the parsed content fails model validation.

    Examples
    --------
    >>> uq = load_uq("uq/spec.toml")
    >>> uq.samples
    100
    >>> list(uq.to_flat().keys())[:2]
    ['surface.pdd.factor_ice', 'surface.pdd.factor_snow']
    """
    data = toml.loads(Path(path).read_text("utf-8"))
    return UQConfig.model_validate(data)


class DistSpec(BaseModel):
    """
    Distribution specification for a single random variable.

    This model captures the SciPy distribution name (``distribution``) and its
    parameters. For **continuous** distributions, ``loc`` and ``scale`` are
    typically required. For **discrete** distributions or special cases
    (e.g., ``randint``, ``truncnorm``), required parameters differ and are
    validated accordingly.

    Attributes
    ----------
    distribution : str
        Name of the SciPy distribution in ``scipy.stats`` (case-insensitive),
        e.g., ``"norm"``, ``"uniform"``, ``"truncnorm"``, ``"randint"``.
    loc : float or None, default: 0.0
        Location parameter for continuous distributions. Optional to allow
        discrete distributions that do not use it.
    scale : float or None, default: 1.0
        Scale parameter for continuous distributions. Must be ``> 0`` when
        applicable; optional for discrete distributions that do not use it.

    Notes
    -----
    This model validates *specification* only. Construction of a frozen SciPy
    RV (e.g., via ``stats.<dist>(...).ppf``) should be done by a helper such as
    ``_make_frozen`` that interprets ``model_extra`` as needed.
    """

    model_config = ConfigDict(extra="allow")

    # Optional so discrete dists (e.g., randint) can omit them cleanly
    loc: float | None = 0.0
    scale: float | None = 1.0
    distribution: str = Field(alias="distribution")

    @field_validator("distribution")
    @classmethod
    def _norm_name(cls, v: str) -> str:
        """
        Normalize the distribution name.

        Parameters
        ----------
        v : str
            Raw distribution name (any case, may contain surrounding spaces).

        Returns
        -------
        str
            Lower-cased, stripped distribution name suitable for
            ``getattr(scipy.stats, name)``.
        """
        return v.strip().lower()

    @model_validator(mode="after")
    def _check_params(self) -> "DistSpec":
        """
        Validate parameter completeness and basic constraints.

        Ensures that the distribution exists in ``scipy.stats`` and that the
        required parameters are present:

        - ``randint``: must provide ``low`` and ``high`` in ``model_extra``.
        - ``truncnorm``: must provide either standardized bounds ``a``, ``b`` or
          raw bounds ``lower``, ``upper`` and must have ``scale > 0``.
        - All other distributions: require ``loc`` and ``scale`` (with
          ``scale > 0``) and any shape parameters declared by
          ``scipy.stats.<dist>.shapes``.

        Returns
        -------
        DistSpec
            The validated instance (possibly unchanged).

        Raises
        ------
        ValueError
            If the distribution is unknown or any required parameters are
            missing/invalid.
        """

        name = self.distribution
        dist = getattr(st, name, None)
        if dist is None or not hasattr(dist, "rvs"):
            raise ValueError(f"unknown SciPy distribution: '{name}'")

        shapes = (dist.shapes or "").replace(" ", "")
        req_shapes = [s for s in shapes.split(",") if s]

        vals = dict(getattr(self, "model_extra", {}) or {})
        present = set(vals)

        # randint: require low, high; loc/scale are not used
        if name == "randint":
            if not {"low", "high"} <= present:
                raise ValueError("randint requires 'low' and 'high'")
            return self

        # truncnorm: require either (a,b) or (lower,upper) and valid scale
        if name == "truncnorm":
            if self.scale is None or float(self.scale) <= 0:
                raise ValueError("truncnorm requires 'scale' > 0")
            has_ab = {"a", "b"} <= present
            has_bounds = {"lower", "upper"} <= present
            if not (has_ab or has_bounds):
                raise ValueError("truncnorm requires 'a' and 'b' or 'lower' and 'upper'")
            return self

        # other distributions: require loc/scale and any shape params
        if self.loc is None or self.scale is None:
            raise ValueError(f"spec for '{name}' must include 'loc' and 'scale'")
        if float(self.scale) <= 0:
            raise ValueError(f"'scale' must be > 0 for '{name}'")

        if req_shapes:
            missing = [s for s in req_shapes if s not in present]
            if missing:
                raise ValueError(
                    f"distribution '{name}' requires shape parameter(s): "
                    f"{', '.join(req_shapes)} (missing: {', '.join(missing)})"
                )
        return self


class UQConfig(BaseModel):
    """
    Uncertainty specification as a flat dotted-key map with sampling metadata.

    This model holds:
      1) the number of ensemble ``samples`` (an integer),
      2) an optional column name ``mapping`` (e.g., to map categorical draws to
         filenames later), and
      3) a ``tree`` mapping from **dotted variable names** (e.g.,
         ``"surface.pdd.factor_ice"``) to :class:`DistSpec` entries that describe
         the probability distribution for each variable.

    Input TOML can be either *nested* (hierarchical tables) or already *flat*
    (quoted dotted-table keys). A ``model_validator(mode="before")`` flattens
    nested inputs by treating any dict that contains a ``"distribution"`` key as
    a *leaf* variable.

    Attributes
    ----------
    samples : int, default=1
        Number of draws to use when generating ensemble samples. Must be > 0.
    mapping : str or None, optional
        Optional column name indicating a mapping key (e.g., to join against a
        lookup table of file paths). Not interpreted by validation; simply
        preserved for downstream use.
    tree : dict[str, DistSpec]
        Flat mapping from dotted variable names to validated :class:`DistSpec`
        objects.

    Notes
    -----
    - The *before* validator accepts either:
        * top-level leaves (possibly nested tables), or
        * a block under the key ``"tree"``.
      In both cases, it extracts leaf specs and assigns them to ``tree``.
    - The presence of a key named ``"distribution"`` is used to detect leaves,
      deferring detailed parameter validation to :class:`DistSpec`.

    Examples
    --------
    Nested TOML (flattened automatically):

    .. code-block:: toml

        samples = 100

        [surface.pdd.factor_ice]
        distribution = "truncnorm"
        loc = 8
        scale = 4
        a = -1
        b =  1

    After parsing:

    >>> uq = UQConfig.model_validate(parsed_toml)
    >>> uq.samples
    100
    >>> uq.mapping
    'surface_choice'
    >>> list(uq.to_flat().keys())
    ['surface.pdd.factor_ice']
    """

    samples: int = Field(default=1, gt=0)
    mapping: dict | None = None
    tree: dict[str, "DistSpec"]  # values parsed as DistSpec after 'before' validator

    @staticmethod
    def _is_leaf(node: Any) -> bool:
        """
        Return ``True`` if a node looks like a distribution spec (a "leaf").

        A *leaf* is any mapping that declares a ``"distribution"`` key; all
        additional validation is delegated to :class:`DistSpec`.

        Parameters
        ----------
        node : Any
            Arbitrary value encountered during flattening.

        Returns
        -------
        bool
            ``True`` if ``node`` is a ``dict`` with a ``"distribution"`` key,
            ``False`` otherwise.

        Examples
        --------
        >>> UQConfig._is_leaf({"distribution": "norm", "loc": 0, "scale": 1})
        True
        >>> UQConfig._is_leaf({"foo": "bar"})
        False
        """
        # Be permissive: any dict with a 'distribution' key is a candidate leaf.
        return isinstance(node, dict) and ("distribution" in node)

    @classmethod
    def _flatten_leaves(cls, node: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
        """
        Recursively flatten nested dicts into dotted-key leaf specifications.

        Parameters
        ----------
        node : Any
            Root of a nested mapping (e.g., parsed TOML tables).
        prefix : str, optional
            Current dotted prefix accumulated during recursion.

        Returns
        -------
        dict[str, dict[str, Any]]
            A flat mapping from dotted names to raw (unvalidated) leaf dicts.

        Notes
        -----
        - Only entries recognized by :meth:`_is_leaf` are emitted as leaves.
        - Non-leaf dicts are traversed recursively.
        - Non-dict values are ignored.

        Examples
        --------
        >>> nested = {"surface": {"pdd": {"factor_ice": {"distribution": "norm"}}}}
        >>> UQConfig._flatten_leaves(nested)
        {'surface.pdd.factor_ice': {'distribution': 'norm'}}
        """
        flat: dict[str, dict[str, Any]] = {}
        if isinstance(node, dict):
            for k, v in node.items():
                key = f"{prefix}.{k}" if prefix else k
                if cls._is_leaf(v):
                    flat[key] = v
                elif isinstance(v, dict):
                    flat.update(cls._flatten_leaves(v, key))
        return flat

    @model_validator(mode="before")
    @classmethod
    def _flatten_input(cls, v: Any) -> dict[str, Any]:
        """
        Normalize input into ``{'samples': ..., 'mapping': ..., 'tree': {...}}``.

        Accepts either nested content (hierarchical TOML tables) or an already
        dotted flat mapping. Top-level ``samples`` and ``mapping`` are preserved
        (also accepted if provided inside the raw block).

        Parameters
        ----------
        v : Any
            Raw object to validate (typically a ``dict`` parsed from TOML).

        Returns
        -------
        dict
            A dict containing normalized keys:
            ``'tree'`` (flat leaf mapping) and optionally ``'samples'``,
            ``'mapping'``.

        Notes
        -----
        - If input contains a key ``"tree"``, that block is used as the source.
        - If any keys appear already dotted (contain ``'.'``), they are copied
          into ``tree`` if their values are ``dict``; otherwise nested flattening
          via :meth:`_flatten_leaves` is applied.

        Examples
        --------
        >>> raw = {"samples": 10, "surface": {"pdd": {"factor_ice": {"distribution": "norm"}}}}
        >>> out = UQConfig._flatten_input(raw)
        >>> sorted(out.keys())
        ['samples', 'tree']
        >>> out['tree'].keys()
        dict_keys(['surface.pdd.factor_ice'])
        """
        if not isinstance(v, dict):
            return v

        # Pull top-level fields if present
        samples = v.get("samples")
        mapping = v.get("mapping")

        # Where the specs live: either under 'tree' or at top level
        raw = v.get("tree", v)
        if not isinstance(raw, dict):
            outv: dict[str, Any] = {"tree": {}}
            if samples is not None:
                outv["samples"] = samples
            if mapping is not None:
                outv["mapping"] = mapping
            return outv

        raw = dict(raw)  # shallow copy so we can pop safely

        # Allow samples/mapping inside the raw block too
        if samples is None and "samples" in raw:
            samples = raw.pop("samples")
        if mapping is None and "mapping" in raw:
            mapping = raw.pop("mapping")

        # Ensure these don't leak into tree
        raw.pop("samples", None)
        raw.pop("mapping", None)

        # If keys are already dotted (['a.b.c']), keep only dict-valued items
        if any(isinstance(k, str) and "." in k for k in raw):
            tree = {k: v for k, v in raw.items() if isinstance(v, dict)}
        else:
            # Nested TOML tables -> flatten by finding leaves (by 'distribution' key)
            tree = cls._flatten_leaves(raw)

        out: dict[str, Any] = {"tree": tree}
        if samples is not None:
            out["samples"] = samples
        if mapping is not None:
            out["mapping"] = mapping
        return out

    # helpers
    def iter_specs(self) -> Iterator[tuple[str, "DistSpec"]]:
        """
        Iterate over variable specifications in the flat tree.

        Yields
        ------
        Iterator[tuple[str, DistSpec]]
            Pairs of ``(name, spec)``, where ``name`` is a dotted variable
            identifier and ``spec`` is the corresponding :class:`DistSpec`.

        Examples
        --------
        >>> for name, spec in uq.iter_specs():
        ...     print(name, spec.distribution)
        surface.pdd.factor_ice truncnorm
        """
        yield from self.tree.items()

    def to_flat(self) -> dict[str, dict[str, Any]]:
        """
        Export the flat mapping to plain dictionaries.

        Returns
        -------
        dict[str, dict[str, Any]]
            A mapping from dotted names to plain dicts created via
            ``DistSpec.model_dump()``. Useful for serialization or passing to
            sampling utilities.

        Examples
        --------
        >>> flat = uq.to_flat()
        >>> flat['surface.pdd.factor_ice']['distribution']
        'truncnorm'
        """
        # Values are DistSpec instances now; dump to plain dicts
        return {name: spec.model_dump() for name, spec in self.tree.items()}


class BaseModelWithDot(BaseModel):
    """
    Normalize dotted keys that may be redundantly prefixed with a section/table name.

    This base class lets Pydantic models accept inputs where field aliases such as
    ``"time.start"`` appear again prefixed inside a same-named table (e.g., under
    ``[time]`` you might see ``"time.time.start"``). The validator rewrites such
    keys to the exact alias expected by the model before normal field parsing.

    Attributes
    ----------
    SECTION : str or None
        The section/table name that may redundantly prefix keys (e.g., ``"time"``).
        Subclasses should set this to enable normalization.
    model_config : pydantic.ConfigDict
        Pydantic configuration for the model. Defaults to ``extra='ignore'``.

    Notes
    -----
    Normalization runs in a ``model_validator`` with ``mode='before'``. For each
    declared field, the following input keys are accepted and mapped to the field's
    alias (or the field name if no alias is set):

    * ``<alias>`` (exact), e.g., ``"time.start"``.
    * ``<SECTION>.<alias>``, e.g., ``"time.time.start"``.
    * ``<SECTION>.<field_name>`` (fallback when no alias is set).

    This enables TOML like:

    .. code-block:: toml

        [time]
        'time.start' = "1980-01-01"
        'time.end'   = "1990-01-01"

    while keeping Pydantic fields with dotted aliases.

    Examples
    --------
    >>> class TimeConfig(BaseModelWithDot):
    ...     SECTION = "time"
    ...     time_start: str = Field(alias="time.start")
    ...     time_end: str   = Field(alias="time.end")
    ...
    >>> cfg = TimeConfig.model_validate({
    ...     "time.time.start": "1980-01-01",
    ...     "time.time.end":   "1990-01-01",
    ... })
    >>> cfg.time_start, cfg.time_end
    ('1980-01-01', '1990-01-01')
    """

    model_config = ConfigDict(extra="ignore")
    SECTION: ClassVar[str | None] = None  # e.g. "time"

    @model_validator(mode="before")
    @classmethod
    def _normalize_dotted_keys(cls, v: Any) -> Any:
        """
        Rewrite redundantly prefixed dotted keys to the model's expected aliases.

        Parameters
        ----------
        v : dict or Any
            Incoming value for validation. If a mapping, keys like
            ``f"{SECTION}.{alias}"`` or ``f"{SECTION}.{field_name}"`` are
            detected and moved to the canonical key (``alias`` or field name).

        Returns
        -------
        dict or Any
            The normalized mapping if ``v`` is a dict; otherwise returns ``v``
            unchanged.
        """
        if not isinstance(v, dict):
            return v
        if cls.SECTION is None:
            return v

        out = dict(v)  # shallow copy
        sec = cls.SECTION

        # For each declared field, ensure its alias is present; if not,
        # look for a redundantly-prefixed variant like "<SECTION>.<alias>".
        for fname, field in cls.model_fields.items():
            target = field.alias or fname  # what Pydantic expects in the input
            if target in out:
                continue  # already good

            # Candidate alternative keys we accept:
            candidates = [f"{sec}.{target}", f"{sec}.{fname}"]
            for cand in candidates:
                if cand in out:
                    out[target] = out.pop(cand)
                    break

        return out


class RunConfig(BaseModel):
    """
    Execution settings for a PISM run.

    Provides executable/launcher options and a helper to export parameters
    for templating. String fields that contain Jinja expressions (e.g.,
    ``"mpirun -np {{ ntasks }}"``) are rendered using the model values.

    Attributes
    ----------
    mpi : str
        MPI launcher template, e.g., ``"mpirun -np {{ ntasks }}"``.
        Defaults to ``"mpirun"``.
    executable : str
        Path to the PISM executable, or command name. Defaults to ``"pism"``.
    ntasks : int
        Total number of MPI ranks. Must be >= 1.

    Notes
    -----
    The :meth:`as_params` method returns only non-empty fields and renders any
    string value containing Jinja delimiters ``{{ ... }}`` using the current
    field values (plus any extra context provided).

    Examples
    --------
    >>> rc = RunConfig(mpi="mpirun -np {{ ntasks }}", executable="/path/pism", ntasks=56)
    >>> rc.as_params()["mpi"]
    'mpirun -np 56'
    """

    mpi: str = Field(default="mpirun")
    executable: str = Field(default="pism")
    ntasks: int = Field(ge=1)

    def as_params(self, **extra: Any) -> dict[str, Any]:
        """
        Export non-empty parameters and render templated strings.

        Any string field containing Jinja expressions is rendered using a
        context composed of the model's own values plus ``extra``.

        Parameters
        ----------
        **extra
            Additional key/value pairs to inject into the Jinja render
            context (these do not mutate the model).

        Returns
        -------
        dict of str to Any
            Dictionary of parameters suitable for template rendering.
            Fields with ``None``/unset/default values are omitted; templated
            strings (e.g., ``mpi``) are rendered to plain strings.
        """
        params = self.model_dump(exclude_none=True, exclude_unset=True, exclude_defaults=True)
        ctx = {**params, **extra}

        def _render(v: Any) -> Any:
            """
            Render templated strings using the current context.

            Parameters
            ----------
            v : Any
                Candidate value to render. If `v` is a string containing Jinja
                delimiters (``{{ ... }}``), it is rendered using the closure
                context ``ctx``; otherwise it is returned unchanged.

            Returns
            -------
            Any
                The rendered string when `v` is a templated string; otherwise the
                original value.

            Raises
            ------
            jinja2.UndefinedError
                If the template references an undefined variable and the Jinja
                environment uses ``StrictUndefined``.
            """
            if isinstance(v, str) and "{{" in v:
                return _JINJA.from_string(v).render(ctx)
            return v

        return {k: _render(v) for k, v in params.items()}


class JobConfig(BaseModelWithDot):
    """
    Scheduler job options parsed from configuration.

    Accepts dotted keys via :class:`BaseModelWithDot` and provides a compact
    export for templating.

    Attributes
    ----------
    queue : str or None
        Scheduler queue/partition name.
    walltime : str or None
        Wall clock limit in ``HH:MM:SS`` (or ``H:MM:SS``) format.
    nodes : int or None
        Number of compute nodes to request (>= 1).

    Notes
    -----
    Unknown/extra keys are forbidden (``extra='forbid'``). Use
    :meth:`as_params` to obtain only present, non-empty keys.

    Examples
    --------
    >>> jc = JobConfig(queue="normal", walltime="04:00:00", nodes=2)
    >>> jc.as_params()
    {'queue': 'normal', 'walltime': '04:00:00', 'nodes': 2}
    """

    model_config = ConfigDict(extra="forbid")

    queue: str | None = None
    walltime: str | None = None
    nodes: int | None = Field(default=None, ge=1)

    @field_validator("walltime")
    @classmethod
    def _hhmmss(cls, v: str | None) -> str | None:
        """
        Validate that ``walltime`` matches ``H:MM:SS`` or ``HH:MM:SS``.

        Parameters
        ----------
        v : str or None
            Candidate walltime string.

        Returns
        -------
        str or None
            The original value if valid; ``None`` is passed through.

        Raises
        ------
        ValueError
            If the value does not match the expected format.
        """
        if v is None:
            return v
        if not re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", v):
            raise ValueError("walltime must look like HH:MM:SS")
        return v

    def as_params(self) -> dict[str, Any]:
        """
        Export only non-empty job parameters for templating.

        Returns
        -------
        dict of str to Any
            A dictionary containing only keys whose values are present
            (i.e., excludes ``None``, unset, and defaulted fields).
        """
        return self.model_dump(exclude_none=True, exclude_unset=True, exclude_defaults=True)


class PismConfig(BaseModelWithDot):
    """
    Top-level configuration model for a PISM run.

    This model aggregates validated sub-sections (execution, job/scheduler,
    time, physics choices, and grid) together with several pass-through
    dictionaries that hold option maps forwarded directly to PISM. Because
    it derives from :class:`BaseModelWithDot`, inputs may use dotted keys
    that are redundantly prefixed by their table/section names.

    Attributes
    ----------
    run : RunConfig
        Execution settings (launcher template, executable path/name,
        number of MPI ranks) with support for rendering Jinja placeholders.
    job : JobConfig
        Scheduler options such as queue/partition, walltime, and number of
        nodes. Unknown keys are forbidden in this section.
    time : TimeConfig
        Time configuration (e.g., ``time.start``, ``time.end``, calendar).
    energy : EnergyConfig
        Energy model selection and its option set (e.g., ``energy.model`` and
        ``energy.options[model]``).
    stress_balance : StressBalanceConfig
        Stress-balance model selection and its option set (e.g.,
        ``stress_balance.model`` and ``stress_balance.options[model]``).
    grid : GridConfig
        Horizontal/vertical grid settings and registration. Derives
        ``grid.dx``/``grid.dy`` from ``resolution`` when not explicitly set.
    atmosphere : dict of str to Any, optional
        Additional atmosphere-related options to pass through (keys are
        typically dotted, e.g., ``"atmosphere.given.file"``). Defaults to ``{}``.
    geometry : dict of str to Any, optional
        Geometry-related options to pass through. Defaults to ``{}``.
    ocean : dict of str to Any, optional
        Ocean-related options to pass through. Defaults to ``{}``.
    calving : dict of str to Any, optional
        Calving-related options to pass through. Defaults to ``{}``.
    iceflow : dict of str to Any, optional
        Ice-flow-related options to pass through. Defaults to ``{}``.
    surface : dict of str to Any, optional
        Surface-related options to pass through. Defaults to ``{}``.
    reporting : dict of str to Any, optional
        Reporting/output options to pass through. Defaults to ``{}``.
    input : dict of str to Any, optional
        Input file options to pass through. Defaults to ``{}``.

    Notes
    -----
    * Dotted option keys inside nested tables are accepted due to
      :class:`BaseModelWithDot` (e.g., inside ``[time]`` you may still use
      ``'time.start'``).
    * The dictionary sections (``atmosphere``, ``geometry``, …, ``input``)
      are intentionally permissive and are stored as-is to be forwarded to
      PISM; validation of their contents is out of scope for this model.

    Examples
    --------
    >>> cfg = PismConfig.model_validate(data)  # 'data' parsed from TOML
    >>> cfg.run.ntasks
    56
    >>> cfg.grid.as_run()["grid.dx"]
    '200m'
    """

    run: RunConfig
    run_info: InfoConfig
    job: JobConfig
    time: TimeConfig
    energy: EnergyConfig
    stress_balance: StressBalanceConfig
    grid: GridConfig
    atmosphere: AtmosphereConfig
    surface: SurfaceConfig
    geometry: dict[str, Any] = {}
    ocean: dict[str, Any] = {}
    calving: dict[str, Any] = {}
    iceflow: dict[str, Any] = {}
    reporting: dict[str, Any] = {}
    input: dict[str, Any] = {}
    time_stepping: dict[str, Any] = {}


class RestartConfig(BaseModelWithDot):
    """
    Info settings for restarting.

    Accepts dotted keys inside the ``[restart]`` table (via BaseModelWithDot)
    and exports params with *quoted* string values for use in templates.
    """

    SECTION = "restart"
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class InfoConfig(BaseModelWithDot):
    """
    Info settings for run metadata.

    Accepts dotted keys inside the ``[run_info]`` table (via BaseModelWithDot)
    and exports params with *quoted* string values for use in templates.
    """

    SECTION = "run_info"
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    institution: str = Field(
        default="University of Alaska Fairbanks",
        alias="run_info.institution",
    )
    title: str = Field(
        default="PISM Campaign",
        alias="run_info.title",
    )

    @staticmethod
    def _quote(v: Any) -> str:
        """
        Return ``v`` as a double-quoted string, escaping inner quotes/backslashes.

        Parameters
        ----------
        v : str
            String to quote.

        Returns
        ----------
        str
            String in quotes.
        """
        s = str(v)
        # strip existing wrapping quotes to avoid double-quoting
        if len(s) >= 2 and s[0] == s[-1] and s[0] in {"'", '"'}:
            s = s[1:-1]
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'

    def as_params(self) -> dict[str, Any]:
        """
        Export run-info parameters with dotted aliases and quoted string values.

        Returns
        -------
        dict[str, Any]
            Dictionary like ``{'run_info.institution': '\"Foo\"', 'run_info.title': '\"Bar\"'}``.
        """
        out = self.model_dump(by_alias=True, exclude_none=True)
        for key in ("run_info.institution", "run_info.title"):
            if key in out and out[key] is not None:
                out[key] = self._quote(out[key])
        return out


class GridConfig(BaseModelWithDot):
    """
    Grid settings.
    """

    SECTION = "grid"
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    # what your TOML shows:
    resolution: str = Field(alias="resolution")  # e.g. "200m" (also accepts "grid.resolution")
    Lbz: int | None = Field(default=None, alias="grid.Lbz")
    Lz: int | float | None = Field(default=None, alias="grid.Lz")
    Mbz: int | None = Field(default=None, alias="grid.Mbz")
    Mz: int | None = Field(default=None, alias="grid.Mz")
    registration: str | None = Field(default=None, alias="grid.registration")

    # derived / optionally provided:
    dx: str | None = Field(default=None, alias="grid.dx")
    dy: str | None = Field(default=None, alias="grid.dy")

    @field_validator("resolution")
    @classmethod
    def _check_resolution(cls, v: str) -> str:
        """
        Validate and normalize the grid resolution string.

        Parameters
        ----------
        v : str
            Resolution value, e.g., ``"200m"``, ``"1500 m"``, ``"0.2km"``, or ``"1 km"``.
            Optional whitespace is allowed; units must be meters (``m``) or kilometers (``km``).

        Returns
        -------
        str
            Normalized resolution with whitespace removed (e.g., ``"200m"``).

        Raises
        ------
        ValueError
            If the value does not match the expected ``<number><unit>`` pattern,
            where ``unit`` is ``m`` or ``km`` (e.g., ``"200m"``, ``"0.2km"``).

        Notes
        -----
        The numeric part may be an integer or a decimal. Only ``m`` and ``km`` are accepted units.
        """
        # accept "200m", "0.2km", "1500 m", "1 km" (with optional space)
        if not re.fullmatch(r"\s*\d+(\.\d+)?\s*(m|km)\s*", v):
            raise ValueError('resolution must look like "200m" or "0.2km"')
        return re.sub(r"\s+", "", v)  # normalize by removing spaces

    @field_validator("registration")
    @classmethod
    def _check_registration(cls, v: str | None) -> str | None:
        """
        Validate and normalize ``grid.registration``.

        Parameters
        ----------
        v : str or None
            Candidate registration string.

        Returns
        -------
        str or None
            Lowercased normalized value (``"center"`` or ``"corner"``) or
            ``None`` if not provided.

        Raises
        ------
        ValueError
            If ``v`` is not one of ``{"center", "corner"}``.

        Notes
        -----
        Leading/trailing whitespace is stripped and the value is lowercased
        before validation.
        """
        if v is None:
            return v
        v_norm = v.strip().lower()
        if v_norm not in {"center", "corner"}:
            raise ValueError('grid.registration must be "center" or "corner"')
        return v_norm

    @model_validator(mode="after")
    def _derive_dx_dy(self) -> "GridConfig":
        """
        Derive ``grid.dx`` and ``grid.dy`` from ``resolution`` when absent.

        Returns
        -------
        GridConfig
            The same instance with ``dx`` and/or ``dy`` populated if they
            were ``None``.

        Notes
        -----
        - If either ``dx`` or ``dy`` is missing, it is set to the normalized
          ``resolution`` string (e.g., ``"200m"``).
        - Existing values for ``dx``/``dy`` are left unchanged.
        """
        if self.dx is None:
            self.dx = self.resolution
        if self.dy is None:
            self.dy = self.resolution
        return self

    def as_params(self) -> dict[str, Any]:
        """
        Export grid parameters as dotted keys suitable for runtime.

        Ensures ``'grid.dx'`` and ``'grid.dy'`` are present (derived from
        ``resolution`` if necessary) and omits the plain ``resolution`` key.

        Returns
        -------
        dict[str, Any]
            Dictionary with dotted aliases (e.g., ``'grid.Mz'``, ``'grid.dx'``,
            ``'grid.dy'``) and no ``None`` values. The ``'resolution'`` key is
            removed from the output.

        Examples
        --------
        >>> g = GridConfig(resolution="200m", Mz=101)
        >>> g.as_params()["grid.dx"], g.as_params()["grid.dy"]
        ('200m', '200m')
        """
        out = self.model_dump(by_alias=True, exclude_none=True)
        out["grid.dx"] = self.dx or self.resolution
        out["grid.dy"] = self.dy or self.resolution
        del out["resolution"]
        return out


class TimeConfig(BaseModelWithDot):
    """
    Time configuration (accepts dotted keys inside the ``[time]`` table).

    Because this inherits from :class:`BaseModelWithDot` with
    ``SECTION = "time"``, inputs may redundantly prefix keys with ``"time."``
    (e.g., ``"time.time.start"``) and they will be normalized to the aliases
    declared below.

    Attributes
    ----------
    time_start : str
        Simulation start time (alias: ``"time.start"``).
    time_end : str
        Simulation end time (alias: ``"time.end"``).
    calendar : str or None
        Calendar name (alias: ``"time.calendar"``), e.g., ``"standard"``.
    reference_date : str or None
        Reference date (alias: ``"time.reference_date"``).
    """

    SECTION = "time"
    model_config = ConfigDict(populate_by_name=True)

    time_start: str = Field(alias="time.start")
    time_end: str = Field(alias="time.end")
    calendar: str | None = Field(default=None, alias="time.calendar")
    reference_date: str | None = Field(default=None, alias="time.reference_date")

    def as_params(self) -> dict[str, Any]:
        """
        Export time parameters using dotted aliases.

        Returns
        -------
        dict of str to Any
            Dictionary with aliases (e.g., ``"time.start"``) and no ``None`` values.
        """
        return self.model_dump(by_alias=True, exclude_none=True)


class ModelWithOptions(BaseModelWithDot):
    """
    Generic "pick a model and its option set" section.

    This shape is used by sub-sections like energy and stress balance that
    specify a chosen ``model`` and an ``options`` mapping keyed by model name.

    Attributes
    ----------
    model : str
        Name of the selected sub-model (e.g., ``"ssa"``).
    options : dict of str to dict
        Mapping from model name to that model's option dictionary.

    Notes
    -----
    Inheriting from :class:`BaseModelWithDot` allows accepting dotted keys
    if the subclass sets ``SECTION`` appropriately.
    """

    model: str
    options: dict[str, dict[str, Any]]

    def selected(self) -> dict[str, Any]:
        """
        Return the option dictionary for the selected model.

        Returns
        -------
        dict
            Options corresponding to ``self.model``, e.g., ``options[model]``.

        Raises
        ------
        ValueError
            If ``self.model`` is not present in ``options``.
        """
        try:
            return self.options[self.model]
        except KeyError as e:
            raise ValueError(f"model '{self.model}' not in options") from e


class AtmosphereConfig(ModelWithOptions):
    """
    Atmosphere model configuration.

    Inherits fields/behavior from :class:`ModelWithOptions`.
    """

    SECTION = "atmosphere"


class SurfaceConfig(ModelWithOptions):
    """
    Surface model configuration.

    Inherits fields/behavior from :class:`ModelWithOptions`.
    """

    SECTION = "surface"


class EnergyConfig(ModelWithOptions):
    """
    Energy model configuration.

    Inherits fields/behavior from :class:`ModelWithOptions`.
    """

    SECTION = "energy"


class StressBalanceConfig(ModelWithOptions):
    """
    Stress-balance model configuration.

    Inherits fields/behavior from :class:`ModelWithOptions`.
    """

    SECTION = "stress_balance"
