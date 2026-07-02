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
"""
Sampling methods.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc, rv_discrete


def _make_frozen(dist_name: str, spec: dict[str, Any]):
    """
    Build a frozen SciPy distribution from a spec dict.

    Parameters
    ----------
    dist_name : str
        Name of the SciPy distribution in `scipy.stats`.
    spec : dict
        Parameter spec, including `distribution`, possibly shape params, and
        `loc`/`scale` if applicable.

    Returns
    -------
    scipy.stats.rv_frozen
        Frozen SciPy distribution ready for `.ppf` / `.rvs`.

    Notes
    -----
    - Uses SciPy's `shapes` string to determine required shape params and
      collects them from `spec` by name (e.g., 'a', 'b', 'n', 'p', 'mu', 's').
    - For `truncnorm`, accepts either standardized `a,b` or raw `lower,upper`
      (converted to `a,b` using `loc` and `scale`).
    - For `randint`, requires `low` and `high`; accepts optional `loc` and
      **does not pass** `scale` (SciPy discrete dists don't accept it).
    - For other **discrete** dists (e.g., `binom`, `poisson`, `geom`, …),
      passes only `loc` (if provided); `scale` is not used.
    - For **continuous** dists, requires `loc` and `scale` and passes both.
    """
    dist_name = dist_name.strip().lower()
    dist = getattr(stats, dist_name)  # validated earlier

    # Parse shape parameter names (comma-separated or None)
    shapes = (dist.shapes or "").replace(" ", "")
    shape_names: list[str] = [s for s in shapes.split(",") if s]

    # Collect shape args by name (populate later, including special-cases)
    args: list[float] = []

    # Special-case: truncnorm (allow lower/upper as convenience)
    if dist_name == "truncnorm":
        # Must have scale > 0 and loc defined (default loc=0 if omitted)
        loc = float(spec.get("loc", 0.0))
        scale = float(spec.get("scale", 1.0))
        if scale <= 0:
            raise ValueError("truncnorm requires 'scale' > 0")

        if {"a", "b"} <= set(spec):
            a = float(spec["a"])
            b = float(spec["b"])
        elif {"lower", "upper"} <= set(spec):
            lower = float(spec["lower"])
            upper = float(spec["upper"])
            if lower >= upper:
                raise ValueError("truncnorm requires lower < upper")
            a = (lower - loc) / scale
            b = (upper - loc) / scale
        else:
            raise ValueError("truncnorm requires a,b or lower,upper")

        return dist(a, b, loc=loc, scale=scale)

    # Special-case: randint (requires low, high; loc optional; no scale)
    if dist_name == "randint":
        if not {"low", "high"} <= set(spec):
            raise ValueError("randint requires 'low' and 'high'")
        low = int(spec["low"])
        high = int(spec["high"])
        loc = int(spec.get("loc", 0))
        # Ignore/forbid scale for randint (not supported)
        if "scale" in spec and float(spec["scale"]) != 1.0:
            raise ValueError("randint does not support 'scale'; remove it or set to 1")
        return dist(low, high, loc=loc)

    # Generic handling: gather shape parameters if required
    if shape_names:
        missing = [n for n in shape_names if n not in spec]
        if missing:
            need = ", ".join(shape_names)
            raise ValueError(
                f"distribution '{dist_name}' requires shape parameter(s): {need}; " f"missing: {', '.join(missing)}"
            )
        args = [float(spec[n]) for n in shape_names]

    is_discrete = isinstance(dist, rv_discrete)

    # Discrete: pass only loc (no scale)
    if is_discrete:
        # loc is optional for most dists; default 0 if absent
        loc = float(spec.get("loc", 0.0))
        return dist(*args, loc=loc)

    # Continuous: require loc & scale
    if spec.get("loc", None) is None or spec.get("scale", None) is None:
        raise ValueError(f"spec for '{dist_name}' must include 'loc' and 'scale'")
    loc = float(spec["loc"])
    scale = float(spec["scale"])
    if scale <= 0:
        raise ValueError(f"'scale' must be > 0 for '{dist_name}'")

    return dist(*args, loc=loc, scale=scale)


def _transform_quantiles(U: np.ndarray, d: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """
    Map unit-cube quantiles to parameter values via each variable's inverse CDF.

    Shared back end for :func:`create_samples` (LHS quantiles) and
    :func:`create_grid_samples` (regular-grid quantiles): both produce a matrix
    ``U`` of quantiles in ``[0, 1]`` and differ only in how those quantiles are
    laid out.

    Parameters
    ----------
    U : numpy.ndarray
        Array of shape ``(n, len(d))`` of quantiles in ``[0, 1]``; column ``i``
        corresponds to the ``i``-th variable in ``d`` (insertion order).
    d : dict
        Mapping ``name -> spec`` where spec includes at least ``distribution``
        and any required parameters.

    Returns
    -------
    pandas.DataFrame
        One column per variable in ``d`` plus an integer ``sample`` column.
        Discrete variables are returned as integer dtype.
    """
    names = list(d.keys())

    # clip away exactly 0 or 1 so ppf is well-defined (ppf(1) is +inf for
    # unbounded distributions)
    eps = np.finfo(float).eps  # pylint: disable=no-member
    U = np.clip(U, eps, 1 - eps)

    # Transform each dimension with that variable's inverse CDF (PPF)
    X = np.empty_like(U, dtype=float)
    discrete_cols: list[int] = []
    for i, name in enumerate(names):
        spec = d[name]
        dist_name = str(spec["distribution"]).strip().lower()
        frozen = _make_frozen(dist_name, spec)

        # mark discrete columns so we can cast to int later
        dist_gen = getattr(stats, dist_name, None)
        if isinstance(dist_gen, rv_discrete):
            discrete_cols.append(i)

        X[:, i] = frozen.ppf(U[:, i])

    # Build DataFrame
    df = pd.DataFrame(X, columns=names)
    # Cast discrete columns to integers
    for i in discrete_cols:
        col = names[i]
        # PPF of discrete dists returns integer-valued floats; round just in case
        df[col] = df[col].round().astype("int64")

    # add sample id
    df.insert(0, "sample", np.arange(len(df), dtype=int))
    return df


def create_samples(d: dict[str, dict[str, Any]], n_samples: int = 10, seed: int | None = None) -> pd.DataFrame:
    """
    Draw Latin Hypercube samples and transform by the specified SciPy distributions.

    Parameters
    ----------
    d : dict
        Mapping ``name -> spec`` where spec includes at least ``distribution`` and any
        required parameters (e.g., ``low, high`` for ``randint`` or ``loc, scale`` for
        continuous distributions; plus any shapes like ``a, b`` for ``truncnorm``).
    n_samples : int, default 10
        Number of samples to draw.
    seed : int or None, default None
        Seed for the Latin Hypercube sampler.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one column per variable in ``d`` and an
        integer ``sample`` column. Discrete variables are returned as integer dtype.
    """
    names = list(d.keys())

    # LHS in [0,1]
    engine = qmc.LatinHypercube(d=len(names), seed=seed)
    U = engine.random(n_samples)

    return _transform_quantiles(U, d)


def create_grid_samples(d: dict[str, dict[str, Any]], n_levels: int = 10, kind: str = "edge") -> pd.DataFrame:
    """
    Build a full-factorial (regular grid) design and transform by the SciPy distributions.

    Unlike :func:`create_samples` (a randomized Latin Hypercube), this lays each
    variable on an equidistant grid of quantiles and takes the **Cartesian
    product** across variables, so the result has ``n_levels ** len(d)`` rows.
    For a single variable this is just ``n_levels`` equidistant draws.

    Parameters
    ----------
    d : dict
        Mapping ``name -> spec`` (same structure as :func:`create_samples`).
    n_levels : int, default 10
        Number of equidistant levels **per variable**. Total rows are
        ``n_levels ** len(d)``.
    kind : {"edge", "midpoint", "endpoint"}, default "edge"
        Quantile convention for the per-variable grid of ``n`` levels:

        - ``"edge"``: ``i / n`` for ``i = 1..n`` (e.g. ``1/3, 2/3, 3/3``). Reaches
          the distribution's upper bound; for *unbounded* distributions the top
          level is an extreme (clipped) value, so prefer ``"midpoint"`` there.
        - ``"midpoint"``: ``(i + 0.5) / n`` for ``i = 0..n-1`` (e.g.
          ``1/6, 1/2, 5/6``). Symmetric; never hits a bound.
        - ``"endpoint"``: ``linspace(0, 1, n)`` (e.g. ``0, 1/2, 1``). Includes
          both bounds.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one column per variable in ``d`` and an integer
        ``sample`` column. Discrete variables are returned as integer dtype.

    Raises
    ------
    ValueError
        If ``n_levels < 1`` or ``kind`` is not recognized.
    """
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")

    names = list(d.keys())

    if kind == "edge":
        q = np.arange(1, n_levels + 1) / n_levels
    elif kind == "midpoint":
        q = (np.arange(n_levels) + 0.5) / n_levels
    elif kind == "endpoint":
        q = np.linspace(0.0, 1.0, n_levels)
    else:
        raise ValueError(f"unknown kind '{kind}'; use 'edge', 'midpoint', or 'endpoint'")

    # Cartesian product of the per-variable grids -> (n_levels ** d, d)
    grids = np.meshgrid(*([q] * len(names)), indexing="ij")
    U = np.stack([g.ravel() for g in grids], axis=1)

    return _transform_quantiles(U, d)


def generate_samples(
    d: dict[str, dict[str, Any]],
    n_samples: int = 10,
    method: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Dispatch to a sampling strategy by name.

    Parameters
    ----------
    d : dict
        Mapping ``name -> spec`` (same structure as :func:`create_samples`).
    n_samples : int, default 10
        For ``method="lhs"`` the total number of draws; for ``method="factorial"``
        the number of equidistant levels **per variable** (total rows are
        ``n_samples ** len(d)``).
    method : str or None, default None
        Sampling strategy. ``None`` or ``"lhs"`` (and aliases) selects the
        Latin Hypercube; ``"factorial"`` (and aliases ``"grid"``,
        ``"full_factorial"``) selects the full-factorial grid. Case-insensitive.
    seed : int or None, default None
        Seed for the Latin Hypercube sampler (ignored for the deterministic grid).

    Returns
    -------
    pandas.DataFrame
        Sample table as returned by the selected sampler.

    Raises
    ------
    ValueError
        If ``method`` is not recognized.
    """
    m = (method or "lhs").strip().lower()
    if m in {"lhs", "latin", "latin_hypercube", "latinhypercube"}:
        return create_samples(d, n_samples=n_samples, seed=seed)
    if m in {"factorial", "grid", "full_factorial"}:
        return create_grid_samples(d, n_levels=n_samples)
    raise ValueError(f"unknown sampling method '{method}'; use 'lhs' or 'factorial'")
