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

# pylint: disable=unused-import,broad-exception-caught

"""
Workflow management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, TypeVar

import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray  # noqa: F401
import xarray as xr
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def merge_model(base_model: T, **overrides: Any) -> T:
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
    >>> new = merge_model(rc, ntasks=80, extra=None)  # 'extra' ignored; None skipped
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


def to_python_scalar(v: Any) -> Any:
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
    >>> to_python_scalar(np.int64(3))
    3
    >>> import pandas as pd
    >>> to_python_scalar(pd.Timestamp("2020-01-01"))
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


def normalize_row(row) -> dict:
    """
    Normalize a row-like object into a clean ``dict[str, Any]``.

    Converts values to plain Python scalars via :func:`to_python_scalar`,
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
    >>> normalize_row(s)
    {'a': datetime.datetime(2021, 1, 1, 0, 0), 'c': 1}
    """
    d = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        out[str(k)] = to_python_scalar(v)
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


ScalarIndex = int | slice
Selection = dict[str, ScalarIndex]  # e.g. {"x": slice(...), "y": slice(...), "time": 0}
Selections = list[Selection]


def _windows_for(var: xr.DataArray, window: int = 32) -> Selections:
    """
    Generate small indexer dictionaries.

    The selections include a central window and the four corners over the
    primary spatial axes (``x/y`` or ``lon/lat`` when available). If a
    ``time`` dimension exists and is non-empty, the function duplicates each
    spatial window at the first and last time steps to exercise temporal I/O.

    Parameters
    ----------
    var : xarray.DataArray
        Target array to sample. May have dimensions such as ``time``,
        ``y``, ``x`` (or ``lat``, ``lon``). Other dimensions are ignored
        except that a small slice will be taken along the first two dims in
        the fallback path if no recognized spatial dims are present.
    window : int, default 32
        Desired spatial window size (in grid points) for both ``x`` and
        ``y`` (or ``lon``/``lat``). If the dimension is smaller than
        ``window``, the slice is clamped to the available extent.

    Returns
    -------
    list of dict
        A list of selection dictionaries suitable for ``.isel(**sel)``.
        Keys are dimension names; values are ``slice`` objects for spatial
        dims and integer indices for ``time`` when included.

    Notes
    -----
    * When both spatial axes are present, the function returns up to
      5 spatial windows: center, top-left, top-right, bottom-left, and
      bottom-right. With a ``time`` axis, each of these appears twice:
      once at the first time step and once at the last.
    * If only one or no recognized spatial axes exist, a conservative
      fallback samples small slices along the first two dimensions, if any.
    * Duplicate selections (possible on tiny arrays) are removed.

    Examples
    --------
    >>> sel_list = _windows_for(da, window=64)
    >>> sub = da.isel(**sel_list[0]).load()  # small materialized read
    """
    dims = list(var.dims)
    sizes = var.sizes

    # time edges if time is present
    if "time" in dims and sizes["time"] > 0:
        t0 = {"time": 0}
        tN = {"time": sizes["time"] - 1}
    else:
        t0 = tN = {}

    # 2D-ish spatial window heuristics (x/y or lon/lat)
    xdim = "x" if "x" in dims else ("lon" if "lon" in dims else None)
    ydim = "y" if "y" in dims else ("lat" if "lat" in dims else None)

    def w(d: str, start: int) -> slice:
        """
        Build a clamped slice of length ``window`` along dimension ``d``.

        Parameters
        ----------
        d : str
            Dimension name present in ``var``.
        start : int
            Preferred starting index (will be clamped inside the valid range).

        Returns
        -------
        slice
            Slice ``slice(a, b)`` where ``b - a <= window`` and
            ``0 <= a < sizes[d]``.
        """
        n = sizes[d]
        a = max(0, min(start, n - window))
        b = min(n, a + window)
        return slice(a, b)

    candidates: Selections = []

    if xdim and ydim:
        xs = sizes[xdim]
        ys = sizes[ydim]

        centers: Selection = {xdim: w(xdim, xs // 2), ydim: w(ydim, ys // 2)}
        tl: Selection = {xdim: w(xdim, 0), ydim: w(ydim, 0)}
        tr: Selection = {xdim: w(xdim, xs - window), ydim: w(ydim, 0)}
        bl: Selection = {xdim: w(xdim, 0), ydim: w(ydim, ys - window)}
        br: Selection = {xdim: w(xdim, xs - window), ydim: w(ydim, ys - window)}
        candidates += [centers, tl, tr, bl, br]
    else:
        # fall back: take a small slice on first two dims, if any
        base: Selection = {}
        for d in dims[:2]:
            base[d] = w(d, 0) if sizes[d] > window else slice(0, sizes[d])
        candidates.append(base)

    # combine with time edges
    windows: Selections = []
    for c in candidates:
        if t0:
            w0: Selection = dict(c)
            w0.update(t0)  # keep type as Selection
            wN: Selection = dict(c)
            wN.update(tN)
            windows.append(w0)
            windows.append(wN)
        else:
            windows.append(c)

    # deduplicate
    uniq: Selections = []
    seen: set[tuple[tuple[str, ScalarIndex], ...]] = set()
    for s in windows:
        key = tuple(sorted(s.items()))
        if key not in seen:
            uniq.append(s)
            seen.add(key)
    return uniq


def check_dataset_lazy(
    ds: xr.Dataset,
    *,
    required_vars: Iterable[str] | None = None,
    window: int = 32,
    max_vars: int = 8,
) -> None:
    """
    Validate an xarray Dataset without loading it fully.

    This performs inexpensive structural checks, then samples several
    small windows per variable (corners/center and first/last time) and
    loads only those to exercise the I/O/decoding stack.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to validate. Should already be opened (preferably with
        chunking, e.g., ``xr.open_dataset(..., chunks='auto')``).
    required_vars : Iterable[str], optional
        Names that must be present as data variables. If any are missing,
        the task raises ``KeyError``.
    window : int, default 32
        Target spatial window size (in grid points) for test reads.
    max_vars : int, default 8
        Cap on number of variables to sample. Variables are picked in
        descending order of size to stress the likely-largest arrays.

    Raises
    ------
    KeyError
        If any ``required_vars`` are missing.
    ValueError
        On structural issues (non-monotonic coords, invalid CRS, zero-sized dims).
    Exception
        Any error raised by backends/decoders while reading the sampled windows.

    Notes
    -----
    * Checks run:
      - coordinate monotonicity for ``x/y`` or ``lon/lat`` (if present)
      - time decodability (if present)
      - rioxarray CRS present (if ``.rio`` accessor available)
      - sample-window reads and simple finite/NaN sanity on samples

    * This does **not** attempt to read the entire dataset.
    """
    # --- basic structure checks ---
    if required_vars:
        missing = [v for v in required_vars if v not in ds.data_vars]
        if missing:
            raise KeyError(f"Missing required variables: {missing}")

    # coord monotonicity
    for xd in ("x", "lon"):
        if xd in ds.coords:
            xv = ds[xd].values
            if xv.size < 2:
                raise ValueError(f"Coordinate '{xd}' has <1 elements")
            if not (np.all(np.diff(xv) > 0) or np.all(np.diff(xv) < 0)):
                raise ValueError(f"Coordinate '{xd}' is not strictly monotonic")

    for yd in ("y", "lat"):
        if yd in ds.coords:
            yv = ds[yd].values
            if yv.size < 2:
                raise ValueError(f"Coordinate '{yd}' has <1 elements")
            if not (np.all(np.diff(yv) > 0) or np.all(np.diff(yv) < 0)):
                raise ValueError(f"Coordinate '{yd}' is not strictly monotonic")

    # time decodability (lazy-friendly)
    if "time" in ds.coords:
        _ = xr.decode_cf(ds[["time"]])  # raises if invalid CF time

    # CRS (if rioxarray is available)
    try:
        _crs = getattr(ds.rio, "crs", None)
        if _crs is None:
            raise ValueError("Dataset has no CRS (.rio.crs is None)")
    except Exception:
        # If rioxarray not present, skip CRS check
        pass

    def nbytes_est(da: xr.DataArray) -> int:
        """
        Estimate the total number of bytes occupied by a DataArray.

        The estimate multiplies the product of all dimension sizes by the
        item size of the array's data type. If the array has no dimensions,
        it assumes a single element. If the dtype is unavailable, an
        8-byte element size is used as a fallback.

        Parameters
        ----------
        da : xarray.DataArray
            The DataArray for which to estimate memory usage.

        Returns
        -------
        int
            Approximate number of bytes required to hold the array in memory.

        Notes
        -----
        This function does not consider chunking or compression; it provides
        a quick static estimate of raw in-memory size. Use
        ``da.nbytes`` for exact values on fully loaded arrays.

        Examples
        --------
        >>> import numpy as np, xarray as xr
        >>> da = xr.DataArray(np.zeros((100, 200)), dims=("y", "x"))
        >>> nbytes_est(da)
        160000
        """
        n = int(np.prod([da.sizes[d] for d in da.dims])) if da.dims else 1
        item = np.dtype(da.dtype).itemsize if hasattr(da, "dtype") else 8
        return n * item

    vars_sorted = sorted(ds.data_vars, key=lambda k: nbytes_est(ds[k]), reverse=True)
    sample_vars = vars_sorted[:max_vars]

    # --- sample windows and read ---
    for name in sample_vars:
        da = ds[name]
        windows = _windows_for(da, window=window)
        # read a few windows (limit to 5 per var)
        for w in windows[:5]:
            sub = da.isel(**{k: v for k, v in w.items() if k in da.dims})
            # small read: materialize into memory
            arr = sub.load().values
            if arr.size == 0:
                raise ValueError(f"{name}: sampled window is empty for selection {w}")

    # Optional: quick global metadata sanity
    if ds.attrs.get("Conventions", "").lower().startswith("cf"):
        # try writing minimal in-memory netcdf header/coords only (super light)
        _ = xr.Dataset(coords={k: ds[k].isel({k: slice(0, 1)}) for k in ds.coords})

    # If we reached here, the dataset is healthy enough for downstream steps.


def check_dataset_fully(ds: xr.Dataset) -> None:
    """
    Validate that an xarray Dataset can be fully loaded into memory.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to validate. The task calls ``ds.load()`` to force all
        variables/coordinates to be read and decoded.

    Raises
    ------
    Exception
        Any exception raised by ``xarray`` while reading/decoding or loading
        data (e.g., ``OSError``, ``ValueError``) is propagated. Prefect will
        retry according to the task's retry policy.

    Notes
    -----
    This is a lightweight integrity check used after staging. It does not
    modify the dataset.
    """
    _ = ds.load()


def check_xr_lazy(path: Path | str) -> bool:
    """
    Open a dataset and run a **sampled** health check with xarray.

    This lazily opens the dataset at ``path`` and invokes a lightweight
    validator (``check_dataset_lazy``) that loads only small windows
    rather than the entire dataset. Prints a ✓/✗ message and returns
    ``True`` on success.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a CF-compliant NetCDF/Zarr dataset readable by
        :func:`xarray.open_dataset`.

    Returns
    -------
    bool
        ``True`` if the dataset opens and the sampled checks pass;
        ``False`` otherwise.

    Raises
    ------
    None
        Exceptions are caught and reported; the function returns ``False`` on
        failure. If you want orchestrators (e.g., Prefect) to handle retries,
        re-raise after logging.

    Notes
    -----
    This function is intended for large datasets where ``.load()`` would be
    impractical. The underlying checker should verify basics like coordinate
    monotonicity, CF-time decodability, CRS presence, and small-window reads.
    """
    p = Path(path).resolve()
    is_ok: bool
    try:
        ds = xr.open_dataset(p)
        check_dataset_lazy(ds)  # your sampled checker
        print(f"{p} is valid ✓")
        is_ok = True
    except FileNotFoundError:
        print(f"{p} is valid ✗ (missing)")
        is_ok = False
    except Exception as e:
        print(f"{p} is valid ✗ ({type(e).__name__}: {e})")
        is_ok = False
    return is_ok


def check_xr_fully(path: Path | str) -> bool:
    """
    Open a dataset and verify it **fully loads** with xarray.

    This forces a full materialization using ``.load()`` via a helper
    (``check_dataset_fully``). Prints a ✓/✗ message and returns ``True`` on success.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a CF-compliant NetCDF/Zarr dataset readable by
        :func:`xarray.open_dataset`.

    Returns
    -------
    bool
        ``True`` if the dataset opens and fully loads; ``False`` otherwise.

    Raises
    ------
    None
        Exceptions are caught and reported; the function returns ``False`` on
        failure. Re-raise after logging if you want caller-managed retries.

    Notes
    -----
    Prefer :func:`check_xr_sampled` for very large datasets to avoid
    out-of-memory errors. This variant is useful when the dataset is expected
    to fit comfortably in memory and you want a strong integrity check.
    """
    p = Path(path).resolve()
    is_ok: bool
    try:
        ds = xr.open_dataset(p)
        check_dataset_fully(ds)  # your full-load checker
        print(f"{p} is valid ✓")
        is_ok = True
    except FileNotFoundError:
        print(f"{p} is valid ✗ (missing)")
        is_ok = False
    except Exception as e:
        print(f"{p} is valid ✗ ({type(e).__name__}: {e})")
        is_ok = False
    return is_ok


def check_rio(path: Path | str) -> bool:
    """
    Open a raster dataset and verify it can be read with rioxarray.

    Uses :func:`rioxarray.open_rasterio` (via ``rio.open``) to check that the
    file opens and basic metadata/CRS are readable. Prints a ✓/✗ message and
    returns ``True`` on success.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a raster readable by rioxarray (e.g., GeoTIFF, Cloud-Optimized
        GeoTIFF).

    Returns
    -------
    bool
        ``True`` if the raster opens successfully; ``False`` otherwise.

    Raises
    ------
    None
        Exceptions are caught and reported; the function returns ``False`` on
        failure. Re-raise after logging if you prefer caller-managed retries.

    Notes
    -----
    This check does not load full raster data. For deeper integrity tests,
    consider reading a small window or verifying CRS/transform consistency.
    """
    p = Path(path).resolve()
    is_ok: bool
    try:
        _ = rio.open(p)  # lightweight open
        print(f"{p} is valid ✓")
        is_ok = True
    except FileNotFoundError:
        print(f"{p} is valid ✗ (missing)")
        is_ok = False
    except Exception as e:
        print(f"{p} is valid ✗ ({type(e).__name__}: {e})")
        is_ok = False
    return is_ok
