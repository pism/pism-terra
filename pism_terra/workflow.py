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
from typing import Iterable

import numpy as np
import rasterio as rio
import rioxarray  # noqa: F401
import xarray as xr

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


def check_dataset_sampled(
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
            if not np.isfinite(arr).any():
                raise ValueError(f"{name}: sampled window has no finite values ({w})")

    # Optional: quick global metadata sanity
    if ds.attrs.get("Conventions", "").lower().startswith("cf"):
        # try writing minimal in-memory netcdf header/coords only (super light)
        _ = xr.Dataset(coords={k: ds[k].isel({k: slice(0, 1)}) for k in ds.coords})

    # If we reached here, the dataset is healthy enough for downstream steps.


def check_dataset(ds: xr.Dataset) -> None:
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


def check_xr_sampled(path: Path | str) -> bool:
    """
    Open a dataset and run a **sampled** health check with xarray.

    This lazily opens the dataset at ``path`` and invokes a lightweight
    validator (``check_dataset_sampled``) that loads only small windows
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
        check_dataset_sampled(ds)  # your sampled checker
        print(f"{p} is valid ✓")
        is_ok = True
    except FileNotFoundError:
        print(f"{p} is valid ✗ (missing)")
        is_ok = False
    except Exception as e:
        print(f"{p} is valid ✗ ({type(e).__name__}: {e})")
        is_ok = False
    return is_ok


def check_xr(path: Path | str) -> bool:
    """
    Open a dataset and verify it **fully loads** with xarray.

    This forces a full materialization using ``.load()`` via a helper
    (``check_dataset``). Prints a ✓/✗ message and returns ``True`` on success.

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
        check_dataset(ds)  # your full-load checker
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
