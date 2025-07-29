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

# mypy: disable-error-code="call-overload"
# pylint: disable=unused-import
"""
Module for data interpolation.
"""

import sys
from collections.abc import Hashable
from typing import Iterable

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import cg, spsolve

try:
    import pyamg

    has_pyamg = True
except ImportError:
    has_pyamg = False


@xr.register_dataarray_accessor("utils")
class InterpolationMethods:
    """
    Interpolation methods for xarray DataArray.

    This class is used to add custom methods to xarray DataArray objects. The methods can be accessed via the 'utils' attribute.

    Parameters
    ----------
    xarray_obj : xr.DataArray
        The xarray DataArray to which to add the custom methods.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        """
        Initialize the InterpolationMethods class.

        Parameters
        ----------
        xarray_obj : xr.DataArray
            The xarray DataArray to which to add the custom methods.
        """
        self._obj = xarray_obj

    def init(self):
        """
        Do-nothing method.

        This method is needed to work with joblib Parallel.
        """

    def __repr__(self):
        """
        Interpolation methods.

        Returns
        -------
        str
            Description of the interpolation methods.
        """
        return """
Interpolation methods for xarray DataArray.

This class is used to add custom methods to xarray DataArray objects. The methods can be accessed via the 'interpolation' attribute.

Parameters
----------

xarray_obj : xr.DataArray
  The xarray DataArray to which to add the custom methods.
      """

    def fillna(
        self,
        dim: str | Iterable[Hashable] = ["y", "x"],
        boundary_condition: str = "neumann",
        solver: str = "direct",
        client: Client | None = None,
    ) -> xr.DataArray:
        """
        Fill missing values in a DataArray using Laplace equation-based interpolation.

        This method applies a 2D Laplacian solver across the specified spatial dimensions
        to fill missing (NaN) values in the DataArray. It supports multiple boundary
        condition types and linear solvers, and can operate on Dask-backed arrays with
        optional progress monitoring.

        Parameters
        ----------
        dim : str or Iterable of hashable, optional
            Dimensions along which to apply the Laplacian solver. Typically spatial
            dimensions like ["y", "x"]. Defaults to ["y", "x"].

        boundary_condition : {"dirichlet", "neumann", "periodic"}, optional
            Type of boundary condition to use at the domain edges:
            - "dirichlet": fixed zero values outside the domain.
            - "neumann": zero-gradient (mirrored edge values).
            - "periodic": domain wraps cyclically.
            Defaults to "neumann".

        solver : {"direct", "cg", "pyamg"}, optional
            Linear solver for the sparse Laplace system:
            - "direct": LU factorization via `scipy.sparse.linalg.spsolve`.
            - "cg": Conjugate Gradient via `scipy.sparse.linalg.cg`.
            - "pyamg": Multigrid solver (requires `pyamg` package).
            Defaults to "direct".

        client : dask.distributed.Client, optional
            A Dask distributed client for parallel execution and progress reporting.
            If provided, progress will be shown via `dask.distributed.progress`.
            Otherwise, a local progress bar is shown via `dask.diagnostics.ProgressBar`.

        Returns
        -------
        xr.DataArray
            A DataArray with missing values filled by Laplacian interpolation.

        Notes
        -----
        - The function is vectorized across all non-spatial dimensions.
        - Works seamlessly with Dask arrays, chunked along non-interpolated axes.
        - Preserves metadata such as `attrs`, `name`, `coords`, and `encoding`.
        - This method is most appropriate for interpolating over 2D grid-like fields.
        """
        da = self._obj

        # Save attributes and metadata
        attrs = da.attrs
        name = da.name
        coords = da.coords
        encoding = da.encoding

        da = xr.apply_ufunc(
            laplace_2d_block,
            da,
            input_core_dims=[dim],
            output_core_dims=[dim],
            vectorize=True,
            dask="allowed",
            keep_attrs=True,
            kwargs={"boundary_condition": boundary_condition, "solver": solver},
        )

        if client is not None:
            # Distributed scheduler: submit and show progress via client
            future = client.compute(da)
            client.wait_for_workers(1)
            progress(future)
            result = client.gather(future)

        else:
            # Local scheduler with tqdm-based progress bar
            with ProgressBar():
                result = da.compute()

        # Manually restore other metadata
        self._obj = result
        self._obj.attrs = attrs
        self._obj.name = name
        self._obj.coords.update(coords)
        self._obj.encoding = encoding

        return self._obj


def laplace_2d_block(data_2d: np.ndarray, boundary_condition: str = "neumann", solver: str = "direct") -> np.ndarray:
    """
    Fill missing values in a single 2D array using Laplace equation.

    Parameters
    ----------
    data_2d : np.ndarray
        2D array with NaNs to be filled.
    boundary_condition : {"dirichlet", "neumann", "periodic"}
        Type of boundary condition to use.
        - "dirichlet": values outside domain are treated as zero (default).
        - "neumann": values outside domain are mirrored (zero-gradient).
        - "periodic": domain wraps around.
    solver : {"direct", "cg", "pyamg"}
        Which solver to use:
        - "direct": sparse LU factorization (fast for small domains)
        - "cg": Conjugate Gradient (suitable for larger problems)
        - "pyamg": Algebraic Multigrid (fastest if available)

    Returns
    -------
    np.ndarray
        The input array with NaNs filled.
    """
    n, m = data_2d.shape
    mask = np.isnan(data_2d)
    data = np.nan_to_num(data_2d.copy())

    interior = np.argwhere(mask)
    if len(interior) == 0:
        return data_2d  # nothing to fill

    index_map = {tuple(pt): idx for idx, pt in enumerate(interior)}
    rows, cols, vals = [], [], []
    b = np.zeros(len(interior))

    for k, (i, j) in enumerate(interior):
        rows.append(k)
        cols.append(k)
        vals.append(-4)

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj

            if not (0 <= ni < n and 0 <= nj < m):
                if boundary_condition == "dirichlet":
                    continue
                if boundary_condition == "neumann":
                    ni = min(max(ni, 0), n - 1)
                    nj = min(max(nj, 0), m - 1)
                elif boundary_condition == "periodic":
                    ni %= n
                    nj %= m
                else:
                    raise ValueError(f"Unknown boundary condition: {boundary_condition}")

            if mask[ni, nj]:
                neighbor_idx = index_map[(ni, nj)]
                rows.append(k)
                cols.append(neighbor_idx)
                vals.append(1)
            else:
                b[k] -= data[ni, nj]

    A = coo_matrix((vals, (rows, cols)), shape=(len(interior), len(interior))).tocsc()

    # Solve the system
    if solver == "direct":
        x = spsolve(A, b)

    elif solver == "cg":
        x, info = cg(A, b, atol=1e-10)
        if info != 0:
            raise RuntimeError(f"CG solver did not converge (info={info})")

    elif solver == "pyamg":
        if not has_pyamg:
            raise ImportError("pyamg not available; install with `pip install pyamg`")
        ml = pyamg.ruge_stuben_solver(A)
        x = ml.solve(b, tol=1e-10)

    else:
        raise ValueError(f"Unknown solver type: {solver}")

    # Fill in the values
    result = data_2d.copy()
    for val, (i, j) in zip(x, interior):
        result[i, j] = val

    return result
