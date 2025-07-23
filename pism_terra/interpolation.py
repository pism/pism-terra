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
from scipy.sparse import coo_matrix, csc_matrix, diags
from scipy.sparse.linalg import spsolve


def create_laplacian_matrix(interior_points: np.ndarray, mask: np.ndarray, n: int, m: int) -> csc_matrix:
    """
    Create the Laplacian matrix for the given interior points and mask.

    Parameters
    ----------
    interior_points : np.ndarray
        Array of interior points where the mask is False.
    mask : np.ndarray
        Boolean mask indicating the missing values.
    n : int
        Number of rows in the data array.
    m : int
        Number of columns in the data array.

    Returns
    -------
    csc_matrix
        The Laplacian matrix in CSC format.
    """
    row_indices = []
    col_indices = []
    data_values = []

    for k, (i, j) in enumerate(interior_points):
        row_indices.append(k)
        col_indices.append(k)
        data_values.append(-4)

        if i > 0:
            if mask[i - 1, j]:
                neighbor_index = np.where((interior_points == [i - 1, j]).all(axis=1))[0][0]
                row_indices.append(k)
                col_indices.append(neighbor_index)
                data_values.append(1)
        if i < n - 1:
            if mask[i + 1, j]:
                neighbor_index = np.where((interior_points == [i + 1, j]).all(axis=1))[0][0]
                row_indices.append(k)
                col_indices.append(neighbor_index)
                data_values.append(1)
        if j > 0:
            if mask[i, j - 1]:
                neighbor_index = np.where((interior_points == [i, j - 1]).all(axis=1))[0][0]
                row_indices.append(k)
                col_indices.append(neighbor_index)
                data_values.append(1)
        if j < m - 1:
            if mask[i, j + 1]:
                neighbor_index = np.where((interior_points == [i, j + 1]).all(axis=1))[0][0]
                row_indices.append(k)
                col_indices.append(neighbor_index)
                data_values.append(1)

    # Create the sparse matrix using COO format
    L = coo_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(len(interior_points), len(interior_points)),
    ).tocsc()
    return L


def create_rhs_vector(data: np.ndarray, interior_points: np.ndarray, mask: np.ndarray, n: int, m: int) -> np.ndarray:
    """
    Create the right-hand side vector for the linear system.

    Parameters
    ----------
    data : np.ndarray
        The data array with missing values.
    interior_points : np.ndarray
        Array of interior points where the mask is False.
    mask : np.ndarray
        Boolean mask indicating the missing values.
    n : int
        Number of rows in the data array.
    m : int
        Number of columns in the data array.

    Returns
    -------
    np.ndarray
        The right-hand side vector.
    """
    b = np.zeros(len(interior_points))

    for k, (i, j) in enumerate(interior_points):
        if i > 0 and ~mask[i - 1, j]:
            b[k] -= data[i - 1, j]
        if i < n - 1 and ~mask[i + 1, j]:
            b[k] -= data[i + 1, j]
        if j > 0 and ~mask[i, j - 1]:
            b[k] -= data[i, j - 1]
        if j < m - 1 and ~mask[i, j + 1]:
            b[k] -= data[i, j + 1]

    return b


def laplace(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fill missing values in the data array using the Laplacian method.

    Parameters
    ----------
    data : np.ndarray
        The data array with missing values.
    mask : np.ndarray
        Boolean mask indicating the missing values.

    Returns
    -------
    np.ndarray
        The data array with missing values filled.
    """

    data = data.copy()
    n, m = data.shape
    interior_points = np.argwhere(mask)

    # Create the Laplacian matrix
    L = create_laplacian_matrix(interior_points, mask, n, m)

    # Create the right-hand side vector
    b = create_rhs_vector(data, interior_points, mask, n, m)

    # Solve the linear system
    u = spsolve(L, b)
    # Fill in the missing values
    for k, (i, j) in enumerate(interior_points):
        data[i, j] = u[k]

    return data


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
    ) -> xr.DataArray:
        """
        Fill missing values using Laplacian.

        Parameters
        ----------
        dim : str | Iterable[Hashable], optional
            The dimensions along which to fill missing values, by default ["y", "x"].

        Returns
        -------
        xr.DataArray
            The DataArray with missing values filled.
        """
        da = self._obj.load()
        data = da.to_numpy()
        mask = da.isnull()
        self._obj = xr.apply_ufunc(
            self._fillna,
            data,
            mask,
            input_core_dims=[dim, dim],
            output_core_dims=[dim],
            vectorize=True,
            dask="forbidden",
        )
        return self._obj

    def _fillna(self, data, mask):
        """
        Fill missing values.

        Parameters
        ----------
        data : np.ndarray
            The data array with missing values.
        mask : np.ndarray
            Boolean mask indicating the missing values.

        Returns
        -------
        np.ndarray
            The data array with missing values filled.
        """

        result = laplace(data, mask)

        return result
