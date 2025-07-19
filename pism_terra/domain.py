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
Create domains.
"""

import numpy as np
import xarray as xr


def create_domain(
    x_bnds: list | np.ndarray,
    y_bnds: list | np.ndarray,
    resolution: float | None = None,
    x_dim: str = "x",
    y_dim: str = "y",
    crs: str = "EPSG:3413",
) -> xr.Dataset:
    """
    Create an xarray.Dataset representing a domain with specified x and y boundaries.

    Parameters
    ----------
    x_bnds : list or numpy.ndarray
        A list or array containing the minimum and maximum x-coordinate boundaries.
    y_bnds : list or numpy.ndarray
        A list or array containing the minimum and maximum y-coordinate boundaries.
    resolution : float or None, optional
        The resolution of the grid, by default None.
    x_dim : str, optional
        The name of the x dimension, by default "x".
    y_dim : str, optional
        The name of the y dimension, by default "y".
    crs : str, optional
        The coordinate reference system (CRS) for the domain, by default "EPSG:3413".

    Returns
    -------
    xarray.Dataset
        An xarray.Dataset containing the domain information, including coordinates,
        boundary data, and mapping attributes.

    Notes
    -----
    The dataset includes:
    - `x` and `y` coordinates with associated metadata.
    - A `mapping` DataArray with polar stereographic projection attributes.
    - A `domain` DataArray with a reference to the `mapping`.
    - `x_bnds` and `y_bnds` DataArrays representing the boundaries of the domain.

    Examples
    --------
    >>> x_bnds = [0, 1000]
    >>> y_bnds = [0, 2000]
    >>> ds = create_domain(x_bnds, y_bnds)
    >>> print(ds)
    """

    if resolution is not None:
        x = np.arange(x_bnds[0] + resolution / 2, x_bnds[1], resolution)
        y = np.arange(y_bnds[0] + resolution / 2, y_bnds[1], resolution)
        xb = np.arange(x_bnds[0], x_bnds[1] + resolution, resolution)
        yb = np.arange(y_bnds[0], y_bnds[1] + resolution, resolution)
        x_bounds = np.stack([xb[:-1], xb[1:]]).T
        y_bounds = np.stack([yb[:-1], yb[1:]]).T
    else:
        x = [0]
        y = [0]
        x_bounds = [[x_bnds[0], x_bnds[1]]]
        y_bounds = [[y_bnds[0], y_bnds[1]]]

    x_bnds_dim = f"{x_dim}_bnds"
    y_bnds_dim = f"{y_dim}_bnds"
    coords = {
        x_dim: (
            [x_dim],
            x,
            {
                "units": "m",
                "axis": x_dim.upper(),
                "bounds": x_bnds_dim,
                "standard_name": "projection_x_coordinate",
                "long_name": f"{x_dim}-coordinate in projected coordinate system",
            },
        ),
        y_dim: (
            [y_dim],
            y,
            {
                "units": "m",
                "axis": y_dim.upper(),
                "bounds": y_bnds_dim,
                "standard_name": "projection_y_coordinate",
                "long_name": f"{y_dim}-coordinate in projected coordinate system",
            },
        ),
    }
    ds = xr.Dataset(
        {
            "domain": xr.DataArray(
                data=0,
                dims=[y_dim, x_dim],
                coords={x_dim: coords[x_dim], y_dim: coords[y_dim]},
                attrs={
                    "dimensions": f"{x_dim} {y_dim}",
                },
            ),
            x_bnds_dim: xr.DataArray(
                data=x_bounds,
                dims=[x_dim, "nv2"],
                coords={x_dim: coords[x_dim]},
                attrs={"_FillValue": False},
            ),
            y_bnds_dim: xr.DataArray(
                data=y_bounds,
                dims=[y_dim, "nv2"],
                coords={y_dim: coords[y_dim]},
                attrs={"_FillValue": False},
            ),
        },
        attrs={"Conventions": "CF-1.8"},
    ).rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
    ds.rio.write_crs(crs, inplace=True)
    return ds
