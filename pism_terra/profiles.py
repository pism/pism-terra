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
Profiles.
"""

import numpy as np
import xarray as xr


@xr.register_dataset_accessor("profiles")
class ProfilesMethods:
    """
    Profiles methods for xarray Dataset.

    This class is used to add custom methods to xarray Dataset objects. The methods can be accessed via the 'profiles' attribute.

    Parameters
    ----------

    xarray_obj : xr.Dataset
      The xarray Dataset to which to add the custom methods.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        """
        Initialize the ProfilesMethods class.

        Parameters
        ----------

        xarray_obj : xr.Dataset
            The xarray Dataset to which to add the custom methods.
        """
        self._obj = xarray_obj

    def init(self):
        """
        Do-nothing method.

        This method is needed to work with joblib Parallel.
        """

    def add_normal_component(
        self,
        x_component: str = "vx",
        y_component: str = "vy",
        normal_name: str = "v_normal",
    ):
        """
        Add a normal component to the xarray Dataset.

        This method computes the normal component of the vectors defined by the x and y components, and adds it to the Dataset.

        Parameters
        ----------
        x_component : str, optional
            The name of the x component variable in the Dataset, by default "vx".
        y_component : str, optional
            The name of the y component variable in the Dataset, by default "vy".
        normal_name : str, optional
            The name of the normal component variable to add to the Dataset, by default "v_normal".

        Returns
        -------
        xr.Dataset
            The xarray Dataset with the normal variables added.
        """
        assert (x_component and y_component) in self._obj.data_vars

        def func(x, x_n, y, y_n):
            """
            Calculate the normal component of a vector.

            This function computes the normal component of a vector by performing a dot product operation.
            The inputs are the x and y components of the vector and their corresponding normal components.

            Parameters
            ----------
            x : float
                The x-component of the vector.
            x_n : float
                The x-component of the normal.
            y : float
                The y-component of the vector.
            y_n : float
                The y-component of the normal.

            Returns
            -------
            float
                The normal component of the vector.

            Examples
            --------
            >>> func(1, 2, 3, 4)
            14
            """
            return x * x_n + y * y_n

        self._obj[normal_name] = xr.apply_ufunc(
            func, self._obj[x_component], self._obj["nx"], self._obj[y_component], self._obj["ny"], dask="allowed"
        )
        return self._obj

    def calculate_stats(
        self,
        obs_var: str = "obs_v_normal",
        sim_var: str = "sim_v_normal",
        dim: str = "profile_axis",
        stats: list[str] = ["rmsd", "pearson_r"],
    ):
        """
        Calculate statistical metrics between observed and simulated data.

        This function calculates the Root Mean Square Deviation (RMSD) and Pearson correlation coefficient between
        observed and simulated data along a specified dimension.

        Parameters
        ----------
        obs_var : str, optional
            The observed data variable name in the xarray Dataset, by default "v".
        sim_var : str, optional
            The simulated data variable name in the xarray Dataset, by default "velsurf_mag".
        dim : str, optional
            The dimension along which to calculate the statistics, by default "profile_axis".
        stats : List[str], optional
            The list of statistical metrics to calculate, by default ["rmsd", "pearson_r"].

        Returns
        -------
        xr.Dataset
            The xarray Dataset with the calculated statistical metrics added as new data variables.
        """
        assert (
            obs_var and sim_var
        ) in self._obj.data_vars, f"{obs_var} or {sim_var} not in {list(self._obj.data_vars)}."

        def rmsd(sim: xr.DataArray, obs: xr.DataArray) -> float:
            """
            Compute the Root Mean Square Deviation (RMSD) between simulated and observed data.

            This function computes the RMSD between two xarray DataArrays. The RMSD is calculated as the square root
            of the mean of the squared differences between the simulated and observed data.

            Parameters
            ----------
            sim : xr.DataArray
                The simulated data as an xarray DataArray.
            obs : xr.DataArray
                The observed data as an xarray DataArray.

            Returns
            -------

            float
              RMSD of sim and obs.
            """
            diff = sim - obs
            return np.sqrt(np.nanmean(diff**2, axis=-1))

        def pearson_r(sim: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
            """
            Compute the Pearson correlation coefficient between simulated and observed data.

            This function computes the Pearson correlation coefficient (r) between two xarray DataArrays along the "profile_axis" dimension.

            Parameters
            ----------
            sim : xr.DataArray
                The simulated data as an xarray DataArray.
            obs : xr.DataArray
                The observed data as an xarray DataArray.

            Returns
            -------
            xr.DataArray
                The computed Pearson correlation coefficient as an xarray DataArray.
            """
            return xr.corr(sim, obs, dim="profile_axis")

        stats_func = {"rmsd": {"func": rmsd, "ufunc": True}, "pearson_r": {"func": pearson_r, "ufunc": False}}
        fluxes = {obs_var: "obs_flux", sim_var: "sim_flux"}

        for k, v in fluxes.items():
            self._obj[v] = self._obj[k].integrate(coord="profile_axis")

        for stat in stats:
            if stats_func[stat]["ufunc"]:
                self._obj[stat] = xr.apply_ufunc(
                    stats_func[stat]["func"],
                    self._obj[obs_var],
                    self._obj[sim_var],
                    dask="allowed",
                    input_core_dims=[[dim], [dim]],
                    output_core_dims=[[]],
                )
            else:
                self._obj[stat] = pearson_r(self._obj[obs_var], self._obj[sim_var])

        return self._obj

    def extract_profile(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        profile_name: str = "Glacier X",
        profile_id: int = 0,
        data_vars: list[str] | None = None,
        normal_var: str = "v_normal",
        normal_error_var: str = "v_err_normal",
        normal_component_vars: dict = {"x": "vx", "y": "vy"},
        normal_component_error_vars: dict = {"x": "vx_err", "y": "vy_err"},
        compute_profile_normal: bool = False,
    ) -> xr.Dataset:
        """
        Extract a profile from a dataset given x and y coordinates.

        Parameters
        ----------
        xs : np.ndarray
            The x-coordinates of the profile.
        ys : np.ndarray
            The y-coordinates of the profile.
        profile_name : str, optional
            The name of the profile, by default "Glacier X".
        profile_id : int, optional
            The id of the profile, by default 0.
        data_vars : Union[None, List[str]], optional
            The list of data variables to include in the profile. If None, all data variables are included, by default None.
        normal_var : str, optional
            The name of the normal variable, by default "v_normal".
        normal_error_var : str, optional
            The name of the normal error variable, by default "v_err_normal".
        normal_component_vars : dict, optional
            The dictionary of normal component variables, by default {"x": "vx", "y": "vy"}.
        normal_component_error_vars : dict, optional
            The dictionary of normal component error variables, by default {"x": "vx_err", "y": "vy_err"}.
        compute_profile_normal : bool, optional
            Whether to compute the profile normal, by default False.

        Returns
        -------
        xr.Dataset
            A new xarray Dataset containing the extracted profile.
        """
        profile_axis = np.sqrt(xs**2 + ys**2).cumsum() - np.sqrt(xs[0] ** 2 + ys[0] ** 2)

        x: xr.DataArray
        y: xr.DataArray
        x = xr.DataArray(
            xs,
            dims="profile_axis",
            coords={"profile_axis": profile_axis},
            attrs=self._obj["x"].attrs,
            name="x",
        )
        y = xr.DataArray(
            ys,
            dims="profile_axis",
            coords={"profile_axis": profile_axis},
            attrs=self._obj["y"].attrs,
            name="y",
        )

        nx, ny = compute_normals(x, y)
        normals = {"nx": nx, "ny": ny}

        das = [
            xr.DataArray(
                val,
                dims="profile_axis",
                coords={"profile_axis": profile_axis},
                name=key,
            )
            for key, val in normals.items()
        ]

        name = xr.DataArray(
            [profile_name],
            dims="profile_id",
            attrs={"units": "m", "long_name": "distance along profile"},
            name="profile_name",
        )
        das.append(name)

        if data_vars is None:
            data_vars = list(self._obj.data_vars)

        for m_var in data_vars:
            da = self._obj[m_var]
            try:
                das.append(da.interp(x=x, y=y, kwargs={"fill_value": np.nan}))
            except:
                pass

        ds = xr.merge(das)
        ds["profile_id"] = [profile_id]
        if compute_profile_normal:
            a = [(v in ds.data_vars) for v in normal_component_vars.values()]
            if np.all(np.array(a)):
                ds.profiles.add_normal_component(
                    x_component=normal_component_vars["x"],
                    y_component=normal_component_vars["y"],
                    normal_name=normal_var,
                )
                assert ds[normal_component_vars["x"]].units == ds[normal_component_vars["y"]].units
                profile_units = ds[normal_component_vars["x"]].units
                ds[normal_var]["units"] = profile_units

            a = [(v in ds.data_vars) for v in normal_component_error_vars.values()]
            if np.all(np.array(a)):
                ds.profiles.add_normal_component(
                    x_component=normal_component_error_vars["x"],
                    y_component=normal_component_error_vars["y"],
                    normal_name=normal_error_var,
                )
                if (normal_component_error_vars["x"] and normal_component_error_vars["y"]) in ds.data_vars:
                    assert ds[normal_component_error_vars["x"]].units == ds[normal_component_error_vars["y"]].units
                    profile_error_units = ds[normal_component_error_vars["x"]].units
                    ds[normal_error_var]["units"] = profile_error_units

        return ds


def normal(point0: np.ndarray, point1: np.ndarray) -> np.ndarray:
    """
    Compute the unit normal vector orthogonal to (point1-point0), pointing 'to the right' of (point1-point0).

    Parameters
    ----------
    point0 : np.ndarray
        The starting point of the vector.
    point1 : np.ndarray
        The ending point of the vector.

    Returns
    -------
    np.ndarray
        The unit normal vector orthogonal to the vector from point0 to point1.

    Notes
    -----
    This function computes the unit normal vector orthogonal to the vector from point0 to point1.
    The normal vector points to the right of the vector from point0 to point1.
    If the dot product of the vector from point0 to point1 and the normal vector is negative, the direction of the normal vector is flipped.
    """

    a = point0 - point1
    n = np.array([-a[1], a[0]])  # compute the normal vector
    n = n / np.linalg.norm(n)  # normalize

    # flip direction if needed:
    if np.cross(a, n) < 0:
        n = -1.0 * n
    return n


def tangential(point0: np.ndarray, point1: np.ndarray) -> np.ndarray:
    """
    Compute the unit tangential vector to (point1-point0), pointing 'to the right' of (point1-point0).

    Parameters
    ----------
    point0 : np.ndarray
        The starting point of the vector.
    point1 : np.ndarray
        The ending point of the vector.

    Returns
    -------
    np.ndarray
        The unit tangential vector pointing from point0 to point1.

    Notes
    -----
    This function computes the unit tangential vector from point0 to point1.
    If the norm of the vector from point0 to point1 is zero, the function returns the zero vector.
    """
    a = point1 - point0
    norm = np.linalg.norm(a)

    # protect from division by zero
    return a if norm == 0 else a / norm


def compute_normals(px: np.ndarray | xr.DataArray | list, py: np.ndarray | xr.DataArray | list):
    """
    Compute normals to a profile described by 'p'. Normals point 'to the right' of the path.

    Parameters
    ----------
    px : Union[np.ndarray, xr.DataArray, list]
        The x-coordinates of the points describing the profile.
    py : Union[np.ndarray, xr.DataArray, list]
        The y-coordinates of the points describing the profile.

    Returns
    -------
    tuple of np.ndarray
        The x and y components of the normal vectors to the profile.

    Notes
    -----
    This function computes the normal vectors to a profile described by the points (px, py).
    The normal vector at each point is computed as the vector from the previous point to the next point, rotated 90 degrees clockwise.
    For the first and last points, the normal vector is computed as the vector from the first point to the second point and from the second last point to the last point, respectively, also rotated 90 degrees clockwise.
    """
    p = np.vstack((px, py)).T

    if len(p) < 2:
        return [0], [0]

    ns = np.zeros_like(p)
    ns[0] = normal(p[0], p[1])
    for j in range(1, len(p) - 1):
        ns[j] = normal(p[j - 1], p[j + 1])

    ns[-1] = normal(p[-2], p[-1])

    return ns[:, 0], ns[:, 1]


def compute_tangentials(px: np.ndarray | list, py: np.ndarray | list):
    """
    Compute tangentials to a profile described by 'p'.

    Parameters
    ----------
    px : Union[np.ndarray, list]
        The x-coordinates of the points describing the profile.
    py : Union[np.ndarray, list]
        The y-coordinates of the points describing the profile.

    Returns
    -------
    tuple of np.ndarray
        The x and y components of the tangential vectors to the profile.

    Notes
    -----
    This function computes the tangential vectors to a profile described by the points (px, py).
    The tangential vector at each point is computed as the vector from the previous point to the next point.
    For the first and last points, the tangential vector is computed as the vector from the first point to the second point and from the second last point to the last point, respectively.
    """
    p = np.vstack((px, py)).T

    if len(p) < 2:
        return [0], [0]

    ts = np.zeros_like(p)
    ts[0] = tangential(p[0], p[1])
    for j in range(1, len(p) - 1):
        ts[j] = tangential(p[j - 1], p[j + 1])

    ts[-1] = tangential(p[-2], p[-1])

    return ts[:, 0], ts[:, 1]
