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
# pylint: disable=too-many-positional-arguments


"""
Processing Functions.
"""

import json
import re
from collections import OrderedDict
from collections.abc import Hashable, Mapping
from typing import Any

import cftime
import numpy as np
import pint_xarray  # pylint: disable=unused-import
import xarray as xr


def integrate_rate(
    da: xr.DataArray,
    dim: str = "time",
    method: str = "left",
    widths: xr.DataArray | None = None,
    to: str | None = None,
    skipna: bool = True,
) -> xr.DataArray:
    """
    Cumulatively integrate a rate over a possibly irregular datetime axis.

    Values carrying a ``[time]`` dimension (Gt/yr, m^2/day, kg/s, W) are weighted by the
    duration of the interval each one represents before accumulating, so unequal spacing
    is handled correctly. The time unit is never parsed out of the unit string: interval
    widths are built in nanoseconds and pint reconciles them against whatever denominator
    the rate carries, which also works for derived units such as W where no explicit time
    unit appears.

    Integration runs along `dim` only; any other dimensions (``uq_id``, ``exp_id``, ``x``,
    ``y``, ...) are carried through independently, so the result keeps the input's shape,
    dimension order and coordinates. `dim` need not be the leading axis.

    Parameters
    ----------
    da : xr.DataArray
        The rate to integrate. Either pint-quantified or carrying ``attrs["units"]``.
        May have any number of dimensions in addition to `dim`.
    dim : str, optional
        The name of the time dimension, by default "time".
    method : str, optional
        Quadrature rule used when `widths` is None, by default "left".
        "left" holds each rate constant until the next sample, which is exact when the
        values are interval means. "trapezoid" interpolates linearly between samples,
        which suits instantaneous rates. Both produce ``n - 1`` increments, so the result
        starts at zero on the first timestamp and the final sample contributes nothing.
    widths : xr.DataArray or None, optional
        Explicit per-sample interval widths, by default None. One-dimensional along `dim`,
        and broadcast against the other dimensions. Either ``timedelta64`` or a
        pint-quantified duration; unquantified numeric widths are assumed to be days, which
        matches ``time.dt.days_in_month``. When given, every sample contributes and `method`
        is ignored.
    to : str or None, optional
        Target unit for the result, by default None. When None the reduced form chosen by
        pint is used, which collapses cleanly only when the time powers cancel -- pass an
        explicit unit for cases such as W (``to="J"``).
    skipna : bool, optional
        If True, treat NaN increments as zero so gaps contribute no mass, by default True.

    Returns
    -------
    xr.DataArray
        Pint-quantified cumulative integral along `dim`.

    Raises
    ------
    ValueError
        If `da` has no ``[time]`` dimension, if `dim` is not a dimension of `da`, if
        `method` is not recognized, or if `widths` is not one-dimensional.

    Notes
    -----
    Conversions involving ``year`` use pint's Julian year of 365.25 days. Pass `widths`
    explicitly if the rate is defined against calendar years.

    Examples
    --------
    >>> import pandas as pd
    >>> import xarray as xr
    >>> time = pd.to_datetime(["2020-01-01", "2020-01-02"])
    >>> rate = xr.DataArray([2.0, 2.0], coords={"time": time}, dims="time").pint.quantify("kg/s")
    >>> integrate_rate(rate).pint.magnitude
    array([     0., 172800.])
    """
    if da.pint.units is None:
        da = da.pint.quantify()
    units = da.pint.units
    ureg = units._REGISTRY  # pylint: disable=protected-access

    if units.dimensionality.get("[time]", 0) == 0:
        raise ValueError(f"{units} has no [time] dimension -- not a rate")
    if dim not in da.dims:
        raise ValueError(f"{dim!r} is not a dimension of the input, got {tuple(da.dims)}")

    magnitude = da.pint.magnitude
    axis = da.get_axis_num(dim)

    def along(arr, window):
        """
        Slice `arr` with `window` on the integration axis, taking every other axis whole.

        Parameters
        ----------
        arr : array_like
            Array to slice, with the same rank as the input rate.
        window : slice
            Slice applied to the integration axis.

        Returns
        -------
        array_like
            View of `arr` restricted to `window` along the integration axis.
        """
        index: list = [slice(None)] * arr.ndim
        index[axis] = window
        return arr[tuple(index)]

    # Interval widths are 1-D along `dim`; reshape so they broadcast over the other dims.
    shape: list[int] = [1] * da.ndim
    shape[axis] = -1

    if widths is not None:
        if widths.ndim != 1:
            raise ValueError(f"widths must be one-dimensional along {dim!r}, got {widths.ndim} dims")
        width_values = np.asarray(widths.pint.magnitude if widths.pint.units else widths)
        if np.issubdtype(width_values.dtype, np.timedelta64):
            delta = ureg.Quantity(width_values / np.timedelta64(1, "ns"), "ns")
        else:
            delta = ureg.Quantity(
                width_values.astype("float64"),
                str(widths.pint.units) if widths.pint.units else "day",
            )
        increments = magnitude * delta.reshape(shape)
    else:
        delta = ureg.Quantity(da[dim].diff(dim).values / np.timedelta64(1, "ns"), "ns").reshape(shape)
        if method == "trapezoid":
            increments = 0.5 * (along(magnitude, slice(None, -1)) + along(magnitude, slice(1, None))) * delta
        elif method == "left":
            increments = along(magnitude, slice(None, -1)) * delta
        else:
            raise ValueError(f"unknown method {method!r}, expected 'left' or 'trapezoid'")

    increments = (increments * units).to(to) if to else (increments * units).to_reduced_units()
    values = np.nan_to_num(increments.magnitude) if skipna else increments.magnitude
    cumulative = np.cumsum(values, axis=axis)
    if widths is None:
        # The diff-based rules yield n - 1 increments; the series starts at zero.
        leading = list(values.shape)
        leading[axis] = 1
        cumulative = np.concatenate([np.zeros(leading, dtype=cumulative.dtype), cumulative], axis=axis)

    out = xr.DataArray(
        ureg.Quantity(cumulative, increments.units),
        coords=da.coords,
        dims=da.dims,
        name=f"cumulative_{da.name}" if da.name else "cumulative",
    )
    if "long_name" in da.attrs:
        out.attrs["long_name"] = f"Cumulative {da.attrs['long_name']}"
    return out


def preprocess_netcdf(
    ds,
    exp_regexp: str = "id_(.+?)_",
    uq_regexp: str | None = r"(RGI2000-v7\.0-C-[^/\s]+)",
    exp_dim: str = "exp_id",
    uq_dim: str | None = "uq_id",
    gcm_dim: str | None = "gcm_id",
    drop_vars: list[str] | None = None,
    drop_dims: list[str] = ["nv4"],
    process_config: bool = True,
) -> xr.Dataset:
    """
    Add experiment identifier to the dataset.

    This function processes the dataset by extracting an experiment identifier from the filename
    using a regular expression, adding it as a new dimension, and optionally dropping specified
    variables and dimensions from the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset to be processed.
    exp_regexp : str, optional
        The regular expression pattern to extract the experiment identifier from the filename, by default "id_(.+?)_".
    uq_regexp : str or None, optional
        The regular expression pattern to extract the UQ identifier from the filename, by default ``r"(RGI2000-v7\\.0-C-[^/\\s]+)"``.
        If None, no UQ dimension is added.
    exp_dim : str, optional
        The name of the new experiment dimension to be added to the dataset, by default "exp_id".
    uq_dim : str or None, optional
        The name of the new UQ dimension to be added to the dataset, by default "uq_id".
        If None, no UQ dimension is added.
    gcm_dim : str or None, optional
        The name of the GCM dimension to be added to the dataset, by default "gcm_id".
        If None, no GCM dimension is added. The GCM name is extracted from the filename
        by matching the pattern ``id_<gcm>_<forcing>``.
    drop_vars : list[str]| None, optional
        A list of variable names to be dropped from the dataset, by default None.
    drop_dims : list[str], optional
        A list of dimension names to be dropped from the dataset, by default ["nv4"].
    process_config : bool, optional
        If True, extract and store pism_config as a JSON-encoded DataArray. If False, simply
        drop the pism_config variable and axis without re-adding it. By default True.

    Returns
    -------
    xarray.Dataset
        The processed dataset with the experiment identifier added as a new dimension, and specified variables and dimensions dropped.

    Raises
    ------
    AssertionError
        If the regular expression does not match any part of the filename.
    """

    m_exp_id_re = re.search(exp_regexp, ds.encoding["source"])
    assert m_exp_id_re is not None
    m_exp_id = m_exp_id_re.group(1)

    if process_config:
        p_config = ds["pism_config"]

    ds = ds.drop_vars(["pism_config"], errors="ignore").drop_dims(["pism_config_axis"], errors="ignore")

    expand_dims = []
    expand_coords = {}

    if gcm_dim is not None:
        gcm_regexp = r"_gcm_(.+?)_exp_"
        m_gcm_re = re.search(gcm_regexp, ds.encoding["source"])
        if m_gcm_re is not None:
            m_gcm_id = m_gcm_re.group(1)
            expand_dims.append(gcm_dim)
            expand_coords[gcm_dim] = [m_gcm_id]

    if uq_regexp is not None and uq_dim is not None and hasattr(ds, "command"):
        m_uq_id_re = re.search(uq_regexp, ds.command)
        assert m_uq_id_re is not None
        m_uq_id = m_uq_id_re.group(1)
        expand_dims.append(uq_dim)
        expand_coords[uq_dim] = [m_uq_id]

    expand_dims.append(exp_dim)
    expand_coords[exp_dim] = [m_exp_id]
    ds = ds.expand_dims(expand_coords)

    if process_config:

        # List of suffixes to exclude
        suffixes_to_exclude = ["_doc", "_type", "_units", "_option", "_choices"]

        # Filter the dictionary and encode as a single JSON string per (uq_id, exp_id)
        config = {
            k: v for k, v in p_config.attrs.items() if not any(k.endswith(suffix) for suffix in suffixes_to_exclude)
        }
        if "geometry.front_retreat.prescribed.file" not in config.keys():
            config["geometry.front_retreat.prescribed.file"] = "false"

        config_json = json.dumps(OrderedDict(sorted(config.items())))
        shape = [1] * len(expand_dims)
        pism_config = xr.DataArray(
            np.array(config_json, dtype=object).reshape(shape),
            dims=expand_dims,
            coords=expand_coords,
            name="pism_config",
        )
        ds = ds.assign_coords(pism_config=pism_config)

    return ds.drop_vars(drop_vars, errors="ignore").drop_dims(drop_dims, errors="ignore")


def preprocess_config_rgi(
    ds,
    exp_regexp: str = "id_(.+?)_",
    rgi_regexp: str = r"(RGI2000-v7\.0-C-[^/\s]+)",
    exp_dim: str = "exp_id",
    rgi_dim: str = "rgi_id",
    drop_vars: list[str] | None = None,
    drop_dims: list[str] = ["nv4"],
) -> xr.Dataset:
    """
    Add experiment identifier to the dataset.

    This function processes the dataset by extracting an experiment identifier from the filename
    using a regular expression, adding it as a new dimension, and optionally dropping specified
    variables and dimensions from the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset to be processed.
    exp_regexp : str, optional
        The regular expression pattern to extract the experiment identifier from the filename, by default "id_(.+?)_".
    rgi_regexp : str, optional
        The regular expression pattern to extract the RGI identifier from the filename, by default ``r"(RGI2000-v7\\.0-C-[^/\\s]+)"``.
    exp_dim : str, optional
        The name of the new experiment dimension to be added to the dataset, by default "exp_id".
    rgi_dim : str, optional
        The name of the new RGI dimension to be added to the dataset, by default "rgi_id".
    drop_vars : list[str]| None, optional
        A list of variable names to be dropped from the dataset, by default None.
    drop_dims : list[str], optional
        A list of dimension names to be dropped from the dataset, by default ["nv4"].

    Returns
    -------
    xarray.Dataset
        The processed dataset with the experiment identifier added as a new dimension, and specified variables and dimensions dropped.

    Raises
    ------
    AssertionError
        If the regular expression does not match any part of the filename.
    """

    m_rgi_id_re = re.search(rgi_regexp, ds.command)
    assert m_rgi_id_re is not None
    m_rgi_id = m_rgi_id_re.group(1)

    m_exp_id_re = re.search(exp_regexp, ds.encoding["source"])
    assert m_exp_id_re is not None
    m_exp_id = m_exp_id_re.group(1)

    p_config = ds["pism_config"]
    ds = ds.drop_vars(["pism_config"], errors="ignore").drop_dims(["pism_config_axis"], errors="ignore")
    ds = ds.expand_dims({rgi_dim: [m_rgi_id], exp_dim: [m_exp_id]})

    # List of suffixes to exclude
    suffixes_to_exclude = ["_doc", "_type", "_units", "_option", "_choices"]

    # Filter the dictionary and encode as a single JSON string per (rgi_id, exp_id)
    config = {k: v for k, v in p_config.attrs.items() if not any(k.endswith(suffix) for suffix in suffixes_to_exclude)}
    if "geometry.front_retreat.prescribed.file" not in config.keys():
        config["geometry.front_retreat.prescribed.file"] = "false"

    config_json = json.dumps(OrderedDict(sorted(config.items())))
    pism_config = xr.DataArray(
        np.array([[config_json]], dtype=object),
        dims=[rgi_dim, exp_dim],
        coords={rgi_dim: [m_rgi_id], exp_dim: [m_exp_id]},
        name="pism_config",
    )
    ds = ds.assign_coords(pism_config=pism_config)

    return ds.drop_vars(drop_vars, errors="ignore").drop_dims(drop_dims, errors="ignore")


def normalize_timeseries(
    ds: xr.Dataset, variables: str | list[str], reference_date: str | cftime.datetime
) -> xr.Dataset:
    """
    Normalize variables in an xarray Dataset by subtracting their values at a reference year.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the cumulative variables to be normalized.
    variables : str or list of str
        The name(s) of the cumulative variables to be normalized.
    reference_date : str or date-like
        The reference date to use for normalization.

    Returns
    -------
    xr.Dataset
        The xarray Dataset with normalized variables.

    Examples
    --------
    >>> import xarray as xr
    >>> import pandas as pd
    >>> time = pd.date_range("1990-01-01", "1995-01-01", freq="A")
    >>> data = xr.Dataset({
    ...     "cumulative_var": ("time", [10, 20, 30, 40, 50, 60]),
    ... }, coords={"time": time})
    >>> normalize_cumulative_variables(data, "cumulative_var", reference_date="1992-01-01")
    <xarray.Dataset>
    Dimensions:         (time: 6)
    Coordinates:
      * time            (time) datetime64[ns] 1990-12-31 1991-12-31 ... 1995-12-31
    Data variables:
        cumulative_var  (time) int64 0 10 20 30 40 50
    """

    ds[variables] -= ds[variables].sel(time=reference_date, method="nearest")
    return ds


def standardize_variable_names(ds: xr.Dataset, name_dict: Mapping[Any, Hashable] | None) -> xr.Dataset:
    """
    Standardize variable names in an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset whose variable names need to be standardized.
    name_dict : Mapping[Any, Hashable] or None
        A dictionary mapping the current variable names to the new standardized names.
        If None, no renaming is performed.

    Returns
    -------
    xr.Dataset
        The xarray Dataset with standardized variable names.

    Examples
    --------
    >>> import xarray as xr
    >>> ds = xr.Dataset({'temp': ('x', [1, 2, 3]), 'precip': ('x', [4, 5, 6])})
    >>> name_dict = {'temp': 'temperature', 'precip': 'precipitation'}
    >>> standardize_variable_names(ds, name_dict)
    <xarray.Dataset>
    Dimensions:      (x: 3)
    Dimensions without coordinates: x
    Data variables:
        temperature   (x) int64 1 2 3
        precipitation (x) int64 4 5 6
    """
    return ds.rename_vars(name_dict)
