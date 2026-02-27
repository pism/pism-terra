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

import re
from collections import OrderedDict
from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import xarray as xr


def preprocess_nc(
    ds: xr.Dataset,
    regexp: str = "id_(.+?)_",
    dim: str = "exp_id",
    drop_vars: list[str] | None = None,
    drop_dims: list[str] | None = None,
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
    regexp : str, optional
        The regular expression pattern to extract the experiment identifier from the filename,
        by default "id_(.+?)_".
    dim : str, optional
        The name of the new dimension to be added to the dataset, by default "exp_id".
    drop_vars : list of str or None, optional
        A list of variable names to be dropped from the dataset, by default None.
    drop_dims : list of str or None, optional
        A list of dimension names to be dropped from the dataset, by default None.

    Returns
    -------
    xarray.Dataset
        The processed dataset with the experiment identifier added as a new dimension,
        and specified variables and dimensions dropped.

    Raises
    ------
    AssertionError
        If the regular expression does not match any part of the filename.

    Notes
    -----
    If `drop_dims` is not provided, it defaults to `["nv4"]`.
    """
    if drop_dims is None:  # Initialize drop_dims if not provided
        drop_dims = ["nv4"]

    m_id_re = re.search(regexp, ds.encoding["source"])
    ds = ds.expand_dims(dim)
    assert m_id_re is not None
    m_id: str | int
    try:
        m_id = int(m_id_re.group(1))
    except ValueError:  # Catch specific exception
        m_id = str(m_id_re.group(1))
    ds[dim] = [m_id]

    return ds.drop_vars(drop_vars, errors="ignore").drop_dims(drop_dims, errors="ignore")


def preprocess_config(
    ds,
    exp_regexp: str = "id_(.+?)_",
    rgi_regexp: str = "(RGI2000-v7\.0-C-[^/\s]+)",
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
        The regular expression pattern to extract the RGI identifier from the filename, by default "(RGI2000-v7\.0-C-[^/\s]+)".
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
    ds = ds.expand_dims({rgi_dim: [m_rgi_id], exp_dim: [m_exp_id]})

    # List of suffixes to exclude
    suffixes_to_exclude = ["_doc", "_type", "_units", "_option", "_choices"]

    # Filter the dictionary
    config = {k: v for k, v in p_config.attrs.items() if not any(k.endswith(suffix) for suffix in suffixes_to_exclude)}
    if "geometry.front_retreat.prescribed.file" not in config.keys():
        config["geometry.front_retreat.prescribed.file"] = "false"

    config_sorted = OrderedDict(sorted(config.items()))

    pc_keys = np.array(list(config_sorted.keys()))
    pc_vals = np.array(list(config_sorted.values()))

    pism_config = xr.DataArray(
        pc_vals.reshape(-1),
        dims=["pism_config_axis"],
        coords={"pism_config_axis": pc_keys},
        name="pism_config",
    )

    ds = xr.merge(
        [
            ds.drop_vars(["pism_config"], errors="ignore").drop_dims(["pism_config_axis"], errors="ignore"),
            pism_config,
        ]
    )
    return ds.drop_vars(drop_vars, errors="ignore").drop_dims(drop_dims, errors="ignore")


def normalize_cumulative_variables(
    ds: xr.Dataset,
    variables: str | list[str] | None = None,
    reference_date: str = "1992-01-01",
) -> xr.Dataset:
    """
    Normalize cumulative variables in an xarray Dataset by subtracting their values at a reference year.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the cumulative variables to be normalized.
    variables : str or list of str
        The name(s) of the cumulative variables to be normalized.
    reference_date : str, optional
        The reference date to use for normalization. Default is "1992-01-01".

    Returns
    -------
    xr.Dataset
        The xarray Dataset with normalized cumulative variables.

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

    if variables is not None:
        ds[variables] -= ds[variables].sel(time=reference_date, method="nearest")
    else:
        pass
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
