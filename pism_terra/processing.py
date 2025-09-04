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

"""
Processing Functions.
"""

import re
from collections import OrderedDict

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
    regexp: str = "id_(.+?)_",
    dim: str = "exp_id",
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
    regexp : str, optional
        The regular expression pattern to extract the experiment identifier from the filename, by default "id_(.+?)_".
    dim : str, optional
        The name of the new dimension to be added to the dataset, by default "exp_id".
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

    if dim not in ds.dims:
        m_id_re = re.search(regexp, ds.encoding["source"])
        ds = ds.expand_dims(dim)
        assert m_id_re is not None
        m_id: str | int
        try:
            m_id = int(m_id_re.group(1))
        except:
            m_id = str(m_id_re.group(1))
        ds[dim] = [m_id]

    p_config = ds["pism_config"]

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
        pc_vals.reshape(-1, 1),
        dims=["pism_config_axis", dim],
        coords={"pism_config_axis": pc_keys, dim: [m_id]},
        name="pism_config",
    )
    ds = xr.merge(
        [
            ds.drop_vars(["pism_config"], errors="ignore").drop_dims(["pism_config_axis"], errors="ignore"),
            pism_config,
        ]
    )
    return ds.drop_vars(drop_vars, errors="ignore").drop_dims(drop_dims, errors="ignore")
