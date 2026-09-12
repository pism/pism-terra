# Copyright (C) 2023 Andy Aschwanden
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
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

# pylint: disable=too-many-positional-arguments

"""
Module for filtering (calibration).
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import xarray as xr

from pism_terra.likelihood import log_normal_xr

try:
    import pint_xarray  # pylint: disable=unused-import
except ImportError:  # pragma: no cover - exercised only where pint-xarray is absent
    pint_xarray = None  # pylint: disable=invalid-name


def sample_with_replacement(weights: np.ndarray, exp_id: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """
    Sample with replacement from exp_id based on the given weights.

    Parameters
    ----------
    weights : np.ndarray
        The probabilities associated with each entry in exp_id.
    exp_id : np.ndarray
        The array of experiment IDs to sample from.
    n_samples : int
        The number of samples to draw.
    seed : int
        The random seed for reproducibility.

    Returns
    -------
    np.ndarray
        An array of sampled experiment IDs.
    """
    rng = np.random.default_rng(seed)
    try:
        ids = rng.choice(exp_id, size=n_samples, p=weights)
    except ValueError:
        ids = rng.choice(exp_id, size=n_samples)
    return ids


def sample_with_replacement_xr(weights, n_samples: int = 100, seed: int = 0, dim="exp_id") -> xr.DataArray:
    """
    Sample with replacement from a DataArray along a specified dimension.

    Parameters
    ----------
    weights : xr.DataArray
        The DataArray containing the weights for sampling.
    n_samples : int, optional
        The number of samples to draw, by default 100.
    seed : int, optional
        The random seed for reproducibility, by default 0.
    dim : str, optional
        The dimension along which to sample, by default "exp_id".

    Returns
    -------
    xr.DataArray
        A DataArray with the sampled values along the specified dimension.
    """
    da = xr.apply_ufunc(
        sample_with_replacement,
        weights.chunk({dim: -1}),
        input_core_dims=[[dim]],
        output_core_dims=[["sample"]],
        vectorize=True,
        dask="parallelized",
        kwargs={
            # ``sample_with_replacement`` names its identifier argument ``exp_id``
            # whatever dimension is being resampled.
            "exp_id": weights[dim].to_numpy(),
            "n_samples": n_samples,
            "seed": seed,
        },
        dask_gufunc_kwargs={"output_sizes": {"sample": n_samples}},
    )
    da.name = dim + "_sampled"
    return da


def _is_quantified(ds: xr.Dataset, names: list[str]) -> bool:
    """
    Tell whether any of ``names`` in ``ds`` carries pint units.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to inspect.
    names : list of str
        Variables to check; names missing from ``ds`` are skipped.

    Returns
    -------
    bool
        ``True`` when pint-xarray is importable and at least one variable is a pint quantity;
        always ``False`` without pint-xarray, so plain data takes the plain code path.
    """
    if pint_xarray is None:
        return False
    return any(ds[name].pint.units is not None for name in names if name in ds)


def _strip_units(
    simulated: xr.Dataset, observed: xr.Dataset, sim_var: str, obs_mean_var: str, obs_std_var: str
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Express quantified input in the observed mean's units and hand back plain arrays.

    The simulated variable and the observed uncertainty are converted to the units of the
    observed mean, then every quantity is stripped back into a ``units`` attribute so the
    likelihood works on ordinary arrays exactly as it does for unquantified input.

    Parameters
    ----------
    simulated : xr.Dataset
        Simulated ensemble; quantified or plain.
    observed : xr.Dataset
        Observations; quantified or plain.
    sim_var : str
        Simulated variable to compare.
    obs_mean_var : str
        Observed mean; its units are the common units.
    obs_std_var : str
        Observed uncertainty, converted to the same units.

    Returns
    -------
    tuple of xr.Dataset
        ``(simulated, observed)`` without pint quantities.
    """
    simulated = simulated.pint.quantify()
    observed = observed.pint.quantify()
    target = observed[obs_mean_var].pint.units
    if target is not None:
        simulated = simulated.pint.to({sim_var: target})
        observed = observed.pint.to({obs_std_var: target})
    return simulated.pint.dequantify(), observed.pint.dequantify()


def importance_sampling(
    simulated: xr.Dataset,
    observed: xr.Dataset,
    log_likelihood: Callable = log_normal_xr,
    likelihood_kwargs: dict | None = None,
    dim: str = "exp_id",
    sum_dims: list = ["time"],
    fudge_factor: float = 3.0,
    n_samples: int = 100,
    obs_mean_var: str = "mass_balance",
    obs_std_var: str = "mass_balance_uncertainty",
    sim_var: str = "mass_balance",
    seed: int = 0,
    compute: bool = True,
) -> xr.Dataset:
    """
    Filter an ensemble of simulated data to match observed data using a likelihood-based approach.

    Parameters
    ----------
    simulated : xr.Dataset
        An xarray Dataset containing the simulated data, on the same coordinates as
        ``observed`` along every shared dimension. Interpolate or reindex it first
        (``simulated.interp_like(observed)``); no interpolation happens here.
    observed : xr.Dataset
        An xarray Dataset containing the observed data.
    log_likelihood : Callable, optional
        The log-likelihood function to use for filtering, by default log_normal_xr.
    likelihood_kwargs : dict, optional
        Additional keyword arguments to pass to the log-likelihood function, by default {}.
    dim : str, optional
        The variable name in `simulated` that identifies each ensemble member, by default "exp_id".
    sum_dims : list, optional
        The dimensions to sum over when computing the log-likelihood, by default ["time"].
    fudge_factor : float, optional
        A multiplicative factor applied to the observed standard deviation to widen the likelihood function,
        allowing for greater tolerance in the matching process, by default 3.0.
    n_samples : int, optional
        The number of samples to draw from the simulated ensemble, by default 100.
    obs_mean_var : str, optional
        The variable name in `observed` that represents the mean observed data, by default "mass_balance".
    obs_std_var : str, optional
        The variable name in `observed` that represents the observed data's standard deviation,
        by default "mass_balance_uncertainty".
    sim_var : str, optional
        The variable name in `simulated` that represents the simulated data to be resampled,
        by default "mass_balance".
    seed : int, optional
        The random seed for reproducibility, by default 0.
    compute : bool, optional
        Set to True if you want to force compute the result.

    Returns
    -------
    xr.Dataset
        A dataset containing the selected members, log_likes, and weights the filtering process.

    Raises
    ------
    ValueError
        If ``simulated[sim_var]`` and the observed variables do not share their coordinates.

    Notes
    -----
    This function implements a filtering algorithm that uses a likelihood-based approach to select ensemble members
    from the simulated dataset that are most consistent with the observed data. The likelihood is computed based on
    the difference between the simulated and observed means, scaled by the observed standard deviation (adjusted by
    the fudge factor). This method allows for the incorporation of observational uncertainty into the ensemble
    selection process.

    If pint-xarray is installed and ``simulated`` or ``observed`` carries pint quantities
    (``.pint.quantify()``), the simulated variable and the observed uncertainty are converted
    to the units of the observed mean before the likelihood is evaluated. Without pint-xarray,
    or with plain arrays, nothing changes.
    """

    # Quantified input (pint-xarray) comes back as plain arrays in the observed mean's units.
    if _is_quantified(observed, [obs_mean_var, obs_std_var]) or _is_quantified(simulated, [sim_var]):
        simulated, observed = _strip_units(simulated, observed, sim_var, obs_mean_var, obs_std_var)

    # The caller aligns the grids; refuse silently broadcasting mismatched coordinates.
    try:
        xr.align(simulated[sim_var], observed[obs_mean_var], observed[obs_std_var], join="exact")
    except ValueError as err:
        raise ValueError(
            f"'{sim_var}' and '{obs_mean_var}'/'{obs_std_var}' must share their coordinates: "
            "interpolate or reindex the simulated ensemble onto the observed grid first, "
            "e.g. simulated.interp_like(observed) or simulated.pint.interp_like(observed)"
        ) from err

    # Calculate the observed mean and adjusted standard deviation
    obs_mean = observed[obs_mean_var]
    obs_std = observed[obs_std_var]

    # Extract the simulated data
    sim = simulated[sim_var]

    if likelihood_kwargs is None:
        likelihood_kwargs = {}

    # Compute the log-likelihood of each simulated data point
    log_likes = log_likelihood(
        sim,
        obs_mean,
        obs_std,
        fudge_factor=fudge_factor,
        sum_dims=sum_dims,
        **likelihood_kwargs,
    )
    log_likes_scaled = log_likes - log_likes.max(dim=dim)
    # Convert log-likelihoods to weights
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"overflow encountered")
        weights = np.exp(log_likes_scaled)
    weights /= weights.sum(dim=dim)
    weights.name = "weights"

    samples = sample_with_replacement_xr(weights, n_samples=n_samples, seed=seed, dim=dim)
    ds = xr.merge([log_likes, weights])
    ds[samples.name] = samples

    if compute:
        ds = ds.compute()
    return ds
