"""Importance sampling on plain and pint-quantified data."""

from typing import Any

import numpy as np
import pytest
import xarray as xr

from pism_terra import filtering
from pism_terra.filtering import importance_sampling


def _datasets(sim_units: str = "m", sim_scale: float = 1.0) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Build a three-member ensemble on a coarse grid and observations on a finer one.

    Member 1 matches the observations, member 0 is biased low and member 2 high.

    Parameters
    ----------
    sim_units : str
        Units attribute of the simulated variable.
    sim_scale : float
        Factor the simulated values are multiplied by (e.g. 100 to express metres in cm).

    Returns
    -------
    tuple of xr.Dataset
        ``(simulated, observed)``.
    """
    x_obs = np.arange(0.0, 1000.0, 100.0)
    y_obs = np.arange(0.0, 800.0, 100.0)
    obs = xr.Dataset(
        {
            "dh": (("y", "x"), np.full((y_obs.size, x_obs.size), -2.0), {"units": "m"}),
            "dh_err": (("y", "x"), np.full((y_obs.size, x_obs.size), 0.5), {"units": "m"}),
        },
        coords={"x": ("x", x_obs, {"units": "m"}), "y": ("y", y_obs, {"units": "m"})},
    )
    x_sim = np.arange(0.0, 1000.0, 200.0)
    y_sim = np.arange(0.0, 800.0, 200.0)
    members = np.array([-3.0, -2.0, -1.0])[:, None, None] * np.ones((1, y_sim.size, x_sim.size))
    sim = xr.Dataset(
        {"usurf": (("exp_id", "y", "x"), members * sim_scale, {"units": sim_units})},
        coords={
            "exp_id": [0, 1, 2],
            "x": ("x", x_sim, {"units": "m"}),
            "y": ("y", y_sim, {"units": "m"}),
        },
    )
    return sim, obs


KW: dict[str, Any] = {
    "obs_mean_var": "dh",
    "obs_std_var": "dh_err",
    "sim_var": "usurf",
    "sum_dims": ["x", "y"],
    "n_samples": 50,
}


def test_plain_arrays_pick_the_matching_member() -> None:
    """Without units the matching member gets the largest weight and the weights sum to one."""
    sim, obs = _datasets()
    out = importance_sampling(sim, obs, **KW)
    assert out.weights.sum().item() == pytest.approx(1.0)
    assert int(out.weights.argmax("exp_id")) == 1


def test_quantified_input_is_interpolated_and_converted() -> None:
    """Quantified input in cm gives the same weights as plain input in m."""
    pytest.importorskip("pint_xarray")
    reference = importance_sampling(*_datasets(), **KW)
    sim, obs = _datasets(sim_units="cm", sim_scale=100.0)
    out = importance_sampling(sim.pint.quantify(), obs.pint.quantify(), **KW)
    np.testing.assert_allclose(out.weights.values, reference.weights.values)
    assert out.log_likelihood.attrs.get("units") in (None, "1", "")


def test_plain_path_without_pint_xarray(monkeypatch: pytest.MonkeyPatch) -> None:
    """With pint-xarray unavailable, plain arrays still work and no pint accessor is touched.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to hide pint-xarray from the module.
    """
    monkeypatch.setattr(filtering, "pint_xarray", None)
    reference_sim, reference_obs = _datasets()
    out = importance_sampling(reference_sim, reference_obs, **KW)
    assert int(out.weights.argmax("exp_id")) == 1
