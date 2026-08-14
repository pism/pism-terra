"""
Tests for the KITP calibration metric.

The block-bootstrap RMSE was rewritten as a streaming ``coarsen`` reduction so
that a dask-backed ensemble is never materialised in memory. These tests pin
the reduction to a straightforward tiling loop and check that the lazy and
eager paths agree exactly.
"""

import numpy as np
import pytest
import xarray as xr

from pism_terra.kitp.calibrate import (
    block_bootstrap_rmse,
    decorrelation_length,
    squared_error_blocks,
)


def reference_blocks(sim, obs, block_size):
    """
    Sum squared error over blocks with an explicit tiling loop.

    This is the original implementation, kept here as an oracle for the
    ``coarsen``-based reduction.

    Parameters
    ----------
    sim : xarray.DataArray
        Per-experiment simulated field with dims ``(exp_id, y, x)``.
    obs : xarray.DataArray
        Observed field with dims ``(y, x)``.
    block_size : int
        Block side in pixels.

    Returns
    -------
    block_sums : numpy.ndarray
        Summed squared error, shape ``(n_exp, n_blocks)``.
    block_counts : numpy.ndarray
        Contributing cells per block, shape ``(n_blocks,)``.
    """
    sim_v = np.asarray(sim.values, dtype=float)
    obs_v = np.asarray(obs.values, dtype=float)
    sq_err = (sim_v - obs_v[None, :, :]) ** 2
    valid = np.isfinite(sq_err).all(axis=0)
    ny, nx = obs_v.shape
    by = max(1, ny // block_size)
    bx = max(1, nx // block_size)
    block_sums = np.zeros((sim_v.shape[0], by, bx))
    block_counts = np.zeros((by, bx), dtype=int)
    for i in range(by):
        for j in range(bx):
            ys = slice(i * block_size, (i + 1) * block_size)
            xs = slice(j * block_size, (j + 1) * block_size)
            v = valid[ys, xs]
            block_counts[i, j] = int(v.sum())
            block_sums[:, i, j] = np.where(v, sq_err[:, ys, xs], 0.0).sum(axis=(1, 2))
    return block_sums.reshape(sim_v.shape[0], -1), block_counts.reshape(-1)


@pytest.fixture(name="fields")
def fixture_fields():
    """
    Build a small ensemble and observation with a ragged NaN footprint.

    Returns
    -------
    tuple of xarray.DataArray
        ``(sim, obs)`` with dims ``(exp_id, y, x)`` and ``(y, x)``.
    """
    rng = np.random.default_rng(42)
    ny, nx, n_exp = 37, 23, 4
    obs_v = rng.normal(size=(ny, nx))
    sim_v = obs_v[None, :, :] + rng.normal(scale=0.5, size=(n_exp, ny, nx))
    # An off-centre hole plus a fully masked row, so blocks differ in count
    # and one block is empty.
    sim_v[:, 5:12, 3:9] = np.nan
    obs_v[0, :] = np.nan
    sim = xr.DataArray(
        sim_v,
        dims=["exp_id", "y", "x"],
        coords={"exp_id": np.arange(n_exp), "y": np.arange(ny), "x": np.arange(nx)},
    )
    obs = xr.DataArray(obs_v, dims=["y", "x"], coords={"y": np.arange(ny), "x": np.arange(nx)})
    return sim, obs


@pytest.mark.parametrize("block_size", [1, 3, 8, 16])
def test_squared_error_blocks_matches_tiling_loop(fields, block_size):
    """
    The coarsen reduction reproduces the explicit tiling loop.

    Parameters
    ----------
    fields : tuple of xarray.DataArray
        The ``(sim, obs)`` pair from the fixture.
    block_size : int
        Block side under test.
    """
    sim, obs = fields
    sums, counts = squared_error_blocks(sim, obs, block_size)
    ref_sums, ref_counts = reference_blocks(sim, obs, block_size)

    np.testing.assert_array_equal(counts, ref_counts)
    np.testing.assert_allclose(sums, ref_sums, rtol=1e-12, atol=0)


def test_squared_error_blocks_is_lazy_and_reads_once(fields):
    """
    A dask-backed ensemble gives the same blocks as an in-memory one.

    Parameters
    ----------
    fields : tuple of xarray.DataArray
        The ``(sim, obs)`` pair from the fixture.
    """
    sim, obs = fields
    lazy = sim.chunk({"exp_id": 1, "y": 8, "x": 8})
    assert lazy.chunks is not None

    lazy_sums, lazy_counts = squared_error_blocks(lazy, obs, 5)
    eager_sums, eager_counts = squared_error_blocks(sim, obs, 5)

    np.testing.assert_array_equal(lazy_counts, eager_counts)
    np.testing.assert_allclose(lazy_sums, eager_sums, rtol=1e-12, atol=0)


def test_squared_error_blocks_block_larger_than_domain(fields):
    """
    A block wider than the grid collapses to a single block.

    Parameters
    ----------
    fields : tuple of xarray.DataArray
        The ``(sim, obs)`` pair from the fixture.
    """
    sim, obs = fields
    sums, counts = squared_error_blocks(sim, obs, 10_000)

    assert counts.shape == (1,)
    assert sums.shape == (sim.sizes["exp_id"], 1)
    # Every cell finite in all experiments contributes exactly once.
    expected = int(np.isfinite((sim - obs).values).all(axis=0).sum())
    assert int(counts[0]) == expected


def test_block_bootstrap_rmse_lazy_matches_eager(fields):
    """
    The public RMSE entry point is unchanged by chunking.

    Parameters
    ----------
    fields : tuple of xarray.DataArray
        The ``(sim, obs)`` pair from the fixture.
    """
    sim, obs = fields
    eager = block_bootstrap_rmse(sim, obs, 5, n_boot=50, seed=3)
    lazy = block_bootstrap_rmse(sim.chunk({"y": 9}), obs, 5, n_boot=50, seed=3)

    assert eager.dims == ("exp_id", "boot")
    np.testing.assert_array_equal(eager.exp_id.values, sim.exp_id.values)
    np.testing.assert_allclose(lazy.values, eager.values, rtol=1e-12, atol=0)


def test_decorrelation_length_returns_nan_without_finite_values():
    """
    An all-NaN field has no decorrelation length.
    """
    assert np.isnan(decorrelation_length(np.full((8, 8), np.nan), 1000.0))
