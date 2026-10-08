"""Reductions of the per-cell log-likelihood."""

import numpy as np
import pytest
import xarray as xr

from pism_terra.likelihood import log_normal_xr, reduce_log_likelihood


@pytest.fixture(name="cells")
def fixture_cells() -> xr.DataArray:
    """
    A ``(member, y, x)`` per-cell log-likelihood with one masked column.

    Returns
    -------
    xr.DataArray
        Values ``-1`` for member 0 and ``-2`` for member 1; ``x = 3`` is NaN.
    """
    da = xr.DataArray(
        np.stack([np.full((4, 4), -1.0), np.full((4, 4), -2.0)]),
        dims=["member", "y", "x"],
        coords={"member": [0, 1], "y": np.arange(4), "x": np.arange(4)},
    )
    return da.where(da.x != 3)


def test_mean_sum_and_blocks_relate_as_documented(cells: xr.DataArray) -> None:
    """
    Mean is the per-cell value, sum is the total over valid cells, blocks divides by the block area.

    Parameters
    ----------
    cells : xr.DataArray
        Fixture.
    """
    mean = reduce_log_likelihood(cells, ["y", "x"], reduction="mean")
    total = reduce_log_likelihood(cells, ["y", "x"], reduction="sum")
    blocks = reduce_log_likelihood(cells, ["y", "x"], reduction="blocks", block_size=2)
    np.testing.assert_allclose(mean, [-1.0, -2.0])
    np.testing.assert_allclose(total, [-12.0, -24.0])  # 12 valid cells
    np.testing.assert_allclose(blocks, total / 4.0)
    np.testing.assert_allclose(reduce_log_likelihood(cells, ["y", "x"], reduction="blocks", block_size=1), total)


def test_missing_dims_are_ignored_and_empty_members_give_nan(cells: xr.DataArray) -> None:
    """
    A dimension absent from the array is skipped, and an all-NaN member reduces to NaN, not 0.

    Parameters
    ----------
    cells : xr.DataArray
        Fixture.
    """
    total = reduce_log_likelihood(cells, ["time", "y", "x"], reduction="sum")
    np.testing.assert_allclose(total, [-12.0, -24.0])
    empty = cells.where(cells.member == 0)
    out = reduce_log_likelihood(empty, ["y", "x"], reduction="blocks", block_size=3)
    assert np.isnan(out.sel(member=1))
    with pytest.raises(ValueError, match="reduction must be one of"):
        reduce_log_likelihood(cells, ["y", "x"], reduction="median")


def test_log_normal_xr_defaults_to_mean_and_forwards_the_reduction() -> None:
    """The xarray likelihood keeps its mean default and accepts the reduction keywords."""
    x = xr.DataArray(np.zeros((2, 3)), dims=["y", "x"])
    mu = xr.DataArray(np.ones((2, 3)), dims=["y", "x"])
    per_cell = -0.5 - 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(log_normal_xr(x, mu, 1.0, fudge_factor=1.0, sum_dims=["y", "x"]), per_cell)
    np.testing.assert_allclose(
        log_normal_xr(x, mu, 1.0, fudge_factor=1.0, sum_dims=["y", "x"], reduction="blocks", block_size=2),
        6 * per_cell / 4,
    )
