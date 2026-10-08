# Copyright (C) 2026 Andy Aschwanden
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

"""
Tests for the particle tracing of :mod:`pism_terra.glacier.render_terrain_3d`.

Covers the pieces that need no renderer: velocity units, the advection step,
tracing with respawns, and trails that never reach back past a respawn.
"""

from __future__ import annotations

import numpy as np
import pytest

# The renderer is a development tool: PyVista is in environment-dev.yml only,
# so the test environment may not have it.
pytest.importorskip("pyvista")

# pylint: disable=wrong-import-position
from pism_terra.glacier.render_terrain_3d import (  # noqa: E402
    SECONDS_PER_YEAR,
    GridSampler,
    advect,
    particle_trails,
    per_year,
    trace_particles,
)

#: 200 m grid, 30 x 20 cells, y increasing as PISM writes it.
X = 200.0 * np.arange(30)
Y = 200.0 * np.arange(20)


def uniform(u: float, v: float, ice: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A uniform velocity field on the test grid.

    Parameters
    ----------
    u, v : float
        Velocity components, m/yr.
    ice : numpy.ndarray or None, optional
        Ice mask; everywhere when None.

    Returns
    -------
    tuple of numpy.ndarray
        ``(u, v, ice)`` as :func:`trace_particles` takes them.
    """
    shape = (Y.size, X.size)
    return np.full(shape, u), np.full(shape, v), np.ones(shape, bool) if ice is None else ice


def test_per_year_converts_only_per_second_units():
    """Per-second velocities are converted to m/yr; per-year ones are left alone."""
    values = np.array([1.0, 2.0])
    for units in ("m s-1", "m s^-1", "m/s", "m second^-1"):
        np.testing.assert_allclose(per_year(values, units), values * SECONDS_PER_YEAR, err_msg=units)
    for units in ("m year^-1", "m/yr", "m year-1", ""):
        np.testing.assert_allclose(per_year(values, units), values, err_msg=units)


def test_advect_moves_with_a_uniform_flow():
    """One step in a uniform flow moves every point by velocity times step."""
    u, v, _ = uniform(100.0, -50.0)
    pos = np.array([[1000.0, 2000.0], [3100.0, 1500.0]])
    moved = advect(pos, u, v, GridSampler(X, Y), dt=2.0)
    np.testing.assert_allclose(moved, pos + [200.0, -100.0])


def test_advect_off_the_grid_is_missing():
    """A point carried outside the grid comes back NaN, so it can respawn."""
    u, v, _ = uniform(1000.0, 0.0)
    moved = advect(np.array([[X[-1] - 10.0, 2000.0]]), u, v, GridSampler(X, Y), dt=1.0)
    assert not np.isfinite(moved).all()


def test_trace_particles_follows_the_flow_and_respawns_on_ice():
    """Between respawns a particle moves by u*step; respawned ones land on ice."""
    ice = np.zeros((Y.size, X.size), bool)
    ice[:, 5:25] = True
    frames = [uniform(50.0, 0.0, ice)] * 30
    positions, births = trace_particles(frames, X, Y, n=200, step=2.0, life=10, min_speed=1.0, seed=1)
    assert positions.shape == (30, 200, 2) and births.shape == (30, 200)

    same = births[1:] == births[:-1]
    np.testing.assert_allclose((positions[1:, :, 0] - positions[:-1, :, 0])[same], 100.0, atol=1e-3)
    np.testing.assert_allclose((positions[1:, :, 1] - positions[:-1, :, 1])[same], 0.0, atol=1e-3)
    # Every position shown lies on the ice strip, and particles do respawn.
    assert (positions[..., 0] >= X[5] - 100.0).all() and (positions[..., 0] <= X[24] + 100.0).all()
    assert (births > 0).any()
    # Lifetimes start staggered, so the particles do not all respawn together.
    assert len(np.unique(births[-1])) > 1


def test_trace_particles_skips_slow_ice():
    """No particle lives where the ice is slower than the minimum speed."""
    frames = [uniform(0.1, 0.0)] * 5
    positions, _ = trace_particles(frames, X, Y, n=50, step=1.0, life=10, min_speed=1.0)
    assert not np.isfinite(positions).any()


def test_trails_stop_at_a_respawn():
    """A trail spans at most ``trail`` frames and never reaches back past a respawn."""
    frames, n = 6, 2
    positions = np.zeros((frames, n, 2), np.float32)
    positions[:, :, 0] = 1000.0 + 100.0 * np.arange(frames)[:, None]
    positions[:, :, 1] = 2000.0
    births = np.zeros((frames, n), np.int32)
    births[4:, 1] = 4  # the second particle respawned at frame 4
    height = np.full((Y.size, X.size), 500.0)

    poly = particle_trails(positions, births, 5, trail=3, height=height, sample=GridSampler(X, Y))
    assert poly is not None
    lengths = []
    cells, i = poly.lines, 0
    while i < len(cells):
        lengths.append(int(cells[i]))
        i += cells[i] + 1
    assert sorted(lengths) == [2, 3]
    np.testing.assert_allclose(poly.points[:, 2], 500.0)
    # The fade rises from the tail to 1 at the head of each trail.
    assert poly["fade"].max() == pytest.approx(1.0)

    # A particle born in this frame has no trail yet.
    births[5, :] = 5
    assert particle_trails(positions, births, 5, trail=3, height=height, sample=GridSampler(X, Y)) is None
