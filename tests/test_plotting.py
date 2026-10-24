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
Tests for the hillshade blending in :mod:`pism_terra.plotting`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pism_terra.plotting import blended_animation, blended_frames, hillshade


def ramp(y_ascending: bool, n_time: int = 3) -> xr.DataArray:
    """
    Build a surface rising to the south-east, on ``(time, y, x)``.

    Parameters
    ----------
    y_ascending : bool
        Store ``y`` south to north (as PISM does) or north to south.
    n_time : int, default 3
        Number of time steps.

    Returns
    -------
    xarray.DataArray
        Heights in m on a 100 m grid, missing in one corner cell.
    """
    x = 100.0 * np.arange(8)
    y = 100.0 * np.arange(6)
    y = y if y_ascending else y[::-1]
    z = np.add.outer(-y, x) / 10.0  # higher to the east and to the south
    z[0, 0] = np.nan
    time = pd.date_range("1990-01-01", periods=n_time, freq="MS") + pd.Timedelta(days=15)
    data = np.stack([z * (1 + i) for i in range(n_time)])
    return xr.DataArray(data, dims=("time", "y", "x"), coords={"time": time, "y": y, "x": x}, name="usurf")


def test_hillshade_does_not_depend_on_the_row_order():
    """
    Shade the same cells the same way whichever way ``y`` runs, missing where the surface is.
    """
    up = hillshade(ramp(True))
    down = hillshade(ramp(False))

    assert up.dims == ("time", "y", "x")
    # The ramp's missing corner sits in a different cell for each order; compare away from both.
    np.testing.assert_allclose(
        up.isel(y=slice(1, -1), x=slice(1, None)).values,
        down.sortby("y").isel(y=slice(1, -1), x=slice(1, None)).values,
    )
    assert np.isnan(up.isel(y=0, x=0)).all()
    assert float(up.min()) >= 0.0 and float(up.max()) <= 1.0


def test_hillshade_is_on_an_absolute_scale():
    """
    Shade flat ground at ``sin(altitude)``, slopes towards the light brighter, away darker.
    """
    surface = ramp(True)
    flat = hillshade(xr.zeros_like(surface), altdeg=45.0)
    np.testing.assert_allclose(flat.values[np.isfinite(flat.values)], np.sin(np.radians(45.0)))

    # The ramp rises to the south-east, so its slopes face the north-western light.
    lit = hillshade(surface, azdeg=315.0)
    dark = hillshade(surface, azdeg=135.0)
    assert float(lit.isel(time=0, y=3, x=3)) > np.sin(np.radians(45.0)) > float(dark.isel(time=0, y=3, x=3))
    # Steeper in later frames: brighter on the lit side, so the scale is shared, not stretched per frame.
    assert float(lit.isel(time=2, y=3, x=3)) > float(lit.isel(time=0, y=3, x=3))


def test_blended_frames_keep_the_shade_where_there_is_no_data():
    """
    Show the hillshade grey where there is no data, and nothing outside the surface.
    """
    surface = ramp(True)
    shade = hillshade(surface)
    data = xr.full_like(surface, 50.0)
    data[:, 2, 3] = np.nan

    images = blended_frames(data, shade, cmap="viridis", clim=(0.0, 100.0))

    assert images.shape == (3, 6, 8, 4)
    grey = shade.values[:, 2, 3]
    np.testing.assert_allclose(images[:, 2, 3, :3], np.stack([grey] * 3, axis=-1))
    assert (images[:, 0, 0, 3] == 0.0).all()
    assert (images[:, 1:, 1:, 3] == 1.0).all()
    # With data, the colour is darkened by the shade, never brightened.
    assert (images[:, 4, 5, :3] <= 1.0).all()
    # A static hillshade serves every frame.
    static = blended_frames(data, shade.isel(time=0), cmap="viridis", clim=(0.0, 100.0))
    np.testing.assert_allclose(static[0], images[0])


def test_blended_frames_on_a_log_scale():
    """
    Treat values at or below zero as missing on a log scale.
    """
    surface = ramp(True)
    data = xr.full_like(surface, 10.0)
    data[:, 2, 3] = 0.0

    images = blended_frames(data, hillshade(surface), cmap="viridis", clim=(1.0, 100.0), log=True)

    np.testing.assert_allclose(images[:, 2, 3, 0], hillshade(surface).values[:, 2, 3])


def test_blended_animation_titles_every_frame_with_its_date():
    """
    One frame per time step, titled with its date, and the last one repeated for the scrubber.
    """
    pytest.importorskip("holoviews")
    pytest.importorskip("panel")
    surface = ramp(True)
    data = xr.full_like(surface, 5.0).rename("velsurf_mag").assign_attrs(units="m/yr")

    pane = blended_animation(data, surface, cmap="viridis", clim=(0.0, 10.0), embed=False)

    animation = pane[0].object
    keys = [k[0] if isinstance(k, tuple) else k for k in animation.keys()]
    assert keys == ["1990-01-16", "1990-02-16", "1990-03-16", "1990-03-16 "]
    assert animation["1990-02-16"].opts.get().kwargs["title"] == "1990-02-16"
