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
"""
Plotting methods.
"""

from __future__ import annotations

from typing import Any

# Imported for their side effect: registering Crameri's "cmc.*" and the
# project's "cmg.*" colormaps, so they can be given by name.
import cmcrameri.cm  # noqa: F401  pylint: disable=unused-import
import cmglaciology.cm  # noqa: F401  pylint: disable=unused-import
import matplotlib
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm, Normalize
from scipy.ndimage import distance_transform_edt

fontsize = 6
rc_params = {
    "axes.linewidth": 0.15,
    "xtick.major.size": 2.0,
    "xtick.major.width": 0.15,
    "ytick.major.size": 2.0,
    "ytick.major.width": 0.15,
    "hatch.linewidth": 0.15,
    "font.size": fontsize,
    "font.family": "DejaVu Sans",
}


def blend_multiply(rgb: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """
    Combine an RGB image with an intensity map using "overlay" blending.

    This function combines an RGB image with an intensity map using "overlay" blending. The RGB image
    and the intensity map are combined by multiplying the RGB values by the intensity values. The resulting
    image is then scaled to have values between 0 and 1.

    Parameters
    ----------
    rgb : np.ndarray
        An (M, N, 3) RGB array of floats ranging from 0 to 1. This represents the color image.
    intensity : np.ndarray
        An (M, N, 1) array of floats ranging from 0 to 1. This represents the grayscale image.

    Returns
    -------
    np.ndarray
        An (M, N, 3) RGB array representing the combined images. The values in the array range from 0 to 1.
    """

    alpha = rgb[..., -1, np.newaxis]
    img_scaled = np.clip(rgb[..., :3] * intensity, 0.0, 1.0)
    return img_scaled * alpha + intensity * (1.0 - alpha)


def _fill_nearest(z: np.ndarray) -> np.ndarray:
    """
    Fill missing cells with the value of the nearest valid cell.

    Parameters
    ----------
    z : numpy.ndarray
        Two-dimensional field with missing values.

    Returns
    -------
    numpy.ndarray
        The field with every missing cell filled; zeros if none is valid.
    """
    missing = ~np.isfinite(z)
    if not missing.any():
        return z
    if missing.all():
        return np.zeros_like(z, dtype=float)
    nearest = distance_transform_edt(missing, return_distances=False, return_indices=True)
    return z[tuple(nearest)]


def hillshade(
    surface: xr.DataArray,
    *,
    azdeg: float = 315.0,
    altdeg: float = 45.0,
    vert_exag: float = 2.0,
) -> xr.DataArray:
    """
    Compute the hillshade of a surface, for every time step it has.

    The illumination is the cosine of the angle between the surface normal
    and the direction of the light, as GDAL's hillshade computes it: flat
    ground gets ``sin(altdeg)`` and every frame is shaded on the same
    absolute scale. (Matplotlib's ``LightSource.hillshade`` instead stretches
    each frame to its own range, which makes an animation flicker.)

    Parameters
    ----------
    surface : xarray.DataArray
        Surface elevation on ``(y, x)``, optionally with further dimensions
        such as ``time``; ``y`` increasing northwards or southwards.
    azdeg : float, default 315
        Direction the light comes from, degrees clockwise from north.
    altdeg : float, default 45
        Angle of the light above the horizon, degrees.
    vert_exag : float, default 2
        Vertical exaggeration applied before shading.

    Returns
    -------
    xarray.DataArray
        Illumination from 0 (shadow) to 1 (lit), named ``hillshade``, with the
        dimensions of ``surface``; missing wherever the surface is.
    """
    x = surface["x"].values.astype(float)
    y = surface["y"].values.astype(float)
    az, alt = np.radians(azdeg), np.radians(altdeg)
    # Towards the light, in (east, north, up).
    light = np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])

    def one(z: np.ndarray) -> np.ndarray:
        """
        Shade one ``(y, x)`` slice.

        Parameters
        ----------
        z : numpy.ndarray
            Heights; missing ones take the nearest valid height, so the
            margin of the surface is not shaded as a cliff.

        Returns
        -------
        numpy.ndarray
            Illumination in ``[0, 1]``.
        """
        # Derivatives against the coordinates themselves, so the row order does not matter.
        dzdy, dzdx = np.gradient(_fill_nearest(z), y, x)
        nx, ny = -vert_exag * dzdx, -vert_exag * dzdy
        intensity = (nx * light[0] + ny * light[1] + light[2]) / np.sqrt(nx**2 + ny**2 + 1.0)
        return np.clip(intensity, 0.0, 1.0)

    shade = xr.apply_ufunc(one, surface, input_core_dims=[["y", "x"]], output_core_dims=[["y", "x"]], vectorize=True)
    return shade.where(surface.notnull()).rename("hillshade").transpose(*surface.dims)


def resolve_cmap(cmap: str | matplotlib.colors.Colormap) -> matplotlib.colors.Colormap:
    """
    Turn a colormap name into a colormap, leaving colormaps alone.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        A registered name (``"cmg.speed"``, ``"cmc.batlow"``, ``"viridis"``) or a colormap.
        The ``cmg.`` and ``cmc.`` colormaps are registered when this module is imported.

    Returns
    -------
    matplotlib.colors.Colormap
        The colormap.
    """
    return matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap


def blended_frames(
    data: xr.DataArray,
    shade: xr.DataArray,
    *,
    cmap: str | matplotlib.colors.Colormap,
    clim: tuple[float, float],
    log: bool = False,
) -> np.ndarray:
    """
    Color a field and multiply it into a hillshade, frame by frame.

    Cells without data show the hillshade alone (:func:`blend_multiply` with
    a transparent colour); cells without a hillshade, outside the surface,
    are transparent.

    Parameters
    ----------
    data : xarray.DataArray
        Field on ``(time, y, x)``.
    shade : xarray.DataArray
        Hillshade (:func:`hillshade`) on ``(time, y, x)`` or ``(y, x)``, on
        the grid of ``data``.
    cmap : str or matplotlib.colors.Colormap
        Colormap of the field.
    clim : tuple of float
        Color limits of the field.
    log : bool, default False
        Color the field on a logarithmic scale.

    Returns
    -------
    numpy.ndarray
        RGBA images in ``[0, 1]``, ``(time, y, x, 4)``.
    """
    colormap = resolve_cmap(cmap).with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
    norm = LogNorm(*clim) if log else Normalize(*clim)
    data = data.transpose("time", "y", "x")
    shade = shade.transpose(..., "y", "x").broadcast_like(data).transpose("time", "y", "x")
    values = data.values
    if log:
        values = np.where(values > 0, values, np.nan)
    # Flattened: LogNorm takes at most two dimensions.
    rgba = colormap(norm(np.ma.masked_invalid(values).ravel()).reshape(values.shape))
    intensity = shade.values[..., None]
    rgb = blend_multiply(rgba, np.nan_to_num(intensity, nan=1.0))
    return np.concatenate([rgb, np.isfinite(intensity).astype(float)], axis=-1)


def blended_animation(
    data: xr.DataArray,
    surface: xr.DataArray,
    *,
    cmap: str | matplotlib.colors.Colormap = "viridis",
    clim: tuple[float, float] | None = None,
    log: bool = False,
    label: str | None = None,
    frame_width: int = 500,
    embed: bool = True,
    **shading: Any,
) -> Any:
    """
    Animate a field over time, its colours multiplied into a hillshade.

    Each frame is the field coloured with ``cmap`` and blended into the
    hillshade of ``surface`` at that time (:func:`blended_frames`), titled
    with its date. The frames are played with a scrubber, a time slider
    with play buttons.

    Parameters
    ----------
    data : xarray.DataArray
        Field on ``(time, y, x)``, e.g. ``ds.velsurf_mag``.
    surface : xarray.DataArray
        Surface elevation to shade, on ``(time, y, x)`` (a hillshade per
        frame) or ``(y, x)`` (one for all), on the grid of ``data``.
    cmap : str or matplotlib.colors.Colormap, default "viridis"
        Colormap of the field: a registered name (``"cmg.speed"``,
        ``"cmc.davos_r"``) or a colormap.
    clim : tuple of float or None, optional
        Color limits; the 2nd to 98th percentile of the field when None.
    log : bool, default False
        Color the field on a logarithmic scale.
    label : str or None, optional
        Colorbar label; the field's name and units when None.
    frame_width : int, default 500
        Width of the map in pixels.
    embed : bool, default True
        Return every frame embedded in the output, which plays in a notebook
        without a live connection to the kernel and keeps playing in a saved
        notebook. False returns the live :mod:`panel` object instead.
    **shading : Any
        Passed to :func:`hillshade` (``azdeg``, ``altdeg``, ``vert_exag``).

    Returns
    -------
    Any
        The embedded animation to display, or the :mod:`panel` object when
        ``embed`` is False.

    Examples
    --------
    >>> blended_animation(ds.velsurf_mag, ds.usurf, cmap="cmg.speed", clim=(0, 10000))  # doctest: +SKIP
    >>> blended_animation(ds.plume_basal_melt_rate, ds.usurf, cmap="cmc.davos_r", clim=(0, 500))  # doctest: +SKIP
    """
    # Optional, notebook-only dependencies: importing them here keeps the module usable without them.
    import holoviews as hv  # pylint: disable=import-outside-toplevel
    import panel as pn  # pylint: disable=import-outside-toplevel

    hv.extension("bokeh")
    if clim is None:
        finite = data.values[np.isfinite(data.values)]
        if log:
            finite = finite[finite > 0]
        clim = (float(np.percentile(finite, 2)), float(np.percentile(finite, 98))) if finite.size else (0.0, 1.0)
    if label is None:
        units = data.attrs.get("units", "")
        label = f"{data.name} ({units})" if units else str(data.name)
    colormap = resolve_cmap(cmap).with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
    images = blended_frames(data, hillshade(surface, **shading), cmap=colormap, clim=clim, log=log)

    x, y = data["x"].values, data["y"].values
    # An RGB image has no colorbar; an invisible image of the field carries it.
    colorbar = hv.Image((x, y, data.isel(time=0).transpose("y", "x").values), kdims=["x", "y"]).opts(
        cmap=colormap,
        clim=clim,
        logz=log,
        alpha=0,
        colorbar=True,
        clabel=label,
        aspect="equal",
        frame_width=frame_width,
    )
    frames = {}
    for t, image in zip(data["time"].values, images):
        date = np.datetime_as_string(t, unit="D") if np.issubdtype(np.asarray(t).dtype, np.datetime64) else str(t)
        rgb = hv.RGB((x, y, *np.moveaxis(image, -1, 0)), kdims=["x", "y"])
        # The title goes on every frame, with the colorbar: on an overlay of the
        # animation and a static colorbar it would stay at the first date.
        frames[date] = (rgb * colorbar).opts(title=date)
    # Panel embeds every position of the scrubber but its last one, so the last
    # frame would never play: repeat it once at the end. Its key sorts after
    # the real one and is never embedded; reaching it leaves the last frame up.
    frames[f"{date} "] = frames[date]
    animation = hv.HoloMap(frames, kdims="date").opts(aspect="equal", frame_width=frame_width)
    pane = pn.panel(animation, widget_type="scrubber", widget_location="bottom")
    if not embed:
        return pane
    return pane.embed(max_states=len(frames), max_opts=len(frames))
