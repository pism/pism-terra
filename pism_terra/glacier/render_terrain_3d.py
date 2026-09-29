#!/usr/bin/env python
"""
Render a 3D terrain from a PISM/netCDF spatial file and drape a field over it.

Builds a 3D surface from a terrain variable (default ``usurf``) and colors it by
a draped variable (default ``velsurf_mag``), one PNG per time step so the frames
can be turned into an animation.

By default the relief is shaded like a GIS hillshade: a hillshade of the terrain
is computed from the grid and multiplied into the colors of both layers, so the
shading does not depend on the camera. ``--shading lit`` uses PyVista's lights
instead.

Examples
--------
    python render_terrain_3d.py path/to/spatial.nc
    python render_terrain_3d.py spatial.nc --z-exaggeration 4 --cmap turbo --log
    python render_terrain_3d.py spatial.nc --overlay-var debris_thickness --overlay-cmap batlow
    # then, e.g.:
    ffmpeg -framerate 15 -i frames/frame_%04d.png -pix_fmt yuv420p out.mp4
"""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Imported for its side effect: registering Crameri's "cmc.*" colormaps.
import cmcrameri.cm  # noqa: F401  pylint: disable=unused-import
import matplotlib
import numpy as np
import pyvista as pv
import xarray as xr
from matplotlib.colors import LightSource, LogNorm, Normalize

from pism_terra.colormaps import register_colormaps

# Register the project's QGIS colormaps (e.g. "speed") into matplotlib's registry.
register_colormaps()


def detect_screen_size(default: tuple[int, int] = (1600, 1200)) -> tuple[int, int]:
    """
    Return the primary display's native pixel resolution.

    On macOS, parses ``system_profiler SPDisplaysDataType``; falls back to
    ``default`` on any other platform or if detection fails.

    Parameters
    ----------
    default : tuple of int, default ``(1600, 1200)``
        Fallback ``(width, height)`` when the resolution can't be detected.

    Returns
    -------
    tuple of int
        ``(width, height)`` in pixels.
    """
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            ).stdout
            m = re.search(r"Resolution:\s*(\d+)\s*x\s*(\d+)", out)
            if m:
                return (int(m.group(1)), int(m.group(2)))
        except (OSError, subprocess.SubprocessError):
            pass
    return default


def resolve_cmap(name: str) -> matplotlib.colors.Colormap:
    """
    Resolve a colormap name to a matplotlib ``Colormap`` object.

    Passing the object (rather than the name) to PyVista bypasses its built-in
    cmocean/colorcet name handling, so project colormaps registered via
    :func:`register_colormaps` (e.g. ``"speed"``) are used instead of a
    same-named cmocean map.

    Crameri's scientific colormaps are registered with a ``cmc.`` prefix
    (``cmc.batlow``) and can also be given without it (``batlow``). A bare name
    that matplotlib already has (``berlin``, ``managua``, ``vanimo``) resolves
    to matplotlib's version; use the prefix for Crameri's.

    Parameters
    ----------
    name : str
        Registered colormap name, or a Crameri colormap name without ``cmc.``.

    Returns
    -------
    matplotlib.colors.Colormap
        The resolved colormap.

    Raises
    ------
    SystemExit
        If ``name`` is not a registered colormap.
    """
    for candidate in (name, f"cmc.{name}"):
        if candidate in matplotlib.colormaps:
            return matplotlib.colormaps[candidate]
    raise SystemExit(f"Unknown colormap {name!r}. Registered: {', '.join(sorted(matplotlib.colormaps))}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("infile", help="Input netCDF file.")
    p.add_argument("-o", "--outdir", default=None, help="Output directory for PNGs (default: <infile>_frames).")
    p.add_argument("--surface-var", default="usurf", help="Variable used for terrain height (default: usurf).")
    p.add_argument("--z-exaggeration", type=float, default=3.0, help="Vertical exaggeration factor (default: 3).")
    # Base terrain layer (elevation).
    p.add_argument("--base-var", default="usurf", help="Variable colored on the base terrain (default: usurf).")
    p.add_argument(
        "--base-cmap",
        default="dem_ak",
        help="Colormap for the base terrain; Crameri maps as batlow or cmc.batlow (default: dem_ak).",
    )
    p.add_argument("--base-clim", type=float, nargs=2, default=None, help="Base color limits MIN MAX (default: auto).")
    p.add_argument(
        "--base-tint",
        type=float,
        default=0.4,
        help="Strength of the base colormap over the grey hillshade, 0 (grey) to 1 (full color); default 0.4.",
    )
    # Shading.
    p.add_argument(
        "--shading",
        choices=["multidirectional", "hillshade", "lit"],
        default="multidirectional",
        help=(
            "multidirectional: hillshade lit from four directions around --light-azimuth (default); "
            "hillshade: lit from --light-azimuth only; lit: PyVista's lights, no hillshade."
        ),
    )
    p.add_argument(
        "--light-azimuth", type=float, default=315.0, help="Hillshade light azimuth, degrees from north (default 315)."
    )
    p.add_argument("--light-altitude", type=float, default=45.0, help="Hillshade light altitude, degrees (default 45).")
    p.add_argument(
        "--hillshade-exaggeration",
        type=float,
        default=2.0,
        help="Vertical exaggeration of the hillshade, independent of --z-exaggeration (default 2).",
    )
    # Overlay layer (velocity on ice).
    p.add_argument("--overlay-var", default="velsurf_mag", help="Variable overlaid on ice (default: velsurf_mag).")
    p.add_argument(
        "--overlay-cmap",
        default="speed",
        help="Colormap for the overlay; Crameri maps as batlow or cmc.batlow (default: speed).",
    )
    p.add_argument(
        "--overlay-clim", type=float, nargs=2, default=None, help="Overlay limits MIN MAX (default: robust)."
    )
    p.add_argument("--overlay-log", action="store_true", help="Log-scale the overlay color mapping.")
    p.add_argument(
        "--overlay-opacity", type=float, default=0.85, help="Opacity of the overlay on ice, 0 to 1 (default 0.85)."
    )
    p.add_argument(
        "--overlay-thk",
        type=float,
        default=10.0,
        help="Overlay the velocity only where thk > this (m); default 10.",
    )
    p.add_argument("--time-stride", type=int, default=1, help="Render every Nth time step (default: 1).")
    p.add_argument(
        "--font-size",
        type=int,
        default=None,
        help="Time-label font size in points (default: auto-scaled to the image height).",
    )
    p.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        default=None,
        help="PNG size W H (default: detected screen resolution).",
    )
    p.add_argument("--azimuth", type=float, default=45.0, help="Camera azimuth from isometric (default: 0).")
    p.add_argument("--elevation", type=float, default=15.0, help="Camera elevation from isometric (default: 15).")
    p.add_argument(
        "--zoom", type=float, default=1.6, help="Camera zoom factor; >1 tightens onto the terrain (default: 1.6)."
    )
    p.add_argument(
        "--pan",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("X", "Y"),
        help=(
            "Shift the picture right by X and up by Y, as fractions of the frame height "
            "(e.g. --pan 0 0.1 moves it up by a tenth of the frame); default 0 0."
        ),
    )
    p.add_argument(
        "--aa",
        choices=["ssaa", "msaa", "fxaa", "none"],
        default="ssaa",
        help="Anti-aliasing: ssaa (best/slowest) ... fxaa/none (fastest). Big speed lever.",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help=(
            "Parallel rendering processes (default: 1). Speeds up headless software "
            "rendering (Linux/OSMesa) but is slower on macOS where the GPU serializes."
        ),
    )
    return p.parse_args()


def robust_clim(values: np.ndarray, log: bool) -> tuple[float, float]:
    """
    Return robust color limits from the 2nd/98th percentiles of finite data.

    Parameters
    ----------
    values : numpy.ndarray
        Data array; non-finite values are ignored.
    log : bool
        If True, restrict to positive values and use the 2nd percentile as the
        lower bound (for a log color scale); otherwise the lower bound is 0.

    Returns
    -------
    tuple of float
        ``(min, max)`` color limits.
    """
    finite = values[np.isfinite(values)]
    if log:
        finite = finite[finite > 0]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(finite, 2)) if log else 0.0
    hi = float(np.percentile(finite, 98))
    if not log:
        lo = min(lo, hi)
    if hi <= lo:
        hi = lo + 1.0
    return (max(lo, 1e-6) if log else lo, hi)


def time_label(ds: xr.Dataset, t: int) -> str:
    """
    Format a human-readable label for a time step.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset providing the ``time`` coordinate.
    t : int
        Time-step index.

    Returns
    -------
    str
        Formatted date/time (or ``"step <t>"`` if no time coordinate).
    """
    if "time" not in ds:
        return f"step {t}"
    val = ds["time"].values[t]
    try:  # datetime64 / cftime
        return str(np.datetime_as_string(val, unit="D"))
    except (TypeError, ValueError):
        return str(getattr(val, "isoformat", lambda: val)()) if hasattr(val, "isoformat") else f"{float(val):.1f}"


def full_clim(values: np.ndarray) -> tuple[float, float]:
    """
    Return the 1st/99th-percentile range of finite data (for elevation).

    Parameters
    ----------
    values : numpy.ndarray
        Data array; non-finite values are ignored.

    Returns
    -------
    tuple of float
        ``(min, max)`` color limits.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
    return (lo, hi if hi > lo else lo + 1.0)


def bar_args(title: str, position_x: float) -> dict:
    """
    Scalar-bar layout for a horizontal bar in the lower half of the frame.

    Parameters
    ----------
    title : str
        Scalar-bar title.
    position_x : float
        Left edge of the bar in normalized viewport coordinates (0-1).

    Returns
    -------
    dict
        Keyword arguments for ``Plotter.add_mesh(scalar_bar_args=...)``.
    """
    return {
        "title": title,
        "color": "black",
        "n_labels": 5,
        "vertical": False,
        "position_x": position_x,
        "position_y": 0.04,
        "width": 0.4,
        "height": 0.05,
    }


def hillshade(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    multidirectional: bool = True,
    vert_exag: float = 1.0,
) -> np.ndarray:
    """
    Compute the hillshade of a gridded surface.

    Parameters
    ----------
    z : numpy.ndarray
        Surface heights, shape ``(ny, nx)``; non-finite values are set to the
        lowest finite height.
    x, y : numpy.ndarray
        Cell-center coordinates in m, lengths ``nx`` and ``ny``, in either order.
    azimuth : float, default 315
        Direction the light comes from, degrees clockwise from north.
    altitude : float, default 45
        Angle of the light above the horizon, degrees.
    multidirectional : bool, default True
        Average the hillshades lit from ``azimuth - 90``, ``- 45``, ``+ 0`` and
        ``+ 45`` degrees, as GDAL's ``-multidirectional`` does, so slopes facing
        away from the main light keep their detail.
    vert_exag : float, default 1
        Vertical exaggeration applied to the heights before shading.

    Returns
    -------
    numpy.ndarray
        Illumination in ``[0, 1]``, same shape and row order as ``z``.
    """
    z = np.asarray(z, dtype=float)
    z = np.where(np.isfinite(z), z, np.nanmin(z) if np.isfinite(z).any() else 0.0)
    dx = abs(float(x[1] - x[0])) if len(x) > 1 else 1.0
    dy = abs(float(y[1] - y[0])) if len(y) > 1 else 1.0
    # LightSource expects the first row to be the northern edge.
    north_up = len(y) < 2 or y[0] > y[-1]
    zz = z if north_up else z[::-1]
    azimuths = [azimuth - 90, azimuth - 45, azimuth, azimuth + 45] if multidirectional else [azimuth]
    hs = np.mean(
        [
            LightSource(azdeg=az % 360, altdeg=altitude).hillshade(zz, vert_exag=vert_exag, dx=dx, dy=dy)
            for az in azimuths
        ],
        axis=0,
    )
    return hs if north_up else hs[::-1]


def shaded_colors(
    values: np.ndarray,
    cmap: matplotlib.colors.Colormap,
    norm: Normalize,
    shade: np.ndarray,
    tint: float = 1.0,
) -> np.ndarray:
    """
    Map values to colors, fade them toward white and multiply in a hillshade.

    Parameters
    ----------
    values : numpy.ndarray
        Data, shape ``(ny, nx)``; non-finite values take the colormap's "bad" color.
    cmap : matplotlib.colors.Colormap
        Colormap.
    norm : matplotlib.colors.Normalize
        Maps ``values`` to ``[0, 1]``.
    shade : numpy.ndarray
        Illumination in ``[0, 1]``, same shape as ``values``.
    tint : float, default 1
        Colormap strength: 1 keeps the colors, 0 makes every cell white, so only
        the hillshade remains.

    Returns
    -------
    numpy.ndarray
        ``uint8`` RGB, shape ``(ny * nx, 3)``, in the Fortran order PyVista's
        ``StructuredGrid`` uses for point data.
    """
    rgb = cmap(norm(np.ma.masked_invalid(values)))[..., :3]
    rgb = (1.0 - tint * (1.0 - rgb)) * shade[..., None]
    rgb = np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
    return np.stack([rgb[..., i].ravel(order="F") for i in range(3)], axis=1)


def faded_cmap(cmap: matplotlib.colors.Colormap, tint: float) -> matplotlib.colors.Colormap:
    """
    Return ``cmap`` faded toward white, for a scalar bar matching :func:`shaded_colors`.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        Colormap to fade.
    tint : float
        Colormap strength, as in :func:`shaded_colors`.

    Returns
    -------
    matplotlib.colors.Colormap
        The faded colormap (``cmap`` itself when ``tint`` is 1).
    """
    if tint >= 1.0:
        return cmap
    colors = cmap(np.linspace(0.0, 1.0, 256))
    colors[:, :3] = 1.0 - tint * (1.0 - colors[:, :3])
    return matplotlib.colors.ListedColormap(colors, name=f"{cmap.name}_faded")


def _add_scalar_bar(
    pl: pv.Plotter,
    cmap: matplotlib.colors.Colormap,
    clim: tuple[float, float],
    log: bool,
    bar_kwargs: dict,
) -> None:
    """
    Add a scalar bar for a mesh drawn with precomputed RGB colors.

    PyVista draws no scalar bar for RGB point data, so the bar hangs off an
    invisible one-point mesh that carries the colormap and limits.

    Parameters
    ----------
    pl : pyvista.Plotter
        Plotter to draw into.
    cmap : matplotlib.colors.Colormap
        Colormap shown on the bar.
    clim : tuple of float
        Color limits.
    log : bool
        Log-scale the bar.
    bar_kwargs : dict
        Scalar-bar layout from :func:`bar_args`.
    """
    anchor = pv.PolyData(np.zeros((1, 3)))
    anchor["value"] = np.array([clim[0]])
    pl.add_mesh(
        anchor,
        scalars="value",
        cmap=cmap,
        clim=clim,
        log_scale=log,
        opacity=0.0,
        show_scalar_bar=True,
        scalar_bar_args=bar_kwargs,
    )


def _var_title(ds: xr.Dataset, var: str) -> str:
    """
    Build a scalar-bar title ``var [units]`` for a dataset variable.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing ``var``.
    var : str
        Variable name.

    Returns
    -------
    str
        ``"<var> [<units>]"``, or just ``"<var>"`` when no units attribute.
    """
    u = ds[var].attrs.get("units", "")
    return f"{var} [{u}]" if u else var


# Per-process state, populated by ``_setup_worker`` (reused across frames).
_WORKER: dict = {}


def _setup_worker(infile: str, cfg: dict) -> None:
    """
    Open the dataset and build a reusable off-screen Plotter for this process.

    Run once per worker (the ``ProcessPoolExecutor`` initializer); subsequent
    :func:`_render_frame` calls reuse the open dataset and Plotter so the GL
    context and grid coordinates are created only once.

    Parameters
    ----------
    infile : str
        Path to the input netCDF file.
    cfg : dict
        Render configuration (variables, colormaps, clims, camera, window
        size, etc.) shared across all frames.
    """
    ds = xr.open_dataset(infile, decode_coords="all")
    xx, yy = np.meshgrid(ds["x"].values, ds["y"].values)
    pl = pv.Plotter(off_screen=True, window_size=list(cfg["window_size"]))
    pl.set_background("white")
    if cfg["aa"] != "none":
        try:
            pl.enable_anti_aliasing(cfg["aa"])
        except (ValueError, AttributeError):
            pass
    _WORKER.update(
        ds=ds,
        x=ds["x"].values,
        y=ds["y"].values,
        xx=xx,
        yy=yy,
        pl=pl,
        cfg=cfg,
        base_cmap=resolve_cmap(cfg["base_cmap"]),
        overlay_cmap=resolve_cmap(cfg["overlay_cmap"]),
    )


def _render_frame(t: int) -> str:
    """
    Render one time step to a PNG using this process's worker state.

    Parameters
    ----------
    t : int
        Time-step index to render.

    Returns
    -------
    str
        Path to the written PNG.
    """
    w = _WORKER
    ds, xx, yy, pl, cfg = w["ds"], w["xx"], w["yy"], w["pl"], w["cfg"]

    z = np.asarray(ds[cfg["surface_var"]].isel(time=t).values, dtype=float)
    z = np.nan_to_num(z, nan=float(np.nanmin(z)) if np.isfinite(z).any() else 0.0)
    shade = None
    if cfg["shading"] != "lit":
        shade = hillshade(
            z,
            w["x"],
            w["y"],
            azimuth=cfg["light_azimuth"],
            altitude=cfg["light_altitude"],
            multidirectional=cfg["shading"] == "multidirectional",
            vert_exag=cfg["hillshade_exaggeration"],
        )
    z = z * cfg["z_exaggeration"]
    b = np.asarray(ds[cfg["base_var"]].isel(time=t).values, dtype=float)
    ov = np.asarray(ds[cfg["overlay_var"]].isel(time=t).values, dtype=float)
    thk = np.asarray(ds["thk"].isel(time=t).values, dtype=float)
    if cfg["overlay_log"]:
        ov = np.where(ov > 0, ov, np.nan)

    grid = pv.StructuredGrid(xx, yy, z)
    grid[cfg["base_var"]] = b.ravel(order="F")
    grid[cfg["overlay_var"]] = ov.ravel(order="F")
    grid["thk"] = thk.ravel(order="F")
    if shade is not None:
        overlay_norm = LogNorm(*cfg["overlay_clim"]) if cfg["overlay_log"] else Normalize(*cfg["overlay_clim"])
        grid["base_rgb"] = shaded_colors(b, w["base_cmap"], Normalize(*cfg["base_clim"]), shade, cfg["base_tint"])
        grid["overlay_rgb"] = shaded_colors(ov, w["overlay_cmap"], overlay_norm, shade)

    # Ice overlay: keep only cells where thk > threshold, lift it slightly.
    ice = grid.threshold(cfg["overlay_thk"], scalars="thk")
    if ice.n_points:
        ice.points[:, 2] += cfg["z_offset"]

    base_bar = bar_args(_var_title(ds, cfg["base_var"]), position_x=0.05)
    overlay_bar = bar_args(_var_title(ds, cfg["overlay_var"]), position_x=0.55)
    pl.clear()
    if shade is not None:
        # Colors carry the hillshade already, so the meshes are drawn unlit.
        pl.add_mesh(grid, scalars="base_rgb", rgb=True, lighting=False)
        _add_scalar_bar(pl, faded_cmap(w["base_cmap"], cfg["base_tint"]), cfg["base_clim"], False, base_bar)
        if ice.n_points:
            pl.add_mesh(ice, scalars="overlay_rgb", rgb=True, lighting=False, opacity=cfg["overlay_opacity"])
        _add_scalar_bar(pl, w["overlay_cmap"], cfg["overlay_clim"], cfg["overlay_log"], overlay_bar)
    else:
        pl.add_mesh(
            grid,
            scalars=cfg["base_var"],
            cmap=w["base_cmap"],
            clim=cfg["base_clim"],
            lighting=True,
            smooth_shading=True,
            show_scalar_bar=True,
            scalar_bar_args=base_bar,
        )
        if ice.n_points:
            pl.add_mesh(
                ice,
                scalars=cfg["overlay_var"],
                cmap=w["overlay_cmap"],
                clim=cfg["overlay_clim"],
                log_scale=cfg["overlay_log"],
                opacity=cfg["overlay_opacity"],
                lighting=True,
                smooth_shading=True,
                show_scalar_bar=True,
                scalar_bar_args=overlay_bar,
            )
    pl.add_text(
        time_label(ds, t),
        position="upper_left",
        font_size=cfg["font_size"],
        color="black",
        shadow=True,
    )
    pl.camera_position = cfg["camera"]["position"]
    # camera_position carries position/focal/up but NOT the view angle that zoom
    # changes, so reapply it explicitly to preserve the zoom.
    pl.camera.view_angle = cfg["camera"]["view_angle"]

    out = Path(cfg["outdir"]) / f"frame_{t:04d}.png"
    pl.screenshot(str(out))
    return str(out)


def _compute_camera(
    xx: np.ndarray, yy: np.ndarray, z0: np.ndarray, args: argparse.Namespace, window_size: list
) -> dict:
    """
    Compute the fixed camera from the first frame as a picklable, zoom-preserving dict.

    Parameters
    ----------
    xx, yy : numpy.ndarray
        Meshgrid coordinate arrays for the surface (shape ``(ny, nx)``).
    z0 : numpy.ndarray
        First-frame (exaggerated) terrain heights, shape ``(ny, nx)``.
    args : argparse.Namespace
        Parsed arguments providing ``azimuth``, ``elevation``, ``zoom`` and ``pan``.
    window_size : list
        ``[width, height]`` in pixels for the off-screen render.

    Returns
    -------
    dict
        Camera state with keys ``"position"`` (position/focal/up tuples) and
        ``"view_angle"`` so the zoom survives broadcast to worker processes.
    """
    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl.add_mesh(pv.StructuredGrid(xx, yy, z0))
    pl.view_isometric()
    pl.camera.azimuth += args.azimuth
    pl.camera.elevation += args.elevation
    pl.camera.zoom(args.zoom)  # shrinks the view angle -> zooms in
    # Pan by moving camera and focal point together, opposite to the shift of the
    # picture, in units of the frame height at the focal point.
    position, focal, up = (np.asarray(v, dtype=float) for v in pl.camera_position)
    view = focal - position
    height = 2.0 * np.linalg.norm(view) * np.tan(np.radians(pl.camera.view_angle) / 2.0)
    up = up / np.linalg.norm(up)
    right = np.cross(view, up)
    right /= np.linalg.norm(right)
    shift = -height * (args.pan[0] * right + args.pan[1] * up)
    pl.camera_position = [tuple(position + shift), tuple(focal + shift), tuple(up)]
    cam = {
        "position": [tuple(p) for p in pl.camera_position],
        "view_angle": float(pl.camera.view_angle),
    }
    pl.close()
    return cam


def main() -> None:
    """Render one PNG per time step, in parallel across processes."""
    args = parse_args()
    infile = Path(args.infile).expanduser()
    outdir = Path(args.outdir) if args.outdir else infile.with_name(infile.stem + "_frames")
    outdir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(infile, decode_coords="all")
    for var in (args.surface_var, args.base_var, args.overlay_var):
        if var not in ds:
            raise SystemExit(f"Variable {var!r} not in {infile} (have: {sorted(ds.data_vars)}).")
    if "thk" not in ds:
        raise SystemExit(f"'thk' not in {infile}; needed to overlay {args.overlay_var} on ice.")

    window_size = list(args.window_size) if args.window_size else list(detect_screen_size())
    # Scale the time-label font to the image height (~2.5%) so it stays legible
    # at high resolution and survives H.264/yuv420p compression.
    font_size = args.font_size if args.font_size else max(14, round(window_size[1] * 0.025))

    xx, yy = np.meshgrid(ds["x"].values, ds["y"].values)  # scalars must ravel(order="F")
    steps = list(range(0, int(ds.sizes.get("time", 1)), args.time_stride))

    # Fixed color limits (defaults match the elevation/velocity scales).
    base_clim = tuple(args.base_clim) if args.base_clim else (-2000.0, 3500.0)
    overlay_clim = tuple(args.overlay_clim) if args.overlay_clim else (1.0, 100.0)

    # Small upward offset so the ice overlay wins the depth test over the base.
    surf0 = np.asarray(ds[args.surface_var].isel(time=steps[0]).values, dtype=float) * args.z_exaggeration
    z_all = np.asarray(ds[args.surface_var].values, dtype=float) * args.z_exaggeration
    z_offset = 0.002 * float(np.nanmax(z_all) - np.nanmin(z_all))
    camera = _compute_camera(xx, yy, np.nan_to_num(surf0, nan=float(np.nanmin(surf0))), args, window_size)
    ds.close()  # each worker opens its own handle

    cfg = {
        "surface_var": args.surface_var,
        "base_var": args.base_var,
        "overlay_var": args.overlay_var,
        "base_cmap": args.base_cmap,
        "overlay_cmap": args.overlay_cmap,
        "base_clim": base_clim,
        "overlay_clim": overlay_clim,
        "overlay_log": args.overlay_log,
        "overlay_thk": args.overlay_thk,
        "overlay_opacity": args.overlay_opacity,
        "base_tint": args.base_tint,
        "shading": args.shading,
        "light_azimuth": args.light_azimuth,
        "light_altitude": args.light_altitude,
        "hillshade_exaggeration": args.hillshade_exaggeration,
        "z_exaggeration": args.z_exaggeration,
        "z_offset": z_offset,
        "window_size": window_size,
        "camera": camera,
        "outdir": str(outdir),
        "aa": args.aa,
        "font_size": font_size,
    }

    total = len(steps)
    jobs = max(1, min(args.jobs, total))
    print(f"Rendering {total} frame(s) at {window_size[0]}x{window_size[1]} with {jobs} process(es)...")
    if jobs == 1:
        _setup_worker(str(infile), cfg)
        for i, t in enumerate(steps, 1):
            print(f"[{i}/{total}] {_render_frame(t)}")
        _WORKER["pl"].close()
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_setup_worker, initargs=(str(infile), cfg)) as ex:
            for i, out in enumerate(ex.map(_render_frame, steps), 1):
                print(f"[{i}/{total}] {out}")

    print(f"\nWrote {total} frame(s) to {outdir}")
    print("Make an animation, e.g.:")
    print(f"  ffmpeg -framerate 15 -i {outdir}/frame_%04d.png -pix_fmt yuv420p {outdir.name}.mp4")


if __name__ == "__main__":
    main()
