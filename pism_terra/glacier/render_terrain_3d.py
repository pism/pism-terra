#!/usr/bin/env python
"""
Render a 3D terrain from a PISM/netCDF spatial file and drape a field over it.

Builds a 3D surface from a terrain variable (default ``usurf``) and colors it by
a draped variable (default ``velsurf_mag``), one PNG per time step so the frames
can be turned into an animation.

Examples
--------
    python render_terrain_3d.py path/to/spatial.nc
    python render_terrain_3d.py spatial.nc --z-exaggeration 4 --cmap turbo --log
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

import matplotlib
import numpy as np
import pyvista as pv
import xarray as xr

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

    Parameters
    ----------
    name : str
        Registered colormap name.

    Returns
    -------
    matplotlib.colors.Colormap
        The resolved colormap.

    Raises
    ------
    SystemExit
        If ``name`` is not a registered colormap.
    """
    try:
        return matplotlib.colormaps[name]
    except KeyError as exc:
        raise SystemExit(f"Unknown colormap {name!r}. Registered: {', '.join(sorted(matplotlib.colormaps))}") from exc


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
    p.add_argument("--base-cmap", default="dem_ak", help="Colormap for the base terrain (default: dem_ak).")
    p.add_argument("--base-clim", type=float, nargs=2, default=None, help="Base color limits MIN MAX (default: auto).")
    # Overlay layer (velocity on ice).
    p.add_argument("--overlay-var", default="velsurf_mag", help="Variable overlaid on ice (default: velsurf_mag).")
    p.add_argument("--overlay-cmap", default="speed", help="Colormap for the overlay (default: speed).")
    p.add_argument(
        "--overlay-clim", type=float, nargs=2, default=None, help="Overlay limits MIN MAX (default: robust)."
    )
    p.add_argument("--overlay-log", action="store_true", help="Log-scale the overlay color mapping.")
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
    z = np.nan_to_num(z, nan=float(np.nanmin(z)) if np.isfinite(z).any() else 0.0) * cfg["z_exaggeration"]
    b = np.asarray(ds[cfg["base_var"]].isel(time=t).values, dtype=float)
    ov = np.asarray(ds[cfg["overlay_var"]].isel(time=t).values, dtype=float)
    thk = np.asarray(ds["thk"].isel(time=t).values, dtype=float)
    if cfg["overlay_log"]:
        ov = np.where(ov > 0, ov, np.nan)

    grid = pv.StructuredGrid(xx, yy, z)
    grid[cfg["base_var"]] = b.ravel(order="F")
    grid[cfg["overlay_var"]] = ov.ravel(order="F")
    grid["thk"] = thk.ravel(order="F")

    # Ice overlay: keep only cells where thk > threshold, lift it slightly.
    ice = grid.threshold(cfg["overlay_thk"], scalars="thk")
    if ice.n_points:
        ice.points[:, 2] += cfg["z_offset"]

    pl.clear()
    pl.add_mesh(
        grid,
        scalars=cfg["base_var"],
        cmap=w["base_cmap"],
        clim=cfg["base_clim"],
        lighting=True,
        smooth_shading=True,
        show_scalar_bar=True,
        scalar_bar_args=bar_args(_var_title(ds, cfg["base_var"]), position_x=0.05),
    )
    if ice.n_points:
        pl.add_mesh(
            ice,
            scalars=cfg["overlay_var"],
            cmap=w["overlay_cmap"],
            clim=cfg["overlay_clim"],
            log_scale=cfg["overlay_log"],
            lighting=True,
            smooth_shading=True,
            show_scalar_bar=True,
            scalar_bar_args=bar_args(_var_title(ds, cfg["overlay_var"]), position_x=0.55),
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
        Parsed arguments providing ``azimuth``, ``elevation``, and ``zoom``.
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
