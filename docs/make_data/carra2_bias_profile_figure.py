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
Illustrate how pism-carra2-bias-correction defines the bias, on and off glacier.

For a CARRA2 cell on glacier and one off glacier at about the same terrain
elevation, side by side, on a typical summer and a typical winter day, plot the
air temperature on the pressure levels against their geopotential height,
the straight line ``T = a + b z`` that :func:`fit_to_ground` fits through
them, carried down to the terrain, and the 2 m air temperature there. The gap
between the line at the ground and the 2 m temperature is the bias. Levels
below CARRA2's terrain are drawn hollow: they are extrapolated, not air, and
only ``--below`` of them enter the fit. A bottom panel shows the bias of each
cell through the record, with its two days marked.

Examples
--------
Make the figure on the S4F CARRA2 bias page::

    python docs/make_data/carra2_bias_profile_figure.py \
        --outlines glacier_s4f_input/s4f_g.gpkg --elevation 1000 \
        -o docs/source/_static/s4f/carra2_bias_profile.png \
        ~/storstrommen/terra/s4f_carra2_3d/carra2_daily_S4F_target_01.zarr
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes

from pism_terra.glacier.carra2_bias_correction import (
    DEFAULT_BELOW,
    _parse_below,
    fit_to_ground,
    glacier_mask,
    open_carra2,
    read_vectors,
)

K0 = 273.15
SEASONS = {"summer": ((6, 7, 8), "tab:red"), "winter": ((12, 1, 2), "tab:blue")}


def pick_cell(ds: xr.Dataset, elevation: float, candidates: np.ndarray) -> tuple[int, int]:
    """
    Find the candidate cell whose terrain is closest to an elevation.

    Parameters
    ----------
    ds : xarray.Dataset
        CARRA2 store with ``orography`` on ``(y, x)``.
    elevation : float
        Terrain elevation to look for, m.
    candidates : numpy.ndarray
        Boolean on ``(y, x)``; only True cells are considered.

    Returns
    -------
    tuple of int
        ``(iy, ix)`` of the cell.
    """
    oro = ds["orography"].values
    distance = np.where(candidates & np.isfinite(oro), np.abs(oro - elevation), np.inf)
    index = np.unravel_index(np.argmin(distance), oro.shape)
    return int(index[0]), int(index[1])


def typical_day(bias: pd.Series, months: tuple[int, ...]) -> pd.Timestamp:
    """
    Find the day of a season whose bias is that season's median.

    Parameters
    ----------
    bias : pandas.Series
        Daily bias, indexed by time.
    months : tuple of int
        Months of the season, e.g. ``(6, 7, 8)``.

    Returns
    -------
    pandas.Timestamp
        The day.
    """
    season = bias[bias.index.month.isin(months)].dropna()
    return season.index[np.argmin(np.abs(season.to_numpy() - season.median()))]


def draw_profile(ax, prof: dict, color: str, bottom: float, labels: bool) -> None:
    """
    Draw levels, fit, terrain, fit at the ground, 2 m temperature and the bias arrow.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    prof : dict
        One day's profile: level heights ``z`` (m), temperatures ``t`` (°C)
        and pressures ``p`` (hPa); boolean ``used`` (in the fit) and
        ``above`` (above the terrain) per level; ``ground`` (m), ``t2m``,
        ``t_ground`` (°C) and the fitted ``lapse`` rate (K km-1).
    color : str
        Colour of the levels and the fit.
    bottom : float
        Lower edge of the shaded ground, m.
    labels : bool
        Label the terrain (main panel) rather than the bias (inset).
    """
    z, t, p, used, above = prof["z"], prof["t"], prof["p"], prof["used"], prof["above"]
    ground, t2m, t_ground, lapse = prof["ground"], prof["t2m"], prof["t_ground"], prof["lapse"]
    zz = np.linspace(ground, z.max(), 50)
    ax.plot(t_ground + lapse / 1000.0 * (zz - ground), zz, color=color, lw=1.5, label=f"fit, {lapse:.1f} K km$^{{-1}}$")
    # Levels: filled above the terrain, hollow below; grey if not in the fit.
    for zi, ti, pi, ui, ai in zip(z, t, p, used, above):
        edge = color if ui else "0.6"
        ax.plot(ti, zi, "o", ms=7, mec=edge, mfc=edge if ai else "white", mew=1.5, zorder=3)
        ax.annotate(
            f"{pi:.0f} hPa",
            (ti, zi),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color="0.35",
            annotation_clip=True,
            clip_on=True,
        )
    ax.axhspan(bottom, ground, color="0.85", zorder=0)
    ax.axhline(ground, color="0.3", lw=1)
    ax.plot(t_ground, ground, "D", color=color, ms=7, zorder=4, label=f"fit at ground {t_ground:.1f} °C")
    ax.plot(t2m, ground, "*", color="k", ms=13, zorder=4, label=f"2 m air {t2m:.1f} °C")
    ax.annotate(
        "",
        xy=(t2m, ground),
        xytext=(t_ground, ground),
        arrowprops={"arrowstyle": "<->", "color": "k", "lw": 1.2, "shrinkA": 6, "shrinkB": 6},
        zorder=5,
    )
    if labels:
        ax.text(0.02, ground, f" terrain {ground:.0f} m", transform=ax.get_yaxis_transform(), va="top", fontsize=8)
    else:
        ax.text(
            (t2m + t_ground) / 2,
            ground,
            f"bias {t_ground - t2m:+.1f} K",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            transform=ax.transData,
            bbox={"fc": "white", "ec": "none", "pad": 1},
            zorder=6,
        )
        ax.texts[-1].set_position(((t2m + t_ground) / 2, ground - 60))


def profile_panel(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    ax, cell: xr.Dataset, fit: xr.Dataset, day: pd.Timestamp, below: int | None, color: str, title: str
) -> None:
    """
    Draw the levels, the fit carried to the ground and the 2 m temperature for one day.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into; a zoom on the ground is added as an inset.
    cell : xarray.Dataset
        One cell's levels, 2 m temperature and orography, levels bottom up.
    fit : xarray.Dataset
        :func:`fit_to_ground` of *cell*.
    day : pandas.Timestamp
        Day to draw.
    below : int or None
        Levels below the terrain in the fit, as in :func:`fit_to_ground`.
    color : str
        Colour of the levels and the fit.
    title : str
        Panel title; the date is appended.
    """
    d = cell.sel(time=day)
    f = fit.sel(time=day)
    z = d["geopotential_height"].values
    t = d["air_temp"].values - K0
    p = d["pressure_level"].values
    ground = float(d["orography"])
    t2m = float(d["air_temp_2m"]) - K0
    t_ground = float(f["t_ground"]) - K0
    lapse = float(f["lapse_rate"])

    n_below = int((z < ground).sum())
    used = np.arange(z.size) >= (0 if below is None else n_below - below)
    above = z >= ground

    prof = {
        "z": z,
        "t": t,
        "p": p,
        "used": used,
        "above": above,
        "ground": ground,
        "t2m": t2m,
        "t_ground": t_ground,
        "lapse": lapse,
    }
    bottom = min(z.min(), ground) - 200
    ax.set_ylim(bottom, z.max() + 300)
    draw_profile(ax, prof, color, bottom, labels=True)
    ax.set_xlim(right=t.max() + 0.12 * (t.max() - t.min()))

    # Zoom on the ground, where the bias is.
    inset = ax.inset_axes([0.62, 0.47, 0.35, 0.31])
    lo, hi = min(t2m, t_ground), max(t2m, t_ground)
    pad = max(2.0, 0.6 * (hi - lo))
    inset.set_xlim(lo - pad, hi + pad)
    inset.set_ylim(ground - 400, ground + 600)
    draw_profile(inset, prof, color, ground - 400, labels=False)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.3)
    ax.indicate_inset_zoom(inset, edgecolor="0.4")

    ax.set_xlabel("temperature (°C)")
    ax.set_ylabel("height (m)")
    ax.set_title(f"{title}: {day:%Y-%m-%d}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)


def load_cell(ds: xr.Dataset, iy: int, ix: int, below: int | None) -> tuple[xr.Dataset, xr.Dataset, pd.Series]:
    """
    Load one cell, fit the levels to its ground every day and compute its bias.

    Parameters
    ----------
    ds : xarray.Dataset
        CARRA2 store.
    iy, ix : int
        Grid indices of the cell.
    below : int or None
        Levels below the terrain in the fit, as in :func:`fit_to_ground`.

    Returns
    -------
    cell : xarray.Dataset
        The cell's fields, levels bottom up.
    fit : xarray.Dataset
        :func:`fit_to_ground` of the cell.
    bias : pandas.Series
        Daily fit at the ground minus 2 m air temperature, K.
    """
    variables = ["air_temp", "geopotential_height", "air_temp_2m", "orography"]
    # fit_to_ground needs the levels bottom up.
    cell = ds[variables].isel(y=iy, x=ix).load().sortby("pressure_level", ascending=False)
    fit = fit_to_ground(cell["air_temp"], cell["geopotential_height"], cell["orography"], below=below)
    return cell, fit, (fit["t_ground"] - cell["air_temp_2m"]).to_series()


def cell_title(ds: xr.Dataset, cell: xr.Dataset, iy: int, ix: int, where: str) -> str:
    """
    Describe a cell: where it is, its terrain and whether it is on glacier.

    Parameters
    ----------
    ds : xarray.Dataset
        CARRA2 store with ``latitude`` and ``longitude``.
    cell : xarray.Dataset
        The cell's fields, with ``orography``.
    iy, ix : int
        Grid indices of the cell.
    where : str
        ``"on glacier"``, ``"off glacier"`` or empty.

    Returns
    -------
    str
        The description.
    """
    lat, lon = float(ds["latitude"][iy, ix]), float(ds["longitude"][iy, ix])
    lon = (lon + 180.0) % 360.0 - 180.0
    return (
        f"{where}{': ' if where else ''}{lat:.2f}° N, {abs(lon):.2f}° {'W' if lon < 0 else 'E'}, "
        f"terrain {float(cell['orography']):.0f} m"
    )


def main() -> None:
    """Make the figure."""
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("store", help="Daily CARRA2 store (zarr or NetCDF).")
    parser.add_argument("--outlines", nargs="*", default=None, help="Glacier outlines; split cells on/off glacier.")
    parser.add_argument(
        "--cells", nargs="+", choices=("on", "off"), default=["on", "off"], help="With --outlines, the columns to show."
    )
    parser.add_argument("--elevation", type=float, default=1000.0, help="Terrain elevation of the cells to show, m.")
    parser.add_argument("--x", type=float, default=None, help="Projected x of a single cell (overrides the rest).")
    parser.add_argument("--y", type=float, default=None, help="Projected y of a single cell (overrides the rest).")
    parser.add_argument("--below", type=_parse_below, default=DEFAULT_BELOW, help="As in pism-carra2-bias-correction.")
    parser.add_argument("--summer-day", default=None, help="Date for the summer panels (default: median JJA bias).")
    parser.add_argument("--winter-day", default=None, help="Date for the winter panels (default: median DJF bias).")
    parser.add_argument("-o", "--output", default="carra2_bias_profile.png", help="Output figure.")
    args = parser.parse_args()

    ds = open_carra2(args.store)
    # (label, iy, ix) per column.
    if args.x is not None and args.y is not None:
        columns = [
            ("", int(np.argmin(np.abs(ds["y"].values - args.y))), int(np.argmin(np.abs(ds["x"].values - args.x))))
        ]
    elif args.outlines:
        mask = glacier_mask(ds, read_vectors(args.outlines)).values
        candidates = {"on": ("on glacier", mask), "off": ("off glacier", ~mask)}
        columns = [(candidates[c][0], *pick_cell(ds, args.elevation, candidates[c][1])) for c in args.cells]
    else:
        columns = [("", *pick_cell(ds, args.elevation, np.ones(ds["orography"].shape, dtype=bool)))]

    fig = plt.figure(figsize=(6.5 * len(columns), 15), layout="constrained")
    grid = fig.add_gridspec(3, len(columns), height_ratios=[2, 2, 1])
    bias_axes: list[Axes] = []
    for col, (where, iy, ix) in enumerate(columns):
        cell, fit, bias = load_cell(ds, iy, ix, args.below)
        days = {
            "summer": pd.Timestamp(args.summer_day) if args.summer_day else typical_day(bias, SEASONS["summer"][0]),
            "winter": pd.Timestamp(args.winter_day) if args.winter_day else typical_day(bias, SEASONS["winter"][0]),
        }
        title = cell_title(ds, cell, iy, ix, where)
        for row, (season, (_, color)) in enumerate(SEASONS.items()):
            ax = fig.add_subplot(grid[row, col])
            profile_panel(ax, cell, fit, days[season], args.below, color, f"typical {season} day")
            if row == 0:
                ax.set_title(f"{title}\n{ax.get_title()}")

        ax = fig.add_subplot(grid[2, col], sharey=bias_axes[0] if bias_axes else None)
        bias_axes.append(ax)
        ax.plot(bias.index, bias.values, color="0.5", lw=0.5, label="daily bias")
        ax.plot(bias.index, bias.rolling(31, center=True).mean(), color="k", lw=1.2, label="31-day mean")
        for season, (_, color) in SEASONS.items():
            ax.axvline(days[season], color=color, ls="--", lw=1)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_ylabel("bias (K)")
        ax.set_title(f"{where or 'cell'}: level fit at the ground − 2 m air temperature", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        print(f"{where or 'cell'}: iy={iy}, ix={ix}; days {days['summer']:%Y-%m-%d}, {days['winter']:%Y-%m-%d}")

    below = "all" if args.below is None else args.below
    fig.suptitle(
        f"{Path(args.store.rstrip('/')).stem}: CARRA2 pressure-level temperature carried to the ground vs. 2 m air "
        f"temperature\n(below={below}; hollow = level below terrain, grey = not in fit)",
        fontsize=11,
    )
    fig.savefig(args.output, dpi=200)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
