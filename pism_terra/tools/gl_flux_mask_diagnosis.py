"""
Find out why two runs that differ only in their front-retreat mask move different ice across the grounding line.

PISM books ``ice_mass_transport_across_grounding_line`` in the ocean cells
(floating or ice-free) that grounded neighbours flow into, and its prescribed
front retreat treats the mask ``f`` of every cell and step as

* ``f <= 0``: the ice is removed;
* ``0 < f < 1``: the thickness is set to zero and ``f`` times it kept as a
  partially filled cell, so the cell is ice-free ocean (or land) as far as the
  grounding line is concerned;
* ``f >= 1``: nothing happens.

A 0/1 mask published on another grid (the CalFin mask at 450 m, offset from
PISM's cell centres) arrives interpolated, so every margin cell gets a
fraction. The two runs then have different fronts, different grounding lines
and different ice at them. This tool compares the two runs' spatial output
and retreat masks on the model grid:

1. the time-weighted grounding-line flux, discharge and mass budget, in total
   and per ISMIP region (Mouginot basins);
2. per record, the grounded cells that touch the ocean -- how many, how thick,
   how fast -- and the flux ``rho H u dx`` they would carry, so a difference
   splits into grounding-line length, thickness and speed;
3. where the flux ends up: floating cells or ice-free ocean cells;
4. the two retreat masks, interpolated to the run grid as PISM does, classed
   into none / partial / full, and how much of the flux difference lies next
   to cells whose class differs;
5. the 9 km tiles with the largest difference, with coordinates and zoom maps.

Usage on chinook (the spatial files stay where they are)::

    python -m pism_terra.tools.gl_flux_mask_diagnosis \\
        2026_10_ismip7_mask_450m/output/spatial/spatial_g900m_id_OCX_1980-01-01_1981-01-01.nc \\
        2026_10_ismip7_mask_900m/output/spatial/spatial_g900m_id_OCX_1980-01-01_1981-01-01.nc \\
        --labels 450m,900m --output-dir gl_mask_diagnosis

The retreat files are read from each file's ``command`` attribute
(``-geometry.front_retreat.prescribed.file``) unless ``--retreat-files`` is
given. The spatial output needs ``ice_mass_transport_across_grounding_line``,
``thk``, ``mask`` and ``velsurf_mag``; ``mass_fluxes`` adds the budget.
Everything written to the output directory is small: ``summary.txt``,
``regions.csv``, ``tiles.csv``, ``timeseries.csv``, ``fields.nc`` and figures.
"""

# pylint: disable=too-many-positional-arguments

from __future__ import annotations

import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import cf_xarray.units  # pylint: disable=unused-import  # noqa: F401  (teaches pint UDUNITS)
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint_xarray  # pylint: disable=unused-import  # noqa: F401  (sets up the unit registry)
import xarray as xr
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage

GL_VAR = "ice_mass_transport_across_grounding_line"
BUDGET_VARS = (
    "tendency_of_ice_mass",
    "tendency_of_ice_mass_due_to_flow",
    "tendency_of_ice_mass_due_to_surface_mass_flux",
    "tendency_of_ice_mass_due_to_basal_mass_flux",
    "tendency_of_ice_mass_due_to_discharge",
    "tendency_of_ice_mass_due_to_conservation_error",
)
RETREAT_VAR = "land_ice_area_fraction_retreat"
RETREAT_RE = re.compile(r"-geometry\.front_retreat\.prescribed\.file\s+(\S+)")
#: PISM's cell types.
GROUNDED, FLOATING, ICE_FREE_OCEAN = 2, 3, 4
#: PISM's defaults: constants.ice.density and the udunits year.
RHO_ICE = 910.0
SECONDS_PER_YEAR = 3.15569259747e7
#: Mask classes after PISM's rounding to 1/1000.
CLASSES = {0: "none", 1: "partial", 2: "full"}


@dataclass
class Run:
    """
    One run's spatial output and what is derived from it.

    Attributes
    ----------
    label : str
        Short name used in tables and figures.
    path : pathlib.Path
        The spatial file.
    ds : xarray.Dataset
        The open spatial file.
    retreat_file : str or None
        The retreat file the run read.
    fields : dict of str to numpy.ndarray
        Time means and last-record fields on the run grid.
    records : list of dict
        Per-record, per-region statistics.
    """

    label: str
    path: Path
    ds: xr.Dataset = field(init=False, repr=False)
    retreat_file: str | None = None
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)


def to_gt_per_year(da: xr.DataArray, cell_area: float) -> xr.DataArray:
    """
    Convert a per-cell mass flux or a flux density to Gt/yr per cell.

    Parameters
    ----------
    da : xarray.DataArray
        Flux with a ``units`` attribute, e.g. ``Gt year^-1`` or ``kg m^-2 year^-1``.
    cell_area : float
        Cell area in m^2, used for flux densities.

    Returns
    -------
    xarray.DataArray
        The flux in Gt/yr per cell, missing values as zero.

    Raises
    ------
    ValueError
        If the units are neither mass per time nor mass per area and time.
    """
    ureg = pint_xarray.unit_registry
    quantity = ureg.Quantity(1.0, da.attrs.get("units", "Gt year^-1"))
    if quantity.check("[mass] / [time]"):
        factor = quantity.to("Gt / year").magnitude
    elif quantity.check("[mass] / [length] ** 2 / [time]"):
        factor = quantity.to("Gt / m ** 2 / year").magnitude * cell_area
    else:
        raise ValueError(f"{da.name}: cannot convert {da.attrs.get('units')!r} to Gt/yr per cell")
    return da.fillna(0.0) * factor


def _seconds(delta: np.ndarray) -> np.ndarray:
    """
    Turn an array of time differences into seconds.

    Parameters
    ----------
    delta : numpy.ndarray
        ``timedelta64`` values or ``datetime.timedelta`` objects (cftime).

    Returns
    -------
    numpy.ndarray
        Seconds as floats.
    """
    try:
        return delta.astype("timedelta64[s]").astype(float)
    except (TypeError, ValueError):
        return np.array([d.total_seconds() for d in delta.ravel()]).reshape(delta.shape)


def record_weights(ds: xr.Dataset) -> tuple[np.ndarray, pd.Timestamp, pd.Timestamp]:
    """
    Length of each record's averaging interval and the run's start and end.

    Parameters
    ----------
    ds : xarray.Dataset
        Spatial output with a ``time`` axis and, ideally, its bounds.

    Returns
    -------
    weights : numpy.ndarray
        Seconds per record (ones when there are no bounds).
    start, end : pandas.Timestamp
        First bound and last bound (or first and last time).
    """
    name = ds["time"].attrs.get("bounds", "time_bounds")
    if name in ds:
        bounds = ds[name].values
        weights = _seconds(bounds[:, 1] - bounds[:, 0])
        start, end = bounds[0, 0], bounds[-1, 1]
    else:
        weights = np.ones(ds.sizes["time"])
        start, end = ds["time"].values[0], ds["time"].values[-1]
    return weights, pd.Timestamp(str(start)), pd.Timestamp(str(end))


def retreat_fraction(
    path: str, x: np.ndarray, y: np.ndarray, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[np.ndarray, int]:
    """
    The retreat mask a run saw, averaged over its period and put on its grid.

    Parameters
    ----------
    path : str
        Retreat file with ``land_ice_area_fraction_retreat``.
    x, y : numpy.ndarray
        The run's cell centres.
    start, end : pandas.Timestamp
        The run's period; the records in ``[start, end)`` are averaged, or the
        one nearest ``start`` when none fall inside.

    Returns
    -------
    fraction : numpy.ndarray
        ``(y, x)`` mask, linearly interpolated when the grids differ (as PISM
        regrids forcing) and rounded to 1/1000 as PISM does.
    n_records : int
        Records averaged.
    """
    with xr.open_dataset(path, chunks={"time": 1}) as ds:
        mask = ds[RETREAT_VAR]
        n_records = 1
        if "time" in mask.dims:
            times = pd.DatetimeIndex([pd.Timestamp(str(t)) for t in mask["time"].values])
            inside = np.flatnonzero((times >= start) & (times < end))
            if inside.size == 0:
                inside = np.array([np.abs(times - start).argmin()])
            n_records = int(inside.size)
            mask = mask.isel(time=inside).mean("time")
        same = (
            mask.sizes["x"] == x.size
            and mask.sizes["y"] == y.size
            and np.allclose(np.sort(mask["x"].values), np.sort(x))
            and np.allclose(np.sort(mask["y"].values), np.sort(y))
        )
        mask = mask.sortby("x").sortby("y")
        mask = mask.reindex(x=x, y=y, method="nearest") if same else mask.interp(x=x, y=y, method="linear")
        values = mask.transpose("y", "x").values.astype(float)
    return np.round(np.nan_to_num(values, nan=0.0) * 1000.0) / 1000.0, n_records


def classify(fraction: np.ndarray) -> np.ndarray:
    """
    Class each cell by what PISM's prescribed retreat does to it.

    Parameters
    ----------
    fraction : numpy.ndarray
        Retreat mask after rounding.

    Returns
    -------
    numpy.ndarray
        0 where the ice is removed, 1 where the cell becomes a partial cell, 2 where it is left alone.
    """
    return np.where(fraction <= 0.0, 0, np.where(fraction < 1.0, 1, 2)).astype("int8")


def touches(cond: np.ndarray) -> np.ndarray:
    """
    Find the cells with at least one of their four neighbours in ``cond``.

    Parameters
    ----------
    cond : numpy.ndarray
        Boolean field.

    Returns
    -------
    numpy.ndarray
        Boolean field of the same shape.
    """
    p = np.pad(cond, 1, constant_values=False)
    return p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]


def region_labels(outline: str, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Label every cell with its ISMIP region, the ocean with the nearest one.

    Parameters
    ----------
    outline : str
        Basin outline with a ``SUBREGION1`` column (Mouginot basins).
    x, y : numpy.ndarray
        Cell centres; y in either order.

    Returns
    -------
    labels : numpy.ndarray
        ``(y, x)`` integers from 1, indexing ``names`` minus one.
    names : list of str
        Region names.
    """
    basins = gpd.read_file(outline)
    if basins.crs is not None and basins.crs.to_epsg() != 3413:
        basins = basins.to_crs("EPSG:3413")
    column = "SUBREGION1" if "SUBREGION1" in basins else basins.columns[0]
    regions = basins.dissolve(column).reset_index()
    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    transform = from_origin(x.min() - dx / 2, y.max() + dy / 2, dx, dy)
    shapes = [(geom, i + 1) for i, geom in enumerate(regions.geometry)]
    labels = rasterize(shapes, out_shape=(y.size, x.size), transform=transform, fill=0, dtype="int32")
    if y[0] < y[-1]:
        labels = labels[::-1]
    # Grounding-line flux is booked in ocean cells, often outside the basins.
    _, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    return labels[iy, ix], [str(n) for n in regions[column]]


def analyse_run(run: Run, labels: np.ndarray, names: list[str], dx: float) -> None:
    """
    Reduce one run's spatial output to time means and per-record, per-region statistics.

    Parameters
    ----------
    run : Run
        The run; ``run.ds`` must be open. Fills ``run.fields`` and ``run.records``.
    labels : numpy.ndarray
        Region label per cell.
    names : list of str
        Region names.
    dx : float
        Grid spacing in m.
    """
    ds = run.ds
    weights, _, _ = record_weights(ds)
    wsum = weights.sum()
    area = dx * dx
    gl = to_gt_per_year(ds[GL_VAR], area)
    budget = {v: to_gt_per_year(ds[v], area) for v in BUDGET_VARS if v in ds}
    mean = {"gl": np.zeros(labels.shape), "gl_freq": np.zeros(labels.shape)}
    mean.update({v: np.zeros(labels.shape) for v in budget})
    speed_name = "velsurf_mag" if "velsurf_mag" in ds else None
    for i in range(ds.sizes["time"]):
        w = weights[i] / wsum
        flux = gl.isel(time=i).values
        cell_type = ds["mask"].isel(time=i).fillna(0).values.astype(int)
        thk = ds["thk"].isel(time=i).fillna(0).values
        speed = ds[speed_name].isel(time=i).fillna(0).values if speed_name else np.zeros_like(thk)
        ocean = (cell_type == FLOATING) | (cell_type == ICE_FREE_OCEAN)
        at_gl = (cell_type == GROUNDED) & touches(ocean)
        mean["gl"] += w * flux
        mean["gl_freq"] += w * at_gl
        for v, da in budget.items():
            mean[v] += w * da.isel(time=i).values
        proxy = RHO_ICE * thk * speed * dx / 1e12  # Gt/yr per grounding-line cell edge
        time = pd.Timestamp(str(ds["time"].values[i]))
        for k, name in enumerate(["GIS", *names]):
            sel = np.ones(labels.shape, bool) if k == 0 else labels == k
            gl_cells = at_gl & sel
            run.records.append(
                {
                    "run": run.label,
                    "time": time,
                    "weight": w,
                    "region": name,
                    "gl_flux": flux[sel].sum(),
                    "gl_flux_into_floating": flux[sel & (cell_type == FLOATING)].sum(),
                    "gl_flux_into_ice_free_ocean": flux[sel & (cell_type == ICE_FREE_OCEAN)].sum(),
                    "gl_cells": int(gl_cells.sum()),
                    "gl_thk_mean": thk[gl_cells].mean() if gl_cells.any() else np.nan,
                    "gl_speed_mean": speed[gl_cells].mean() if gl_cells.any() else np.nan,
                    "gl_proxy_flux": -proxy[gl_cells].sum(),
                    "grounded_area_km2": (sel & (cell_type == GROUNDED)).sum() * area / 1e6,
                    "floating_area_km2": (sel & (cell_type == FLOATING)).sum() * area / 1e6,
                    **{
                        v.replace("tendency_of_ice_mass", "dmdt"): budget[v].isel(time=i).values[sel].sum()
                        for v in budget
                    },
                }
            )
    last = ds.sizes["time"] - 1
    mean["thk_last"] = ds["thk"].isel(time=last).fillna(0).values
    mean["mask_last"] = ds["mask"].isel(time=last).fillna(0).values
    mean["speed_last"] = ds[speed_name].isel(time=last).fillna(0).values if speed_name else np.zeros(labels.shape)
    run.fields = mean


def region_table(runs: Sequence[Run]) -> pd.DataFrame:
    """
    Time-weighted means per run and region, side by side, with the split of the difference.

    Parameters
    ----------
    runs : sequence of Run
        The two analysed runs.

    Returns
    -------
    pandas.DataFrame
        One row per region; columns ``<quantity>_<label>`` plus the differences
        and the log-ratio split of the flux into length, thickness and speed.
    """
    frames = []
    for run in runs:
        rec = pd.DataFrame(run.records)
        numeric = rec.drop(columns=["run", "time", "weight"]).set_index("region")
        mean = numeric.mul(rec.set_index("region")["weight"], axis=0).groupby(level=0, sort=False).sum()
        frames.append(mean.add_suffix(f"_{run.label}"))
    table = pd.concat(frames, axis=1)
    a, b = (r.label for r in runs)
    table[f"gl_flux_{b}-{a}"] = table[f"gl_flux_{b}"] - table[f"gl_flux_{a}"]
    table[f"gl_flux_{b}/{a}"] = table[f"gl_flux_{b}"] / table[f"gl_flux_{a}"]
    for q, short in [("gl_cells", "length"), ("gl_thk_mean", "thickness"), ("gl_speed_mean", "speed")]:
        table[f"log_ratio_{short}"] = np.log(table[f"{q}_{b}"] / table[f"{q}_{a}"])
    table["log_ratio_proxy"] = np.log(table[f"gl_proxy_flux_{b}"] / table[f"gl_proxy_flux_{a}"])
    table["log_ratio_gl_flux"] = np.log(table[f"gl_flux_{b}"] / table[f"gl_flux_{a}"])
    return table


def tile_table(runs: Sequence[Run], masks: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, labels: np.ndarray,
               names: list[str], tile: int, top: int) -> pd.DataFrame:  # fmt: skip
    """
    Rank ``tile`` x ``tile`` blocks by the difference in grounding-line flux.

    Parameters
    ----------
    runs : sequence of Run
        The two analysed runs.
    masks : dict of str to numpy.ndarray
        Retreat-mask class per cell and run label.
    x, y : numpy.ndarray
        Cell centres.
    labels : numpy.ndarray
        Region label per cell.
    names : list of str
        Region names.
    tile : int
        Block size in cells.
    top : int
        Blocks kept.

    Returns
    -------
    pandas.DataFrame
        The ``top`` blocks by absolute difference, with centre, region and per-run statistics.
    """
    a, b = runs
    ny, nx = labels.shape
    by, bx = np.arange(ny) // tile, np.arange(nx) // tile
    block = by[:, None] * (bx.max() + 1) + bx[None, :]
    nblocks = (by.max() + 1) * (bx.max() + 1)

    def bsum(values):
        """
        Sum a field over each block.

        Parameters
        ----------
        values : array-like
            Field on the run grid.

        Returns
        -------
        numpy.ndarray
            One sum per block.
        """
        return np.bincount(block.ravel(), weights=np.asarray(values, float).ravel(), minlength=nblocks)

    diff = bsum(b.fields["gl"]) - bsum(a.fields["gl"])
    order = [k for k in np.argsort(-np.abs(diff))[:top] if diff[k] != 0]
    disagree = masks[a.label] != masks[b.label]
    to_lonlat = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
    at_gl = {}
    for run in runs:
        cell_type = run.fields["mask_last"]
        at_gl[run.label] = (cell_type == GROUNDED) & touches((cell_type == FLOATING) | (cell_type == ICE_FREE_OCEAN))
    rows = []
    for rank, k in enumerate(order, 1):
        cells = block == k
        # The flux is booked in the ocean cell; the grounded cell feeding it
        # may sit across the tile edge.
        grown = ndimage.binary_dilation(cells)
        iy, ix = np.nonzero(cells)
        cx, cy = x[ix].mean(), y[iy].mean()
        lon, lat = to_lonlat.transform(cx, cy)
        row = {
            "rank": rank,
            "x": cx,
            "y": cy,
            "lon": lon,
            "lat": lat,
            "region": names[np.bincount(labels[cells]).argmax() - 1],
            f"gl_flux_{b.label}-{a.label}": diff[k],
            "mask_class_disagreements": int(disagree[cells].sum()),
        }
        for run in runs:
            f = run.fields
            gl_cells = at_gl[run.label] & grown
            row[f"gl_flux_{run.label}"] = f["gl"][cells].sum()
            row[f"gl_cells_{run.label}"] = int(gl_cells.sum())
            row[f"gl_thk_{run.label}"] = f["thk_last"][gl_cells].mean() if gl_cells.any() else np.nan
            row[f"gl_speed_{run.label}"] = f["speed_last"][gl_cells].mean() if gl_cells.any() else np.nan
            row[f"partial_cells_{run.label}"] = int((masks[run.label][cells] == 1).sum())
            row[f"grounded_cells_{run.label}"] = int((f["mask_last"][cells] == GROUNDED).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_timeseries(records: pd.DataFrame, filename: Path) -> None:
    """
    Plot the ice-sheet totals per record for both runs.

    Parameters
    ----------
    records : pandas.DataFrame
        Per-record statistics of both runs.
    filename : pathlib.Path
        Figure to write.
    """
    gis = records[records["region"] == "GIS"]
    panels = [
        ("gl_flux", "GL flux (Gt/yr)"),
        ("gl_proxy_flux", "rho H u dx at GL (Gt/yr)"),
        ("dmdt_due_to_discharge", "discharge (Gt/yr)"),
        ("gl_cells", "GL cells"),
        ("grounded_area_km2", "grounded area (km2)"),
        ("floating_area_km2", "floating area (km2)"),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for ax, (col, label) in zip(axs.flat, panels):
        if col not in gis:
            ax.set_visible(False)
            continue
        for run, sub in gis.groupby("run", sort=False):
            ax.plot(sub["time"], sub[col], marker="o", ms=3, label=run)
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)
    axs.flat[0].legend()
    fig.savefig(filename, dpi=120)
    plt.close(fig)


def plot_regions(table: pd.DataFrame, labels: Sequence[str], filename: Path) -> None:
    """
    Bar chart of the grounding-line flux and its proxy per region.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`region_table`.
    labels : sequence of str
        The two run labels.
    filename : pathlib.Path
        Figure to write.
    """
    regions = [r for r in table.index if r != "GIS"]
    xpos = np.arange(len(regions))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, col, title in [(axs[0], "gl_flux", "GL flux (Gt/yr)"), (axs[1], "gl_proxy_flux", "rho H u dx at GL")]:
        for j, label in enumerate(labels):
            ax.bar(xpos + (j - 0.5) * 0.4, table.loc[regions, f"{col}_{label}"], 0.4, label=label)
        ax.set_xticks(xpos, regions)
        ax.set_title(title)
    axs[0].legend()
    fig.savefig(filename, dpi=120)
    plt.close(fig)


def plot_tile(runs: Sequence[Run], fractions: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray,
              row: pd.Series, half_width: float, filename: Path) -> None:  # fmt: skip
    """
    Zoom on one tile: thickness with the grounding line, the flux, the retreat mask, per run.

    Parameters
    ----------
    runs : sequence of Run
        The two analysed runs.
    fractions : dict of str to numpy.ndarray
        Retreat mask per run label on the run grid.
    x, y : numpy.ndarray
        Cell centres.
    row : pandas.Series
        The tile's row of :func:`tile_table`.
    half_width : float
        Half the window, in m.
    filename : pathlib.Path
        Figure to write.
    """
    ix = np.flatnonzero(np.abs(x - row["x"]) <= half_width)
    iy = np.flatnonzero(np.abs(y - row["y"]) <= half_width)
    win = np.ix_(iy, ix)
    ext = [x[ix[0]] / 1e3, x[ix[-1]] / 1e3, y[iy[0]] / 1e3, y[iy[-1]] / 1e3]
    origin = "lower" if y[0] < y[-1] else "upper"
    vmax = max(np.abs(r.fields["gl"][win]).max() for r in runs) or 1.0
    fig, axs = plt.subplots(len(runs), 3, figsize=(12, 4 * len(runs)), layout="constrained", squeeze=False)
    for axrow, run in zip(axs, runs):
        f = run.fields
        grounded = (f["mask_last"][win] == GROUNDED).astype(float)
        im = axrow[0].imshow(f["thk_last"][win], origin=origin, extent=ext, cmap="Blues", vmin=0)
        axrow[0].contour(grounded, levels=[0.5], colors="k", linewidths=0.8, origin=origin, extent=ext)
        fig.colorbar(im, ax=axrow[0], shrink=0.8, label="thk (m), last record; black: grounded")
        gl = np.where(f["gl"][win] != 0, f["gl"][win], np.nan)
        im = axrow[1].imshow(gl, origin=origin, extent=ext, cmap="RdBu", vmin=-vmax, vmax=vmax)
        axrow[1].contour(grounded, levels=[0.5], colors="k", linewidths=0.5, origin=origin, extent=ext)
        fig.colorbar(im, ax=axrow[1], shrink=0.8, label="GL flux per cell (Gt/yr), mean")
        im = axrow[2].imshow(
            fractions[run.label][win], origin=origin, extent=ext, cmap=mpl.colormaps["viridis"], vmin=0, vmax=1
        )
        partial = (fractions[run.label][win] > 0) & (fractions[run.label][win] < 1)
        pyx = np.nonzero(partial)
        axrow[2].scatter(x[ix][pyx[1]] / 1e3, y[iy][pyx[0]] / 1e3, s=4, c="r", marker="s", label="0 < f < 1")
        axrow[2].contour(grounded, levels=[0.5], colors="w", linewidths=0.5, origin=origin, extent=ext)
        fig.colorbar(im, ax=axrow[2], shrink=0.8, label="retreat mask f on the run grid")
        axrow[2].legend(fontsize=7, loc="lower right")
        axrow[0].set_ylabel(run.label)
        for ax in axrow:
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
    fig.suptitle(
        f"#{int(row['rank'])} {row['region']} ({row['lat']:.2f}N, {-row['lon']:.2f}W): "
        f"GL flux {runs[1].label}-{runs[0].label} = {row[f'gl_flux_{runs[1].label}-{runs[0].label}']:.2f} Gt/yr"
    )
    fig.savefig(filename, dpi=110)
    plt.close(fig)


def summarize(runs: Sequence[Run], table: pd.DataFrame, fractions: dict[str, np.ndarray],
              classes: dict[str, np.ndarray], n_records: dict[str, int]) -> str:  # fmt: skip
    """
    Put the headline numbers and what they point at into words.

    Parameters
    ----------
    runs : sequence of Run
        The two analysed runs.
    table : pandas.DataFrame
        Output of :func:`region_table`.
    fractions : dict of str to numpy.ndarray
        Retreat mask per run label.
    classes : dict of str to numpy.ndarray
        Retreat-mask class per run label.
    n_records : dict of str to int
        Retreat-mask records averaged per run label.

    Returns
    -------
    str
        The summary.
    """
    a, b = runs
    gis = table.loc["GIS"]
    lines = [f"Runs: {a.label} = {a.path}", f"      {b.label} = {b.path}", ""]
    for run in runs:
        c = classes[run.label]
        lines.append(
            f"Retreat mask {run.label}: {run.retreat_file} ({n_records[run.label]} record(s) averaged); "
            f"cells none {int((c == 0).sum())}, partial {int((c == 1).sum())}, full {int((c == 2).sum())}; "
            f"distinct values {np.unique(fractions[run.label]).size}"
        )
    lines.append("")
    lines.append("Ice sheet, time-weighted mean (Gt/yr; negative = grounded ice lost to the ocean):")
    for q in ["gl_flux", "gl_flux_into_floating", "gl_flux_into_ice_free_ocean", "gl_proxy_flux",
              "dmdt_due_to_discharge", "dmdt_due_to_flow", "dmdt", "gl_cells", "gl_thk_mean", "gl_speed_mean",
              "grounded_area_km2", "floating_area_km2"]:  # fmt: skip
        if f"{q}_{a.label}" in gis:
            va, vb = gis[f"{q}_{a.label}"], gis[f"{q}_{b.label}"]
            lines.append(f"  {q:30s} {a.label:>8s} {va:12.2f}   {b.label:>8s} {vb:12.2f}   diff {vb - va:12.2f}")
    lines.append("")
    lines.append(
        f"log({b.label}/{a.label}) of the GL flux {gis['log_ratio_gl_flux']:+.3f}; of rho H u dx "
        f"{gis['log_ratio_proxy']:+.3f} = length {gis['log_ratio_length']:+.3f} + thickness "
        f"{gis['log_ratio_thickness']:+.3f} + speed {gis['log_ratio_speed']:+.3f} (+ covariance)"
    )
    disagree = classes[a.label] != classes[b.label]
    near = ndimage.binary_dilation(disagree, iterations=2)
    dgl = np.abs(b.fields["gl"] - a.fields["gl"])
    share = dgl[near].sum() / dgl.sum() if dgl.sum() else np.nan
    lines.append(
        f"Cells whose mask class differs: {int(disagree.sum())}; they and their 2-cell neighbourhood "
        f"hold {share:.0%} of the cell-wise |GL flux difference|."
    )
    for run in runs:
        own = classes[run.label]
        gl = run.fields["gl"]
        parts = ", ".join(f"{name} {gl[own == k].sum():.1f}" for k, name in CLASSES.items())
        lines.append(f"GL flux of {run.label} booked in cells whose own mask is: {parts} Gt/yr")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Compare the grounding-line flux of two runs and write tables and figures.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; ``sys.argv[1:]`` when None.

    Returns
    -------
    int
        Exit status.
    """
    parser = ArgumentParser(description=__doc__.split("\n\n", 1)[0], formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("FILES", nargs=2, help="The two spatial files.")
    parser.add_argument("--labels", default="A,B", help="Comma-separated names of the two runs.")
    parser.add_argument("--retreat-files", default=None, help="Comma-separated retreat files, overriding 'command'.")
    parser.add_argument("--outline", default=str(files("pism_terra.data") / "mouginot_basins_w_shelves.gpkg"))
    parser.add_argument("--output-dir", default="gl_mask_diagnosis")
    parser.add_argument("--tile", type=int, default=10, help="Block size in cells for ranking the differences.")
    parser.add_argument("--top", type=int, default=25, help="Blocks listed in tiles.csv.")
    parser.add_argument("--plot-top", type=int, default=8, help="Blocks drawn as zoom maps.")
    parser.add_argument("--half-width", type=float, default=25e3, help="Half-width of the zoom maps in m.")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = [s.strip() for s in args.labels.split(",")]
    runs = [Run(label, Path(p)) for label, p in zip(labels, args.FILES)]
    overrides = args.retreat_files.split(",") if args.retreat_files else [None, None]
    for run, override in zip(runs, overrides):
        run.ds = xr.open_dataset(run.path, chunks={"time": 1})
        found = RETREAT_RE.search(run.ds.attrs.get("command", ""))
        run.retreat_file = override or (found.group(1) if found else None)
        if run.retreat_file is None:
            raise SystemExit(f"{run.path}: no retreat file in 'command'; pass --retreat-files")
        missing = [v for v in (GL_VAR, "thk", "mask") if v not in run.ds]
        if missing:
            raise SystemExit(f"{run.path}: missing {missing}")
    x, y = runs[0].ds["x"].values.astype(float), runs[0].ds["y"].values.astype(float)
    if not (np.array_equal(x, runs[1].ds["x"].values) and np.array_equal(y, runs[1].ds["y"].values)):
        raise SystemExit("The two runs are not on the same grid.")
    dx = abs(x[1] - x[0])
    region, names = region_labels(args.outline, x, y)

    fractions, classes, n_records = {}, {}, {}
    for run in runs:
        _, start, end = record_weights(run.ds)
        print(f"{run.label}: reading {run.retreat_file} for {start:%Y-%m-%d} to {end:%Y-%m-%d}", flush=True)
        fractions[run.label], n_records[run.label] = retreat_fraction(str(run.retreat_file), x, y, start, end)
        classes[run.label] = classify(fractions[run.label])
        print(f"{run.label}: reducing {run.ds.sizes['time']} record(s)", flush=True)
        analyse_run(run, region, names, dx)

    records = pd.concat([pd.DataFrame(r.records) for r in runs], ignore_index=True)
    records.to_csv(out / "timeseries.csv", index=False)
    table = region_table(runs)
    table.to_csv(out / "regions.csv")
    tiles = tile_table(runs, classes, x, y, region, names, args.tile, args.top)
    tiles.to_csv(out / "tiles.csv", index=False)

    a, b = runs
    fields = xr.Dataset(
        {
            **{f"gl_flux_{r.label}": (("y", "x"), r.fields["gl"]) for r in runs},
            f"gl_flux_{b.label}_minus_{a.label}": (("y", "x"), b.fields["gl"] - a.fields["gl"]),
            **{f"retreat_fraction_{r.label}": (("y", "x"), fractions[r.label]) for r in runs},
            **{f"retreat_class_{r.label}": (("y", "x"), classes[r.label]) for r in runs},
            **{f"thk_last_{r.label}": (("y", "x"), r.fields["thk_last"]) for r in runs},
            **{f"mask_last_{r.label}": (("y", "x"), r.fields["mask_last"].astype("int8")) for r in runs},
            "region": (("y", "x"), region),
        },
        coords={"x": x, "y": y},
        attrs={"regions": ", ".join(f"{i + 1}: {n}" for i, n in enumerate(names)), "proj": "EPSG:3413"},
    )
    fields.to_netcdf(out / "fields.nc")

    plot_timeseries(records, out / "timeseries.png")
    plot_regions(table, labels, out / "regions.png")
    for _, row in tiles.head(args.plot_top).iterrows():
        plot_tile(runs, fractions, x, y, row, args.half_width, out / f"tile_{int(row['rank']):02d}.png")

    text = summarize(runs, table, fractions, classes, n_records)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 2)
    cols = [
        f"{q}_{r.label}" for q in ("gl_flux", "gl_proxy_flux", "gl_cells", "gl_thk_mean", "gl_speed_mean") for r in runs
    ]
    cols += [f"gl_flux_{b.label}-{a.label}", "log_ratio_gl_flux", "log_ratio_length", "log_ratio_thickness",
             "log_ratio_speed"]  # fmt: skip
    text += "\n\nPer region (flux Gt/yr, thickness m, speed m/yr):\n" + table[cols].to_string()
    show = ["rank", "region", "lat", "lon", f"gl_flux_{b.label}-{a.label}", "mask_class_disagreements"]
    show += [f"{q}_{r.label}" for q in ("gl_cells", "gl_thk", "gl_speed", "partial_cells") for r in runs]
    text += "\n\nLargest differences by tile:\n" + tiles[show].head(15).to_string(index=False)
    (out / "summary.txt").write_text(text + "\n")
    print(text)
    print(f"\nWritten to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
