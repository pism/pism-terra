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
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
CDO grid description files for common projections, and point sampling of gridded data.

:func:`sample_points` pulls the values of a projected ``(y, x)`` dataset at a
set of sites, for any dataset with a CF grid mapping — a PISM output, a CARRA2
or ERA5 forcing store, a boot file — rather than the stake-specific sampler in
:mod:`pism_terra.glacier.usgs_benchmark_stakes`.
"""

import logging
from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr

from pism_terra.workflow import dataset_crs

logger = logging.getLogger(__name__)

_GRIDS_DIR = Path(__file__).parent / "grids"


def load_grid(name: str) -> str:
    """
    Load a CDO grid description by name.

    Parameters
    ----------
    name : str
        Grid name (e.g., ``"ismip6"``, ``"carra2"``, ``"hirham"``, ``"bedmachine"``).
        The ``.txt`` extension is appended automatically.

    Returns
    -------
    str
        Contents of the grid description file.
    """
    return (_GRIDS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def _spacing(coord: np.ndarray) -> float:
    """
    Mean absolute step of a one-dimensional coordinate.

    Parameters
    ----------
    coord : numpy.ndarray
        Coordinate values.

    Returns
    -------
    float
        The spacing; zero for a single-element coordinate.
    """
    return float(np.abs(np.diff(coord)).mean()) if coord.size > 1 else 0.0


def sample_points(
    ds: xr.Dataset,
    points: gpd.GeoDataFrame,
    *,
    variables: Iterable[str] | None = None,
    method: str = "nearest",
    crs: str | None = None,
    dim: str = "site",
    label: str | None = None,
    x_dim: str = "x",
    y_dim: str = "y",
) -> xr.Dataset:
    """
    Sample a projected dataset at a set of points.

    The sites are reprojected to the dataset's own CRS, so the grid is never
    touched. Only the bounding box of the sites (plus a one-cell margin for
    the bilinear stencil) is read, so sampling a large or remote store — a
    daily CARRA2 Zarr on S3, say — costs a few cells per time step rather
    than the whole grid. Any extra dimensions (``time``, ``pressure_level``,
    …) are carried through untouched.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset on a projected ``(y, x)`` grid with its grid-mapping
        variable still present (``mapping`` or ``spatial_ref`` with
        ``crs_wkt``), unless *crs* is given.
    points : geopandas.GeoDataFrame
        Point geometries with a CRS. Every non-geometry column becomes a
        coordinate on *dim*.
    variables : iterable of str or None, optional
        Variables to sample; ``ds.data_vars`` works. None samples every
        variable that spans both horizontal dimensions. A listed variable
        that lacks them is carried through unchanged when it has other
        dimensions (``time_bounds``) and dropped when it is a scalar (the
        grid mapping). A listed variable the dataset lacks is skipped with
        a warning.
    method : {"nearest", "linear"}, default "nearest"
        Nearest cell, or bilinear interpolation from the four neighbours. A
        bilinear value whose stencil touches a NaN cell falls back to the
        nearest cell for that site and step.
    crs : str or None, optional
        CRS of the grid when the dataset carries no usable grid mapping.
    dim : str, default "site"
        Name of the point dimension in the result.
    label : str or None, optional
        Column of *points* that labels the sites. Defaults to a ``site``
        column when there is one, else the frame's index.
    x_dim, y_dim : str, default "x", "y"
        Names of the horizontal dimensions.

    Returns
    -------
    xarray.Dataset
        The sampled variables on ``(…, dim)``, with the point columns,
        ``x``/``y`` in the grid's CRS and ``lon``/``lat`` as coordinates on
        *dim*. Coordinates that span the grid (2-D ``latitude``,
        ``longitude``) are dropped in favour of those. ``attrs["sampling"]``
        records the method and ``attrs["crs"]`` the grid's CRS.

    Raises
    ------
    ValueError
        When no point lies inside the grid, *method* is unknown, *label* is
        not a column of *points*, or the dataset has no grid mapping and no
        *crs* was given.
    """
    if method not in ("nearest", "linear"):
        raise ValueError(f"unknown sampling method {method!r}")
    if label is None and "site" in points.columns:
        label = "site"
    if label is not None and label not in points.columns:
        raise ValueError(f"{label!r} is not a column of the points")

    grid_crs = dataset_crs(ds, crs)
    for name in (x_dim, y_dim):
        if ds[name].size > 1 and ds[name].values[1] < ds[name].values[0]:
            ds = ds.sortby(name)
    dx, dy = _spacing(ds[x_dim].values), _spacing(ds[y_dim].values)

    projected = points.to_crs(grid_crs) if points.crs is not None else points
    xs = projected.geometry.x.to_numpy()
    ys = projected.geometry.y.to_numpy()
    inside = (
        (xs >= float(ds[x_dim].min()) - dx / 2)
        & (xs <= float(ds[x_dim].max()) + dx / 2)
        & (ys >= float(ds[y_dim].min()) - dy / 2)
        & (ys <= float(ds[y_dim].max()) + dy / 2)
    )
    if not inside.all():
        dropped = points[~inside]
        names = dropped[label].tolist() if label is not None else dropped.index.tolist()
        logger.warning("%d of %d points fall outside the grid and are dropped: %s", len(names), len(points), names)
    if not inside.any():
        raise ValueError("no point lies inside the grid")
    points = points[inside].reset_index(drop=True)
    xs, ys = xs[inside], ys[inside]

    spatial = [v for v in ds.data_vars if x_dim in ds[v].dims and y_dim in ds[v].dims]
    if variables is None:
        to_sample, passthrough = spatial, []
    else:
        requested = list(dict.fromkeys(variables))
        missing = [v for v in requested if v not in ds]
        if missing:
            logger.warning("requested variables not in the dataset: %s; skipping them", missing)
        to_sample = [v for v in requested if v in spatial]
        passthrough = [v for v in requested if v in ds and v not in spatial and ds[v].ndim > 0]
    if not to_sample:
        raise ValueError("no variable spans both horizontal dimensions")

    ix = np.asarray(ds.indexes[x_dim].get_indexer(xs, method="nearest"))
    iy = np.asarray(ds.indexes[y_dim].get_indexer(ys, method="nearest"))
    crop = ds[to_sample].isel(
        {
            x_dim: slice(max(int(ix.min()) - 1, 0), int(ix.max()) + 2),
            y_dim: slice(max(int(iy.min()) - 1, 0), int(iy.max()) + 2),
        }
    )
    grid_coords = [c for c in crop.coords if c not in (x_dim, y_dim) and ({x_dim, y_dim} & set(crop[c].dims))]
    crop = crop.drop_vars(grid_coords).load()

    x_sel = xr.DataArray(xs, dims=dim)
    y_sel = xr.DataArray(ys, dims=dim)
    nearest = crop.sel({x_dim: x_sel, y_dim: y_sel}, method="nearest")
    if method == "nearest":
        sampled = nearest
    else:
        sampled = crop.interp({x_dim: x_sel, y_dim: y_sel}, method="linear")
        for var in to_sample:
            gaps = sampled[var].isnull() & nearest[var].notnull()
            if bool(gaps.any()):
                logger.info(
                    "%s: bilinear stencil touches a NaN cell at %d site-steps; using nearest", var, int(gaps.sum())
                )
                sampled[var] = sampled[var].where(~gaps, nearest[var])
            sampled[var].attrs = dict(ds[var].attrs)
    sampled = sampled.drop_vars([x_dim, y_dim], errors="ignore")
    for var in passthrough:
        sampled[var] = ds[var].load()

    labels = points[label].to_numpy() if label is not None else points.index.to_numpy()
    coords: dict[str, tuple[str, np.ndarray]] = {dim: (dim, labels)}
    for column in points.columns:
        if column != points.geometry.name and column != label and column not in sampled.dims:
            coords[column] = (dim, points[column].to_numpy())
    coords["x"] = (dim, xs)
    coords["y"] = (dim, ys)
    if points.crs is not None:
        geographic = points.to_crs("EPSG:4326")
        coords["lon"] = (dim, geographic.geometry.x.to_numpy())
        coords["lat"] = (dim, geographic.geometry.y.to_numpy())
    sampled = sampled.assign_coords(coords)
    sampled.attrs = dict(ds.attrs)
    sampled.attrs["sampling"] = method
    sampled.attrs["crs"] = grid_crs
    return sampled
