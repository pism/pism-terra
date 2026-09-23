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
Per-basin integrals of the ISMIP7 flux variables.

An ISMIP7 submission directory holds one file per variable, each a full
``(time, y, x)`` field. :data:`FLUX_VARS` are the ones reported *per unit
area* — a surface mass balance in kg m^-2 s^-1, a geothermal heat flux in
W m^-2, a thickness rate in m s^-1 — so the meaningful per-basin number is
the integral over the basin, ``sum(flux * cell_area)``, not the plain sum
:mod:`pism_terra.postprocess_scalar` takes over extensive fields. That
integral is what turns ``acabf`` into a basin's ``tendacabf``, and the units
change with it: per-area units lose their ``m^-2``.

The eight files are opened together with :func:`xarray.open_mfdataset` — they
share a grid and a time axis, so the merge is one dataset with eight
variables — and reduced over every basin in a single Dask pass, the way
:func:`pism_terra.postprocess_scalar.process_file` does it.

The time axis is passed through untouched: read undecoded, written back with
the dtype and attributes it arrived with. ISMIP7 flux time stamps have
already been re-stamped onto the middle of their averaging interval by
:mod:`pism_terra.ismip7.fix_time_flux_variables`, and decoding and
re-encoding them here would quietly undo that.

Console entry point ``pism-ismip7-postprocess-flux``; the forward-run
templates call it on the submission directory at the end of a counter-driven
run.
"""

from __future__ import annotations

import logging
import re
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from dask.distributed import Client, progress

from pism_terra.log import setup_logging
from pism_terra.postprocess_scalar import (
    DEFAULT_COLUMNS,
    _raise_fd_limit,
    basin_masks,
)
from pism_terra.workflow import dataset_crs, drop_grid_mapping, make_cdo_readable

xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)

#: ISMIP7 variables reported per unit area, which therefore need integrating
#: over the basin rather than summing.
FLUX_VARS = {
    "hfgeoubed",
    "acabf",
    "libmassbfgr",
    "libmassbffl",
    "dlithkdt",
    "licalvf",
    "ligroundf",
    "lifmassbf",
}

#: Variables carried through the merge that are not fluxes.
CARRIED_VARS = ("time_bounds", "time_bnds")


def find_flux_files(experiment_dir: Path | str, variables: set[str] = FLUX_VARS) -> dict[str, Path]:
    """
    Locate one file per flux variable in an ISMIP7 experiment directory.

    Submission files are named ``<variable>_<domain>_<group>_…_<years>.nc``,
    so the variable is the part before the first underscore. Matching on that
    rather than on a substring keeps ``acabf`` from also picking up
    ``tendacabf``.

    Parameters
    ----------
    experiment_dir : pathlib.Path or str
        A submission directory, e.g. ``output/GrIS/UAF/PISM/CORE/C005``.
    variables : set of str, optional
        Variable names to look for; defaults to :data:`FLUX_VARS`.

    Returns
    -------
    dict
        Variable name to file, for those present. Missing variables are
        logged and left out rather than raising: a run may legitimately not
        report all of them.
    """
    experiment_dir = Path(experiment_dir)
    found: dict[str, Path] = {}
    for variable in sorted(variables):
        matches = sorted(f for f in experiment_dir.glob(f"{variable}_*.nc") if f.name.split("_")[0] == variable)
        if not matches:
            logger.warning("%s: no file for %s", experiment_dir.name, variable)
            continue
        if len(matches) > 1:
            logger.warning(
                "%s: %d files for %s, using %s", experiment_dir.name, len(matches), variable, matches[0].name
            )
        found[variable] = matches[0]
    return found


def output_name(files: dict[str, Path], prefix: str = "basin_flux") -> str:
    """
    Name the basin file after the experiment the flux files belong to.

    Every file in a submission directory shares everything after its variable
    name, so dropping that prefix from any of them leaves the experiment's own
    stem.

    Parameters
    ----------
    files : dict
        Variable name to file, from :func:`find_flux_files`.
    prefix : str, optional
        Prefix of the written file.

    Returns
    -------
    str
        File name, e.g.
        ``basin_flux_GrIS_UAF_PISM_m001_CESM2-WACCM_f001_ssp126_C005_2015-2299.nc``.

    Raises
    ------
    ValueError
        If *files* is empty.
    """
    if not files:
        raise ValueError("no flux files to name the output after")
    variable, path = sorted(files.items())[0]
    return f"{prefix}_{path.name[len(variable) + 1 :]}"


def cell_area(ds: xr.Dataset) -> float:
    """
    Area of one grid cell, in m^2.

    ISMIP7 submissions sit on a regular projected grid, so this is the
    spacing product and the same everywhere. It is read from the coordinates
    rather than assumed, since the submission grid (1 km) is not the grid the
    simulation ran on.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with ``x`` and ``y`` coordinates in metres.

    Returns
    -------
    float
        Cell area in m^2.

    Raises
    ------
    ValueError
        If either coordinate has fewer than two points, or is not evenly
        spaced to within a tenth of a percent — in which case a single cell
        area is the wrong model and the caller needs a per-cell one.
    """
    spacings = []
    for dim in ("x", "y"):
        values = np.asarray(ds[dim].values, dtype=float)
        if values.size < 2:
            raise ValueError(f"cannot infer cell area: {dim} has {values.size} point(s)")
        steps = np.abs(np.diff(values))
        spacing = float(steps.mean())
        if np.ptp(steps) > 1e-3 * spacing:
            raise ValueError(f"{dim} is not evenly spaced ({steps.min()}..{steps.max()} m); cell area is not constant")
        spacings.append(spacing)
    return spacings[0] * spacings[1]


def integrated_units(units: str | None) -> str | None:
    """
    Give the units of a field once it has been multiplied by an area.

    Not simply "drop the ``m^-2``": ``dlithkdt`` is a thickness rate in
    m s^-1, and integrating *that* over an area gives a volume rate in
    m^3 s^-1. So the metre exponent gains two wherever it appears, and the
    factor disappears only when that takes it to zero.

    Parameters
    ----------
    units : str or None
        Units as the submission file states them, e.g. ``"kg m^-2 s^-1"``,
        ``"W m^-2"`` or ``"m s^-1"``.

    Returns
    -------
    str or None
        The units of the integral: ``"kg s^-1"``, ``"W"``, ``"m^3 s^-1"``.
        A string with no metre factor gets one appended; ``None`` and the
        empty string pass through.
    """
    if not units:
        return units
    # A metre factor is a standalone "m", optionally with an exponent written
    # ^N, ^-N, **N or a bare -N. Anything else (kg, s, W, second) is left be.
    pattern = re.compile(r"(?<![A-Za-z])m(?:\^|\*\*)?(-?\d+)?(?![A-Za-z0-9])")
    match = pattern.search(units)
    if match is None:
        return f"{units} m^2"
    exponent = int(match.group(1)) if match.group(1) else 1
    exponent += 2
    if exponent == 0:
        # Drop the factor and the separator that went with it.
        out = units[: match.start()] + units[match.end() :]
        return " ".join(out.split())
    replacement = "m" if exponent == 1 else f"m^{exponent}"
    return units[: match.start()] + replacement + units[match.end() :]


def submission_crs(ds: xr.Dataset, crs: str | None = None) -> str:
    """
    CRS of an ISMIP7 submission file.

    The conforming files carry a CF grid mapping, but it states the
    projection as PISM's ``proj_params`` (``"EPSG:3413"``) rather than the
    ``crs_wkt`` :func:`pism_terra.workflow.dataset_crs` looks for, so that
    helper alone rejects them. Try it first — it handles everything PISM
    writes directly — and fall back to ``proj_params``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with a grid-mapping variable.
    crs : str or None, optional
        Explicit override; returned as given.

    Returns
    -------
    str
        Something :meth:`geopandas.GeoDataFrame.to_crs` accepts.

    Raises
    ------
    ValueError
        If neither route finds a projection.
    """
    if crs is not None:
        return crs
    try:
        return dataset_crs(ds, None)
    except ValueError:
        for name in ("mapping", "polar_stereographic", "crs", "spatial_ref"):
            if name in ds.variables:
                params = ds[name].attrs.get("proj_params")
                if params:
                    logger.info("Read the projection from %s:proj_params (%s)", name, params)
                    return str(params)
        raise


def process_experiment(
    experiment_dir: Path | str,
    output_dir: Path | str,
    outlinefile: Path | str,
    client: Client,
    *,
    column: str | None = None,
    crs: str | None = None,
    dim_name: str = "basin",
    total_name: str | None = None,
    all_touched: bool = False,
) -> Path | None:
    """
    Integrate one experiment's flux variables over every basin and write the result.

    Parameters
    ----------
    experiment_dir : pathlib.Path or str
        Submission directory holding the per-variable files.
    output_dir : pathlib.Path or str
        Directory the basin file is written to; created if absent.
    outlinefile : pathlib.Path or str
        Basin outlines (GeoPackage/shapefile), in any CRS.
    client : dask.distributed.Client
        Cluster the reduction runs on.
    column : str or None, optional
        Outline column holding the basin name; ``None`` picks the first of
        :data:`pism_terra.postprocess_scalar.DEFAULT_COLUMNS` present.
    crs : str or None, optional
        CRS of the input grid; ``None`` reads it from the grid mapping.
    dim_name : str, optional
        Name of the basin dimension in the output.
    total_name : str or None, optional
        Append a whole-domain row summing every basin under this name.
    all_touched : bool, optional
        Count every cell an outline touches, not only those whose centre it
        contains.

    Returns
    -------
    pathlib.Path or None
        The file written, or None when the directory holds no flux files.
    """
    experiment_dir = Path(experiment_dir)
    files = find_flux_files(experiment_dir)
    if not files:
        logger.warning("%s: no flux variables found; nothing to do", experiment_dir)
        return None
    logger.info("%s: %d flux variables: %s", experiment_dir.name, len(files), ", ".join(sorted(files)))

    started = time.time()
    # Undecoded on the way in and untouched on the way out: these time stamps
    # have already been placed inside their averaging interval by
    # pism-ismip7-fix-time-flux-variables.
    ds = xr.open_mfdataset(
        [str(p) for p in files.values()],
        decode_times=False,
        decode_timedelta=False,
        combine="by_coords",
        compat="no_conflicts",
        join="exact",
        chunks={"time": 1},
        engine="h5netcdf",
    )

    dst_crs = submission_crs(ds, crs)
    outline = gpd.read_file(outlinefile).to_crs(dst_crs)
    area = cell_area(ds)
    logger.info("Cell area %.0f m^2 on the submission grid", area)

    time_var = ds["time"]
    bounds = {name: ds[name] for name in CARRIED_VARS if name in ds.variables}

    fluxes = xr.Dataset({name: ds[name] for name in sorted(files) if name in ds.data_vars})
    fluxes = fluxes.drop_vars(["mapping", "spatial_ref", "crs", "polar_stereographic"], errors="ignore")
    fluxes = fluxes.rio.write_crs(dst_crs).rio.set_spatial_dims(x_dim="x", y_dim="y")

    masks = basin_masks(fluxes, outline, column=column, all_touched=all_touched, client=client)
    logger.info("Integrating over %d basins in a single pass", len(masks))

    # flux * cell area, so the per-basin number is an integral rather than a
    # sum of per-area values. One graph over every basin, so each chunk of the
    # inputs is read once and fans out to all of them.
    integrated = fluxes * area
    lazy = xr.concat(
        [integrated.where(mask).sum(dim=["y", "x"]).expand_dims({dim_name: [name]}) for name, mask in masks],
        dim=dim_name,
    )

    future = client.compute(lazy)
    progress(future)
    basins = future.result()

    for name in basins.data_vars:
        attrs = dict(ds[name].attrs)
        attrs["units"] = integrated_units(attrs.get("units"))
        attrs["cell_area"] = area
        attrs["comment"] = "integrated over the basin: sum(flux * cell_area)"
        basins[name].attrs = attrs

    basins["area"] = xr.DataArray(
        np.asarray([geom.area for geom in outline.geometry], dtype="float64"),
        dims=(dim_name,),
        coords={dim_name: basins[dim_name]},
        attrs={"units": "m^2", "long_name": "outline area"},
    )

    if total_name is not None and total_name not in set(basins[dim_name].values):
        total = basins.sum(dim=dim_name).expand_dims({dim_name: [total_name]})
        basins = xr.concat([basins, total], dim=dim_name)
        logger.info("Added %s as the sum over %d basins", total_name, len(masks))

    # Put time back exactly as it came in — same values, same attributes — and
    # carry its bounds along so the averaging interval survives too.
    basins = basins.assign_coords(time=time_var)
    for name, values in bounds.items():
        basins[name] = values
    basins = drop_grid_mapping(basins)
    basins = make_cdo_readable(basins, dim_name)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / output_name(files)

    encoding = {var: {"zlib": True, "complevel": 2} for var in basins.data_vars}
    # Write time with the on-disk dtype it had, so the axis round-trips
    # bit-for-bit rather than being widened to float64.
    if "dtype" in time_var.encoding:
        encoding["time"] = {"dtype": time_var.encoding["dtype"]}
    logger.info("Writing %s", outfile)
    basins.to_netcdf(outfile, encoding=encoding)
    logger.info("Time elapsed for %s: %.0fs", experiment_dir.name, time.time() - started)
    return outfile


def postprocess_flux(
    experiment_dir: Path | str,
    output_dir: Path | str,
    outlinefile: Path | str,
    n_workers: int = 4,
    local_directory: str | Path | None = None,
    **kwargs,
) -> Path | None:
    """
    Integrate an experiment's fluxes per basin, managing the Dask cluster.

    Parameters
    ----------
    experiment_dir : pathlib.Path or str
        Submission directory holding the per-variable files.
    output_dir : pathlib.Path or str
        Directory the basin file is written to.
    outlinefile : pathlib.Path or str
        Basin outlines.
    n_workers : int, optional
        Dask workers.
    local_directory : str or pathlib.Path or None, optional
        Worker scratch directory; ``None`` uses ``$TMPDIR``. Keep it off
        Lustre.
    **kwargs
        Forwarded to :func:`process_experiment`.

    Returns
    -------
    pathlib.Path or None
        The file written, or None when there was nothing to do.
    """
    import os  # pylint: disable=import-outside-toplevel
    import tempfile  # pylint: disable=import-outside-toplevel

    started = time.time()
    _raise_fd_limit()
    scratch = str(local_directory) if local_directory is not None else tempfile.gettempdir()
    os.makedirs(scratch, exist_ok=True)
    logger.info("Dask worker scratch directory: %s", scratch)

    client = Client(n_workers=n_workers, threads_per_worker=1, local_directory=scratch)
    logger.info("Dask dashboard: %s", client.dashboard_link)
    try:
        outfile = process_experiment(experiment_dir, output_dir, outlinefile, client, **kwargs)
    finally:
        client.close()
    logger.info("Time elapsed %.0fs", time.time() - started)
    return outfile


def main() -> None:
    """
    Run the ``pism-ismip7-postprocess-flux`` command line tool.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = (
        "Integrate the ISMIP7 flux variables of one experiment over each basin. The per-area "
        "fluxes are multiplied by the cell area before summing, so the result is a flux "
        "through the basin rather than a sum of per-area values."
    )
    parser.add_argument("--ntasks", help="Number of Dask workers.", type=int, default=4)
    parser.add_argument(
        "--local-directory",
        help="Dask worker scratch. Point at node-local disk on a cluster; defaults to $TMPDIR.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--column",
        help=f"Outline column holding the basin name. Tried in order {list(DEFAULT_COLUMNS)} when unset.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--crs", help="CRS of the input grid. Read from the grid mapping when unset.", type=str, default=None
    )
    parser.add_argument("--dim-name", help="Name of the basin dimension in the output.", type=str, default="basin")
    parser.add_argument(
        "--total-name",
        help="Append a whole-domain row summing every basin under this name. Off when unset.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--all-touched",
        help="Count every cell an outline touches, not only those whose centre it contains.",
        action="store_true",
        default=False,
    )
    parser.add_argument("EXPERIMENT_DIR", help="Submission directory, e.g. output/GrIS/UAF/PISM/CORE/C005.", nargs=1)
    parser.add_argument("OUTDIR", help="Directory the basin file is written to.", nargs=1)
    parser.add_argument("OUTLINEFILE", help="Basin outline file (GeoPackage/shapefile).", nargs=1)

    options = parser.parse_args()
    output_dir = Path(options.OUTDIR[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "postprocess_flux.log")

    postprocess_flux(
        options.EXPERIMENT_DIR[0],
        output_dir,
        options.OUTLINEFILE[0],
        n_workers=options.ntasks,
        local_directory=options.local_directory,
        column=options.column,
        crs=options.crs,
        dim_name=options.dim_name,
        total_name=options.total_name,
        all_touched=options.all_touched,
    )


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
