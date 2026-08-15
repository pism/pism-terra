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

# pylint: disable=unused-import,unused-variable,too-many-positional-arguments

"""
Postprocessing.
"""

import json
import logging
import os
import resource
import tempfile
import time
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter
from pathlib import Path

import cf_xarray
import dask
import dask.array as dask_array
import geopandas as gpd
import numpy as np
import rioxarray
import toml
import xarray as xr
from dask.distributed import Client, progress
from pyfiglet import Figlet
from rasterio.features import geometry_mask

from pism_terra.log import setup_logging
from pism_terra.workflow import make_cdo_readable, pism_config_value

xr.set_options(keep_attrs=True)
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

logger = logging.getLogger(__name__)


def _raise_fd_limit() -> None:
    """
    Raise the soft open-file limit toward the hard limit.

    Each Dask worker (nanny + process) needs several file descriptors for its
    pipes and sockets, so starting a handful of workers under a low soft
    ``RLIMIT_NOFILE`` (1024 on many clusters) fails with
    ``OSError: [Errno 24] Too many open files`` before any work begins. The
    hard limit is usually far higher, and raising the soft limit up to it needs
    no privileges. Failures are logged and ignored — the caller can still run
    with fewer workers.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # An "unlimited" hard limit cannot be applied verbatim on every platform
        # (macOS rejects it for NOFILE), so aim for a large finite target.
        target = 65536 if hard == resource.RLIM_INFINITY else hard
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            logger.info("Raised open-file limit from %s to %s", soft, target)
    except (ValueError, OSError) as exc:  # pragma: no cover - platform dependent
        logger.warning("Could not raise open-file limit: %s", exc)


def _dim_chunks(ds: xr.Dataset, dim: str, fallback: int) -> tuple[int, ...] | int:
    """
    Chunk sizes along ``dim``, ignoring disagreement along the other dimensions.

    ``Dataset.chunksizes`` raises ``ValueError`` as soon as *any* dimension is
    chunked differently across variables, and ``chunks="auto"`` routinely does
    that along ``time`` because the chunking it picks depends on each
    variable's dtype and rank. Only the y/x chunking is wanted here, so read it
    off the variables directly and take the most common chunking of ``dim``.
    Any leftover mismatch is harmless — Dask rechunks the mask to match its
    operand; matching up front only avoids that extra work.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to inspect. Variables not backed by Dask are ignored.
    dim : str
        Dimension name, e.g. ``"y"``.
    fallback : int
        Returned when no Dask-backed variable carries ``dim`` (a single chunk
        spanning the dimension).

    Returns
    -------
    tuple of int, or int
        Chunk sizes along ``dim``, or ``fallback``.
    """
    counts = Counter(var.chunksizes[dim] for var in ds.variables.values() if dim in var.dims and var.chunks is not None)
    if not counts:
        return fallback
    return counts.most_common(1)[0][0]


def basin_masks(
    ds: xr.Dataset,
    basin: gpd.GeoDataFrame,
    column: str = "SUBREGION1",
    all_touched: bool = False,
    client: Client | None = None,
) -> list[tuple[str, xr.DataArray]]:
    """
    Rasterize each basin polygon onto the dataset grid, once.

    Returns one boolean mask per basin, wrapped as a Dask-backed
    :class:`xarray.DataArray` chunked to match ``ds`` so that masking does not
    force a rechunk. The rasterization matches ``rio.clip(..., drop=False)``
    exactly: :func:`rasterio.features.geometry_mask` with the same transform and
    ``all_touched``, so a cell belongs to a basin under the same rule.

    Separate per-basin masks are used rather than a single integer label array
    because nothing guarantees the outlines partition the grid. Older basin
    files bundle a whole-ice-sheet ``GIS`` polygon that overlaps every regional
    basin, and even ``mouginot_basins_w_shelves.gpkg`` leaves one cell claimed
    by both ``NO`` and ``NW``; a label array can hold only one basin per cell
    and would silently drop the rest.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with a CRS and spatial dims set (see ``rio.write_crs`` /
        ``rio.set_spatial_dims``). Only its grid is used.
    basin : geopandas.GeoDataFrame
        Basin outlines, already in the dataset CRS.
    column : str, default "SUBREGION1"
        Column holding the basin name.
    all_touched : bool, default False
        Passed to :func:`rasterio.features.geometry_mask`. ``False`` selects
        cells whose center falls inside the polygon.
    client : dask.distributed.Client or None, default None
        When given, the rasters are scattered to the workers instead of being
        embedded in the task graph (see below). Without a client the masks ride
        along in the graph, which is fine for small grids.

    Returns
    -------
    list of (str, xarray.DataArray)
        ``(basin name, boolean mask)`` pairs, in ``basin`` row order.

    Raises
    ------
    ValueError
        If ``basin`` has no features (e.g. a truncated or freshly-created
        outline file) or lacks ``column``. Without the check an empty
        outline surfaces as ``client.scatter([])`` blowing up deep inside
        distributed ("not enough values to unpack").
    """
    if basin.empty:
        raise ValueError("outline file contains no features; check the file (e.g. `ogrinfo -so`) and re-copy it")
    if column not in basin.columns:
        raise ValueError(f"outline file has no column {column!r}; available: {list(basin.columns)}")

    transform = ds.rio.transform(recalc=True)
    shape = (int(ds.rio.height), int(ds.rio.width))
    # Match the dataset's own y/x chunking so ``ds.where(mask)`` stays blockwise.
    chunks = (_dim_chunks(ds, "y", shape[0]), _dim_chunks(ds, "x", shape[1]))

    names, rasters = [], []
    for _, row in basin.iterrows():
        mask = geometry_mask(
            [row.geometry],
            out_shape=shape,
            transform=transform,
            invert=True,
            all_touched=all_touched,
        )
        name = row[column]
        if not mask.any():
            logger.warning("Basin %s covers no grid cells; its sums will be zero", name)
        names.append(name)
        rasters.append(mask)

    if client is None:
        arrays = [dask_array.from_array(raster, chunks=chunks) for raster in rasters]
    else:
        # ``from_array`` puts the raster itself into the task graph, so the graph
        # the client ships to the scheduler grows with the grid: one bool per
        # cell per basin, which on the 1500 m Greenland grid is ~1.6 MB x 7
        # basins and trips Dask's "Sending large graph" warning. Scattering
        # instead moves them once, as data, and leaves only future references in
        # the graph.
        arrays = [
            dask_array.from_delayed(dask.delayed(future), shape=shape, dtype=bool).rechunk(chunks)
            for future in client.scatter(rasters)
        ]

    return [(name, xr.DataArray(array, dims=("y", "x"))) for name, array in zip(names, arrays)]


def resolve_outfile(infile: str | Path, outfile: str | Path) -> Path:
    """
    Turn a directory destination into a concrete per-basin output file.

    ``pism-*-postprocess`` takes an output *file*, but pointing it at an
    output *directory* is a natural thing to try — and used to surface only
    at the very end, as a ``PermissionError`` from ``to_netcdf`` after the
    whole reduction had already been computed. A directory is now accepted
    and named following the run scripts' convention: the input's
    ``spatial_`` prefix becomes ``basin_``.

    Parameters
    ----------
    infile : str or pathlib.Path
        The spatial file being reduced; supplies the name.
    outfile : str or pathlib.Path
        Destination file, or a directory to put the file in.

    Returns
    -------
    pathlib.Path
        Path of the file to write. Its parent directory is created.

    Raises
    ------
    PermissionError
        If the destination directory cannot be written to. Raised up front,
        rather than after the reduction has run.
    """
    # Check the raw argument: Path() strips a trailing separator, which is
    # how a shell tab-completes a directory that does not exist yet.
    looks_like_dir = os.fspath(outfile).endswith((os.sep, "/"))
    outfile = Path(outfile)
    if outfile.is_dir() or looks_like_dir:
        name = Path(infile).name
        stem = name[len("spatial_") :] if name.startswith("spatial_") else name
        outfile = outfile / f"basin_{stem}"

    outfile.parent.mkdir(parents=True, exist_ok=True)
    if not os.access(outfile.parent, os.W_OK):
        raise PermissionError(f"cannot write to {outfile.parent}: check permissions before re-running")
    return outfile


def process_file(
    infile: str | Path,
    outfile: str | Path,
    outlinefile: str | Path,
    client: Client,
    column: str = "SUBREGION1",
    crs: str = "EPSG:3413",
    total_name: str | None = "GIS",
):
    """
    Reduce a NetCDF dataset to per-basin scalar sums and write them to ``outfile``.

    The basin outlines are rasterized onto the dataset grid once, then every
    basin sum is assembled into a single Dask graph and evaluated in one pass:
    each chunk of ``infile`` is read and decompressed exactly once and feeds all
    the per-basin reductions. Peak memory is a chunk, not the dataset, so this
    does not depend on the cluster being able to hold the file.

    No spatial (clipped) file is written.

    Parameters
    ----------
    infile : str or Path
        Path to the NetCDF file to reduce. Must contain x/y spatial dimensions.
    outfile : str or Path
        Path to the output netCDF, or a directory to write it into. A
        directory is named following the run scripts' convention: the
        input's ``spatial_`` prefix becomes ``basin_``.
    outlinefile : str or Path
        Path to the BASIN glacier outline file (e.g., GeoPackage or shapefile) that defines
        the basins to reduce over.
    client : dask.Client
        Dask client.
    column : str, default "SUBREGION1"
        Name of the column in ``basin_file`` used to identify basins.
    crs : str, default "EPSG:3413"
        CRS code applied to the input dataset before rasterizing the outlines.
    total_name : str or None, default "GIS"
        Name of an extra whole-domain basin appended as the sum over the basins
        in ``outlinefile``. Skipped when the outlines already contain a basin of
        this name (older files carry their own ``GIS`` polygon), or when
        ``None``. Note that this total covers only what the outlines cover: with
        ``mouginot_basins_w_shelves.gpkg`` about 0.2% of icy cells — peripheral
        glaciers and ice caps — fall outside every basin and are not counted.
    """

    infile_name = Path(infile).name
    outfile = resolve_outfile(infile, outfile)
    basin = gpd.read_file(outlinefile)

    start = time.time()
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=False)

    ds = xr.open_dataset(
        infile,
        decode_timedelta=False,
        decode_times=False,
        chunks="auto",
        engine="h5netcdf",
    )

    # Bounds/mapping vars (``x_bnds``/``y_bnds``/``mapping``) carry only one of
    # the x/y dims; keeping them would inject a dangling x/y dimension back into
    # the per-basin scalar output on merge. Drop them up front.
    #
    # ``spatial_ref`` is dropped for a different reason: it is the CF grid-mapping
    # variable rioxarray writes, so any file that has round-tripped through
    # rioxarray carries one as a scalar data variable. ``rio.write_crs`` below
    # recreates it as a *coordinate*, and merging a dataset that has it as a
    # coordinate with one that has it as a data variable fails with "unable to
    # determine if these variables should be coordinates or not".
    ds = ds.drop_vars(
        ["x_bnds", "x_bounds", "y_bnds", "y_bounds", "mapping", "spatial_ref", "crs", "polar_stereographic"],
        errors="ignore",
    )

    # Read before the ``pism_config`` variable is split off below.
    ice_free_thickness = float(pism_config_value(ds, "output.ice_free_thickness_standard", 10.0))

    # Separate variables that lack spatial (x, y) dimensions, as they cannot be
    # reduced over x/y
    non_spatial_vars = [var for var in ds.data_vars if "x" not in ds[var].dims or "y" not in ds[var].dims]
    ds_non_spatial = ds[non_spatial_vars]
    ds = ds.drop_vars(non_spatial_vars).rio.write_crs(crs).rio.set_spatial_dims(x_dim="x", y_dim="y")

    # ``ice_mass_glacierized`` needs both ice_mass and thk in the spatial file;
    # some configs write a reduced var set, so only compute it when both are
    # present. Masking is elementwise, so deriving it before the basin masks are
    # applied gives the same per-basin sums as deriving it after.
    #
    # The threshold is PISM's own reporting standard, taken from the run's
    # configuration rather than restated here, so a config that overrides it
    # stays consistent between the simulation and this diagnostic.
    if "ice_mass" in ds.data_vars and "thk" in ds.data_vars:
        logger.info("Ice-free thickness standard: %.4g m", ice_free_thickness)
        ds["ice_mass_glacierized"] = ds["ice_mass"].where(ds["thk"] > ice_free_thickness)

    if "grounding_line_flux" in ds.data_vars:
        ds["grounding_line_flux_nonneg"] = ds["grounding_line_flux"].where(ds["grounding_line_flux"] < 0)

    # ``where`` promotes integer variables to float so it can write NaN outside
    # the basin. Remember them and restore the dtype after summing, so the output
    # schema does not depend on how the masking is done. (Sums of small integers
    # over a Greenland grid stay far inside float64's exact-integer range.)
    integer_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.integer)]

    masks = basin_masks(ds, basin, column=column, client=client)
    logger.info("Reducing over %d basins in a single pass", len(masks))

    # One graph for every basin. The per-basin branches share the same source
    # chunk tasks, so Dask reads each chunk once and fans it out to all of them.
    lazy = xr.concat(
        [ds.where(mask).sum(dim=["y", "x"]).expand_dims({"basin": [name]}) for name, mask in masks],
        dim="basin",
    )

    future = client.compute(lazy)
    progress(future)
    scalar = future.result()

    for var in integer_vars:
        scalar[var] = scalar[var].round().astype(np.int64)

    if total_name is not None and total_name not in set(scalar["basin"].values):
        total = scalar.sum(dim="basin").expand_dims({"basin": [total_name]})
        scalar = xr.concat([scalar, total], dim="basin")
        logger.info("Added %s as the sum over %d basins", total_name, len(masks))

    logger.info("Writing %s", outfile)
    # Keep non-spatial vars (e.g. pism_config)
    extra_vars = [v for v in ds_non_spatial.data_vars if "time" not in ds_non_spatial[v].dims]
    if extra_vars:
        scalar = xr.merge([scalar, ds_non_spatial[extra_vars].compute()])
    # Put time first and swap the string basin labels for an integer index, so
    # CDO can open the result at all. See ``make_cdo_readable``.
    scalar = make_cdo_readable(scalar, "basin")
    comp = {"zlib": True, "complevel": 2}
    encoding_scalar = {var: comp for var in scalar.data_vars}
    scalar.to_netcdf(outfile, encoding=encoding_scalar)

    end = time.time()
    time_elapsed = end - start
    logger.info("Time elapsed for %s: %.0fs", infile_name, time_elapsed)


def postprocess_glacier(
    infile: str | Path,
    outfile: str | Path,
    outlinefile: str | Path,
    n_workers: int = 4,
    local_directory: str | Path | None = None,
):
    """
    Postprocess ISMIP7 Greenland output by clipping to basin geometries.

    Opens ``infile`` and clips it to the basin outline using a Dask client,
    writing per-basin scalar sums to ``outfile``.

    Parameters
    ----------
    infile : str or Path
        Path to the NetCDF file to be clipped. Must contain x/y spatial dimensions.
    outfile : str or Path
        Path to the output netCDF, or a directory to write it into. A
        directory is named following the run scripts' convention: the
        input's ``spatial_`` prefix becomes ``basin_``.
    outlinefile : str or Path
        Path to the BASIN glacier outline file (e.g., GeoPackage or shapefile) that defines
        the geometry to clip the dataset to.
    n_workers : int, optional
        Number of Dask workers, by default 4.
    local_directory : str or Path or None, optional
        Directory where Dask workers write scratch data. On network file
        systems (e.g. Lustre on Chinook) the default scratch location is slow
        and Dask warns about it; point this at node-local disk instead. When
        ``None`` (default), fall back to :func:`tempfile.gettempdir`, which
        honours ``$TMPDIR`` (SLURM sets it to node-local storage).
    """

    start = time.time()

    # Each worker costs several file descriptors; make sure the soft limit is
    # high enough that the nannies can actually spawn.
    _raise_fd_limit()

    # Keep Dask worker scratch off the (slow, networked) Lustre home/cwd.
    scratch_dir = str(local_directory) if local_directory is not None else tempfile.gettempdir()
    os.makedirs(scratch_dir, exist_ok=True)
    logger.info("Dask worker scratch directory: %s", scratch_dir)

    client = Client(n_workers=n_workers, threads_per_worker=1, local_directory=scratch_dir)
    logger.info("Dask dashboard: %s", client.dashboard_link)

    process_file(infile, outfile, outlinefile, client)

    client.close()

    end = time.time()
    time_elapsed = end - start
    logger.info("Time elapsed %.0fs", time_elapsed)


def main():
    """
    Run main script.
    """

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Postprocess ISMIP7 Greenland."
    parser.add_argument(
        "--ntasks",
        help="Sets number of tasks.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--local-directory",
        help="Directory for Dask worker scratch data. On network file systems "
        "(e.g. Lustre) point this at node-local disk to avoid slow scratch I/O. "
        "Defaults to $TMPDIR (or the system temp dir).",
        type=str,
        default=None,
    )
    parser.add_argument(
        "INFILE",
        help="input file.",
        nargs=1,
    )
    parser.add_argument(
        "OUTFILE",
        help="output file, or a directory to write basin_<input>.nc into.",
        nargs=1,
    )
    parser.add_argument(
        "OUTLINEFILE",
        help="basin outline file (GeoPackage/shapefile).",
        nargs=1,
    )

    options, unknown = parser.parse_known_args()
    infile = options.INFILE[0]
    outfile = options.OUTFILE[0]
    outlinefile = options.OUTLINEFILE[0]
    ntasks = options.ntasks
    local_directory = options.local_directory

    # Resolve first, so a directory destination puts the log next to the
    # output file rather than one level up.
    outfile = resolve_outfile(infile, outfile)
    setup_logging(outfile.resolve().parent / "postprocess.log")

    postprocess_glacier(infile, outfile, outlinefile, n_workers=ntasks, local_directory=local_directory)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
