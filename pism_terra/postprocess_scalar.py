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
Per-region scalar post-processing.

Reduces a PISM spatial output file to per-region sums over ``x``/``y`` — one
row per outline in a GeoPackage/shapefile. The same code serves Greenland
drainage basins and RGI glaciers: the outlines are rasterized onto the model
grid once, and every region's reduction goes into a single Dask graph, so each
chunk of the input is read and decompressed exactly once and peak memory is a
chunk rather than the whole dataset.

See :mod:`pism_terra.postprocess_spatial` for the variant that keeps the
spatial dimensions instead of summing over them.
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
from pism_terra.workflow import drop_grid_mapping, make_cdo_readable, pism_config_value

xr.set_options(keep_attrs=True)
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

logger = logging.getLogger(__name__)

# Outline column holding the region name, tried in this order when the caller
# does not name one: the packaged Mouginot basins carry ``glacier_id``, RGI
# files ``rgi_id``, and older Greenland outlines only ``SUBREGION1``.
DEFAULT_COLUMNS = ("glacier_id", "rgi_id", "SUBREGION1")


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


def dataset_crs(ds: xr.Dataset, crs: str | None = None) -> str:
    """
    Determine the CRS of a PISM output file.

    PISM writes a CF grid-mapping variable (``mapping``, or ``spatial_ref``
    once a file has round-tripped through rioxarray) carrying ``crs_wkt``,
    so the projection can be read from the file itself. That matters because
    the projection is campaign-specific — polar stereographic for Greenland,
    a UTM zone for an Alaskan glacier — and a hard-coded default would
    silently misplace one of them.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset as opened from disk, before the grid-mapping variable is
        dropped.
    crs : str or None, optional
        Explicit override (e.g. ``"EPSG:3413"``). When given it wins, which
        is the escape hatch for files that carry no usable grid mapping.

    Returns
    -------
    str
        CRS as WKT or as the given override string.

    Raises
    ------
    ValueError
        If no override was given and the file has no grid-mapping variable
        with ``crs_wkt``.
    """
    if crs is not None:
        return crs

    grid_mapping = ds.rio.grid_mapping
    if grid_mapping in ds.variables:
        crs_wkt = ds[grid_mapping].attrs.get("crs_wkt")
        if crs_wkt:
            logger.info("Using CRS from '%s'", grid_mapping)
            return str(crs_wkt)

    raise ValueError(
        "input file carries no grid-mapping variable with 'crs_wkt'; pass --crs explicitly (e.g. --crs EPSG:3413)"
    )


def resolve_column(outline: gpd.GeoDataFrame, column: str | None = None) -> str:
    """
    Pick the outline column that holds the region names.

    Outline files disagree on what the name column is called: the packaged
    Mouginot basins use ``glacier_id``, RGI files use ``rgi_id``, and older
    Greenland files only have ``SUBREGION1``. Rather than force every file to
    be re-saved, try them in order of preference.

    Parameters
    ----------
    outline : geopandas.GeoDataFrame
        The outlines being reduced over.
    column : str or None, optional
        Explicit column name. When given it must exist — no fallback, so a
        typo is an error rather than a silently different labelling.

    Returns
    -------
    str
        Name of the column to label regions with.

    Raises
    ------
    ValueError
        If ``column`` is given but absent, or if none of
        :data:`DEFAULT_COLUMNS` is present.
    """
    if column is not None:
        if column not in outline.columns:
            raise ValueError(f"outline file has no column {column!r}; available: {list(outline.columns)}")
        return column

    for candidate in DEFAULT_COLUMNS:
        if candidate in outline.columns:
            logger.info("Labelling regions by '%s'", candidate)
            return candidate

    raise ValueError(
        f"outline file has none of {list(DEFAULT_COLUMNS)}; "
        f"available: {list(outline.columns)}. Pass --column to choose one."
    )


def basin_masks(
    ds: xr.Dataset,
    basin: gpd.GeoDataFrame,
    column: str | None = None,
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
    column : str or None, optional
        Column holding the region name. ``None`` (default) resolves it with
        :func:`resolve_column`.
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
    column = resolve_column(basin, column)

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
    column: str | None = None,
    crs: str | None = None,
    dim_name: str = "glacier_id",
    total_name: str | None = None,
    all_touched: bool = False,
):
    """
    Reduce a NetCDF dataset to per-region scalar sums and write them to ``outfile``.

    The outlines are rasterized onto the dataset grid once, then every region's
    sum is assembled into a single Dask graph and evaluated in one pass: each
    chunk of ``infile`` is read and decompressed exactly once and feeds all the
    per-region reductions. Peak memory is a chunk, not the dataset, so this does
    not depend on the cluster being able to hold the file.

    No spatial (clipped) file is written; see
    :mod:`pism_terra.postprocess_spatial` for that.

    Parameters
    ----------
    infile : str or Path
        Path to the NetCDF file to reduce. Must contain x/y spatial dimensions.
    outfile : str or Path
        Path to the output netCDF, or a directory to write it into. A
        directory is named following the run scripts' convention: the
        input's ``spatial_`` prefix becomes ``basin_``.
    outlinefile : str or Path
        Outline file (GeoPackage/shapefile) defining the regions to reduce
        over — Greenland drainage basins, RGI glaciers or complexes. It is
        reprojected to the dataset's CRS, so it may be stored in any.
    client : dask.Client
        Dask client.
    column : str or None, optional
        Name of the column in ``outlinefile`` used to label each region.
        ``None`` (default) picks the first of :data:`DEFAULT_COLUMNS` that
        the file has.
    crs : str or None, optional
        CRS applied to the input before rasterizing. ``None`` (default) reads
        it from the file; see :func:`dataset_crs`.
    dim_name : str, default "glacier_id"
        Name of the region dimension in the output. Greenland campaigns pass
        ``"basin"`` to stay compatible with existing analysis code.
    total_name : str or None, optional
        Name of an extra whole-domain region appended as the sum over all the
        outlines. ``None`` (default) writes no total, which is what per-glacier
        runs want. Skipped when the outlines already contain a region of this
        name (older Greenland files carry their own ``GIS`` polygon). Note that
        such a total covers only what the outlines cover: with
        ``mouginot_basins_w_shelves.gpkg`` about 0.2% of icy cells — peripheral
        glaciers and ice caps — fall outside every basin and are not counted.
    all_touched : bool, default False
        Count a cell as part of a region when the outline touches it at all,
        rather than only when the cell center falls inside. Useful for
        glaciers small relative to the grid spacing.
    """

    infile_name = Path(infile).name
    outfile = resolve_outfile(infile, outfile)
    outline = gpd.read_file(outlinefile)

    start = time.time()
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=False)

    ds = xr.open_dataset(
        infile,
        decode_timedelta=False,
        decode_times=False,
        chunks="auto",
        engine="h5netcdf",
    )

    # Read the projection off the grid-mapping variable before it is dropped
    # below, then bring the outlines onto it. The outlines are commonly stored
    # in EPSG:4326 (the RGI ones are) while the model grid is projected, so
    # rasterizing without this would place every region outside the domain.
    dst_crs = dataset_crs(ds, crs)
    outline = outline.to_crs(dst_crs)

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
    ds = ds.drop_vars(non_spatial_vars).rio.write_crs(dst_crs).rio.set_spatial_dims(x_dim="x", y_dim="y")

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

    masks = basin_masks(ds, outline, column=column, all_touched=all_touched, client=client)
    logger.info("Reducing over %d regions in a single pass", len(masks))

    # One graph for every region. The per-region branches share the same source
    # chunk tasks, so Dask reads each chunk once and fans it out to all of them.
    lazy = xr.concat(
        [ds.where(mask).sum(dim=["y", "x"]).expand_dims({dim_name: [name]}) for name, mask in masks],
        dim=dim_name,
    )

    future = client.compute(lazy)
    progress(future)
    scalar = future.result()

    for var in integer_vars:
        scalar[var] = scalar[var].round().astype(np.int64)

    # Outline area, in the same row order as the reduction. This is the
    # polygon's own area, not the area of the cells that were summed, so it is
    # independent of the grid resolution.
    scalar["area"] = xr.DataArray(
        np.asarray([geom.area for geom in outline.geometry], dtype="float64"),
        dims=(dim_name,),
        coords={dim_name: scalar[dim_name]},
        attrs={"units": "m^2", "long_name": "outline area"},
    )

    if total_name is not None and total_name not in set(scalar[dim_name].values):
        total = scalar.sum(dim=dim_name).expand_dims({dim_name: [total_name]})
        scalar = xr.concat([scalar, total], dim=dim_name)
        logger.info("Added %s as the sum over %d regions", total_name, len(masks))

    logger.info("Writing %s", outfile)
    # Keep non-spatial vars (e.g. pism_config)
    extra_vars = [v for v in ds_non_spatial.data_vars if "time" not in ds_non_spatial[v].dims]
    if extra_vars:
        scalar = xr.merge([scalar, ds_non_spatial[extra_vars].compute()])
    # The grid mapping describes a grid this output no longer has.
    scalar = drop_grid_mapping(scalar)
    # Put time first and swap the string region labels for an integer index, so
    # CDO can open the result at all. See ``make_cdo_readable``.
    scalar = make_cdo_readable(scalar, dim_name)
    comp = {"zlib": True, "complevel": 2}
    encoding_scalar = {var: comp for var in scalar.data_vars}
    scalar.to_netcdf(outfile, encoding=encoding_scalar)

    end = time.time()
    time_elapsed = end - start
    logger.info("Time elapsed for %s: %.0fs", infile_name, time_elapsed)


def postprocess_scalar(
    infile: str | Path,
    outfile: str | Path,
    outlinefile: str | Path,
    n_workers: int = 4,
    local_directory: str | Path | None = None,
    **kwargs,
):
    """
    Reduce a PISM spatial file to per-region sums, managing the Dask cluster.

    Thin wrapper around :func:`process_file`: raises the open-file limit,
    starts a local Dask cluster with sensible scratch, runs the reduction and
    shuts the cluster down again.

    Parameters
    ----------
    infile : str or Path
        Path to the NetCDF file to be reduced. Must contain x/y spatial dimensions.
    outfile : str or Path
        Path to the output netCDF, or a directory to write it into. A
        directory is named following the run scripts' convention: the
        input's ``spatial_`` prefix becomes ``basin_``.
    outlinefile : str or Path
        Outline file (GeoPackage/shapefile) defining the regions to reduce over.
    n_workers : int, optional
        Number of Dask workers, by default 4.
    local_directory : str or Path or None, optional
        Directory where Dask workers write scratch data. On network file
        systems (e.g. Lustre on Chinook) the default scratch location is slow
        and Dask warns about it; point this at node-local disk instead. When
        ``None`` (default), fall back to :func:`tempfile.gettempdir`, which
        honours ``$TMPDIR`` (SLURM sets it to node-local storage).
    **kwargs
        Forwarded to :func:`process_file` (``column``, ``crs``, ``dim_name``,
        ``total_name``, ``all_touched``).
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

    try:
        process_file(infile, outfile, outlinefile, client, **kwargs)
    finally:
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
    parser.description = "Reduce a PISM spatial file to per-region scalar sums."
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
        "--column",
        help="Column in OUTLINEFILE holding the region name. " f"Tried in order {list(DEFAULT_COLUMNS)} when unset.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--crs",
        help="CRS of the input grid. Read from the file's grid mapping when unset.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dim-name",
        help="Name of the region dimension in the output.",
        type=str,
        default="glacier_id",
    )
    parser.add_argument(
        "--total-name",
        help="Append a whole-domain region summing all outlines under this name. Off when unset.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--all-touched",
        help="Count every cell the outline touches, not only those whose center it contains.",
        action="store_true",
        default=False,
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
        help="region outline file (GeoPackage/shapefile).",
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

    postprocess_scalar(
        infile,
        outfile,
        outlinefile,
        n_workers=ntasks,
        local_directory=local_directory,
        column=options.column,
        crs=options.crs,
        dim_name=options.dim_name,
        total_name=options.total_name,
        all_touched=options.all_touched,
    )


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
