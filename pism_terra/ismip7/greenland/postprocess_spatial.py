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

# pylint: disable=too-many-positional-arguments,too-many-locals,too-many-arguments

"""
Per-basin spatial postprocessing.

Unlike :mod:`pism_terra.ismip7.greenland.postprocess`, which reduces PISM
spatial output to per-basin scalar sums, this module extracts the masked
spatial fields themselves — one NetCDF file per basin, no summing. Output can
approach the input in size (>50 GB in production), so every step here is
lazy: the input is opened chunked, masking and cropping stay Dask-backed, and
the write streams chunk by chunk. Nothing ever materializes a full basin.

Three write strategies are provided behind ``--method``; they produce
identical files and differ only in how the stream reaches disk (see
``benchmarks/bench_postprocess_spatial.py`` for the comparison that picked
the default):

- ``netcdf``: every basin's ``to_netcdf(..., compute=False)`` is collected
  into one graph and computed at once — basins are separate files, so their
  writes proceed in parallel, and shared input chunks are read once.
- ``zarr``: each basin goes to a scratch Zarr store first (lock-free parallel
  write), then streams from Zarr to NetCDF.
- ``shards``: each basin is written serially in ``--time-batch`` slabs to
  scratch shards, which are then concatenated and streamed to the final file.
  Slowest, but the most conservative in memory and file-handle use.

Integer variables (PISM's ``mask`` and other flags) are cropped but not
NaN-masked: ``where`` would promote them to float, doubling their size and
destroying flag semantics. Cells outside the basin polygon keep their
original values inside the bounding box; use the float variables (which are
NaN outside the polygon) or re-rasterize the outline to distinguish
inside from outside.
"""

import logging
import shutil
import tempfile
import time
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import reduce
from operator import or_
from pathlib import Path

import dask
import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401  pylint: disable=unused-import
import xarray as xr
from dask.distributed import Client, progress

from pism_terra.ismip7.greenland.postprocess import _raise_fd_limit, basin_masks
from pism_terra.log import setup_logging

xr.set_options(keep_attrs=True)
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)

logger = logging.getLogger(__name__)

# Set from benchmarks/bench_postprocess_spatial.py at 2 GB (see benchmarks/README.md).
DEFAULT_METHOD = "netcdf"

#: Bounds and grid-mapping variables dropped up front — same list and reasons
#: as the scalar module (dangling dims on merge; ``spatial_ref``
#: coordinate/data-variable conflict).
DROP_VARS = ["x_bnds", "x_bounds", "y_bnds", "y_bounds", "mapping", "spatial_ref", "crs", "polar_stereographic"]


def _bbox_slices(mask: xr.DataArray) -> tuple[slice, slice]:
    """
    Index slices spanning the ``True`` cells of a boolean (y, x) mask.

    Only two tiny boolean vectors (``mask.any`` per axis) are computed, so
    this is cheap even for a Dask-backed mask on a fine grid. The resulting
    window is the extent of cells actually inside the polygon under the
    rasterization rule, which can be up to one cell tighter per edge than
    ``rio.clip(..., drop=True)`` — that crops to the polygon *bounds* and can
    keep all-NaN border rows/columns.

    Parameters
    ----------
    mask : xarray.DataArray
        Boolean mask with dims ``(y, x)``.

    Returns
    -------
    tuple of slice
        ``(y_slice, x_slice)`` covering every ``True`` cell.

    Raises
    ------
    ValueError
        If the mask selects no cells.
    """
    y_any, x_any = dask.compute(mask.any("x"), mask.any("y"))
    y_idx = np.flatnonzero(np.asarray(y_any))
    x_idx = np.flatnonzero(np.asarray(x_any))
    if y_idx.size == 0 or x_idx.size == 0:
        raise ValueError("mask selects no cells")
    return slice(int(y_idx[0]), int(y_idx[-1]) + 1), slice(int(x_idx[0]), int(x_idx[-1]) + 1)


def extract_basin(ds: xr.Dataset, mask: xr.DataArray, crop: bool = True) -> xr.Dataset:
    """
    Lazy per-basin view of ``ds``: optionally bbox-cropped, NaN-masked floats.

    Float variables are masked with ``where(mask)`` (NaN outside the
    polygon); integer variables are cropped but keep their dtype and values
    (see module docstring). Variables with extra dimensions (e.g. 3D
    ``enthalpy`` with a ``z`` dim) work unchanged — the (y, x) mask
    broadcasts blockwise.

    Parameters
    ----------
    ds : xarray.Dataset
        Spatial dataset; every data variable must carry ``y`` and ``x``.
    mask : xarray.DataArray
        Boolean basin mask with dims ``(y, x)`` on the same grid.
    crop : bool, default True
        Crop to the mask's bounding box (``rio.clip(drop=True)`` semantics up
        to all-NaN border rows/columns, see :func:`_bbox_slices`). ``False``
        keeps the full grid (``drop=False`` semantics).

    Returns
    -------
    xarray.Dataset
        Lazy masked (and optionally cropped) dataset, variable order
        preserved.
    """
    if crop:
        y_slice, x_slice = _bbox_slices(mask)
        ds = ds.isel(y=y_slice, x=x_slice)
        mask = mask.isel(y=y_slice, x=x_slice)

    integer_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.integer)]
    out = ds.drop_vars(integer_vars).where(mask)
    for var in integer_vars:
        out[var] = ds[var]
    # ``where`` + re-adding moved the integer vars to the back; restore order.
    return out[list(ds.data_vars)]


def _encoding(ds: xr.Dataset) -> dict[str, dict]:
    """
    Fresh write encoding: compressed data variables, no coord fill values.

    Also clears every variable's stale on-disk encoding **in place**: the
    input's encoding carries ``chunksizes`` from the source file, which after
    a bbox crop can exceed the (smaller) dimension sizes and make h5netcdf
    refuse the write. The one encoding entry worth keeping is
    ``grid_mapping`` — rioxarray moves that CF attribute into the encoding on
    ``write_crs``, and dropping it would leave the output's grid-mapping
    variable orphaned (``rio.crs`` = None downstream) — so it is promoted
    back to a variable attribute.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset about to be written. Modified in place (encodings cleared).

    Returns
    -------
    dict
        Encoding mapping for :meth:`xarray.Dataset.to_netcdf`.
    """
    comp = {"zlib": True, "complevel": 2, "shuffle": True}
    enc: dict[str, dict] = {}
    for name, var in ds.variables.items():
        grid_mapping = var.encoding.get("grid_mapping")
        var.encoding = {}
        if grid_mapping and grid_mapping in ds.variables:
            var.attrs.setdefault("grid_mapping", grid_mapping)
        if name in ds.data_vars and np.issubdtype(var.dtype, np.number):
            enc[name] = dict(comp)
        elif name in ds.coords:
            enc[name] = {"_FillValue": None}
    return enc


def _unlimited(ds: xr.Dataset) -> list[str]:
    """
    Unlimited-dimension list for a write: ``time`` when present.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset about to be written.

    Returns
    -------
    list of str
        ``["time"]`` or ``[]``.
    """
    return ["time"] if "time" in ds.dims else []


def _zarr_safe_chunks(sub: xr.Dataset) -> xr.Dataset:
    """
    Rechunk dimensions whose Dask chunking Zarr would reject.

    Zarr requires every chunk of an array to be the same size, except the
    last, which may only be smaller. A bbox ``isel`` routinely violates
    that — slicing chunk boundaries can leave a *small first* chunk, e.g.
    ``(23, 222)`` along x — and ``to_zarr`` then refuses the write. Only the
    offending dimensions are rechunked (to their largest chunk size), so a
    dataset that is already regular passes through untouched.

    Parameters
    ----------
    sub : xarray.Dataset
        Dataset about to go to a Zarr store.

    Returns
    -------
    xarray.Dataset
        The same dataset, rechunked where necessary.
    """
    rechunk: dict[str, int] = {}
    for var in sub.data_vars.values():
        if var.chunks is None:
            continue
        for dim, sizes in zip(var.dims, var.chunks):
            irregular = len(set(sizes[:-1])) > 1 or sizes[-1] > sizes[0]
            if len(sizes) > 1 and irregular:
                rechunk[str(dim)] = max(*sizes, rechunk.get(str(dim), 0))
    if rechunk:
        logger.info("Rechunking %s for the Zarr store", rechunk)
        sub = sub.chunk(rechunk)
    return sub


def _write_zarr(sub: xr.Dataset, path: Path, scratch: Path) -> None:
    """
    Write ``sub`` via a scratch Zarr store, then stream Zarr → NetCDF.

    Zarr writes are lock-free and fully parallel, which is the whole point of
    the detour; the cost is one extra (uncompressed-ish) copy on scratch
    disk. Non-numeric variables (e.g. the ``pism_config`` char blob) skip the
    store — Zarr's string handling is version-dependent — and are merged back
    in at the final write; they are tiny.

    Parameters
    ----------
    sub : xarray.Dataset
        Lazy per-basin dataset.
    path : pathlib.Path
        Final NetCDF path.
    scratch : pathlib.Path
        Directory for the temporary Zarr store.
    """
    non_numeric = [v for v in sub.data_vars if not np.issubdtype(sub[v].dtype, np.number)]
    store = scratch / (path.stem + ".zarr")
    shutil.rmtree(store, ignore_errors=True)
    try:
        sub.drop_vars(non_numeric).pipe(_zarr_safe_chunks).to_zarr(store, mode="w", consolidated=True)
        staged = xr.open_zarr(store)
        # Non-indexed coordinates (``spatial_ref``) can come back as data
        # variables; restore their coordinate status so the three write
        # methods produce identical files.
        staged = staged.set_coords([c for c in sub.coords if c in staged.data_vars])
        for var in non_numeric:
            staged[var] = sub[var].compute()
        # The conversion must not run on the live distributed client: with an
        # active client xarray routes the netCDF write lock through
        # ``distributed.Lock``, which can deadlock an in-process cluster (see
        # ``_write_shards``). The store is on local scratch, so a synchronous
        # chunk-by-chunk copy is cheap and needs no cluster.
        with dask.config.set(scheduler="synchronous"):
            staged.to_netcdf(path, engine="h5netcdf", encoding=_encoding(staged), unlimited_dims=_unlimited(staged))
        staged.close()
    finally:
        shutil.rmtree(store, ignore_errors=True)


def _write_shards(sub: xr.Dataset, path: Path, scratch: Path, time_batch: int) -> None:
    """
    Write ``sub`` serially in time slabs, then concatenate the shards.

    Each slab of ``time_batch`` steps is computed (bbox × slab memory) and
    written uncompressed to scratch; the shards are then reopened lazily,
    concatenated along ``time``, and streamed compressed to ``path``. The
    slowest method, but memory and open-file use are bounded and no
    distributed write locks are involved.

    Parameters
    ----------
    sub : xarray.Dataset
        Lazy per-basin dataset.
    path : pathlib.Path
        Final NetCDF path.
    scratch : pathlib.Path
        Directory for the temporary shard files.
    time_batch : int
        Time steps per shard.
    """
    if "time" not in sub.dims:
        sub.to_netcdf(path, engine="h5netcdf", encoding=_encoding(sub), unlimited_dims=_unlimited(sub))
        return

    shard_dir = scratch / (path.stem + "_shards")
    shutil.rmtree(shard_dir, ignore_errors=True)
    shard_dir.mkdir(parents=True)
    try:
        timeless = [v for v in sub.data_vars if "time" not in sub[v].dims]
        shard_paths = []
        for idx, start in enumerate(range(0, sub.sizes["time"], time_batch)):
            batch = sub.drop_vars(timeless).isel(time=slice(start, start + time_batch)).compute()
            shard = shard_dir / f"batch_{idx:04d}.nc"
            for var in batch.variables.values():
                var.encoding = {}
            batch.to_netcdf(shard, engine="h5netcdf")
            shard_paths.append(shard)
        shards = [xr.open_dataset(p, chunks={}, decode_times=False, decode_timedelta=False) for p in shard_paths]
        stitched = xr.concat(shards, dim="time")
        stitched = stitched.set_coords([c for c in sub.coords if c in stitched.data_vars])
        for var in timeless:
            stitched[var] = sub[var].compute()
        stitched = stitched[list(sub.data_vars)]
        stitched.attrs = sub.attrs
        # Reading the shards (h5netcdf) while writing the final file
        # (h5netcdf) in one compute must not run on the live distributed
        # client: xarray then takes the write lock via ``distributed.Lock``,
        # which intermittently deadlocks an in-process cluster (observed as a
        # hung test with the shard files still open). The synchronous
        # scheduler serializes read and write in this thread with a plain
        # lock; memory stays one chunk.
        with dask.config.set(scheduler="synchronous"):
            stitched.to_netcdf(path, engine="h5netcdf", encoding=_encoding(stitched), unlimited_dims=["time"])
        for shard in shards:
            shard.close()
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)


def process_file_spatial(
    infile: str | Path,
    outdir: str | Path,
    outlinefile: str | Path,
    client: Client,
    column: str = "SUBREGION1",
    crs: str = "EPSG:3413",
    crop: bool = True,
    variables: list[str] | None = None,
    method: str = DEFAULT_METHOD,
    total: bool = False,
    total_name: str = "GIS",
    time_batch: int = 12,
    scratch: str | Path | None = None,
) -> list[Path]:
    """
    Extract per-basin spatial fields from ``infile`` into one file per basin.

    The counterpart of
    :func:`pism_terra.ismip7.greenland.postprocess.process_file` that keeps
    the spatial dimensions instead of summing over them. Every step is lazy;
    peak memory is bounded by (workers × chunk) for the ``netcdf``/``zarr``
    methods and by one time slab for ``shards``, independent of file size.

    Parameters
    ----------
    infile : str or Path
        Path to the NetCDF file to extract from. Must contain x/y spatial
        dimensions.
    outdir : str or Path
        Directory for the per-basin files, created if needed. Files are
        named ``spatial_<BASIN>_<input-stem>.nc`` (a leading ``spatial_`` on
        the input stem is not repeated).
    outlinefile : str or Path
        Basin outline file (GeoPackage/shapefile) defining the basins.
    client : dask.distributed.Client
        Dask client.
    column : str, default "SUBREGION1"
        Column in ``outlinefile`` holding the basin names.
    crs : str, default "EPSG:3413"
        CRS applied to the input before rasterizing the outlines.
    crop : bool, default True
        Crop each basin file to its bounding box. ``False`` keeps every file
        on the full input grid, NaN-masked outside the basin.
    variables : list of str or None, optional
        Spatial variables to keep. ``None`` (default) keeps all of them,
        including 3D fields. Unknown names raise ``ValueError`` listing what
        is available.
    method : {"netcdf", "zarr", "shards"}, optional
        Write strategy (see module docstring). Default is the benchmark
        winner.
    total : bool, default False
        Also write a whole-domain file masked to the union of all basins.
        Off by default — it is nearly a copy of the input.
    total_name : str, default "GIS"
        Basin name used for the union file; skipped when the outlines
        already contain a basin of this name.
    time_batch : int, default 12
        Time steps per shard (``shards`` method only).
    scratch : str or Path or None, optional
        Scratch directory for the ``zarr``/``shards`` intermediates.
        Defaults to :func:`tempfile.gettempdir` (honours ``$TMPDIR``).

    Returns
    -------
    list of pathlib.Path
        The written per-basin files, in outline row order.

    Raises
    ------
    ValueError
        On unknown ``variables`` or ``method``.
    """
    if method not in ("netcdf", "zarr", "shards"):
        raise ValueError(f"unknown method {method!r}; expected 'netcdf', 'zarr', or 'shards'")

    infile = Path(infile)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scratch = Path(scratch) if scratch is not None else Path(tempfile.gettempdir())
    basin = gpd.read_file(outlinefile)

    start = time.time()

    ds = xr.open_dataset(
        infile,
        decode_timedelta=False,
        decode_times=False,
        chunks="auto",
        engine="h5netcdf",
    )
    ds = ds.drop_vars(DROP_VARS, errors="ignore")

    # Split off variables that lack spatial dims; the time-less ones (e.g.
    # ``pism_config``) are carried into every basin file, like the scalar module.
    non_spatial_vars = [var for var in ds.data_vars if "x" not in ds[var].dims or "y" not in ds[var].dims]
    extra = ds[[v for v in non_spatial_vars if "time" not in ds[v].dims]].compute()
    ds = ds.drop_vars(non_spatial_vars)

    if variables is not None:
        unknown = sorted(set(variables) - set(ds.data_vars))
        if unknown:
            raise ValueError(f"unknown variables {unknown}; available: {sorted(ds.data_vars)}")
        ds = ds[variables]

    # Vertical (z-like) dims are short; one chunk each keeps the mask
    # broadcast blockwise. Never rechunk time/y/x — respect the input.
    extra_dims = {str(d): -1 for d in ds.dims if d not in ("time", "y", "x")}
    if extra_dims:
        ds = ds.chunk(extra_dims)

    ds = ds.rio.write_crs(crs).rio.set_spatial_dims(x_dim="x", y_dim="y")

    masks = basin_masks(ds, basin, column=column, client=client)
    if total and total_name not in {name for name, _ in masks}:
        union = reduce(or_, (mask for _, mask in masks))
        masks.append((total_name, union))

    stem = infile.stem
    stem = stem[len("spatial_") :] if stem.startswith("spatial_") else stem

    subs: list[tuple[Path, xr.Dataset]] = []
    for name, mask in masks:
        try:
            # The union file is a whole-domain product; never crop it.
            sub = extract_basin(ds, mask, crop=crop and name != total_name)
        except ValueError:
            logger.warning("Basin %s covers no grid cells; skipped", name)
            continue
        for var in extra.data_vars:
            sub[var] = extra[var]
        sub.attrs["basin"] = str(name)
        sub = sub.rio.write_crs(crs)
        path = outdir / f"spatial_{name}_{stem}.nc"
        path.unlink(missing_ok=True)
        subs.append((path, sub))

    logger.info("Writing %d basin files via method=%s", len(subs), method)
    if method == "netcdf":
        # One graph for all basins: separate target files write in parallel
        # (no lock contention), shared input chunks are read once.
        writes = [
            sub.to_netcdf(
                path, engine="h5netcdf", encoding=_encoding(sub), unlimited_dims=_unlimited(sub), compute=False
            )
            for path, sub in subs
        ]
        futures = client.compute(writes)
        progress(futures)
        client.gather(futures)
    else:
        for path, sub in subs:
            logger.info("Writing %s", path.name)
            if method == "zarr":
                _write_zarr(sub, path, scratch)
            else:
                _write_shards(sub, path, scratch, time_batch)

    ds.close()
    logger.info("Time elapsed for %s: %.0fs", infile.name, time.time() - start)
    return [path for path, _ in subs]


def postprocess_glacier_spatial(
    infile: str | Path,
    outdir: str | Path,
    outlinefile: str | Path,
    n_workers: int = 4,
    local_directory: str | Path | None = None,
    **kwargs,
) -> list[Path]:
    """
    Extract per-basin spatial fields with a locally started Dask client.

    Wrapper around :func:`process_file_spatial` mirroring the scalar
    module's ``postprocess_glacier``: raises the fd limit, keeps Dask worker
    scratch off networked filesystems, and starts/stops the client.

    Parameters
    ----------
    infile : str or Path
        Path to the NetCDF file to extract from.
    outdir : str or Path
        Directory for the per-basin files.
    outlinefile : str or Path
        Basin outline file (GeoPackage/shapefile).
    n_workers : int, optional
        Number of Dask workers, by default 4.
    local_directory : str or Path or None, optional
        Directory for Dask worker scratch data and the zarr/shards
        intermediates. On network file systems point this at node-local
        disk. Defaults to :func:`tempfile.gettempdir` (honours ``$TMPDIR``).
    **kwargs : dict
        Passed through to :func:`process_file_spatial` (``crop``,
        ``variables``, ``method``, ``total``, ``time_batch``, ...).

    Returns
    -------
    list of pathlib.Path
        The written per-basin files.
    """
    start = time.time()
    _raise_fd_limit()

    scratch_dir = Path(local_directory) if local_directory is not None else Path(tempfile.gettempdir())
    scratch_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Dask worker scratch directory: %s", scratch_dir)

    client = Client(n_workers=n_workers, threads_per_worker=1, local_directory=str(scratch_dir))
    logger.info("Dask dashboard: %s", client.dashboard_link)
    try:
        written = process_file_spatial(infile, outdir, outlinefile, client, scratch=scratch_dir, **kwargs)
    finally:
        client.close()

    logger.info("Time elapsed %.0fs", time.time() - start)
    return written


def main():
    """
    Run main script.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Extract per-basin spatial fields (masked, not summed) from ISMIP7 Greenland output."
    parser.add_argument(
        "--ntasks",
        help="Sets number of tasks.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--local-directory",
        help="Directory for Dask worker scratch data and zarr/shards intermediates. "
        "On network file systems (e.g. Lustre) point this at node-local disk. "
        "Defaults to $TMPDIR (or the system temp dir).",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--vars",
        help="Comma-separated spatial variables to keep (default: all, including 3D fields).",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--crop",
        help="'bbox' crops each basin file to its bounding box; 'full' keeps the full grid, NaN-masked.",
        choices=["bbox", "full"],
        default="bbox",
    )
    parser.add_argument(
        "--method",
        help="Write strategy; the default won the benchmark in benchmarks/.",
        choices=["netcdf", "zarr", "shards"],
        default=DEFAULT_METHOD,
    )
    parser.add_argument(
        "--time-batch",
        help="Time steps per shard (shards method only).",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--total",
        help="Also write a whole-domain file masked to the union of all basins.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--column",
        help="Column in OUTLINEFILE holding the basin names.",
        type=str,
        default="SUBREGION1",
    )
    parser.add_argument(
        "INFILE",
        help="input file.",
        nargs=1,
    )
    parser.add_argument(
        "OUTDIR",
        help="output directory for the per-basin files.",
        nargs=1,
    )
    parser.add_argument(
        "OUTLINEFILE",
        help="basin outline file (GeoPackage/shapefile).",
        nargs=1,
    )

    options, _ = parser.parse_known_args()
    outdir = Path(options.OUTDIR[0]).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    setup_logging(outdir / "postprocess_spatial.log")

    postprocess_glacier_spatial(
        options.INFILE[0],
        outdir,
        options.OUTLINEFILE[0],
        n_workers=options.ntasks,
        local_directory=options.local_directory,
        crop=options.crop == "bbox",
        variables=options.vars.split(",") if options.vars else None,
        method=options.method,
        total=options.total,
        time_batch=options.time_batch,
        column=options.column,
    )


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
