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
from pathlib import Path

import cf_xarray
import geopandas as gpd
import rioxarray
import toml
import xarray as xr
from dask.distributed import Client, progress
from pyfiglet import Figlet
from tqdm import tqdm

from pism_terra.log import setup_logging

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


def process_file(
    infile: str | Path,
    outfile: str | Path,
    outlinefile: str | Path,
    client: Client,
    column: str = "SUBREGION1",
    crs: str = "EPSG:3413",
):
    """
    Clip a NetCDF dataset to basin geometries and write per-basin scalar sums.

    Reads ``infile``, clips it to each geometry in ``outlinefile``, field-sums
    the clipped result over x/y per basin, and writes the resulting per-basin
    scalar dataset to ``outfile``. No spatial (clipped) file is written.

    Parameters
    ----------
    infile : str or Path
        Path to the NetCDF file to be clipped. Must contain x/y spatial dimensions.
    outfile : str or Path
        Path to the output netCDF.
    outlinefile : str or Path
        Path to the BASIN glacier outline file (e.g., GeoPackage or shapefile) that defines
        the geometry to clip the dataset to.
    client : dask.Client
        Dask client.
    column : str, default "SUBREGION1"
        Name of the column in ``basin_file`` used to identify basins (e.g.
        ``"GIS"`` is selected for the merged-basin clip).
    crs : str, default "EPSG:3413"
        CRS code applied to the input dataset before clipping.
    """

    infile_name = Path(infile).name
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
    ds = ds.drop_vars(["x_bnds", "x_bounds", "y_bnds", "y_bounds", "mapping"], errors="ignore")

    # Separate variables that lack spatial (x, y) dimensions, as rio.clip cannot handle them
    non_spatial_vars = [var for var in ds.data_vars if "x" not in ds[var].dims or "y" not in ds[var].dims]
    ds_non_spatial = ds[non_spatial_vars]
    ds = ds.drop_vars(non_spatial_vars).rio.write_crs(crs).rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds = client.persist(ds)
    progress(ds)

    dss = []
    for _, row in tqdm(basin.iterrows(), total=len(basin), desc="Clipping basins"):
        ds_clipped = ds.rio.clip([row.geometry], drop=False)
        # ``ice_mass_glacierized`` needs both ice_mass and thk in the spatial
        # file; some configs write a reduced var set, so only compute it when
        # both are present.
        if "ice_mass" in ds_clipped.data_vars and "thk" in ds_clipped.data_vars:
            ds_clipped["ice_mass_glacierized"] = ds_clipped["ice_mass"].where(ds_clipped["thk"] > 10)
        ds_sum = ds_clipped.sum(dim=["y", "x"]).compute()
        dss.append(ds_sum.expand_dims({"basin": [row[column]]}))

    scalar = xr.concat(dss, dim="basin")

    logger.info("Writing %s", outfile)
    # Keep non-spatial vars (e.g. pism_config)
    extra_vars = [v for v in ds_non_spatial.data_vars if "time" not in ds_non_spatial[v].dims]
    if extra_vars:
        scalar = xr.merge([scalar, ds_non_spatial[extra_vars].compute()])
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
        Path to the output netCDF.
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
        help="output file.",
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

    config_path = Path(outfile).resolve().parent
    setup_logging(config_path / "postprocess.log")

    postprocess_glacier(infile, outfile, outlinefile, n_workers=ntasks, local_directory=local_directory)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
