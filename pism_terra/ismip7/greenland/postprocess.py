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

# pylint: disable=unused-import,unused-variable

"""
Postprocessing.
"""

import json
import logging
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

xr.set_options(keep_attrs=True)
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

logger = logging.getLogger(__name__)


def process_file(
    infile: str | Path, basin_file: str | Path, client: Client, column: str = "SUBREGION1", crs: str = "EPSG:3413"
):
    """
    Clip a NetCDF dataset to the glacier geometry defined in an BASIN file.

    This function reads a NetCDF file containing geospatial data and clips it to the
    geometry defined in a glacier outline file (e.g., BASIN shapefile). The clipped dataset
    is saved to a new NetCDF file prefixed with "clipped_".

    Parameters
    ----------
    infile : str or Path
        Path to the NetCDF file to be clipped. Must contain x/y spatial dimensions.
    basin_file : str or Path
        Path to the BASIN glacier outline file (e.g., GeoPackage or shapefile) that defines
        the geometry to clip the dataset to. Must include an `epsg` column to define the CRS.
    client : dask.Client
        Dask client.
    column : str
        Column.
    crs : str
        CRS code.
    """

    infile = Path(infile)
    infile_name = infile.name
    infile_path = infile.parent
    clipped_file = infile_path / Path("clipped_" + infile_name)
    scalar_file = infile_path / Path("fldsum_" + infile_name)

    basin = gpd.read_file(basin_file)

    start = time.time()
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=False)

    ds = xr.open_dataset(
        infile,
        decode_timedelta=False,
        decode_times=False,
        chunks="auto",
        engine="h5netcdf",
    )

    # Separate variables that lack spatial (x, y) dimensions, as rio.clip cannot handle them
    non_spatial_vars = [var for var in ds.data_vars if "x" not in ds[var].dims or "y" not in ds[var].dims]
    ds_non_spatial = ds[non_spatial_vars]
    ds = ds.drop_vars(non_spatial_vars).rio.write_crs(crs).rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds = client.persist(ds)
    progress(ds)

    gis_clipped = ds.rio.clip(basin[basin[column] == "GIS"].geometry, drop=False)
    gis_clipped = xr.merge([gis_clipped, ds_non_spatial])

    logger.info("Writing %s", clipped_file)
    comp = {"zlib": True, "complevel": 2}
    encoding = {var: comp for var in gis_clipped.data_vars}
    write_clipped = gis_clipped.to_netcdf(clipped_file, encoding=encoding, compute=False)
    future_clipped = client.compute(write_clipped)
    progress(future_clipped)

    dss = []
    for _, row in tqdm(basin.iterrows(), total=len(basin), desc="Clipping basins"):
        ds_clipped = ds.rio.clip([row.geometry], drop=False)
        ds_sum = ds_clipped.sum(dim=["y", "x"]).compute()
        dss.append(ds_sum.expand_dims({"basin": [row[column]]}))

    scalar = xr.concat(dss, dim="basin")

    logger.info("Writing %s", scalar_file)
    # Keep non-spatial vars (e.g. pism_config)
    extra_vars = [v for v in ds_non_spatial.data_vars if "time" not in ds_non_spatial[v].dims]
    if extra_vars:
        scalar = xr.merge([scalar, ds_non_spatial[extra_vars].compute()])
    encoding_scalar = {var: comp for var in scalar.data_vars}
    scalar.to_netcdf(scalar_file, encoding=encoding_scalar)

    end = time.time()
    time_elapsed = end - start
    logger.info("Time elapsed for %s: %.0fs", infile_name, time_elapsed)


def postprocess_glacier(config_file: str | Path, n_workers: int = 4):
    """
    Configure and print a PISM model run command for a glacier.

    This function reads glacier metadata from a CSV file and simulation settings
    from a TOML configuration file, then builds and prints a full PISM command-line
    string for executing a model run. It sets up output directories and constructs
    appropriate output filenames.

    Parameters
    ----------
    config_file : str or Path
        Path to a TOML file containing PISM run configuration, including time,
        energy model, stress balance model, and reporting options.
    n_workers : int, optional
        Number of Dask workers, by default 4.
    """

    config_toml = toml.load(config_file)
    config = json.loads(json.dumps(config_toml))

    start = time.time()
    outline_file = config["basin"]["outline"]

    client = Client(n_workers=n_workers, threads_per_worker=1)
    logger.info("Dask dashboard: %s", client.dashboard_link)

    for o in ["spatial"]:
        s_file = Path(config["output"][o])
        process_file(s_file, outline_file, client)

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
    parser.description = "Postprocess KITP Greenland."
    parser.add_argument(
        "--ntasks",
        help="Sets number of tasks.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "RUN_FILE",
        help="CONFIG TOML.",
        nargs=1,
    )

    options, unknown = parser.parse_known_args()
    config_file = options.RUN_FILE[0]
    ntasks = options.ntasks

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.WARNING, format=log_format)
    for handler in logging.root.handlers:
        handler.setLevel(logging.WARNING)
    config_path = Path(config_file).resolve().parent
    file_handler = logging.FileHandler(config_path / "postprocess.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger("pism_terra").setLevel(logging.INFO)
    logging.getLogger("pism_terra").addHandler(file_handler)

    postprocess_glacier(config_file, n_workers=ntasks)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
