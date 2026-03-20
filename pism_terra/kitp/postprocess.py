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

    print(f"Writing {clipped_file}")
    comp = {"zlib": True, "complevel": 2}
    encoding = {var: comp for var in gis_clipped.data_vars}
    write_clipped = gis_clipped.to_netcdf(clipped_file, encoding=encoding, compute=False)
    future_clipped = client.compute(write_clipped)
    progress(future_clipped)

    dss = []
    for _, row in tqdm(basin.iterrows(), total=len(basin), desc="Clipping basins"):
        ds_clipped = ds.rio.clip([row.geometry], drop=False)
        dss.append(ds_clipped.expand_dims({"basin": [row[column]]}))

    clipped = xr.concat(dss, dim="basin")

    print(f"Writing {scalar_file}")
    scalar = clipped.sum(dim=["y", "x"])
    # Keep dimensionless non-spatial vars (e.g. pism_config) but skip those with a time dim to avoid duplicates
    scalar_extras = ds_non_spatial[list(ds_non_spatial.data_vars)]
    if scalar_extras.data_vars:
        scalar = xr.merge([scalar, scalar_extras])
    encoding_scalar = {var: comp for var in scalar.data_vars}
    write_scalar = scalar.to_netcdf(scalar_file, encoding=encoding_scalar, compute=False)
    future_scalar = client.compute(write_scalar)
    progress(future_scalar)

    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed for {infile_name}: {time_elapsed:.0f}s")


def postprocess_glacier(config_file: str | Path):
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
    """

    config_toml = toml.load(config_file)
    config = json.loads(json.dumps(config_toml))

    start = time.time()
    outline_file = config["basin"]["outline"]

    client = Client(n_workers=4, threads_per_worker=1, memory_limit="8GiB")
    print(f"Dask dashboard: {client.dashboard_link}")

    for o in ["spatial"]:
        s_file = Path(config["output"][o])
        process_file(s_file, outline_file, client)

    client.close()

    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed {time_elapsed:.0f}s")


def main():
    """
    Run main script.
    """

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Postprocess KITP Greenland."
    parser.add_argument(
        "RUN_FILE",
        help="CONFIG TOML.",
        nargs=1,
    )

    options, unknown = parser.parse_known_args()
    config_file = options.RUN_FILE[0]

    postprocess_glacier(config_file)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
