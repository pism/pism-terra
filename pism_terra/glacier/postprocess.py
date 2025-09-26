# Copyright (C) 2025 Andy Aschwanden
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
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import cf_xarray
import dask
import geopandas as gpd
import rioxarray
import toml
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress
from pyfiglet import Figlet

xr.set_options(keep_attrs=True)


def process_file(infile: str | Path, rgi_file: str | Path):
    """
    Clip a NetCDF dataset to the glacier geometry defined in an RGI file.

    This function reads a NetCDF file containing geospatial data and clips it to the
    geometry defined in a glacier outline file (e.g., RGI shapefile). The clipped dataset
    is saved to a new NetCDF file prefixed with "clipped_".

    Parameters
    ----------
    infile : str or Path
        Path to the NetCDF file to be clipped. Must contain x/y spatial dimensions.
    rgi_file : str or Path
        Path to the RGI glacier outline file (e.g., GeoPackage or shapefile) that defines
        the geometry to clip the dataset to. Must include an `epsg` column to define the CRS.
    """

    infile = Path(infile)
    infile_name = infile.name
    infile_path = infile.parent
    clipped_file = infile_path / Path("clipped_" + infile_name)
    speed_clipped_file = infile_path / Path("clipped_speed_" + infile_name)
    scalar_file = infile_path / Path("fldsum_" + infile_name)

    rgi = gpd.read_file(rgi_file)
    crs = rgi.iloc[0]["epsg"]
    rgi_projected = rgi.to_crs(crs)
    geometry = rgi_projected.geometry

    start = time.time()

    ds = (
        xr.open_dataset(
            infile,
            decode_times=False,
            decode_timedelta=False,
            chunks="auto",
            engine="h5netcdf",
        )
        .drop_vars("time_bounds", errors="ignore")
        .rio.set_spatial_dims(x_dim="x", y_dim="y")
    )

    ds = ds.rio.write_crs(crs, inplace=False)
    ds_clipped = ds.rio.clip(geometry, drop=False)
    ds_clipped.to_netcdf(clipped_file)
    if "velsurf_mag" in ds.data_vars:
        ds_clipped[["spatial_ref", "pism_config", "velsurf_mag"]].to_netcdf(clipped_file)
    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed for postprocessing: {time_elapsed:.0f}s")
    pism_config = ds["pism_config"]
    ds_scalar = ds_clipped.drop_vars(["pism_config"], errors="ignore").sum(dim=["x", "y"])
    ds_scalar = xr.merge([ds_scalar, pism_config])
    ds_scalar.to_netcdf(scalar_file)


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
    rgi_file = config["rgi"]["outline"]

    for o in ["spatial", "state"]:
        s_file = Path(config["output"][o])
        print(s_file)
        with ProgressBar():
            process_file(s_file, rgi_file)

    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed {time_elapsed:.0f}s")


def main():
    """
    Run main script.
    """

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Postprocess RGI Glacier."
    parser.add_argument(
        "RUN_FILE",
        help="""CONFIG TOML.""",
        nargs=1,
    )

    options, unknown = parser.parse_known_args()
    config_file = options.RUN_FILE[0]

    postprocess_glacier(config_file)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
