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

# pylint: disable=unused-import

"""
Running.
"""
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import cf_xarray
import geopandas as gpd
import rioxarray
import toml
import xarray as xr


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
    # time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    rgi = gpd.read_file(config["outline"])
    crs = rgi.iloc[0]["epsg"]
    rgi_projected = rgi.to_crs(crs)
    for o in ["state", "spatial"]:
        s_file = Path(config["output"][o])
        s_file_name = s_file.name
        s_path = s_file.parent
        s_clipped_file = s_path / Path("clipped_" + s_file_name)

        ds = (
            xr.open_dataset(s_file, decode_times=False, decode_timedelta=False)
            .drop_vars("time_bounds")
            .rio.set_spatial_dims(x_dim="x", y_dim="y")
        )
        ds.rio.write_crs(crs, inplace=True)
        ds_clipped = ds.rio.clip(rgi_projected.geometry)
        ds_clipped.to_netcdf(s_clipped_file)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
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
