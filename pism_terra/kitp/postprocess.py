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

from pism_terra.log import setup_logging

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
    geometry defined in a glacier outline file (e.g., BASIN shapefile). Clipped spatial
    output is written to ``<output_root>/processed_spatial/clipped_<name>.nc`` and
    per-basin scalar sums to ``<output_root>/processed_scalar/fldsum_<name>.nc``,
    where ``output_root`` is the parent of the input file's directory.

    Parameters
    ----------
    infile : str or Path
        Path to the NetCDF file to be clipped. Must contain x/y spatial dimensions.
    basin_file : str or Path
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

    infile = Path(infile)
    infile_name = infile.name
    output_root = infile.parent.parent
    clipped_dir = output_root / "processed_spatial"
    scalar_dir = output_root / "processed_scalar"
    clipped_dir.mkdir(parents=True, exist_ok=True)
    scalar_dir.mkdir(parents=True, exist_ok=True)
    clipped_file = clipped_dir / Path("clipped_" + infile_name)
    scalar_file = scalar_dir / Path("fldsum_" + infile_name)

    basin = gpd.read_file(basin_file)
    if basin.crs is None or str(basin.crs) != str(crs):
        basin = basin.to_crs(crs)

    start = time.time()
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=False)

    ds = xr.open_dataset(
        infile,
        decode_timedelta=False,
        decode_times=False,
        chunks="auto",
        engine="netcdf4",
    )

    # Spatial bounds vars (``x_bnds``/``y_bnds``) reference the pre-clip
    # x/y sizes and would inject dangling dimensions back into the output
    # after merge — h5netcdf serializes those as duplicate "x" dims. Drop
    # them before splitting; PISM doesn't require them on the clipped output.
    ds = ds.drop_vars(["x_bnds", "x_bounds", "y_bnds", "y_bounds", "mapping"], errors="ignore")

    # Separate variables that lack BOTH spatial (x, y) dimensions, as
    # rio.clip cannot handle them. Use ``and`` so that vars carrying only
    # one spatial dim (rare, but possible) still go down the spatial path.
    non_spatial_vars = [var for var in ds.data_vars if "x" not in ds[var].dims and "y" not in ds[var].dims]
    ds_non_spatial = ds[non_spatial_vars]
    ds = ds.drop_vars(non_spatial_vars).rio.write_crs(crs).rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds = client.persist(ds)
    progress(ds)

    comp = {"zlib": True, "complevel": 2}

    dss = []
    for _, row in tqdm(basin.iterrows(), total=len(basin), desc="Clipping basins"):
        ds_clipped = ds.rio.clip([row.geometry], drop=False)
        ds_sum = ds_clipped.sum(dim=["y", "x"]).compute()
        ds_sum["area"] = row.geometry.area
        ds_sum["area"].attrs.update({"units": "m^2"})
        dss.append(ds_sum.expand_dims({"basin": [row[column]]}))

    scalar = xr.concat(dss, dim="basin")

    logger.info("Writing %s", scalar_file)
    # Keep non-spatial vars (e.g. pism_config)
    extra_vars = [v for v in ds_non_spatial.data_vars if "time" not in ds_non_spatial[v].dims]
    if extra_vars:
        scalar = xr.merge([scalar, ds_non_spatial[extra_vars].compute()])
    encoding_scalar = {var: comp for var in scalar.data_vars}
    scalar.to_netcdf(scalar_file, encoding=encoding_scalar, engine="netcdf4")

    end = time.time()
    time_elapsed = end - start
    logger.info("Time elapsed for %s: %.0fs", infile_name, time_elapsed)


def postprocess_glacier(config_file: str | Path, n_workers: int = 4):
    """
    Postprocess KITP output by clipping spatial output to basin geometries.

    Reads a TOML run-configuration, opens the configured ``spatial`` output
    NetCDF, and clips it to the basin outline using a Dask client.

    Parameters
    ----------
    config_file : str or Path
        Path to a TOML file containing PISM run configuration with at least
        ``[basin].outline`` and ``[output].spatial`` keys.
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

    config_path = Path(config_file).resolve().parent
    setup_logging(config_path / "postprocess.log")

    postprocess_glacier(config_file, n_workers=ntasks)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
