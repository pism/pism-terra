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
import logging
import time
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Literal

import cf_xarray
import dask
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
    infile: str | Path,
    outline_file: str | Path,
    rgi_type: Literal["G", "C"],
    client: Client,
    column: str = "rgi_id",
):
    """
    Clip a NetCDF dataset to glacier outlines and write per-outline scalar sums.

    Reads ``infile``, reprojects the geometries in ``outline_file`` to the
    dataset's CRS, and clips the dataset to each outline. For every outline the
    spatial variables are summed over ``x``/``y`` (with the outline ``area``
    attached) and stacked along an ``rgi_id`` dimension. The result is written
    to ``<output_root>/processed_scalar/fldsum_<rgi_type>_<name>.nc``, where
    ``output_root`` is the grandparent of ``infile`` (``infile.parent.parent``)
    and ``<name>`` is the input filename.

    Parameters
    ----------
    infile : str or Path
        Path to the NetCDF file to be clipped. Must contain x/y spatial dimensions.
    outline_file : str or Path
        Path to the glacier outline file (e.g., GeoPackage or shapefile) whose
        geometries the dataset is clipped to.
    rgi_type : {"G", "C"}
        RGI outline type being processed — glacier ("G") or glacier complex
        ("C"). Used as a prefix in the output filename (``fldsum_<rgi_type>_...``).
    client : dask.distributed.Client
        Dask client used to persist the dataset before clipping.
    column : str, default "rgi_id"
        Name of the column in ``outline_file`` used to label each clipped
        outline along the output ``rgi_id`` dimension.
    """

    infile = Path(infile)
    infile_name = infile.name
    output_root = infile.parent.parent
    scalar_dir = output_root / "processed_scalar"
    scalar_dir.mkdir(parents=True, exist_ok=True)
    scalar_file = scalar_dir / Path(f"fldsum_{rgi_type}_" + infile_name)

    outline = gpd.read_file(outline_file)

    start = time.time()
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=False)

    ds = xr.open_dataset(
        infile,
        decode_timedelta=False,
        decode_times=False,
        chunks="auto",
        engine="netcdf4",
    )
    mapping_var = ds.rio.grid_mapping
    dst_crs = ds[mapping_var].attrs["crs_wkt"]

    outline = outline.to_crs(dst_crs)

    # Spatial bounds vars (``x_bnds``/``y_bnds``) reference the pre-clip
    # x/y sizes and would inject dangling dimensions back into the output
    # after merge — h5netcdf serializes those as duplicate "x" dims. Drop
    # them before splitting; PISM doesn't require them on the clipped output.
    ds = ds.drop_vars(["x_bnds", "x_bounds", "y_bnds", "y_bounds", "mapping", "spatial_ref"], errors="ignore")

    # Separate variables that lack BOTH spatial (x, y) dimensions, as
    # rio.clip cannot handle them. Use ``and`` so that vars carrying only
    # one spatial dim (rare, but possible) still go down the spatial path.
    non_spatial_vars = [var for var in ds.data_vars if "x" not in ds[var].dims and "y" not in ds[var].dims]
    ds_non_spatial = ds[non_spatial_vars]
    ds = ds.drop_vars(non_spatial_vars).rio.write_crs(dst_crs).rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds = client.persist(ds)
    progress(ds)

    comp = {"zlib": True, "complevel": 2}

    dss = []
    for _, row in tqdm(outline.iterrows(), total=len(outline), desc="Clipping outlines"):
        ds_clipped = ds.rio.clip([row.geometry], drop=False)
        ds_sum = ds_clipped.sum(dim=["y", "x"]).compute()
        ds_sum["area"] = row.geometry.area
        ds_sum["area"].attrs.update({"units": "m^2"})
        dss.append(ds_sum.expand_dims({"RGIid": [row[column]]}))

    scalar = xr.concat(dss, dim="RGIid")

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


def postprocess_glacier(config_file: str | Path, rgi_type: Literal["G", "C"], n_workers: int = 4):
    """
    Postprocess a PISM glacier ``spatial`` output for one RGI outline type.

    Reads simulation paths from a TOML run configuration and clips the
    ``spatial`` output NetCDF to the requested RGI outline (glacier ``"G"`` or
    complex ``"C"``) using a Dask client, writing per-outline scalar sums via
    :func:`process_file`.

    Parameters
    ----------
    config_file : str or Path
        Path to a TOML file containing the PISM run configuration. The
        ``[rgi]`` table provides the outline paths (``outline_g``/``outline_c``,
        with a legacy ``outline`` fallback) and ``[output]`` the file paths.
    rgi_type : {"G", "C"}
        RGI outline type to process. Selects ``outline_<rgi_type>`` from the
        config's ``[rgi]`` table.
    n_workers : int, optional
        Number of Dask workers, by default 4.
    """

    config_toml = toml.load(config_file)
    config = json.loads(json.dumps(config_toml))

    start = time.time()
    rgi = config["rgi"]
    outline_file = rgi.get(f"outline_{rgi_type.lower()}", rgi.get("outline"))

    client = Client(n_workers=n_workers, threads_per_worker=1)
    logger.info("Dask dashboard: %s", client.dashboard_link)

    for o in ["spatial"]:
        s_file = Path(config["output"][o])
        process_file(s_file, outline_file, rgi_type, client)

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
    parser.description = "Postprocess RGI Glacier."
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

    postprocess_glacier(config_file, "C", n_workers=ntasks)
    postprocess_glacier(config_file, "G", n_workers=ntasks)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
