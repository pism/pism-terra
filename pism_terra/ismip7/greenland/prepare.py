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
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

# pylint: disable=too-many-positional-arguments
"""
Prepare ISMIP7 Greenland data sets.
"""

import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Sequence

import cf_xarray
import geopandas as gpd
import pandas as pd
import rioxarray  # pylint: disable=unused-import
import toml
import xarray as xr
import xarray_regrid.methods.conservative  # pylint: disable=unused-import
from dask.distributed import Client, as_completed
from joblib import Parallel, delayed
from pyfiglet import Figlet
from tqdm.auto import tqdm

from pism_terra.domain import create_domain
from pism_terra.download import download_earthaccess, download_netcdf
from pism_terra.raster import create_ds
from pism_terra.vector import aggregate, dissolve
from pism_terra.workflow import check_xr_fully, check_xr_lazy, tqdm_joblib

xr.set_options(keep_attrs=True)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Prepare ISMIP7 Greenland input data sets.

    This function is the programmatic entry point. It parses command-line style
    arguments, creates the target grid, downloads and processes observation data,
    and prepares climate/ocean forcing files for PISM simulations.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments **excluding** the program name (i.e., like
        ``sys.argv[1:]``). If ``None`` (default), arguments are taken from the
        current process' ``sys.argv[1:]``. Passing ``argv=[]`` is recommended
        when calling from a Jupyter notebook to avoid ipykernel arguments.

    Returns
    -------
    dict[str, Any]
        Results dictionary containing:

        - ``"config"``: dict
          The parsed TOML configuration used for processing.
    """

    parser = ArgumentParser()
    parser.add_argument("--obs-path", default="data/obs")
    parser.add_argument(
        "--force-overwrite",
        help="Force downloading all files.",
        action="store_true",
        default=False,
    )
    parser.add_argument("CONFIG_FILE", nargs=1)
    parser.add_argument("DATA_PATH", nargs=1)
    parser.add_argument("OUTPUT_PATH", nargs=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_file = args.CONFIG_FILE[0]
    data_path = Path(args.DATA_PATH[0])
    force_overwrite = args.force_overwrite
    obs_path = Path(args.obs_path)
    output_path = Path(args.OUTPUT_PATH[0])
    output_path.mkdir(parents=True, exist_ok=True)

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    print("=" * 120)
    print(banner)
    print("=" * 120)
    print("Preparing ISMIP7 Greenland data")
    print("-" * 120)
    print("")

    config = toml.loads(Path(config_file).read_text("utf-8"))

    # ISMIP6 grid
    resolution = 1000.0
    x_bnds = [-720000.0 - resolution / 2, 960000.0 + resolution / 2]
    y_bnds = [-3450000.0 - resolution / 2, -570000.0 + resolution / 2]

    print("-" * 120)
    print("Grid File")
    print("-" * 120)

    grid_ds = create_domain(x_bnds, y_bnds, resolution=resolution)
    grid_file = output_path / Path("ismip7_greenland_grid.nc")
    grid_ds.to_netcdf(grid_file)
    check_xr_fully(grid_file)

    print("-" * 120)
    print("Calfin Glacier Fronts File")
    print("-" * 120)

    calfin_file = prepare_calfin(output_path, resolution=600)

    url = "https://g-ab4495.8c185.08cc.data.globus.org/ISMIP6/ISMIP7_Prep/Observations/Greenland/GreenlandObsISMIP7-v1.3.nc"
    print("-" * 120)
    print("Boot File")
    print("-" * 120)
    boot_file = prepare_observations(url, obs_path, output_path, config, thin=4, force_overwrite=force_overwrite)
    check_xr_lazy(boot_file)

    print("-" * 120)
    print("Forcings")
    print("-" * 120)
    forcing_files = prepare_forcing(data_path, output_path, config)

    return {"config": config, "grid_file": grid_file, "boot_file": boot_file, "forcing_files": forcing_files}


def _process_single_forcing(
    gcm: str,
    forcing: str,
    base_path: Path,
    output_path: Path,
    pathway: str,
    version: str,
    start_year: int,
    end_year: int,
    short_hand: str,
    fields: list[str],
    ismip7_to_pism: dict[str, str],
    freq: str = "MS",
    calendar: str = "noleap",
) -> Path:
    """
    Process a single GCM/forcing combination.

    Parameters
    ----------
    gcm : str
        GCM name.
    forcing : str
        Forcing type.
    base_path : Path
        Base path to input data.
    output_path : Path
        Output directory.
    pathway : str
        Pathway name (e.g., "historical", "ssp585").
    version : str
        Version string (e.g., "v1").
    start_year : int
        Start year for time selection.
    end_year : int
        End year for time selection.
    short_hand : str
        Short hand identifier for forcing.
    fields : list[str]
        List of climate fields to process.
    ismip7_to_pism : dict[str, str]
        Variable name mapping from ISMIP7 to PISM conventions.
    freq : str, optional
        Frequency string for date range generation. Default is "MS".
    calendar : str, optional
        Calendar type for xarray encoding. Default is "noleap".

    Returns
    -------
    Path
        Path to the output NetCDF file.
    """
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    dss = []
    for m_var in fields:
        p = (
            base_path / Path(gcm) / Path(pathway) / Path(short_hand) / Path(m_var) / Path(version)
            if short_hand != "none"
            else base_path / Path(gcm) / Path(pathway) / Path(m_var) / Path(version)
        )

        urls = list(p.glob(f"{m_var}_{gcm}_{pathway}_*.nc"))
        ds = xr.open_mfdataset(
            urls,
            parallel=True,
            decode_times=time_coder,
            engine="h5netcdf",
            chunks={"time": 12, "y": -1, "x": -1},
            data_vars="minimal",
        ).sel({"time": slice(str(start_year), str(end_year))})
        dss.append(ds)

    ds = xr.merge(dss)
    ds = ds.rename_vars({k: v for k, v in ismip7_to_pism.items() if k in ds})
    ds.rio.write_crs("EPSG:3413", inplace=True).rio.write_coordinate_system(inplace=True)

    time = xr.date_range(start=str(start_year), freq=freq, periods=ds.time.size + 1, use_cftime=True, calendar=calendar)
    time_centered = time[:-1] + (time[1:] - time[:-1]) / 2
    ds = ds.assign_coords(time=time_centered)
    time_bounds = xr.DataArray(
        data=list(zip(time[:-1], time[1:])),
        dims=["time", "bnds"],
    )
    ds["time_bounds"] = time_bounds
    ds["time"].attrs.update({"bounds": "time_bounds"})
    ds["time"].encoding.update({"units": "days since 1850-01-01", "_FillValue": None, "dtype": float})
    ds["time_bounds"].encoding.update({"units": "days since 1850-01-01", "_FillValue": None, "dtype": float})

    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {"_FillValue": None}
        if var not in ("spatial_ref", "time", "time_bounds"):
            encoding[var].update({"zlib": True, "complevel": 2})

    output_file = output_path / Path(f"ismip7_greenland_{forcing}_{pathway}_{gcm}_{version}_{start_year}_{end_year}.nc")
    ds.to_netcdf(output_file, encoding=encoding, engine="h5netcdf")

    return output_file


def prepare_observations(
    url: str,
    input_path: Path | str,
    output_path: Path | str,
    config: dict,
    thin: int = 4,
    force_overwrite: bool = False,
) -> Path | str:
    """
    Download and prepare ISMIP7 Greenland observation data.

    Downloads the observation NetCDF file from the given URL (if not already
    cached or if ``force_overwrite`` is True), extracts relevant variables
    (mapping, geothermal heat flux, bed, thickness), renames them according
    to the config mapping, and writes the result to the output directory.

    Parameters
    ----------
    url : str
        URL to the ISMIP7 Greenland observation NetCDF file.
    input_path : Path or str
        Directory where the raw downloaded file is cached.
    output_path : Path or str
        Directory where the processed boot file is written.
    config : dict
        Configuration dictionary with variable name mappings. Keys present
        in the dataset are renamed to their corresponding values.
    thin : int
        Factor to reduce BedMachine grid (150m). thin=4 -> 600m grid.
        Interpolation is done conservatively.
    force_overwrite : bool, default False
        If True, re-download the file even if it already exists locally.

    Returns
    -------
    Path or str
        Path to the output boot NetCDF file.
    """

    name = url.split("/")[-1]
    obs_file = Path(input_path) / Path(name)
    if (not check_xr_lazy(obs_file)) or force_overwrite:
        ds = download_netcdf(url)
    else:
        ds = xr.open_dataset(obs_file)

    ds_bm = ds[["bed", "thickness"]]
    da_1km = ds["geothermal_heat_flux1"]
    da_1km = da_1km.where(da_1km != -9999, 0.042)

    if thin > 1:
        target_da = ds_bm.thin({"x": thin, "y": thin})
        ds_bm_regridded = ds_bm.regrid.conservative(target_da)
    else:
        ds_bm_regridded = ds_bm
    ds = xr.merge([ds_bm_regridded, da_1km, ds["mapping"]])

    ds = ds.rename_vars({k: v for k, v in config.items() if k in ds})
    obs_file = output_path / Path("boot_GreenlandObsISMIP7-v1.3.nc")
    ds.to_netcdf(obs_file)
    return obs_file


def prepare_calfin(output_path: Path | str, resolution: int = 150, n_workers: int = 4) -> str | Path:
    """
    Prepare CALFIN glacier front retreat data as a gridded NetCDF.

    Downloads CALFIN terminus positions, groups by month, computes cumulative
    retreat extent, and rasterizes to the target resolution.

    Parameters
    ----------
    output_path : Path or str
        Directory for output files.
    resolution : int, default 150
        Grid resolution in meters.
    n_workers : int, default 4
        Number of parallel workers.

    Returns
    -------
    Path
        Path to the output NetCDF file.
    """
    x_min, x_max = -653000, 879700
    y_min, y_max = -632750, -3384350
    geom = {
        "type": "Polygon",
        "crs": {"properties": {"name": "EPSG:3413"}},
        "bbox": [x_min, y_min, x_max, y_max],
        "coordinates": [[(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]],
    }

    output_path = Path(output_path)
    tmp_path = output_path.parent / Path(output_path.name + "_tmp")

    # Download CALFIN data
    calfin_files = download_earthaccess(
        doi="10.5067/7FILV218JZA2", filter_str="Greenland_polygons", result_dir=tmp_path
    )
    calfin_file = next(f for f in calfin_files if f.suffix == ".shp")

    crs = "EPSG:3413"

    # Load reference data and CALFIN
    imbie = gpd.read_file("s3://pism-cloud-data/ismip7_extra/GRE_Basins_IMBIE2_v1.3_w_shelves.gpkg").to_crs(crs)
    calfin = gpd.read_file(calfin_file).to_crs(crs)

    # Prepare CALFIN timestamps and geometry
    calfin["Date"] = pd.DatetimeIndex(calfin["Date"])
    calfin = calfin.set_index("Date").sort_index()
    calfin.geometry = calfin.geometry.make_valid()

    # Create base union geometry
    imbie_gis = imbie[imbie["SUBREGION1"] == "GIS"].dissolve()
    imbie_union = imbie_gis.union(calfin.dissolve())

    freq = "ME"

    # Step 1: Group by month and dissolve each group
    groups = [(date, df) for date, df in calfin.groupby(pd.Grouper(freq=freq)) if len(df) > 0]
    with tqdm_joblib(tqdm(desc="Grouping geometries", total=len(groups))):
        grouped_results = Parallel(n_jobs=n_workers)(delayed(dissolve)(df, date) for date, df in groups)
    calfin_grouped = pd.concat(grouped_results).reset_index()

    # Step 2: Cumulative union (O(n) instead of O(n²))
    print("Computing cumulative unions...")
    cumulative_geoms = []
    cumulative = None
    for idx, row in tqdm(calfin_grouped.iterrows(), total=len(calfin_grouped), desc="Cumulative dissolve"):
        if cumulative is None:
            cumulative = row.geometry
        else:
            cumulative = cumulative.union(row.geometry)
        cumulative_geoms.append({"Date": row["Date"], "geometry": cumulative})

    calfin_aggregated = gpd.GeoDataFrame(cumulative_geoms[1:], crs=crs).set_index("Date")

    # Step 3: Rasterize to grid
    agg_groups = [(date, df) for date, df in calfin_aggregated.groupby(pd.Grouper(freq=freq)) if len(df) > 0]
    with tqdm_joblib(tqdm(desc="Rasterizing geometries", total=len(agg_groups))):
        raster_results = Parallel(n_jobs=n_workers)(
            delayed(create_ds)(date, df, imbie_union, geom=geom, resolution=resolution) for date, df in agg_groups
        )

    result_filtered = [r for r in raster_results if r is not None]

    # Merge and save
    p_fn = output_path / Path(f"pism_g{resolution}m_frontretreat_calfin_1972_2019_{freq}.nc")
    print(f"Merging datasets and saving to {p_fn.absolute()}")

    ds = xr.open_mfdataset(result_filtered).load()
    ds = ds.cf.add_bounds("time")
    ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    ds.rio.write_crs(crs, inplace=True)
    ds.to_netcdf(p_fn)

    return p_fn


def prepare_forcing(
    base_path: Path | str,
    output_path: Path | str,
    config: dict,
    n_workers: int = 4,
) -> Sequence[Path | str]:
    """
    Process forcing data for all GCMs and forcings in parallel.

    Parameters
    ----------
    base_path : Path or str
        Base path to input data.
    output_path : Path or str
        Output directory.
    config : dict
        Configuration dictionary.
    n_workers : int, optional
        Number of dask workers, by default 4.

    Returns
    -------
    list[Path | str]
        List of output file paths.
    """
    start_time = time.perf_counter()

    base_path = Path(base_path)
    output_path = Path(output_path)

    ismip7_to_pism = config["ismip7_to_pism"]
    # Build list of tasks
    tasks = []

    for gcm in config["gcms"]:
        for pathway in config["pathway"]:
            version = "v" + str(config["pathway"][pathway]["version"])
            start_year = config["pathway"][pathway]["start_year"]
            end_year = config["pathway"][pathway]["end_year"]
            for forcing, forcing_dict in config["forcing"].items():
                short_hand = forcing_dict["short_hand"]
                fields = forcing_dict["fields"]
                tasks.append((gcm, forcing, version, start_year, end_year, pathway, short_hand, fields))

    # Process in parallel using dask.distributed
    with Client(n_workers=n_workers, threads_per_worker=1) as client:
        print(f"Dask dashboard: {client.dashboard_link}")

        futures = []
        for gcm, forcing, version, start_year, end_year, pathway, short_hand, fields in tasks:
            future = client.submit(
                _process_single_forcing,
                gcm,
                forcing,
                base_path,
                output_path,
                pathway,
                version,
                start_year,
                end_year,
                short_hand,
                fields,
                ismip7_to_pism,
            )
            futures.append(future)

        # Collect results as they complete
        processed_files = []
        for future in as_completed(futures):
            output_file = future.result()
            print(f"Completed: {output_file}")
            processed_files.append(output_file)

    elapsed = time.perf_counter() - start_time
    print(f"Total processing time: {elapsed:.2f} seconds")

    return processed_files


def cli(argv: Sequence[str] | None = None) -> int:
    """
    Console entry point.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments (without the program name). If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    _ = main(argv=argv)
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(cli())
