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

# pylint: disable=too-many-positional-arguments,unused-import
"""
Prepare ISMIP7 Greenland data sets.
"""

import logging
import os
import re
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Sequence

import cf_xarray
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # pylint: disable=unused-import
import toml
import xarray as xr
import xarray_regrid.methods.conservative  # pylint: disable=unused-import
from cdo import Cdo
from dask.distributed import Client, as_completed
from pyfiglet import Figlet
from tqdm.auto import tqdm

from pism_terra.domain import create_domain
from pism_terra.download import (
    download_earthaccess,
    download_gebco,
    download_netcdf,
    file_localizer,
)
from pism_terra.raster import create_ds
from pism_terra.vector import dissolve
from pism_terra.workflow import check_xr_fully, check_xr_lazy

xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)


def _make_path(year, base_path, gcm, pathway, short_hand, m_var, version):
    """
    Build the resolved file path for an ISMIP7 forcing variable and year.

    Parameters
    ----------
    year : int
        Year of the forcing file.
    base_path : Path
        Root directory for the forcing data.
    gcm : str
        GCM name (e.g. "CESM2-WACCM").
    pathway : str
        Emissions pathway (e.g. "historical", "ssp585").
    short_hand : str
        Short-hand identifier for the forcing type, or "none".
    m_var : str
        Variable name (e.g. "acabf", "tas").
    version : str
        Version string (e.g. "v1").

    Returns
    -------
    str
        Resolved file path as a string.
    """
    p = (
        base_path / Path(gcm) / Path(pathway) / Path(short_hand) / Path(m_var) / Path(version)
        if short_hand != "none"
        else base_path / Path(gcm) / Path(pathway) / Path(m_var) / Path(version)
    )
    v = (
        Path(f"{m_var}_{gcm}_{pathway}_{short_hand}_{year}.nc")
        if short_hand != "none"
        else Path(f"{m_var}_{gcm}_{pathway}_{year}.nc")
    )
    url = (p / v).resolve()
    return str(url)


def _process_single_forcing(
    gcm: str,
    forcing: str,
    base_path: Path,
    output_path: Path,
    pathway: str,
    version: str,
    hist_start_year: int,
    hist_end_year: int,
    proj_start_year: int,
    proj_end_year: int,
    short_hand: str,
    fields: list[str],
    ismip7_to_pism: dict[str, str],
    freq: str = "1mon",
    calendar: str = "365_day",
) -> list[Path]:
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
    hist_start_year : int
        Start year for time selection.
    hist_end_year : int
        End year for time selection.
    proj_start_year : int
        Start year for time selection.
    proj_end_year : int
        End year for time selection.
    short_hand : str
        Short hand identifier for forcing.
    fields : list[str]
        List of climate fields to process.
    ismip7_to_pism : dict[str, str]
        Variable name mapping from ISMIP7 to PISM conventions.
    freq : str, optional
        Frequency string for CDO time axis. Default is "1mon".
    calendar : str, optional
        Calendar type for CDO time axis. Default is "365_day".

    Returns
    -------
    Path
        Path to the output NetCDF file.
    """
    os.environ["HDF5_LOG_LEVEL"] = "0"
    cdo = Cdo()
    cdo.debug = True

    grid_file = file_localizer("s3://pism-cloud-data/ismip7_extra/grid.txt", dest=output_path)

    output_files = []

    hist_merge_cmds = []
    for m_var in fields:
        urls = [
            _make_path(year, base_path, gcm, "historical", short_hand, m_var, version)
            for year in range(hist_start_year, hist_end_year)
        ]
        k, v = m_var, ismip7_to_pism[m_var]
        tas_replace = "-setrtoc,0,230,230 -setrtoc,303,403,303" if m_var == "tas" else ""
        merge_cmd = (
            f"-chname,{k},{v} {tas_replace} -setgrid,{str(grid_file)} -mergetime [ "
            + " ".join(str(f) for f in urls)
            + " ]"
        )
        hist_merge_cmds.append(merge_cmd)

    proj_merge_cmds = []
    for m_var in fields:
        urls = [
            _make_path(year, base_path, gcm, pathway, short_hand, m_var, version)
            for year in range(proj_start_year, proj_end_year)
        ]
        k, v = m_var, ismip7_to_pism[m_var]
        tas_replace = "-setrtoc,0,230,230 -setrtoc,303,403,303" if m_var == "tas" else ""
        merge_cmd = (
            f"-chname,{k},{v} {tas_replace} -setgrid,{str(grid_file)} -mergetime [ "
            + " ".join(str(f) for f in urls)
            + " ]"
        )
        proj_merge_cmds.append(merge_cmd)

    hist_output_file = output_path / Path(
        f"ismip7_greenland_{forcing}_historical_{gcm}_{version}_{hist_start_year}_{hist_end_year}.nc"
    )

    cdo.setmisstoc(
        0,
        input=f"""-setgrid,{str(grid_file)} -settbounds,{freq} -setreftime,1850-01-01 -settunits,hours -setcalendar,{calendar} -settaxis,'{hist_start_year}-01-16 12:00,,{freq}' -merge """
        + " ".join(hist_merge_cmds),
        output=str(hist_output_file.resolve()),
        options="-f nc4 -z zip_2",
    )
    output_files.append(hist_output_file)

    proj_output_file = output_path / Path(
        f"ismip7_greenland_{forcing}_{pathway}_{gcm}_{version}_{proj_start_year}_{proj_end_year}.nc"
    )

    cdo.setmisstoc(
        0,
        input=f"""-setgrid,{str(grid_file)} -settbounds,{freq} -setreftime,1850-01-01 -settunits,hours -setcalendar,{calendar} -settaxis,'{proj_start_year}-01-16 12:00,,{freq}' -merge """
        + " ".join(proj_merge_cmds),
        output=str(proj_output_file.resolve()),
        options="-f nc4 -z zip_2",
    )
    output_files.append(proj_output_file)

    input_files = " ".join(str(f.resolve()) for f in output_files)
    merged_file = output_path / Path(
        f"ismip7_greenland_{forcing}_{pathway}_{gcm}_{version}_{hist_start_year}_{proj_end_year}.nc"
    )
    cdo.mergetime(
        input=input_files,
        options="-f nc4 -z zip_2",
        output=str(merged_file.resolve()),
    )
    output_files.append(merged_file)
    return output_files


def prepare_observations(
    url: str,
    input_path: Path | str,
    output_path: Path | str,
    config: dict,
    surface_dem: str | None = None,
    target_grid: xr.Dataset | xr.DataArray | None = None,
    force_overwrite: bool = False,
) -> dict[str, Path | str]:
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
    surface_dem : str or None, optional
        URL or path to an alternative surface DEM. When provided, the
        surface elevation is recalculated as bed + thickness using this DEM.
    target_grid : xarray.Dataset, xarray.DataArray, or None, optional
        Target grid for conservative regridding. When provided, bed and
        thickness are regridded and GEBCO bathymetry fills missing bed values.
    force_overwrite : bool, default False
        If True, re-download the file even if it already exists locally.

    Returns
    -------
    dict[str, Path or str]
        Dictionary with keys ``"boot_file"`` and ``"heatflux_file"``
        mapping to their respective output paths.
    """

    name = url.split("/")[-1]
    obs_file = Path(input_path) / Path(name)
    if (not check_xr_lazy(obs_file)) or force_overwrite:
        ds_bm = download_netcdf(url)
    else:
        ds_bm = xr.open_dataset(obs_file)

    if target_grid is not None:
        ds_bm_regridded = ds_bm[["bed", "thickness"]].regrid.conservative(target_grid)
        gebco_p = download_gebco(target_dir=input_path)
        gebco = xr.open_dataset(gebco_p, chunks="auto").rio.write_crs("EPSG:4326")
        gebco_bm_regridded = gebco.rio.reproject_match(ds_bm_regridded.rio.write_crs("EPSG:3413")).compute()
        ds_bm_regridded["bed"] = ds_bm_regridded["bed"].where(
            ds_bm_regridded["bed"].notnull(), gebco_bm_regridded["elevation"]
        )
        ds_bm_regridded = ds_bm_regridded.fillna(0)
    else:
        ds_bm_regridded = ds_bm

    ftt_mask = xr.where(ds_bm_regridded["thickness"] > 0, 1, 0)
    ftt_mask.name = "ftt_mask"
    if surface_dem is not None:
        surface_file = Path(input_path) / Path("surface_dem.nc")
        bed = ds_bm_regridded["bed"]
        if (not check_xr_lazy(surface_file)) or force_overwrite:
            ds = download_netcdf(surface_dem)
            ds.to_netcdf(surface_file)
        else:
            ds = xr.open_dataset(surface_file)
        surface = ds["surface"].regrid.conservative(target_grid)
        thickness = xr.where(surface > 0, surface - bed, 0)
        thickness = thickness.where(thickness > 10, 0)
        thickness.name = "thickness"
        thickness.attrs.update(ds_bm_regridded["thickness"].attrs)
        boot = xr.merge([bed, ftt_mask, surface, thickness])
    else:
        boot = xr.merge([ds_bm_regridded[["bed", "thickness"]], ftt_mask])
    boot = boot.fillna(0)
    ds = xr.merge([boot, ds_bm["mapping"]])

    geo = (
        ds_bm[["geothermal_heat_flux1"]]
        .rename_dims({"x1km": "x", "y1km": "y"})
        .rename_vars({"x1km": "x", "y1km": "y", "geothermal_heat_flux1": "bheatflx"})
        .regrid.conservative(target_grid)
    )
    geo = geo.where(geo != -9999, 0.042)

    ds["bed"].attrs.update({"standard_name": "bedrock_altitude", "units": "m"})
    ds = ds.rename_vars({k: v for k, v in config["ismip7_to_pism"].items() if k in ds}).drop_vars(
        ["crs", "spatial_ref"], errors="ignore"
    )
    for v in ds.data_vars:
        ds[v].attrs.pop("coordinates", None)
        ds[v].encoding.pop("coordinates", None)

    resolution = int(ds.x[1] - ds.x[0])
    obs_file = output_path / Path(f"boot_g{resolution}m_GreenlandObsISMIP7-v1.3.nc")
    comp = {"zlib": True, "complevel": 2}
    encoding = {var: comp for var in ds.data_vars}
    encoding.update({var: {"_FillValue": None} for var in list(ds.data_vars) + list(ds.coords)})
    ds.to_netcdf(obs_file, encoding=encoding)

    geo["bheatflx"].attrs.pop("coordinates", None)
    geo["bheatflx"].encoding.pop("coordinates", None)
    geo = geo.drop_vars("spatial_ref", errors="ignore")
    geo["mapping"] = ds_bm["mapping"]
    for v in geo.data_vars:
        geo[v].attrs.pop("coordinates", None)
        geo[v].encoding.pop("coordinates", None)
    geo_file = output_path / Path(f"heatflux_g{resolution}m_GreenlandObsISMIP7-v1.3.nc")
    geo_encoding = {var: {"_FillValue": None} for var in list(geo.data_vars) + list(geo.coords)}
    geo.to_netcdf(geo_file, encoding=geo_encoding)

    return {"boot_file": obs_file, "heatflux_file": geo_file}


def prepare_calfin(
    output_path: Path | str,
    resolution: int,
    x_bnds: list | np.ndarray,
    y_bnds: list | np.ndarray,
    freq: str = "ME",
    force_overwrite: bool = False,
    n_workers: int = 4,
) -> str | Path:
    """
    Prepare CALFIN glacier front retreat data as a gridded NetCDF.

    Downloads CALFIN terminus positions, groups by month, computes cumulative
    retreat extent, and rasterizes to the target resolution.

    Parameters
    ----------
    output_path : Path or str
        Directory for output files.
    resolution : int
        Grid resolution in meters.
    x_bnds : list or numpy.ndarray
        A list or array containing the minimum and maximum x-coordinate boundaries.
    y_bnds : list or numpy.ndarray
        A list or array containing the minimum and maximum y-coordinate boundaries.
    freq : str, default "ME"
        Pandas frequency string for temporal grouping.
    force_overwrite : bool, default False
        If True, reprocess even if the output file already exists.
    n_workers : int, default 4
        Number of parallel workers.

    Returns
    -------
    Path
        Path to the output NetCDF file.
    """
    x_min, x_max = x_bnds[0], x_bnds[1]
    y_min, y_max = y_bnds[1], y_bnds[0]
    geom = {
        "type": "Polygon",
        "crs": {"properties": {"name": "EPSG:3413"}},
        "bbox": [x_min, y_min, x_max, y_max],
        "coordinates": [[(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]],
    }

    output_path = Path(output_path)
    p_fn = output_path / Path(f"pism_g{resolution}m_frontretreat_calfin_1972_2019_{freq}.nc")

    if (not check_xr_lazy(p_fn)) or force_overwrite:

        tmp_path = output_path.parent / Path("calfin")

        # Download CALFIN data
        retreat_files = download_earthaccess(
            doi="10.5067/7FILV218JZA2", filter_str="Greenland_polygons", result_dir=tmp_path
        )
        retreat_file = next(f for f in retreat_files if f.suffix == ".shp")

        crs = "EPSG:3413"

        # Load reference data and CALFIN
        imbie = gpd.read_file(
            "s3://pism-cloud-data/ismip7_greenland_extra/GRE_Basins_IMBIE2_v1.3_w_shelves.gpkg"
        ).to_crs(crs)
        calfin = gpd.read_file(retreat_file).to_crs(crs)

        # Prepare CALFIN timestamps and geometry
        calfin["Date"] = pd.DatetimeIndex(calfin["Date"])
        calfin = calfin.set_index("Date").sort_index()
        calfin.geometry = calfin.geometry.make_valid()
        calfin_dissolved = calfin.dissolve()

        # Create base union geometry
        imbie_dissolved = imbie.dissolve()
        imbie_union = imbie_dissolved.union(calfin_dissolved)

        # Step 1: Group by month and dissolve each group
        groups = [(date, df) for date, df in calfin.groupby(pd.Grouper(freq=freq)) if len(df) > 0]

        with Client(n_workers=n_workers, threads_per_worker=1) as client:
            logger.info("Dask dashboard: %s", client.dashboard_link)

            futures = [client.submit(dissolve, df, date) for date, df in groups]
            grouped_results = []
            for future in tqdm(as_completed(futures), desc="Grouping geometries", total=len(futures)):
                grouped_results.append(future.result())

        calfin_grouped = pd.concat(grouped_results).reset_index()

        # Step 2: Cumulative union (O(n) instead of O(n²))
        logger.info("Computing cumulative unions...")
        cumulative_geoms = []
        cumulative = None
        for _, row in tqdm(calfin_grouped.iterrows(), total=len(calfin_grouped), desc="Cumulative dissolve"):
            if cumulative is None:
                cumulative = row.geometry
            else:
                cumulative = cumulative.union(row.geometry)
            cumulative_geoms.append({"Date": row["Date"], "geometry": cumulative})

        calfin_aggregated = gpd.GeoDataFrame(cumulative_geoms[1:], crs=crs).set_index("Date")

        # Step 3: Rasterize to grid
        agg_groups = [(date, df) for date, df in calfin_aggregated.groupby(pd.Grouper(freq=freq)) if len(df) > 0]

        with Client(n_workers=n_workers, threads_per_worker=1) as client:
            logger.info("Dask dashboard: %s", client.dashboard_link)

            futures = [
                client.submit(
                    create_ds,
                    tmp_path / f"frontretreat_g{resolution}m_{date.year}-{date.month}-{date.day}.nc",
                    date,
                    df,
                    imbie_union,
                    geom=geom,
                    resolution=resolution,
                )
                for date, df in agg_groups
            ]
            raster_results = []
            for future in tqdm(as_completed(futures), desc="Rasterizing geometries", total=len(futures)):
                raster_results.append(future.result())

        result_filtered = [r for r in raster_results if r is not None]

        # Merge and save
        logger.info("Merging datasets and saving to %s", p_fn.resolve())

        cdo = Cdo()
        cdo.settbounds(
            "1mon",
            input="-mergetime " + " ".join(str(f) for f in result_filtered),
            output=str(p_fn.resolve()),
            options="-f nc4 -z zip_2",
        )
    return p_fn


def prepare_ismip7_forcing(
    base_path: Path | str,
    output_path: Path | str,
    config: dict,
    n_workers: int = 2,
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
        Number of dask workers, by default 2.

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
            hist_start_year, hist_end_year = config["pathway"][pathway]["historical"]
            proj_start_year, proj_end_year = config["pathway"][pathway]["projection"]
            for forcing, forcing_dict in config["forcing"].items():
                short_hand = forcing_dict["short_hand"]
                fields = forcing_dict["fields"]
                tasks.append(
                    (
                        gcm,
                        forcing,
                        version,
                        hist_start_year,
                        hist_end_year,
                        proj_start_year,
                        proj_end_year,
                        pathway,
                        short_hand,
                        fields,
                    )
                )

    # Process in parallel using dask.distributed
    with Client(n_workers=n_workers, threads_per_worker=1) as client:
        logger.info("Dask dashboard: %s", client.dashboard_link)

        futures = []
        for (
            gcm,
            forcing,
            version,
            hist_start_year,
            hist_end_year,
            proj_start_year,
            proj_end_year,
            pathway,
            short_hand,
            fields,
        ) in tasks:
            future = client.submit(
                _process_single_forcing,
                gcm,
                forcing,
                base_path,
                output_path,
                pathway,
                version,
                hist_start_year,
                hist_end_year,
                proj_start_year,
                proj_end_year,
                short_hand,
                fields,
                ismip7_to_pism,
            )
            futures.append(future)

        # Collect results as they complete
        processed_files = []
        for future in as_completed(futures):
            output_files = future.result()
            logger.info("Completed: %s", output_files)
            processed_files.extend(output_files)

    elapsed = time.perf_counter() - start_time
    logger.info("Total processing time: %.2f seconds", elapsed)

    return processed_files
