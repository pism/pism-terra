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

# pylint: disable=too-many-positional-arguments,unused-import,broad-exception-caught

"""
Prepare KITP Greenland data sets.
"""

import os
import re
import shutil
import tempfile
import time
from argparse import ArgumentParser
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed as cf_as_completed
from pathlib import Path
from typing import Any

import cf_xarray
import cftime
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
from tqdm import tqdm

from pism_terra.aws import s3_to_local
from pism_terra.domain import create_domain
from pism_terra.download import download_hirham, unzip_files
from pism_terra.raster import create_ds
from pism_terra.vector import dissolve
from pism_terra.workflow import check_xr_fully, check_xr_lazy

xr.set_options(keep_attrs=True)

hirham_grid = """
gridtype = projection
xname = rlon
xsize = 402
ysize = 602
xfirst = -16
xinc = 0.05
xunits = degree
yname = rlat
yfirst = -14
yinc = 0.05
yunits = degree
grid_mapping_name = rotated_latitude_longitude
grid_north_pole_longitude = 160
grid_north_pole_latitude = 18
north_pole_grid_longitude = 0
"""

bedmachine_grid = """
gridtype = projection
xsize     = 10218
ysize     = 18346
xunits   = "meter"
yunits   = "meter"
xfirst    = -65300
xinc      = 150
yfirst    = -3384425
yinc      = 150
grid_mapping = crs
grid_mapping_name = polar_stereographic
straight_vertical_longitude_from_pole = -39.
standard_parallel = 71.
latitude_of_projection_origin = 90.
false_easting = 0.
false_northing = 0.
"""

ismip6_grid = """
gridtype = projection
xsize     = 1496
ysize     = 2700
xunits   = "meter"
yunits   = "meter"
xfirst    = -640000
xinc      = 1000
yfirst    = -3355000
yinc      = 1000
grid_mapping = crs
grid_mapping_name = polar_stereographic
straight_vertical_longitude_from_pole = -45.
standard_parallel = 70.
latitude_of_projection_origin = 90.
false_easting = 0.
false_northing = 0.
"""


def process_hirham_cdo(
    data_dir: str | Path,
    output_file: str | Path,
    base_url: str,
    vars_dict: dict,
    overwrite: bool = False,
    max_workers: int = 4,
    start_year: int = 1980,
    end_year: int = 2021,
) -> None:
    """
    Prepare and process HIRHAM data and save the output to a NetCDF file.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Directory containing the input data.
    output_file : Union[str, Path]
        Path to the output NetCDF file.
    base_url : str
        Base URL for downloading HIRHAM data.
    vars_dict : Dict
        Dictionary of variables to process with their attributes.
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    max_workers : int, optional
        Maximum number of parallel workers, by default 4.
    start_year : int, optional
        Starting year for processing, by default 1980.
    end_year : int, optional
        Ending year for processing, by default 2021.
    """
    print("Processing HIRHAM")

    hirham_dir = data_dir / Path("hirham")
    hirham_dir.mkdir(parents=True, exist_ok=True)
    hirham_nc_dir = hirham_dir / Path("nc")
    hirham_nc_dir.mkdir(parents=True, exist_ok=True)
    hirham_zip_dir = hirham_dir / Path("zip")
    hirham_zip_dir.mkdir(parents=True, exist_ok=True)

    responses = download_hirham(
        base_url,
        start_year,
        end_year,
        output_dir=hirham_zip_dir,
        max_workers=max_workers,
    )

    responses = unzip_files(
        responses,
        output_dir=hirham_nc_dir,
        overwrite=overwrite,
        max_workers=max_workers,
    )
    hirham_grid_path = hirham_dir / Path("hirham_grid.txt")
    hirham_grid_path.write_text(hirham_grid)
    target_grid_path = hirham_dir / Path("ismip6_grid.txt")
    target_grid_path.write_text(ismip6_grid)

    # Initialize an empty list to store the parts of the string
    chname_parts = []

    # Iterate over the dictionary items
    for key, value in vars_dict.items():
        chname_parts.append(key)
        chname_parts.append(value["pism_name"])
    chname = ",".join(chname_parts)

    # Initialize an empty list to store the parts of the string
    setattribute_parts = []

    # Iterate over the dictionary items
    for key, value in vars_dict.items():
        setattribute_parts.append(f"""{key}@units='{value["units"]}'""")
    setattribute = ",".join(setattribute_parts)

    print("Merging daily files and calculate multi-year monthly means.")

    cdo = Cdo()
    cdo.debug = True

    # First merge daily files in batches to avoid "Argument list too long"
    merged_file = hirham_nc_dir / "merged_daily.nc"
    print("Merging daily files in batches...")
    batches = []
    for year in range(start_year, end_year + 1):
        responses = list((hirham_nc_dir / str(year)).glob("Daily*.nc"))
        batch = sorted([str(p.resolve()) for p in responses])
        batch_out = str((hirham_nc_dir / f"batch_{year}.nc").resolve())
        batches.append((year, batch, batch_out))

    def _merge_batch(args):
        """
        Merge and process a single year-batch of daily HIRHAM files.

        Parameters
        ----------
        args : tuple
            A ``(batch, batch_out)`` pair where *batch* is a list of input
            file paths and *batch_out* is the output file path.

        Returns
        -------
        str
            Path to the merged output file.
        """
        year, batch, batch_out = args
        tmpdir = tempfile.mkdtemp()
        cdo_local = Cdo(tempdir=tmpdir)
        merged = os.path.join(tmpdir, f"merged_{year}.nc")
        cdo_local.setrtomiss(
            "-1e40,-1e10",
            input=f"""-setmissval,-9e33 -selvar,precipitation,air_temp -setreftime,{start_year}-01-01 -settbounds,day -settaxis,"{year}-01-01" -setattribute,precipitation@standard_name="precipitation_flux" -setattribute,precipitation@units="kg m^-2 day^-1"  -aexpr,"precipitation=snowfall+rainfall" -chname,{chname} -setattribute,{setattribute} -selvar,{",".join(vars_dict.keys())} -setgrid,{str(hirham_grid_path.resolve())} -mergetime """
            + " ".join(batch),
            output=merged,
            options="-f nc4 -z zip_2 -P 1",
        )
        monmean = os.path.join(tmpdir, f"monmean_{year}.nc")
        cdo_local.monmean(
            input=merged,
            output=monmean,
            options="-f nc4 -z zip_2 -P 1",
        )
        monstd = os.path.join(tmpdir, f"monstd_{year}.nc")
        cdo_local.setattribute(
            """air_temp_sd@long_name="standard deviation of 2-m air temperature" """,
            input=f"""-delattribute,"air_temp_sd@standard_name" -chname,air_temp,air_temp_sd -monstd -selvar,air_temp {merged}""",
            output=monstd,
            options="-f nc4 -z zip_2 -P 1",
        )
        cdo_local.merge(
            input=f"{monmean} {monstd}",
            output=batch_out,
            options="-f nc4 -z zip_2 -P 1",
        )
        shutil.rmtree(tmpdir, ignore_errors=True)

        return batch_out

    batch_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_merge_batch, b): b for b in batches}
        for future in tqdm(cf_as_completed(futures), total=len(futures), desc="Merging batches"):
            batch_files.append(future.result())
    batch_files.sort()

    if len(batch_files) > 1:
        cdo.mergetime(
            input=" ".join(batch_files), output=str(merged_file.resolve()), options=f"-f nc4 -z zip_2 -P {max_workers}"
        )
        for bf in batch_files:
            Path(bf).unlink(missing_ok=True)
    else:
        Path(batch_files[0]).rename(merged_file)

    start = time.time()
    ds = cdo.setmisstodis(
        input=f"""-remapycon,{str(target_grid_path.resolve())} -ymonmean {merged_file}""",
        options=f"-f nc4 -z zip_2 -P {max_workers}",
        returnXDataset=True,
    )
    merged_file.unlink(missing_ok=True)

    ds = ds.drop_vars("time_bnds", errors="ignore")

    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    bounds_start = np.cumsum([0] + month_lengths[:-1]).astype("float64")
    bounds_end = np.cumsum(month_lengths).astype("float64")
    time_mid = (bounds_start + bounds_end) / 2.0

    time_bounds = np.column_stack([bounds_start, bounds_end])

    ds = ds.assign_coords(time=("time", time_mid))
    ds["time"].attrs.update(
        {"standard_name": "time", "units": "days since 0001-01-01", "calendar": "365_day", "bounds": "time_bounds"}
    )
    ds["time_bounds"] = (("time", "nv"), time_bounds)

    for var in list(ds.data_vars) + list(ds.coords):
        ds[var].attrs.pop("missing_value", None)
        ds[var].attrs.pop("_FillValue", None)
        ds[var].encoding["missing_value"] = None
        ds[var].encoding["_FillValue"] = None

    ds.to_netcdf(output_file)

    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed {time_elapsed:.0f}s")


def prepare_baseline_climatology(
    output_path: Path | str,
    start_year: int,
    end_year: int,
    version: str,
    n_workers: int = 4,
    force_overwrite: bool = False,
) -> Path:
    """
    Process baseline monthly climatology.

    Parameters
    ----------
    output_path : Path or str
        Output directory.
    start_year : int
        First year of the baseline period.
    end_year : int
        Last year of the baseline period.
    version : str
        Version string appended to the output filename.
    n_workers : int, optional
        Number of dask workers, by default 4.
    force_overwrite : bool, default ``False``
        If ``True``, downstream helpers may regenerate intermediate/final artifacts
        even if cache files exist.

    Returns
    -------
    Path
        Path to the output climatology file.
    """
    start_time = time.perf_counter()

    hirham_url = "http://ensemblesrt3.dmi.dk/data/prudence/temp/nichan/Daily2D_GrIS/"
    hirham_vars_dict: dict[str, dict[str, str]] = {
        "tas": {"pism_name": "air_temp", "units": "kelvin"},
        "gld": {"pism_name": "climatic_mass_balance", "units": "kg m^-2 day^-1"},
        "rainfall": {"pism_name": "rainfall", "units": "kg m^-2 day^-1"},
        "snfall": {"pism_name": "snowfall", "units": "kg m^-2 day^-1"},
    }

    output_path = Path(output_path)

    output_file = output_path / Path(f"HIRHAM5-ERA5_YMM_{start_year}_{end_year}_{version}.nc")
    if (not check_xr_lazy(output_file)) or force_overwrite:
        process_hirham_cdo(
            data_dir=output_path,
            vars_dict=hirham_vars_dict,
            start_year=start_year,
            end_year=end_year,
            output_file=output_file,
            base_url=hirham_url,
            max_workers=n_workers,
        )

    elapsed = time.perf_counter() - start_time
    print(f"Total processing time: {elapsed:.2f} seconds")

    return output_file


def prepare_anomalies(
    output_path: Path | str,
    bucket: str,
    prefix: str,
    gcms: list[str],
    present_day_forcings: list[str],
    future_forcings: list[str],
    version: str,
    n_workers: int = 4,
    force_overwrite: bool = False,
) -> list[Path]:
    """
    Process forcing data for all GCMs and forcings in parallel.

    Parameters
    ----------
    output_path : Path or str
        Output directory.
    bucket : str
        AWS S3 bucket name containing the forcing data.
    prefix : str
        S3 key prefix for the forcing data.
    gcms : Sequence[str]
        List of GCM names to process.
    present_day_forcings : list of str
        Present-day forcing experiment names (e.g. ``["pdSST-pdSIC"]``).
    future_forcings : list of str
        Future forcing experiment names (e.g. ``["futSST-pdSIC"]``).
    version : str
        Version string appended to the output filename.
    n_workers : int, optional
        Number of dask workers, by default 4.
    force_overwrite : bool, default ``False``
        If ``True``, downstream helpers may regenerate intermediate/final artifacts
        even if cache files exist.

    Returns
    -------
    list[Path]
        List of output file paths.
    """

    _ = n_workers

    forcing_path = output_path / Path(prefix)
    s3_to_local(bucket, prefix=prefix, dest=forcing_path)

    target_grid_path = output_path / Path("ismip6_grid.txt")
    target_grid_path.write_text(ismip6_grid)

    height_file = forcing_path / Path("height.nc")
    start = time.perf_counter()

    # Build list of all (gcm, pd_forcing, ff_forcing) tasks
    tasks = [(gcm, pd_forcing, ff) for gcm in gcms for pd_forcing in present_day_forcings for ff in future_forcings]

    def _process_anomaly(args):
        """
        Process a single GCM anomaly combination.

        Parameters
        ----------
        args : tuple
            A ``(gcm, pd_forcing, ff)`` triple.

        Returns
        -------
        Path
            Path to the output file.
        """
        gcm, pd_forcing, ff = args
        ff_tas_file = forcing_path / Path(ff) / Path(f"tas_Amon_{gcm}_{ff}.nc")
        if ff == "pa-futArcSIC-ext":
            ff_pr_file = forcing_path / Path(ff) / Path(f"pr_day_{gcm}_{ff}.nc")
        else:
            ff_pr_file = forcing_path / Path(ff) / Path(f"pr_Amon_{gcm}_{ff}.nc")
        if pd_forcing == "pa-pdSIC-ext":
            pd_pr_file = forcing_path / Path(pd_forcing) / Path(f"pr_day_{gcm}_{pd_forcing}.nc")
        else:
            pd_pr_file = forcing_path / Path(pd_forcing) / Path(f"pr_Amon_{gcm}_{pd_forcing}.nc")

        pd_tas_file = forcing_path / Path(pd_forcing) / Path(f"tas_Amon_{gcm}_{pd_forcing}.nc")
        output_file = forcing_path / Path(f"{gcm}_anomalies_{ff}_{pd_forcing}_{version}.nc")

        if (not check_xr_lazy(output_file, verbose=False)) or force_overwrite:
            with tempfile.TemporaryDirectory() as tmpdir:
                cdo_local = Cdo(tempdir=tmpdir)

                ds = cdo_local.setmisstodis(
                    input=f"""-remapycon,{str(target_grid_path.resolve())} -chname,pr,precipitation,tas,air_temp -merge -setattribute,height@units="m",height@standard_name="surface_altitude" -selvar,height {height_file} -sub -merge [ -selvar,tas {str(ff_tas_file.resolve())} -selvar,pr {str(ff_pr_file.resolve())} ] -merge [ -selvar,tas {str(pd_tas_file.resolve())} -selvar,pr {str(pd_pr_file.resolve())} ] """,
                    returnXDataset=True,
                    options="-f nc4 -z zip_2 -P 1",
                )

                ds["air_temp"].attrs["units"] = "kelvin"
                ds["precipitation"] = ds["precipitation"] * 86400.0
                ds["precipitation"].attrs["units"] = "kg m^-2 day^-1"
                ds = ds.drop_vars("time_bnds", errors="ignore")

            month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            bounds_start = np.cumsum([0] + month_lengths[:-1]).astype("float64")
            bounds_end = np.cumsum(month_lengths).astype("float64")
            time_mid = (bounds_start + bounds_end) / 2.0

            time_bounds = np.column_stack([bounds_start, bounds_end])

            ds = ds.assign_coords(time=("time", time_mid))
            ds["time"].attrs.update(
                {
                    "standard_name": "time",
                    "units": "days since 0001-01-01",
                    "calendar": "365_day",
                    "bounds": "time_bounds",
                }
            )
            ds["time_bounds"] = (("time", "nv"), time_bounds)

            for var in list(ds.data_vars) + list(ds.coords):
                ds[var].attrs.pop("missing_value", None)
                ds[var].attrs.pop("_FillValue", None)
                ds[var].encoding["missing_value"] = None
                ds[var].encoding["_FillValue"] = None

            ds.to_netcdf(output_file)

        return output_file

    result = []
    for task in tqdm(tasks, desc="Processing anomalies"):
        gcm, pd_forcing, ff = task
        try:
            result.append(_process_anomaly(task))
        except Exception as exc:
            print(f"Failed {gcm} {ff} vs {pd_forcing}: {exc}")

    elapsed = time.perf_counter() - start
    print(f"Total processing time: {elapsed:.2f} seconds")

    return result


def baseline_with_anomalies(
    baseline_file: str | Path,
    forcing_files: Sequence[str | Path],
    force_overwrite: bool = False,
) -> list[Path]:
    """
    Add baseline climatology to each anomaly forcing file.

    For precipitation and air_temp the baseline values are added to the
    anomaly.  The ``height`` variable is taken from the anomaly file
    unchanged.  Output files are written next to the baseline file with
    a combined name.

    Parameters
    ----------
    baseline_file : str or Path
        Path to the baseline climatology NetCDF file
        (e.g. ``HIRHAM5-ERA5_YMM_1990_2019_v1.nc``).
    forcing_files : list of str or Path
        Anomaly forcing files
        (e.g. ``CESM1-WACCM-SC_anomalies_futSST-pdSIC_pdSST-pdSICSIT_v1.nc``).
    force_overwrite : bool, default False
        If True, regenerate output even if it already exists.

    Returns
    -------
    list of Path
        Paths to the output files.
    """
    baseline_file = Path(baseline_file)
    # Strip version suffix (e.g. _v1) from baseline stem
    baseline_stem = re.sub(r"_v\d+$", "", baseline_file.stem)  # e.g. HIRHAM5-ERA5_YMM_1990_2019
    output_dir = baseline_file.parent

    result = []
    for forcing_file in tqdm(forcing_files, desc="Adding anomalies to baseline"):
        forcing_file = Path(forcing_file)
        forcing_stem = forcing_file.stem  # e.g. CESM1-WACCM-SC_anomalies_futSST-pdSIC_pdSST-pdSICSIT_v1
        output_file = output_dir / f"{baseline_stem}_{forcing_stem}.nc"

        if (not check_xr_lazy(output_file, verbose=False)) or force_overwrite:
            baseline = xr.open_dataset(baseline_file)
            anomaly = xr.open_dataset(forcing_file)

            ds = baseline.copy(deep=True)
            ds["precipitation"] = baseline["precipitation"] + anomaly["precipitation"]
            ds["air_temp"] = baseline["air_temp"] + anomaly["air_temp"]
            ds["height"] = anomaly["height"]

            for var in list(ds.data_vars) + list(ds.coords):
                ds[var].attrs.pop("missing_value", None)
                ds[var].attrs.pop("_FillValue", None)
                ds[var].encoding["missing_value"] = None
                ds[var].encoding["_FillValue"] = None

            ds.to_netcdf(output_file)
            baseline.close()
            anomaly.close()

        result.append(output_file)

    return result
