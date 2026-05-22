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
Prepare ISMIP7 Greenland data sets.
"""

import logging
import os
import re
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed as cf_as_completed
from pathlib import Path
from typing import Any, Literal, Sequence

import cf_xarray
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rioxarray  # pylint: disable=unused-import
import toml
import xarray as xr
import xarray_regrid.methods.conservative  # pylint: disable=unused-import
from cdo import Cdo
from dask.distributed import Client, as_completed
from pyfiglet import Figlet
from rasterio.enums import Resampling
from tqdm.auto import tqdm

from pism_terra.domain import create_domain
from pism_terra.download import (
    download_earthaccess,
    download_file,
    download_gebco,
    download_netcdf,
    file_localizer,
)
from pism_terra.raster import create_ds
from pism_terra.vector import dissolve
from pism_terra.workflow import check_xr_fully, check_xr_lazy

xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)


ISMIP7_GLOBUS_BASE = "https://g-ab4495.8c185.08cc.data.globus.org/ISMIP7/GrIS"


def _make_url(year, ice_sheet, gcm, pathway, short_hand, m_var, version):
    """
    Build the Globus HTTPS URL for one ISMIP7 GrIS forcing file.

    Mirrors the public Globus directory layout::

        {base}/{gcm}/{pathway}/{short_hand}/{m_var}/{version}/<file>.nc

    where ``<file>`` is ``{m_var}_{ice_sheet}_{gcm}_{pathway}_{short_hand}_{version}_{year}.nc``
    (without the ``short_hand`` segments when it equals ``"none"``).

    Parameters
    ----------
    year : int
        Year of the forcing file.
    ice_sheet : str
        Ice-sheet identifier embedded in the filename (e.g. ``"GrIS"`` or
        ``"AIS"``).
    gcm : str
        GCM name (e.g. ``"CESM2-WACCM"``).
    pathway : str
        Emissions pathway (e.g. ``"historical"``, ``"ssp585"``).
    short_hand : str
        Short-hand identifier for the forcing type (e.g. ``"SDBN1-1000m"``)
        or ``"none"`` when the variable lives in a per-GCM/pathway tree
        without the short-hand segment.
    m_var : str
        Variable name (e.g. ``"acabf"``, ``"tas"``).
    version : str
        Version string (e.g. ``"v1"``).

    Returns
    -------
    str
        Globus HTTPS URL for the file.
    """
    fname = (
        f"{m_var}_{ice_sheet}_{gcm}_{pathway}_{short_hand}_{version}_{year}.nc"
        if short_hand != "none"
        else f"{m_var}_{ice_sheet}_{gcm}_{pathway}_{version}_{year}.nc"
    )
    parts = [ISMIP7_GLOBUS_BASE, gcm, pathway]
    if short_hand != "none":
        parts.append(short_hand)
    parts.extend([m_var, version, fname])
    return "/".join(parts)


class GlobusAuthRequired(RuntimeError):
    """Globus refused the download because no valid bearer token was supplied."""


def _globus_headers() -> dict[str, str]:
    """
    Build header.

    Build the Authorization header for Globus HTTPS downloads, if a token
    is available. Reads ``GLOBUS_ACCESS_TOKEN`` from the environment.

    Returns
    -------
    dict
        Authorization header.
    """
    token = os.environ.get("GLOBUS_ACCESS_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _download_one(url: str, dest: Path, force_overwrite: bool = False, timeout: int = 600) -> Path:
    """
    Download a single ISMIP7 NetCDF from Globus to ``dest``.

    Streams the response to a temporary ``.part`` file and renames on
    completion so partial downloads never look like valid caches. If
    ``dest`` already exists and ``force_overwrite`` is False, it is
    returned unchanged.

    If ``GLOBUS_ACCESS_TOKEN`` is set in the environment, it is sent as a
    Bearer token in the ``Authorization`` header. If the server tries to
    redirect to ``auth.globus.org`` (i.e. the collection requires login),
    the function aborts with :class:`GlobusAuthRequired` rather than
    chasing the auth flow.

    Parameters
    ----------
    url : str
        Source URL (typically built by :func:`_make_url`).
    dest : Path
        Local target path.
    force_overwrite : bool, default False
        Re-download even if ``dest`` is already present.
    timeout : int, default 600
        Per-request timeout in seconds.

    Returns
    -------
    Path
        ``dest`` (after a successful download or as a cache hit).

    Raises
    ------
    GlobusAuthRequired
        If the request gets redirected to Globus Auth (collection requires
        authenticated access).
    """
    if dest.exists() and not force_overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = _globus_headers()
    with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=False) as r:
        # Follow data-server redirects (g-…data.globus.org → real node) but
        # bail loudly on auth redirects so we don't hammer auth.globus.org.
        while r.is_redirect:
            location = r.headers.get("Location", "")
            if "auth.globus.org" in location:
                raise GlobusAuthRequired(
                    "Globus collection requires authentication. "
                    "Set GLOBUS_ACCESS_TOKEN with a valid Bearer token "
                    "(see notes in pism_terra.ismip7.greenland.forcing)."
                )
            r.close()
            r = requests.get(location, headers=headers, stream=True, timeout=timeout, allow_redirects=False)
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.rename(dest)
    return dest


def _download_many(
    pairs: Sequence[tuple[str, Path]],
    max_workers: int = 2,
    desc: str = "Downloading ISMIP7",
    force_overwrite: bool = False,
) -> list[Path]:
    """
    Download a list of ``(url, dest)`` pairs in parallel.

    Parameters
    ----------
    pairs : sequence of (str, pathlib.Path)
        URLs to fetch and the local destinations to write them to.
    max_workers : int, default 8
        Number of concurrent download workers.
    desc : str, default "Downloading ISMIP7"
        Progress-bar description.
    force_overwrite : bool, default False
        Re-download even if a cached file already exists at the destination.

    Returns
    -------
    list of Path
        Paths to all successfully downloaded files (in completion order).
    """
    results: list[Path] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, url, dest, force_overwrite): (url, dest) for url, dest in pairs}
        pbar = tqdm(cf_as_completed(futures), total=len(futures), desc=desc, unit="file")
        for fut in pbar:
            url, dest = futures[fut]
            try:
                results.append(fut.result())
                pbar.set_postfix_str(f"{dest.name} ✓")
            except GlobusAuthRequired as exc:
                # Don't keep hammering: cancel remaining work and surface a clear error.
                pbar.set_postfix_str("auth required — aborting")
                for pending in futures:
                    pending.cancel()
                raise exc
            except Exception as exc:
                pbar.set_postfix_str(f"{dest.name} ✗")
                logger.error("Failed to download %s: %s", url, exc)
    return results


def _local_path(year, data_path, ice_sheet, gcm, pathway, short_hand, m_var, version):
    """
    Build the local file path for an ISMIP7 forcing variable and year.

    Used when reading from a local mirror of the Globus tree. The on-disk
    layout matches the Globus URL exactly — same subdir hierarchy and same
    filename — so the only flexibility is whether *data_path* already includes
    the trailing ``GrIS`` segment or sits above it.

    Parameters
    ----------
    year : int
        Year of the forcing file.
    data_path : pathlib.Path
        Root of the local mirror; expected to contain a ``<ice_sheet>/``
        subdirectory that mirrors the Globus tree (e.g.
        ``~/storstrommen/ISMIP7/`` for ``ice_sheet="GrIS"``).
    ice_sheet : str
        Ice-sheet subdirectory under *data_path* and segment of the local
        filename (``"GrIS"`` or ``"AIS"``).
    gcm : str
        GCM name (e.g. ``"CESM2-WACCM"``).
    pathway : str
        Emissions pathway (e.g. ``"historical"``, ``"ssp585"``).
    short_hand : str
        Short-hand identifier for the forcing type, or ``"none"``.
    m_var : str
        Variable name (e.g. ``"acabf"``, ``"tas"``).
    version : str
        Version string (e.g. ``"v1"``).

    Returns
    -------
    pathlib.Path
        Path to the file under *data_path* (or under *data_path/GrIS* if that
        is where the file actually lives). When neither candidate exists, the
        first candidate is returned so callers can surface a sensible error.
    """
    fname = (
        f"{m_var}_{ice_sheet}_{gcm}_{pathway}_{short_hand}_{version}_{year}.nc"
        if short_hand != "none"
        else f"{m_var}_{ice_sheet}_{gcm}_{pathway}_{version}_{year}.nc"
    )
    rel_parts = [gcm, pathway]
    if short_hand != "none":
        rel_parts.append(short_hand)
    rel_parts.extend([m_var, version, fname])
    rel = Path(*rel_parts)
    return data_path / ice_sheet / rel


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
    ice_sheet: Literal["AIS", "GrIS"],
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
    data_path: Path | None = None,
) -> list[Path]:
    """
    Process a single GCM/forcing combination.

    Parameters
    ----------
    ice_sheet : {"AIS", "GrIS"}
        Ice-sheet identifier; selects the subtree under the Globus base or
        local mirror and is embedded in source filenames.
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
    data_path : pathlib.Path or None, optional
        If given, read forcing files from this local mirror of the Globus
        tree instead of downloading from ``base_path``. Files are looked up
        with their Globus filename under ``data_path`` (or under
        ``data_path/GrIS``). When ``None`` (default), the function downloads
        from Globus and stores files under ``base_path``.

    Returns
    -------
    list[Path]
        Paths to the historical, projection, and merged NetCDF files.
    """
    os.environ["HDF5_LOG_LEVEL"] = "0"
    cdo = Cdo()
    cdo.debug = True

    grid_file = file_localizer("s3://pism-cloud-data/ismip7_extra/grid.txt", dest=output_path)
    tas_replace = ""

    output_files = []

    def _resolve(year: int, pathway_name: str, m_var: str) -> Path:
        """
        Return the local file path for one (year, pathway, var).

        Parameters
        ----------
        year : int
            Calendar year of the requested forcing slice.
        pathway_name : str
            Emissions pathway segment of the path (e.g. ``"historical"``).
        m_var : str
            ISMIP7 variable name (e.g. ``"acabf"``).

        Returns
        -------
        pathlib.Path
            Path under ``data_path`` (Globus-mirror filename) when
            ``data_path`` was supplied, else the legacy ``_make_path``
            location under ``base_path``.
        """
        if data_path is not None:
            return _local_path(year, data_path, ice_sheet, gcm, pathway_name, short_hand, m_var, version)
        # _make_path doesn't take ice_sheet (the segment isn't part of the legacy
        # base_path layout).
        return Path(_make_path(year, base_path, gcm, pathway_name, short_hand, m_var, version))

    if data_path is None:
        # Build (url, local_path) pairs for every (variable, year) we need,
        # across both historical and projection epochs. Then download
        # everything in parallel before any cdo step touches the files.
        download_pairs: list[tuple[str, Path]] = []
        for pathway_name, start_year, end_year in (
            ("historical", hist_start_year, hist_end_year),
            (pathway, proj_start_year, proj_end_year),
        ):
            for m_var in fields:
                for year in range(start_year, end_year):
                    url = _make_url(year, ice_sheet, gcm, pathway_name, short_hand, m_var, version)
                    download_pairs.append((url, _resolve(year, pathway_name, m_var)))

        _download_many(download_pairs, desc=f"Download {gcm}/{forcing}")
    else:
        logger.info("Using local ISMIP7 forcing under %s for %s/%s", data_path, gcm, forcing)

    # cdo merges run on the resolved local paths (downloaded or pre-existing).
    # Doing the per-variable mergetime in-process (alongside the final
    # cross-variable merge) produced one shell invocation listing every
    # (variable, year) source file. With ~6 variables × ~300 years that
    # easily exceeds ARG_MAX. Split the work: per-variable mergetime/chname
    # writes a tmp file; the final per-epoch cdo only sees one tmp per var.
    import tempfile  # pylint: disable=import-outside-toplevel

    def _merge_one_var(
        tmp_root: Path, epoch_label: str, pathway_name: str, start_year: int, end_year: int, m_var: str
    ) -> Path:
        """
        Mergetime + chname for one (epoch, variable) into a tmp NetCDF.

        Splitting the per-variable merge off the final cross-variable merge
        keeps any one ``cdo`` invocation well under the ARG_MAX limit, even
        for projection epochs that span hundreds of years.

        Parameters
        ----------
        tmp_root : pathlib.Path
            Directory for the per-variable tmp output.
        epoch_label : str
            Short label embedded in the tmp filename (e.g. ``"hist"`` or
            ``"proj"``) to keep epochs distinct in the same tmp dir.
        pathway_name : str
            Emissions pathway segment of the source path (e.g.
            ``"historical"`` or ``"ssp585"``).
        start_year, end_year : int
            Inclusive/exclusive year range for the source files.
        m_var : str
            ISMIP7 variable name (e.g. ``"acabf"``).

        Returns
        -------
        pathlib.Path
            Path to the per-variable tmp NetCDF.
        """
        paths = [_resolve(year, pathway_name, m_var) for year in range(start_year, end_year)]
        k, v = m_var, ismip7_to_pism[m_var]
        out = tmp_root / f"{epoch_label}_{m_var}.nc"
        cdo.chname(
            f"{k},{v}",
            input=(f"{tas_replace} -setgrid,{str(grid_file)} -mergetime [ " + " ".join(str(p) for p in paths) + " ]"),
            output=str(out.resolve()),
            options="-f nc4 -z zip_2",
        )
        return out

    hist_output_file = output_path / Path(
        f"ismip7_greenland_{forcing}_historical_{gcm}_{version}_{hist_start_year}_{hist_end_year}.nc"
    )
    proj_output_file = output_path / Path(
        f"ismip7_greenland_{forcing}_{pathway}_{gcm}_{version}_{proj_start_year}_{proj_end_year}.nc"
    )

    with tempfile.TemporaryDirectory(prefix=f"_ismip7_{gcm}_{forcing}_", dir=str(output_path)) as _tmp:
        tmp_root = Path(_tmp)
        hist_tmp = [
            _merge_one_var(tmp_root, "hist", "historical", hist_start_year, hist_end_year, m_var) for m_var in fields
        ]
        proj_tmp = [
            _merge_one_var(tmp_root, "proj", pathway, proj_start_year, proj_end_year, m_var) for m_var in fields
        ]

        cdo.setmisstoc(
            0,
            input=(
                f"-setgrid,{str(grid_file)} -settbounds,{freq} "
                f"-setreftime,1850-01-01 -settunits,hours -setcalendar,{calendar} "
                f"-settaxis,'{hist_start_year}-01-16 12:00,,{freq}' -merge "
                + " ".join(str(p.resolve()) for p in hist_tmp)
            ),
            output=str(hist_output_file.resolve()),
            options="-f nc4 -z zip_2",
        )
        output_files.append(hist_output_file)

        cdo.setmisstoc(
            0,
            input=(
                f"-setgrid,{str(grid_file)} -settbounds,{freq} "
                f"-setreftime,1850-01-01 -settunits,hours -setcalendar,{calendar} "
                f"-settaxis,'{proj_start_year}-01-16 12:00,,{freq}' -merge "
                + " ".join(str(p.resolve()) for p in proj_tmp)
            ),
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
    url: Path | str,
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

    # ``url`` may be a Path (when reading from a local mirror) or a string
    # (when downloading from Globus). Normalize to str for split/download.
    url_str = str(url)
    name = url_str.rsplit("/", maxsplit=1)[-1]
    obs_file = Path(input_path) / Path(name)
    if (not check_xr_lazy(obs_file)) or force_overwrite:
        ds_bm = download_netcdf(url_str)
    else:
        ds_bm = xr.open_dataset(obs_file)

    ds_bm = ds_bm.rename_vars({"surface_grimp": "surface"})
    ds_bm["surface"].attrs.update(
        {"standard_name": "surface_altitude", "long_name": "ice surface elevation", "units": "m"}
    )

    if target_grid is not None:
        ds_bm_regridded = ds_bm[["bed", "thickness", "surface", "mask"]].regrid.conservative(target_grid)
        gebco_p = download_gebco(target_dir=input_path)
        gebco = xr.open_dataset(gebco_p, chunks="auto").rio.write_crs("EPSG:4326")
        gebco_bm_regridded = gebco.rio.reproject_match(
            ds_bm_regridded.rio.write_crs("EPSG:3413"), resampling=Resampling.bilinear
        ).compute()
        ds_bm_regridded["bed"] = ds_bm_regridded["bed"].where(
            ds_bm_regridded["bed"].notnull(), gebco_bm_regridded["elevation"]
        )
        ds_bm_regridded = ds_bm_regridded.fillna(0)
    else:
        ds_bm_regridded = ds_bm

    ftt_mask = xr.where(ds_bm_regridded["thickness"] > 0, 1, 0)
    ftt_mask.name = "ftt_mask"

    liafr = xr.where(ds_bm_regridded["mask"] == 0, 0, 1)
    liafr.name = "land_ice_area_fraction_retreat"
    liafr.attrs.update({"units": "1"})
    liafr = liafr.astype("bool")

    if surface_dem is not None:
        surface_file = Path(input_path) / Path("surface_dem.nc")
        bed = ds_bm_regridded["bed"]
        if (not check_xr_lazy(surface_file)) or force_overwrite:
            ds = download_netcdf(surface_dem)
            ds.to_netcdf(surface_file)
        else:
            ds = xr.open_dataset(surface_file)
        surface = ds["surface"].regrid.conservative(target_grid)
        surface.name = "surface"
        thickness = xr.where(surface > 0, surface - bed, 0)
        thickness = thickness.where(thickness > 10, 0)
        thickness.name = "thickness"
        thickness.attrs.update(ds_bm_regridded["thickness"].attrs)
        boot = xr.merge([bed, ftt_mask, surface, thickness, liafr])
    else:
        boot = xr.merge([ds_bm_regridded[["bed", "thickness", "surface"]], ftt_mask, liafr])
    boot = boot.fillna(0)
    ds = xr.merge([boot, ds_bm["mapping"]])

    geo = (
        ds_bm[["geothermal_heat_flux1"]]
        .rename_dims({"x1km": "x", "y1km": "y"})
        .rename_vars({"x1km": "x", "y1km": "y", "geothermal_heat_flux1": "bheatflx"})
        .regrid.conservative(target_grid)
    )
    geo = geo.where(geo != -9999, 0.042)

    ds["surface"].attrs.update({"standard_name": "surface_altitude", "units": "m"})
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
    ds.to_netcdf(obs_file, encoding=encoding, engine="h5netcdf")

    geo["bheatflx"].attrs.pop("coordinates", None)
    geo["bheatflx"].encoding.pop("coordinates", None)
    geo = geo.drop_vars("spatial_ref", errors="ignore")
    geo["mapping"] = ds_bm["mapping"]
    for v in geo.data_vars:
        geo[v].attrs.pop("coordinates", None)
        geo[v].encoding.pop("coordinates", None)
    geo_file = output_path / Path(f"heatflux_g{resolution}m_GreenlandObsISMIP7-v1.3.nc")
    geo_encoding = {var: {"_FillValue": None} for var in list(geo.data_vars) + list(geo.coords)}
    geo.to_netcdf(geo_file, encoding=geo_encoding, engine="h5netcdf")

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
    data_path: Path | str | None = None,
    n_workers: int = 2,
) -> Sequence[Path | str]:
    """
    Process forcing data for all GCMs and forcings in parallel.

    Parameters
    ----------
    base_path : Path or str
        Base path (or URL) to the remote ISMIP7 forcing tree. Used only when
        ``data_path`` is ``None``; otherwise downloads are skipped entirely.
    output_path : Path or str
        Output directory.
    config : dict
        Configuration dictionary.
    data_path : Path or str or None, optional
        If given, read forcing files from this local mirror of the Globus
        tree instead of downloading. Layout is expected to match the
        Globus tree, with *data_path* either containing a ``GrIS/`` subdir
        or being that ``GrIS/`` directory itself.
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
    if data_path is not None:
        data_path = Path(data_path)

    ismip7_to_pism = config["ismip7_to_pism"]
    # Build list of tasks
    tasks = []

    for gcm, _gcm_config in config["gcms"].items():
        for pathway, _pathway_config in _gcm_config.items():
            ice_sheet = config["ice_sheet"]
            version = "v" + str(_pathway_config["version"])
            hist_start_year, hist_end_year = _pathway_config["historical"]
            proj_start_year, proj_end_year = _pathway_config["projection"]
            for forcing, forcing_dict in config["forcing"].items():
                short_hand = forcing_dict["short_hand"]
                fields = forcing_dict["fields"]
                tasks.append(
                    (
                        ice_sheet,
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
            ice_sheet,
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
                ice_sheet,
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
                data_path=data_path,
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
