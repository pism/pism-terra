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

# mypy: disable-error-code="call-overload"
# pylint: disable=unused-import,too-many-positional-arguments,broad-exception-caught


"""
Prepare Heatflow Maps.
"""

import logging
import zipfile
from pathlib import Path

import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.warp import Resampling

from pism_terra.download import download_archive, extract_archive
from pism_terra.workflow import check_xr_lazy

logger = logging.getLogger(__name__)

# Lucazeau (2019), "Analysis and Mapping of an Updated Terrestrial Heat Flow
# Data Set", G-Cubed, doi:10.1029/2019GC008389. The gridded "similarity"
# prediction (``HFgrid14.csv``) ships in Supporting Information S01
# (supplement file ``sup-0003``); the ``sup-0004``/S02 archive is the
# *point* database (``NGHF.csv``), not the map.
HF_URL = (
    "https://agupubs.onlinelibrary.wiley.com/action/downloadSupplement"
    "?doi=10.1029%2F2019GC008389&file=2019GC008389-sup-0003-Data_Set_SI-S01.zip"
)
HF_ARCHIVE_NAME = "2019GC008389-sup-0003-Data_Set_SI-S01.zip"
HF_CSV_NAME = "HFgrid14.csv"


def _find_hfgrid_csv(search_dir: Path) -> Path | None:
    """
    Locate the ``HFgrid14.csv`` grid file anywhere under *search_dir*.

    Parameters
    ----------
    search_dir : pathlib.Path
        Directory searched recursively for the Lucazeau grid CSV.

    Returns
    -------
    pathlib.Path or None
        Path to the first match, or ``None`` if no match is found.
    """
    matches = sorted(search_dir.rglob(HF_CSV_NAME))
    return matches[0] if matches else None


def prepare_heatflux_lucazeau(
    output_path: Path | str,
    extract_path: Path | str,
    force_overwrite: bool = False,
) -> Path:
    """
    Stage the Lucazeau (2019) global heat-flow map as a cloud-optimized Zarr.

    Reads the gridded "similarity" heat-flow prediction (``HFgrid14.csv``),
    reshapes the point list onto its native 0.5-degree latitude-longitude grid,
    attaches CF metadata and an EPSG:4326 CRS, and writes a consolidated
    (cloud-readable) Zarr store.

    The source CSV is obtained in this order: an already-extracted
    ``HFgrid14.csv`` under *extract_path*; any ``.zip`` placed under
    *extract_path* (extracted and searched); otherwise an automatic download
    from Wiley. The Wiley endpoint sits behind a Cloudflare JS challenge that
    plain HTTP clients cannot pass, so for unattended runs download the archive
    manually and drop it in *extract_path*.

    Parameters
    ----------
    output_path : Path or str
        Directory in which the ``heatflux_lucazeau.zarr`` store is written.
    extract_path : Path or str
        Working directory for the downloaded/placed archive and its extracted
        CSV.
    force_overwrite : bool, default False
        If True, rebuild the Zarr even when it already exists (the source CSV is
        reused when present; only re-downloaded if missing).

    Returns
    -------
    pathlib.Path
        Path to the written Zarr store.

    Raises
    ------
    RuntimeError
        If the automatic download does not yield a valid zip (e.g. blocked by
        Cloudflare); the message explains the manual-download fallback.
    FileNotFoundError
        If ``HFgrid14.csv`` cannot be located after download/extraction.

    Notes
    -----
    Heat-flow values are in milliwatts per square metre (``mW m-2``) and are
    kept as published, including the small number of negative / very large
    cells. PISM's ``bheatflx`` expects ``W m-2``; convert (``* 1e-3``) and clamp
    downstream when consuming this field.
    """
    extract_path = Path(extract_path)
    output_path = Path(output_path)
    extract_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    zarr_store = output_path / "heatflux_lucazeau.zarr"
    if zarr_store.exists() and not force_overwrite:
        logger.info("Heat-flow Zarr already exists, skipping: %s", zarr_store)
        return zarr_store

    # 1) already-extracted CSV  2) a pre-placed zip  3) download from Wiley
    csv_path = _find_hfgrid_csv(extract_path)
    if csv_path is None:
        for zip_path in sorted(extract_path.glob("*.zip")):
            if zipfile.is_zipfile(zip_path):
                logger.info("Extracting pre-placed archive %s", zip_path.name)
                extract_archive(zip_path, extract_path / "lucazeau", force_overwrite=force_overwrite, verbose=False)
        csv_path = _find_hfgrid_csv(extract_path)

    if csv_path is None:
        archive_dest = extract_path / HF_ARCHIVE_NAME
        logger.info("Downloading Lucazeau heat-flow map from %s", HF_URL)
        archive = download_archive(HF_URL, dest=archive_dest, force_overwrite=force_overwrite, verbose=True)
        if not zipfile.is_zipfile(archive):
            archive.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the Lucazeau heat-flow archive automatically — the Wiley "
                "endpoint is behind a Cloudflare challenge that HTTP clients cannot pass. "
                f"Manually download\n  {HF_URL}\n"
                f"and place the .zip (or the extracted {HF_CSV_NAME}) in\n  {extract_path}"
            )
        extract_archive(archive, extract_path / "lucazeau", force_overwrite=force_overwrite, verbose=True)
        csv_path = _find_hfgrid_csv(extract_path)

    if csv_path is None:
        raise FileNotFoundError(f"{HF_CSV_NAME} not found under {extract_path} after download/extraction")

    logger.info("Reading heat-flow grid from %s", csv_path)
    # Semicolon-delimited; the published header misspells "longitude" as
    # "longiyude" and leaves Hf_obs empty (NaN) for unobserved cells.
    df = pd.read_csv(csv_path, sep=";").rename(columns={"longiyude": "longitude"})

    # Point list -> regular lat/lon grid (the file is a complete 0.5-degree grid).
    ds = (
        df.set_index(["latitude", "longitude"])[["HF_pred", "sHF_pred", "Hf_obs"]]
        .to_xarray()
        .rename(
            {
                "latitude": "lat",
                "longitude": "lon",
                "HF_pred": "heatflux",
                "sHF_pred": "heatflux_uncertainty",
                "Hf_obs": "heatflux_observed",
            }
        )
        .sortby("lat")
        .sortby("lon")
    )

    ds["lat"].attrs.update(standard_name="latitude", long_name="latitude", units="degrees_north")
    ds["lon"].attrs.update(standard_name="longitude", long_name="longitude", units="degrees_east")
    ds["heatflux"].attrs.update(
        standard_name="upward_geothermal_heat_flux_at_ground_level",
        long_name="predicted surface heat flow (Lucazeau 2019 similarity method)",
        units="mW m-2",
    )
    ds["heatflux_uncertainty"].attrs.update(
        long_name="standard deviation of predicted surface heat flow", units="mW m-2"
    )
    ds["heatflux_observed"].attrs.update(long_name="observed surface heat flow where available", units="mW m-2")
    ds.attrs.update(
        title="Lucazeau (2019) global terrestrial heat-flow map (HFgrid14)",
        references="Lucazeau, F. (2019), doi:10.1029/2019GC008389",
        source=HF_URL,
    )

    ds = (
        ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
        .rio.write_crs("EPSG:4326", inplace=False)
        .rio.write_grid_mapping("spatial_ref", inplace=False)
    )

    # Small global grid -> a single chunk; consolidated metadata makes the store
    # cloud-readable in one metadata request.
    ds = ds.chunk({"lat": ds.sizes["lat"], "lon": ds.sizes["lon"]})
    logger.info("Writing cloud-optimized Zarr to %s", zarr_store)
    ds.to_zarr(zarr_store, mode="w", consolidated=True)

    return zarr_store


def heatflux_from_grid(
    target_grid: xr.Dataset,
    dataset: str = "lucazeau",
    path: Path | str = "tmp.nc",
    bucket: str = "pism-cloud-data",
    prefix: str = "glacier",
    force_overwrite: bool = False,
) -> xr.Dataset:
    """
    Interpolate a staged heat-flow map onto a target grid as PISM ``bheatflx``.

    Opens the cloud-optimized Zarr produced by
    :func:`prepare_heatflux_lucazeau` from
    ``s3://{bucket}/{prefix}/heatflux/heatflux_{dataset}.zarr``, reprojects the
    global field onto *target_grid*, converts to PISM's geothermal-flux units,
    and writes the result to ``path``. A valid cache at ``path`` is reused unless
    ``force_overwrite=True``.

    Parameters
    ----------
    target_grid : xarray.Dataset
        Grid (with a projected CRS in ``spatial_ref``) the heat flow is aligned to.
    dataset : str, default ``"lucazeau"``
        Heat-flow dataset name; selects ``heatflux_{dataset}.zarr`` on S3.
    path : str or pathlib.Path, default ``"tmp.nc"``
        Output/cache NetCDF (e.g. ``bheatflux_{rgi_id}.nc``).
    bucket : str, default ``"pism-cloud-data"``
        S3 bucket holding the staged heat-flow Zarr.
    prefix : str, default ``"glacier"``
        S3 key prefix under *bucket*.
    force_overwrite : bool, default ``False``
        If ``True``, ignore any cache at ``path`` and regenerate.

    Returns
    -------
    xarray.Dataset
        Single-variable dataset ``bheatflx`` (``W m-2``) on *target_grid*.

    Notes
    -----
    The source map is in ``mW m-2``; it is converted to ``W m-2`` (``* 1e-3``)
    and the handful of unphysical negative source cells are clamped to zero so
    the field is a valid PISM geothermal-flux input.
    """
    print("")
    print("Generate Geothermal Heat Flux")
    print("-" * 120)

    if check_xr_lazy(path) and not force_overwrite:
        logger.info("Using cached heat-flow file %s", path)
        return xr.open_dataset(path)

    path = Path(path)
    path.unlink(missing_ok=True)

    uri = f"s3://{bucket}/{prefix}/heatflux/heatflux_{dataset}.zarr".replace("//", "/").replace("s3:/", "s3://")
    logger.info("Opening heat-flow Zarr %s", uri)
    storage_options = {"anon": True} if uri.startswith("s3://") else None
    ds = xr.open_zarr(uri, consolidated=True, storage_options=storage_options, chunks={})

    hf = ds["heatflux"].rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    if hf.rio.crs is None:
        hf = hf.rio.write_crs("EPSG:4326")

    # Interpolate the global 0.5-degree field onto the projected target grid.
    bheatflx = hf.rio.reproject_match(target_grid, resampling=Resampling.bilinear)

    # mW/m^2 -> W/m^2 (PISM `bheatflx` units); clamp unphysical negative cells.
    bheatflx = (bheatflx * 1.0e-3).clip(min=0.0)
    bheatflx.name = "bheatflx"
    bheatflx.attrs = {
        "standard_name": "upward_geothermal_heat_flux_at_ground_level",
        "long_name": "upward geothermal flux at the bottom bedrock surface",
        "units": "W m-2",
        "source": f"Lucazeau (2019) heat-flow map ({dataset})",
    }
    bheatflx = bheatflx.rio.write_crs(target_grid.rio.crs).rio.write_grid_mapping()

    out = bheatflx.to_dataset()
    out.to_netcdf(path, engine="h5netcdf")
    logger.info("Heat flux saved to %s", path)
    return out
