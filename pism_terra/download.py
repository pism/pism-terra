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

# pylint: disable=consider-using-with,too-many-positional-arguments,broad-exception-caught
"""
Module for downloading data.
"""

from __future__ import annotations

import logging
import os
import re
import tarfile
import tempfile
import time
import zipfile
from collections.abc import Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

import boto3
import earthaccess
import numpy as np
import requests
import xarray as xr
from ecmwf.datastores import Client as _DatastoresClient
from ecmwf.datastores import Remote as _DatastoresRemote
from tqdm.auto import tqdm

from pism_terra.aws import download_from_s3
from pism_terra.workflow import check_xr_lazy

# Silence noisy third-party INFO chatter that interleaves with tqdm bars.
for _name in ("cdsapi", "datapi", "multiurl", "ecmwf", "botocore", "s3transfer", "boto3"):
    logging.getLogger(_name).setLevel(logging.WARNING)


class FileInfo(NamedTuple):
    """
    FileInfo class.
    """

    variable: str
    units: str
    month: str
    year: str


def parse_filename(path: str) -> FileInfo:
    """
    Extract variable name, units, month, and year from a climate file path.

    The filename is expected to follow a structure like:
    ``archive/<var>/<var>_<descriptor>_<units>_<source>_<experiment>_<month>_<year>.tif``

    Parameters
    ----------
    path : str
        Full path or filename (e.g., ``"archive/pr/pr_total_mm_CRU_TS40_historical_01_1901.tif"``).

    Returns
    -------
    FileInfo
        Named tuple containing:
        - ``variable`` : str
          Variable short name (e.g., ``"pr"``)
        - ``units`` : str
          Units substring (e.g., ``"mm"``)
        - ``month`` : str
          Two-digit month string (e.g., ``"01"``)
        - ``year`` : str
          Four-digit year string (e.g., ``"1901"``)

    Raises
    ------
    ValueError
        If the filename does not match the expected pattern.

    Examples
    --------
    >>> parse_filename("archive/pr/pr_total_mm_CRU_TS40_historical_01_1901.tif")
    FileInfo(variable='pr', units='mm', month='01', year='1901')
    """
    pattern = (
        r".*/(?P<variable>[a-zA-Z0-9]+)/"
        r"(?P=variable)_[A-Za-z0-9_]*_(?P<units>[A-Za-z]+)_[A-Za-z0-9_]+_"
        r"(?P<month>\d{2})_(?P<year>\d{4})\.tif$"
    )

    m = re.match(pattern, path)
    if not m:
        raise ValueError(f"Filename pattern not recognized: {path}")
    return FileInfo(**m.groupdict())


def unzip_files(
    files=list[str | Path],
    output_dir: str | Path = ".",
    overwrite: bool = False,
    max_workers: int = 4,
) -> list[Path]:
    """
    Unzip files in parallel.

    Parameters
    ----------
    files : list[Union[str, Path]]
        List of file paths to unzip.
    output_dir : Union[str, Path], optional
        The directory where the unzipped files will be saved, by default ".".
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    max_workers : int, optional
        The maximum number of threads to use for unzipping, by default 4.

    Returns
    -------
    list[Path]
        List of paths to the unzipped files.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(unzip_file, f, str(output_dir), overwrite=overwrite, verbose=False): f for f in files
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Extracting archives", unit="file")
        for future in pbar:
            try:
                future.result()
                pbar.set_postfix_str(f"{Path(futures[future]).stem} ✓")
            except (IOError, ValueError) as e:
                pbar.set_postfix_str(f"{Path(futures[future]).stem} ✗")
                print(f"An error occurred: {e}")

    responses = list(Path(output_dir).rglob("*.nc"))
    return responses


def unzip_file(zip_path: str, extract_to: str, overwrite: bool = False, verbose: bool = True) -> None:
    """
    Unzip a file to a specified directory with a progress bar and optional overwrite.

    Parameters
    ----------
    zip_path : str
        The path to the ZIP file.
    extract_to : str
        The directory where the contents will be extracted.
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    verbose : bool, default True
        If True, show a per-file progress bar.
    """
    # Ensure the extract_to directory exists
    Path(extract_to).mkdir(parents=True, exist_ok=True)

    # Open the ZIP file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Get the list of file names in the zip file
        file_list = zip_ref.namelist()

        # Iterate over the file names with a progress bar
        for file in tqdm(file_list, desc="Extracting files", unit="file", disable=not verbose):
            file_path = Path(extract_to) / file
            if not file_path.exists() or overwrite:
                zip_ref.extract(member=file, path=extract_to)


def extract_archive(
    archive: tarfile.TarFile | zipfile.ZipFile | str | Path,
    extract_to: str | Path = Path("archive"),
    force_overwrite: bool = False,
    verbose: bool = True,
) -> list[str]:
    """
    Extract a ZIP or TAR archive to a specified directory with a progress bar.

    Supports `.zip`, `.tar`, `.tar.gz`, `.tgz`, and other common formats. Files
    are extracted to the specified directory. Existing files are skipped unless
    `overwrite` is True.

    Parameters
    ----------
    archive : str, Path, tarfile.TarFile, or zipfile.ZipFile
        Path to the archive file or an already opened archive object.
    extract_to : str or Path, optional
        Directory to extract the archive contents into. Defaults to "archive".
    force_overwrite : bool, optional
        Whether to overwrite existing files. Defaults to False.
    verbose : bool, optional
        Show progress bar during extraction. Defaults to True.

    Returns
    -------
    list of str
        List of paths (as strings) to the extracted files.

    Raises
    ------
    ValueError
        If the archive format is not supported or the file is invalid.

    Notes
    -----
    - Uses `tqdm` for a progress bar when *verbose* is True.
    - Automatically creates the `extract_to` directory if needed.
    - Automatically closes the archive if opened internally.
    """
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    opened_internally = False

    if isinstance(archive, (str, Path)):
        archive_path = Path(archive)
        if archive_path.suffix == ".zip":
            archive = zipfile.ZipFile(archive_path, "r")
            opened_internally = True
        elif archive_path.suffix in [".tar", ".gz", ".tgz", ".bz2", ".xz"]:
            archive = tarfile.open(archive_path, "r:*")
            opened_internally = True
        else:
            raise ValueError(f"Unsupported archive type: {archive_path}")

    extracted_files = []

    members: str | list[str] | list[zipfile.ZipInfo] | list[tarfile.TarInfo]
    if isinstance(archive, zipfile.ZipFile):
        members = archive.namelist()
        for member in tqdm(members, desc="Extracting files", unit="file", disable=not verbose):
            file_path = extract_to / member
            if (not file_path.exists()) or force_overwrite:
                archive.extract(member, path=extract_to)
            extracted_files.append(str(file_path))

    elif isinstance(archive, tarfile.TarFile):
        members = archive.getmembers()
        for member in tqdm(members, desc="Extracting files", unit="file", disable=not verbose):
            file_path = extract_to / member.name
            if (not file_path.exists()) or force_overwrite:
                archive.extract(member, path=extract_to)
            extracted_files.append(str(file_path))

    else:
        raise ValueError(f"Unsupported archive object: {type(archive)}")

    if opened_internally:
        archive.close()

    return extracted_files


def _cds_year_cache_path(dataset: str, request: dict, year: str, dest: Path, suffix: str = ".nc") -> Path:
    """
    Build a collision-resistant cache filename for a per-year CDS download.

    Parameters
    ----------
    dataset : str
        CDS dataset identifier (used in the filename).
    request : dict
        CDS request dict; only ``request["variable"]`` is read, to avoid
        cache collisions when different variables are requested for the
        same dataset/year.
    year : str
        Four-digit year string.
    dest : Path
        Directory the cache file lives in.
    suffix : str, default ``".nc"``
        File extension for the cached download (including the leading dot).
        Use ``".grib"`` when ``request["format"]`` (or ``"data_format"``) is
        ``"grib"`` so the on-disk extension matches the actual content.

    Returns
    -------
    Path
        Path of the form ``<dest>/_cds_<dataset>_<variables>_<year><suffix>``.
    """
    var_key = "_".join(sorted(request.get("variable", [])))
    return dest / f"_cds_{dataset}_{var_key}_{year}{suffix}"


def _cds_finish_year(remote: _DatastoresRemote, year: str, dest: Path, nc_path: Path) -> Path:
    """
    Wait on a previously-submitted CDS job and download its result.

    Parameters
    ----------
    remote : ecmwf.datastores.Remote
        Job handle returned by :meth:`Client.submit`.
    year : str
        Four-digit year string (used for filenames and zip extraction folder).
    dest : Path
        Directory in which to write the downloaded file.
    nc_path : Path
        Final cache path used when the server returns the file directly. The
        suffix of this path (``.nc`` or ``.grib``) is also used to filter
        zip-archive contents.

    Returns
    -------
    Path
        Path to the resulting file (extracted from a ZIP if needed).
    """
    results = remote.get_results()  # blocks until the job is finished

    if results.content_type == "application/zip":
        dl_path = dest / f"_cds_{year}.zip"
    else:
        dl_path = nc_path

    results.download(str(dl_path))

    if str(dl_path).endswith(".zip"):
        extracted = extract_archive(dl_path, extract_to=dest / f"_cds_{year}", force_overwrite=True, verbose=False)
        want_suffix = nc_path.suffix.lower()
        matching = [p for p in extracted if str(p).lower().endswith(want_suffix)]
        if matching:
            return Path(matching[0])
        raise FileNotFoundError(f"No '{want_suffix}' files found in archive for year {year}")

    return nc_path


def _cds_download_years(
    client: _DatastoresClient,
    dataset: str,
    request: dict,
    years: Sequence[str],
    dest: Path,
    force_overwrite: bool = False,
    max_workers: int = 5,
    desc: str = "Downloading years",
    verbose: bool = True,
    suffix: str = ".nc",
) -> list[Path]:
    """
    Download many CDS years using a submit-all-then-download pattern.

    All requests are submitted up front (cheap HTTP POSTs) so the server
    queues them concurrently. Results are then waited on and downloaded in
    parallel via a thread pool of size ``max_workers``.

    Parameters
    ----------
    client : ecmwf.datastores.Client
        Authenticated ECMWF Data Stores client.
    dataset : str
        CDS dataset identifier.
    request : dict
        Base CDS request dict (without ``year``).
    years : sequence of str
        Years to retrieve.
    dest : Path
        Directory in which to write per-year NetCDF files.
    force_overwrite : bool, default False
        Re-download even if a cached file already exists.
    max_workers : int, default 5
        Maximum number of concurrent download/wait workers.
    desc : str, default "Downloading years"
        Progress-bar description.
    verbose : bool, default True
        If True, show submission/heartbeat progress and CDS request IDs.
        Set False for quieter logs.
    suffix : str, default ``".nc"``
        File extension applied to per-year cache files. Pass ``".grib"`` when
        the CDS request asks for GRIB so the on-disk extension is honest.

    Returns
    -------
    list of Path
        Per-year cache paths (in submission order). Failed years are logged
        and omitted.
    """
    # Phase 1: submit (or reuse cached) — sequential, no waiting
    pending: list[tuple[str, Path, _DatastoresRemote]] = []  # (year, nc_path, remote)
    cached: list[Path] = []
    submit_pbar = tqdm(years, desc="Submitting", unit="yr", disable=not verbose)
    for yr in submit_pbar:
        nc_path = _cds_year_cache_path(dataset, request, yr, dest, suffix=suffix)
        if nc_path.exists() and not force_overwrite:
            cached.append(nc_path)
            submit_pbar.set_postfix_str(f"{yr} cached")
            continue
        remote = client.submit(dataset, {**request, "year": [yr]})
        pending.append((yr, nc_path, remote))
        submit_pbar.set_postfix_str(f"{yr} → {remote.request_id}")

    if verbose and pending:
        ids = ", ".join(f"{yr}={r.request_id}" for yr, _, r in pending)
        tqdm.write(f"Submitted {len(pending)} CDS jobs (cached={len(cached)}): {ids}")

    if not pending:
        return cached

    # Phase 2: wait + download in parallel. Periodically poll job status
    # so something prints even when CDS is slow.
    downloaded: list[Path] = list(cached)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_cds_finish_year, remote, yr, dest, nc_path): yr for yr, nc_path, remote in pending}
        pbar = tqdm(total=len(futures), desc=desc, unit="yr")
        remaining = dict(futures)  # future -> year
        last_status_log = 0.0
        while remaining:
            done, _ = wait(list(remaining), timeout=60.0, return_when=FIRST_COMPLETED)
            for future in done:
                yr = remaining.pop(future)
                try:
                    downloaded.append(future.result())
                    pbar.set_postfix_str(f"{yr} done")
                except Exception as exc:
                    pbar.set_postfix_str(f"{yr} failed")
                    tqdm.write(f"Failed to download year {yr}: {exc}")
                pbar.update(1)
            # Heartbeat: every ~60 s with no completions, print job statuses
            if not done and verbose:
                now = time.time()
                if now - last_status_log > 60:
                    last_status_log = now
                    statuses = {}
                    for yr in remaining.values():
                        # The (yr, nc_path, remote) tuple in pending mirrors futures' order
                        try:
                            remote = next(r for y, _, r in pending if y == yr)
                            statuses[yr] = remote.status
                        except Exception:
                            statuses[yr] = "?"
                    summary = ", ".join(f"{yr}={s}" for yr, s in sorted(statuses.items()))
                    tqdm.write(f"[CDS heartbeat] {summary}")
        pbar.close()
    return downloaded


def carra_download_request(
    dataset: str,
    request: dict,
    file_path: Path | str = "tmp.nc",
    force_overwrite: bool = False,
    max_workers: int = 5,
) -> xr.Dataset:
    """
    Download CARRA-style reanalysis data from CDS and return the file paths.

    The ``request`` dict is used verbatim as the CDS request. Requests are
    split by year and submitted concurrently (up to ``max_workers`` in
    parallel) so the CDS queue processes them faster.

    Parameters
    ----------
    dataset : str
        CDS dataset identifier to retrieve.
    request : dict
        CDS request used verbatim. The ``year`` key will be split for
        parallel download. Useful for CARRA or other datasets with different
        request schemas.
    file_path : str or pathlib.Path, default ``"tmp.nc"``
        Cache file. If it exists and opens successfully, it is re-used unless
        ``force_overwrite`` is set.
    force_overwrite : bool, default ``False``
        If ``True``, ignore any existing cache at ``path`` and perform a fresh
        download.
    max_workers : int, default 5
        Maximum number of concurrent CDS requests.

    Returns
    -------
    list of pathlib.Path
        Paths to the downloaded NetCDF files (one per year).

    Raises
    ------
    Exception
        CDS request/authentication/parameter failures (raised by
        :mod:`ecmwf.datastores`).
    OSError
        Problems opening/writing downloaded files.
    ValueError
        Incompatible files for merge.

    Notes
    -----
    - Requires a valid CDS API key in ``~/.ecmwfdatastoresrc``
      (or ``ECMWF_DATASTORES_URL`` / ``ECMWF_DATASTORES_KEY`` env vars).
    - If CDS provides a ZIP, contents are extracted before loading/merging.
    """

    file_path = Path(file_path)
    suffix = file_path.suffix or ".nc"

    client = _DatastoresClient()

    path = file_path.parent
    carra2_path = path / Path("_".join(v for v in request["variable"]))
    carra2_path.mkdir(exist_ok=True)

    file_path.unlink(missing_ok=True)

    years = [str(y) for y in request.pop("year")]
    # Remove "year" from the base request; each worker adds its own.

    return _cds_download_years(
        client,
        dataset,
        request,
        years,
        carra2_path,
        force_overwrite=force_overwrite,
        max_workers=max_workers,
        suffix=suffix,
    )


def download_request(
    dataset: str = "reanalysis-era5-single-levels-monthly-means",
    area: Sequence[float] | None = (90.0, -90.0, 45.0, 90.0),
    year: Iterable[int] = range(1980, 2025),
    variable: Sequence[str] = ("2m_temperature", "total_precipitation"),
    file_path: Path | str = "tmp.nc",
    force_overwrite: bool = False,
    request_override: dict | None = None,
    max_workers: int = 5,
    **kwargs,
) -> xr.Dataset:
    """
    Download reanalysis data from CDS and return it as an xarray Dataset.

    By default, sends a request to the Copernicus Climate Data Store (CDS)
    API for monthly ERA5 averages. For other datasets (e.g., CARRA), pass a
    fully formed ``request_override`` dict — it will be used as-is, ignoring
    ``area``, ``year``, and ``variable``.

    Requests are split by year and submitted concurrently (up to
    ``max_workers`` in parallel) so the CDS queue processes them faster.

    Parameters
    ----------
    dataset : str, default ``"reanalysis-era5-single-levels-monthly-means"``
        CDS dataset identifier to retrieve.
    area : sequence of float or None, default ``(90, -90, 45, 90)``
        Geographic bounding box **[North, West, South, East]** in degrees (WGS84).
        Ignored when ``request_override`` is provided.

    year : iterable of int, default ``range(1980, 2025)``
        Years to request. Ignored when ``request_override`` is provided.
    variable : sequence of str, default ``("2m_temperature", "total_precipitation")``
        Variable names to download. Ignored when ``request_override`` is provided.
    file_path : str or pathlib.Path, default ``"tmp.nc"``
        Cache file. If it exists and opens successfully, it is re-used unless
        ``force_overwrite`` is set.
    force_overwrite : bool, default ``False``
        If ``True``, ignore any existing cache at ``path`` and perform a fresh
        download.
    request_override : dict or None, optional
        If provided, used as the CDS request verbatim, replacing the default
        ERA5 request. The ``year`` key will be split for parallel download.
        Useful for CARRA or other datasets with different request schemas.
    max_workers : int, default 5
        Maximum number of concurrent CDS requests.
    **kwargs
        Additional keyword arguments for compatibility. Currently not used.

    Returns
    -------
    xarray.Dataset
        Dataset containing the requested data.

    Raises
    ------
    Exception
        CDS request/authentication/parameter failures (raised by
        :mod:`ecmwf.datastores`).
    OSError
        Problems opening/writing downloaded files.
    ValueError
        Incompatible files for merge.

    Notes
    -----
    - Requires a valid CDS API key in ``~/.ecmwfdatastoresrc``
      (or ``ECMWF_DATASTORES_URL`` / ``ECMWF_DATASTORES_KEY`` env vars).
    - If CDS provides a ZIP, contents are extracted before loading/merging.
    """

    _ = kwargs

    file_path = Path(file_path)

    if request_override is not None:
        request = request_override
    else:
        request = {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": list(variable),
            "year": [str(y) for y in year],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        if area is not None:
            request["area"] = list(area)

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=False)

    if (not check_xr_lazy(file_path)) or force_overwrite:
        client = _DatastoresClient()

        path = file_path.parent
        file_path.unlink(missing_ok=True)

        years = [str(y) for y in request.pop("year")]
        # Remove "year" from the base request; each call adds its own.

        downloaded = _cds_download_years(
            client,
            dataset,
            request,
            years,
            path,
            force_overwrite=force_overwrite,
            max_workers=max_workers,
        )

        dss = []
        for nc in sorted(downloaded):
            ds_part = xr.open_dataset(nc, decode_times=time_coder, decode_timedelta=True)
            if "valid_time" in ds_part.coords:
                ds_part["valid_time"] = ds_part["valid_time"].dt.floor("D")
            dss.append(ds_part)

        ds = xr.merge(dss).drop_vars(["number", "expver"], errors="ignore")

        if "latitude" in ds.coords:
            ds = ds.sortby("latitude")
            ds["latitude"].attrs["stored_direction"] = "increasing"
            ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
            ds.rio.write_crs("EPSG:4326", inplace=True)

        ds.to_netcdf(file_path)
    else:
        ds = xr.open_dataset(file_path, decode_times=time_coder, decode_timedelta=True)
        if "latitude" in ds.coords:
            ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
            ds.rio.write_crs("EPSG:4326", inplace=True)

    return ds


def save_netcdf(
    ds: xr.Dataset,
    output_filename: str | Path = "output.nc",
    comp: dict = {"zlib": True, "complevel": 2},
    **kwargs,
):
    """
    Save the xarray dataset to a NetCDF file with specified compression.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be saved.
    output_filename : str or Path, optional
        The output filename for the NetCDF file.
    comp : dict, optional
        Compression settings for numerical variables.
    **kwargs
        Additional keyword arguments passed to xarray.Dataset.to_netcdf.
    """
    valid_encoding_keys = {
        "zlib",
        "complevel",
        "shuffle",
        "fletcher32",
        "contiguous",
        "chunksizes",
        "dtype",
        "endian",
        "least_significant_digit",
        "_FillValue",
    }

    encoding = {}

    for var in ds.data_vars:
        if np.issubdtype(ds[var].dtype, np.number):
            enc = ds[var].encoding.copy() if hasattr(ds[var], "encoding") else {}
            # Clean and apply compression
            enc = {k: v for k, v in enc.items() if k in valid_encoding_keys}
            enc.update(comp)
            encoding[var] = enc

    ds.to_netcdf(output_filename, encoding=encoding, engine="h5netcdf", **kwargs)


def download_archive(
    url: str, dest: Path | str | None = None, force_overwrite: bool = False, verbose: bool = True
) -> Path:
    """
    Download an archive file from a URL and save it to disk.

    If *dest* already exists and *force_overwrite* is ``False`` the download
    is skipped and the existing path is returned immediately.

    Parameters
    ----------
    url : str
        The URL of the archive file to download.
    dest : Path or str or None, optional
        Local file path for the downloaded archive.  When ``None`` the
        filename is derived from the URL and placed in the current directory.
    force_overwrite : bool, optional
        Re-download even when *dest* already exists.  Defaults to ``False``.
    verbose : bool, optional
        Show progress bar and status messages. Defaults to ``True``.

    Returns
    -------
    Path
        Path to the downloaded archive file on disk.
    """
    if dest is None:
        dest = Path(Path(urlparse(url).path).name)
    else:
        dest = Path(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force_overwrite:
        if verbose:
            print(f"Archive already exists, skipping download: {dest}")
        return dest

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("Content-Length", 0))

    with (
        open(dest, "wb") as f,
        tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {dest.name}",
            disable=not verbose,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    return dest


def file_localizer(file_path: str, dest: str | Path = Path.cwd()) -> Path:
    """
    Localize files to the ``dest`` directory if the don't already exist on the local filesystem.

    This function will ensure files are available in a local directory, either by downloading the HTTP/S3 file, or
    finding an appropriate file bundled with the pism-terra package.

    Parameters
    ----------
    file_path : str
        URI, local path, or path within pism-terra to a file.
    dest : str or Path, optional
        If a file is localized, place it in this directory. Defaults to the current working directory.

    Returns
    -------
    Path
        Localized file path.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if Path(file_path).exists():
        return Path(file_path).resolve()
    elif (package_path := Path(__file__).parent / Path(file_path)).exists():
        return package_path.resolve()
    elif file_path.startswith("s3://"):
        return download_from_s3(file_path, dest / Path(file_path).name)
    elif file_path.startswith("https://") or file_path.startswith("http://"):
        return Path(download_file(file_path, dest / Path(file_path).name))

    raise ValueError(f"Unable to find local path to {file_path}")


def download_file(url: str, output_path: Path | str, force_overwrite: bool = False) -> str:
    """
    Download a file from a URL and write it to ``output_path``.

    The function streams the response, displays a progress bar when the
    content length is known, and writes atomically by first saving to a
    temporary file and then renaming it into place.

    Parameters
    ----------
    url : str
        HTTP(S) URL to download.
    output_path : str or pathlib.Path
        Destination file path. Parent directories are created if missing.
    force_overwrite : bool, default False
        If ``True``, download and overwrite even when ``output_path`` exists.
        If ``False``, an existing file short-circuits and is returned as-is.

    Returns
    -------
    str
        Absolute path to the downloaded file.

    Raises
    ------
    requests.HTTPError
        If the server responds with an error status.
    requests.RequestException
        On connection/timeouts or other request-related failures.
    OSError
        If writing the file to disk fails.

    Notes
    -----
    - Uses streamed download with chunked writes (8 KiB).
    - If the server does not provide ``Content-Length``, the progress bar
      still updates by bytes received, but without a total.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force_overwrite:
        return str(dest.resolve())

    # Stream the response to avoid loading the whole file in memory.
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))

        # Write atomically: temp file in same directory, then rename.
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(dest.parent), delete=False, prefix=dest.name + ".", suffix=".part"
        ) as tmp:
            tmp_name = tmp.name
            with tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {dest.name}",
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    tmp.write(chunk)
                    pbar.update(len(chunk))
            tmp.flush()
            os.fsync(tmp.fileno())

    # Move temp file into place
    Path(tmp_name).replace(dest)

    return str(dest.resolve())


def download_earthaccess(filter_str: str | None = None, result_dir: Path | str = ".", **kwargs) -> list:
    """
    Download datasets via Earthaccess.

    Parameters
    ----------
    filter_str : str, optional
        A string to filter the search results. Default is None.
    result_dir : Union[Path, str], optional
        The directory where the downloaded files will be saved. Default is ".".
    **kwargs : dict
        Additional keyword arguments to pass to the Earthaccess search function.

    Returns
    -------
    list
        A list of paths to the downloaded files.
    """
    p = Path(result_dir)
    p.mkdir(parents=True, exist_ok=True)

    earthaccess.login()
    results = earthaccess.search_data(**kwargs)
    if filter_str is not None:
        results = [
            granule
            for granule in results
            if filter_str in granule["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
        ]
    earthaccess.get_s3_credentials(results=results)
    result = earthaccess.download(results, p)
    return [Path(f) for f in result]


def download_netcdf(
    url: str = "https://dataverse.geus.dk/api/access/datafile/:persistentId?persistentId=doi:10.22008/FK2/OHI23Z/MRSBQR",
    chunk_size: int = 1024,
) -> xr.Dataset:
    """
    Download a dataset from the specified URL and return it as an xarray Dataset.

    Supports both HTTP(S) and S3 URLs (``s3://bucket/key``).

    Parameters
    ----------
    url : str, optional
        The URL of the dataset to download. Default is the mass balance dataset URL.
    chunk_size : int, optional
        The size of the chunks to download at a time, in bytes. Default is 1024 bytes.

    Returns
    -------
    xr.Dataset
        The downloaded dataset as an xarray Dataset.

    Examples
    --------
    >>> dataset = download_dataset()
    >>> print(dataset)
    """
    tmp = Path(tempfile.mktemp(suffix=".nc"))
    try:
        if url.startswith("s3://"):
            parts = url.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1]
            s3 = boto3.client("s3")
            meta = s3.head_object(Bucket=bucket, Key=key)
            file_size = meta["ContentLength"]
            progress = tqdm(total=file_size, unit="iB", unit_scale=True)
            print(f"Downloading {url}")
            s3.download_file(bucket, key, str(tmp), Callback=progress.update)
            progress.close()
        else:
            response = requests.head(url, timeout=10)
            file_size = int(response.headers.get("content-length", 0))
            progress = tqdm(total=file_size, unit="iB", unit_scale=True)
            print(f"Downloading {url}")
            with requests.get(url, stream=True, timeout=10) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        progress.update(len(chunk))
            progress.close()

        return xr.open_dataset(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def download_gebco(
    url: str = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/ice_surface_elevation/netcdf/GEBCO_2026.zip?download=1",
    target_dir: os.PathLike | str = ".",
) -> Path:
    """
    Download and extract GEBCO 2026 ice surface elevation NetCDF if needed.

    Parameters
    ----------
    url : str, optional
        URL to the GEBCO 2026 ZIP archive.
    target_dir : str or PathLike, optional
        Directory where the ZIP and NetCDF file should be stored.

    Returns
    -------
    pathlib.Path
        Path to the extracted NetCDF file.

    Notes
    -----
    - If a valid NetCDF file already exists in `target_dir`, it is returned
      without re-downloading.
    - The function searches for any ``*.nc`` file in `target_dir` and uses
      the first valid one.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    # 1. Check for existing valid NetCDF in target_dir
    existing_nc_files = sorted(target_dir.glob("GEBCO*.nc"))
    for nc_path in existing_nc_files:
        if check_xr_lazy(nc_path):
            return nc_path
    # 2. No valid NetCDF found. Reuse a previously-downloaded zip if present;
    #    otherwise stream the archive from the upstream URL.
    zip_path = target_dir / "gebco.zip"
    if not zip_path.exists():
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            chunk_size = 1024 * 1024  # 1 MB
            tmp = zip_path.with_suffix(zip_path.suffix + ".part")
            with (
                open(tmp, "wb") as f,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading gebco.zip",
                ) as pbar,
            ):
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        pbar.update(len(chunk))
            tmp.rename(zip_path)
    # 3. Extract ZIP
    print(f"Extracting {zip_path} to {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    # 4. Find NetCDF file in target_dir
    nc_files = sorted(target_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF (*.nc) files found in {target_dir} after extracting {zip_path}")
    # Prefer the first valid one
    for nc_path in nc_files:
        if check_xr_lazy(nc_path):
            return nc_path
    raise RuntimeError(f"Found NetCDF files in {target_dir}, but none could be opened successfully.")


def download_hirham(
    base_url: str,
    start_year: int,
    end_year: int,
    output_dir: str | Path = ".",
    max_workers: int = 4,
) -> list[Path]:
    """
    Download HIRHAM files in parallel.

    Parameters
    ----------
    base_url : str
        The base URL for downloading HIRHAM data.
    start_year : int
        The starting year of the files to download.
    end_year : int
        The ending year of the files to download.
    output_dir : Union[str, Path], optional
        The directory where the downloaded files will be saved, by default ".".
    max_workers : int, optional
        The maximum number of threads to use for downloading, by default 4.

    Returns
    -------
    list[Path]
        List of paths to the downloaded files.
    """
    print(f"Downloading HIRHAM5 from {base_url}")
    responses = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for year in range(start_year, end_year + 1):
            year_file = f"{year}.zip"
            url = base_url + year_file
            output_path = output_dir / Path(year_file)
            futures.append(executor.submit(download_file, url, output_path))
            responses.append(output_path)
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

    return responses
