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

# pylint: disable=consider-using-with,too-many-positional-arguments
"""
Module for data processing.
"""

from __future__ import annotations

import os
import re
import tarfile
import tempfile
import zipfile
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple

import cdsapi
import earthaccess
import numpy as np
import requests
import xarray as xr
from tqdm.auto import tqdm

from pism_terra.workflow import check_xr_sampled


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
    files : List[Union[str, Path]]
        List of file paths to unzip.
    output_dir : Union[str, Path], optional
        The directory where the unzipped files will be saved, by default ".".
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    max_workers : int, optional
        The maximum number of threads to use for unzipping, by default 4.

    Returns
    -------
    List[Path]
        List of paths to the unzipped files.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for f in files:
            futures.append(executor.submit(unzip_file, f, str(output_dir), overwrite=overwrite))
        for future in as_completed(futures):
            try:
                future.result()
            except (IOError, ValueError) as e:
                print(f"An error occurred: {e}", unzip_file)

    responses = list(Path(output_dir).rglob("*.nc"))
    return responses


def unzip_file(zip_path: str, extract_to: str, overwrite: bool = False) -> None:
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
    """
    # Ensure the extract_to directory exists
    Path(extract_to).mkdir(parents=True, exist_ok=True)

    # Open the ZIP file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Get the list of file names in the zip file
        file_list = zip_ref.namelist()

        # Iterate over the file names with a progress bar
        for file in tqdm(file_list, desc="Extracting files", unit="file"):
            file_path = Path(extract_to) / file
            if not file_path.exists() or overwrite:
                zip_ref.extract(member=file, path=extract_to)


def extract_archive(
    archive: tarfile.TarFile | zipfile.ZipFile | str | Path,
    extract_to: str | Path = Path("archive"),
    force_overwrite: bool = False,
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
    - Uses `tqdm` for a progress bar.
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
        for member in tqdm(members, desc="Extracting files", unit="file"):
            file_path = extract_to / member
            if (not file_path.exists()) or force_overwrite:
                archive.extract(member, path=extract_to)
            extracted_files.append(str(file_path))

    elif isinstance(archive, tarfile.TarFile):
        members = archive.getmembers()
        for member in tqdm(members, desc="Extracting files", unit="file"):
            file_path = extract_to / member.name
            if (not file_path.exists()) or force_overwrite:
                archive.extract(member, path=extract_to)
            extracted_files.append(str(file_path))

    else:
        raise ValueError(f"Unsupported archive object: {type(archive)}")

    if opened_internally:
        archive.close()

    return extracted_files


def download_request(
    dataset: str = "reanalysis-era5-single-levels-monthly-means",
    area: Sequence[float] = (90.0, -90.0, 45.0, 90.0),
    years: Iterable[int] = range(1980, 2025),
    variable: Sequence[str] = ("2m_temperature", "total_precipitation"),
    path: Path | str = "tmp.nc",
    force_overwrite: bool = False,
) -> xr.Dataset:
    """
    Download monthly ERA5 reanalysis from CDS and return it as an xarray Dataset.

    Sends a request to the Copernicus Climate Data Store (CDS) API for monthly
    averages of the specified single-level variables over ``area`` and ``years``.
    If CDS returns multiple NetCDF files (e.g., one per year), they are opened
    and merged into a single dataset. The merged dataset is cached at ``path`` and
    re-used on subsequent calls unless ``force_overwrite=True``.

    Parameters
    ----------
    dataset : str, default ``"reanalysis-era5-single-levels-monthly-means"``
        CDS dataset identifier to retrieve. Use a different name to target
        ERA5-Land or other collections.
    area : sequence of float, default ``(90, -90, 45, 90)``
        Geographic bounding box **[North, West, South, East]** in degrees (WGS84).
        Note the CDS-specific ordering.
    years : iterable of int, default ``range(1980, 2025)``
        Years to request (e.g., ``range(1980, 2025)`` or ``[1990, 1991]``).
    variable : sequence of str, default ``("2m_temperature", "total_precipitation")``
        ERA5 variable names to download (e.g., ``"2m_temperature"``,
        ``"total_precipitation"``, ``"geopotential"``). Availability depends on
        the chosen ``dataset``.
    path : str or pathlib.Path, default ``"tmp.nc"``
        Cache file. If it exists and opens successfully, it is re-used unless
        ``force_overwrite`` is set.
    force_overwrite : bool, default ``False``
        If ``True``, ignore any existing cache at ``path`` and perform a fresh
        download.

    Returns
    -------
    xarray.Dataset
        Dataset containing requested monthly means. Typical variables include:
        - ``t2m`` : 2-m air temperature [K]
        - ``tp`` : total precipitation [m]
        Coordinates may include a monthly ``valid_time``; when present it is
        floored to daily resolution.

    Raises
    ------
    cdsapi.api.ClientError
        CDS request/authentication/parameter failures.
    OSError
        Problems opening/writing downloaded files.
    ValueError
        Incompatible files for merge.
    Exception
        Other I/O/decoding errors during assembly.

    Notes
    -----
    - Requires a valid CDS API key in ``~/.cdsapirc``.
    - The request uses ``product_type="monthly_averaged_reanalysis"``,
      months ``"01"``–``"12"``, and time ``"00:00"``.
    - If CDS provides a ZIP, contents are extracted before loading/merging.
    """
    path = Path(path)

    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": list(variable),
        "year": list(years),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": list(area),  # [N, W, S, E]
    }

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=False)

    if (not check_xr_sampled(path)) or force_overwrite:
        client = cdsapi.Client()

        path = Path(path)
        path.unlink(missing_ok=True)

        f = client.retrieve(dataset, request).download()

        if f.endswith(".zip"):
            era_files = extract_archive(f)
            dss = []
            for era_file in era_files:
                ds_part = xr.open_dataset(era_file, decode_times=time_coder, decode_timedelta=True)
                if "valid_time" in ds_part.coords:
                    ds_part["valid_time"] = ds_part["valid_time"].dt.floor("D")
                dss.append(ds_part)
            ds = xr.merge(dss).drop_vars(["number", "expver"], errors="ignore")
        else:
            ds = xr.open_dataset(f, decode_times=time_coder, decode_timedelta=True).drop_vars(
                ["number", "expver"], errors="ignore"
            )

        ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        ds.rio.write_crs("EPSG:4326", inplace=True)
        ds.to_netcdf(path)
    else:
        ds = xr.open_dataset(path, decode_times=time_coder, decode_timedelta=True)
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

    ds.to_netcdf(output_filename, encoding=encoding, **kwargs)


def download_archive(url: str) -> tarfile.TarFile | zipfile.ZipFile:
    """
    Download an archive file from a URL and return it as a tarfile or ZipFile object.

    Parameters
    ----------
    url : str
        The URL of the archive file to download. The file can be either a .tar.gz or a .zip file.

    Returns
    -------
    Union[tarfile.TarFile, zipfile.ZipFile]
        The downloaded archive file as a tarfile.TarFile object if the file is a .tar.gz,
        or as a ZipFile object if the file is a .zip.
    """
    response = requests.get(url, stream=True, timeout=5)
    response.raise_for_status()

    total_size = int(response.headers.get("Content-Length", 0))
    buffer = BytesIO()

    with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading archive") as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            buffer.write(chunk)
            pbar.update(len(chunk))

    buffer.seek(0)

    if url.endswith((".tar.gz", ".tgz")):
        return tarfile.open(fileobj=buffer, mode="r:gz")
    elif url.endswith(".zip"):
        return zipfile.ZipFile(buffer)
    else:
        raise ValueError("Unsupported archive format: must end with .zip or .tar.gz")


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
    List
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
    return earthaccess.download(results, p)


def download_netcdf(
    url: str = "https://dataverse.geus.dk/api/access/datafile/:persistentId?persistentId=doi:10.22008/FK2/OHI23Z/MRSBQR",
    chunk_size: int = 1024,
) -> xr.Dataset:
    """
    Download a dataset from the specified URL and return it as an xarray Dataset.

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
    # Get the file size from the headers
    response = requests.head(url, timeout=10)
    file_size = int(response.headers.get("content-length", 0))

    # Initialize the progress bar
    progress = tqdm(total=file_size, unit="iB", unit_scale=True)

    # Download the file in chunks and update the progress bar
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=10) as r:
        r.raise_for_status()
        with open("temp.nc", "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                progress.update(len(chunk))
    progress.close()

    with NamedTemporaryFile(suffix=".nc", delete=False) as xr_file:
        # Open the downloaded file with xarray
        return xr.open_dataset(xr_file)
