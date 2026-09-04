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

import json
import logging
import os
import re
import shutil
import subprocess
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
import rioxarray  # pylint: disable=unused-import
import s3fs
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
    download_gebco,
    download_netcdf,
    file_localizer,
)
from pism_terra.raster import create_ds
from pism_terra.vector import dissolve
from pism_terra.workflow import (
    check_xr_fully,
    check_xr_lazy,
    drop_geotransform_attr,
    stamp_grid_mapping,
)

xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)


# Public source.coop mirror of the ISMIP7 GrIS forcing tree
# (https://source.coop/ismip/ismip7-gris-forcing). Unlike the original Globus
# collection there is no per-version directory level: only the newest version
# of each file is present and the ``v?`` tag lives in the filename only, so
# files are discovered by listing a variable directory rather than by
# constructing names from a configured version.
SOURCE_COOP_ENDPOINT = "https://data.source.coop"
SOURCE_COOP_PREFIX = "ismip/ismip7-gris-forcing/data"

# Trailing ``_YYYY.nc`` year tag of the per-year forcing files.
_YEAR_RE = re.compile(r"_(\d{4})\.nc$")


def _split_source_spec(spec: str | None) -> tuple[str | None, str]:
    """
    Split a per-forcing ``source`` spec into ``(source, short_hand)``.

    The setup TOML declares, per GCM and forcing, the directory subpath
    between the pathway slot and the variable directory. A plain value like
    ``"SDBN1-1000m"`` is the short-hand segment of a standard GCM tree
    (``{gcm}/{pathway}/SDBN1-1000m/{var}``). A slash form like
    ``"RACMO2.3p2-ERA/SDBN1-1000m"`` selects a reanalysis-forced tree (OCX):
    the first segment replaces the pathway directory and takes the GCM slot in
    filenames, the second is the short-hand
    (``OCX/RACMO2.3p2-ERA/SDBN1-1000m/{var}``).

    Parameters
    ----------
    spec : str or None
        The configured spec, or ``None``/``"none"`` for trees without the
        short-hand segment.

    Returns
    -------
    tuple
        ``(source, short_hand)`` as consumed by :func:`_local_path` and the
        cloud sync: ``source`` is ``None`` for standard trees and
        ``short_hand`` is ``"none"`` when the segment is absent.
    """
    if not spec or spec == "none":
        return None, "none"
    if "/" in spec:
        source, short_hand = spec.split("/", 1)
        return source, short_hand
    return None, spec


def _cloud_fs() -> s3fs.S3FileSystem:
    """
    Open an anonymous S3 filesystem on the source.coop data endpoint.

    A genuinely fresh instance is returned on purpose (fsspec caches
    instances by constructor args, and ``s3fs`` caches directory listings
    per instance): change detection needs the current upstream ETags, not a
    stale listing from an earlier sync in the same process.

    Returns
    -------
    s3fs.S3FileSystem
        Anonymous filesystem rooted at the source.coop S3 endpoint.
    """
    # Generous read timeout: a dozen parallel ~76 MB streams (several dask
    # tasks x download threads) can starve an individual socket well past
    # botocore's 60 s default, which shows up as AioReadTimeoutError.
    return s3fs.S3FileSystem(
        anon=True,
        endpoint_url=SOURCE_COOP_ENDPOINT,
        skip_instance_cache=True,
        config_kwargs={
            "connect_timeout": 60,
            "read_timeout": 300,
            "retries": {"max_attempts": 5, "mode": "adaptive"},
        },
    )


def _remote_var_dir(gcm: str, pathway: str, short_hand: str, m_var: str, source: str | None = None) -> str:
    """
    Build the source.coop key of one variable directory.

    Parameters
    ----------
    gcm : str
        GCM name (e.g. ``"CESM2-WACCM"``), the top-level directory.
    pathway : str
        Emissions pathway (e.g. ``"historical"``, ``"ssp585"``).
    short_hand : str
        Short-hand segment (e.g. ``"SDBN1-1000m"``) or ``"none"``.
    m_var : str
        Variable name (e.g. ``"acabf"``).
    source : str or None, optional
        Reanalysis source for OCX-style trees; replaces the pathway segment.

    Returns
    -------
    str
        Key of the directory holding that variable's per-year files.
    """
    parts = [SOURCE_COOP_PREFIX, gcm, pathway if source is None else source]
    if short_hand != "none":
        parts.append(short_hand)
    parts.append(m_var)
    return "/".join(parts)


def _meta_path(local: Path) -> Path:
    """
    Sidecar file recording the upstream identity of a cached download.

    Parameters
    ----------
    local : pathlib.Path
        The cached NetCDF.

    Returns
    -------
    pathlib.Path
        Path of the JSON sidecar next to it.
    """
    return local.with_name(local.name + ".s3meta.json")


def _write_meta(local: Path, remote: dict) -> None:
    """
    Record the upstream ETag/size/date of a cached file in its sidecar.

    Parameters
    ----------
    local : pathlib.Path
        The cached NetCDF the sidecar belongs to.
    remote : dict
        S3 listing entry for the upstream object (``ETag``, ``size``,
        ``LastModified``).
    """
    meta = {
        "etag": str(remote.get("ETag", "")).strip('"'),
        "size": int(remote["size"]),
        "last_modified": str(remote.get("LastModified", "")),
    }
    _meta_path(local).write_text(json.dumps(meta), encoding="utf-8")


def _needs_download(local: Path, remote: dict) -> bool:
    """
    Decide whether an upstream object must be (re-)fetched into the cache.

    A cached file is considered current when its sidecar records the same
    ETag and size the upstream listing reports. A file without a sidecar
    (e.g. hand-copied into the cache) is adopted when its size matches the
    upstream object — the sidecar is written so later runs compare ETags —
    and re-downloaded otherwise.

    Parameters
    ----------
    local : pathlib.Path
        Cached file location (may not exist yet).
    remote : dict
        S3 listing entry for the upstream object.

    Returns
    -------
    bool
        ``True`` when the file must be downloaded.
    """
    if not local.exists():
        return True
    etag = str(remote.get("ETag", "")).strip('"')
    meta_file = _meta_path(local)
    if meta_file.exists():
        try:
            recorded = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return True
        return recorded.get("etag") != etag or int(recorded.get("size", -1)) != int(remote["size"])
    if local.stat().st_size == int(remote["size"]):
        _write_meta(local, remote)
        return False
    return True


def _fetch_one(fs: s3fs.S3FileSystem, rkey: str, local: Path, remote: dict, attempts: int = 4) -> Path:
    """
    Download one object to the cache, atomically, and stamp its sidecar.

    Transient network failures (socket read timeouts, dropped connections)
    are retried with exponential backoff instead of aborting the whole sync;
    a bad ``.part`` file is discarded before each retry.

    Parameters
    ----------
    fs : s3fs.S3FileSystem
        Filesystem to fetch through.
    rkey : str
        Remote object key.
    local : pathlib.Path
        Cache destination.
    remote : dict
        S3 listing entry for the object (recorded in the sidecar).
    attempts : int, default 4
        Total tries before the last error is re-raised.

    Returns
    -------
    pathlib.Path
        ``local``, once the download completed.
    """
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            fs.get_file(rkey, str(tmp))
            break
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            wait = 2**attempt
            logger.warning(
                "Download %s failed (%s: %s); retry %d/%d in %ds",
                rkey.rsplit("/", 1)[-1],
                type(exc).__name__,
                exc,
                attempt,
                attempts - 1,
                wait,
            )
            time.sleep(wait)
    tmp.replace(local)
    _write_meta(local, remote)
    return local


def _sync_cloud_files(
    gcm: str,
    pathway: str,
    short_hand: str,
    fields: Sequence[str],
    start_year: int,
    end_year: int,
    cache_path: Path | str,
    source: str | None = None,
    max_workers: int = 4,
    label: str = "",
) -> dict[tuple[str, int], Path]:
    """
    Mirror the required per-year forcing files from source.coop into a cache.

    Lists each variable directory once (which also discovers the current
    filename, version tag included), then downloads only the files that are
    missing locally or whose upstream ETag/size changed since they were
    cached (see :func:`_needs_download`). The cache mirrors the cloud layout
    under *cache_path*, with a small JSON sidecar per file recording the
    upstream identity.

    Parameters
    ----------
    gcm : str
        GCM name (top-level directory).
    pathway : str
        Emissions pathway.
    short_hand : str
        Short-hand segment or ``"none"``.
    fields : sequence of str
        Variable names to fetch.
    start_year, end_year : int
        Inclusive year range.
    cache_path : str or pathlib.Path
        Root of the local cache.
    source : str or None, optional
        Reanalysis source for OCX-style trees.
    max_workers : int, default 4
        Concurrent downloads.
    label : str, optional
        Progress/log label; defaults to ``{gcm}/{pathway}``.

    Returns
    -------
    dict
        Mapping of ``(variable, year)`` to the cached local path.

    Raises
    ------
    FileNotFoundError
        When a variable directory does not exist upstream, or years inside
        the requested range are missing from it.
    """
    fs = _cloud_fs()
    cache_path = Path(cache_path)
    label = label or f"{gcm}/{pathway}"

    wanted: dict[tuple[str, int], dict] = {}
    for m_var in fields:
        rdir = _remote_var_dir(gcm, pathway, short_hand, m_var, source=source)
        try:
            entries = fs.ls(rdir, detail=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"No such forcing directory upstream: {SOURCE_COOP_ENDPOINT}/{rdir}") from exc
        by_year: dict[int, dict] = {}
        for entry in entries:
            m = _YEAR_RE.search(entry["name"])
            if m:
                by_year[int(m.group(1))] = entry
        missing = [y for y in range(start_year, end_year + 1) if y not in by_year]
        if missing:
            raise FileNotFoundError(
                f"{SOURCE_COOP_ENDPOINT}/{rdir} is missing {len(missing)} year(s) in "
                f"{start_year}..{end_year} (first: {missing[0]}, last: {missing[-1]})"
            )
        for year in range(start_year, end_year + 1):
            wanted[(m_var, year)] = by_year[year]

    local_paths: dict[tuple[str, int], Path] = {}
    to_fetch: list[tuple[str, Path, dict]] = []
    for key, entry in wanted.items():
        rel = entry["name"].removeprefix(SOURCE_COOP_PREFIX).lstrip("/")
        local = cache_path / rel
        local_paths[key] = local
        if _needs_download(local, entry):
            to_fetch.append((entry["name"], local, entry))

    logger.info("%s: %d file(s) cached, %d to download", label, len(wanted) - len(to_fetch), len(to_fetch))
    if to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, fs, rkey, local, entry): local for rkey, local, entry in to_fetch}
            for fut in tqdm(cf_as_completed(futures), total=len(futures), desc=f"Download {label}", unit="file"):
                fut.result()
    return local_paths


def _local_path(year, data_path, ice_sheet, gcm, pathway, short_hand, m_var, version, source=None):
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
    source : str or None, optional
        Source dataset for reanalysis-forced trees (e.g. OCX uses
        ``"RACMO2.3p2-ERA"`` for climate and ``"EN4"`` for ocean). When given,
        the directory is ``{gcm}/{source}/...`` (no pathway segment) and the
        filename embeds ``{source}_{gcm}`` instead of ``{gcm}_{pathway}``.

    Returns
    -------
    pathlib.Path
        Path to the file under *data_path* (or under *data_path/GrIS* if that
        is where the file actually lives). When neither candidate exists, the
        first candidate is returned so callers can surface a sensible error.
    """
    name_gcm, name_pathway = (gcm, pathway) if source is None else (source, gcm)
    fname = (
        f"{m_var}_{ice_sheet}_{name_gcm}_{name_pathway}_{short_hand}_{version}_{year}.nc"
        if short_hand != "none"
        else f"{m_var}_{ice_sheet}_{name_gcm}_{name_pathway}_{version}_{year}.nc"
    )
    rel_parts = [gcm, pathway if source is None else source]
    if short_hand != "none":
        rel_parts.append(short_hand)
    rel_parts.extend([m_var, version, fname])
    rel = Path(*rel_parts)
    return data_path / ice_sheet / rel


def _strip_fill_attrs(path: Path) -> None:
    """
    Strip ``_FillValue`` and ``missing_value`` attrs in place.

    CDO and netCDF-4 both treat ``_FillValue`` as a reserved attribute and
    refuse to delete it via the public API. NCO's ``ncatted`` is the only
    reliable way to remove it as a pure metadata edit (no data rewrite).
    All NaN / missing cells have already been filled by ``setmisstoc`` (and
    the per-variable temperature fill in ``_merge_one_var``) by the time
    this runs, so the attributes carry no remaining information.

    Parameters
    ----------
    path : pathlib.Path
        NetCDF file to edit in place.

    Raises
    ------
    RuntimeError
        If ``ncatted`` is not installed; install ``nco`` in the active
        conda env to satisfy this dependency.
    subprocess.CalledProcessError
        If ``ncatted`` runs but fails (e.g. permissions, corrupt file).
    """
    if shutil.which("ncatted") is None:
        raise RuntimeError("ncatted not found on PATH; install the 'nco' conda package")
    # The "...,,d,," form means: attribute name ".._FillValue", any var
    # (empty middle field), action "d" (delete), no value, no type.
    subprocess.run(
        [
            "ncatted",
            "-O",
            "-h",  # don't append a history line; the file stays bit-stable across reruns
            "-a",
            "_FillValue,,d,,",
            "-a",
            "missing_value,,d,,",
            str(path),
        ],
        check=True,
    )


def _process_single_forcing(
    ice_sheet: Literal["AIS", "GrIS"],
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
    freq: str = "1mon",
    calendar: str = "365_day",
    data_path: Path | None = None,
    staging_path: Path | None = None,
    source: str | None = None,
) -> list[Path]:
    """
    Process a single (GCM, pathway, forcing) combination into one output file.

    Each pathway (historical, ssp???) is now its own task: no more coupling
    the historical and projection epochs inside one worker. The output
    filename embeds ``pathway`` and the ``start_year``/``end_year`` span,
    e.g. ``ismip7_greenland_climate_historical_CESM2-WACCM_v2_1978_2014.nc``.

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
        Root of the local download cache (cloud mode); the source.coop tree
        is mirrored below it.
    output_path : Path
        Output directory.
    pathway : str
        Pathway name (e.g., ``"historical"``, ``"ssp585"``).
    version : str
        Version label embedded in the output filename (e.g., ``"v1"``). The
        source files themselves are discovered by listing the cloud tree, so
        this does not select what is downloaded.
    start_year : int
        First year in the epoch (inclusive).
    end_year : int
        Last year in the epoch (**inclusive** — ``end_year = 2300`` means
        the year 2300 is processed).
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
        If given, read forcing files from this local mirror of the original
        Globus tree instead of downloading. Files are looked up with their
        Globus filename under ``data_path`` (or under ``data_path/GrIS``).
        When ``None`` (default), the function syncs the required files from
        the public source.coop mirror into a cache under ``base_path``,
        re-downloading only files whose upstream ETag/size changed.
    staging_path : pathlib.Path or None, optional
        Directory for intermediate scratch (the per-variable cdo
        ``mergetime`` tmp files). Auto-cleaned at the end of the function
        via ``TemporaryDirectory``. When ``None`` (default),
        ``output_path`` is used — but only the final merged file is left
        in ``output_path`` either way.
    source : str or None, optional
        Reanalysis source for OCX-style trees (e.g. ``"RACMO2.3p2-ERA"`` for
        climate, ``"EN4"`` for ocean), from the first segment of a slash-form
        ``source`` spec (see :func:`_split_source_spec`). Changes the layout
        to ``{gcm}/{source}/...`` and the filenames to
        ``..._{source}_{gcm}_...``; ``None`` for standard GCM/pathway trees.

    Returns
    -------
    list[Path]
        Paths to the produced NetCDF files (one per group: ``climate`` and
        optionally ``climate_gradient`` when ``forcing == "climate"``).
    """
    os.environ["HDF5_LOG_LEVEL"] = "0"
    cdo = Cdo()
    cdo.debug = True

    # The cdo grid description is scratch, not a shipped product; keep it out
    # of the output directory so that stays 1:1 syncable to S3.
    grid_file = file_localizer(
        "s3://pism-cloud-data/ismip7_extra/grid.txt",
        dest=staging_path if staging_path is not None else output_path,
    )
    tas_replace = ""

    output_files = []

    cloud_files: dict[tuple[str, int], Path] = {}

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
            ``data_path`` was supplied, else the cache location the cloud
            sync placed the file at (discovered, version tag included, from
            the source.coop listing).
        """
        if data_path is not None:
            return _local_path(year, data_path, ice_sheet, gcm, pathway_name, short_hand, m_var, version, source=source)
        return cloud_files[(m_var, year)]

    if data_path is None:
        # ``end_year`` is inclusive per the campaign config convention.
        cloud_files = _sync_cloud_files(
            gcm,
            pathway,
            short_hand,
            fields,
            start_year,
            end_year,
            base_path,
            source=source,
            label=f"{gcm}/{pathway}/{forcing}",
        )
    else:
        logger.info("Using local ISMIP7 forcing under %s for %s/%s/%s", data_path, gcm, pathway, forcing)

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
        paths = [_resolve(year, pathway_name, m_var) for year in range(start_year, end_year + 1)]
        k, v = m_var, ismip7_to_pism[m_var]
        out = tmp_root / f"{epoch_label}_{m_var}.nc"
        # Per-variable fill applied in the per-tmp stage so the outer
        # ``setmisstoc,0`` has nothing to do for this variable. The outer
        # fill is correct for SMB / runoff (no-ice cells == 0 mass change)
        # but unphysical for surface temperature, which the source masks
        # out as NaN over non-ice cells and (in some files) ships with
        # literal 0 K values. Replace both with 260 K — well below the
        # outer fill range, so any real cold temperature is preserved.
        if m_var in ("ts", "tas"):
            fill_op = " -setrtoc,-1,1,260 -setmisstoc,260"
        elif m_var == "tf":
            # Ocean thermal forcing: extrapolate valid ocean values into the
            # masked gaps (nearest-neighbour) instead of letting the outer
            # ``setmisstoc,0`` zero-fill them. A zero thermal forcing (ocean at
            # the freezing point) breaks PICO's box-1 quadratic (T_star ~ 0 ->
            # negative sqrt); real thermal forcing can dip slightly below 0, so
            # do NOT treat near-zero values as fill here.
            fill_op = " -setmisstonn"
        elif m_var == "so":
            # Ocean salinity: convert the near-zero land/fill values to missing,
            # then extrapolate valid ocean salinity into the gaps
            # (nearest-neighbour) rather than filling with a constant. Real ocean
            # salinity is ~30-35 g/kg, never near zero, so [-1, 1] is safe to drop.
            fill_op = " -setmisstonn -setrtomiss,-1,1"
        else:
            fill_op = ""
        # Some per-year ISMIP7 source files carry an extra grid-mapping variable
        # (``crs``) — and define it inconsistently between years — while others
        # omit it. ``-mergetime`` then aborts with "Input streams have different
        # number of variables per timestep" (preceded by a flood of "Inconsistent
        # variable definition for crs!"). Select only the physical variable from
        # each input first (``-apply,-selname``) so every stream has an identical,
        # single-variable structure; the grid is re-attached by ``-setgrid`` below.
        merge_inputs = f"-apply,-selname,{k} [ " + " ".join(str(p) for p in paths) + " ]"
        mergetime_chain = f"{tas_replace}{fill_op} -setgrid,{str(grid_file)} -mergetime {merge_inputs}"
        if m_var == "so":
            # CMIP6 sea-water salinity ships with ``units = "psu"`` (practical
            # salinity unit). PISM's ``ocean.th`` requires the numerically
            # identical but udunits-parsable ``g/kg`` and refuses to read the
            # file otherwise. Patch the attribute on the renamed variable in
            # the same cdo invocation so we don't pay an extra read/write.
            cdo.setattribute(
                f"{v}@units=g/kg",
                input=f"-chname,{k},{v} {mergetime_chain}",
                output=str(out.resolve()),
                options="-f nc4 -z zip_2",
            )
        else:
            cdo.chname(
                f"{k},{v}",
                input=mergetime_chain,
                output=str(out.resolve()),
                options="-f nc4 -z zip_2",
            )
        return out

    # ISMIP7 publishes "climate" fields at two cadences: ``acabf`` / ``mrro``
    # / ``ts`` are monthly, while the elevation-gradient fields ``dacabfdz``
    # and ``dmrrodz`` are annual. ``cdo -merge`` requires aligned time axes,
    # so mixing the two would silently truncate the monthly streams to the 5
    # annual timesteps (warning ``Input stream 2 has 5 timesteps. Stream 1
    # has more timesteps, skipped!``). Emit two separate output files
    # instead: the standard ``..._climate_...`` for the monthly fields and
    # ``..._climate_gradient_...`` for the annual ones. Other forcings
    # (ocean, …) keep the original single-group behavior.
    annual_fields_set = {"dacabfdz", "dmrrodz", "dtsdz"}
    if forcing == "climate":
        monthly_fields = [f for f in fields if f not in annual_fields_set]
        annual_fields = [f for f in fields if f in annual_fields_set]
        groups: list[tuple[str, list[str], str, str]] = []
        if monthly_fields:
            groups.append(("climate", monthly_fields, freq, "01-16 12:00"))
        if annual_fields:
            # Mid-year anchor for the annual gradient timestamps. CDO's
            # ``settbounds`` only accepts hour/day/month frequency tokens,
            # so use ``12month`` (which both ``settbounds`` and ``settaxis``
            # parse as one calendar year) instead of ``1yr``.
            groups.append(("climate_gradient", annual_fields, "12month", "07-02 12:00"))
    else:
        groups = [(forcing, fields, freq, "01-16 12:00")]

    # Intermediates (cdo ``mergetime`` tmps) live under ``staging_path``
    # instead of ``output_path``. The tempdir is removed on ``with`` exit,
    # so disk usage drops back to just the single output file per group.
    staging_root = Path(staging_path) if staging_path is not None else output_path
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"_ismip7_{gcm}_{pathway}_{forcing}_", dir=str(staging_root)) as _tmp:
        tmp_root = Path(_tmp)

        for label, sub_fields, sub_freq, time_anchor in groups:
            tmps = [
                _merge_one_var(tmp_root, f"{pathway}_{label}", pathway, start_year, end_year, m_var)
                for m_var in sub_fields
            ]

            # Output name embeds the epoch: the historical file and each
            # projection file live side-by-side and are consumed
            # independently by PISM's two-invocation forward driver
            # (see pism_terra.ismip7.greenland.run._render_forward_run).
            out_file = output_path / Path(
                f"ismip7_greenland_{label}_{pathway}_{gcm}_{version}_{start_year}_{end_year}.nc"
            )
            # ``-merge`` is variadic; ``[ ... ]`` delimits the file list
            # so python-cdo doesn't misinterpret the ``output=`` argument
            # as another input stream.
            cdo.setmisstoc(
                0,
                input=(
                    f"-setgrid,{str(grid_file)} -settbounds,{sub_freq} "
                    f"-setreftime,1850-01-01 -settunits,hours -setcalendar,{calendar} "
                    f"-settaxis,'{start_year}-{time_anchor},,{sub_freq}' -merge [ "
                    + " ".join(str(p.resolve()) for p in tmps)
                    + " ]"
                ),
                output=str(out_file.resolve()),
                options="-f nc4 -z zip_2",
            )
            _strip_fill_attrs(out_file)
            output_files.append(out_file)

    return output_files


# Packaged GrIS basin polygons (attribute ``basin`` in 1..7) used for the mask.
_BASIN_GPKG = Path(__file__).resolve().parents[2] / "data" / "gris-basins-w-shelves.gpkg"


def basin_mask(target_grid: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """
    Rasterize the GrIS basins onto ``target_grid`` as an integer mask.

    Reads the packaged basin polygons (``gris-basins-w-shelves.gpkg``, integer
    attribute ``basin`` in ``1..7``) and assigns each grid cell the ``basin`` value
    of the polygon that covers it; cells outside every basin are ``0``.

    Parameters
    ----------
    target_grid : xarray.Dataset or xarray.DataArray
        Grid to rasterize onto. Must carry ``x``/``y`` coordinates in EPSG:3413.

    Returns
    -------
    xarray.DataArray
        ``int8`` basin mask named ``"basins"`` with values ``0..7``.
    """
    basins_gdf = gpd.read_file(_BASIN_GPKG).to_crs("EPSG:3413")

    # Zero reference field on the target grid, with x/y + CRS so rio.clip works.
    ref = xr.DataArray(
        np.zeros((target_grid["y"].size, target_grid["x"].size), dtype="float32"),
        dims=("y", "x"),
        coords={"y": target_grid["y"], "x": target_grid["x"]},
    ).rio.write_crs("EPSG:3413")

    basins = xr.zeros_like(ref, dtype="int8")
    for _, row in basins_gdf.iterrows():
        try:
            # Inside-polygon mask (clip keeps interior, sets exterior -> NaN).
            inside = ref.rio.clip([row.geometry], drop=False, all_touched=True).notnull()
        except Exception:  # pylint: disable=broad-exception-caught  # basin outside the grid
            continue
        basins = basins.where(~inside, int(row["basin"]))
    basins = basins.astype("int8")
    basins.name = "basins"
    basins.attrs = {"units": "1", "long_name": "GrIS basin id (1-7, 0=outside)"}
    return basins


def add_basins_to_ocean_files(ocean_files: list[Path]) -> None:
    """
    Add the GrIS basin mask to each ocean forcing file, in place.

    Reads only each file's ``x``/``y`` coordinates, rasterizes the packaged basin
    polygons onto that grid via :func:`basin_mask`, and appends the result as an
    ``int8`` ``basins`` variable **in place** (``mode="a"``). A file whose grid
    cannot be matched is logged and skipped rather than aborting the whole batch.

    The forcing payload is never read. A projection ocean file spans 2015-2300 at
    monthly cadence (3432 timesteps x ``tf`` + ``so``); loading it to rewrite the
    whole dataset needs tens of GB and gets the process OOM-killed on a shared
    node. Appending one 2-D mask touches only the new variable, so cost is
    independent of the record length.

    Parameters
    ----------
    ocean_files : list of pathlib.Path
        Ocean forcing NetCDFs to annotate.
    """
    for ocean_file in ocean_files:
        try:
            # ``decode_times=False``: the time axis is irrelevant here and
            # projections run past the datetime64[ns] ceiling (year 2262), which
            # would otherwise trigger a cftime fallback warning.
            with xr.open_dataset(ocean_file, decode_times=False, decode_timedelta=False) as ds:
                grid = xr.Dataset(coords={"x": ds["x"].to_numpy(), "y": ds["y"].to_numpy()})
                # Georeference basins like the file's other data variables (match
                # their ``grid_mapping``, e.g. ``crs``, rather than the source field's).
                grid_mapping = next(
                    (
                        ds[v].attrs["grid_mapping"]
                        for v in ds.data_vars
                        if v != "basins" and "grid_mapping" in ds[v].attrs
                    ),
                    None,
                )
                already_present = "basins" in ds.variables
            basins = basin_mask(grid)
            if grid_mapping:
                basins.attrs["grid_mapping"] = grid_mapping
            # ``basin_mask`` builds its reference field through rioxarray, which
            # attaches a ``spatial_ref`` coordinate. The real grid-mapping
            # variable is already in the file, so don't append a second one.
            basins = basins.drop_vars("spatial_ref", errors="ignore")
            # Encoding may only be set when the variable is created; on a rerun
            # the existing (already compressed) variable is overwritten in place.
            encoding = {} if already_present else {"basins": {"zlib": True, "complevel": 2, "_FillValue": None}}
            xr.Dataset({"basins": basins}).to_netcdf(ocean_file, mode="a", engine="h5netcdf", encoding=encoding)
            logger.info("Added basin mask to %s", ocean_file.name)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Could not add basin mask to %s: %s", ocean_file, exc)


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
        Dictionary with keys ``"boot_file"``, ``"heatflux_file"``, and
        ``"obs_file"`` mapping to their respective output paths. The
        ``"obs_file"`` is an inverse-distance-weighted collapse of
        ``vx_timeseries``/``vy_timeseries``, post-processed in the same
        shape as :func:`pism_terra.glacier.observations.glacier_velocities_from_grid`.
    """

    rho_ice = 910.0  # constants.ice.density
    rho_sea_water = 1028.0  # constants.sea_water.density
    ice_free_thickness = 0.01  # geometry.ice_free_thickness_standard
    sea_level = 0.0
    alpha = 1.0 - rho_ice / rho_sea_water

    # ``url`` may be a Path (when reading from a local mirror) or a string
    # (when downloading from Globus). Normalize to str for split/download.
    url_str = str(url)
    name = url_str.rsplit("/", maxsplit=1)[-1]
    boot_file = Path(input_path) / Path(name)
    if (not check_xr_lazy(boot_file)) or force_overwrite:
        ds_bm = download_netcdf(url_str)
    else:
        ds_bm = xr.open_dataset(boot_file)

    ds_bm = ds_bm.rename_vars({"surface_grimp": "surface"})
    ds_bm["surface"].attrs.update(
        {"standard_name": "surface_altitude", "long_name": "ice surface elevation", "units": "m"}
    )

    if target_grid is not None:
        ds_bm_regridded = ds_bm[["bed", "thickness", "surface"]].regrid.conservative(target_grid)
        # ``mask`` is a categorical class label (0 = ocean, 1 = ice-free land,
        # 2 = grounded ice, 3 = floating ice, 4 = non-Greenland land), so it must
        # not be area-averaged. Conservative regridding blends classes across
        # every coastline and shelf edge — over Petermann it turns 4 classes into
        # 41 distinct values, 4.4% of cells non-integer — which makes the exact
        # ``== 0`` / ``== 3`` tests below miss precisely the mixed cells they need
        # to catch. Majority resampling keeps the field categorical. Ties are
        # broken by ``idxmax`` (lowest class wins), so the result is reproducible.
        # ``sortby("y")``: BedMachine's y axis descends, and unlike the
        # conservative regridder the flox-based majority reduction needs
        # ascending coordinates (it otherwise reduces over an empty bin and
        # raises "zero-size array to reduction operation maximum").
        ds_bm_regridded["mask"] = (
            ds_bm["mask"].sortby("y").regrid.most_common(target_grid, values=np.array([0, 1, 2, 3, 4]), time_dim=None)
        )
        gebco_p = download_gebco(target_dir=input_path)
        gebco = xr.open_dataset(gebco_p, chunks="auto").rio.write_crs("EPSG:4326")
        gebco_bm_regridded = gebco.rio.reproject_match(
            ds_bm_regridded.rio.write_crs("EPSG:3413"), resampling=Resampling.bilinear
        ).compute()
        # GEBCO fills only what lies outside BedMachine's domain (where the target
        # grid extends past its coverage, leaving NaN after regridding). It must
        # NOT replace BedMachine over ocean: BedMachine resolves Greenland's fjord
        # bathymetry, where GEBCO is far too shallow — in Ilulissat Icefjord a
        # median -209 m against BedMachine's -558 m. Substituting it there both
        # publishes a bed that grounds ice which should float and, via the
        # flotation cap below, carves the coarse bathymetry into the ice
        # thickness. BedMachine has no NaN bed inside its own domain, so this
        # keeps its bed everywhere it exists.
        use_gebco = ds_bm_regridded["bed"].isnull()
        ds_bm_regridded["bed"] = ds_bm_regridded["bed"].where(~use_gebco, gebco_bm_regridded["elevation"])
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
        dem_year = "1985"
        surface_file = Path(input_path) / Path("surface_dem.nc")
        bed = ds_bm_regridded["bed"]
        if (not check_xr_lazy(surface_file)) or force_overwrite:
            ds = download_netcdf(surface_dem)
            ds.to_netcdf(surface_file)
        else:
            ds = xr.open_dataset(surface_file)
        surface = ds["surface"].regrid.conservative(target_grid)
        surface = surface.where(surface > 0, 0)
        surface.name = "surface"
        mask = ds_bm_regridded["mask"]
        # Bed-referenced thickness only where the ice is actually grounded. The
        # surface DEM and BedMachine's mask are different products on different
        # grids, so their coastlines disagree; applying ``surface - bed`` to a
        # cell whose surface is a valley wall but whose bed is the fjord floor
        # produced >1500 m of phantom ice along Petermann's margins.
        thickness = xr.where((surface > 0) & (mask == 2), surface - bed, 0)
        # In the ocean (mask == 0) and on ice shelves (mask == 3) a positive
        # surface is floating freeboard, not bed-referenced elevation, so recover
        # the thickness from flotation: freeboard = alpha * H  =>  H = surface / alpha.
        # Cap that by the local water depth: a floating column of thickness H
        # draws (rho_ice / rho_sea_water) * H of water and cannot float in less,
        # so H <= depth * rho_sea_water / rho_ice. Without the cap the 1/alpha
        # (~8.7x) amplification turns a single DEM cell contaminated by the fjord
        # wall into hundreds of metres of ice right at the calving front.
        max_floating_thickness = (-bed * rho_sea_water / rho_ice).clip(min=0)
        thickness_from_flotation = np.minimum(surface / alpha, max_floating_thickness)
        # Only recover floating thickness inside the GrIS basin polygons
        # (basin > 0); outside them, mask 0/3 cells are spurious ocean/shelf
        # that would otherwise pick up bogus thicknesses.
        in_basin = basin_mask(target_grid) > 0
        is_floating = (surface > 0) & ((mask == 0) | (mask == 3)) & in_basin
        thickness = xr.where(is_floating, thickness_from_flotation, thickness)
        thickness = thickness.where(thickness > 10, 0)
        thickness = thickness.where(in_basin, 0)
        thickness.name = "thickness"
        thickness.attrs.update(ds_bm_regridded["thickness"].attrs)
        boot = xr.merge([bed, ftt_mask, surface, thickness, liafr])
    else:
        dem_year = "2007"
        thickness = ds_bm_regridded["thickness"]
        thickness = thickness.where(thickness > 10, 0)
        thickness.name = "thickness"
        thickness.attrs.update(ds_bm_regridded["thickness"].attrs)
        boot = xr.merge([ds_bm_regridded[["bed", "surface"]], thickness, ftt_mask, liafr])

    # Ellesmere Island sits inside the domain but is not part of the modeled
    # Greenland ice sheet; force it to deep ocean so no ice can grow there.
    # Inside the polygon set bed = -2000 m, surface = 0 m, thickness = 0 m.
    ellesmere_gpkg = Path(__file__).resolve().parents[2] / "data" / "ellsmere.gpkg"
    ellesmere = gpd.read_file(ellesmere_gpkg).to_crs("EPSG:3413")
    if len(ellesmere) == 0:
        logger.warning("%s has no geometry; skipping Ellesmere override", ellesmere_gpkg.name)
    else:
        try:
            # Inside-polygon mask on the boot grid: clip a constant field (inside
            # kept, outside -> NaN), so notnull() marks the polygon interior.
            inside = (
                xr.ones_like(boot["bed"])
                .rio.write_crs("EPSG:3413")
                .rio.clip(ellesmere.geometry, drop=False, all_touched=True)
                .notnull()
            )
            boot["bed"] = boot["bed"].where(~inside, -2000.0)
            boot["surface"] = boot["surface"].where(~inside, 0.0)
            boot["thickness"] = boot["thickness"].where(~inside, 0.0)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Ellesmere override skipped (%s)", exc)

    boot = boot.fillna(0)
    ds = boot
    geo = (
        ds_bm[["geothermal_heat_flux1"]]
        .rename_dims({"x1km": "x", "y1km": "y"})
        .rename_vars({"x1km": "x", "y1km": "y", "geothermal_heat_flux1": "bheatflx"})
        .regrid.conservative(target_grid)
    )
    geo = geo.where(geo != -9999, 0.042)
    geo = geo.where(geo <= 1e36, 0.042)

    ds["surface"].attrs.update({"standard_name": "surface_altitude", "units": "m"})
    ds["bed"].attrs.update({"standard_name": "bedrock_altitude", "units": "m"})
    ds = ds.rename_vars({k: v for k, v in config["ismip7_to_pism"].items() if k in ds}).drop_vars(
        ["crs", "spatial_ref"], errors="ignore"
    )
    for v in ds.data_vars:
        ds[v].attrs.pop("coordinates", None)
        ds[v].encoding.pop("coordinates", None)
        # BedMachine variables ship with stale per-variable ``grid_mapping``
        # attrs (e.g. ``surface:grid_mapping = "polar_stereographic"``) that
        # point at variables we never write. ``stamp_grid_mapping`` would
        # canonicalize them on its own, but stripping here keeps the
        # intermediate state cleaner and avoids surprises if the helper's
        # heuristics change.
        ds[v].attrs.pop("grid_mapping", None)

    # ``write_crs`` returns a new dataset — the old code threw the return
    # away, leaving ``ds`` without any ``mapping`` variable and with the
    # stale BedMachine attrs as the only grid-mapping evidence.
    ds = ds.rio.write_crs("EPSG:3413", grid_mapping_name="mapping").rio.write_coordinate_system()
    drop_geotransform_attr(ds)
    ds = stamp_grid_mapping(ds, name="mapping")

    comp = {"zlib": True, "complevel": 2}
    for var in list(ds.data_vars) + list(ds.coords):
        ds[var].encoding.update({"_FillValue": None})
    for var in list(ds.data_vars):
        ds[var].encoding.update(comp)

    resolution = int(ds.x[1] - ds.x[0])
    boot_file = output_path / Path(f"boot_{dem_year}_g{resolution}m_GreenlandObsISMIP7-v1.3.nc")

    ds.to_netcdf(boot_file, engine="h5netcdf")

    geo["bheatflx"].attrs.pop("coordinates", None)
    geo["bheatflx"].encoding.pop("coordinates", None)
    geo = geo.drop_vars("spatial_ref", errors="ignore")
    for v in geo.data_vars:
        geo[v].attrs.pop("coordinates", None)
        geo[v].encoding.pop("coordinates", None)
        geo[v].attrs.pop("grid_mapping", None)
    geo = geo.rio.write_crs("EPSG:3413", grid_mapping_name="mapping").rio.write_coordinate_system()
    drop_geotransform_attr(geo)
    geo = stamp_grid_mapping(geo, name="mapping")
    geo_file = output_path / Path(f"heatflux_g{resolution}m_GreenlandObsISMIP7-v1.3.nc")
    geo_encoding = {var: {"_FillValue": None} for var in list(geo.data_vars) + list(geo.coords)}
    for var in geo.data_vars:
        # See the obs write below: a per-variable encoding dict replaces the
        # variable's ``.encoding``, so preserve the CF ``grid_mapping`` key.
        grid_mapping = geo[var].encoding.get("grid_mapping")
        if grid_mapping:
            geo_encoding[var]["grid_mapping"] = grid_mapping
    geo.to_netcdf(geo_file, encoding=geo_encoding, engine="h5netcdf")

    ice_mask = ds_bm["icemask_promice"]
    vel = ds_bm[["vx_mosaic", "vy_mosaic"]].rename_vars({"vx_mosaic": "vx", "vy_mosaic": "vy"})

    if target_grid is not None:
        vel = vel.regrid.conservative(target_grid)
        ice_mask = ice_mask.regrid.conservative(target_grid)

    ice_mask = ice_mask > 0.5

    # Integer GrIS basin mask (1..7, 0 outside) rasterized from the packaged
    # basin polygons onto the target grid.
    basins = basin_mask(target_grid)

    # Grounded ice via PISM's flotation criterion (src/util/Mask.hh): ice is
    # grounded where its base rests on the bed (not floating) and ice is present.
    #   hgrounded = bed + thickness;  hfloating = sea_level + alpha * thickness
    #   alpha = 1 - rho_ice / rho_sea_water;  floating if hfloating > hgrounded;
    #   ice_free if thickness <= ice_free_thickness_standard.
    # Uses the (regridded) boot geometry, which is on the same grid as ``vel``.
    bed = boot["bed"]
    thk = boot["thickness"]
    grounded_ice = (bed + thk >= sea_level + alpha * thk) & (thk > ice_free_thickness) & ice_mask

    vel["v"] = ((vel["vx"].fillna(0) ** 2 + vel["vy"].fillna(0) ** 2) ** 0.5).astype("float32")
    vel["u_observed"] = vel["vx"].fillna(0).astype("float32")
    vel["v_observed"] = vel["vy"].fillna(0).astype("float32")
    # zeta is FREE (0) where there is grounded ice and FIXED (1) elsewhere;
    # the misfit weight is the inverse (1 = trust obs on grounded ice, 0 = ignore).
    vel["zeta_fixed_mask"] = xr.where(grounded_ice, 0, 1).astype("int8")
    vel["zeta_fixed_mask"].attrs.update({"units": "1", "long_name": "tauc_unchanging integer mask (1=fixed)"})
    vel["vel_misfit_weight"] = xr.where(grounded_ice, 1, 0).astype("int8")
    vel["vel_misfit_weight"].attrs.update({"units": "1", "long_name": "misfit weight (1=trust obs, 0=ignore)"})
    vel["basins"] = basins
    # Constant prior for the till yield stress used by the PISM inverse run.
    vel["tauc_prior"] = xr.full_like(vel["v"], 1.4e5, dtype="float32")
    vel["tauc_prior"].attrs.update({"units": "Pa", "long_name": "prior till yield stress (tauc)"})

    vel = vel.rio.write_crs("EPSG:3413", grid_mapping_name="mapping").rio.write_coordinate_system()
    vel["x"].attrs.update(
        {
            "standard_name": "projection_x_coordinate",
            "long_name": "x coordinate of projection",
            "units": "m",
            "axis": "X",
        }
    )
    vel["y"].attrs.update(
        {
            "standard_name": "projection_y_coordinate",
            "long_name": "y coordinate of projection",
            "units": "m",
            "axis": "Y",
        }
    )
    vel["x"].encoding["_FillValue"] = None
    vel["y"].encoding["_FillValue"] = None
    for v in vel.data_vars:
        vel[v].attrs.pop("coordinates", None)
        vel[v].encoding.pop("coordinates", None)
        vel[v].attrs.pop("grid_mapping", None)
        for k in ("scale_factor", "add_offset", "AREA_OR_POINT"):
            vel[v].attrs.pop(k, None)
            vel[v].encoding.pop(k, None)
    drop_geotransform_attr(vel)
    vel = stamp_grid_mapping(vel, name="mapping")
    vel = vel.drop_vars(["crs", "spatial_ref"], errors="ignore")

    obs_file = output_path / Path(f"obs_{dem_year}_g{resolution}m_GreenlandObsISMIP7-v1.3.nc")
    vel_encoding: dict[str, dict[str, Any]] = {
        var: {"_FillValue": None} for var in list(vel.data_vars) + list(vel.coords)
    }
    for var in vel.data_vars:
        vel_encoding[var].update(comp)
        # A per-variable encoding dict passed to ``to_netcdf`` replaces the
        # variable's ``.encoding``, so the CF ``grid_mapping`` key set by
        # stamp_grid_mapping would be dropped. Carry it through explicitly.
        grid_mapping = vel[var].encoding.get("grid_mapping")
        if grid_mapping:
            vel_encoding[var]["grid_mapping"] = grid_mapping
    vel.to_netcdf(obs_file, encoding=vel_encoding, engine="h5netcdf")

    return {"boot_file": boot_file, "heatflux_file": geo_file, "obs_file": obs_file}


def prepare_calfin(
    output_path: Path | str,
    resolution: int,
    x_bnds: list | np.ndarray,
    y_bnds: list | np.ndarray,
    freq: str = "MS",
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


def _forcing_tasks(config: dict) -> list[tuple]:
    """
    Expand the setup TOML into one task per (GCM, pathway, forcing).

    Each ``pathway`` (``historical`` / ``ssp???``) is its own task; the
    caller decides which forward run pairs them up (see run.py where
    ``run_hist`` uses the historical file and ``run_proj`` uses the ssp
    file). ``end`` is inclusive per the setup TOML convention.

    ``source`` and ``version`` may sit at the GCM level (defaults for every
    pathway) with optional per-pathway overrides. A pathway may also carry
    ``fields = {climate = [...]}`` to override the ``[forcing]`` field list
    for that pathway alone (the ctrl pathways publish no ``mrro``). ``source`` maps each
    forcing to its subtree and (optionally) a per-forcing version label:
    the table form ``{climate = {dataset = "SDBN1-1000m", version = 3}}``
    carries both, while a plain string (``{climate = "SDBN1-1000m"}``)
    names just the subtree and takes the pathway/GCM-level ``version``.
    The dataset is the directory subpath between the pathway slot and the
    variable dir; the slash form (``"RACMO2.3p2-ERA/SDBN1-1000m"``) names a
    reanalysis-forced tree like OCX (see :func:`_split_source_spec`).
    ``version`` labels the merged output filenames only; the cloud sync
    discovers the actual per-field file versions by listing the
    source.coop tree (which tags versions per variable).

    Parameters
    ----------
    config : dict
        Parsed setup TOML with ``ice_sheet``, ``[gcms]`` and ``[forcing]``.

    Returns
    -------
    list of tuple
        ``(ice_sheet, gcm, forcing, version, pathway, start_year, end_year,
        short_hand, fields, source)`` per task.

    Raises
    ------
    ValueError
        When a forcing has no ``version`` at the forcing, pathway, or GCM
        level.
    """
    tasks = []
    for gcm, _gcm_config in config["gcms"].items():
        gcm_sources = _gcm_config.get("source", {})
        gcm_version = _gcm_config.get("version")
        for pathway, _pathway_config in _gcm_config.items():
            if not isinstance(_pathway_config, dict) or "start" not in _pathway_config:
                # GCM-level keys (``source``, ``version``) are not pathways.
                continue
            ice_sheet = config["ice_sheet"]
            pathway_version = _pathway_config.get("version", gcm_version)
            start_year = int(_pathway_config["start"])
            end_year = int(_pathway_config["end"])
            sources = {**gcm_sources, **_pathway_config.get("source", {})}
            # Per-pathway field list, for pathways that are published with a
            # different set of variables than the rest of the tree (the ctrl
            # runs carry no ``mrro``).
            field_overrides = _pathway_config.get("fields", {})
            for forcing, forcing_dict in config["forcing"].items():
                fields = field_overrides.get(forcing, forcing_dict["fields"])
                # Fall back to a legacy global ``[forcing] short_hand`` so
                # older setup TOMLs keep working.
                spec = sources.get(forcing, forcing_dict.get("short_hand", "none"))
                # Table form: {dataset = ..., version = ...} carries a
                # per-forcing version (climate and ocean are published on
                # independent version tracks); a plain string falls back to
                # the pathway/GCM-level version.
                spec_version = None
                if isinstance(spec, dict):
                    spec_version = spec.get("version")
                    spec = spec.get("dataset")
                version_value = spec_version if spec_version is not None else pathway_version
                if version_value is None:
                    raise ValueError(
                        f"[gcms] {gcm}/{pathway}/{forcing}: no 'version' at the forcing, pathway, or GCM level"
                    )
                version = "v" + str(version_value)
                source, short_hand = _split_source_spec(spec)
                tasks.append(
                    (
                        ice_sheet,
                        gcm,
                        forcing,
                        version,
                        pathway,
                        start_year,
                        end_year,
                        short_hand,
                        fields,
                        source,
                    )
                )
    return tasks


def prepare_ismip7_forcing(
    base_path: Path | str,
    output_path: Path | str,
    config: dict,
    data_path: Path | str | None = None,
    n_workers: int = 2,
    staging_path: Path | str | None = None,
) -> Sequence[Path | str]:
    """
    Process forcing data for all GCMs and forcings in parallel.

    Parameters
    ----------
    base_path : Path or str
        Root of the local download cache. Used only when ``data_path`` is
        ``None``: the files required by ``config`` are synced from the public
        source.coop mirror into a tree below it, and later runs re-download
        only files whose upstream ETag/size changed.
    output_path : Path or str
        Output directory. Only the final merged forcing files end up here.
    config : dict
        Configuration dictionary.
    data_path : Path or str or None, optional
        If given, read forcing files from this local mirror of the Globus
        tree instead of downloading. Layout is expected to match the
        Globus tree, with *data_path* either containing a ``GrIS/`` subdir
        or being that ``GrIS/`` directory itself.
    n_workers : int, optional
        Number of dask workers, by default 2.
    staging_path : Path or str or None, optional
        Directory for intermediate scratch (per-variable cdo tmps and the
        per-epoch hist/proj outputs). The whole staging tree is removed
        after each forcing finishes, so the only artifact left on disk is
        the final merged file in ``output_path``. Defaults to
        ``output_path`` when omitted, matching the legacy behavior.

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
    if staging_path is not None:
        staging_path = Path(staging_path)
        staging_path.mkdir(parents=True, exist_ok=True)

    ismip7_to_pism = config["ismip7_to_pism"]
    tasks = _forcing_tasks(config)

    # Process in parallel using dask.distributed
    with Client(n_workers=n_workers, threads_per_worker=1) as client:
        logger.info("Dask dashboard: %s", client.dashboard_link)

        futures = []
        for (
            ice_sheet,
            gcm,
            forcing,
            version,
            pathway,
            start_year,
            end_year,
            short_hand,
            fields,
            source,
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
                start_year,
                end_year,
                short_hand,
                fields,
                ismip7_to_pism,
                data_path=data_path,
                staging_path=staging_path,
                source=source,
            )
            futures.append(future)

        # Collect results as they complete
        processed_files = []
        for future in as_completed(futures):
            output_files = future.result()
            logger.info("Completed: %s", output_files)
            processed_files.extend(output_files)

    # Stamp the GrIS basin mask (from the packaged polygons) onto every generated
    # ocean forcing file, after the dask loop.
    ocean_files = [Path(f) for f in processed_files if "_ocean_" in Path(f).name]
    if ocean_files:
        logger.info("Adding basin mask to %d ocean forcing file(s)", len(ocean_files))
        add_basins_to_ocean_files(ocean_files)

    elapsed = time.perf_counter() - start_time
    logger.info("Total processing time: %.2f seconds", elapsed)

    return processed_files
