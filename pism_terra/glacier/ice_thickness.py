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

# pylint: disable=too-many-positional-arguments,broad-exception-caught

"""
Prepare ice thickness.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed as cf_as_completed
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import xarray as xr
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform_bounds,
)
from shapely.geometry import box
from tqdm.auto import tqdm

from pism_terra.aws import download_from_s3, s3_to_local
from pism_terra.download import download_archive, extract_archive
from pism_terra.raster import check_overlap
from pism_terra.workflow import check_xr_lazy

logger = logging.getLogger(__name__)


def _load_rgi6_links(rgi_extract_path: Path) -> pd.DataFrame:
    """
    Build a global RGI6 → RGI7-G ID lookup from RGI7 G-archive link CSVs.

    Each RGI7 G regional zip ships a ``*-rgi6_links.csv`` that maps RGI7
    glacier IDs (``RGI2000-v7.0-G-XX-YYYYY``) to their RGI6 counterparts
    (``RGI60-XX.YYYYY``). This helper finds every such CSV under
    *rgi_extract_path*, normalizes the column names, and returns a single
    concatenated DataFrame with columns ``rgi_id`` (RGI7) and ``rgi6_id``.

    Parameters
    ----------
    rgi_extract_path : pathlib.Path
        Directory containing extracted RGI7 G archives. Searched recursively
        for files matching ``*rgi6_links*.csv``.

    Returns
    -------
    pandas.DataFrame
        Two-column frame ``["rgi_id", "rgi6_id"]``. Rows with missing IDs are
        dropped. Empty if no link files are found.
    """
    csvs = sorted(rgi_extract_path.rglob("*rgi6_links*.csv"))
    if not csvs:
        return pd.DataFrame(columns=["rgi_id", "rgi6_id"])

    # Column names vary across RGI7 releases; accept the common spellings.
    rgi7_aliases = ["rgi_id", "rgi7_id", "rgi70_id", "RGIId", "rgi_id_v7"]
    rgi6_aliases = ["rgi6_id", "rgi60_id", "RGIId_v6", "rgi_id_v6", "v6_id"]

    frames = []
    for csv in csvs:
        df = pd.read_csv(csv)
        rgi7_col = next((c for c in rgi7_aliases if c in df.columns), None)
        rgi6_col = next((c for c in rgi6_aliases if c in df.columns), None)
        if rgi7_col is None or rgi6_col is None:
            logger.warning(
                "rgi6_links file %s has unexpected columns %s; skipping",
                csv,
                list(df.columns),
            )
            continue
        frames.append(df[[rgi7_col, rgi6_col]].rename(columns={rgi7_col: "rgi_id", rgi6_col: "rgi6_id"}))

    if not frames:
        return pd.DataFrame(columns=["rgi_id", "rgi6_id"])

    links = pd.concat(frames, ignore_index=True)
    links = links.dropna(subset=["rgi_id", "rgi6_id"])
    links["rgi_id"] = links["rgi_id"].astype(str)
    links["rgi6_id"] = links["rgi6_id"].astype(str)
    return links


def prepare_ice_thickness_frank(
    regions: list,
    complexes: gpd.GeoDataFrame,
    glaciers: gpd.GeoDataFrame,
    output_path: Path | str,
    extract_path: Path | str,
    rgi_extract_path: Path | str | None = None,
    ntasks: int = 8,
    force_overwrite: bool = False,
):
    """
    Download Frank et al. ice thickness and merge into RGI7 complex rasters.

    The Frank dataset is keyed on RGI6 glacier IDs; this routine pulls the
    single Figshare archive and matches each per-glacier ``.tif`` to its
    parent RGI7 complex by **spatial intersection** of the tif footprint
    with the complex outline. The RGI v6 and v7 numbering schemes are
    unrelated, and the per-region ``rgi6_links.csv`` files shipped with
    RGI7 are a sparse subset, so an ID-based join cannot recover the full
    mapping.

    Parameters
    ----------
    regions : list
        Region codes (e.g. ``["01", "06"]``).
    complexes : geopandas.GeoDataFrame
        Complex outlines with ``rgi_id``, ``crs``, ``geometry``, and (for
        regular complexes) ``o1region``. Aggregates (no ``o1region``) are
        merged in a second pass.
    glaciers : geopandas.GeoDataFrame
        Glacier outlines with ``rgi_id``, ``rgi_id_c``, ``o1region``, and
        (optionally) ``rgi_id_c_aggregate``. Used only to resolve aggregate
        parent complexes; the per-complex match is purely spatial.
    output_path : Path or str
        Root directory for the per-complex / per-aggregate output rasters.
    extract_path : Path or str
        Working directory for the Frank archive (``Thk.zip``) and its
        extracted per-RGI6 ``.tif`` files.
    rgi_extract_path : Path or str or None, optional
        Unused; kept for signature parity with :func:`prepare_ice_thickness_maffezzoli`.
    ntasks : int, default 8
        Maximum number of parallel workers for the merge phases.
    force_overwrite : bool, default False
        If True, re-download / re-extract / re-merge.
    """

    # Frank et al. (2025) global glacier ice thickness — single archive at
    # Springer Nature Figshare (file id 57288257), RGI6-keyed per-glacier tifs.
    url: str = "https://springernature.figshare.com/ndownloader/files/57288257"
    extract_path = Path(extract_path)
    output_path = Path(output_path)
    extract_path.mkdir(parents=True, exist_ok=True)
    _ = rgi_extract_path  # unused; kept for signature parity

    archive_dest = extract_path / "Thk.zip"
    logger.info("Downloading Frank ice thickness from %s", url)
    archive = download_archive(url, dest=archive_dest, force_overwrite=force_overwrite, verbose=True)

    frank_dir = extract_path / "frank"
    frank_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s into %s", archive, frank_dir)
    extract_archive(archive, frank_dir, force_overwrite=force_overwrite, verbose=True)

    # Frank's Thk.zip is a zip-of-zips: a per-region ``RGI-NN_thk.zip`` for
    # each o1 region. Extract only the regions in scope so we don't unpack
    # the whole world when, say, only Alaska is requested.
    region_codes = {str(r).zfill(2) for r in regions}
    for inner_zip in sorted(frank_dir.glob("RGI-*_thk.zip")):
        # Filename: RGI-NN_thk.zip — pull NN out of the stem.
        m = re.match(r"^RGI-(\d{2})_thk$", inner_zip.stem)
        if not m or m.group(1) not in region_codes:
            continue
        out_dir = frank_dir / inner_zip.stem
        if out_dir.exists() and not force_overwrite and any(out_dir.rglob("RGI60-*.tif")):
            continue
        logger.info("Extracting per-region %s into %s", inner_zip.name, out_dir)
        extract_archive(inner_zip, out_dir, force_overwrite=force_overwrite, verbose=True)

    # Index the extracted Frank tifs by RGI6 id. The RGI7-shipped
    # ``rgi6_links.csv`` files are a sparse subset (only the renumberings
    # with non-trivial overlap) and the v6/v7 numbering schemes are
    # unrelated, so an id-based join can't recover the full mapping.
    # Instead, build a spatial index over Frank's per-glacier footprints
    # and find the tifs that physically overlap each RGI7 complex.
    frank_tifs: dict[str, Path] = {p.stem.removesuffix("_thk"): p for p in frank_dir.rglob("RGI60-*.tif")}
    if not frank_tifs:
        logger.error("No RGI60-*.tif files found under %s after extraction", frank_dir)
        return
    logger.info("Indexed %d Frank per-glacier tifs", len(frank_tifs))

    def _frank_footprint_4326(item):
        """
        Read a Frank tif's bounding box and reproject it to EPSG:4326.

        Parameters
        ----------
        item : tuple of (str, pathlib.Path)
            ``(rgi6_id, path)`` pair from the indexed Frank tif dict.

        Returns
        -------
        dict or None
            Mapping with ``rgi6_id`` and ``geometry`` (a Polygon in
            EPSG:4326), or ``None`` if the tif could not be read.
        """

        rgi6_id, tif_path = item
        try:
            with rasterio.open(tif_path) as src:
                if src.crs is None:
                    return None
                b = src.bounds
                if src.crs.to_epsg() == 4326:
                    poly = box(b.left, b.bottom, b.right, b.top)
                else:
                    poly = box(*transform_bounds(src.crs, "EPSG:4326", *b))
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        return {"rgi6_id": rgi6_id, "geometry": poly}

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, ntasks)) as executor:
        for rec in tqdm(
            executor.map(_frank_footprint_4326, frank_tifs.items()),
            total=len(frank_tifs),
            desc="Indexing Frank footprints",
        ):
            if rec is not None:
                records.append(rec)
    frank_footprints = gpd.GeoDataFrame(records, crs="EPSG:4326")
    logger.info("Built spatial index over %d Frank footprints", len(frank_footprints))

    # Project complex outlines to 4326 once so the per-complex intersect
    # call is cheap. The .sindex on frank_footprints accelerates lookups.
    complexes_4326 = complexes.to_crs("EPSG:4326")
    _ = frank_footprints.sindex  # warm the rtree

    def _reproject_and_merge(input_files, dst_crs, output_file, tmp_label):
        """
        Reproject inputs to ``dst_crs`` and write their mosaic to ``output_file``.

        Parameters
        ----------
        input_files : list[Path]
            Source GeoTIFFs to merge.
        dst_crs : str
            Target CRS string (e.g. ``"EPSG:32606"``).
        output_file : Path
            Target path for the merged GeoTIFF.
        tmp_label : str
            Label embedded in temporary reproject filenames so concurrent
            tasks operating on the same source raster don't collide.

        Returns
        -------
        Path
            ``output_file`` on success.
        """
        reprojected: list[Path] = []
        for fpath in input_files:
            with rasterio.open(fpath) as src:
                if src.crs == rasterio.crs.CRS.from_user_input(dst_crs):
                    reprojected.append(fpath)
                    continue
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                meta = src.meta.copy()
                meta.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})
                tmp_path = fpath.parent / f"{fpath.stem}__{tmp_label}_reproj.tif"
                with rasterio.open(tmp_path, "w", **meta) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                    )
                reprojected.append(tmp_path)

        # Frank tifs are per-glacier and their bounding boxes overlap at
        # adjacent-glacier edges. Two quirks dictate the merge call:
        #   1. The declared nodata (9999) never appears in the actual data
        #      — Frank uses literal 0 for "outside the v6 outline" cells.
        #      Overriding ``nodata=0`` here treats those zero borders as
        #      transparent during merge.
        #   2. ``method="max"`` then picks the larger value across any
        #      remaining overlap (where two adjacent tifs both have real
        #      thickness near a shared edge).
        # Without (1), a neighbor's bbox 0s overwrite real thickness on the
        # downstream side of a glacier, producing the visible "cut" pattern.
        mosaic, out_transform = merge(reprojected, method="max", nodata=0)
        with rasterio.open(reprojected[0]) as src:
            out_meta = src.meta.copy()
        predictor = 3 if np.issubdtype(mosaic.dtype, np.floating) else 2
        out_meta.update(
            {
                "driver": "COG",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "compress": "DEFLATE",
                "predictor": predictor,
                "level": 6,
                "blocksize": 512,
                "overview_resampling": "AVERAGE",
                "BIGTIFF": "YES",
                "num_threads": "ALL_CPUS",
            }
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(mosaic)

        for fpath in reprojected:
            if fpath.stem.endswith("_reproj"):
                fpath.unlink(missing_ok=True)
        return output_file

    def _merge_complex(rgi_c_id, o1region, dst_crs):
        """
        Mosaic Frank rasters for every RGI6 glacier inside one RGI7 complex.

        Parameters
        ----------
        rgi_c_id : str
            RGI7 complex identifier (e.g. ``"RGI2000-v7.0-C-01-09429"``).
        o1region : str or int
            Region code of the complex; used to build the output subdirectory.
        dst_crs : str
            Target CRS string for the merged output (e.g. ``"EPSG:32606"``).

        Returns
        -------
        pathlib.Path or None
            Path to the merged GeoTIFF, or ``None`` if no Frank rasters
            resolved to this complex.
        """
        row = complexes_4326.loc[complexes_4326["rgi_id"] == rgi_c_id]
        if row.empty:
            logger.warning("Complex %s not found in complexes table; skipping", rgi_c_id)
            return None
        geom = row.geometry.iloc[0]
        matched = frank_footprints[frank_footprints.intersects(geom)]
        if matched.empty:
            logger.warning("Complex %s has no overlapping Frank tifs; skipping", rgi_c_id)
            return None
        files = [frank_tifs[r] for r in matched["rgi6_id"].tolist() if r in frank_tifs]
        merged_path = output_path / Path(f"RGI2000-v7.0-C-{str(o1region).zfill(2)}")
        merged_file = merged_path / Path(f"{rgi_c_id}_thickness.tif")
        return _reproject_and_merge(files, dst_crs, merged_file, rgi_c_id)

    def _merge_aggregate(agg_id, dst_crs):
        """
        Mosaic the phase-1 per-complex outputs for an aggregate (e.g. ``S4F_AK``).

        Parameters
        ----------
        agg_id : str
            Aggregate identifier (e.g. ``"S4F_AK"``).
        dst_crs : str
            Target CRS string for the merged output.

        Returns
        -------
        pathlib.Path or None
            Path to the merged aggregate GeoTIFF, or ``None`` if no parent
            phase-1 thickness files were found.
        """
        agg_col = glaciers.get("rgi_id_c_aggregate")
        if agg_col is None:
            return None
        sel = agg_col.fillna("").str.split(";").apply(lambda parts: agg_id in parts)
        parent_ids = glaciers.loc[sel, "rgi_id_c"].dropna().unique()
        if len(parent_ids) == 0:
            logger.debug("No parent complexes resolved for aggregate %s", agg_id)
            return None
        parent_files: list[Path] = []
        for parent_id in parent_ids:
            for o1_dir in output_path.glob("RGI2000-v7.0-C-*"):
                p = o1_dir / f"{parent_id}_thickness.tif"
                if p.exists():
                    parent_files.append(p)
                    break
        if not parent_files:
            logger.debug("No phase-1 thickness files for aggregate %s", agg_id)
            return None
        merged_file = output_path / Path(agg_id) / Path(f"{agg_id}_thickness.tif")
        return _reproject_and_merge(parent_files, dst_crs, merged_file, agg_id)

    # Same two-phase scheduling as Maffezzoli: regular complexes first
    # (they only depend on the Frank tifs), then aggregates (which depend
    # on the phase-1 per-complex outputs).
    regular_tasks: list[tuple[str, object, str]] = []
    aggregate_tasks: list[tuple[str, str]] = []
    for _, row in complexes.iterrows():
        crs_value = row.get("crs")
        if not isinstance(crs_value, str) or not crs_value:
            logger.warning("Complex %s has no CRS; skipping", row["rgi_id"])
            continue
        if pd.notna(row.get("o1region")):
            regular_tasks.append((row["rgi_id"], row.get("o1region"), crs_value))
        else:
            aggregate_tasks.append((row["rgi_id"], crs_value))

    failed: list[tuple[str, Exception]] = []
    if regular_tasks:
        with ThreadPoolExecutor(max_workers=min(ntasks, max(1, len(regular_tasks)))) as executor:
            futures = {
                executor.submit(_merge_complex, rgi_c_id, o1region, dst_crs): rgi_c_id
                for rgi_c_id, o1region, dst_crs in regular_tasks
            }
            for future in tqdm(cf_as_completed(futures), total=len(futures), desc="Merging Frank thickness"):
                rgi_c_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed.append((rgi_c_id, exc))

    if aggregate_tasks:
        with ThreadPoolExecutor(max_workers=min(ntasks, max(1, len(aggregate_tasks)))) as executor:
            futures = {
                executor.submit(_merge_aggregate, agg_id, dst_crs): agg_id for agg_id, dst_crs in aggregate_tasks
            }
            for future in tqdm(cf_as_completed(futures), total=len(futures), desc="Merging Frank aggregates"):
                agg_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed.append((agg_id, exc))

    for rgi_c_id, err in failed:
        logger.error("Failed merging %s: %s", rgi_c_id, err)
    logger.info("Frank ice thickness merging complete")
    _ = regions  # listed for parity with the Maffezzoli signature


def prepare_ice_thickness_maffezzoli(
    regions: list,
    complexes: gpd.GeoDataFrame,
    glaciers: gpd.GeoDataFrame,
    output_path: Path | str,
    extract_path: Path | str,
    ntasks: int = 8,
    force_overwrite: bool = False,
):
    """
    Download Maffezzoli (IceBoost) ice thickness and clip into RGI7 complex rasters.

    The IceBoost v2.0 "complexes" dataset is computed directly on the RGI7
    glacier *complex* outlines and shipped as a single archive of per-region,
    per-UTM-zone mosaics (``complex/rgi{N}/iceboost_..._epsg_{EPSG}.tif``).
    Because the thickness is already keyed to RGI7 complexes, no per-glacier
    merging or RGI6/RGI7 reconciliation is needed: each complex raster is just
    the slice of its region's mosaic(s) that overlaps the complex outline.

    Parameters
    ----------
    regions : list
        Region codes (e.g. ``["01", "06"]``). Used to limit which region
        mosaics are indexed from the archive.
    complexes : geopandas.GeoDataFrame
        Complex outlines with ``rgi_id``, ``crs``, ``geometry``, and (for
        regular complexes) ``o1region``. Aggregates (no ``o1region``) are
        merged in a second pass from the per-complex outputs.
    glaciers : geopandas.GeoDataFrame
        Glacier outlines with ``rgi_id_c`` and ``rgi_id_c_aggregate`` columns.
        Used only to resolve aggregate parent complexes in the second pass.
    output_path : Path or str
        Root directory for the per-complex / per-aggregate output rasters.
    extract_path : Path or str
        Working directory for the IceBoost archive and its extracted mosaics.
    ntasks : int, default 8
        Maximum number of parallel workers for the clip / merge phases.
    force_overwrite : bool, default False
        If True, re-download / re-extract / re-clip.
    """

    # Maffezzoli et al. IceBoost v2.0 global glacier ice thickness, computed on
    # RGI7 complex outlines — single Zenodo archive of per-region/per-UTM-zone
    # mosaics (record 20463551).
    url: str = "https://zenodo.org/records/20463551/files/IceBoostv20_complexes_RGI_7.zip?download=1"
    extract_path = Path(extract_path)
    output_path = Path(output_path)
    extract_path.mkdir(parents=True, exist_ok=True)

    archive_dest = extract_path / "IceBoostv20_complexes_RGI_7.zip"
    logger.info("Downloading Maffezzoli (IceBoost) ice thickness from %s", url)
    archive = download_archive(url, dest=archive_dest, force_overwrite=force_overwrite, verbose=True)

    maff_dir = extract_path / "maffezzoli"
    maff_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s into %s", archive, maff_dir)
    extract_archive(archive, maff_dir, force_overwrite=force_overwrite, verbose=True)

    # Restrict the index to the regions in scope so we don't open mosaics for
    # the whole world when only a few regions were requested.
    requested_regions: set[int] = set()
    for r in regions:
        try:
            requested_regions.add(int(str(r)))
        except (TypeError, ValueError):
            continue

    # Index the extracted mosaics by RGI primary region. Each entry records the
    # tif path, its CRS, and its footprint (a box in its own CRS) so the
    # per-complex overlap test is a cheap geometry intersection.
    mosaic_re = re.compile(r"_rgi(\d+)_v70G_epsg_(\d+)\.tif$")
    region_mosaics: dict[int, list[tuple[Path, object, object]]] = {}
    for tif in sorted(maff_dir.rglob("iceboost_*_rgi*_epsg_*.tif")):
        m = mosaic_re.search(tif.name)
        if not m:
            continue
        region = int(m.group(1))
        if requested_regions and region not in requested_regions:
            continue
        with rasterio.open(tif) as src:
            region_mosaics.setdefault(region, []).append((tif, src.crs, box(*src.bounds)))
    if not region_mosaics:
        logger.error("No IceBoost mosaics found under %s after extraction", maff_dir)
        return
    logger.info("Indexed IceBoost mosaics for regions %s", sorted(region_mosaics))

    # Project complex outlines to 4326 once; each per-mosaic overlap test then
    # reprojects the single complex geometry into that mosaic's CRS.
    complexes_4326 = complexes.to_crs("EPSG:4326")

    def _reproject_and_merge(input_files, dst_crs, output_file, tmp_label):
        """
        Reproject inputs to ``dst_crs`` and write their mosaic to ``output_file``.

        Parameters
        ----------
        input_files : list[Path]
            Source GeoTIFFs to merge.
        dst_crs : str
            Target CRS string (e.g. ``"EPSG:32606"``).
        output_file : Path
            Target path for the merged GeoTIFF.
        tmp_label : str
            Label embedded in temporary reproject filenames so concurrent
            tasks operating on the same source raster don't collide.

        Returns
        -------
        Path
            ``output_file`` on success.
        """
        reprojected: list[Path] = []
        for fpath in input_files:
            with rasterio.open(fpath) as src:
                if src.crs == rasterio.crs.CRS.from_user_input(dst_crs):
                    reprojected.append(fpath)
                    continue
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                meta = src.meta.copy()
                meta.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})
                tmp_path = fpath.parent / f"{fpath.stem}__{tmp_label}_reproj.tif"
                with rasterio.open(tmp_path, "w", **meta) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                    )
                reprojected.append(tmp_path)

        mosaic, out_transform = merge(reprojected)
        with rasterio.open(reprojected[0]) as src:
            out_meta = src.meta.copy()
        # Cloud-Optimized GeoTIFF: tiled + DEFLATE + overviews, range-readable
        # from S3 (so QGIS via /vsis3/ streams without a full download).
        # predictor=3 for floats, 2 for ints. BIGTIFF=YES for >4 GB outputs.
        predictor = 3 if np.issubdtype(mosaic.dtype, np.floating) else 2
        out_meta.update(
            {
                "driver": "COG",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "compress": "DEFLATE",
                "predictor": predictor,
                "level": 6,
                "blocksize": 512,
                "overview_resampling": "AVERAGE",
                "BIGTIFF": "YES",
                "num_threads": "ALL_CPUS",
            }
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(mosaic)

        for fpath in reprojected:
            if fpath.stem.endswith("_reproj"):
                fpath.unlink(missing_ok=True)
        return output_file

    def _clip_to_geom(mosaic_path, geom, tmp_path):
        """
        Crop the thickness band of a mosaic to ``geom`` and write the window.

        IceBoost mosaics are multi-band (``thickness``, ``thickness_err``,
        ``jensen_gap``, ``h_wgs84``, ``n_geoid``); only band 1 (``thickness``)
        is kept so the per-complex output stays single-band like the rest of
        the pipeline expects.

        Parameters
        ----------
        mosaic_path : pathlib.Path
            Source regional mosaic GeoTIFF.
        geom : shapely.geometry.base.BaseGeometry
            Complex outline expressed in *mosaic_path*'s CRS.
        tmp_path : pathlib.Path
            Destination for the cropped GeoTIFF.

        Returns
        -------
        pathlib.Path or None
            ``tmp_path`` on success, or ``None`` if ``geom`` does not overlap
            the raster (``rasterio.mask`` raises in that case).
        """
        with rasterio.open(mosaic_path) as src:
            try:
                out_image, out_transform = rio_mask(src, [geom], crop=True, all_touched=True, indexes=[1])
            except ValueError:
                return None  # geometry doesn't overlap this mosaic
            out_meta = src.meta.copy()
        out_meta.update(
            {"count": 1, "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform}
        )
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(tmp_path, "w", **out_meta) as dst:
            dst.write(out_image)
        return tmp_path

    def _merge_complex(rgi_c_id, o1region, dst_crs):
        """
        Build a per-complex thickness raster by clipping the region mosaic(s).

        Phase-1: find the region's IceBoost mosaic(s) that overlap the complex
        outline, crop each to the complex extent, and reproject/mosaic the
        crop(s) into one per-complex GeoTIFF in ``dst_crs``.

        Parameters
        ----------
        rgi_c_id : str
            RGI7 complex identifier (e.g. ``"RGI2000-v7.0-C-01-09429"``).
        o1region : str or int
            Region code of the complex; used to build the output subdirectory.
        dst_crs : str
            Target CRS string for the merged output (e.g. ``"EPSG:32606"``).

        Returns
        -------
        pathlib.Path or None
            Path to the merged GeoTIFF, or ``None`` if no mosaic overlapped
            this complex.
        """
        candidates = region_mosaics.get(int(o1region), [])
        if not candidates:
            logger.warning("No IceBoost mosaics for region %s (complex %s); skipping", o1region, rgi_c_id)
            return None
        row = complexes_4326.loc[complexes_4326["rgi_id"] == rgi_c_id]
        if row.empty:
            logger.warning("Complex %s not found in complexes table; skipping", rgi_c_id)
            return None
        geom_4326 = row.geometry.iloc[0]

        clipped_files: list[Path] = []
        for idx, (mosaic_path, mcrs, mbounds) in enumerate(candidates):
            geom_m = gpd.GeoSeries([geom_4326], crs="EPSG:4326").to_crs(mcrs).iloc[0]
            if not geom_m.intersects(mbounds):
                continue
            tmp_path = mosaic_path.parent / f"{rgi_c_id}__clip{idx}.tif"
            clip = _clip_to_geom(mosaic_path, geom_m, tmp_path)
            if clip is not None:
                clipped_files.append(clip)
        if not clipped_files:
            logger.warning("Complex %s does not overlap any IceBoost mosaic; skipping", rgi_c_id)
            return None

        merged_path = output_path / Path(f"RGI2000-v7.0-C-{str(o1region).zfill(2)}")
        merged_file = merged_path / Path(f"{rgi_c_id}_thickness.tif")
        try:
            return _reproject_and_merge(clipped_files, dst_crs, merged_file, rgi_c_id)
        finally:
            for fpath in clipped_files:
                fpath.unlink(missing_ok=True)

    def _merge_aggregate(agg_id, dst_crs):
        """
        Build an aggregate thickness raster from per-complex outputs.

        Phase-2: looks up the parent complexes that own glaciers tagged with
        ``agg_id`` (via the ``rgi_id_c_aggregate`` column), reads each parent's
        already-produced ``<parent>_thickness.tif`` from phase 1, and
        reprojects/mosaics them into one aggregate GeoTIFF.

        Parameters
        ----------
        agg_id : str
            Aggregate identifier (e.g. ``"S4F_AK"``).
        dst_crs : str
            Target CRS string for the merged output.

        Returns
        -------
        pathlib.Path or None
            Path to the merged GeoTIFF, or ``None`` if no parent thickness
            files (from phase 1) were found on disk.
        """
        agg_col = glaciers.get("rgi_id_c_aggregate")
        if agg_col is None:
            return None
        sel = agg_col.fillna("").str.split(";").apply(lambda parts: agg_id in parts)
        parent_ids = glaciers.loc[sel, "rgi_id_c"].dropna().unique()
        if len(parent_ids) == 0:
            logger.debug("No parent complexes resolved for aggregate %s", agg_id)
            return None

        parent_files: list[Path] = []
        for parent_id in parent_ids:
            # Per-region subdirs are output_path/RGI2000-v7.0-C-XX/<parent>_thickness.tif
            for o1_dir in output_path.glob("RGI2000-v7.0-C-*"):
                p = o1_dir / f"{parent_id}_thickness.tif"
                if p.exists():
                    parent_files.append(p)
                    break
        if not parent_files:
            logger.debug("No parent thickness files found for aggregate %s", agg_id)
            return None

        merged_file = output_path / Path(agg_id) / Path(f"{agg_id}_thickness.tif")
        return _reproject_and_merge(parent_files, dst_crs, merged_file, agg_id)

    # Split tasks: regular complexes (have an o1region) run first; aggregates
    # depend on their parents' merged outputs and run after.
    regular_tasks: list[tuple[str, object, str]] = []
    aggregate_tasks: list[tuple[str, str]] = []
    for _, row in complexes.iterrows():
        crs_value = row.get("crs")
        if not isinstance(crs_value, str) or not crs_value:
            logger.warning("Complex %s has no CRS; skipping", row["rgi_id"])
            continue
        if pd.notna(row.get("o1region")):
            regular_tasks.append((row["rgi_id"], row.get("o1region"), crs_value))
        else:
            aggregate_tasks.append((row["rgi_id"], crs_value))

    failed: list[tuple[str, Exception]] = []
    if regular_tasks:
        with ThreadPoolExecutor(max_workers=min(ntasks, max(1, len(regular_tasks)))) as executor:
            futures = {
                executor.submit(_merge_complex, rgi_c_id, o1region, dst_crs): rgi_c_id
                for rgi_c_id, o1region, dst_crs in regular_tasks
            }
            for future in tqdm(cf_as_completed(futures), total=len(futures), desc="Clipping ice thickness"):
                rgi_c_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed.append((rgi_c_id, exc))

    if aggregate_tasks:
        with ThreadPoolExecutor(max_workers=min(ntasks, max(1, len(aggregate_tasks)))) as executor:
            futures = {
                executor.submit(_merge_aggregate, agg_id, dst_crs): agg_id for agg_id, dst_crs in aggregate_tasks
            }
            for future in tqdm(cf_as_completed(futures), total=len(futures), desc="Merging aggregates"):
                agg_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed.append((agg_id, exc))

    for rgi_c_id, err in failed:
        logger.error("Failed merging %s: %s", rgi_c_id, err)
    logger.info("Ice thickness merging complete")


def get_ice_thickness(
    rgi_id: str,
    target_grid: xr.Dataset | xr.DataArray,
    dataset: Literal["frank", "maffezzoli", "millan"] = "maffezzoli",
    path: str | Path = "input_files",
    **kwargs,
) -> xr.DataArray:
    """
    Prepare ice thickness data for a given glacier and target grid.

    This function dispatches to a dataset-specific loader to prepare an
    ice thickness field interpolated to the resolution and bounds of a
    specified target grid.

    Parameters
    ----------
    rgi_id : str
        Glacier identifier (e.g., ``"RGI2000-v7.0-C-06-00014"``) used to locate
        the relevant ice thickness tiles.
    target_grid : xarray.Dataset or xarray.DataArray
        Grid to which the output ice thickness will be interpolated.
    dataset : {"millan", "maffezzoli"}, default ``"maffezzoli"``
        The name of the ice thickness dataset to use.
    path : str or pathlib.Path, default ``"input_files"``
        Working directory used by helper routines to cache/write intermediate rasters/grids.
    **kwargs
        Additional keyword arguments passed to the dataset-specific function,
        e.g., `target_crs="EPSG:32641"`.

    Returns
    -------
    xarray.DataArray
        Ice thickness interpolated to the target grid.

    Raises
    ------
    NotImplementedError
        If the specified dataset is not supported.
    """
    logger.info("Getting ice thickness from dataset '%s'", dataset)
    if dataset == "frank":
        thickness = get_ice_thickness_frank(rgi_id, target_grid, path=path, **kwargs)
    elif dataset == "maffezzoli":
        thickness = get_ice_thickness_maffezzoli(rgi_id, target_grid, path=path, **kwargs)
    elif dataset == "millan":
        thickness = get_ice_thickness_millan(rgi_id, target_grid, path=path, **kwargs)
    else:
        raise NotImplementedError(f"Ice thickness dataset '{dataset}' not implemented.")
    thickness = thickness.where(thickness > 0, 0)
    thickness.attrs.update({"standard_name": "land_ice_thickness", "units": "m"})

    return thickness


def get_ice_thickness_frank(
    rgi_id: str, target_grid: xr.Dataset | xr.DataArray, path: str | Path = "input_files", **kwargs
):
    """
    Load and interpolate Frank et al. (2026) ice thickness data to a target grid.

    Parameters
    ----------
    rgi_id : str
        Glacier identifier (e.g., ``"RGI2000-v7.0-C-06-00014"``) used to locate
        the overlapping Maffezzoli thickness rasters.
    target_grid : xarray.Dataset or xarray.DataArray
        Target grid to which ice thickness should be interpolated.
    path : str or pathlib.Path, default ``"input_files"``
        Working directory used by helper routines to cache/write intermediate rasters/grids.
    **kwargs
        Additional keyword arguments. Must include:
        - target_crs : str
            CRS to reproject rasters to (e.g., "EPSG:32641").

    Returns
    -------
    xarray.DataArray
        Interpolated ice thickness field on the target grid.

    Notes
    -----
    - Uses `rioxarray` to load and project raster files.
    - Assumes a fixed reprojected resolution of 100 meters.
    """

    force_overwrite: bool = bool(kwargs.pop("force_overwrite", False))
    bucket: str = kwargs.pop("bucket", "pism-cloud-data")
    prefix: str = kwargs.pop("prefix", "glacier")

    out_dir = Path(path)
    thickness_file = out_dir / f"thickness_frank_{rgi_id}.nc"

    if (not check_xr_lazy(thickness_file)) or force_overwrite:
        thickness_file.unlink(missing_ok=True)

        # Regular complex IDs strip the trailing glacier-number segment to get
        # the per-region subdir; aggregate IDs (no hyphens) live under their
        # own name.
        region = "-".join(rgi_id.split("-")[:-1]) or rgi_id
        s3_uri = f"s3://{bucket}/{prefix}/ice_thickness/frank/{region}/{rgi_id}_thickness.tif"
        local_tif = out_dir / f"{rgi_id}_thickness.tif"
        print(f"Downloading Frank thickness from {s3_uri}", flush=True)
        download_from_s3(s3_uri, local_tif)

        logger.info("Reprojecting and aligning thickness to target grid")
        da = rxr.open_rasterio(local_tif).sel(band=1).drop_vars("band")
        thickness = da.rio.reproject_match(target_grid, resampling=Resampling.bilinear)
        thickness.rio.write_nodata(None, inplace=True)
        thickness.name = "thickness"
        thickness = thickness.rio.write_crs(target_grid.rio.crs).rio.write_grid_mapping()
        thickness.to_netcdf(thickness_file)
        logger.info("Thickness saved to %s", thickness_file)
        # Return the in-memory result; CRS doesn't always survive the netCDF
        # round-trip, and we just computed a CRS-correct value.
        return thickness

    logger.info("Using cached thickness file %s", thickness_file)
    with xr.open_dataset(thickness_file) as ds:
        src_crs = ds.rio.crs
        if src_crs is None and "spatial_ref" in ds.coords:
            sr = ds["spatial_ref"]
            src_crs = sr.attrs.get("crs_wkt") or sr.attrs.get("spatial_ref")
        thickness = ds["thickness"].load()
    if src_crs is None:
        # Last resort: trust target_grid's CRS (the cache was generated against
        # an aligned grid, so this is correct by construction).
        src_crs = target_grid.rio.crs
    thickness = thickness.rio.write_crs(src_crs)
    return thickness


def get_ice_thickness_maffezzoli(
    rgi_id: str, target_grid: xr.Dataset | xr.DataArray, path: str | Path = "input_files", **kwargs
):
    """
    Load and interpolate Maffezzoli et al. (2026) ice thickness data to a target grid.

    Parameters
    ----------
    rgi_id : str
        Glacier identifier (e.g., ``"RGI2000-v7.0-C-06-00014"``) used to locate
        the overlapping Maffezzoli thickness rasters.
    target_grid : xarray.Dataset or xarray.DataArray
        Target grid to which ice thickness should be interpolated.
    path : str or pathlib.Path, default ``"input_files"``
        Working directory used by helper routines to cache/write intermediate rasters/grids.
    **kwargs
        Additional keyword arguments. Must include:
        - target_crs : str
            CRS to reproject rasters to (e.g., "EPSG:32641").

    Returns
    -------
    xarray.DataArray
        Interpolated ice thickness field on the target grid.

    Notes
    -----
    - Uses `rioxarray` to load and project raster files.
    - Assumes a fixed reprojected resolution of 100 meters.
    """

    force_overwrite: bool = bool(kwargs.pop("force_overwrite", False))
    bucket: str = kwargs.pop("bucket", "pism-cloud-data")
    prefix: str = kwargs.pop("prefix", "glacier")

    out_dir = Path(path)
    thickness_file = out_dir / f"thickness_maffezzoli_{rgi_id}.nc"

    if (not check_xr_lazy(thickness_file)) or force_overwrite:
        thickness_file.unlink(missing_ok=True)

        # Regular complex IDs strip the trailing glacier-number segment to get
        # the per-region subdir; aggregate IDs (no hyphens) live under their
        # own name.
        region = "-".join(rgi_id.split("-")[:-1]) or rgi_id
        s3_uri = f"s3://{bucket}/{prefix}/ice_thickness/maffezzoli/{region}/{rgi_id}_thickness.tif"
        local_tif = out_dir / f"{rgi_id}_thickness.tif"
        print(f"Downloading Maffezzoli thickness from {s3_uri}", flush=True)
        download_from_s3(s3_uri, local_tif)

        logger.info("Reprojecting and aligning thickness to target grid")
        da = rxr.open_rasterio(local_tif).sel(band=1).drop_vars("band")
        thickness = da.rio.reproject_match(target_grid, resampling=Resampling.bilinear)
        thickness.rio.write_nodata(None, inplace=True)
        thickness.name = "thickness"
        thickness = thickness.rio.write_crs(target_grid.rio.crs).rio.write_grid_mapping()
        thickness.to_netcdf(thickness_file)
        logger.info("Thickness saved to %s", thickness_file)
        # Return the in-memory result; CRS doesn't always survive the netCDF
        # round-trip, and we just computed a CRS-correct value.
        return thickness

    logger.info("Using cached thickness file %s", thickness_file)
    with xr.open_dataset(thickness_file) as ds:
        src_crs = ds.rio.crs
        if src_crs is None and "spatial_ref" in ds.coords:
            sr = ds["spatial_ref"]
            src_crs = sr.attrs.get("crs_wkt") or sr.attrs.get("spatial_ref")
        thickness = ds["thickness"].load()
    if src_crs is None:
        # Last resort: trust target_grid's CRS (the cache was generated against
        # an aligned grid, so this is correct by construction).
        src_crs = target_grid.rio.crs
    thickness = thickness.rio.write_crs(src_crs)
    return thickness


def _millan_region_dir(rgi_id: str) -> str:
    """
    Map an RGI ID to its Millan region directory.

    Millan tiles are organised by RGI primary region under ``RGI-{N}/``,
    with the Central Asia regions (13–15) merged into ``RGI-13-15/``.

    Parameters
    ----------
    rgi_id : str
        Glacier identifier, e.g. ``"RGI2000-v7.0-C-01-04374"``.

    Returns
    -------
    str
        Millan region subdirectory name (e.g. ``"RGI-1"``, ``"RGI-13-15"``).
    """
    parts = rgi_id.split("-")
    if len(parts) < 4:
        raise ValueError(f"Cannot derive Millan region from RGI ID {rgi_id!r}")
    region = int(parts[3])
    if region in (13, 14, 15):
        return "RGI-13-15"
    return f"RGI-{region}"


def get_ice_thickness_millan(
    rgi_id: str, target_grid: xr.Dataset | xr.DataArray, path: str | Path = "input_files", **kwargs
):
    """
    Load and interpolate Millan et al. (2022) ice thickness data to a target grid.

    Downloads only the Millan tiles for the RGI primary region matching
    ``rgi_id``, filters them by overlap against the supplied glacier
    geometry, reprojects each overlapping tile onto ``target_grid`` and
    returns the per-pixel sum.

    Parameters
    ----------
    rgi_id : str
        Glacier identifier (e.g., ``"RGI2000-v7.0-C-01-04374"``) used to
        locate the per-region tile directory.
    target_grid : xarray.Dataset or xarray.DataArray
        Target grid to which ice thickness should be interpolated.
    path : str or pathlib.Path, default ``"input_files"``
        Working directory used by helper routines to cache/write intermediate rasters/grids.
    **kwargs
        Additional keyword arguments. Recognised keys:

        - geometries : geopandas.GeoSeries / GeoDataFrame / iterable
            Glacier outline used to filter overlapping tiles. Required.
        - bucket : str, default ``"pism-cloud-data"``
        - prefix : str, default ``"rgi/glacier"``
            S3 prefix; tiles are read from
            ``{prefix}/ice_thickness/millan/RGI-{region}/``.
        - force_overwrite : bool, default False

    Returns
    -------
    xarray.DataArray
        Interpolated and summed ice thickness field on the target grid.
    """

    force_overwrite: bool = bool(kwargs.pop("force_overwrite", False))
    bucket: str = kwargs.pop("bucket", "pism-cloud-data")
    prefix: str = kwargs.pop("prefix", "rgi/glacier")
    geometries = kwargs.pop("geometries", None)
    if geometries is None:
        raise ValueError("get_ice_thickness_millan requires `geometries` for tile overlap check")
    # raster_overlaps_glacier's GeoSeries branch has a missing to_crs (FIXME);
    # wrapping as a GeoDataFrame routes through the branch that handles CRS
    # correctly, since Millan tiles and the glacier are in different CRSes.
    if isinstance(geometries, gpd.GeoSeries):
        geometries = gpd.GeoDataFrame(geometry=geometries, crs=geometries.crs)

    out_dir = Path(path)
    thickness_file = out_dir / f"thickness_millan_{rgi_id}.nc"

    if (not check_xr_lazy(thickness_file)) or force_overwrite:

        thickness_file.unlink(missing_ok=True)
        region_dir = _millan_region_dir(rgi_id)
        s3_prefix = f"{prefix}/ice_thickness/millan/{region_dir}"
        local_root = Path("data/ice_thickness/millan") / region_dir
        logger.info("Downloading Millan thickness tiles from s3://%s/%s", bucket, s3_prefix)
        s3_to_local(bucket, prefix=s3_prefix, dest=local_root)
        ice_thickness_files = list(local_root.rglob("THICKNESS_*.tif"))
        logger.info("Found %d Millan thickness tiles in %s, checking overlap", len(ice_thickness_files), region_dir)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(check_overlap, tile, geometries) for tile in ice_thickness_files]
            overlapping_rasters = [f.result() for f in cf_as_completed(futures) if f.result() is not None]

        logger.info("Found %d overlapping rasters, reprojecting and interpolating", len(overlapping_rasters))
        thicknesses = []
        for k, p in enumerate(overlapping_rasters):
            if p is not None:
                da = rxr.open_rasterio(p).squeeze().drop_vars("band", errors="ignore")
                thickness = da.rio.reproject_match(target_grid, resampling=Resampling.bilinear)
                thickness.rio.write_nodata(None, inplace=True)
                thickness.name = "thickness"
                thickness["raster"] = k
                thicknesses.append(thickness)

        thickness = xr.concat(thicknesses, dim="raster").sum(dim="raster")
        thickness = thickness.rio.write_crs(target_grid.rio.crs).rio.write_grid_mapping()
        thickness.to_netcdf(thickness_file)
        logger.info("Millan thickness saved to %s", thickness_file)
        return thickness

    logger.info("Using cached thickness file %s", thickness_file)
    with xr.open_dataset(thickness_file) as ds:
        src_crs = ds.rio.crs
        if src_crs is None and "spatial_ref" in ds.coords:
            sr = ds["spatial_ref"]
            src_crs = sr.attrs.get("crs_wkt") or sr.attrs.get("spatial_ref")
        thickness = ds["thickness"].load()
    if src_crs is None:
        src_crs = target_grid.rio.crs
    thickness = thickness.rio.write_crs(src_crs)

    return thickness


def add_malaspina_bed(
    ds: xr.Dataset,
    target_crs: str,
    bed_file: str | Path = "data/ice_thickness/malaspina/malaspina_bed_3338.tif",
    outline_file: str | Path = "data/rgi/rgi-malaspina.shp",
) -> xr.Dataset:
    """
    Replace bed topography in a dataset using the Malaspina Glacier bed dataset.

    This function reads a GeoTIFF file containing bed topography data for the Malaspina Glacier,
    clips it to the glacier outline, reprojects it to match the target dataset's CRS, and
    replaces the corresponding region in the input dataset. It also updates the `thickness`
    and `surface` fields accordingly.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing at least the variables 'bed', 'thickness', and 'surface'.
    target_crs : str
        The target coordinate reference system (CRS) to use for reprojection (e.g., "EPSG:3413").
    bed_file : str or Path, optional
        Path to the GeoTIFF file containing the Malaspina bed topography.
        Default is "data/ice_thickness/malaspina/malaspina_bed_3338.tif".
    outline_file : str or Path, optional
        Path to the glacier outline shapefile used to clip the bed topography.
        Default is "data/rgi/rgi-malaspina.shp".

    Returns
    -------
    xr.Dataset
        Modified dataset with updated 'bed', 'thickness', and 'surface' fields within the Malaspina region.

    Notes
    -----
    - Bed values of -9999.0 are treated as nodata and replaced with NaN.
    - Replaces `bed` where new values are available and recalculates `thickness = surface - bed`.
    - Ensures that thickness and surface are non-negative.
    - Updates CF-convention attributes and CRS metadata.
    """

    outline = gpd.read_file(outline_file).to_crs(target_crs)
    da = (
        rxr.open_rasterio(bed_file, mask=True)
        .squeeze()
        .drop_vars("band", errors="ignore")
        .rio.reproject_match(ds["bed"])
    )
    clipped_da = da.rio.clip(outline.geometry, drop=False)
    clipped_da = clipped_da.where(clipped_da != -9999.0, other=np.nan).drop_vars("spatial_ref")
    ds["bed"] = xr.where(~np.isnan(clipped_da), clipped_da, ds["bed"], keep_attrs=True)
    ds["thickness"] = xr.where(~np.isnan(clipped_da), ds["surface"] - clipped_da, ds["thickness"], keep_attrs=True)

    ds["thickness"] = ds["thickness"].where(ds["thickness"] > 0.0, 0.0)
    ds["surface"] = ds["surface"].where(ds["thickness"] > 0.0, 0.0)
    ds["surface"].attrs.update({"standard_name": "surface_altitude", "units": "m"})

    ds["thickness"].attrs.update({"standard_name": "land_ice_thickness", "units": "m"})

    ds["bed"].attrs.update({"standard_name": "bedrock_altitude", "units": "m"})
    return ds
