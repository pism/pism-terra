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
Prepare RGI.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed as cf_as_completed
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm

from pism_terra.download import download_archive, extract_archive

logger = logging.getLogger(__name__)


def get_rgi_url(url_template: str, region: str, outline_type: str = "C") -> str:
    """
    Format the given URL template with the provided region and outline type.

    Parameters
    ----------
    url_template : str
        A string with `{region}` and `{outline_type}` placeholders to be replaced.
    region : str
        The region code to insert into the template.
    outline_type : str, optional
        The outline type ("C" or "G"). Defaults to "C".

    Returns
    -------
    str
        The formatted URL.
    """
    return url_template.format(region=region, outline_type=outline_type)


def prepare_rgi_region(
    region,
    outline_type: str = "C",
    url_template: str = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-C/RGI2000-v7.0-{outline_type}-{region}.zip",
    extract_path: Path | str = "rgi_archive",
    area_threshold: float = 1.0,
    force_overwrite: bool = False,
):
    """
    Download, extract, and preprocess a specific RGI region shapefile.

    This function downloads the zipped shapefile for a given RGI region, extracts it,
    filters out glaciers smaller than a specified area threshold, and computes the EPSG
    code and CRS for each glacier based on its UTM zone and latitude.

    Parameters
    ----------
    region : pandas.Series or Mapping
        Region descriptor with a ``"region"`` key (e.g., ``"01_alaska"``) used
        to fill in the URL template. May optionally include a ``"crs"`` entry
        to override the auto-derived UTM CRS.
    outline_type : str, optional
       Either C or G (complex or glacier).
    url_template : str, optional
        URL template containing a `{region}` placeholder. Defaults to the NSIDC RGI v7 template.
    extract_path : str or Path, optional
        Path to the directory where the archive will be extracted. Default is "rgi_archive".
    area_threshold : float, optional
        Minimum glacier area (in square kilometers) for inclusion in the result. Defaults to 1.0.
    force_overwrite : bool, default False
        If True, re-download the file even if it already exists locally.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame of glaciers in the region with an added column:
        - "crs": EPSG code string based on hemisphere and UTM zone (or the
          ``crs`` value from the supplied ``region`` if present).

    See Also
    --------
    download_archive : Downloads an archive from a URL.
    extract_archive : Extracts a zip archive to a directory.

    Notes
    -----
    This function assumes that each region shapefile includes columns `"utm_zone"` and `"cenlat"`.
    """

    extract_path = Path(extract_path)
    region_code = region["region"]
    url = get_rgi_url(url_template, region_code, outline_type)
    archive_dest = extract_path / Path(url.rsplit("/", 1)[-1])
    logger.info("Downloading RGI region %s (%s)", region_code, outline_type)
    archive = download_archive(url, dest=archive_dest, force_overwrite=force_overwrite, verbose=False)
    logger.info("Extracting RGI region %s (%s)", region_code, outline_type)
    extract_archive(archive, extract_path, force_overwrite=force_overwrite, verbose=False)

    logger.info("Processing RGI region %s (%s)", region_code, outline_type)
    rgi = gpd.read_file(extract_path / f"RGI2000-v7.0-{outline_type}-{region_code}.shp")
    rgi = rgi[rgi["area_km2"] > area_threshold]
    crs_value = region.get("crs") if hasattr(region, "get") else None
    if isinstance(crs_value, str) and crs_value:
        rgi["crs"] = crs_value
    else:
        rgi["crs"] = rgi.apply(
            lambda row: f"""EPSG:{32600 + int(row["utm_zone"]) if row["cenlat"] >= 0 else 32700 + int(row["utm_zone"])}""",
            axis=1,
        )
    return rgi


def prepare_rgi(
    regions: pd.DataFrame,
    output_path: Path,
    glaciers: pd.DataFrame | list[str] | None = None,
    extract_path: Path | str = "rgi_archive",
    force_overwrite: bool = False,
    ntasks: int = 8,
):
    """
    Download, extract, and merge RGI region shapefiles for complex and glacier outlines.

    For each region and outline type (C and G), the corresponding shapefile is
    downloaded in parallel, filtered, and concatenated.  Each glacier outline (G)
    is then spatially joined to its parent complex outline (C) via a per-region
    spatial join, and the results are saved as GeoPackage files.

    Parameters
    ----------
    regions : pandas.DataFrame
        Table of regions with at least a ``"region"`` column whose values are
        region codes such as ``"01_alaska"`` or ``"06_iceland"``.
    output_path : Path
        Root directory for output files.  A ``rgi/`` subdirectory is created
        inside it.
    glaciers : pandas.DataFrame, list of str, or None, optional
        Optional whitelist of RGI IDs. May contain glacier IDs (``...-G-...``),
        complex IDs (``...-C-...``), or both. If a DataFrame is given, the
        ``rgi_id`` column is used. The output is restricted to:

        - any complex IDs listed,
        - any glacier IDs listed, and
        - the parent complexes of any listed glaciers (so glacier outlines
          always come paired with their containing complex).

        If None (default), no filtering is applied.
    extract_path : str or Path, optional
        Path to the directory where the archive will be extracted. Default is "rgi_archive".
    force_overwrite : bool, default False
        If True, re-download the file even if it already exists locally.
    ntasks : int, default 8
        Maximum number of parallel workers.

    Returns
    -------
    dict
        Dictionary with keys ``"rgi_complexes"`` and ``"rgi_glaciers"``
        pointing to the saved GeoPackage file paths.
    """

    rgi_archive_path = extract_path / Path("archive")
    rgi_archive_path.mkdir(parents=True, exist_ok=True)

    outline_types = ["C", "G"]

    # Optional: tune this
    total_tasks = len(regions) * len(outline_types)
    MAX_WORKERS = min(ntasks, total_tasks)  # or os.cpu_count() if CPU-bound

    url_template = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-{outline_type}/RGI2000-v7.0-{outline_type}-{region}.zip"

    rgis = []
    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                prepare_rgi_region,
                row,
                outline_type=outline_type,
                url_template=url_template,
                extract_path=rgi_archive_path,
                force_overwrite=force_overwrite,
            ): (row["region"], outline_type)
            for _, row in regions.iterrows()
            for outline_type in outline_types
        }

        pbar = tqdm(cf_as_completed(futures), total=len(futures), desc="Downloading RGI regions")
        for future in pbar:
            region_code, outline_type = futures[future]
            try:
                rgis.append(future.result())
                pbar.set_postfix_str(f"{region_code} ({outline_type}) ✓")
            except Exception as err:
                failed.append((region_code, outline_type, err))
                pbar.set_postfix_str(f"{region_code} ({outline_type}) ✗")
    for region_code, outline_type, exc in failed:
        logger.error("Failed region: %s, outline_type: %s with error: %s", region_code, outline_type, exc)

    logger.info("Concatenating %d RGI dataframes", len(rgis))
    rgi = pd.concat(rgis, ignore_index=True)
    rgi_c = rgi[rgi["rgi_id"].str.contains("-C-")].copy()
    rgi_g = rgi[rgi["rgi_id"].str.contains("-G-")].copy()

    # Assign parent complex rgi_id to each glacier outline, searching per o1region
    def _sjoin_region(region):
        """
        Return a Series mapping glacier indices to their parent complex rgi_id.

        Parameters
        ----------
        region : str
            The o1region code to process.

        Returns
        -------
        pandas.Series
            Series indexed by glacier row index with parent complex ``rgi_id`` values.
        """
        g_region = rgi_g.loc[rgi_g["o1region"] == region]
        c_region = rgi_c.loc[rgi_c["o1region"] == region]
        if c_region.empty:
            return pd.Series(dtype=object)
        # Use representative points to avoid floating-point boundary issues
        g_points = g_region[["rgi_id"]].copy()
        g_points["geometry"] = g_region.geometry.representative_point()
        g_points = gpd.GeoDataFrame(g_points, crs=g_region.crs)
        joined = gpd.sjoin(
            g_points,
            c_region[["rgi_id", "geometry"]],
            how="left",
            predicate="within",
        )
        joined = joined[~joined.index.duplicated(keep="first")]
        return joined["rgi_id_right"]

    regions_unique = rgi_g["o1region"].unique()
    rgi_g["rgi_id_c"] = None
    with ThreadPoolExecutor(max_workers=min(8, len(regions_unique))) as executor:
        futures = {executor.submit(_sjoin_region, region): region for region in regions_unique}
        for future in tqdm(cf_as_completed(futures), total=len(futures), desc="Assigning complex IDs"):
            result = future.result()
            if not result.empty:
                rgi_g.loc[result.index, "rgi_id_c"] = result.values

    if glaciers is not None:
        if isinstance(glaciers, pd.DataFrame):
            wanted = glaciers["rgi_id"].astype(str).tolist()
        else:
            wanted = [str(g) for g in glaciers]

        wanted_g = {i for i in wanted if "-G-" in i}
        wanted_c = {i for i in wanted if "-C-" in i}

        if wanted_g:
            parents = rgi_g.loc[rgi_g["rgi_id"].isin(wanted_g), "rgi_id_c"].dropna().unique()
            wanted_c.update(parents.tolist())

        rgi_c = rgi_c[rgi_c["rgi_id"].isin(wanted_c)].copy()
        rgi_g = rgi_g[rgi_g["rgi_id"].isin(wanted_g)].copy()
        logger.info("Filtered to %d complexes and %d glaciers", len(rgi_c), len(rgi_g))

    complex_path = output_path / "rgi_c.gpkg"
    logger.info("Saving complexes to %s", complex_path)
    rgi_c.to_file(complex_path)
    glaciers_path = output_path / "rgi_g.gpkg"
    logger.info("Saving glaciers to %s", glaciers_path)
    rgi_g.to_file(glaciers_path)

    logger.info("RGI preparation complete: %d complexes, %d glaciers", len(rgi_c), len(rgi_g))
    return {"rgi_complexes": complex_path, "rgi_glaciers": glaciers_path}
