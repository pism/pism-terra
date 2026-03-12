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

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed as cf_as_completed
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm

from pism_terra.download import download_archive, extract_archive


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
    region: str,
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
    region : str or None
        The region code (e.g., "01_alaska", "06_iceland") used to fill in the URL template.
    outline_type : str, optional
       Either C or G (complex or glacier).
    url_template : str, optional
        URL template containing a `{region}` placeholder. Defaults to the NSIDC RGI v7 template.
    extract_path : str or Path, optional
        Path to the directory where the archive will be extracted. Default is "rgi".
    area_threshold : float, optional
        Minimum glacier area (in square kilometers) for inclusion in the result. Defaults to 1.0.
    force_overwrite : bool, default False
        If True, re-download the file even if it already exists locally.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame of glaciers in the region, each with added columns:
        - "crs": EPSG code string based on hemisphere and UTM zone
        - "epsg_code": Integer EPSG code for reprojection

    See Also
    --------
    download_archive : Downloads an archive from a URL.
    extract_archive : Extracts a zip archive to a directory.

    Notes
    -----
    This function assumes that each region shapefile includes columns `"utm_zone"` and `"cenlat"`.
    """

    extract_path = Path(extract_path)
    url = get_rgi_url(url_template, region, outline_type)
    archive_dest = extract_path / Path(url.rsplit("/", 1)[-1])
    archive = download_archive(url, dest=archive_dest, force_overwrite=force_overwrite, verbose=False)
    extract_archive(archive, extract_path, force_overwrite=force_overwrite, verbose=False)

    rgi = gpd.read_file(extract_path / f"RGI2000-v7.0-{outline_type}-{region}.shp")
    rgi = rgi[rgi["area_km2"] > area_threshold]
    rgi["epsg"] = rgi.apply(
        lambda row: f"""EPSG:{32600 + int(row["utm_zone"]) if row["cenlat"] >= 0 else 32700 + int(row["utm_zone"])}""",
        axis=1,
    )
    rgi["epsg_code"] = rgi.apply(
        lambda row: f"""{32600 + int(row["utm_zone"]) if row["cenlat"] >= 0 else 32700 + int(row["utm_zone"])}""",
        axis=1,
    )
    return rgi


def prepare_rgi(
    regions: list,
    output_path: Path,
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
    regions : list
        Region codes (e.g. ``["01_alaska", "06_iceland"]``).
    output_path : Path
        Root directory for output files.  A ``rgi/`` subdirectory is created
        inside it.
    extract_path : str or Path, optional
        Path to the directory where the archive will be extracted. Default is "rgi".
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
    rgi_path = output_path / Path("rgi")
    rgi_path.mkdir(parents=True, exist_ok=True)
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
                region,
                outline_type=outline_type,
                url_template=url_template,
                extract_path=rgi_archive_path,
                force_overwrite=force_overwrite,
            ): (region, outline_type)
            for region in regions
            for outline_type in outline_types
        }

        pbar = tqdm(cf_as_completed(futures), total=len(futures), desc="Downloading RGI regions")
        for future in pbar:
            region, outline_type = futures[future]
            try:
                rgis.append(future.result())
                pbar.set_postfix_str(f"{region} ({outline_type}) ✓")
            except Exception as err:
                failed.append((region, outline_type, err))
                pbar.set_postfix_str(f"{region} ({outline_type}) ✗")
    for region, outline_type, exc in failed:
        print(f"✗ Failed region: {region}, outline_type: {outline_type} with error: {exc}")

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

    complex_path = output_path / "rgi_c.gpkg"
    rgi_c.to_file(complex_path)
    glaciers_path = output_path / "rgi_g.gpkg"
    rgi_g.to_file(glaciers_path)

    return {"rgi_complexes": complex_path, "rgi_glaciers": glaciers_path}
