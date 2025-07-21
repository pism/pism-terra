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
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Prepare RGI.
"""
# pylint: disable=broad-exception-caught

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import pandas as pd

from pism_terra.download import download_archive, extract_archive


def get_rgi_url(url_template: str, region: str) -> str:
    """
    Format the given URL template with the provided region.

    Parameters
    ----------
    url_template : str
        A string with a `{region}` placeholder to be replaced.
    region : str
        The region code to insert into the template.

    Returns
    -------
    str
        The formatted URL.
    """
    return url_template.format(region=region)


def prepare_rgi_region(
    region: str,
    url_template: str = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-C/RGI2000-v7.0-C-{region}.zip",
    extract_to: Path | str = "rgi",
    area_threshold: float = 1.0,
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
    url_template : str, optional
        URL template containing a `{region}` placeholder. Defaults to the NSIDC RGI v7 template.
    extract_to : str or Path, optional
        Path to the directory where the archive will be extracted. Default is "rgi".
    area_threshold : float, optional
        Minimum glacier area (in square kilometers) for inclusion in the result. Defaults to 1.0.

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

    extract_to = Path(extract_to)
    url = get_rgi_url(url_template, region)
    archive = download_archive(url)
    extract_archive(archive, extract_to)

    rgi = gpd.read_file(extract_to / f"RGI2000-v7.0-C-{region}.shp")
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


rgi_path = Path("rgi")
rgi_path.mkdir(parents=True, exist_ok=True)
rgi_archive_path = rgi_path / Path("archive")
rgi_archive_path.mkdir(parents=True, exist_ok=True)

regions = pd.read_csv(rgi_path / "regions.csv", dtype={"id": str, "name": str})
regions["region"] = regions["id"] + "_" + regions["name"]
# Optional: tune this
MAX_WORKERS = min(8, len(regions))  # or os.cpu_count() if CPU-bound


url_template = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-C/RGI2000-v7.0-C-{region}.zip"
wgs84 = "EPSG:4326"


rgis = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(prepare_rgi_region, region, url_template=url_template, extract_to=rgi_archive_path): region
        for region in regions["region"]
    }

    for future in as_completed(futures):
        region = futures[future]
        try:
            rgis.append(future.result())
            print(f"✓ Finished region: {region}")
        except Exception as e:
            print(f"✗ Failed region: {region} with error: {e}")

rgi = pd.concat(rgis)
rgi.to_file(rgi_path / "rgi.gpkg")
shutil.rmtree(rgi_archive_path)
