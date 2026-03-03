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
Prepare RGI data sets.
"""

import os
import re
import shutil
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed as cf_as_completed
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import cf_xarray
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # pylint: disable=unused-import
import toml
import xarray as xr
import xarray_regrid.methods.conservative  # pylint: disable=unused-import
from cdo import Cdo
from dask.distributed import Client, as_completed
from pyfiglet import Figlet
from rasterio.merge import merge
from tqdm.auto import tqdm

from pism_terra.download import download_archive, extract_archive
from pism_terra.vector import glaciers_in_complex

xr.set_options(keep_attrs=True)


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


def get_maffezzoli_url(url_template: str, region: str) -> str:
    """
    Format the given URL template with the provided region.

    Parameters
    ----------
    url_template : str
        A string with `{region}` and `{outline_type}` placeholders to be replaced.
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
    outline_type: str = "C",
    url_template: str = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-C/RGI2000-v7.0-{outline_type}-{region}.zip",
    extract_to: Path | str = "rgi",
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
    extract_to : str or Path, optional
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

    extract_to = Path(extract_to)
    url = get_rgi_url(url_template, region, outline_type)
    archive_dest = extract_to / Path(url.rsplit("/", 1)[-1])
    archive = download_archive(url, dest=archive_dest, force_overwrite=force_overwrite)
    extract_archive(archive, extract_to, force_overwrite=force_overwrite)

    rgi = gpd.read_file(extract_to / f"RGI2000-v7.0-{outline_type}-{region}.shp")
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


def prepare_ice_thickness_maffezzoli(
    regions: list,
    complexes: gpd.GeoDataFrame,
    glaciers: gpd.GeoDataFrame,
    output_path: Path,
    extract_to: Path | str = "ice_thickness",
    ntasks: int = 8,
    force_overwrite: bool = False,
):
    """
    Download, extract, and merge RGI region ice thickness.

    Parameters
    ----------
    regions : list
        Region codes (e.g. ``["01", "06"]``).
    complexes : geopandas.GeoDataFrame
        Complex outlines with an ``rgi_id`` and ``o1region`` column.
    glaciers : geopandas.GeoDataFrame
        Glacier outlines with ``rgi_id``, ``rgi_id_c``, and ``o1region`` columns.
    output_path : Path
        Root directory for output files.
    extract_to : Path or str, optional
        Subdirectory under *output_path* for extracted archives.
        Defaults to ``"ice_thickness"``.
    ntasks : int, default 8
        Maximum number of parallel workers.
    force_overwrite : bool, default False
        If True, re-download the file even if it already exists locally.
    """

    url_template: str = "https://zenodo.org/records/17724512/files/RGI70G_rgi{region}.zip?download=1"
    extract_to = Path(extract_to)

    def _download_and_extract(region):
        """
        Download and extract ice thickness archive for a single region.

        Parameters
        ----------
        region : str
            Region code (e.g. ``"01"``).

        Returns
        -------
        str
            The region code that was processed.
        """
        url = get_maffezzoli_url(url_template, region)
        archive_dest = output_path / extract_to / Path(urlparse(url).path).name
        archive = download_archive(url, dest=archive_dest, force_overwrite=force_overwrite)
        extract_archive(archive, output_path / extract_to, force_overwrite=force_overwrite)
        return region

    MAX_WORKERS = min(ntasks, len(regions))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_download_and_extract, region): region for region in regions}
        for future in tqdm(cf_as_completed(futures), total=len(futures), desc="Downloading ice thickness"):
            region = futures[future]
            try:
                future.result()
                print(f"✓ Finished ice thickness: {region}")
            except Exception as e:
                print(f"✗ Failed ice thickness: {region} with error: {e}")

    def _merge_complex(rgi_c_id, region_g, region_code):
        """
        Merge glacier thickness rasters for a single complex.

        Parameters
        ----------
        rgi_c_id : str
            The complex outline identifier.
        region_g : geopandas.GeoDataFrame
            Glacier outlines for the region.
        region_code : str
            Region code (e.g. ``"1"``).

        Returns
        -------
        str or None
            Path to the merged file, or ``None`` if no files found.
        """
        glaciers_list = glaciers_in_complex(rgi_c_id, region_g)
        glaciers_files = [
            output_path / extract_to / Path(f"rgi{region_code}") / Path(f"{g}.tif") for g in glaciers_list
        ]
        glaciers_files = [f for f in glaciers_files if f.exists()]
        if not glaciers_files:
            return None
        mosaic, out_transform = merge(glaciers_files)
        with rasterio.open(glaciers_files[0]) as src:
            out_meta = src.meta.copy()
        out_meta.update(
            {"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_transform}
        )
        merged_path = output_path / extract_to / Path(f"{rgi_c_id}_thickness.tif")
        with rasterio.open(merged_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        return str(merged_path)

    # Build list of all (rgi_c_id, region_g, region_code) tasks across regions
    merge_tasks = []
    for region in regions:
        region_c = complexes[complexes["o1region"] == region.zfill(2)]
        region_g = glaciers[glaciers["o1region"] == region.zfill(2)]
        for rgi_c_id in region_c["rgi_id"]:
            merge_tasks.append((rgi_c_id, region_g, region))

    with ThreadPoolExecutor(max_workers=min(ntasks, max(1, len(merge_tasks)))) as executor:
        futures = {
            executor.submit(_merge_complex, rgi_c_id, region_g, region_code): rgi_c_id
            for rgi_c_id, region_g, region_code in merge_tasks
        }
        failed = []
        for future in tqdm(cf_as_completed(futures), total=len(futures), desc="Merging ice thickness"):
            rgi_c_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                failed.append((rgi_c_id, exc))
        for rgi_c_id, err in failed:
            print(f"✗ Failed {rgi_c_id}: {err}")


def prepare_rgi(regions: list, output_path: Path, force_overwrite: bool = False, ntasks: int = 8):
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
    rgi_archive_path = rgi_path / Path("archive")
    rgi_archive_path.mkdir(parents=True, exist_ok=True)

    outline_types = ["C", "G"]

    # Optional: tune this
    total_tasks = len(regions) * len(outline_types)
    MAX_WORKERS = min(ntasks, total_tasks)  # or os.cpu_count() if CPU-bound

    url_template = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-{outline_type}/RGI2000-v7.0-{outline_type}-{region}.zip"

    rgis = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                prepare_rgi_region,
                region,
                outline_type=outline_type,
                url_template=url_template,
                extract_to=rgi_archive_path,
                force_overwrite=force_overwrite,
            ): (region, outline_type)
            for region in regions
            for outline_type in outline_types
        }

        for future in cf_as_completed(futures):
            region, outline_type = futures[future]
            try:
                rgis.append(future.result())
                print(f"✓ Finished region: {region}, outline_type: {outline_type}")
            except Exception as e:
                print(f"✗ Failed region: {region}, outline_type: {outline_type} with error: {e}")

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


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Prepare ISMIP7 Greenland input data sets.

    This function is the programmatic entry point. It parses command-line style
    arguments, creates the target grid, downloads and processes observation data,
    and prepares climate/ocean forcing files for PISM simulations.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments **excluding** the program name (i.e., like
        ``sys.argv[1:]``). If ``None`` (default), arguments are taken from the
        current process' ``sys.argv[1:]``. Passing ``argv=[]`` is recommended
        when calling from a Jupyter notebook to avoid ipykernel arguments.

    Returns
    -------
    dict[str, Any]
        Results dictionary containing:

        - ``"config"``: dict
          The parsed TOML configuration used for processing.
    """

    parser = ArgumentParser()
    parser.add_argument("--obs-path", default="data/obs")
    parser.add_argument(
        "--force-overwrite",
        help="Force downloading all files.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ntasks",
        help="Parallel tasks.",
        type=int,
        default=8,
    )
    parser.add_argument("CONFIG_FILE", nargs=1)
    parser.add_argument("OUTPUT_PATH", nargs=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_file = args.CONFIG_FILE[0]
    force_overwrite = args.force_overwrite
    ntasks = args.ntasks
    output_path = Path(args.OUTPUT_PATH[0])
    output_path.mkdir(parents=True, exist_ok=True)

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    print("=" * 120)
    print(banner)
    print("=" * 120)
    print("Preparing RGI data")
    print("-" * 120)
    print("")

    config = toml.loads(Path(config_file).read_text("utf-8"))
    regions = pd.DataFrame.from_dict(config["regions"], orient="index", columns=["name"])
    regions["region"] = regions.index.astype(str).str.zfill(2) + "_" + regions["name"]

    result = prepare_rgi(regions["region"], output_path=output_path, force_overwrite=force_overwrite)

    complexes = gpd.read_file(result["rgi_complexes"])
    glaciers = gpd.read_file(result["rgi_glaciers"])

    prepare_ice_thickness_maffezzoli(
        regions.index,
        complexes=complexes,
        glaciers=glaciers,
        output_path=output_path,
        force_overwrite=force_overwrite,
        ntasks=ntasks,
    )

    return result


def cli(argv: Sequence[str] | None = None) -> int:
    """
    Console entry point.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments (without the program name). If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    _ = main(argv=argv)
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(cli())
