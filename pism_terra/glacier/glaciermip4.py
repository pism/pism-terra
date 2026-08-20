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
GlacierMIP4 preparation.

Region-level analog of :mod:`pism_terra.glacier.s4f`: builds one aggregated
"complex" per RGI o1 region listed in the setup TOML
(e.g. ``RGI7_01``, ``RGI7_03``, ``RGI7_07``) whose geometry is the union of
all native RGI7 complexes in that region, in the per-region CRS the TOML
specifies. Downstream, ``pism-glacier-stage RGI7_NN ...`` looks the
aggregate up in ``rgi_c.gpkg`` and stages it like any other complex.
"""

from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # pylint: disable=unused-import
import toml
import xarray as xr
from pyfiglet import Figlet
from shapely.geometry import MultiPolygon, Polygon

from pism_terra.download import download_gebco
from pism_terra.glacier.ice_thickness import (
    prepare_ice_thickness_frank,
    prepare_ice_thickness_maffezzoli,
)
from pism_terra.glacier.rgi import prepare_rgi
from pism_terra.log import setup_logging

xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)


def _add_region_aggregates(
    complexes: gpd.GeoDataFrame,
    glaciers: gpd.GeoDataFrame,
    regions: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Append ``RGI7_NN`` aggregate complexes to *complexes* and tag *glaciers*.

    For every row in ``regions`` (indexed by o1 region code) a single
    aggregate complex is added whose ``rgi_id`` is ``f"RGI7_{NN}"`` with
    zero-padded ``NN``, whose geometry is the polygon-union of every
    native ``-C-NN-`` complex in that region, and whose ``crs`` is taken
    verbatim from the TOML's ``crs`` field. Each constituent glacier is
    back-linked to the aggregate via the ``rgi_id_c_aggregate`` column
    (semicolon-joined to support membership in multiple aggregates).

    Parameters
    ----------
    complexes : geopandas.GeoDataFrame
        Output of :func:`pism_terra.glacier.rgi.prepare_rgi`'s
        ``rgi_complexes`` layer. Modified in-place via concatenation.
    glaciers : geopandas.GeoDataFrame
        The matching ``rgi_glaciers`` layer.
    regions : pandas.DataFrame
        TOML's ``[regions]`` table, indexed by o1 code, with columns
        ``name`` and ``crs``.

    Returns
    -------
    tuple of geopandas.GeoDataFrame
        ``(complexes_with_aggregates, glaciers_with_back_links)``.
    """
    if "rgi_id_c_aggregate" not in glaciers.columns:
        glaciers["rgi_id_c_aggregate"] = ""

    extra_rows: list[dict[str, Any]] = []
    for r_idx, r_row in regions.iterrows():
        nn = str(r_idx).zfill(2)
        agg_name = f"RGI7_{nn}"
        crs_value = r_row["crs"]

        in_region = glaciers["rgi_id"].str.contains(f"-G-{nn}-", na=False)
        g_ids = glaciers.loc[in_region, "rgi_id"].tolist()
        if not g_ids:
            logger.warning("No RGI v7 glaciers found for region %s", nn)
            continue

        parent_c_ids = glaciers.loc[glaciers["rgi_id"].isin(g_ids), "rgi_id_c"].dropna().unique()
        parents = complexes.loc[complexes["rgi_id"].isin(parent_c_ids)]
        if parents.empty:
            logger.warning("No parent complexes resolved for %s", agg_name)
            continue

        merged = parents.geometry.union_all()
        if isinstance(merged, Polygon):
            merged = MultiPolygon([merged])

        area_km2 = float(gpd.GeoSeries([merged], crs=complexes.crs).to_crs(crs_value).geometry.area.sum()) / 1e6

        extra_rows.append(
            {
                "rgi_id": agg_name,
                "geometry": merged,
                "crs": crs_value,
                "epsg_code": None,
                "area_km2": area_km2,
            }
        )

        # Back-link constituent glaciers. Semicolon-join so a glacier can
        # belong to multiple aggregates without row duplication.
        sel = glaciers["rgi_id"].isin(g_ids)
        existing = glaciers.loc[sel, "rgi_id_c_aggregate"].fillna("")
        updated = existing.where(existing == "", existing + ";") + agg_name
        glaciers.loc[sel, "rgi_id_c_aggregate"] = updated

    if extra_rows:
        extras = gpd.GeoDataFrame(extra_rows, geometry="geometry", crs=complexes.crs)
        complexes = gpd.GeoDataFrame(
            pd.concat([complexes, extras], ignore_index=True),
            geometry="geometry",
            crs=complexes.crs,
        )
        logger.info(
            "Added %d GlacierMIP4 aggregates: %s",
            len(extra_rows),
            ", ".join(r["rgi_id"] for r in extra_rows),
        )

    return complexes, glaciers


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Prepare GlacierMIP4 region-level data sets.

    Reads a setup TOML describing one or more RGI o1 regions (with their
    name and target CRS), builds a per-region ``RGI7_NN`` aggregate
    complex, and stages the shared inputs (Maffezzoli + Frank ice
    thickness, GEBCO bathymetry) so that subsequent
    ``pism-glacier-stage RGI7_NN ...`` invocations succeed.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments (excluding the program name). When
        ``None``, :data:`sys.argv` is used. Passing ``argv=[]`` is useful
        from a notebook to bypass ipykernel arguments.

    Returns
    -------
    dict
        Mapping returned by :func:`pism_terra.glacier.rgi.prepare_rgi`
        with paths to the (now aggregate-augmented) ``rgi_c.gpkg`` and
        ``rgi_g.gpkg``.

    Notes
    -----
    The TOML must contain a ``[regions]`` table keyed by o1 code, e.g.::

        [regions]
        1 = {name = "alaska", crs = "EPSG:5936"}
        3 = {name = "arctic_canada_north", crs = "EPSG:3413"}
        7 = {name = "svalbard_jan_mayen", crs = "EPSG:3413"}
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--force-overwrite",
        help="Force re-downloading/re-extracting all source files.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ntasks",
        help="Number of parallel worker tasks.",
        type=int,
        default=8,
    )
    parser.add_argument("CONFIG_FILE", nargs=1)
    parser.add_argument(
        "OUTPUT_PATH",
        nargs="?",
        default=".",
        help="Root directory for staged outputs.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_file = args.CONFIG_FILE[0]
    force_overwrite = args.force_overwrite
    ntasks = args.ntasks
    output_path = Path(args.OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    setup_logging(output_path / "prepare.log")

    fig = Figlet(font="standard")
    banner = fig.renderText("pism-terra")
    logger.info("=" * 120)
    logger.info("\n%s", banner)
    logger.info("=" * 120)
    logger.info("Preparing GlacierMIP4 data")
    logger.info("-" * 120)

    config = toml.loads(Path(config_file).read_text("utf-8"))
    regions = pd.DataFrame.from_dict(config["regions"], orient="index")
    if "crs" not in regions.columns:
        raise ValueError(
            'Every [regions.<NN>] entry in the setup TOML must declare a `crs` (e.g. `crs = "EPSG:5936"`).'
        )
    regions["region"] = regions.index.astype(str).str.zfill(2) + "_" + regions["name"]

    glacier_path = output_path / "glacier"
    glacier_path.mkdir(parents=True, exist_ok=True)
    staging_path = output_path / "staging"
    staging_path.mkdir(parents=True, exist_ok=True)

    # --- RGI ---
    rgi_path = glacier_path / "rgi"
    rgi_path.mkdir(parents=True, exist_ok=True)
    rgi_staging = staging_path / "rgi"
    rgi_staging.mkdir(parents=True, exist_ok=True)

    rgi_files = prepare_rgi(
        regions,
        glaciers=None,
        glacier_groups=None,
        output_path=rgi_path,
        extract_path=rgi_staging,
        force_overwrite=force_overwrite,
        ntasks=ntasks,
    )

    complexes = gpd.read_file(rgi_files["rgi_complexes"])
    glaciers = gpd.read_file(rgi_files["rgi_glaciers"])

    # --- Add one ``RGI7_NN`` aggregate complex per region ---
    complexes, glaciers = _add_region_aggregates(complexes, glaciers, regions)
    logger.info("Writing aggregated complexes back to %s", rgi_files["rgi_complexes"])
    complexes.to_file(rgi_files["rgi_complexes"])
    logger.info("Writing back-linked glaciers to %s", rgi_files["rgi_glaciers"])
    glaciers.to_file(rgi_files["rgi_glaciers"])

    # --- Ice thickness (Frank + Maffezzoli) ---
    ice_thickness_path = glacier_path / "ice_thickness"
    ice_thickness_path.mkdir(parents=True, exist_ok=True)
    ice_thickness_staging = staging_path / "ice_thickness"
    ice_thickness_staging.mkdir(parents=True, exist_ok=True)

    frank_path = ice_thickness_path / "frank"
    frank_path.mkdir(parents=True, exist_ok=True)
    prepare_ice_thickness_frank(
        regions.index,
        complexes=complexes,
        glaciers=glaciers,
        output_path=frank_path,
        extract_path=ice_thickness_staging,
        rgi_extract_path=rgi_staging,
        force_overwrite=force_overwrite,
        ntasks=ntasks,
    )

    maffezzoli_path = ice_thickness_path / "maffezzoli"
    maffezzoli_path.mkdir(parents=True, exist_ok=True)
    prepare_ice_thickness_maffezzoli(
        regions.index,
        complexes=complexes,
        glaciers=glaciers,
        output_path=maffezzoli_path,
        extract_path=ice_thickness_staging,
        force_overwrite=force_overwrite,
        ntasks=ntasks,
    )

    # --- GEBCO bathymetry ---
    gebco_path = glacier_path / "gebco"
    gebco_path.mkdir(parents=True, exist_ok=True)
    gebco_staging = staging_path / "gebco"
    gebco_staging.mkdir(parents=True, exist_ok=True)
    gebco_nc = download_gebco(target_dir=gebco_staging)
    cog_gebco_p = gebco_path / "bathymetry.tif"

    ds = xr.open_dataset(gebco_nc, chunks={"lat": 1024, "lon": 1024})
    da = ds["elevation"].rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326")
    predictor = 3 if np.issubdtype(da.dtype, np.floating) else 2
    da.rio.to_raster(
        cog_gebco_p,
        driver="COG",
        compress="DEFLATE",
        predictor=predictor,
        blocksize=512,
        bigtiff="YES",
        overview_resampling="AVERAGE",
        num_threads="ALL_CPUS",
    )

    logger.info("GlacierMIP4 preparation complete.")
    logger.info("Aggregates available: %s", ", ".join(f"RGI7_{str(i).zfill(2)}" for i in regions.index))
    logger.info("Next: pism-glacier-stage RGI7_NN <run-config.toml>")
    return rgi_files


def cli(argv: Sequence[str] | None = None) -> int:
    """
    Console entry point.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments (excluding the program name).

    Returns
    -------
    int
        Exit code (``0`` on success).
    """
    _ = main(argv=argv)
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(cli())
