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

"""
USGS benchmark-glacier stake measurements as a GeoPackage.

The ScienceBase release behind :mod:`pism_terra.glacier.usgs_benchmark`
also ships the point (stake, pit, probe) measurements every glacier-wide
balance was built from: seasonal ``bw``/``ba`` per site and year in
``Input_<Glacier>_Glaciological_Data.csv``, and for some glaciers the
sub-seasonal ``db`` between two visits in
``Input_<Glacier>_SubSeasonal_Glaciological_Data.csv``. This module joins
them with the site coordinates and writes one GeoPackage with a layer per
table, keeping the release's units (m w.e.) and turning its two date
notations into proper date fields.
"""

import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import pandas as pd

from pism_terra.glacier.usgs_benchmark import DEFAULT_DATA_DIR, download_usgs_benchmark, load_sites
from pism_terra.log import setup_logging

logger = logging.getLogger("pism_terra.glacier.usgs_stakes")

# Every layer: the per-glacier CSV suffix and its date columns.
LAYERS = {
    "stakes": ("Glaciological_Data", ("spring_date", "fall_date")),
    "subseasonal": ("SubSeasonal_Glaciological_Data", ("Date1", "Date2")),
}
# The release writes dates as YYYY/MM/DD in most files and M/D/YYYY in a few
# (Kennicott); both are tried, in this order.
DATE_FORMATS = ("%Y/%m/%d", "%m/%d/%Y")


def parse_dates(values: pd.Series) -> pd.Series:
    """
    Parse the release's date strings, whichever of its notations they use.

    Parameters
    ----------
    values : pandas.Series
        Date strings (``nan`` for missing).

    Returns
    -------
    pandas.Series
        ``datetime64[ns]`` values, ``NaT`` where missing.

    Raises
    ------
    ValueError
        If a non-missing value matches none of :data:`DATE_FORMATS`.
    """
    text = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        parsed = parsed.fillna(pd.to_datetime(text, format=fmt, errors="coerce"))
    bad = text.notna() & parsed.isna() & ~text.str.lower().isin(["nan", ""])
    if bad.any():
        raise ValueError(f"Unparseable dates: {sorted(text[bad].unique())[:10]}")
    return parsed


def load_measurements(data_dir: Path | str, glacier: str, layer: str = "stakes") -> pd.DataFrame | None:
    """
    Read one glacier's point measurements with typed dates.

    Parameters
    ----------
    data_dir : Path or str
        Extracted ``glacier_massBalance_data`` directory.
    glacier : str
        Glacier name as used in the release's directory and file names.
    layer : str, default "stakes"
        Which table, a key of :data:`LAYERS`.

    Returns
    -------
    pandas.DataFrame or None
        The CSV with a leading ``glacier`` column, stripped ``site_name``,
        integer ``Year`` and ``datetime64`` date columns; None when the
        glacier has no such file.
    """
    suffix, date_columns = LAYERS[layer]
    csv = Path(data_dir) / glacier / f"Input_{glacier}_{suffix}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv, na_values=["nan", "NaN", "NAN"])
    df.insert(0, "glacier", glacier)
    df["Year"] = df["Year"].astype(int)
    missing_site = df["site_name"].isna()
    if missing_site.any():
        logger.warning("%s: %d %s rows without a site name", glacier, int(missing_site.sum()), layer)
    df["site_name"] = df["site_name"].astype("string").str.strip()
    for column in date_columns:
        df[column] = parse_dates(df[column])
    return df


def build_stake_layers(data_dir: Path | str, sites: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    """
    Assemble every glacier's measurements into one point layer per table.

    Parameters
    ----------
    data_dir : Path or str
        Extracted ``glacier_massBalance_data`` directory; each
        sub-directory is a glacier.
    sites : geopandas.GeoDataFrame
        Stake locations from :func:`pism_terra.glacier.usgs_benchmark.load_sites`.

    Returns
    -------
    dict of str to geopandas.GeoDataFrame
        ``"sites"`` (one point per stake) plus a layer per key of
        :data:`LAYERS` that has data. Measurements at a site the release
        gives no coordinates for keep a null geometry rather than being
        dropped; they are listed in the log.
    """
    data_dir = Path(data_dir)
    glaciers = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    locations = sites[["Glacier", "site_name", "elevation", "geometry"]] if "elevation" in sites else sites
    locations = locations.rename(columns={"Glacier": "glacier"})[["glacier", "site_name", "geometry"]]
    locations = locations.drop_duplicates(subset=["glacier", "site_name"])

    layers: dict[str, gpd.GeoDataFrame] = {"sites": gpd.GeoDataFrame(sites.rename(columns={"Glacier": "glacier"}))}
    for layer in LAYERS:
        frames = [df for glacier in glaciers if (df := load_measurements(data_dir, glacier, layer)) is not None]
        if not frames:
            continue
        table = pd.concat(frames, ignore_index=True)
        merged = table.merge(locations, on=["glacier", "site_name"], how="left")
        unlocated = merged[merged["geometry"].isna()]
        if not unlocated.empty:
            names = unlocated.groupby("glacier")["site_name"].agg(lambda s: sorted(set(s.dropna())))
            logger.warning(
                "%s: %d of %d rows at sites without coordinates: %s",
                layer,
                len(unlocated),
                len(merged),
                names.to_dict(),
            )
        layers[layer] = gpd.GeoDataFrame(merged, geometry="geometry", crs=sites.crs)
        logger.info("%s: %d rows from %d glaciers", layer, len(merged), len(frames))
    return layers


def write_stake_geopackage(layers: dict[str, gpd.GeoDataFrame], output: Path | str) -> Path:
    """
    Write the layers to one GeoPackage, replacing any existing file.

    Parameters
    ----------
    layers : dict of str to geopandas.GeoDataFrame
        Output of :func:`build_stake_layers`.
    output : Path or str
        GeoPackage to write.

    Returns
    -------
    Path
        The file written.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    for name, gdf in layers.items():
        gdf.to_file(output, layer=name, driver="GPKG")
        logger.info("Wrote layer %s (%d rows) to %s", name, len(gdf), output)
    return output


def main(argv: Sequence[str] | None = None) -> Path:
    """
    Write the USGS benchmark-glacier stake measurements to a GeoPackage.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments (excluding the program name). When
        ``None``, :data:`sys.argv` is used.

    Returns
    -------
    Path
        The GeoPackage written.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = (
        "Write the USGS benchmark-glacier point measurements (m w.e.) to a GeoPackage with layers "
        "'sites', 'stakes' (seasonal bw/ba per site and year) and 'subseasonal' (db between two visits)."
    )
    parser.add_argument("OUTPUT_GPKG", help="GeoPackage to write (replaced if it exists).")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Cache for the USGS archives.")
    parser.add_argument("--force-overwrite", action="store_true", default=False, help="Re-download the archives.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output = Path(args.OUTPUT_GPKG).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(output.with_suffix(".log"))

    paths = download_usgs_benchmark(args.data_dir, force_overwrite=args.force_overwrite)
    layers = build_stake_layers(paths["data"], load_sites(paths["sites"]))
    return write_stake_geopackage(layers, output)


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
