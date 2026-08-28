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

# pylint: disable=unused-import

"""
USGS benchmark-glacier mass balances against PISM output.

The USGS ScienceBase release *Compiled Input Data and Glacier-Wide Mass
Balances* (https://doi.org/10.5066/F7HD7SRF) publishes geodetically
calibrated glacier-wide winter, summer and annual balances for a handful of
North American glaciers in metres water equivalent, keyed by glacier *name*
rather than RGI ID. This module downloads the release, finds the RGI v7
glacier (``-G-``) outline each glacier lives in, converts the balances to
Gt/yr with the release's own time-varying glacier area, and plots each
glacier's series with whatever PISM ``tendency_of_ice_mass`` exists for that
RGI ID under a run directory.
"""

import logging
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import cf_xarray
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pint_xarray
import requests
import xarray as xr
from matplotlib.ticker import MaxNLocator

from pism_terra.download import download_archive, extract_archive
from pism_terra.glacier.rgi import prepare_rgi_region
from pism_terra.kitp.analyze import REGION_DIM, rc_params, with_region_labels
from pism_terra.log import setup_logging

logger = logging.getLogger("pism_terra.glacier.usgs_benchmark")

SCIENCEBASE_API = "https://www.sciencebase.gov/catalog"
SCIENCEBASE_ITEM = "6441de03d34ee8d4ade7d2a5"
DATA_ARCHIVE = "glacier_massBalance_data.zip"
SITES_ARCHIVE = "Glacier_Mass_Balance_Sites.zip"

# The release covers Alaska, Washington and Montana only, so two RGI o1
# regions are enough to place every glacier in it. The project GeoPackages
# (``s4f_g.gpkg`` etc.) are study-area subsets and miss most of these
# glaciers, hence the full regional outlines by default.
RGI_REGIONS = ("01_alaska", "02_western_canada_usa")
# NSIDC keeps the complex and glacier products in sibling directories; the
# template ``prepare_rgi_region`` defaults to only knows the complex one.
RGI_URL_TEMPLATE = (
    "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/"
    "RGI2000-v7.0-{outline_type}/RGI2000-v7.0-{outline_type}-{region}.zip"
)

BALANCE_VARS = {"Bw": "winter", "Bs": "summer", "Ba": "annual"}
# Column suffixes recognised as a one-sigma uncertainty for a balance
# column. Version 10 of the release has none; they are honoured if a future
# version (or a user-edited CSV) adds them.
UNCERTAINTY_SUFFIXES = ("_unc", "_uncertainty", "_err", "_sigma")
MODEL_VAR = "tendency_of_ice_mass"
#: Model counterpart of a glaciological seasonal balance. A stake network
#: measures accumulation minus melt at the surface; within a season flow and
#: discharge move mass around without a stake seeing it, so the surface flux is
#: the like-for-like variable — not the total ``MODEL_VAR`` tendency.
MODEL_SEASONAL_VAR = "tendency_of_ice_mass_due_to_surface_mass_flux"
#: Columns of the release's glacier-wide solution giving the measured season
#: boundaries: the winter maximum ends the winter season, the annual minimum
#: ends the balance year.
SEASON_DATE_COLUMNS = ("Bw_Date", "Ba_Date")
DEFAULT_DATA_DIR = "~/base/pism-terra/usgs_benchmark"

RHO_WATER = 1000.0  # constants.fresh_water.density

OBS_STYLE = {
    "Bw": {"color": "#4477AA", "marker": "v"},
    "Bs": {"color": "#EE6677", "marker": "^"},
    "Ba": {"color": "#000000", "marker": "o"},
}
MODEL_COLOR = "#228833"
#: Modelled seasonal balances borrow their observation's colour so the eye
#: pairs them, and are drawn as lines against the observations' markers.
MODEL_SEASON_STYLE = {"Bw": {"color": "#4477AA", "ls": "--"}, "Bs": {"color": "#EE6677", "ls": "--"}}
# Fewest overlapping years for which a correlation coefficient is reported; the
# error statistics are given from a single year on.
SKILL_MIN_YEARS = 3
SPECIFIC_UNITS = "m year^-1"
SPECIFIC_LABEL = "m w.e./yr"
SKILL_COLUMNS = ["variable", "season", "source", "n", "r", "mae", "bias"]


def sciencebase_file_urls(item_id: str = SCIENCEBASE_ITEM) -> dict[str, str]:
    """
    Resolve the download URL of every file attached to a ScienceBase item.

    The file URLs embed a content hash that changes whenever USGS posts a
    new version, so they are looked up by file name at run time rather than
    hard-coded.

    Parameters
    ----------
    item_id : str, optional
        ScienceBase item identifier.

    Returns
    -------
    dict of str to str
        Mapping from attached file name to its download URL.
    """
    response = requests.get(f"{SCIENCEBASE_API}/item/{item_id}", params={"format": "json"}, timeout=30)
    response.raise_for_status()
    return {entry["name"]: entry["url"] for entry in response.json().get("files", [])}


def download_usgs_benchmark(data_dir: Path | str, force_overwrite: bool = False) -> dict[str, Path]:
    """
    Fetch and extract the mass-balance and site archives of the release.

    Parameters
    ----------
    data_dir : Path or str
        Directory the archives are downloaded to and extracted under.
    force_overwrite : bool, default False
        Re-download and re-extract even when the files already exist.

    Returns
    -------
    dict of str to Path
        ``"data"`` — directory with one sub-directory per glacier;
        ``"sites"`` — directory with the measurement-site CSV.
    """
    data_dir = Path(data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    urls: dict[str, str] | None = None
    result: dict[str, Path] = {}
    for key, name in (("data", DATA_ARCHIVE), ("sites", SITES_ARCHIVE)):
        archive = data_dir / name
        extract_to = data_dir / Path(name).stem
        if archive.exists() and extract_to.exists() and not force_overwrite:
            logger.info("Using cached %s", extract_to)
            result[key] = extract_to
            continue
        if not archive.exists() or force_overwrite:
            if urls is None:
                urls = sciencebase_file_urls()
            if name not in urls:
                raise KeyError(f"{name} is not attached to ScienceBase item {SCIENCEBASE_ITEM}")
            logger.info("Downloading %s", name)
            download_archive(urls[name], dest=archive, force_overwrite=force_overwrite, verbose=False)
        logger.info("Extracting %s to %s", name, extract_to)
        extract_archive(archive, extract_to, force_overwrite=force_overwrite, verbose=False)
        result[key] = extract_to
    return result


def load_sites(sites_dir: Path | str) -> gpd.GeoDataFrame:
    """
    Read the glaciological measurement sites as points.

    Weather stations are dropped: they sit off-glacier and would drag the
    RGI match toward the wrong outline.

    Parameters
    ----------
    sites_dir : Path or str
        Directory holding ``Glacier_Mass_Balance_Data_Sites.csv``.

    Returns
    -------
    geopandas.GeoDataFrame
        One EPSG:4326 point per stake, with the release's ``Glacier`` and
        ``site_name`` columns.
    """
    sites_dir = Path(sites_dir)
    csv = next(iter(sorted(sites_dir.glob("*Sites*.csv"))), None)
    if csv is None:
        raise FileNotFoundError(f"No site CSV found in {sites_dir}")
    df = pd.read_csv(csv)
    df = df[df["Type"].str.strip().str.lower() == "glaciological"].copy()
    df["Glacier"] = df["Glacier"].str.strip()
    df["site_name"] = df["site_name"].astype(str).str.strip()
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    ).reset_index(drop=True)


def load_rgi_glaciers(
    data_dir: Path | str,
    rgi_file: Path | str | None = None,
    regions: Iterable[str] = RGI_REGIONS,
    force_overwrite: bool = False,
) -> gpd.GeoDataFrame:
    """
    Load RGI v7 glacier (``-G-``) outlines to match the sites against.

    Parameters
    ----------
    data_dir : Path or str
        Directory the regional RGI archives are cached under
        (``<data_dir>/rgi_archive``).
    rgi_file : Path or str or None, optional
        An existing outline file (GeoPackage or shapefile) to use instead of
        downloading the regions.
    regions : iterable of str, optional
        RGI region names to download when *rgi_file* is not given.
    force_overwrite : bool, default False
        Re-download the regional archives.

    Returns
    -------
    geopandas.GeoDataFrame
        Outlines with ``rgi_id``, ``glac_name``, ``area_km2`` and geometry.
    """
    columns = ["rgi_id", "glac_name", "area_km2", "geometry"]
    if rgi_file is not None:
        rgi = gpd.read_file(rgi_file)
        if "glac_name" not in rgi:
            rgi["glac_name"] = None
        return rgi[columns]

    extract_path = Path(data_dir).expanduser() / "rgi_archive"
    frames = [
        prepare_rgi_region(
            {"region": region},
            outline_type="G",
            url_template=RGI_URL_TEMPLATE,
            extract_path=extract_path,
            area_threshold=0.0,
            force_overwrite=force_overwrite,
        )[columns]
        for region in regions
    ]
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def match_rgi_ids(sites: gpd.GeoDataFrame, rgi: gpd.GeoDataFrame, max_distance_km: float = 5.0) -> pd.DataFrame:
    """
    Assign each glacier in the release to the RGI outline holding most of its stakes.

    A glacier's stake network can spill onto tributaries with their own RGI
    entry (Taku's does), so the modal outline is taken rather than requiring
    every stake to agree. A glacier with no stake inside any outline falls
    back to the outline nearest the stakes' centroid, but only within
    *max_distance_km*: with a study-area subset of outlines the nearest one
    can be hundreds of kilometres away, and such a glacier is better left
    unmatched (``rgi_id`` NaN) than plotted against the wrong outline.

    Parameters
    ----------
    sites : geopandas.GeoDataFrame
        Stake points from :func:`load_sites`.
    rgi : geopandas.GeoDataFrame
        Outlines from :func:`load_rgi_glaciers`.
    max_distance_km : float, default 5.0
        Farthest an outline may be from the stakes' centroid for the
        nearest-neighbour fallback.

    Returns
    -------
    pandas.DataFrame
        One row per glacier with ``glacier``, ``rgi_id``, ``glac_name``,
        ``area_km2``, ``n_sites`` and ``n_matched`` (stakes inside the
        chosen outline; 0 for a nearest-neighbour fallback).
    """
    outlines = rgi[["rgi_id", "glac_name", "area_km2", "geometry"]]
    points = sites.to_crs(outlines.crs)
    joined = gpd.sjoin(points, outlines, how="left", predicate="within")

    rows: list[dict[str, Any]] = []
    for glacier, group in joined.groupby("Glacier", sort=True):
        n_sites = group.index.nunique()
        matched = group.dropna(subset=["rgi_id"])
        if matched.empty:
            nearest = _nearest_outline(points.loc[group.index.unique()], outlines, max_distance_km)
            if nearest is None:
                logger.warning("%s: no RGI outline within %.0f km of its stakes; unmatched", glacier, max_distance_km)
                nearest = {"rgi_id": None, "glac_name": None, "area_km2": np.nan}
            else:
                logger.warning("%s: no stake inside an RGI outline; using nearest %s", glacier, nearest["rgi_id"])
            rows.append({"glacier": glacier, **nearest, "n_sites": n_sites, "n_matched": 0})
            continue
        counts = matched["rgi_id"].value_counts()
        rgi_id = counts.index[0]
        n_matched = int(counts.iloc[0])
        if n_matched * 2 < n_sites:
            logger.warning("%s: only %d of %d stakes fall in %s", glacier, n_matched, n_sites, rgi_id)
        else:
            logger.info("%s -> %s (%d of %d stakes)", glacier, rgi_id, n_matched, n_sites)
        first = matched[matched["rgi_id"] == rgi_id].iloc[0]
        rows.append(
            {
                "glacier": glacier,
                "rgi_id": rgi_id,
                "glac_name": first["glac_name"],
                "area_km2": float(first["area_km2"]),
                "n_sites": n_sites,
                "n_matched": n_matched,
            }
        )
    return pd.DataFrame(rows, columns=["glacier", "rgi_id", "glac_name", "area_km2", "n_sites", "n_matched"])


def _nearest_outline(
    points: gpd.GeoDataFrame, outlines: gpd.GeoDataFrame, max_distance_km: float
) -> dict[str, Any] | None:
    """
    Find the outline nearest the centroid of a group of points.

    Parameters
    ----------
    points : geopandas.GeoDataFrame
        Stakes of one glacier.
    outlines : geopandas.GeoDataFrame
        Candidate outlines in the same CRS as *points*.
    max_distance_km : float
        Outlines farther than this are not considered.

    Returns
    -------
    dict or None
        ``rgi_id``, ``glac_name`` and ``area_km2`` of the nearest outline,
        or None when none lies within *max_distance_km*.
    """
    centroid = gpd.GeoDataFrame(geometry=[points.union_all().centroid], crs=points.crs)
    # Distances are meaningless in degrees, and reprojecting every outline
    # in a region for one lookup is slow, so work in a local UTM on the
    # outlines within a degree of the stakes.
    x, y = centroid.geometry.iloc[0].x, centroid.geometry.iloc[0].y
    candidates = outlines.cx[x - 1 : x + 1, y - 1 : y + 1]
    if candidates.empty:
        candidates = outlines
    utm = centroid.estimate_utm_crs()
    nearest = gpd.sjoin_nearest(
        centroid.to_crs(utm), candidates.to_crs(utm), how="left", max_distance=max_distance_km * 1000.0
    )
    row = nearest.iloc[0]
    if pd.isna(row["rgi_id"]):
        return None
    return {"rgi_id": row["rgi_id"], "glac_name": row["glac_name"], "area_km2": float(row["area_km2"])}


def _uncertainty_var(ds: xr.Dataset | pd.DataFrame, var: str) -> str | None:
    """
    Name the uncertainty column that goes with a balance column, if any.

    Parameters
    ----------
    ds : xarray.Dataset or pandas.DataFrame
        Container to look in.
    var : str
        Balance variable name (``Bw``, ``Bs`` or ``Ba``).

    Returns
    -------
    str or None
        The first ``<var><suffix>`` present, or None.
    """
    return next((f"{var}{suffix}" for suffix in UNCERTAINTY_SUFFIXES if f"{var}{suffix}" in ds), None)


def load_glacier_wide(data_dir: Path | str, glacier: str, fallback_area_km2: float | None = None) -> xr.Dataset | None:
    """
    Read a glacier's calibrated glacier-wide balances and its area per balance year.

    Parameters
    ----------
    data_dir : Path or str
        Extracted ``glacier_massBalance_data`` directory.
    glacier : str
        Glacier name as used in the release's directory and file names.
    fallback_area_km2 : float or None, optional
        Area to use when the release has no area-altitude distribution for
        the glacier (the RGI area, typically).

    Returns
    -------
    xarray.Dataset or None
        ``Bw``/``Bs``/``Ba`` (and any uncertainty columns) in
        ``m year^-1`` — metres water equivalent per balance year — with
        ``area`` in km² on an integer ``time`` axis of balance years. None
        when the glacier has no glacier-wide solution file (point data only).
    """
    glacier_dir = Path(data_dir) / glacier
    output_csv = glacier_dir / f"Output_{glacier}_Glacier_Wide_solutions_calibrated.csv"
    if not output_csv.exists():
        return None

    df = pd.read_csv(output_csv, na_values=["nan", "NaN", "NAN"])
    years = df["Year"].astype(int).to_numpy()
    ds = xr.Dataset(coords={"time": ("time", years, {"long_name": "balance year"})})
    for var, season in BALANCE_VARS.items():
        if var not in df:
            continue
        ds[var] = (
            "time",
            df[var].astype(float).to_numpy(),
            {"units": "m year^-1", "long_name": f"glacier-wide {season} mass balance (m w.e.)"},
        )
        unc = _uncertainty_var(df, var)
        if unc is not None:
            ds[unc] = (
                "time",
                df[unc].astype(float).to_numpy(),
                {"units": "m year^-1", "long_name": f"uncertainty of glacier-wide {season} mass balance (m w.e.)"},
            )

    # The measured season boundaries. Keeping them on the dataset lets the
    # model be integrated over exactly the interval each observation covers.
    for column in SEASON_DATE_COLUMNS:
        if column in df:
            ds[column] = ("time", pd.to_datetime(df[column], errors="coerce").to_numpy())
            ds[column].attrs = {"long_name": f"measured {column.split('_', maxsplit=1)[0]} date"}

    aad_csv = glacier_dir / f"Input_{glacier}_Area_Altitude_Distribution.csv"
    if aad_csv.exists():
        aad = pd.read_csv(aad_csv).set_index("Year")
        area = aad.sum(axis=1, numeric_only=True)
        area.index = area.index.astype(int)
        full_index = pd.RangeIndex(min(area.index.min(), years.min()), max(area.index.max(), years.max()) + 1)
        # Years without a DEM-derived area (the tail of the record, usually)
        # keep the last known geometry.
        filled = area.reindex(full_index).ffill().bfill()
        missing = sorted(int(year) for year in set(years) - set(area.index))
        if missing:
            logger.info("%s: no area for %s; carrying the nearest year forward", glacier, missing)
        area_values = filled.reindex(years).to_numpy()
        source = "area-altitude distribution"
    elif fallback_area_km2 is not None:
        logger.warning("%s: no area-altitude distribution; using constant %.3f km2", glacier, fallback_area_km2)
        area_values = np.full(years.shape, float(fallback_area_km2))
        source = "RGI"
    else:
        raise FileNotFoundError(f"{aad_csv} is missing and no fallback area was given")
    ds["area"] = ("time", area_values, {"units": "km^2", "long_name": "glacier area", "source": source})
    ds.attrs["glacier"] = glacier
    ds.attrs["source"] = f"USGS Glacier Project, Compiled Input Data and Glacier-Wide Mass Balances, {output_csv.name}"
    return ds


def to_mass_rate(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert glacier-wide specific balances to mass-change rates.

    A balance of 1 m w.e. over the glacier is a mass change of
    ``1 m x rho_water x area``; with the area in km² that is ``1e-3`` Gt per
    km².

    Parameters
    ----------
    ds : xarray.Dataset
        Output of :func:`load_glacier_wide`.

    Returns
    -------
    xarray.Dataset
        The balance (and uncertainty) variables in Gt/yr, plus ``area``.
    """
    quantified = ds.pint.quantify()
    rho = xr.DataArray(RHO_WATER).pint.quantify("kg m^-3")
    out: dict[str, xr.DataArray] = {}
    for var in list(ds.data_vars):
        if var == "area" or not np.issubdtype(ds[var].dtype, np.number):
            continue
        rate = (quantified[var] * rho * quantified["area"]).pint.to("Gt/year")
        rate.attrs["long_name"] = ds[var].attrs.get("long_name", var).replace("(m w.e.)", "").strip()
        out[var] = rate
    result = xr.Dataset(out).pint.dequantify()
    for var in result.data_vars:
        result[var].attrs["units"] = "Gt year^-1"
    result["area"] = ds["area"]
    for column in SEASON_DATE_COLUMNS:
        if column in ds:
            result[column] = ds[column]
    result.attrs.update(ds.attrs)
    return result


def find_model_files(root: Path | str, pattern: str = "scalar_G_*.nc") -> list[Path]:
    """
    Find per-glacier scalar files below a run directory.

    Parameters
    ----------
    root : Path or str
        Directory searched recursively.
    pattern : str, optional
        Glob pattern of the post-processed per-glacier scalar files.

    Returns
    -------
    list of Path
        Matching files, sorted.
    """
    return sorted(Path(root).expanduser().rglob(pattern))


def _mean_years(ds: xr.Dataset) -> np.ndarray:
    """
    Label each reporting interval by the calendar year it covers.

    Uses the interval start from ``time_bounds`` when the file has it. The
    post-processed files drop the bounds and keep the interval midpoint as
    the time stamp, so a stamp on 1 January is taken as the *end* of the
    previous year's interval and any other stamp as lying inside its year.

    Parameters
    ----------
    ds : xarray.Dataset
        Scalar dataset with a decoded (cftime or datetime) ``time`` axis.

    Returns
    -------
    numpy.ndarray
        Integer year per time step.
    """
    bounds_name = ds["time"].attrs.get("bounds", "time_bounds")
    if bounds_name in ds:
        starts = ds[bounds_name].isel({ds[bounds_name].dims[-1]: 0}).values
        return np.array([_year(t) for t in starts], dtype=int)
    years = []
    for t in ds["time"].values:
        t = pd.Timestamp(t) if isinstance(t, np.datetime64) else t
        years.append(_year(t) - 1 if (t.month, t.day) == (1, 1) else _year(t))
    return np.array(years, dtype=int)


def _year(t: Any) -> int:
    """
    Calendar year of a cftime, pandas or numpy time stamp.

    Parameters
    ----------
    t : object
        Time stamp.

    Returns
    -------
    int
        Its year.
    """
    if isinstance(t, np.datetime64):
        t = pd.Timestamp(t)
    return int(t.year)


def _run_label(file: Path, root: Path | str | None) -> str:
    """
    Name a run by its directory under the search root and its ensemble id.

    Parameters
    ----------
    file : Path
        Scalar file.
    root : Path or str or None
        Search root; None labels the run by file name.

    Returns
    -------
    str
        ``<subdirectory> id_<n>`` when both are known, otherwise the file
        name (or ``<subdirectory>/<file name>`` when no ``id_<n>`` token exists).
    """
    if root is None or not file.is_relative_to(root):
        return file.name
    relative = file.relative_to(root)
    if len(relative.parts) == 1:
        return file.name
    subdir = relative.parts[0]
    match = re.search(r"_id_(\d+)_", file.name)
    return f"{subdir} id_{match.group(1)}" if match else f"{subdir}/{file.name}"


def _grid_spacing(ds: xr.Dataset) -> tuple[float, float] | None:
    """
    Read the grid spacing PISM records in ``pism_config``.

    Parameters
    ----------
    ds : xarray.Dataset
        Scalar file as opened, before ``pism_config`` is dropped.

    Returns
    -------
    tuple of float or None
        ``(dx, dy)`` in metres, or None when the file carries no config.
    """
    if "pism_config" not in ds:
        return None
    attrs = ds["pism_config"].attrs
    try:
        return float(attrs["grid.dx"]), float(attrs["grid.dy"])
    except (KeyError, TypeError, ValueError):
        return None


def ice_area(ds: xr.Dataset, rgi_id: str, spacing: tuple[float, float] | None) -> xr.DataArray | None:
    """
    Ice-covered area of a glacier through time, from a per-glacier scalar file.

    The post-processing sums ``sftgif`` (the ice-covered fraction of each
    cell) over the outline, so it is a cell count; times the cell area it is
    the glacierized area. Without it the outline area the post-processing
    stores is used, which does not change in time.

    Parameters
    ----------
    ds : xarray.Dataset
        Scalar dataset with region labels restored.
    rgi_id : str
        RGI ``-G-`` identifier to select.
    spacing : tuple of float or None
        ``(dx, dy)`` in metres from :func:`_grid_spacing`.

    Returns
    -------
    xarray.DataArray or None
        Area in m² on ``time``; None when the file has neither ``sftgif``
        nor ``area``.
    """
    if "sftgif" in ds and spacing is not None:
        dx, dy = spacing
        area = ds["sftgif"].sel({REGION_DIM: rgi_id}).load() * dx * dy
        area.attrs = {"units": "m^2", "long_name": "ice-covered area", "source": "sftgif * dx * dy"}
    elif "area" in ds:
        static = ds["area"].sel({REGION_DIM: rgi_id}).load()
        area = xr.full_like(ds["time"], float(static), dtype=float)
        area.attrs = {"units": "m^2", "long_name": "outline area", "source": "outline"}
    else:
        return None
    return area.drop_vars(REGION_DIM, errors="ignore")


def specific_balance(values: xr.DataArray, area: xr.DataArray) -> xr.DataArray:
    """
    Turn a mass-change rate into a specific balance in metres water equivalent.

    Parameters
    ----------
    values : xarray.DataArray
        Rates in Gt/yr (``units`` attribute required).
    area : xarray.DataArray
        Glacier area in m² (``units`` attribute required), broadcastable
        against *values*.

    Returns
    -------
    xarray.DataArray
        ``values / (area * rho_water)`` in ``m year^-1``.
    """
    rho = xr.DataArray(RHO_WATER).pint.quantify("kg m^-3")
    out = (values.pint.quantify() / (area.pint.quantify() * rho)).pint.to("m/year").pint.dequantify()
    # the area used is reported separately; as a coordinate it would clash with
    # the observed area when the series are merged into one dataset
    out = out.drop_vars("area", errors="ignore")
    out.attrs = {"units": SPECIFIC_UNITS, "long_name": f"{values.attrs.get('long_name', values.name)} (m w.e.)"}
    return out


def to_specific_balances(
    model: xr.DataArray | None, seasons: xr.Dataset | None, obs: xr.Dataset
) -> tuple[xr.DataArray | None, xr.Dataset | None]:
    """
    Express the modelled series per unit area, like the observations.

    Each series is divided by the model's own ice area for that year (the
    ``area`` coordinate/variable the loaders attach). Runs whose files
    carry no area fall back to the observed area of the year.

    Parameters
    ----------
    model : xarray.DataArray or None
        Annual total rates in Gt/yr from :func:`load_model_series`.
    seasons : xarray.Dataset or None
        Seasonal balances in Gt from :func:`model_seasonal_balances`.
    obs : xarray.Dataset
        Observations from :func:`load_glacier_wide` (``area`` in km²).

    Returns
    -------
    tuple
        ``(model, seasons)`` in ``m year^-1``, None where the input was None.
    """
    obs_area = (obs["area"].pint.quantify().pint.to("m^2").pint.dequantify()).astype(float)
    obs_area.attrs["units"] = "m^2"

    def _area_for(values: xr.DataArray, area: xr.DataArray | None) -> xr.DataArray:
        """
        Area to divide by, filled from the observations where the model has none.

        Parameters
        ----------
        values : xarray.DataArray
            Series being converted (for its ``time`` axis and name).
        area : xarray.DataArray or None
            Model ice area in m², possibly with NaN years.

        Returns
        -------
        xarray.DataArray
            Area in m² on the series' years.
        """
        fallback = obs_area.reindex(time=values["time"])
        if area is None:
            logger.warning("%s: model files carry no ice area; using the observed area", values.name)
            area = fallback
        else:
            area = area.where(np.isfinite(area), fallback).astype(float)
        area.attrs["units"] = "m^2"
        return area

    model_mwe = None
    if model is not None:
        model_mwe = specific_balance(model, _area_for(model, model.coords.get("area")))
        model_mwe.name = model.name
    seasons_mwe = None
    if seasons is not None:
        area = _area_for(seasons["Bw"], seasons["area"] if "area" in seasons else None)
        seasons_mwe = xr.Dataset({var: specific_balance(seasons[var], area) for var in BALANCE_VARS if var in seasons})
    return model_mwe, seasons_mwe


def load_model_series(files: Sequence[Path | str], rgi_id: str, root: Path | str | None = None) -> xr.DataArray | None:
    """
    Collect a glacier's annual ``tendency_of_ice_mass`` from every file that has it.

    Parameters
    ----------
    files : sequence of Path or str
        Post-processed per-glacier scalar files.
    rgi_id : str
        RGI ``-G-`` identifier to select.
    root : Path or str or None, optional
        Run directory the ``run`` labels are made relative to.

    Returns
    -------
    xarray.DataArray or None
        Annual-mean rate in Gt/yr on ``(run, time)`` with integer years,
        carrying the year's mean ice area (m²) as the ``area`` coordinate
        (NaN for files without one); None when no file contains the glacier.
    """
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    delta_coder = xr.coders.CFTimedeltaCoder()
    series: list[xr.DataArray] = []
    labels: list[str] = []
    for file in files:
        file = Path(file)
        with xr.open_dataset(file, decode_times=time_coder, decode_timedelta=delta_coder) as ds:
            spacing = _grid_spacing(ds)
            ds = with_region_labels(ds.drop_vars("pism_config", errors="ignore"))
            names = ds[REGION_DIM].values
            if names.dtype.kind == "S":
                ds = ds.assign_coords({REGION_DIM: names.astype(str)})
            if MODEL_VAR not in ds or rgi_id not in set(ds[REGION_DIM].values.tolist()):
                continue
            da = ds[MODEL_VAR].sel({REGION_DIM: rgi_id}).pint.quantify().pint.to("Gt/year").pint.dequantify().load()
            years = _mean_years(ds)
            area = ice_area(ds, rgi_id, spacing)
            if area is None:
                area = xr.full_like(ds["time"], np.nan, dtype=float)
            da = da.assign_coords(year=("time", years)).swap_dims({"time": "year"}).drop_vars("time")
            da = da.groupby("year").mean("year").rename({"year": "time"})
            area = area.assign_coords(year=("time", years)).swap_dims({"time": "year"}).drop_vars("time")
            area = area.groupby("year").mean("year").rename({"year": "time"})
            da = da.assign_coords(area=("time", area.values)).drop_vars(REGION_DIM, errors="ignore")
        label = _run_label(file, root)
        series.append(da)
        labels.append(label)
        logger.info("%s found in %s", rgi_id, label)
    if not series:
        return None
    model = xr.concat(series, dim=pd.Index(labels, name="run"), join="outer").sortby("time")
    model.attrs = {"units": "Gt year^-1", "long_name": "rate of change of the ice mass"}
    model.name = MODEL_VAR
    return model


def _month_edges(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calendar-month interval each monthly stamp represents.

    PISM stamps a monthly mean at the middle of its month, so the interval is
    the calendar month the stamp falls in.

    Parameters
    ----------
    times : numpy.ndarray
        Decoded time stamps (cftime or numpy datetimes).

    Returns
    -------
    tuple of numpy.ndarray
        ``(starts, ends)`` as pandas Timestamps, ``ends`` exclusive.
    """
    starts, ends = [], []
    for t in times:
        stamp = pd.Timestamp(str(t)[:19])
        start = stamp.normalize().replace(day=1)
        starts.append(start)
        ends.append(start + pd.offsets.MonthBegin(1))
    return np.array(starts), np.array(ends)


def is_monthly(times: np.ndarray) -> bool:
    """
    Report whether a time axis is monthly rather than annual.

    Parameters
    ----------
    times : numpy.ndarray
        Decoded time stamps.

    Returns
    -------
    bool
        True when consecutive stamps are less than 45 days apart.
    """
    if len(times) < 2:
        return False
    spans = [(pd.Timestamp(str(b)[:19]) - pd.Timestamp(str(a)[:19])).days for a, b in zip(times[:-1], times[1:])]
    return bool(np.median(spans) < 45)


def integrate_rate(rate: xr.DataArray, start: pd.Timestamp, end: pd.Timestamp, days_per_year: float = 365.0) -> float:
    """
    Integrate a monthly rate over an arbitrary interval.

    Each monthly value is a mean rate over its calendar month, so a month
    straddling a season boundary contributes in proportion to the days it
    shares with the interval. The seasons are bounded by *measured* dates,
    which fall mid-month far more often than not.

    Parameters
    ----------
    rate : xarray.DataArray
        Monthly rate in Gt/yr, with ``month_start``/``month_end`` coordinates.
    start, end : pandas.Timestamp
        Interval to integrate over; ``end`` exclusive.
    days_per_year : float, default 365.0
        Days per year of the model calendar, converting the rate to a mass.

    Returns
    -------
    float
        Mass change over the interval, in Gt. NaN when the interval is not
        fully covered by the model record.
    """
    # xarray stores the coordinates as datetime64, so work in that dtype
    # throughout rather than assuming Timestamps survive the round trip.
    starts = pd.DatetimeIndex(rate["month_start"].values).values
    ends = pd.DatetimeIndex(rate["month_end"].values).values
    lower, upper = np.datetime64(start), np.datetime64(end)
    if lower < starts.min() or upper > ends.max():
        return float("nan")

    overlap = (np.minimum(ends, upper) - np.maximum(starts, lower)) / np.timedelta64(1, "D")
    overlap = np.clip(overlap.astype(float), 0.0, None)
    return float(np.nansum(rate.values * overlap) / days_per_year)


def model_seasonal_balances(
    files: Sequence[Path | str],
    rgi_id: str,
    obs: xr.Dataset,
    root: Path | str | None = None,
    variable: str = MODEL_SEASONAL_VAR,
) -> xr.Dataset | None:
    """
    Integrate the modelled surface mass flux over each measured season.

    A balance year runs from one annual minimum to the next and is split at the
    winter maximum, both dated in the release. The winter balance is therefore
    the mass change from the *previous* year's ``Ba_Date`` to this year's
    ``Bw_Date``, and the summer balance runs from there to this year's
    ``Ba_Date``. Years where the release gives no dates — or that the model
    record does not fully span — are left out, so every value plotted covers
    exactly the interval its observation does.

    Requires monthly model output; annual files carry no seasonal information
    and yield ``None``.

    Parameters
    ----------
    files : sequence of Path or str
        Post-processed per-glacier scalar files.
    rgi_id : str
        RGI ``-G-`` identifier to select.
    obs : xarray.Dataset
        Observations carrying ``Bw_Date``/``Ba_Date`` (see
        :func:`load_glacier_wide`).
    root : Path or str or None, optional
        Run directory the ``run`` labels are made relative to.
    variable : str, optional
        Model variable to integrate. Defaults to
        :data:`MODEL_SEASONAL_VAR`.

    Returns
    -------
    xarray.Dataset or None
        ``Bw``/``Bs`` in Gt on ``(run, time)`` over balance years, plus the
        derived annual ``Ba``. None when no file has monthly output for the
        glacier, or the release dates no season.
    """
    if not all(column in obs for column in SEASON_DATE_COLUMNS):
        logger.info("%s: release gives no season dates; no modelled seasons", rgi_id)
        return None

    years = obs["time"].values.astype(int)
    bw_dates = pd.to_datetime(obs["Bw_Date"].values)
    ba_dates = pd.to_datetime(obs["Ba_Date"].values)
    # Winter starts at the previous balance year's minimum, so a year needs its
    # own two dates and its predecessor's ``Ba_Date``.
    previous = {int(y): ba for y, ba in zip(years, ba_dates)}

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    delta_coder = xr.coders.CFTimedeltaCoder()
    per_run: list[xr.Dataset] = []
    labels: list[str] = []
    for file in files:
        file = Path(file)
        with xr.open_dataset(file, decode_times=time_coder, decode_timedelta=delta_coder) as ds:
            spacing = _grid_spacing(ds)
            ds = with_region_labels(ds.drop_vars("pism_config", errors="ignore"))
            names = ds[REGION_DIM].values
            if names.dtype.kind == "S":
                ds = ds.assign_coords({REGION_DIM: names.astype(str)})
            if variable not in ds or rgi_id not in set(ds[REGION_DIM].values.tolist()):
                continue
            if not is_monthly(ds["time"].values):
                logger.info("%s: %s is not monthly; no modelled seasons from it", rgi_id, file.name)
                continue
            area = ice_area(ds, rgi_id, spacing)
            area_by_year = (
                None
                if area is None
                else area.assign_coords(year=("time", _mean_years(ds)))
                .swap_dims({"time": "year"})
                .groupby("year")
                .mean("year")
            )
            rate = (
                ds[variable]
                .sel({REGION_DIM: rgi_id})
                .pint.quantify()
                .pint.to("Gt/year")
                .pint.dequantify()
                .load()
                .drop_vars(REGION_DIM, errors="ignore")
            )
            starts, ends = _month_edges(ds["time"].values)
            rate = rate.assign_coords(month_start=("time", starts), month_end=("time", ends))

        winter, summer, areas, kept = [], [], [], []
        for year, bw_date, ba_date in zip(years, bw_dates, ba_dates):
            year_start = previous.get(int(year) - 1)
            if pd.isna(bw_date) or pd.isna(ba_date) or year_start is None or pd.isna(year_start):
                continue
            b_w = integrate_rate(rate, year_start, bw_date)
            b_s = integrate_rate(rate, bw_date, ba_date)
            if np.isnan(b_w) and np.isnan(b_s):
                continue
            winter.append(b_w)
            summer.append(b_s)
            # the year's mean ice area, for the conversion to metres water equivalent
            areas.append(
                float(area_by_year.sel(year=int(year)))
                if area_by_year is not None and int(year) in area_by_year["year"].values
                else np.nan
            )
            kept.append(int(year))

        if not kept:
            continue
        per_run.append(
            xr.Dataset(
                {
                    "Bw": ("time", np.array(winter)),
                    "Bs": ("time", np.array(summer)),
                    "area": ("time", np.array(areas), {"units": "m^2", "long_name": "ice-covered area"}),
                },
                coords={"time": np.array(kept)},
            )
        )
        labels.append(_run_label(file, root))
        logger.info("%s: %d modelled seasons from %s", rgi_id, len(kept), labels[-1])

    if not per_run:
        return None

    out = xr.concat(per_run, dim=pd.Index(labels, name="run"), join="outer").sortby("time")
    out["Ba"] = out["Bw"] + out["Bs"]
    for var, season in BALANCE_VARS.items():
        out[var].attrs = {
            "units": "Gt year^-1",
            "long_name": f"modelled {season} surface mass balance",
            "source_variable": variable,
        }
    return out


def _ensemble_line(values: xr.DataArray) -> xr.DataArray:
    """
    Collapse an ensemble to the series the plot draws.

    Parameters
    ----------
    values : xarray.DataArray
        Modelled values, on ``(run, time)`` or just ``time``.

    Returns
    -------
    xarray.DataArray
        The single run, or the median across runs.
    """
    if "run" in values.dims:
        return values.median(dim="run") if values.sizes["run"] > 1 else values.isel(run=0, drop=True)
    return values


def _score(observed: xr.DataArray, modelled: xr.DataArray) -> dict[str, float]:
    """
    Compare two annual series over the years both have a value for.

    Parameters
    ----------
    observed : xarray.DataArray
        Observed balances on integer ``time``.
    modelled : xarray.DataArray
        Modelled balances on integer ``time``.

    Returns
    -------
    dict
        ``n`` overlapping years, Pearson ``r`` (NaN below
        :data:`SKILL_MIN_YEARS` years or for a constant series), ``mae``
        and ``bias`` (model minus observation), in the series' units.
    """
    obs, mod = xr.align(observed, modelled, join="inner")
    o = np.asarray(obs.values, dtype=float)
    m = np.asarray(mod.values, dtype=float)
    keep = np.isfinite(o) & np.isfinite(m)
    o, m = o[keep], m[keep]
    n = int(o.size)
    if n == 0:
        return {"n": 0, "r": np.nan, "mae": np.nan, "bias": np.nan}
    r = np.nan
    if n >= SKILL_MIN_YEARS and o.std() > 0 and m.std() > 0:
        r = float(np.corrcoef(o, m)[0, 1])
    return {"n": n, "r": r, "mae": float(np.mean(np.abs(m - o))), "bias": float(np.mean(m - o))}


def skill_scores(obs: xr.Dataset, model: xr.DataArray | None = None, seasons: xr.Dataset | None = None) -> pd.DataFrame:
    """
    Score the modelled balances against the observations.

    Every comparison uses the series the plot draws — the single run, or the
    ensemble median — over the years both sides have. The seasonal and annual
    surface balances are the natural counterparts of the stake-derived
    observations; the total ``tendency_of_ice_mass`` is scored against the
    annual balance as well, labelled ``total``, since it is what the annual
    model line shows.

    Parameters
    ----------
    obs : xarray.Dataset
        Balances in Gt/yr from :func:`to_mass_rate`.
    model : xarray.DataArray or None, optional
        Annual total mass-change rates from :func:`load_model_series`.
    seasons : xarray.Dataset or None, optional
        Modelled seasonal balances from :func:`model_seasonal_balances`.

    Returns
    -------
    pandas.DataFrame
        One row per available comparison with :data:`SKILL_COLUMNS`;
        empty when there is nothing to compare.
    """
    rows: list[dict[str, Any]] = []
    if seasons is not None:
        for var, season in BALANCE_VARS.items():
            if var in obs and var in seasons:
                rows.append(
                    {
                        "variable": var,
                        "season": season,
                        "source": "surface",
                        **_score(obs[var], _ensemble_line(seasons[var])),
                    }
                )
    if model is not None and "Ba" in obs:
        rows.append(
            {"variable": "Ba", "season": "annual", "source": "total", **_score(obs["Ba"], _ensemble_line(model))}
        )
    return pd.DataFrame(rows, columns=SKILL_COLUMNS)


def format_skill(skill: pd.DataFrame, units: str = "Gt/yr") -> str:
    """
    Lay the scores out as the text block drawn on the figure.

    Parameters
    ----------
    skill : pandas.DataFrame
        Output of :func:`skill_scores`.
    units : str, optional
        Units appended to the error statistics.

    Returns
    -------
    str
        One line per comparison, e.g. ``winter  r=0.62  MAE=0.11 Gt/yr  n=30``.
    """
    lines = []
    for row in skill.itertuples(index=False):
        if row.n == 0:
            continue
        label = f"{row.season} ({row.source})" if row.source != "surface" else row.season
        r = f"r={row.r:.2f}" if np.isfinite(row.r) else "r=n/a"
        lines.append(f"{label:16s} {r:>7s}  MAE={row.mae:.3g} {units}  n={row.n}")
    return "\n".join(lines)


def plot_glacier(
    obs: xr.Dataset,
    model: xr.DataArray | None,
    glacier: str,
    rgi_id: str,
    output_dir: Path | str,
    *,
    seasons: xr.Dataset | None = None,
    skill: pd.DataFrame | None = None,
    area_km2: float | None = None,
) -> Path:
    """
    Plot observed seasonal and annual balances with the modelled mass-change rate.

    Everything is drawn as a specific balance in metres water equivalent per
    year. A second axis on the right reads the same curves in Gt/yr at a
    fixed area — the mean observed area — since one scale factor cannot
    follow the year-to-year area changes that went into the conversion.

    Parameters
    ----------
    obs : xarray.Dataset
        Balances in m w.e. per year from :func:`load_glacier_wide`.
    model : xarray.DataArray or None
        Modelled specific balances on ``(run, time)`` from
        :func:`to_specific_balances`. Several runs are drawn as their median
        and 5-95 % range.
    glacier : str
        Glacier name for the title and file name.
    rgi_id : str
        RGI identifier for the title and file name.
    output_dir : Path or str
        Directory the figure is written to.
    seasons : xarray.Dataset or None, optional
        Modelled seasonal balances from :func:`model_seasonal_balances`,
        drawn as lines in their observation's colour. ``None`` (the case for
        annual model output) leaves the plot as it was.
    skill : pandas.DataFrame or None, optional
        Scores from :func:`skill_scores`, written into the top-left corner.
    area_km2 : float or None, optional
        Glacier area the right-hand Gt/yr axis is scaled with. Defaults to
        the mean of ``obs["area"]``.

    Returns
    -------
    Path
        The PNG written (a PDF sits next to it).
    """
    output_dir = Path(output_dir)
    if area_km2 is None and "area" in obs:
        area_km2 = float(obs["area"].mean())
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"usgs_benchmark_{glacier}_{rgi_id}"

    with mpl.rc_context(rc=rc_params):
        fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.6), layout="constrained")
        for var, season in BALANCE_VARS.items():
            if var not in obs:
                continue
            style = OBS_STYLE[var]
            unc = _uncertainty_var(obs, var)
            kwargs: dict[str, Any] = {
                "color": style["color"],
                "marker": style["marker"],
                "ms": 2,
                "lw": 0.75,
                "label": f"Observed {season}",
            }
            if unc is not None:
                ax.errorbar(obs["time"], obs[var], yerr=obs[unc], capsize=1, elinewidth=0.5, **kwargs)
            else:
                ax.plot(obs["time"], obs[var], **kwargs)

        if seasons is not None:
            for var, style in MODEL_SEASON_STYLE.items():
                if var not in seasons:
                    continue
                season = BALANCE_VARS[var]
                values = seasons[var]
                if values.sizes["run"] == 1:
                    ax.plot(
                        values["time"],
                        values.isel(run=0),
                        color=style["color"],
                        ls=style["ls"],
                        lw=1.0,
                        label=f"PISM {season} (surface)",
                    )
                else:
                    ax.fill_between(
                        values["time"],
                        values.quantile(0.05, dim="run"),
                        values.quantile(0.95, dim="run"),
                        color=style["color"],
                        alpha=0.2,
                        lw=0,
                    )
                    ax.plot(
                        values["time"],
                        values.median(dim="run"),
                        color=style["color"],
                        ls=style["ls"],
                        lw=1.0,
                        label=f"PISM {season} (surface, n={values.sizes['run']})",
                    )

        if model is not None:
            if model.sizes["run"] == 1:
                ax.plot(
                    model["time"],
                    model.isel(run=0),
                    color=MODEL_COLOR,
                    lw=1.0,
                    label=f"PISM ({model['run'].values[0]})",
                )
            else:
                low = model.quantile(0.05, dim="run")
                high = model.quantile(0.95, dim="run")
                ax.fill_between(model["time"], low, high, color=MODEL_COLOR, alpha=0.25, lw=0)
                ax.plot(
                    model["time"],
                    model.median(dim="run"),
                    color=MODEL_COLOR,
                    lw=1.0,
                    label=f"PISM median, 5-95% (n={model.sizes['run']})",
                )

        if skill is not None and (text := format_skill(skill, units=SPECIFIC_LABEL)):
            ax.text(
                0.02,
                0.98,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5,
                family="monospace",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )

        ax.axhline(y=0, color="k", ls="dotted", lw=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Balance year")
        ax.set_ylabel("Mass balance (m w.e. yr$^{-1}$)")
        if area_km2 is not None and np.isfinite(area_km2) and area_km2 > 0:
            # 1 m w.e. over 1 km^2 is 1e-3 Gt
            factor = area_km2 * 1e-3
            secondary = ax.secondary_yaxis(
                "right", functions=(lambda m: np.asarray(m) * factor, lambda g: np.asarray(g) / factor)
            )
            secondary.set_ylabel(f"Mass balance (Gt yr$^{{-1}}$ at {area_km2:.3g} km$^2$)")
        ax.set_title(f"{glacier} ({rgi_id})")
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc="outside lower center", ncol=2)
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)
        png = output_dir / f"{stem}.png"
        fig.savefig(png, dpi=300)
        fig.savefig(output_dir / f"{stem}.pdf")
        plt.close(fig)
    return png


def run_pipeline(
    run_dir: Path | str | None,
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    output_dir: Path | str = ".",
    rgi_file: Path | str | None = None,
    uncertainty: float | None = None,
    force_overwrite: bool = False,
) -> pd.DataFrame:
    """
    Download, match, convert and plot every glacier in the release.

    Parameters
    ----------
    run_dir : Path or str or None
        Directory searched recursively for ``scalar_G_*.nc``; None plots
        the observations alone.
    data_dir : Path or str, optional
        Cache for the ScienceBase archives and RGI outlines.
    output_dir : Path or str, optional
        Where figures, per-glacier NetCDF files and the match table go.
    rgi_file : Path or str or None, optional
        Outline file to match against instead of the downloaded regions.
    uncertainty : float or None, optional
        Constant one-sigma uncertainty in m w.e. applied to every balance
        when the CSV carries none.
    force_overwrite : bool, default False
        Re-download the archives.

    Returns
    -------
    pandas.DataFrame
        The match table with ``figure`` and ``n_runs`` columns added.
    """
    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = download_usgs_benchmark(data_dir, force_overwrite=force_overwrite)
    sites = load_sites(paths["sites"])
    rgi = load_rgi_glaciers(data_dir, rgi_file=rgi_file, force_overwrite=force_overwrite)
    matches = match_rgi_ids(sites, rgi)

    model_files = find_model_files(run_dir) if run_dir is not None else []
    logger.info("%d scalar_G files under %s", len(model_files), run_dir)

    figures: list[str | None] = []
    n_runs: list[int] = []
    skills: list[pd.DataFrame] = []
    for row in matches.itertuples(index=False):
        if pd.isna(row.rgi_id):
            figures.append(None)
            n_runs.append(0)
            continue
        obs = load_glacier_wide(paths["data"], row.glacier, fallback_area_km2=row.area_km2)
        if obs is None:
            logger.info("%s: point data only, no glacier-wide balances; skipping", row.glacier)
            figures.append(None)
            n_runs.append(0)
            continue
        if uncertainty is not None:
            for var in BALANCE_VARS:
                if var in obs and _uncertainty_var(obs, var) is None:
                    obs[f"{var}_unc"] = xr.full_like(obs[var], uncertainty)
                    obs[f"{var}_unc"].attrs = {
                        "units": "m year^-1",
                        "long_name": f"assumed uncertainty of {var} (m w.e.)",
                    }
        rate = to_mass_rate(obs)
        model = load_model_series(model_files, row.rgi_id, root=run_dir) if model_files else None
        seasons = model_seasonal_balances(model_files, row.rgi_id, rate, root=run_dir) if model_files else None
        model_mwe, seasons_mwe = to_specific_balances(model, seasons, obs)
        skill = skill_scores(obs, model_mwe, seasons_mwe)
        if not skill.empty:
            skills.append(skill.assign(glacier=row.glacier, rgi_id=row.rgi_id, units=SPECIFIC_UNITS))
            for line in format_skill(skill, units=SPECIFIC_LABEL).splitlines():
                logger.info("%s: %s", row.glacier, line)
        png = plot_glacier(obs, model_mwe, row.glacier, row.rgi_id, output_dir, seasons=seasons_mwe, skill=skill)
        figures.append(str(png))
        n_runs.append(0 if model is None else int(model.sizes["run"]))

        mwe = [v for v in obs.data_vars if v != "area" and np.issubdtype(obs[v].dtype, np.number)]
        out = xr.merge([rate, obs[mwe].rename({v: f"{v}_mwe" for v in mwe})])
        if model is not None:
            out[MODEL_VAR] = model.drop_vars("area", errors="ignore")
            out[f"{MODEL_VAR}_mwe"] = model_mwe
            if "area" in model.coords:
                area = model.coords["area"]
                out["model_area"] = xr.DataArray(
                    area.values,
                    dims=area.dims,
                    coords={dim: model.coords[dim] for dim in area.dims},
                    attrs={"units": "m^2", "long_name": "modelled ice-covered area"},
                )
        if seasons is not None and seasons_mwe is not None:
            for var in BALANCE_VARS:
                if var in seasons:
                    out[f"{var}_model"] = seasons[var]
                    out[f"{var}_model_mwe"] = seasons_mwe[var]
        for score in skill.itertuples(index=False):
            name = f"{MODEL_VAR}_mwe" if score.source == "total" else f"{score.variable}_model_mwe"
            if name in out:
                out[name].attrs.update(
                    {
                        "pearson_r": score.r,
                        "mae": score.mae,
                        "bias": score.bias,
                        "n_years": score.n,
                        "skill_units": SPECIFIC_UNITS,
                    }
                )
        out.attrs["rgi_id"] = row.rgi_id
        out.to_netcdf(output_dir / f"usgs_benchmark_{row.glacier}_{row.rgi_id}.nc")

    matches = matches.assign(figure=figures, n_runs=n_runs)
    matches.to_csv(output_dir / "usgs_benchmark_rgi_match.csv", index=False)
    columns = ["glacier", "rgi_id", *SKILL_COLUMNS, "units"]
    skill_table = pd.concat(skills, ignore_index=True) if skills else pd.DataFrame(columns=columns)
    skill_table = skill_table[columns]
    skill_table.to_csv(output_dir / "usgs_benchmark_skill.csv", index=False)
    return matches


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Compare USGS benchmark-glacier mass balances with PISM output.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments (excluding the program name). When
        ``None``, :data:`sys.argv` is used.

    Returns
    -------
    pandas.DataFrame
        The glacier-to-RGI match table.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Plot USGS glacier-wide mass balances (Gt/yr) against PISM tendency_of_ice_mass."
    parser.add_argument(
        "RUN_DIR",
        nargs="?",
        default=None,
        help="Directory searched recursively for scalar_G_*.nc files. Omit to plot the observations alone.",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Cache for the USGS archives and RGI outlines.")
    parser.add_argument("--output-dir", default=".", help="Directory for figures, NetCDF files and the match table.")
    parser.add_argument(
        "--rgi-glacier-file",
        default=None,
        help="RGI v7 glacier (-G-) outlines to match against instead of downloading regions 01 and 02.",
    )
    parser.add_argument(
        "--uncertainty",
        type=float,
        default=None,
        help="Constant one-sigma uncertainty (m w.e.) to draw as error bars when the CSVs carry none.",
    )
    parser.add_argument("--force-overwrite", action="store_true", default=False, help="Re-download the archives.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "usgs_benchmark.log")

    matches = run_pipeline(
        args.RUN_DIR,
        data_dir=args.data_dir,
        output_dir=output_dir,
        rgi_file=args.rgi_glacier_file,
        uncertainty=args.uncertainty,
        force_overwrite=args.force_overwrite,
    )
    print(matches.to_string(index=False))
    return matches


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
