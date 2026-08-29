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
Shared pieces of the USGS benchmark-glacier comparisons.

The USGS ScienceBase release *Compiled Input Data and Glacier-Wide Mass
Balances* (https://doi.org/10.5066/F7HD7SRF) publishes glacier-wide winter,
summer and annual balances plus the stake measurements behind them for a
handful of North American glaciers, keyed by glacier *name* rather than RGI
ID. This module holds what every consumer needs: reading the release's
tables, placing its glaciers in RGI v7 outlines, the time bookkeeping that
turns PISM's monthly rates into balances over measured intervals, and the
skill statistics. The command-line tools live in
:mod:`pism_terra.glacier.usgs_benchmark_glaciers`,
:mod:`pism_terra.glacier.usgs_benchmark_stakes` and
:mod:`pism_terra.glacier.usgs_generate_geopackage`; downloading is in
:func:`pism_terra.download.download_usgs_benchmark`.
"""

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from pism_terra.glacier.rgi import prepare_rgi_region
from pism_terra.workflow import pism_config_value

logger = logging.getLogger("pism_terra.glacier.usgs")

DEFAULT_DATA_DIR = "~/base/pism-terra/usgs_benchmark"

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
#: Columns of the release's glacier-wide solution giving the measured season
#: boundaries: the winter maximum ends the winter season, the annual minimum
#: ends the balance year.
SEASON_DATE_COLUMNS = ("Bw_Date", "Ba_Date")

# The point-measurement tables: the per-glacier CSV suffix and its date columns.
LAYERS = {
    "stakes": ("Glaciological_Data", ("spring_date", "fall_date")),
    "subseasonal": ("SubSeasonal_Glaciological_Data", ("Date1", "Date2")),
}
# The release writes dates as YYYY/MM/DD in most files and M/D/YYYY in a few
# (Kennicott); both are tried, in this order.
DATE_FORMATS = ("%Y/%m/%d", "%m/%d/%Y")

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


# --- The release ---------------------------------------------------------------


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


def uncertainty_var(ds: xr.Dataset | pd.DataFrame, var: str) -> str | None:
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
        unc = uncertainty_var(df, var)
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


# --- RGI ------------------------------------------------------------------------


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


def rgi_output_dir(output_dir: Path | str, rgi_id: str) -> Path:
    """
    Directory a glacier's outputs go to, created on demand.

    Parameters
    ----------
    output_dir : Path or str
        Top-level output directory.
    rgi_id : str
        RGI identifier naming the sub-directory.

    Returns
    -------
    Path
        ``<output_dir>/<rgi_id>``.
    """
    path = Path(output_dir).expanduser() / str(rgi_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- Model files and time bookkeeping ----------------------------------------------


def find_model_files(root: Path | str, pattern: str = "scalar_G_*.nc") -> list[Path]:
    """
    Find model output files below a run directory.

    Parameters
    ----------
    root : Path or str
        Directory searched recursively.
    pattern : str, optional
        Glob pattern; the default is the post-processed per-glacier scalar files.

    Returns
    -------
    list of Path
        Matching files, sorted.
    """
    return sorted(Path(root).expanduser().rglob(pattern))


def open_pism(file: Path | str, **kwargs: Any) -> xr.Dataset:
    """
    Open a PISM output file with its time axis decoded to ``cftime`` objects.

    Parameters
    ----------
    file : Path or str
        NetCDF file.
    **kwargs
        Passed on to :func:`xarray.open_dataset`.

    Returns
    -------
    xarray.Dataset
        The dataset, lazily loaded.
    """
    return xr.open_dataset(
        file,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
        decode_timedelta=xr.coders.CFTimedeltaCoder(),
        **kwargs,
    )


def grid_spacing(ds: xr.Dataset) -> tuple[float, float] | None:
    """
    Read the grid spacing PISM records in ``pism_config``.

    Parameters
    ----------
    ds : xarray.Dataset
        File as opened, before ``pism_config`` is dropped.

    Returns
    -------
    tuple of float or None
        ``(dx, dy)`` in metres, or None when the file carries no config.
    """
    dx = pism_config_value(ds, "grid.dx")
    dy = pism_config_value(ds, "grid.dy")
    try:
        return float(dx), float(dy)
    except (TypeError, ValueError):
        return None


def run_label(file: Path, root: Path | str | None) -> str:
    """
    Name a run by its directory under the search root and its ensemble id.

    Parameters
    ----------
    file : Path
        Model output file.
    root : Path or str or None
        Search root; None labels the run by file name.

    Returns
    -------
    str
        ``<subdirectory> id_<n>`` (with `` uq_<m>`` appended for an
        ensemble member) when both are known, otherwise the file name (or
        ``<subdirectory>/<file name>`` when no ``id_<n>`` token exists).
    """
    if root is None or not file.is_relative_to(root):
        return file.name
    relative = file.relative_to(root)
    if len(relative.parts) == 1:
        return file.name
    subdir = relative.parts[0]
    match = re.search(r"_id_(\d+)(?:_uq_(\d+))?_", file.name)
    if match is None:
        return f"{subdir}/{file.name}"
    label = f"{subdir} id_{match.group(1)}"
    return f"{label} uq_{match.group(2)}" if match.group(2) is not None else label


def to_timestamp(t: Any) -> pd.Timestamp:
    """
    Convert a cftime, numpy or pandas time stamp to a pandas Timestamp.

    The conversion goes through the calendar date, so a ``365_day`` or
    ``standard`` model date maps to the same day of the proleptic Gregorian
    calendar — which is what comparing against measured dates needs.

    Parameters
    ----------
    t : object
        Time stamp.

    Returns
    -------
    pandas.Timestamp
        The same calendar date and time.
    """
    if isinstance(t, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(t)
    return pd.Timestamp(str(t)[:19])


def year_of(t: Any) -> int:
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


def mean_years(ds: xr.Dataset) -> np.ndarray:
    """
    Label each reporting interval by the calendar year it covers.

    Uses the interval start from ``time_bounds`` when the file has it. The
    post-processed files drop the bounds and keep the interval midpoint as
    the time stamp, so a stamp on 1 January is taken as the *end* of the
    previous year's interval and any other stamp as lying inside its year.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with a decoded (cftime or datetime) ``time`` axis.

    Returns
    -------
    numpy.ndarray
        Integer year per time step.
    """
    bounds_name = ds["time"].attrs.get("bounds", "time_bounds")
    if bounds_name in ds:
        starts = ds[bounds_name].isel({ds[bounds_name].dims[-1]: 0}).values
        return np.array([year_of(t) for t in starts], dtype=int)
    years = []
    for t in ds["time"].values:
        t = pd.Timestamp(t) if isinstance(t, np.datetime64) else t
        years.append(year_of(t) - 1 if (t.month, t.day) == (1, 1) else year_of(t))
    return np.array(years, dtype=int)


def month_edges(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        start = to_timestamp(t).normalize().replace(day=1)
        starts.append(start)
        ends.append(start + pd.offsets.MonthBegin(1))
    return np.array(starts), np.array(ends)


def interval_edges(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Reporting interval of each time step.

    Taken from the file's ``time_bounds`` when it has decoded bounds, which
    is exact whatever the reporting frequency; otherwise from the stamps,
    assuming monthly means stamped mid-month (:func:`month_edges`).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with a decoded ``time`` axis.

    Returns
    -------
    tuple of numpy.ndarray
        ``(starts, ends)`` as pandas Timestamps, ``ends`` exclusive.
    """
    bounds_name = ds["time"].attrs.get("bounds", "time_bounds")
    if bounds_name in ds and not np.issubdtype(ds[bounds_name].dtype, np.number):
        bounds = ds[bounds_name].values
        starts = np.array([to_timestamp(t) for t in bounds[:, 0]])
        ends = np.array([to_timestamp(t) for t in bounds[:, 1]])
        return starts, ends
    return month_edges(ds["time"].values)


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
    spans = [(to_timestamp(b) - to_timestamp(a)).days for a, b in zip(times[:-1], times[1:])]
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
        Monthly rate (any per-year unit), with ``month_start``/``month_end``
        coordinates on its ``time`` dimension.
    start, end : pandas.Timestamp
        Interval to integrate over; ``end`` exclusive.
    days_per_year : float, default 365.0
        Days per year of the model calendar, converting the rate to an amount.

    Returns
    -------
    float
        Amount accumulated over the interval, in the rate's unit times years.
        NaN when the interval is not fully covered by the model record or
        the rate is missing for a month inside it.
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
    covered = overlap > 0
    values = np.asarray(rate.values, dtype=float)[covered]
    # A gap inside the interval (a cell the model has deglaciated, say) makes
    # the balance unknown rather than smaller.
    if np.isnan(values).any():
        return float("nan")
    return float(np.sum(values * overlap[covered]) / days_per_year)


# --- Skill --------------------------------------------------------------------------


def ensemble_line(values: xr.DataArray) -> xr.DataArray:
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


def score(observed: xr.DataArray | np.ndarray, modelled: xr.DataArray | np.ndarray) -> dict[str, float]:
    """
    Compare two series over the entries both have a value for.

    Parameters
    ----------
    observed : xarray.DataArray or numpy.ndarray
        Observed balances; DataArrays are aligned on their coordinates.
    modelled : xarray.DataArray or numpy.ndarray
        Modelled balances.

    Returns
    -------
    dict
        ``n`` overlapping entries, Pearson ``r`` (NaN below
        :data:`SKILL_MIN_YEARS` entries or for a constant series), ``mae``
        and ``bias`` (model minus observation), in the series' units.
    """
    if isinstance(observed, xr.DataArray) and isinstance(modelled, xr.DataArray):
        observed, modelled = xr.align(observed, modelled, join="inner")
    o = np.asarray(observed, dtype=float).ravel()
    m = np.asarray(modelled, dtype=float).ravel()
    keep = np.isfinite(o) & np.isfinite(m)
    o, m = o[keep], m[keep]
    n = int(o.size)
    if n == 0:
        return {"n": 0, "r": np.nan, "mae": np.nan, "bias": np.nan}
    r = np.nan
    if n >= SKILL_MIN_YEARS and o.std() > 0 and m.std() > 0:
        r = float(np.corrcoef(o, m)[0, 1])
    return {"n": n, "r": r, "mae": float(np.mean(np.abs(m - o))), "bias": float(np.mean(m - o))}


def format_skill(skill: pd.DataFrame, units: str = "Gt/yr") -> str:
    """
    Lay the scores out as the text block drawn on the figure.

    Parameters
    ----------
    skill : pandas.DataFrame
        Rows with ``season``, ``source``, ``n``, ``r`` and ``mae`` columns.
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
