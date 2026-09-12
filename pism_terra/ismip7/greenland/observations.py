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
Observed Greenland mass balance, for validating ISMIP7 runs against.

Three independent estimates of how much mass Greenland has lost, put on a
common footing so a run's ``basin_*`` output can be compared with them:

- **GRACE GSFC mascons** — 0.5 degree equivalent-water-thickness fields,
  integrated to mass per cell so they can be summed over any region.
- **GRACE Tellus mascons** — the ice-sheet-wide time series PO.DAAC
  publishes, an independent processing of the same missions.
- **Mankoff et al. (2021)** — the input-output estimate, per Mouginot basin
  and for the ice sheet as a whole, which resolves the flux components
  (discharge, SMB, basal melt) that GRACE only sees in sum.

Downloads are cached and re-used: the GSFC file alone is ~500 MB, and the
derived products are only rebuilt when missing or when ``--force-overwrite``
is given. Point ``--cache-path`` at a shared directory to prepare these once
for many runs.

The three products are copied into ``<OUTPUT_PATH>/output/observations/``
beside a run's own output, which is where the analysis reads them from.
``pism-ismip7-greenland-stage`` does this as part of staging, so the
observations are in place before the run starts; the console entry point
here is for preparing or refreshing them on their own.
"""

from __future__ import annotations

import datetime
import logging
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Sequence

import cf_xarray.units  # pylint: disable=unused-import  # noqa: F401  (teaches pint UDUNITS)
import numpy as np
import pandas as pd
import pint_xarray  # pylint: disable=unused-import  # noqa: F401  (registers the .pint accessor)
import xarray as xr
from pyproj import Proj, Transformer
from shapely.geometry import Polygon
from shapely.ops import transform

from pism_terra.download import download_earthaccess, download_file, save_netcdf
from pism_terra.log import setup_logging

logger = logging.getLogger(__name__)

xr.set_options(keep_attrs=True)

#: GSFC mascon solution, 0.5 degree, equivalent water thickness.
GRACE_GSFC_URL = (
    "https://earth.gsfc.nasa.gov/sites/default/files/geo/gsfc.glb_.200204_202410_rl06v2.0_obp-ice6gd_halfdegree.nc"
)

#: PO.DAAC collection holding the ice-sheet-wide GRACE Tellus time series.
#: Collections are retired as releases land; RL06.1_V3 no longer resolves.
GRACE_TELLUS_SHORT_NAME = "GREENLAND_MASS_TELLUS_MASCON_CRI_TIME_SERIES_RL06.3_V4"

#: Mankoff et al. (2021) mass balance, GEUS Dataverse.
MANKOFF_URL = "https://dataverse.geus.dk/api/access/datafile/:persistentId?persistentId=doi:10.22008/FK2/OHI23Z/MRSBQR"

#: Mankoff per-basin variables, source name to ours. ``_ROI`` are the
#: per-region series; the un-suffixed names are the ice-sheet totals.
MANKOFF_BASIN_VARS: dict[str, str] = {
    "D_ROI": "grounding_line_flux",
    "MB_ROI": "mass_balance",
    "SMB_ROI": "surface_mass_balance",
    "BMB_ROI": "basal_mass_balance",
    "D_ROI_err": "grounding_line_flux_uncertainty",
    "MB_ROI_err": "mass_balance_uncertainty",
    "SMB_ROI_err": "surface_mass_balance_uncertainty",
    "BMB_ROI_err": "basal_mass_balance_uncertainty",
}

#: The same quantities for the ice sheet as a whole, concatenated on as
#: basin ``"GIS"`` so one selection covers both.
MANKOFF_GIS_VARS: dict[str, str] = {
    "D": "grounding_line_flux",
    "MB": "mass_balance",
    "SMB": "surface_mass_balance",
    "BMB": "basal_mass_balance",
    "D_err": "grounding_line_flux_uncertainty",
    "MB_err": "mass_balance_uncertainty",
    "SMB_err": "surface_mass_balance_uncertainty",
    "BMB_err": "basal_mass_balance_uncertainty",
}

#: Mankoff variables whose ``units`` attribute is missing upstream. The rest
#: of the file states them UDUNITS-style ("Gt d-1"), which is why
#: ``cf_xarray.units`` is imported above: plain pint reads the "d-1" as a
#: subtraction and raises.
MANKOFF_MISSING_UNITS = ("MB_err", "BMB_err", "MB_ROI", "MB_ROI_err", "BMB_ROI_err")

#: Density of water, for turning equivalent water thickness into mass.
#: The magnitude is kept separate: pint refuses a scaling factor inside a
#: unit expression ("1000 kg m^-3" is a ValueError, not 1000 kg/m3).
WATER_DENSITY = 1000.0
WATER_DENSITY_UNITS = "kg m^-3"

#: Filenames of the three products, in the order they are prepared.
PRODUCTS = (
    "grace_gsfc_greenland_mass_balance.nc",
    "grace_greenland_mass_balance.nc",
    "mankoff_greenland_mass_balance.nc",
)

#: Greenland, generously: the GSFC file is global.
GREENLAND_BOUNDS = {"lon": slice(360 - 75, 360 - 10), "lat": slice(59, 84)}

_WGS84 = Proj(proj="latlong", datum="WGS84")
_EQUAL_AREA = Proj(proj="aea", lat_1=0, lat_2=90)
_TO_EQUAL_AREA = Transformer.from_crs(_WGS84.crs, _EQUAL_AREA.crs, always_xy=True).transform


def polygon_area(lat_0: float, lat_1: float, lon_0: float, lon_1: float) -> float:
    """
    Compute the area of one lat/lon cell on the ellipsoid.

    The cell is projected into an Albers equal-area projection first, so a
    cell near the pole is not counted as though it were at the equator —
    over Greenland that is most of the signal.

    Parameters
    ----------
    lat_0, lat_1 : float
        Southern and northern edge of the cell, degrees north.
    lon_0, lon_1 : float
        Western and eastern edge of the cell, degrees east.

    Returns
    -------
    float
        Area in square metres.
    """
    polygon = Polygon([(lon_0, lat_0), (lon_1, lat_0), (lon_1, lat_1), (lon_0, lat_1)])
    return transform(_TO_EQUAL_AREA, polygon).area


def quantify_data_only(obj):
    """
    Quantify the data, leaving the index coordinates alone.

    A plain ``.pint.quantify()`` also wraps any indexed coordinate carrying
    a ``units`` attribute, which pint-xarray >= 0.6 represents as a
    ``PintIndex``. Operations that rebuild coordinates — ``xr.broadcast``,
    ``xr.apply_ufunc`` — hand back a plain ``PandasIndex`` instead, and
    xarray then refuses to align the two::

        AlignmentError: cannot align objects on coordinate 'lat'
        because of conflicting indexes

    Nothing here converts degrees or datetimes, so the coordinates gain
    nothing from being quantified. Every operand needs the same treatment,
    not just the first: quantifying only one of them reintroduces the
    mismatch from the other side.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        Object whose data carries ``units`` attributes.

    Returns
    -------
    xr.Dataset or xr.DataArray
        The same object with quantified data and untouched coordinates.
    """
    return obj.pint.quantify(**{name: None for name in obj.indexes})


def cell_areas(ds: xr.Dataset) -> xr.DataArray:
    """
    Compute the area of every cell of a bounded lat/lon grid.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset carrying ``lat_bounds`` and ``lon_bounds``.

    Returns
    -------
    xr.DataArray
        Per-cell area on the ``(lat, lon)`` grid, with ``units`` set.
    """
    lat_bounds, lon_bounds = xr.broadcast(ds["lat_bounds"], ds["lon_bounds"])
    lat_bounds = lat_bounds.transpose("lat", "lon", "bounds")
    lon_bounds = lon_bounds.transpose("lat", "lon", "bounds")
    area = xr.apply_ufunc(
        polygon_area,
        lat_bounds.isel({"bounds": 0}),
        lat_bounds.isel({"bounds": 1}),
        lon_bounds.isel({"bounds": 0}),
        lon_bounds.isel({"bounds": 1}),
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    area.name = "area"
    area.attrs.update({"units": "m^2", "long_name": "area of grid cell"})
    return area


def decimal_year_to_datetime(decimal_year: float) -> datetime.datetime:
    """
    Convert a decimal year to a date, rounded to the nearest day.

    Parameters
    ----------
    decimal_year : float
        Year with a fractional part, e.g. ``2002.2877``.

    Returns
    -------
    datetime.datetime
        Midnight of the corresponding day.
    """
    year = int(decimal_year)
    start = datetime.datetime(year, 1, 1)
    days = (datetime.datetime(year + 1, 1, 1) - start).days
    date = start + datetime.timedelta(days=(decimal_year - year) * days)
    if date.hour >= 12:
        date = date + datetime.timedelta(days=1)
    return date.replace(hour=0, minute=0, second=0, microsecond=0)


def read_mass_time_series(path: Path | str) -> pd.DataFrame:
    """
    Read a PO.DAAC mass time series, whatever length its header is.

    The file is a comment header followed by three whitespace-separated
    columns. The header length is not stable across releases, so rather than
    skipping a fixed number of lines this keeps the lines that parse as three
    numbers and drops the rest.

    Parameters
    ----------
    path : Path or str
        The downloaded ``.txt`` time series.

    Returns
    -------
    pd.DataFrame
        Columns ``year`` (decimal), ``cumulative_mass_balance`` and
        ``mass_balance_uncertainty``.

    Raises
    ------
    ValueError
        If no data rows were found, which means the format has changed.
    """
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            rows.append([float(field) for field in fields])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"no three-column data rows found in {path}")
    return pd.DataFrame(rows, columns=["year", "cumulative_mass_balance", "mass_balance_uncertainty"])


def _write(ds: xr.Dataset, path: Path) -> Path:
    """
    Write a derived dataset, dropping encoding inherited from its source.

    A variable built with ``diff`` is one timestep shorter than the variable
    it came from but keeps its ``chunksizes``, and netCDF4 then refuses with
    "chunksize cannot exceed dimension size".

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to write.
    path : pathlib.Path
        Destination file.

    Returns
    -------
    pathlib.Path
        ``path``.
    """
    for name in ds.data_vars:
        ds[name].encoding.pop("chunksizes", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_netcdf(ds, path)
    logger.info("Wrote %s", path)
    return path


def prepare_grace_gsfc(cache_path: Path, url: str = GRACE_GSFC_URL, force_overwrite: bool = False) -> Path:
    """
    Turn the GSFC mascon solution into per-cell mass over Greenland.

    The published field is equivalent water thickness, which cannot be summed
    over a region as it stands; multiplying by the true cell area and the
    density of water gives a mass per cell that can.

    Parameters
    ----------
    cache_path : pathlib.Path
        Directory for the download and the derived product.
    url : str, optional
        Source of the global half-degree file.
    force_overwrite : bool, optional
        Rebuild even when the product is already there.

    Returns
    -------
    pathlib.Path
        The written product.
    """
    product = cache_path / "grace_gsfc_greenland_mass_balance.nc"
    if product.exists() and not force_overwrite:
        logger.info("Using existing %s", product)
        return product

    source = Path(download_file(url, cache_path / Path(url).name.split("?")[0], force_overwrite=force_overwrite))
    with xr.open_dataset(source) as raw:
        ds = raw.sel(GREENLAND_BOUNDS).load()

    # "binary" is not a unit pint knows, and the mask is dimensionless anyway.
    ds["land_mask"].attrs.update({"units": ""})
    coord_units = {name: ds[name].attrs["units"] for name in ds.indexes if "units" in ds[name].attrs}
    ds = quantify_data_only(ds)

    area = cell_areas(ds)
    water_density = xr.DataArray(WATER_DENSITY).pint.quantify(WATER_DENSITY_UNITS).pint.to("Gt m^-3")
    ds["cumulative_mass_balance"] = (
        ds["lwe_thickness"].where(ds["land_mask"]).pint.to("m") * quantify_data_only(area) * water_density
    )
    # Renamed before quantifying: the difference of ``ds.time`` is itself
    # named ``time``, and quantify() would stamp seconds onto the ``time``
    # *coordinate* as well, which no other operand shares.
    interval = (ds.time.diff(dim="time") / np.timedelta64(1, "s")).rename("interval").pint.quantify("s").pint.to("year")
    ds["mass_balance"] = ds["cumulative_mass_balance"].diff(dim="time") / interval
    ds["lwe_thickness_err"] = xr.zeros_like(ds["lwe_thickness"]) + 4
    ds["mass_balance_err"] = (
        xr.zeros_like(ds["mass_balance"])
        + xr.DataArray(4).pint.quantify("cm yr^-1").pint.to("m yr^-1") * quantify_data_only(area) * water_density
    )

    out = ds.pint.dequantify()
    for name, units in coord_units.items():
        out[name].attrs["units"] = units
    return _write(out, product)


def prepare_grace_tellus(
    cache_path: Path, short_name: str = GRACE_TELLUS_SHORT_NAME, force_overwrite: bool = False
) -> Path:
    """
    Build the ice-sheet-wide GRACE Tellus mass balance time series.

    Parameters
    ----------
    cache_path : pathlib.Path
        Directory for the download and the derived product.
    short_name : str, optional
        PO.DAAC collection to fetch.
    force_overwrite : bool, optional
        Rebuild even when the product is already there.

    Returns
    -------
    pathlib.Path
        The written product.
    """
    product = cache_path / "grace_greenland_mass_balance.nc"
    if product.exists() and not force_overwrite:
        logger.info("Using existing %s", product)
        return product

    results = download_earthaccess(result_dir=cache_path, short_name=short_name)
    df = read_mass_time_series(results[0])
    df["time"] = np.vectorize(decimal_year_to_datetime)(df["year"])

    ds = xr.Dataset.from_dataframe(df.set_index(df["time"]))
    ds["cumulative_mass_balance"].attrs.update({"units": "Gt"})
    ds["cumulative_mass_balance_uncertainty"] = np.sqrt((ds["mass_balance_uncertainty"] ** 2).cumsum(dim="time"))
    ds["cumulative_mass_balance_uncertainty"].attrs.update({"units": "Gt"})
    ds = quantify_data_only(ds)

    interval = (ds.time.diff(dim="time") / np.timedelta64(1, "s")).rename("interval").pint.quantify("s").pint.to("year")
    ds["mass_balance"] = ds["cumulative_mass_balance"].diff(dim="time") / interval
    return _write(ds.pint.dequantify(), product)


def prepare_mankoff(cache_path: Path, url: str = MANKOFF_URL, force_overwrite: bool = False) -> Path:
    """
    Build the Mankoff per-basin and ice-sheet mass balance.

    The published series are daily rates; this adds the running totals and
    flips the sign of the grounding-line flux so every variable is positive
    for mass gained, as PISM reports them.

    Parameters
    ----------
    cache_path : pathlib.Path
        Directory for the download and the derived product.
    url : str, optional
        GEUS Dataverse URL of the source NetCDF.
    force_overwrite : bool, optional
        Rebuild even when the product is already there.

    Returns
    -------
    pathlib.Path
        The written product.
    """
    product = cache_path / "mankoff_greenland_mass_balance.nc"
    if product.exists() and not force_overwrite:
        logger.info("Using existing %s", product)
        return product

    source = Path(
        download_file(url, cache_path / "mankoff_greenland_mass_balance_source.nc", force_overwrite=force_overwrite)
    )
    with xr.open_dataset(source) as raw:
        ds = raw.load()
    for name in MANKOFF_MISSING_UNITS:
        ds[name].attrs["units"] = "Gt day-1"
    ds = quantify_data_only(ds)

    gis = ds[list(MANKOFF_GIS_VARS)].rename_vars(MANKOFF_GIS_VARS)[list(MANKOFF_GIS_VARS.values())]
    gis = gis.expand_dims("basin")
    gis["basin"] = ["GIS"]

    basins = ds.rename_vars(MANKOFF_BASIN_VARS)[list(MANKOFF_BASIN_VARS.values())].rename({"region": "basin"})
    ds = xr.concat([basins, gis], dim="basin")
    ds["basin"] = ds["basin"].astype("<U3")

    # The record is unevenly spaced: yearly until 1986, daily after, so the
    # running totals have to weight each step by its own length.
    # Renamed before quantifying, as above.
    interval = (
        (ds.time.diff(dim="time", label="lower") / np.timedelta64(1, "s"))
        .rename("interval")
        .pint.quantify("s")
        .pint.to("day")
    )
    for name in MANKOFF_BASIN_VARS.values():
        ds[f"cumulative_{name}"] = (ds[name].pint.to("Gt day^-1") * interval).cumsum(dim="time")
        ds[f"cumulative_{name}"] = ds[f"cumulative_{name}"].pint.to("Gt")
        ds[name] = ds[name].pint.to("Gt year^-1")

    # The final step's cumulative uncertainty is NaN, having no interval.
    ds = ds.isel({"time": slice(0, -1)})
    ds["grounding_line_flux"] = ds["grounding_line_flux"] * xr.DataArray(-1).pint.quantify("1")
    return _write(ds.pint.dequantify(), product)


def prepare_observations(
    output_path: Path | str,
    cache_path: Path | str | None = None,
    force_overwrite: bool = False,
    skip_errors: bool = False,
) -> list[Path]:
    """
    Prepare all three observational products and place them beside a run.

    Parameters
    ----------
    output_path : Path or str
        Run directory; the products land in ``<output_path>/output/observations``.
    cache_path : Path or str or None, optional
        Directory for downloads and derived products. Defaults to
        ``<output_path>/observations_cache``; point it at a shared directory
        to prepare these once for many runs.
    force_overwrite : bool, optional
        Re-download and rebuild everything.
    skip_errors : bool, optional
        Log and carry on when one product cannot be built, rather than
        raising. Staging sets this: the GRACE Tellus product needs an
        Earthdata login, and a run's *inputs* must not fail to stage over
        data that is only wanted for the analysis afterwards.

    Returns
    -------
    list of pathlib.Path
        The products as they sit in the run's observations directory. Short
        of three when ``skip_errors`` swallowed a failure.
    """
    output_path = Path(output_path)
    cache_path = Path(cache_path) if cache_path is not None else output_path / "observations_cache"
    cache_path.mkdir(parents=True, exist_ok=True)

    builders = (
        ("GRACE GSFC", prepare_grace_gsfc),
        ("GRACE Tellus", prepare_grace_tellus),
        ("Mankoff", prepare_mankoff),
    )
    products = []
    for label, builder in builders:
        try:
            products.append(builder(cache_path, force_overwrite=force_overwrite))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if not skip_errors:
                raise
            logger.error("Could not prepare the %s observations: %s", label, exc)

    destination = output_path / "output" / "observations"
    destination.mkdir(parents=True, exist_ok=True)
    placed = []
    for product in products:
        target = destination / product.name
        # copy2 rather than a link: the cache may be a shared directory that
        # outlives, or is cleaned independently of, this run.
        shutil.copy2(product, target)
        logger.info("Placed %s", target)
        placed.append(target)
    return placed


def main(argv: Sequence[str] | None = None) -> int:
    """
    Console entry point.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments without the program name. ``None`` uses
        ``sys.argv``.

    Returns
    -------
    int
        Exit code, ``0`` on success.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Prepare observed Greenland mass balance for validating ISMIP7 runs."
    parser.add_argument(
        "--cache-path",
        help="Directory for downloads and derived products, re-used across runs. "
        "Defaults to <OUTPUT_PATH>/observations_cache.",
        default=None,
    )
    parser.add_argument(
        "--force-overwrite",
        help="Re-download and rebuild everything.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "OUTPUT_PATH", nargs=1, help="Run directory; products land in <OUTPUT_PATH>/output/observations."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_path = Path(args.OUTPUT_PATH[0])
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(output_path / "observations.log")

    placed = prepare_observations(output_path, cache_path=args.cache_path, force_overwrite=args.force_overwrite)
    logger.info("-" * 100)
    for path in placed:
        logger.info("%s", path)
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(main())
