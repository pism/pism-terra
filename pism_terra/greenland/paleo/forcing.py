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

# pylint: disable=too-many-positional-arguments

"""
Forcing for Greenland paleo simulations.

Two kinds of forcing drive a glacial cycle: a present-day base climate, taken
as a multi-year monthly mean of the ISMIP7 OCX product, and scalar time series
of the departure from it, taken from the SeaRISE Greenland data set (GRIP
temperature anomaly and SPECMAP sea level).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr
from cdo import Cdo

from pism_terra.glacier.climate import stamp_monthly_climatology_axis
from pism_terra.ismip7.greenland.forcing import _forcing_tasks, prepare_ismip7_forcing
from pism_terra.workflow import check_xr_lazy

logger = logging.getLogger(__name__)

#: Time axis of the scalar series, the one PISM's ``std-greenland`` example writes.
SERIES_TIME_UNITS = "common_years since 1-1-1"
SERIES_CALENDAR = "365_day"

#: Per forcing, the variable that is zero exactly where the source has no data
#: (the open ocean, for the regional climate model fields): precipitation over
#: land is never exactly zero in a monthly mean. The ocean fields are
#: extrapolated by the ISMIP7 machinery already.
EXTRAPOLATE_FROM = {"climate": "precipitation"}

#: SeaRISE variables the series are read from: (time coordinate, values).
SEARISE_TEMPERATURE = ("oisotopestimes", "temp_time_series")
SEARISE_SEA_LEVEL = ("sealeveltimes", "sealevel_time_series")


def climatology_filename(forcing: str, gcm: str, start_year: int, end_year: int) -> str:
    """
    Name the multi-year monthly mean of one forcing.

    Parameters
    ----------
    forcing : str
        ``"climate"`` or ``"ocean"``.
    gcm : str
        Product the mean is taken of, e.g. ``"OCX"``.
    start_year : int
        First year of the mean.
    end_year : int
        Last year of the mean, inclusive.

    Returns
    -------
    str
        ``paleo_greenland_{forcing}_{gcm}_YMM_{start}_{end}.nc``.
    """
    return f"paleo_greenland_{forcing}_{gcm}_YMM_{start_year}_{end_year}.nc"


def monthly_climatology(monthly_file: Path | str, output_file: Path | str, extrapolate_from: str | None = None) -> Path:
    """
    Reduce a monthly forcing file to the 12-step climatology PISM cycles.

    Parameters
    ----------
    monthly_file : Path or str
        Monthly forcing spanning whole years, as written by
        :func:`pism_terra.ismip7.greenland.forcing.prepare_ismip7_forcing`.
    output_file : Path or str
        Climatology to write.
    extrapolate_from : str or None, optional
        Variable that is zero exactly where the source had no data. When
        given, every field is replaced there by its nearest valid neighbour:
        the ISMIP7 machinery fills cells without data with a constant, which a
        present-day run never sees but a glacial ice sheet grows into.

    Returns
    -------
    pathlib.Path
        ``output_file``.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cdo = Cdo()
    mean = f"-ymonmean {Path(monthly_file).resolve()}"
    if extrapolate_from is not None:
        valid = f"-gtc,0 -timmax -selname,{extrapolate_from} {mean}"
        mean = f"-setmisstonn -ifthen {valid} {mean}"
    ds = cdo.copy(input=mean, options="-f nc4", returnXDataset=True).load()
    # ``basins`` (ocean files) has no time axis and passes through ymonmean.
    ds = stamp_monthly_climatology_axis(ds)
    encoding: dict[str, dict[str, Any]] = {
        name: {"zlib": True, "complevel": 2, "_FillValue": None} for name in ds.data_vars if ds[name].ndim >= 2
    }
    encoding["time"] = {"_FillValue": None}
    encoding["time_bounds"] = {"_FillValue": None}
    ds.to_netcdf(output_file, encoding=encoding)
    ds.close()
    return output_file


def prepare_ocx_climatology(
    config: dict,
    cache_path: Path | str,
    output_path: Path | str,
    staging_path: Path | str,
    data_path: Path | str | None = None,
    forcings: str | Sequence[str] | None = None,
    force_overwrite: bool = False,
) -> dict[str, Path]:
    """
    Build the monthly climatologies of the configured base-climate period.

    The setup TOML names the period as the ``historical`` span of each
    ``[gcms]`` entry; the monthly files of that span are built by the ISMIP7
    machinery into ``staging_path`` and reduced to their multi-year monthly
    mean in ``output_path``.

    Parameters
    ----------
    config : dict
        Parsed setup TOML (``setup_greenland_paleo.toml``).
    cache_path : Path or str
        Download cache of the per-year source files.
    output_path : Path or str
        Directory the climatologies are written to.
    staging_path : Path or str
        Scratch directory for the monthly files the means are taken of.
    data_path : Path or str or None, optional
        Local mirror of the forcing tree, used instead of downloading.
    forcings : str or sequence of str or None, optional
        Only build these forcings (``"climate"``, ``"ocean"``).
    force_overwrite : bool, optional
        Rebuild climatologies that already exist.

    Returns
    -------
    dict of str to pathlib.Path
        Climatology per forcing, keyed ``"climate"`` / ``"ocean"`` (the last
        GCM wins when the setup lists several).
    """
    output_path = Path(output_path)
    staging_path = Path(staging_path)
    wanted = None
    if forcings is not None:
        wanted = {f.strip() for f in (forcings.split(",") if isinstance(forcings, str) else forcings)}

    targets: dict[tuple[str, str, str], Path] = {}
    for _, gcm, forcing, _, pathway, start_year, end_year, *_ in _forcing_tasks(config):
        if wanted is not None and forcing not in wanted:
            continue
        targets[(gcm, pathway, forcing)] = output_path / climatology_filename(forcing, gcm, start_year, end_year)

    todo = {key: path for key, path in targets.items() if force_overwrite or not check_xr_lazy(path, verbose=False)}
    if todo:
        monthly_files = prepare_ismip7_forcing(
            cache_path,
            staging_path,
            config,
            data_path=data_path,
            staging_path=staging_path / "tmp",
            gcms=sorted({gcm for gcm, _, _ in todo}),
            pathways=sorted({pathway for _, pathway, _ in todo}),
            forcings=sorted({forcing for _, _, forcing in todo}),
        )
        for (gcm, pathway, forcing), target in todo.items():
            stem = f"ismip7_greenland_{forcing}_{pathway}_{gcm}_"
            matches = [Path(f) for f in monthly_files if Path(f).name.startswith(stem)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one monthly file starting with {stem!r}, got {matches}")
            logger.info("Monthly climatology of %s -> %s", matches[0].name, target.name)
            monthly_climatology(matches[0], target, extrapolate_from=EXTRAPOLATE_FROM.get(forcing))

    return {forcing: path for (_, _, forcing), path in targets.items()}


def _series_dataset(
    years_before_present: np.ndarray, values: np.ndarray, name: str, attrs: dict[str, str]
) -> xr.Dataset:
    """
    Put a series given in years before present on PISM's scalar forcing axis.

    Time runs forward and is negative before present; each record closes the
    interval that starts at the previous one, the convention of the
    ``std-greenland`` example.

    Parameters
    ----------
    years_before_present : numpy.ndarray
        Positive ages, in any order.
    values : numpy.ndarray
        Series values at those ages.
    name : str
        Name of the forcing variable (``"delta_T"`` or ``"delta_SL"``).
    attrs : dict of str to str
        Attributes of the forcing variable.

    Returns
    -------
    xarray.Dataset
        The series with ``time`` and ``time_bnds``.
    """
    # ``+ 0.0`` turns the -0.0 of "present" into 0.0.
    time = -np.asarray(years_before_present, dtype="float64") + 0.0
    order = np.argsort(time)
    time = time[order]
    values = np.asarray(values, dtype="float64")[order]

    bounds = np.empty((time.size, 2), dtype="float64")
    bounds[:, 1] = time
    bounds[1:, 0] = time[:-1]
    bounds[0, 0] = 2 * time[0] - time[1]

    ds = xr.Dataset(
        {name: (("time",), values, attrs), "time_bnds": (("time", "nv"), bounds)},
        coords={
            "time": (
                ("time",),
                time,
                {
                    "units": SERIES_TIME_UNITS,
                    "calendar": SERIES_CALENDAR,
                    "bounds": "time_bnds",
                    "axis": "T",
                    "long_name": "time",
                },
            )
        },
    )
    return ds


def _write_series(ds: xr.Dataset, path: Path) -> Path:
    """
    Write a scalar series without fill values or decoded times.

    Parameters
    ----------
    ds : xarray.Dataset
        Series from :func:`_series_dataset`.
    path : pathlib.Path
        File to write.

    Returns
    -------
    pathlib.Path
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding={name: {"_FillValue": None} for name in ds.variables})
    return path


def prepare_searise_series(
    searise_file: Path | str,
    output_path: Path | str,
    ocean_scale: float = 0.25,
    start: float | None = None,
) -> dict[str, Path]:
    """
    Write the SeaRISE temperature and sea-level series as PISM scalar forcing.

    A port of ``examples/std-greenland/preprocess.sh`` in PISM, plus a scaled
    copy of the temperature series for the ocean.

    Parameters
    ----------
    searise_file : Path or str
        ``Greenland_5km_v1.1.nc`` of the SeaRISE project.
    output_path : Path or str
        Directory the series are written to.
    ocean_scale : float, optional
        Fraction of the air-temperature anomaly applied to the ocean.
    start : float or None, optional
        Drop records before this year (negative for years before present).

    Returns
    -------
    dict of str to pathlib.Path
        ``"delta_T_file"`` (``pism_dT.nc``), ``"delta_SL_file"``
        (``pism_dSL.nc``) and ``"ocean_delta_T_file"`` (``pism_ocean_dT.nc``).
    """
    output_path = Path(output_path)
    source = Path(searise_file).name

    with xr.open_dataset(searise_file, decode_times=False) as searise:
        t_time, t_name = SEARISE_TEMPERATURE
        sl_time, sl_name = SEARISE_SEA_LEVEL
        delta_t = _series_dataset(
            searise[t_time].values,
            searise[t_name].values,
            "delta_T",
            {"units": "kelvin", "long_name": "air temperature anomaly (GRIP ice core)", "source": source},
        )
        delta_sl = _series_dataset(
            searise[sl_time].values,
            searise[sl_name].values,
            "delta_SL",
            {
                "units": "meters",
                "long_name": "sea level anomaly (SPECMAP)",
                "standard_name": "global_average_sea_level_change",
                "source": source,
            },
        )

    if start is not None:
        delta_t = delta_t.sel(time=slice(start, None))
        delta_sl = delta_sl.sel(time=slice(start, None))

    ocean_delta_t = delta_t.copy(deep=True)
    ocean_delta_t["delta_T"] = ocean_delta_t["delta_T"] * ocean_scale
    ocean_delta_t["delta_T"].attrs = {
        "units": "kelvin",
        "long_name": f"ocean temperature anomaly ({ocean_scale} x GRIP air temperature anomaly)",
        "source": source,
    }

    return {
        "delta_T_file": _write_series(delta_t, output_path / "pism_dT.nc"),
        "delta_SL_file": _write_series(delta_sl, output_path / "pism_dSL.nc"),
        "ocean_delta_T_file": _write_series(ocean_delta_t, output_path / "pism_ocean_dT.nc"),
    }
