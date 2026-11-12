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
Shared fixtures for the USGS benchmark-glacier tests.

Everything is synthetic: a two-glacier release written to ``tmp_path``, two
square RGI outlines, and a scalar file in the layout ``pism-postprocess-scalar``
writes. No network access.
"""

from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

GLACIER_A = "RGI2000-v7.0-G-01-00001"
GLACIER_B = "RGI2000-v7.0-G-01-00002"


@pytest.fixture(name="usgs_release")
def fixture_usgs_release(tmp_path: Path) -> dict[str, Path]:
    """
    Write a miniature copy of the ScienceBase release.

    ``Foo`` has glacier-wide balances (with a ``Ba_unc`` column) for three
    years but an area-altitude distribution for only the first two; ``Bar``
    has point data only, like Kennicott and Kahiltna in the real release.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    dict of str to Path
        ``"data"`` and ``"sites"`` directories, as
        :func:`pism_terra.download.download_usgs_benchmark` returns them.
    """
    data = tmp_path / "glacier_massBalance_data"
    foo_dir = data / "Foo"
    foo_dir.mkdir(parents=True)
    (foo_dir / "Output_Foo_Glacier_Wide_solutions_calibrated.csv").write_text(
        "Year,Bw,Bs,Ba,ELA,Bw_Date,Ba_Date,Calibration,Ba_unc\n"
        "2000,2.0,-3.0,-1.0,1500,2000/05/01,2000/09/01,0.1,0.2\n"
        "2001,nan,nan,-2.0,nan,nan,nan,0.1,0.2\n"
        "2002,1.0,-1.5,-0.5,1400,2002/05/01,2002/09/01,0.1,0.2\n"
    )
    (foo_dir / "Input_Foo_Area_Altitude_Distribution.csv").write_text(
        "Year,1250,1350\n2000,4.0,6.0\n2001,3.0,5.0\n",
    )
    bar_dir = data / "Bar"
    bar_dir.mkdir()
    (bar_dir / "Input_Bar_Glaciological_Data.csv").write_text("Year,site_name,ba\n2000,X,-1\n")

    sites = tmp_path / "Glacier_Mass_Balance_Sites"
    sites.mkdir()
    (sites / "Glacier_Mass_Balance_Data_Sites.csv").write_text(
        "Glacier,USGS Benchmark Glacier,Type,site_name,Easting,Northing,UTM_EPSG,latitude,longitude\n"
        "Foo,Yes,Glaciological,A,0,0,32606,60.2,-148.2\n"
        "Foo,Yes,Glaciological,B,0,0,32606,60.3,-148.3\n"
        "Foo,Yes,Glaciological,C,0,0,32606,61.2,-147.2\n"
        "Foo,Yes,Weather,Airport,0,0,32606,65.0,-140.0\n"
        "Bar,No,Glaciological,X,0,0,32606,62.01,-147.5\n"
        "Baz,No,Glaciological,Y,0,0,32606,63.5,-145.5\n"
    )
    return {"data": data, "sites": sites}


@pytest.fixture(name="usgs_rgi")
def fixture_usgs_rgi() -> gpd.GeoDataFrame:
    """
    Two square glacier outlines.

    Returns
    -------
    geopandas.GeoDataFrame
        Outline A spans 60-61 N / 149-148 W, outline B 61-62 N / 148-147 W.
    """
    return gpd.GeoDataFrame(
        {
            "rgi_id": [GLACIER_A, GLACIER_B],
            "glac_name": ["Foo Glacier", None],
            "area_km2": [10.0, 20.0],
        },
        geometry=[box(-149, 60, -148, 61), box(-148, 61, -147, 62)],
        crs="EPSG:4326",
    )


def write_scalar_file(
    path: Path,
    labels=(GLACIER_A, GLACIER_B),
    values=(1.0, 2.0),
    n_time: int = 3,
    *,
    sftgif=None,
    spacing: float = 500.0,
) -> Path:
    """
    Write a per-glacier scalar file as ``pism-postprocess-scalar`` does.

    Parameters
    ----------
    path : Path
        File to write.
    labels : tuple of str
        Region labels stored in ``glacier_id_name``.
    values : tuple of float
        Constant ``tendency_of_ice_mass`` per region, in Gt/yr.
    n_time : int, default 3
        Number of annual time steps, stamped at mid-year from 2000 on.
    sftgif : tuple of float or None, optional
        Ice-covered cell count per region; when given, ``sftgif`` and a
        ``pism_config`` variable carrying ``grid.dx``/``grid.dy`` are written.
    spacing : float, default 500.0
        Grid spacing in metres recorded in ``pism_config``.

    Returns
    -------
    Path
        The file written.
    """
    data = np.tile(np.array(values, dtype="float64"), (n_time, 1))
    ds = xr.Dataset(
        {"tendency_of_ice_mass": (("time", "glacier_id"), data, {"units": "Gt year^-1"})},
        coords={
            "time": ("time", pd.to_datetime([f"{2000 + i}-07-02" for i in range(n_time)])),
            "glacier_id": np.arange(len(labels), dtype="int32"),
            "glacier_id_name": ("glacier_id", list(labels)),
        },
    )
    if sftgif is not None:
        cells = np.tile(np.array(sftgif, dtype="float64"), (n_time, 1))
        ds["sftgif"] = (("time", "glacier_id"), cells, {"units": "1"})
        ds["pism_config"] = ((), np.int32(0), {"grid.dx": spacing, "grid.dy": spacing})
    ds.to_netcdf(path)
    return path


@pytest.fixture(name="scalar_file")
def fixture_scalar_file() -> Callable[..., Path]:
    """
    Hand tests the scalar-file writer.

    Returns
    -------
    callable
        :func:`write_scalar_file`.
    """
    return write_scalar_file
