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
Tests for the USGS stake-measurement GeoPackage.

A miniature release with both of the date notations the real one uses, a
site without coordinates and a glacier without a sub-seasonal table.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from pism_terra.glacier import usgs
from pism_terra.glacier import usgs_generate_geopackage as us
from pism_terra.glacier.usgs import load_sites


@pytest.fixture(name="release")
def fixture_release(tmp_path: Path) -> dict[str, Path]:
    """
    Write point-measurement CSVs for two glaciers plus their sites.

    ``Foo`` uses ``YYYY/MM/DD`` dates and has a sub-seasonal table; ``Bar``
    uses ``M/D/YYYY`` dates and a site (``Q``) the site list does not know.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    dict of str to Path
        ``"data"`` and ``"sites"`` directories.
    """
    data = tmp_path / "glacier_massBalance_data"
    foo_dir = data / "Foo"
    foo_dir.mkdir(parents=True)
    (foo_dir / "Input_Foo_Glaciological_Data.csv").write_text(
        "Year,site_name,spring_date,fall_date,elevation,bw_snow_depth,bw_density,bw,ba_snow_depth,ba_density,ba,"
        "winter_ablation,summer_accumulation_snow_depth,summer_accumulation_density,summer_accumulation\n"
        "2000,A,2000/05/01,2000/09/10,1200,3.0,400,1.2,nan,nan,-1.5,nan,nan,nan,0\n"
        "2001,A ,2001/05/03,nan,1200,nan,nan,1.0,nan,nan,nan,0.1,nan,nan,nan\n"
    )
    (foo_dir / "Input_Foo_SubSeasonal_Glaciological_Data.csv").write_text(
        "Year,site_name,Elevation,Date1,Surface1,Date2,Surface2,db\n2000,A,1200,2000/06/01,snow,2000/07/01,ice,-0.8\n"
    )
    bar_dir = data / "Bar"
    bar_dir.mkdir()
    (bar_dir / "Input_Bar_Glaciological_Data.csv").write_text(
        "Year,site_name,spring_date,fall_date,elevation,bw_snow_depth,bw_density,bw,ba_snow_depth,ba_density,ba,"
        "winter_ablation,summer_accumulation_snow_depth,summer_accumulation_density,summer_accumulation\n"
        "2016,Q,5/26/2016,8/20/2016,2181,nan,nan,1.94,nan,nan,1.3,nan,nan,nan,0\n"
    )
    sites = tmp_path / "Glacier_Mass_Balance_Sites"
    sites.mkdir()
    (sites / "Glacier_Mass_Balance_Data_Sites.csv").write_text(
        "Glacier,USGS Benchmark Glacier,Type,site_name,Easting,Northing,UTM_EPSG,latitude,longitude\n"
        "Foo,Yes,Glaciological,A,0,0,32606,60.2,-148.2\n"
        "Foo,Yes,Weather,Airport,0,0,32606,65.0,-140.0\n"
        "Bar,No,Glaciological,X,0,0,32606,63.5,-145.5\n"
    )
    return {"data": data, "sites": sites}


def test_parse_dates_accepts_both_notations():
    """
    ``YYYY/MM/DD`` and ``M/D/YYYY`` both parse; ``nan`` becomes ``NaT``; junk raises.
    """
    parsed = usgs.parse_dates(pd.Series(["2000/05/01", "5/26/2016", "nan", np.nan]))
    assert parsed.dtype == "datetime64[ns]"
    assert parsed.iloc[0] == pd.Timestamp("2000-05-01")
    assert parsed.iloc[1] == pd.Timestamp("2016-05-26")
    assert parsed.iloc[2:].isna().all()
    with pytest.raises(ValueError, match="Unparseable"):
        usgs.parse_dates(pd.Series(["May 1 2000"]))


def test_measurements_are_typed(release):
    """
    Type dates as datetimes and years as integers, and strip site names.

    Parameters
    ----------
    release : dict of str to Path
        The synthetic release directories.
    """
    df = usgs.load_measurements(release["data"], "Foo")
    assert df is not None
    assert df["glacier"].tolist() == ["Foo", "Foo"]
    assert df["site_name"].tolist() == ["A", "A"]
    assert df["Year"].dtype.kind == "i"
    assert df["spring_date"].tolist() == [pd.Timestamp("2000-05-01"), pd.Timestamp("2001-05-03")]
    assert pd.isna(df["fall_date"].iloc[1])
    assert usgs.load_measurements(release["data"], "Bar", "subseasonal") is None


def test_layers_join_site_coordinates(release):
    """
    Join each measurement to its site's point; unknown sites keep a null geometry.

    Parameters
    ----------
    release : dict of str to Path
        The synthetic release directories.
    """
    layers = us.build_stake_layers(release["data"], load_sites(release["sites"]))
    assert set(layers) == {"sites", "stakes", "subseasonal"}
    stakes = layers["stakes"]
    assert len(stakes) == 3
    foo_rows = stakes[stakes["glacier"] == "Foo"]
    assert (foo_rows.geometry.x == -148.2).all()
    assert stakes[stakes["glacier"] == "Bar"].geometry.isna().all()
    assert stakes.crs.to_epsg() == 4326
    assert len(layers["subseasonal"]) == 1
    assert layers["sites"]["Type"].eq("Glaciological").all()


def test_cli_writes_datetime_fields(release, tmp_path, monkeypatch):
    """
    The GeoPackage round-trips the dates as datetime fields and m w.e. values untouched.

    Parameters
    ----------
    release : dict of str to Path
        The synthetic release directories.
    tmp_path : Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to skip the download.
    """
    monkeypatch.setattr(us, "download_usgs_benchmark", lambda data_dir, force_overwrite=False: release)
    output = tmp_path / "out" / "stakes.gpkg"
    assert us.cli([str(output), "--data-dir", str(tmp_path)]) == 0
    assert output.exists()

    stakes = gpd.read_file(output, layer="stakes")
    assert stakes["spring_date"].dtype.kind == "M"
    assert (
        pd.Timestamp(stakes.loc[stakes["glacier"] == "Bar", "spring_date"].iloc[0]).date()
        == pd.Timestamp("2016-05-26").date()
    )
    assert stakes.loc[stakes["glacier"] == "Bar", "bw"].iloc[0] == 1.94
    assert stakes["Year"].tolist() == [2016, 2000, 2001]
    assert gpd.list_layers(output)["name"].tolist() == ["sites", "stakes", "subseasonal"]

    # Running again replaces the file rather than appending to its layers.
    assert us.cli([str(output), "--data-dir", str(tmp_path)]) == 0
    assert len(gpd.read_file(output, layer="stakes")) == 3
