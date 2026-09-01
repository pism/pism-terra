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
Tests for the USGS stake comparison against spatial output.

A tiny UTM grid with a piecewise-constant seasonal balance — accumulation
from October to April, runoff from May to September — so every integral
has a closed form, one column the model never had ice in, and two stakes
placed on cell centres from the grid itself. No network access.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS, Transformer

from pism_terra.glacier import usgs
from pism_terra.glacier import usgs_benchmark_stakes as ubs

GLACIER_A = "RGI2000-v7.0-G-01-00001"
EPSG = 32606
DX = 100.0
NX, NY = 8, 6
ACCUMULATION = 1000.0  # kg m^-2 yr^-1, October to April
RUNOFF = 3000.0  # kg m^-2 yr^-1, May to September
ICE_FREE_COLUMN = 0
STAKE_CELL = (3, 2)  # (i, j) of the on-ice stake
BARE_CELL = (ICE_FREE_COLUMN, 4)


def grid() -> tuple[np.ndarray, np.ndarray]:
    """
    Cell centres of the synthetic grid, placed inside outline A of the ``usgs_rgi`` fixture.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x, y)`` in EPSG:32606 metres.
    """
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)
    x0, y0 = to_utm.transform(-148.5, 60.5)
    x0, y0 = round(x0 / DX) * DX, round(y0 / DX) * DX
    return x0 + DX * np.arange(NX), y0 + DX * np.arange(NY)


def lonlat(i: int, j: int) -> tuple[float, float]:
    """
    Geographic position of a cell centre.

    Parameters
    ----------
    i, j : int
        Column and row.

    Returns
    -------
    tuple of float
        ``(lon, lat)``.
    """
    x, y = grid()
    to_geo = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)
    return to_geo.transform(x[i], y[j])


def spatial_file(path: Path, *, years: int = 2, start: int = 2000, monthly: bool = True) -> Path:
    """
    Write a spatial file in PISM's layout.

    Parameters
    ----------
    path : Path
        File to write.
    years : int, default 2
        Number of years.
    start : int, default 2000
        First year.
    monthly : bool, default True
        Monthly means; False writes annual means, which carry no seasons.

    Returns
    -------
    Path
        The file written.
    """
    x, y = grid()
    freq = "MS" if monthly else "YS"
    starts = pd.date_range(f"{start}-01-01", periods=12 * years if monthly else years, freq=freq)
    ends = starts.shift(1, freq=freq)
    mid = starts + (ends - starts) / 2
    epoch = pd.Timestamp("1980-01-01")
    seconds = ((mid - epoch) / pd.Timedelta(seconds=1)).to_numpy()
    bounds = np.stack([(starts - epoch) / pd.Timedelta(seconds=1), (ends - epoch) / pd.Timedelta(seconds=1)], axis=1)

    months = np.array([stamp.month for stamp in starts])
    winter = (months >= 10) | (months <= 4)
    acc = np.where(winter, ACCUMULATION, 0.0)[:, None, None] * np.ones((1, NY, NX))
    runoff = np.where(winter, 0.0, RUNOFF)[:, None, None] * np.ones((1, NY, NX))
    if not monthly:
        acc[:] = ACCUMULATION * 7 / 12
        runoff[:] = RUNOFF * 5 / 12
    thk = np.full((len(mid), NY, NX), 100.0)
    thk[:, :, ICE_FREE_COLUMN] = 0.0
    usurf = np.broadcast_to(1500.0 + 10.0 * np.arange(NY)[:, None], (len(mid), NY, NX)).copy()

    ds = xr.Dataset(
        {
            "surface_accumulation_flux": (("time", "y", "x"), acc, {"units": "kg m^-2 year^-1"}),
            "surface_runoff_flux": (("time", "y", "x"), runoff, {"units": "kg m^-2 year^-1"}),
            "thk": (("time", "y", "x"), thk, {"units": "m"}),
            "usurf": (("time", "y", "x"), usurf, {"units": "m"}),
            "time_bounds": (("time", "nv"), bounds),
            "mapping": (
                (),
                np.int32(0),
                {"crs_wkt": CRS.from_epsg(EPSG).to_wkt(), "grid_mapping_name": "transverse_mercator"},
            ),
            "pism_config": ((), np.int32(0), {"grid.dx": DX, "grid.dy": DX}),
        },
        coords={
            "time": (
                "time",
                seconds,
                {"units": "seconds since 1980-01-01", "calendar": "standard", "bounds": "time_bounds"},
            ),
            "x": ("x", x, {"units": "m", "standard_name": "projection_x_coordinate"}),
            "y": ("y", y, {"units": "m", "standard_name": "projection_y_coordinate"}),
        },
    )
    for var in ("surface_accumulation_flux", "surface_runoff_flux", "thk", "usurf"):
        ds[var].attrs["grid_mapping"] = "mapping"
    ds["time_bounds"].attrs["units"] = "seconds since 1980-01-01"
    ds["time_bounds"].attrs["calendar"] = "standard"
    ds.to_netcdf(path)
    return path


@pytest.fixture(name="stake_release")
def fixture_stake_release(tmp_path: Path) -> dict[str, Path]:
    """
    A release with one glacier, ``Foo``, and two stakes: ``A`` on ice, ``Z`` on the bare column.

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
        "Year,site_name,spring_date,fall_date,elevation,bw,ba\n"
        "2000,A,2000/05/01,2000/09/15,1530,1.0,-1.0\n"
        "2001,A,2001/05/01,2001/09/15,1530,0.5,-0.7\n"
        "2000,Z,2000/05/01,2000/09/15,1540,0.8,-1.2\n"
        "2001,Z,2001/05/01,2001/09/15,1540,0.9,-0.9\n"
    )
    (foo_dir / "Input_Foo_SubSeasonal_Glaciological_Data.csv").write_text(
        "Year,site_name,Elevation,Date1,Surface1,Date2,Surface2,db\n2001,A,1530,2001/06/01,snow,2001/07/01,ice,-0.3\n"
    )
    sites = tmp_path / "Glacier_Mass_Balance_Sites"
    sites.mkdir()
    lon_a, lat_a = lonlat(*STAKE_CELL)
    lon_z, lat_z = lonlat(*BARE_CELL)
    (sites / "Glacier_Mass_Balance_Data_Sites.csv").write_text(
        "Glacier,USGS Benchmark Glacier,Type,site_name,Easting,Northing,UTM_EPSG,latitude,longitude\n"
        f"Foo,Yes,Glaciological,A,0,0,{EPSG},{lat_a:.6f},{lon_a:.6f}\n"
        f"Foo,Yes,Glaciological,Z,0,0,{EPSG},{lat_z:.6f},{lon_z:.6f}\n"
        "Foo,Yes,Weather,Airport,0,0,32606,65.0,-140.0\n"
    )
    return {"data": data, "sites": sites}


def expected_balances() -> dict[str, float]:
    """
    Closed-form balances at stake A in 2001 (fall 2000-09-15 to fall 2001-09-15).

    Returns
    -------
    dict
        ``bw``, ``bs``, ``ba`` and ``db`` in m w.e.
    """
    winter_days = 31 + 30 + 31 + 31 + 28 + 31 + 30  # October to April
    bw = (ACCUMULATION * winter_days - RUNOFF * 16) / 365 / 1000  # 15-30 September is runoff
    bs = -RUNOFF * (31 + 30 + 31 + 31 + 14) / 365 / 1000  # 1 May to 15 September
    return {"bw": bw, "bs": bs, "ba": bw + bs, "db": -RUNOFF * 30 / 365 / 1000}


@pytest.mark.parametrize("method", ["nearest", "linear"])
def test_sampling_and_integration(tmp_path, stake_release, method):
    """
    The stake's cell is integrated over exactly the measured interval; the bare column gives NaN.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    stake_release : dict of str to Path
        The synthetic release.
    method : str
        Sampling method.
    """
    file = spatial_file(tmp_path / "spatial_g100m_X_id_0_2000-01-01_2002-01-01.nc")
    sites = usgs.load_sites(stake_release["sites"])
    stakes = usgs.load_measurements(stake_release["data"], "Foo")
    subseasonal = usgs.load_measurements(stake_release["data"], "Foo", "subseasonal")
    points = ubs.stake_points(sites, [stakes, subseasonal])
    assert points["site"].tolist() == ["Foo:A", "Foo:Z"]

    with usgs.open_pism(file) as ds:
        sampled = ubs.sample_points(ds, points, method=method)
    assert sampled is not None
    assert sampled.sizes == {"site": 2, "time": 24}
    assert sampled["smb"].attrs["units"] == "m year^-1"
    assert sampled["smb"].sel(site="Foo:Z").isnull().all()
    np.testing.assert_allclose(sampled["smb"].sel(site="Foo:A").values[:4], ACCUMULATION / 1000)
    np.testing.assert_allclose(sampled["usurf"].sel(site="Foo:A").values, 1500.0 + 10.0 * STAKE_CELL[1], atol=0.1)
    np.testing.assert_allclose(sampled["lon"].sel(site="Foo:A"), lonlat(*STAKE_CELL)[0])

    table = ubs.stake_balances(sampled, stakes, subseasonal, run="test")
    at_a = table[(table["site"] == "A") & (table["year"] == 2001)].set_index("variable")
    expected = expected_balances()
    for var, value in expected.items():
        assert at_a.loc[var, "model"] == pytest.approx(value, abs=1e-9), var
    assert at_a.loc["bs", "obs"] == pytest.approx(-0.7 - 0.5)
    assert at_a.loc["ba", "usurf"] == pytest.approx(1520.0, abs=0.1)
    # 2000 has no previous fall visit, so no winter or annual model value; summer still integrates
    at_a_2000 = table[(table["site"] == "A") & (table["year"] == 2000)].set_index("variable")
    assert np.isnan(at_a_2000.loc["bw", "model"]) and np.isnan(at_a_2000.loc["ba", "model"])
    assert at_a_2000.loc["bs", "model"] == pytest.approx(expected["bs"])
    assert table[table["site"] == "Z"]["model"].isna().all()
    assert (table[table["site"] == "Z"]["obs"].notna()).all()


def test_previous_fall_falls_back_to_the_glacier_wide_date(tmp_path, stake_release):
    """
    Without a previous fall visit, the previous balance year's ``Ba_Date`` opens the interval.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    stake_release : dict of str to Path
        The synthetic release.
    """
    file = spatial_file(tmp_path / "spatial.nc")
    sites = usgs.load_sites(stake_release["sites"])
    stakes = usgs.load_measurements(stake_release["data"], "Foo")
    assert stakes is not None
    stakes = stakes[stakes["Year"] == 2001]
    wide = xr.Dataset({"Ba_Date": ("time", pd.to_datetime(["2000-09-15"]).to_numpy())}, coords={"time": [2000]})
    with usgs.open_pism(file) as ds:
        sampled = ubs.sample_points(ds, ubs.stake_points(sites, [stakes]))
    table = ubs.stake_balances(sampled, stakes, glacier_wide=wide).set_index(["site", "variable"])
    assert table.loc[("A", "bw"), "model"] == pytest.approx(expected_balances()["bw"])
    assert ubs.stake_balances(sampled, stakes).set_index(["site", "variable"])["model"].isna().loc[("A", "bw")]


def test_skill_per_site_and_pooled():
    """
    Each site is scored on its own and all sites together; the ensemble median is scored.
    """
    rows = []
    for run, shift in (("a", 0.0), ("b", 1.0)):
        for site, obs in (("A", [1.0, 2.0, 3.0]), ("B", [0.0, -1.0, -2.0])):
            for year, value in zip((2000, 2001, 2002), obs):
                rows.append(
                    {
                        "glacier": "Foo",
                        "site": site,
                        "year": year,
                        "variable": "ba",
                        "start": pd.Timestamp(f"{year - 1}-09-15"),
                        "end": pd.Timestamp(f"{year}-09-15"),
                        "obs": value,
                        "model": value + shift,
                        "elevation": 1500.0,
                        "usurf": 1520.0,
                        "visit": -1,
                        "run": run,
                    }
                )
    ice_free = pd.Series({("Foo", "A"): 0, ("Foo", "B"): 5})
    skill = ubs.stake_skill(pd.DataFrame(rows), ice_free).set_index("site")
    assert skill.loc["A", "n"] == 3 and skill.loc["A", "bias"] == pytest.approx(0.5)
    assert skill.loc["A", "r"] == pytest.approx(1.0)
    assert skill.loc["A", "elevation_offset"] == pytest.approx(20.0)
    assert skill.loc["B", "n_ice_free_months"] == 5
    assert skill.loc[ubs.POOLED, "n"] == 6 and skill.loc[ubs.POOLED, "mae"] == pytest.approx(0.5)
    assert ubs.stake_skill(pd.DataFrame(columns=ubs.TABLE_COLUMNS)).empty


def test_cli_end_to_end(tmp_path, stake_release, usgs_rgi, monkeypatch):
    """
    Figures, NetCDF and tables land under the RGI ID; an annual-only file is ignored.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    stake_release : dict of str to Path
        The synthetic release.
    usgs_rgi : geopandas.GeoDataFrame
        The two synthetic outlines.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to skip the download.
    """
    monkeypatch.setattr(ubs, "download_usgs_benchmark", lambda data_dir, force_overwrite=False: stake_release)
    rgi_file = tmp_path / "rgi.gpkg"
    usgs_rgi.to_file(rgi_file, driver="GPKG")
    runs = tmp_path / "runs"
    spatial = runs / "RGI2000-v7.0-C-01-00001" / "output" / "spatial"
    spatial.mkdir(parents=True)
    spatial_file(spatial / "spatial_g100m_RGI2000-v7.0-C-01-00001_id_0_2000-01-01_2002-01-01.nc")
    spatial_file(spatial / "spatial_g100m_RGI2000-v7.0-C-01-00001_id_0_uq_1_2000-01-01_2002-01-01.nc")
    annual = runs / "annual" / "output" / "spatial"
    annual.mkdir(parents=True)
    spatial_file(annual / "spatial_g100m_RGI2000-v7.0-C-01-00001_id_0_2000-01-01_2002-01-01.nc", monthly=False)
    processed = runs / "RGI2000-v7.0-C-01-00001" / "output" / "processed_spatial"
    processed.mkdir(parents=True)
    spatial_file(processed / "spatial_g100m_RGI2000-v7.0-C-01-00001_id_0_2000-01-01_2002-01-01_TM.nc")
    assert len(ubs.find_spatial_files(runs)) == 3
    output_dir = tmp_path / "out"

    argv = [
        str(runs),
        "--data-dir",
        str(tmp_path),
        "--output-dir",
        str(output_dir),
        "--rgi-glacier-file",
        str(rgi_file),
    ]
    assert ubs.cli(argv) == 0

    matches = pd.read_csv(output_dir / "usgs_benchmark_stakes_rgi_match.csv").set_index("glacier")
    assert matches.loc["Foo", "rgi_id"] == GLACIER_A
    assert matches.loc["Foo", "n_stakes"] == 2 and matches.loc["Foo", "n_sampled"] == 2
    assert matches.loc["Foo", "n_runs"] == 2  # the annual file is skipped
    glacier_dir = output_dir / GLACIER_A
    stem = f"usgs_benchmark_stakes_Foo_{GLACIER_A}"
    assert matches.loc["Foo", "figure"] == str(glacier_dir / f"{stem}.png")
    for suffix in (".png", ".pdf", "_scatter.png", "_scatter.pdf", "_gradient.png", "_gradient.pdf", ".nc", ".csv"):
        assert (glacier_dir / f"{stem}{suffix}").exists(), suffix

    skill = pd.read_csv(output_dir / "usgs_benchmark_stakes_skill.csv")
    gradients = pd.read_csv(output_dir / "usgs_benchmark_stakes_gradients.csv")
    assert set(gradients["source"]) <= {"obs", "model"} and (gradients["rgi_id"] == GLACIER_A).all()
    assert set(skill["site"]) == {"A", "Z", ubs.POOLED}
    bare = skill[skill["site"] == "Z"]  # never on ice: nothing to score, but the reason is reported
    assert (bare["n"] == 0).all() and (bare["n_ice_free_months"] == 24).all()
    annual_a = skill[(skill["site"] == "A") & (skill["variable"] == "ba")].iloc[0]
    assert annual_a["n"] == 1 and annual_a["units"] == "m year^-1"
    assert annual_a["elevation_offset"] == pytest.approx(1520.0 - 1530.0)
    assert annual_a["n_ice_free_months"] == 0
    assert skill[(skill["site"] == "A") & (skill["variable"] == "db")].iloc[0]["n"] == 1

    out = xr.open_dataset(glacier_dir / f"{stem}.nc")
    assert out["run"].values.tolist() == ["RGI2000-v7.0-C-01-00001 id_0", "RGI2000-v7.0-C-01-00001 id_0 uq_1"]
    assert out["site"].values.tolist() == ["A", "Z"]
    expected = expected_balances()
    np.testing.assert_allclose(out["ba_model"].sel(site="A", time=2001).values, expected["ba"])
    assert out["ba"].sel(site="A", time=2001) == -0.7
    assert out["ba_model"].sel(site="Z").isnull().all()
    assert out["db_model"].sizes["visit"] == 1
    np.testing.assert_allclose(out["db_model"].values, expected["db"])
    np.testing.assert_allclose(out["lon"].sel(site="A"), lonlat(*STAKE_CELL)[0])
    assert out.attrs["rgi_id"] == GLACIER_A

    # Observations alone still produce the per-site figure.
    obs_only = tmp_path / "obs_only"
    assert (
        ubs.cli(["--data-dir", str(tmp_path), "--output-dir", str(obs_only), "--rgi-glacier-file", str(rgi_file)]) == 0
    )
    assert (obs_only / GLACIER_A / f"{stem}.png").exists()
    assert not (obs_only / GLACIER_A / f"{stem}_scatter.png").exists()
    assert (obs_only / GLACIER_A / f"{stem}_gradient.png").exists()
    assert (obs_only / GLACIER_A / f"{stem}_gradient.csv").exists()
    assert pd.read_csv(obs_only / "usgs_benchmark_stakes_skill.csv").empty


def test_gradient_fit_recovers_a_hinged_profile():
    """
    A two-piece profile is recovered exactly; a one-sided profile still gets a slope and an extrapolated ELA.
    """
    z = np.array([1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0])
    ela, below, above = 1550.0, 0.008, 0.003
    b = np.where(z < ela, below * (z - ela), above * (z - ela))
    fit = ubs.fit_gradient(b, z)
    assert fit is not None
    assert fit["ela"] == pytest.approx(ela, abs=5.0)
    assert fit["gradient_below"] == pytest.approx(below, rel=0.02)
    assert fit["gradient_above"] == pytest.approx(above, rel=0.02)
    assert fit["ela_in_range"] == 1.0 and fit["rmse"] < 0.01 and fit["n"] == 8

    winter = 0.5 + 0.001 * (z - 1200.0)  # positive everywhere: the ELA lies below the stakes
    fit = ubs.fit_gradient(winter, z)
    assert fit is not None
    assert fit["ela"] < z.min() and fit["ela_in_range"] == 0.0
    assert fit["gradient_above"] == pytest.approx(0.001, rel=0.05)
    assert np.isnan(fit["gradient_below"])
    assert ubs.fit_gradient(b[:3], z[:3]) is None
    assert ubs.fit_gradient(b, np.full_like(z, 1500.0)) is None
