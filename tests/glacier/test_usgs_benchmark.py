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
Tests for the USGS benchmark-glacier comparison.

Everything is synthetic: a two-glacier release written to ``tmp_path``, two
square RGI outlines, and a scalar file in the layout ``pism-postprocess-scalar``
writes. No network access.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from pism_terra.glacier import usgs_benchmark as ub

GLACIER_A = "RGI2000-v7.0-G-01-00001"
GLACIER_B = "RGI2000-v7.0-G-01-00002"


@pytest.fixture(name="release")
def fixture_release(tmp_path: Path) -> dict[str, Path]:
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
        :func:`pism_terra.glacier.usgs_benchmark.download_usgs_benchmark`
        returns them.
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


@pytest.fixture(name="rgi")
def fixture_rgi() -> gpd.GeoDataFrame:
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


def scalar_file(
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


def test_sites_keep_only_stakes(release):
    """
    Weather stations are dropped and the points carry the glacier name.

    Parameters
    ----------
    release : dict of str to Path
        The synthetic release directories.
    """
    sites = ub.load_sites(release["sites"])
    assert sites["Type"].eq("Glaciological").all()
    assert sites["Glacier"].tolist() == ["Foo", "Foo", "Foo", "Bar", "Baz"]
    assert sites.crs.to_epsg() == 4326


def test_match_takes_the_modal_outline(release, rgi):
    """
    Two of Foo's three stakes sit in A, so A wins; the count is reported.

    Parameters
    ----------
    release : dict of str to Path
        The synthetic release directories.
    rgi : geopandas.GeoDataFrame
        The two synthetic outlines.
    """
    matches = ub.match_rgi_ids(ub.load_sites(release["sites"]), rgi).set_index("glacier")
    assert matches.loc["Foo", "rgi_id"] == GLACIER_A
    assert matches.loc["Foo", "n_sites"] == 3
    assert matches.loc["Foo", "n_matched"] == 2
    assert matches.loc["Foo", "area_km2"] == 10.0


def test_match_falls_back_to_nearest(release, rgi):
    """
    Fall back to the nearer outline when no stake lies inside any, within a distance cap.

    Bar's stake is a kilometre north of outline B; Baz's is more than a
    hundred kilometres from either outline and stays unmatched.

    Parameters
    ----------
    release : dict of str to Path
        The synthetic release directories.
    rgi : geopandas.GeoDataFrame
        The two synthetic outlines.
    """
    matches = ub.match_rgi_ids(ub.load_sites(release["sites"]), rgi).set_index("glacier")
    assert matches.loc["Bar", "rgi_id"] == GLACIER_B
    assert matches.loc["Bar", "n_matched"] == 0
    assert pd.isna(matches.loc["Baz", "rgi_id"])
    assert pd.isna(matches.loc["Baz", "area_km2"])


def test_glacier_wide_area_is_carried_forward(release):
    """
    The year without an area-altitude distribution keeps the last known area.

    Parameters
    ----------
    release : dict of str to Path
        The synthetic release directories.
    """
    obs = ub.load_glacier_wide(release["data"], "Foo")
    assert obs is not None
    assert obs["time"].values.tolist() == [2000, 2001, 2002]
    np.testing.assert_allclose(obs["area"].values, [10.0, 8.0, 8.0])
    assert obs["Ba"].attrs["units"] == "m year^-1"
    assert "Ba_unc" in obs
    assert np.isnan(obs["Bw"].values[1])


def test_glacier_wide_missing_for_point_only_glacier(release):
    """
    A glacier without a glacier-wide solution file yields None.

    Parameters
    ----------
    release : dict of str to Path
        The synthetic release directories.
    """
    assert ub.load_glacier_wide(release["data"], "Bar") is None


def test_to_mass_rate(release):
    """
    One metre water equivalent over one square kilometre is 1e-3 Gt.

    Parameters
    ----------
    release : dict of str to Path
        The synthetic release directories.
    """
    rate = ub.to_mass_rate(ub.load_glacier_wide(release["data"], "Foo"))
    np.testing.assert_allclose(rate["Ba"].values, [-1.0 * 10 * 1e-3, -2.0 * 8 * 1e-3, -0.5 * 8 * 1e-3])
    np.testing.assert_allclose(rate["Ba_unc"].values, [0.2 * 10 * 1e-3, 0.2 * 8 * 1e-3, 0.2 * 8 * 1e-3])
    assert rate["Ba"].attrs["units"] == "Gt year^-1"
    assert "area" in rate


def test_model_series_selects_glacier_and_years(tmp_path):
    """
    Mid-year stamps map to their own year and only the requested glacier is kept.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    """
    run_a = tmp_path / "run_a" / "output" / "processed_scalar"
    run_b = tmp_path / "run_b" / "output" / "processed_scalar"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    scalar_file(run_a / "scalar_G_first.nc", values=(1.0, 2.0))
    scalar_file(run_b / "scalar_G_second.nc", values=(3.0, 4.0))
    scalar_file(run_b / "scalar_G_other.nc", labels=("RGI2000-v7.0-G-01-99999",), values=(9.0,))
    files = ub.find_model_files(tmp_path)
    assert len(files) == 3

    model = ub.load_model_series(files, GLACIER_B, root=tmp_path)
    assert model is not None
    assert model.sizes["run"] == 2
    assert model["time"].values.tolist() == [2000, 2001, 2002]
    assert sorted(model["run"].values.tolist()) == ["run_a/scalar_G_first.nc", "run_b/scalar_G_second.nc"]
    np.testing.assert_allclose(model.sel(run="run_a/scalar_G_first.nc").values, 2.0)
    np.testing.assert_allclose(model.median("run").values, 3.0)
    assert ub.load_model_series(files, "RGI2000-v7.0-G-01-12345") is None


def test_mean_years_handles_bounds_and_new_year_stamps():
    """
    Interval starts come from ``time_bounds``; a 1 January stamp closes the previous year.
    """
    times = pd.to_datetime(["2001-01-01", "2002-01-01"])
    ds = xr.Dataset(coords={"time": ("time", times)})
    assert ub._mean_years(ds).tolist() == [2000, 2001]  # pylint: disable=protected-access

    bounds = np.array([pd.to_datetime(["1990-01-01", "1991-01-01"]), pd.to_datetime(["1991-01-01", "1992-01-01"])])
    ds = ds.assign(time_bounds=(("time", "nv"), bounds))
    ds["time"].attrs["bounds"] = "time_bounds"
    assert ub._mean_years(ds).tolist() == [1990, 1991]  # pylint: disable=protected-access


def test_download_uses_resolved_urls(tmp_path, monkeypatch):
    """
    Fetch archives by name from the item's file list and extract them next to it.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the downloads.
    """
    fetched: list[str] = []

    def fake_urls(item_id=ub.SCIENCEBASE_ITEM):  # pylint: disable=unused-argument
        """
        Stand in for the ScienceBase lookup.

        Parameters
        ----------
        item_id : str
            Ignored.

        Returns
        -------
        dict of str to str
            File-name to URL mapping.
        """
        return {ub.DATA_ARCHIVE: "https://example.org/data.zip", ub.SITES_ARCHIVE: "https://example.org/sites.zip"}

    def fake_download(url, dest=None, force_overwrite=False, verbose=True):  # pylint: disable=unused-argument
        """
        Record the URL and write an empty zip.

        Parameters
        ----------
        url : str
            Source URL.
        dest : Path
            Destination.
        force_overwrite : bool
            Ignored.
        verbose : bool
            Ignored.

        Returns
        -------
        Path
            The destination.
        """
        import zipfile  # pylint: disable=import-outside-toplevel

        fetched.append(url)
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("placeholder.txt", "")
        return Path(dest)

    monkeypatch.setattr(ub, "sciencebase_file_urls", fake_urls)
    monkeypatch.setattr(ub, "download_archive", fake_download)

    paths = ub.download_usgs_benchmark(tmp_path)
    assert fetched == ["https://example.org/data.zip", "https://example.org/sites.zip"]
    assert (paths["data"] / "placeholder.txt").exists()
    assert paths["sites"] == tmp_path / "Glacier_Mass_Balance_Sites"

    ub.download_usgs_benchmark(tmp_path)
    assert len(fetched) == 2


def test_cli_end_to_end(tmp_path, release, rgi, monkeypatch):
    """
    The command writes a figure, a NetCDF and the match table, and skips point-only glaciers.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    release : dict of str to Path
        The synthetic release directories.
    rgi : geopandas.GeoDataFrame
        The two synthetic outlines.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the downloads.
    """
    monkeypatch.setattr(ub, "download_usgs_benchmark", lambda data_dir, force_overwrite=False: release)
    rgi_file = tmp_path / "rgi.gpkg"
    rgi.to_file(rgi_file, driver="GPKG")
    run_dir = tmp_path / "runs" / "RGI2000-v7.0-C-01-00001" / "output" / "processed_scalar"
    run_dir.mkdir(parents=True)
    scalar_file(run_dir / "scalar_G_g500m_RGI2000-v7.0-C-01-00001_id_0_2000-01-01_2003-01-01.nc")
    output_dir = tmp_path / "out"

    assert (
        ub.cli(
            [
                str(tmp_path / "runs"),
                "--data-dir",
                str(tmp_path),
                "--output-dir",
                str(output_dir),
                "--rgi-glacier-file",
                str(rgi_file),
                "--uncertainty",
                "0.3",
            ]
        )
        == 0
    )

    matches = pd.read_csv(output_dir / "usgs_benchmark_rgi_match.csv").set_index("glacier")
    assert matches.loc["Foo", "n_runs"] == 1
    skill = pd.read_csv(output_dir / "usgs_benchmark_skill.csv")
    total = skill[(skill["glacier"] == "Foo") & (skill["source"] == "total")].iloc[0]
    assert total["variable"] == "Ba" and total["n"] == 3 and total["units"] == "m year^-1"
    # the constant 1 Gt/yr becomes 100/125/125 m w.e. over the observed areas, so r is defined
    assert np.isfinite(total["r"]) and np.isfinite(total["mae"])
    assert pd.isna(matches.loc["Bar", "figure"])
    assert pd.isna(matches.loc["Baz", "rgi_id"])
    assert (output_dir / f"usgs_benchmark_Foo_{GLACIER_A}.png").exists()
    assert (output_dir / f"usgs_benchmark_Foo_{GLACIER_A}.pdf").exists()
    out = xr.open_dataset(output_dir / f"usgs_benchmark_Foo_{GLACIER_A}.nc")
    assert {"Ba", "Ba_mwe", "Bw_unc", "tendency_of_ice_mass", "tendency_of_ice_mass_mwe", "area"} <= set(out.data_vars)
    # 1 Gt/yr over the observed area (10 km^2, then 8) is 100 and 125 m w.e./yr
    np.testing.assert_allclose(out["tendency_of_ice_mass_mwe"].values.ravel(), [100.0, 125.0, 125.0])
    assert out["tendency_of_ice_mass"].sizes["run"] == 1
    assert out["run"].values.tolist() == ["RGI2000-v7.0-C-01-00001 id_0"]
    np.testing.assert_allclose(out["tendency_of_ice_mass"].values, 1.0)


def test_skill_scores_and_formatting():
    """
    Score on the ensemble median over the common finite years; withhold r below three.
    """
    obs = xr.Dataset(
        {
            "Ba": ("time", [1.0, 2.0, 3.0, 4.0, np.nan]),
            "Bw": ("time", [2.0, 2.5, 3.0, 3.5, 4.0]),
            "Bs": ("time", [-1.0, -0.5, 0.0, 0.5, 1.0]),
        },
        coords={"time": [2000, 2001, 2002, 2003, 2004]},
    )
    runs = pd.Index(["a", "b", "c"], name="run")
    seasons = xr.Dataset(
        {
            # median across runs is obs + 0.5
            "Bw": (("run", "time"), [[2.5, 3.0, 3.5, 4.0, 4.5], [0.0, 0.0, 0.0, 0.0, 0.0], [9.0, 9.0, 9.0, 9.0, 9.0]]),
            # only two overlapping years: error statistics but no r
            "Bs": (("run", "time"), np.array([[-2.0, -1.0, np.nan, np.nan, np.nan]] * 3)),
        },
        coords={"run": runs, "time": obs["time"]},
    )
    # total model: perfectly anti-correlated with Ba on 2001-2003 (2000 missing, 2004 obs NaN)
    model = xr.DataArray(
        [[-2.0, -3.0, -4.0, 0.0]], dims=("run", "time"), coords={"run": ["x"], "time": [2001, 2002, 2003, 2004]}
    )

    skill = ub.skill_scores(obs, model, seasons).set_index(["variable", "source"])
    assert list(skill.columns) == ["season", "n", "r", "mae", "bias"]
    winter = skill.loc[("Bw", "surface")]
    assert winter["n"] == 5 and winter["r"] == pytest.approx(1.0) and winter["mae"] == pytest.approx(0.5)
    summer = skill.loc[("Bs", "surface")]
    assert summer["n"] == 2 and pd.isna(summer["r"]) and summer["mae"] == pytest.approx(0.75)
    assert ("Ba", "surface") not in skill.index  # seasons carry no Ba here
    total = skill.loc[("Ba", "total")]
    assert total["n"] == 3 and total["r"] == pytest.approx(-1.0) and total["bias"] == pytest.approx(-6.0)

    text = ub.format_skill(skill.reset_index())
    assert "winter" in text and "r=1.00" in text and "MAE=0.5 Gt/yr" in text and "n=5" in text
    assert "summer" in text and "r=n/a" in text
    assert "annual (total)" in text
    assert ub.skill_scores(obs).empty
    assert ub.format_skill(ub.skill_scores(obs)) == ""


def test_model_series_carries_ice_area(tmp_path):
    """
    ``sftgif`` times the cell area rides along as the ``area`` coordinate; without it, NaN.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    """
    run = tmp_path / "run" / "output" / "processed_scalar"
    run.mkdir(parents=True)
    scalar_file(run / "scalar_G_with.nc", values=(1.0, 2.0), sftgif=(4000.0, 40.0), spacing=500.0)
    model = ub.load_model_series([run / "scalar_G_with.nc"], GLACIER_A, root=tmp_path)
    assert model is not None
    np.testing.assert_allclose(model["area"].values, 4000.0 * 500.0**2)  # 1000 km^2
    assert ub.load_model_series([scalar_file(run / "scalar_G_without.nc")], GLACIER_A)["area"].isnull().all()


def test_specific_balance_conversion():
    """
    Convert one gigatonne a year over a thousand square kilometres to one metre water equivalent.
    """
    rate = xr.DataArray([1.0, -0.5], dims="time", coords={"time": [2000, 2001]}, attrs={"units": "Gt year^-1"})
    area = xr.DataArray([1e9, 5e8], dims="time", coords={"time": [2000, 2001]}, attrs={"units": "m^2"})
    mwe = ub.specific_balance(rate, area)
    np.testing.assert_allclose(mwe.values, [1.0, -1.0])
    assert mwe.attrs["units"] == "m year^-1"

    obs = xr.Dataset({"area": ("time", [1000.0, 500.0], {"units": "km^2"})}, coords={"time": [2000, 2001]})
    # a run without an area of its own falls back to the observed area
    model = rate.expand_dims(run=["a"]).assign_attrs(units="Gt year^-1")
    model.name = "tendency_of_ice_mass"
    model_mwe, seasons_mwe = ub.to_specific_balances(model, None, obs)
    np.testing.assert_allclose(model_mwe.values, [[1.0, -1.0]])
    assert seasons_mwe is None
