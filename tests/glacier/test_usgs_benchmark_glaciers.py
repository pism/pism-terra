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
Tests for the USGS benchmark-glacier comparison against scalar output.

The release, outlines and scalar-file writer come from ``conftest.py``;
nothing touches the network.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pism_terra.glacier import usgs_benchmark_glaciers as ub

GLACIER_A = "RGI2000-v7.0-G-01-00001"
GLACIER_B = "RGI2000-v7.0-G-01-00002"


def test_to_mass_rate(usgs_release):
    """
    One metre water equivalent over one square kilometre is 1e-3 Gt.

    Parameters
    ----------
    usgs_release : dict of str to Path
        The synthetic release directories.
    """
    rate = ub.to_mass_rate(ub.load_glacier_wide(usgs_release["data"], "Foo"))
    np.testing.assert_allclose(rate["Ba"].values, [-1.0 * 10 * 1e-3, -2.0 * 8 * 1e-3, -0.5 * 8 * 1e-3])
    np.testing.assert_allclose(rate["Ba_unc"].values, [0.2 * 10 * 1e-3, 0.2 * 8 * 1e-3, 0.2 * 8 * 1e-3])
    assert rate["Ba"].attrs["units"] == "Gt year^-1"
    assert "area" in rate


def test_model_series_selects_glacier_and_years(tmp_path, scalar_file):
    """
    Mid-year stamps map to their own year and only the requested glacier is kept.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    scalar_file : callable
        Writes a synthetic per-glacier scalar file.
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


def test_cli_end_to_end(tmp_path, usgs_release, usgs_rgi, monkeypatch, scalar_file):
    """
    The command writes a figure and a NetCDF under the RGI ID, plus the match table, and skips point-only glaciers.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    usgs_release : dict of str to Path
        The synthetic release directories.
    usgs_rgi : geopandas.GeoDataFrame
        The two synthetic outlines.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the downloads.
    scalar_file : callable
        Writes a synthetic per-glacier scalar file.
    """
    monkeypatch.setattr(ub, "download_usgs_benchmark", lambda data_dir, force_overwrite=False: usgs_release)
    rgi_file = tmp_path / "rgi.gpkg"
    usgs_rgi.to_file(rgi_file, driver="GPKG")
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
    glacier_dir = output_dir / GLACIER_A
    assert matches.loc["Foo", "figure"] == str(glacier_dir / f"usgs_benchmark_Foo_{GLACIER_A}.png")
    assert (glacier_dir / f"usgs_benchmark_Foo_{GLACIER_A}.png").exists()
    assert (glacier_dir / f"usgs_benchmark_Foo_{GLACIER_A}.pdf").exists()
    out = xr.open_dataset(glacier_dir / f"usgs_benchmark_Foo_{GLACIER_A}.nc")
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


def test_model_series_carries_ice_area(tmp_path, scalar_file):
    """
    ``sftgif`` times the cell area rides along as the ``area`` coordinate; without it, NaN.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    scalar_file : callable
        Writes a synthetic per-glacier scalar file.
    """
    run = tmp_path / "run" / "output" / "processed_scalar"
    run.mkdir(parents=True)
    scalar_file(run / "scalar_G_with.nc", values=(1.0, 2.0), sftgif=(4000.0, 40.0), spacing=500.0)
    model = ub.load_model_series([run / "scalar_G_with.nc"], GLACIER_A, root=tmp_path)
    assert model is not None
    np.testing.assert_allclose(model["area"].values, 4000.0 * 500.0**2)  # 1000 km^2
    without = ub.load_model_series([scalar_file(run / "scalar_G_without.nc")], GLACIER_A)
    assert without is not None
    assert without["area"].isnull().all()


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
    assert model_mwe is not None
    np.testing.assert_allclose(model_mwe.values, [[1.0, -1.0]])
    assert seasons_mwe is None
