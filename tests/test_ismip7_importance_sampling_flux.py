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
Tests for :mod:`pism_terra.ismip7.greenland.importance_sampling_flux`.

Builds a miniature ensemble of per-basin flux series -- one member on a
synthetic truth, one a little off, one far too negative -- in the layout
``pism-ismip7-postprocess-flux`` writes (monthly, stamped mid-month, regions
named ``GIS_<basin>``) against a synthetic Mankoff product (daily, regions
named after the basin, the last year half reported), and covers:

- pairing the model's regions with the observed ones by their basin;
- averaging onto calendar bins and dropping the bins a record only partly
  covers, on either clock;
- the sign flip of the basal mass balance and the unit conversion;
- the alignment on the bins both fully cover and the sampling window;
- an end-to-end run that rejects the biased member in every region, joins
  the basins and writes the figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pism_terra.ismip7.greenland.importance_sampling_flux import (
    MEMBER_DIM,
    align_ensemble_and_observations,
    find_region_files,
    load_ensemble,
    load_observations,
    load_uq_parameters,
    main,
    match_regions,
    resample_complete,
    run_pipeline,
    sampling_window,
    score_region,
    window_span,
)

#: Model basins and their observed names.
REGIONS = {"GIS_NW": "NW", "GIS_SW": "SW", "GIS_GIS": "GIS"}

#: Offset of every member from the truth, Gt/yr, keyed by UQ draw.
MEMBER_BIAS = {"0": 0.0, "1": 2.0, "2": -40.0}

#: Observed uncertainty, Gt/yr.
UNCERTAINTY = 5.0

#: First and last year the model covers (mid-month stamps, whole years).
MODEL_YEARS = (1990, 1999)


def truth(time: pd.DatetimeIndex, level: float) -> np.ndarray:
    """
    A slowly trending flux, so daily and mid-month sampling agree on the bin means.

    Parameters
    ----------
    time : pandas.DatetimeIndex
        Instants.
    level : float
        Value at the start of 1990.

    Returns
    -------
    numpy.ndarray
        The flux at every instant.
    """
    years = time.year + (time.dayofyear - 1) / 365.25
    return level - 2.0 * (years - 1990.0)


def write_case(directory: Path) -> tuple[Path, Path]:
    """
    Write the miniature ensemble, its parameter table and the observations.

    Parameters
    ----------
    directory : pathlib.Path
        Directory to fill.

    Returns
    -------
    tuple of pathlib.Path
        The run directory and the observation file.
    """
    run_dir = directory / "run"
    scalar_dir = run_dir / "output" / "scalar"
    scalar_dir.mkdir(parents=True)
    obs_dir = run_dir / "output" / "observations"
    obs_dir.mkdir(parents=True)

    model_time = pd.date_range(f"{MODEL_YEARS[0]}-01-01", f"{MODEL_YEARS[1]}-12-01", freq="MS") + pd.Timedelta(days=15)
    levels = {"GIS_NW": -50.0, "GIS_SW": -10.0, "GIS_GIS": -200.0}
    for uq, bias in MEMBER_BIAS.items():
        mb = np.stack([truth(model_time, levels[r]) + bias for r in REGIONS], axis=1)
        ds = xr.Dataset(
            {
                "tendency_of_ice_mass": (("time", "region"), mb, {"units": "Gt year^-1"}),
                "tendency_of_ice_mass_due_to_surface_mass_flux": (
                    ("time", "region"),
                    mb + 100.0,
                    {"units": "Gt year^-1"},
                ),
                "ice_mass_transport_across_grounding_line": (("time", "region"), mb - 100.0, {"units": "Gt year^-1"}),
                "tendency_of_ice_mass_due_to_basal_mass_flux": (
                    ("time", "region"),
                    np.full_like(mb, -3.0),
                    {"units": "Gt year^-1"},
                ),
                "area": (("region",), np.arange(len(REGIONS), dtype=float), {"units": "m^2"}),
            },
            coords={
                "time": model_time,
                "region": np.arange(len(REGIONS), dtype="int32"),
                "region_name": ("region", list(REGIONS)),
            },
        )
        ds.to_netcdf(scalar_dir / f"region_g1500m_id_OCX_uq_{uq}_ocx_prescribed_1990-01-01_2000-01-01.nc")

    pd.DataFrame({"uq": list(MEMBER_BIAS), "a.b": [1.0, 2.0, 3.0], "c.d": [0.1, 0.2, 0.3]}).to_csv(
        run_dir / "output" / "uq.csv", index=False
    )

    # Daily, starting before the model and stopping half-way through its last year.
    obs_time = pd.date_range("1986-01-01", f"{MODEL_YEARS[1]}-06-30", freq="D")
    mb = np.stack([truth(obs_time, levels[r]) for r in REGIONS], axis=1)
    units = {"units": "gigametric_ton / year"}
    obs = xr.Dataset(
        {
            "mass_balance": (("time", "region"), mb, units),
            "mass_balance_uncertainty": (("time", "region"), np.full_like(mb, UNCERTAINTY), units),
            "surface_mass_balance": (("time", "region"), mb + 100.0, units),
            "surface_mass_balance_uncertainty": (("time", "region"), np.full_like(mb, UNCERTAINTY), units),
            "grounding_line_flux": (("time", "region"), mb - 100.0, units),
            "grounding_line_flux_uncertainty": (("time", "region"), np.full_like(mb, UNCERTAINTY), units),
            # Mankoff counts basal melt as a positive loss.
            "basal_mass_balance": (("time", "region"), np.full_like(mb, 3.0), units),
            "basal_mass_balance_uncertainty": (("time", "region"), np.full_like(mb, 1.0), units),
        },
        coords={"time": obs_time, "region": list(REGIONS.values())},
    )
    obs_file = obs_dir / "mankoff_greenland_mass_balance.nc"
    obs.to_netcdf(obs_file)
    return run_dir, obs_file


@pytest.fixture(name="case")
def fixture_case(tmp_path: Path) -> tuple[Path, Path]:
    """
    The miniature ensemble and its observations.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    tuple of pathlib.Path
        The run directory and the observation file.
    """
    return write_case(tmp_path)


def test_match_regions_pairs_basins_by_name():
    """
    Pair a model region with its namesake, else with the basin its name ends in; report the rest.
    """
    pairs = match_regions(["GIS_NW", "GIS", "GIS_XX", "NO"], ["NW", "GIS", "NO"])
    assert pairs == {"GIS_NW": "NW", "GIS": "GIS", "NO": "NO"}


def test_resample_complete_drops_the_bins_a_record_only_reaches_into():
    """
    A daily record that stops in June has no complete year; a mid-month monthly record covers its months.
    """
    daily = pd.date_range("1990-01-01", "1991-06-30", freq="D")
    ds = xr.Dataset({"v": ("time", np.ones(daily.size))}, coords={"time": daily})
    years = resample_complete(ds, "YS")
    assert list(years.time.dt.year.values) == [1990]
    months = resample_complete(ds, "ME")
    assert months.sizes["time"] == 18

    mid_month = pd.date_range("1990-01-01", "1990-12-01", freq="MS") + pd.Timedelta(days=15)
    ds = xr.Dataset({"v": ("time", np.arange(12.0))}, coords={"time": mid_month})
    assert resample_complete(ds, "MS").sizes["time"] == 12
    assert resample_complete(ds, "YS").sizes["time"] == 1
    np.testing.assert_allclose(resample_complete(ds, "YE")["v"], 5.5)

    with pytest.raises(ValueError, match="freq must be one of"):
        resample_complete(ds, "W")


def test_load_ensemble_labels_members_and_regions(case):
    """
    The UQ draw comes from the file name, the region from ``region_name``, the fluxes under their compared names.

    Parameters
    ----------
    case : tuple of pathlib.Path
        The run directory and the observation file.
    """
    run_dir, _ = case
    sim = load_ensemble(find_region_files(run_dir))
    assert list(sim[MEMBER_DIM].values) == list(MEMBER_BIAS)
    assert list(sim.region.values) == list(REGIONS)
    assert set(sim.data_vars) == {"mass_balance", "surface_mass_balance", "grounding_line_flux", "basal_mass_balance"}
    assert sim.sizes["time"] == 12 * (MODEL_YEARS[1] - MODEL_YEARS[0] + 1)


def test_load_observations_flips_the_basal_sign(case):
    """
    Basal melt is a positive loss for Mankoff and a negative flux for PISM.

    Parameters
    ----------
    case : tuple of pathlib.Path
        The run directory and the observation file.
    """
    _, obs_file = case
    obs = load_observations(obs_file)
    assert float(obs["basal_mass_balance"].isel(time=0, region=0)) == -3.0
    assert float(obs["mass_balance"].sel(region="NW").isel(time=0)) < 0


def test_align_keeps_the_years_both_fully_cover(case):
    """
    The observations' half year and the years before the model are dropped; units follow the model.

    Parameters
    ----------
    case : tuple of pathlib.Path
        The run directory and the observation file.
    """
    run_dir, obs_file = case
    sim, obs = align_ensemble_and_observations(
        load_ensemble(find_region_files(run_dir)), load_observations(obs_file), "YS"
    )
    years = list(sim.time.dt.year.values)
    assert years == list(range(MODEL_YEARS[0], MODEL_YEARS[1]))
    assert list(obs.time.values) == list(sim.time.values)
    assert list(obs.region.values) == list(REGIONS)
    assert obs["mass_balance"].attrs["units"] == sim["mass_balance"].attrs["units"]
    # The truth member agrees with the observations on every year, the biased one does not.
    np.testing.assert_allclose(sim["mass_balance"].sel(uq_id="0"), obs["mass_balance"], atol=0.05)
    assert abs(sim["mass_balance"].sel(uq_id="2") - obs["mass_balance"]).min() > 39

    with pytest.raises(ValueError, match="share no region"):
        align_ensemble_and_observations(
            load_ensemble(find_region_files(run_dir)).assign_coords(region=["A", "B", "C"]),
            load_observations(obs_file),
            "YS",
        )


def test_sampling_window_and_its_span(case):
    """
    The window is inclusive on both sides and spans whole bins; an empty one is an error.

    Parameters
    ----------
    case : tuple of pathlib.Path
        The run directory and the observation file.
    """
    run_dir, obs_file = case
    sim, _ = align_ensemble_and_observations(
        load_ensemble(find_region_files(run_dir)), load_observations(obs_file), "YS"
    )
    window = sampling_window(sim, "1992", "1994-06-01")
    assert list(window.time.dt.year.values) == [1992, 1993, 1994]
    first, last = window_span(window, "YS")
    assert (first, last) == (pd.Timestamp("1992-01-01"), pd.Timestamp("1994-12-31"))
    assert sampling_window(sim, None, None).sizes["time"] == sim.sizes["time"]
    with pytest.raises(ValueError, match="no instant between"):
        sampling_window(sim, "2050", None)


def test_score_region_needs_the_flux_on_both_sides(case):
    """
    A flux the observations lack cannot be sampled on, nor can an unknown reduction be used.

    Parameters
    ----------
    case : tuple of pathlib.Path
        The run directory and the observation file.
    """
    run_dir, obs_file = case
    sim, obs = align_ensemble_and_observations(
        load_ensemble(find_region_files(run_dir)), load_observations(obs_file), "YS"
    )
    region = sim.sel(region="GIS_NW", drop=True), obs.sel(region="GIS_NW", drop=True)
    with pytest.raises(ValueError, match="must be on both sides"):
        score_region(region[0], region[1].drop_vars("mass_balance_uncertainty"), "mass_balance")
    with pytest.raises(ValueError, match="reduction must be one of"):
        score_region(region[0], region[1], "mass_balance", reduction="blocks")


def test_run_pipeline_rejects_the_biased_member(tmp_path: Path, case):
    """
    The member 40 Gt/yr off gets no weight anywhere, the truth leads, and the basins join.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    case : tuple of pathlib.Path
        The run directory and the observation file.
    """
    run_dir, _ = case
    out = tmp_path / "out"
    summary = run_pipeline(run_dir, out, fudge_factors=(1.0, 10.0), n_samples=1000, seed=1)

    assert set(summary["region"]) == set(REGIONS) | {"joint"}
    assert (summary["top_uq_id"] == "0").all()
    assert (summary.loc[summary["region"] != "joint", "n_instants"] == MODEL_YEARS[1] - MODEL_YEARS[0]).all()
    assert (summary.loc[summary["region"] == "joint", "n_regions"] == 2).all()

    weights = pd.read_csv(out / "importance_sampling_weights.csv")
    biased = weights[weights[MEMBER_DIM] == 2]
    assert (biased["weights_ff_1"] < 1e-6).all()
    assert (biased["counts_ff_1"] == 0).all()
    # The parameters ride along, so the posterior histograms can be drawn from the table.
    assert {"a.b", "c.d", "rmse", "mae", "bias"} <= set(weights.columns)
    assert weights.loc[weights[MEMBER_DIM] == 2, "rmse"].dropna().gt(39).all()

    for region in REGIONS:
        for name in (
            f"fluxes_{region}_mass_balance_ff_1.png",
            f"fluxes_{region}_mass_balance_ff_10.png",
            f"posterior_{region}_mass_balance_ff_1.png",
            "importance_sampling_mass_balance.nc",
            "importance_sampling_mass_balance.csv",
        ):
            assert (out / region / name).is_file(), name
    assert (out / "joint" / "posterior_joint_mass_balance_ff_1.png").is_file()
    with xr.open_dataset(out / "GIS_NW" / "importance_sampling_mass_balance.nc") as ds:
        assert ds.attrs["start"] == f"{MODEL_YEARS[0]}-01-01"
        assert ds.attrs["end"] == f"{MODEL_YEARS[1] - 1}-12-31"
        assert list(ds.fudge_factor.values) == [1.0, 10.0]
    with xr.open_dataset(out / "joint" / "importance_sampling_mass_balance.nc") as ds:
        assert sorted(ds.attrs["regions"]) == ["GIS_NW", "GIS_SW"]


def test_run_pipeline_honours_the_window_and_the_variable(tmp_path: Path, case):
    """
    A monthly window of one year samples twelve instants, on the flux asked for.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    case : tuple of pathlib.Path
        The run directory and the observation file.
    """
    run_dir, _ = case
    out = tmp_path / "out"
    summary = run_pipeline(
        run_dir,
        out,
        variable="grounding_line_flux",
        freq="MS",
        start="1995-01-01",
        end="1995-12-31",
        fudge_factors=(3.0,),
        n_samples=100,
    )
    assert (summary["variable"] == "grounding_line_flux").all()
    assert (summary.loc[summary["region"] != "joint", "n_instants"] == 12).all()
    assert (out / "GIS_SW" / "fluxes_GIS_SW_grounding_line_flux_ff_3.png").is_file()

    with pytest.raises(ValueError, match="variable must be one of"):
        run_pipeline(run_dir, out, variable="thickness")


def test_load_uq_parameters_finds_the_table(case, tmp_path: Path):
    """
    The table is found below the run directory or taken from the path given; a missing one is an error.

    Parameters
    ----------
    case : tuple of pathlib.Path
        The run directory and the observation file.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    run_dir, _ = case
    df = load_uq_parameters(run_dir)
    assert list(df.index) == list(MEMBER_BIAS)
    assert df.index.name == MEMBER_DIM
    assert list(load_uq_parameters(tmp_path, run_dir / "output" / "uq.csv").columns) == ["a.b", "c.d"]
    with pytest.raises(FileNotFoundError):
        load_uq_parameters(tmp_path / "nowhere")


def test_cli_runs_from_the_run_directory(tmp_path: Path, case, monkeypatch):
    """
    The console entry point finds the observations and the parameter table below RUN_DIR.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    case : tuple of pathlib.Path
        The run directory and the observation file.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture for the working directory.
    """
    run_dir, _ = case
    out = tmp_path / "cli"
    monkeypatch.chdir(tmp_path)
    assert main([str(run_dir), str(out), "--fudge-factors", "2", "--n-samples", "100", "--start", "1991"]) == 0
    summary = pd.read_csv(out / "importance_sampling_summary.csv")
    assert list(summary["fudge_factor"].unique()) == [2.0]
    assert (out / "importance_sampling.log").stat().st_size > 0
