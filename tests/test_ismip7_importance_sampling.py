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
Tests for :mod:`pism_terra.ismip7.greenland.importance_sampling`.

Builds a miniature ensemble -- one member following a synthetic truth, one
thinning several metres too much -- against synthetic Khan and Smith
products, and covers:

- naming a member after its set counter, which is the key into the
  protocol's parameter table, and falling back to a UQ draw.
- leaving a member on the submission grid alone while regridding a 900 m one
  conservatively, and that the conservative step preserves the integral.
- error statistics over the cells both sides cover, holes excluded.
- an end-to-end run that rejects the biased member and joins the two
  products into one posterior.
"""

from __future__ import annotations

from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pism_terra.ismip7.greenland.importance_sampling import (
    ERROR_VAR,
    MEMBER_DIM,
    SIM_VARS,
    align_to_observations,
    error_stats,
    find_member_files,
    load_ensemble,
    load_observations,
    main,
    member_label,
    product_name,
    run_pipeline,
    same_grid,
    simulated_variable,
    to_observed_grid,
)

#: Size of the synthetic observation grid, cells.
N = 32

#: Cells of the observation hole, per side.
HOLE = 4


def grid(n: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a synthetic grid, with ``y`` descending as the real files have it.

    Parameters
    ----------
    n : int
        Cells per side.
    spacing : float
        Cell size, metres.

    Returns
    -------
    tuple of numpy.ndarray
        The ``x`` and ``y`` coordinates.
    """
    return (
        -720000.0 + spacing * np.arange(n, dtype="float64"),
        -570000.0 - spacing * np.arange(n, dtype="float64"),
    )


def truth_field(n: int, n_time: int) -> np.ndarray:
    """
    A synthetic cumulative thinning: smooth in space, growing with time.

    Smooth because the block reduction measures the field's decorrelation
    length, and white noise would make every block one cell.

    Parameters
    ----------
    n : int
        Cells per side.
    n_time : int
        Number of records.

    Returns
    -------
    numpy.ndarray
        Field of shape ``(n_time, n, n)``.
    """
    ramp = np.linspace(1.0, 0.0, n)[None, :] * np.ones((n, 1))
    return np.stack([-(i + 1) * 2.0 * ramp for i in range(n_time)])


def write_case(directory: Path, n_time: int = 4) -> tuple[Path, Path, list[Path]]:
    """
    Write a miniature ensemble and the two observed products it is scored against.

    Parameters
    ----------
    directory : pathlib.Path
        Directory to fill.
    n_time : int, optional
        Number of records the annual product carries.

    Returns
    -------
    tuple
        The run directory, the observation directory and the observed files.
    """
    x, y = grid(N, 1000.0)
    times = [cftime.DatetimeGregorian(2004 + i, 1, 1) for i in range(n_time)]
    truth = truth_field(N, n_time)
    truth[:, :HOLE, :HOLE] = np.nan  # the observations have a hole

    obs_dir = directory / "obs"
    obs_dir.mkdir(parents=True, exist_ok=True)
    observed = []
    for name, data, stamps in (
        ("dh_khan_g1000m_test.nc", truth, times),
        ("dh_smith_g1000m_test.nc", truth[-1:], times[-1:]),
    ):
        ds = xr.Dataset(
            {"dh": (("time", "y", "x"), data.astype("float32"), {"units": "m"})},
            coords={"time": stamps, "y": y, "x": x},
        )
        ds.to_netcdf(obs_dir / name)
        observed.append(obs_dir / name)

    run_dir = directory / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    # The model has no holes, so the truth's are filled before the members are
    # built: the statistics must then exclude them on the observations' side.
    filled = np.nan_to_num(truth, nan=0.0)
    for label, bias in (("C011", 0.0), ("C012", -4.0)):
        ds = xr.Dataset(
            {"lithk": (("time", "y", "x"), (filled + bias).astype("float32"), {"units": "m"})},
            coords={"time": times, "y": y, "x": x},
        )
        ds.to_netcdf(run_dir / f"dh_lithk_lithk_GrIS_UAF_PISM_m001_OCX_f001_OCX_{label}_1990-2024.nc")
    return run_dir, obs_dir, observed


@pytest.fixture(name="case")
def fixture_case(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    """
    A miniature ensemble and its observations.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    tuple
        The run directory, the observation directory and the observed files.
    """
    return write_case(tmp_path)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # The set counter wins: it is what the protocol's parameter table is keyed on.
        ("dh_lithk_lithk_GrIS_UAF_PISM_m001_OCX_f001_OCX_C011_1990-2024.nc", "C011"),
        ("dh_lithk_lithk_GrIS_UAF_PISM_m007_CESM2-WACCM_f001_ssp585_P207_2015-2299.nc", "P207"),
        ("dh_lithk_lithk_GrIS_UAF_PISM_m002_MRI-ESM2-0_f002_ssp126_E015_2015-2299.nc", "E015"),
        # Glacier-style names carry a UQ draw instead.
        ("dh_g900m_id_CTRL_uq_0042_2003_2019.nc", "0042"),
        # Neither: fall back to something that is at least unique per file.
        ("dh_something_else.nc", "dh_something_else"),
    ],
)
def test_member_label(name, expected):
    """
    Name a member after its set counter, then its UQ draw, then its file.

    Parameters
    ----------
    name : str
        Member file name.
    expected : str
        Label it should be given.
    """
    assert member_label(Path(name)) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("dh_smith_g5000m_ICESat1-ICESat2-2021.nc", "smith"),
        ("elsewhere.nc", "elsewhere"),
    ],
)
def test_product_name(name, expected):
    """
    Read the product out of a staged observation file name.

    Parameters
    ----------
    name : str
        Observation file name.
    expected : str
        Product it names.
    """
    assert product_name(Path(name)) == expected


def test_find_member_files_needs_members(tmp_path: Path):
    """
    Refuse an empty ensemble rather than reporting statistics over no members.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    with pytest.raises(FileNotFoundError, match="no files matching"):
        find_member_files(tmp_path)


def test_same_grid_ignores_the_axis_direction():
    """
    Run ``y`` the other way and it is still the same grid.

    The submission files descend in ``y`` where the observations ascend;
    alignment sorts that out, so it must not trigger a regrid.
    """
    x, y = grid(8, 1000.0)
    a = xr.Dataset(coords={"y": y, "x": x})
    assert same_grid(a, xr.Dataset(coords={"y": y[::-1], "x": x}))
    assert not same_grid(a, xr.Dataset(coords={"y": y[:4], "x": x}))
    assert not same_grid(a, xr.Dataset(coords={"y": y + 500.0, "x": x}))


def test_to_observed_grid_leaves_the_submission_grid_alone(case):
    """
    Do not regrid what is already on the observed grid.

    Members written into the submission tree share the observations' 1 km
    grid exactly; regridding them would only blur them.

    Parameters
    ----------
    case : tuple
        Run directory, observation directory and observed files.
    """
    run_dir, _, observed = case
    obs = load_observations(observed[0], 0.1, 0.5)
    sim = load_ensemble(find_member_files(run_dir))
    result = to_observed_grid(sim, obs)
    assert result is sim


def test_to_observed_grid_conserves_the_integral(case):
    """
    Regrid a finer ensemble conservatively, preserving what it integrates to.

    A 900 m member has to reach the 1 km observations, and the quantity is a
    thickness change that sums into a mass budget: the mean over the shared
    extent must survive the change of support, which bilinear interpolation
    does not guarantee.

    Parameters
    ----------
    case : tuple
        Run directory, observation directory and observed files.
    """
    _, _, observed = case
    obs = load_observations(observed[0], 0.1, 0.5)

    # A 900 m grid covering exactly the same extent as the 1 km observations,
    # carrying a smooth field so the comparison is not about resolution.
    n_fine = N * 1000 // 900
    x_fine = float(obs["x"][0]) + 900.0 * np.arange(n_fine)
    y_fine = float(obs["y"][0]) - 900.0 * np.arange(n_fine)
    fine = xr.Dataset(
        {"dh": (("y", "x"), truth_field(n_fine, 1)[0])},
        coords={"y": y_fine, "x": x_fine},
    )

    coarse = to_observed_grid(fine, obs)
    assert coarse.sizes["x"] == obs.sizes["x"] and coarse.sizes["y"] == obs.sizes["y"]
    # Compare over the cells the finer grid actually covers -- it stops
    # 400 m short of the 1 km extent, and those edge cells come back empty.
    covered = coarse["dh"].notnull()
    assert float(coarse["dh"].where(covered).mean()) == pytest.approx(float(fine["dh"].mean()), rel=0.02)


def test_error_stats_excludes_the_observation_holes(case):
    """
    Score only the cells both sides have, and say how many those were.

    Members can cover different parts of the observations, so the count is
    part of the result rather than a detail: without it two RMSEs are not
    necessarily comparable.

    Parameters
    ----------
    case : tuple
        Run directory, observation directory and observed files.
    """
    run_dir, _, observed = case
    obs = load_observations(observed[0], 0.1, 0.5)
    sim, obs = align_to_observations(load_ensemble(find_member_files(run_dir)), obs)
    stats = error_stats(sim["dh"], obs["dh"]).compute()

    n_time = obs.sizes["time"]
    assert stats["n_cells"].values.tolist() == [n_time * (N * N - HOLE * HOLE)] * 2
    # The members differ from the truth by a constant, so every statistic is
    # that constant and the signed bias separates them.
    np.testing.assert_allclose(stats["rmse"].values, [0.0, 4.0], atol=1e-5)
    np.testing.assert_allclose(stats["mae"].values, [0.0, 4.0], atol=1e-5)
    np.testing.assert_allclose(stats["bias"].values, [0.0, -4.0], atol=1e-5)
    assert stats["rmse"].attrs["units"] == "m"


def test_align_to_observations_needs_a_shared_record(case):
    """
    Refuse an ensemble that overlaps the observations nowhere in time.

    Silently returning an empty comparison would look like a perfect fit.

    Parameters
    ----------
    case : tuple
        Run directory, observation directory and observed files.
    """
    run_dir, _, observed = case
    obs = load_observations(observed[0], 0.1, 0.5)
    shifted = obs.assign_coords(time=[t.replace(year=t.year + 100) for t in obs["time"].values])
    with pytest.raises(ValueError, match="share no time step"):
        align_to_observations(load_ensemble(find_member_files(run_dir)), shifted)


def test_observation_uncertainty_has_a_floor(case):
    """
    Give the observations an error, since neither product ships one.

    The floor is what keeps the cells that did not change -- where a purely
    relative error goes to zero -- from dominating the likelihood.

    Parameters
    ----------
    case : tuple
        Run directory, observation directory and observed files.
    """
    _, _, observed = case
    obs = load_observations(observed[0], 0.1, 0.5)
    assert "dh_error" in obs
    assert float(obs["dh_error"].min()) == pytest.approx(0.5)
    expected = np.maximum(0.1 * abs(obs["dh"]), 0.5)
    np.testing.assert_allclose(obs["dh_error"].values, expected.values, equal_nan=True)


def test_run_pipeline_rejects_the_biased_member(tmp_path: Path, case):
    """
    Score both products end to end and join them into one posterior.

    The member that follows the truth must take essentially all the weight
    and lead the bootstrap ranking; the one thinning four metres too much
    must be rejected and not be tied with the best.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    case : tuple
        Run directory, observation directory and observed files.
    """
    run_dir, _, observed = case
    out = tmp_path / "out"
    summary = run_pipeline(run_dir, observed, out, n_samples=1000, n_boot=50)

    assert set(summary["product"]) == {"khan", "smith", "joint"}
    for product in ("khan", "smith", "joint"):
        assert (out / product / "importance_sampling.csv").exists()
        assert (out / product / "importance_sampling.nc").exists()
    assert (out / "importance_sampling_summary.csv").exists()

    n_samples = 1000
    for product, rows in summary.groupby("product"):
        for fudge_factor, block in rows.groupby("fudge_factor"):
            block = block.set_index(MEMBER_DIM)
            assert float(block.loc["C011", "weights"]) > 0.99, (product, fudge_factor)
            assert float(block.loc["C012", "weights"]) < 0.01, (product, fudge_factor)
            # The counts follow the weights rather than vanishing: at the
            # largest fudge factor the inflated error does let the worse
            # member through, just rarely, which is the point of the sweep.
            assert int(block.loc["C012", "counts"]) < 0.02 * n_samples, (product, fudge_factor)
            assert int(block.loc["C011", "counts"]) > 0.98 * n_samples, (product, fudge_factor)

    # The ranking is per product, so it is only on the per-product rows.
    khan = summary[summary["product"] == "khan"].set_index(MEMBER_DIM)
    assert bool(khan.loc["C011", "tied_with_best"].iloc[0])
    assert not bool(khan.loc["C012", "tied_with_best"].iloc[0])

    written = pd.read_csv(out / "khan" / "importance_sampling.csv")
    assert list(written.columns[:2]) == ["fudge_factor", MEMBER_DIM]


def write_observations_with_error(path: Path, error: np.ndarray) -> Path:
    """
    Write an observed product that ships its own per-cell uncertainty.

    Parameters
    ----------
    path : pathlib.Path
        File to write.
    error : numpy.ndarray
        Uncertainty field of shape ``(N, N)``.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    x, y = grid(N, 1000.0)
    times = [cftime.DatetimeGregorian(2019, 1, 1)]
    ds = xr.Dataset(
        {
            "dh": (("time", "y", "x"), truth_field(N, 1).astype("float32"), {"units": "m"}),
            "dh_error": (("time", "y", "x"), error[None].astype("float32"), {"units": "m"}),
        },
        coords={"time": times, "y": y, "x": x},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def test_load_observations_prefers_the_reported_uncertainty(tmp_path: Path):
    """
    Use the product's own error rather than inventing one.

    The Smith archive ships a per-cell RMSE; substituting a flat relative
    error for it would throw away the one thing that distinguishes a
    well-surveyed cell from an unconstrained one.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    error = np.full((N, N), 2.0)
    path = write_observations_with_error(tmp_path / "dh_smith_g5000m_test.nc", error)
    obs = load_observations(path, 0.1, 0.5, max_error=None)
    np.testing.assert_allclose(obs[ERROR_VAR].isel(time=0).values, 2.0)


def test_load_observations_floors_the_reported_uncertainty(tmp_path: Path):
    """
    Never let a cell claim an accuracy the survey does not have.

    The archive reports RMSEs down to a fraction of a millimetre. Taken at
    face value such a cell is worth thousands of ordinary ones and would
    decide the posterior on its own.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    error = np.full((N, N), 2.0)
    error[0, 0] = 1e-6
    path = write_observations_with_error(tmp_path / "dh_smith_g5000m_test.nc", error)
    obs = load_observations(path, 0.1, 0.5, max_error=None)
    assert float(obs[ERROR_VAR].isel(time=0)[0, 0]) == pytest.approx(0.5)
    # Everything already above the floor is left alone.
    assert float(obs[ERROR_VAR].isel(time=0)[1, 1]) == pytest.approx(2.0)


def test_load_observations_drops_hopeless_cells(tmp_path: Path):
    """
    Remove cells whose uncertainty is past being informative.

    The likelihood mutes them on its own, but the plain RMSE and MAE have no
    such defence, and in the real archive those same cells carry the most
    extreme values in the field -- so left in they would dominate exactly the
    statistics that have no way to discount them.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    error = np.full((N, N), 2.0)
    error[0, :] = 5000.0
    path = write_observations_with_error(tmp_path / "dh_smith_g5000m_test.nc", error)

    obs = load_observations(path, 0.1, 0.5, max_error=1000.0)
    assert np.isnan(obs["dh"].isel(time=0).values[0, :]).all()
    assert np.isfinite(obs["dh"].isel(time=0).values[1:, :]).all()

    # Opting out keeps them.
    kept = load_observations(path, 0.1, 0.5, max_error=None)
    assert np.isfinite(kept["dh"].isel(time=0).values).all()


def test_load_observations_falls_back_when_there_is_no_error(case):
    """
    Synthesise an uncertainty only for a product that ships none.

    Parameters
    ----------
    case : tuple
        Run directory, observation directory and observed files.
    """
    _, _, observed = case
    obs = load_observations(observed[0], 0.1, 0.5, max_error=None)
    assert ERROR_VAR in obs
    expected = np.maximum(0.1 * abs(obs["dh"]), 0.5)
    np.testing.assert_allclose(obs[ERROR_VAR].values, expected.values, equal_nan=True)


def _member(**variables) -> xr.Dataset:
    """
    A one-record member file carrying the named fields.

    Parameters
    ----------
    **variables : numpy.ndarray or float
        Field name to constant value.

    Returns
    -------
    xarray.Dataset
        The member.
    """
    x, y = grid(4, 1000.0)
    coords = {"time": [cftime.DatetimeGregorian(2019, 1, 1)], "y": y, "x": x}
    data = {name: (("time", "y", "x"), np.full((1, 4, 4), value, dtype="float32")) for name, value in variables.items()}
    ds = xr.Dataset(data, coords=coords)
    ds["time_bnds"] = (("time", "bnds"), np.array([[cftime.DatetimeGregorian(2003, 1, 1), coords["time"][0]]]))
    ds["mapping"] = ((), np.int8(0))
    return ds


def test_simulated_variable_prefers_thickness():
    """
    Take ``lithk`` when the run reports it.

    It is the thickness change the observations measure; ``usurf`` beside it
    is a different quantity and must not win by being first alphabetically or
    first in the file.
    """
    assert simulated_variable(_member(usurf=1.0, lithk=2.0)) == "lithk"
    assert SIM_VARS[0] == "lithk"


def test_simulated_variable_accepts_what_a_plain_run_offers():
    """
    Fall back to ``usurf`` for a run that reports nothing else.

    A non-submission run writes surface elevation, and comparing it is better
    than refusing -- provided the caller is told they are different things.
    """
    assert simulated_variable(_member(usurf=1.0)) == "usurf"


def test_simulated_variable_ignores_the_bookkeeping_variables():
    """
    Do not mistake ``time_bnds`` or ``mapping`` for the field.

    They sit in every file and would otherwise make a single-field file look
    ambiguous.
    """
    assert simulated_variable(_member(dhdt=1.0)) == "dhdt"


def test_simulated_variable_honours_an_explicit_choice():
    """
    Let the caller override the preference order.
    """
    assert simulated_variable(_member(usurf=1.0, lithk=2.0), "usurf") == "usurf"


def test_simulated_variable_rejects_a_field_that_is_not_there():
    """
    Name what is available rather than failing further down.
    """
    with pytest.raises(ValueError, match="'orog' not among"):
        simulated_variable(_member(lithk=1.0), "orog")


def test_simulated_variable_refuses_to_guess_between_several():
    """
    Ask rather than pick when the file carries several plausible fields.

    Choosing silently would put an arbitrary quantity into the likelihood.
    """
    with pytest.raises(ValueError, match="cannot tell which"):
        simulated_variable(_member(sftgif=1.0, velsurf_mag=2.0))


def test_load_ensemble_reads_a_usurf_only_run(tmp_path: Path):
    """
    Open a plain run's dh file, which carries no ``lithk`` at all.

    This is what a non-submission OCX run produces, and it used to fail
    outright.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _member(usurf=-2.0).to_netcdf(run_dir / "dh_usurf_spatial_g900m_id_OCX_ocx_prescribed_1990-01-01_2025-01-01.nc")
    sim = load_ensemble(find_member_files(run_dir))
    assert "dh" in sim
    assert float(sim["dh"].isel({MEMBER_DIM: 0}).max()) == pytest.approx(-2.0)


def test_cli_does_not_need_a_separator_before_the_positionals(tmp_path: Path, case, monkeypatch):
    """
    Reach ``RUN_DIR`` and ``OUTPUT_PATH`` without a ``--`` in the way.

    A greedy ``nargs="+"`` on ``--observations`` swallowed both positionals,
    so the command could only be run with a separator -- and ``-h`` landed
    inside the observation list instead of printing help.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    case : tuple
        Run directory, observation directory and observed files.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to silence the log file handler.
    """
    run_dir, _, observed = case
    out = tmp_path / "cli"
    monkeypatch.setattr("pism_terra.ismip7.greenland.importance_sampling.setup_logging", lambda *_: None)
    assert (
        main(
            [
                "--observations",
                str(observed[0]),
                "--n-samples",
                "100",
                "--n-boot",
                "10",
                "--fudge-factors",
                "1,5",
                str(run_dir),
                str(out),
            ]
        )
        == 0
    )
    written = pd.read_csv(out / "importance_sampling_summary.csv")
    assert sorted(set(written["fudge_factor"])) == [1.0, 5.0]
