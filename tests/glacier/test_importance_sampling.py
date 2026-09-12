"""End-to-end test of pism-glacier-importance-sampling on a synthetic project."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pism_terra.glacier import importance_sampling as gis

GLACIERS = ["RGI2000-v7.0-C-01-00001", "RGI2000-v7.0-C-01-00002"]
MEMBERS = ["0", "1", "2", "3"]


def _grid(n: int, step: float) -> dict[str, tuple[str, np.ndarray, dict[str, str]]]:
    """
    Build projected ``x``/``y`` coordinates with units attributes.

    Parameters
    ----------
    n : int
        Points per axis.
    step : float
        Spacing in metres.

    Returns
    -------
    dict
        Coordinates for :class:`xarray.Dataset`.
    """
    axis = np.arange(n) * step
    return {"x": ("x", axis, {"units": "m"}), "y": ("y", axis, {"units": "m"})}


@pytest.fixture(name="project")
def fixture_project(tmp_path: Path) -> Path:
    """
    Write two glaciers with four ensemble members each, plus observations and uq.csv.

    Member 1 matches the observations, the others are offset by 1, 2 and 3 m.
    The second glacier misses member 3, so the joint posterior has three members.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Project directory.
    """
    rng = np.random.default_rng(0)
    for g, rgi_id in enumerate(GLACIERS):
        obs_dir = tmp_path / rgi_id / "input"
        obs_dir.mkdir(parents=True)
        truth = -2.0 + 0.3 * rng.normal(size=(20, 20))
        truth[:4, :] = np.nan  # holes in the observations
        landice = np.ones((20, 20), dtype=np.int8)
        obs = xr.Dataset(
            {
                "dh": (("y", "x"), truth, {"units": "m"}),
                "dh_err": (("y", "x"), np.full((20, 20), 0.5), {"units": "m"}),
                "landice": (("y", "x"), landice),
            },
            coords=_grid(20, 100.0),
        )
        obs.to_netcdf(obs_dir / f"obs_{rgi_id}.nc")

        dh_dir = tmp_path / rgi_id / "output" / "dh"
        dh_dir.mkdir(parents=True)
        members = MEMBERS if g == 0 else MEMBERS[:-1]
        for member in members:
            field = truth[::2, ::2] + (int(member) - 1) * 1.0
            sim = xr.Dataset(
                {"usurf": (("time", "y", "x"), field[None], {"units": "m"})},
                coords={"time": [np.datetime64("2010-01-01")], **_grid(10, 200.0)},
            )
            sim.to_netcdf(dh_dir / f"dh_{rgi_id}_id_0_uq_{member}_2000-01-01_2020-01-01.nc")
        pd.DataFrame(
            {
                "uq": [int(m) for m in MEMBERS],
                "surface.pdd.factor_ice": np.linspace(0.004, 0.008, 4),
                "surface.pdd.refreeze": [0.2, 0.4, 0.6, 0.8],
            }
        ).to_csv(tmp_path / rgi_id / "output" / "uq.csv", index=False)
    return tmp_path


def test_pipeline_weights_members_and_combines_glaciers(project: Path) -> None:
    """
    The matching member wins per glacier and jointly, and the USGS-like layout is written.

    Parameters
    ----------
    project : Path
        Synthetic project fixture.
    """
    out = project / "results"
    summary = gis.run_pipeline(project, output_path=out, fudge_factors=(1.0, 10.0), n_samples=500, n_boot=20)

    assert set(summary["rgi_id"]) == set(GLACIERS) | {"joint"}
    for rgi_id in GLACIERS:
        rows = summary[(summary.rgi_id == rgi_id) & (summary.fudge_factor == 1.0)]
        assert rows["top_uq_id"].item() == "1"
        assert rows["best_rmse_uq_id"].item() == "1"
        # four of twenty rows are holes, and linear interpolation drops the rows and columns
        # next to them and beyond the coarser ensemble grid
        assert 0.6 < rows["coverage"].item() < 0.8
        glacier_dir = out / rgi_id
        assert (glacier_dir / f"importance_sampling_{rgi_id}.nc").is_file()
        assert (glacier_dir / "importance_usurf.csv").is_file()
        assert (glacier_dir / "importance_usurf_ff_1.png").is_file()
        assert (glacier_dir / "importance_usurf_tied.png").is_file()
        assert (glacier_dir / "importance_usurf_best_rmse.png").is_file()
        ds = xr.open_dataset(glacier_dir / f"importance_sampling_{rgi_id}.nc")
        assert set(ds["variable"].values) == {"usurf"}
        assert "surface.pdd.factor_ice" in ds
        np.testing.assert_allclose(ds["weights"].sum("uq_id"), 1.0)

    assert set(summary["reduction"]) == {"blocks"}
    joint = summary[(summary.rgi_id == "joint") & (summary.fudge_factor == 1.0)]
    assert joint["top_uq_id"].item() == "1"
    assert joint["n_members"].item() == 3  # member 3 is missing from the second glacier
    assert (out / "joint" / "importance_sampling_joint.nc").is_file()
    assert (out / "joint" / "importance_joint_usurf_ff_1.png").is_file()
    assert (out / "importance_sampling_summary.csv").is_file()
    long = pd.read_csv(out / "importance_sampling_weights.csv")
    assert set(long["rgi_id"]) == set(GLACIERS) | {"joint"}


def test_cli_returns_zero_and_honours_no_bootstrap(project: Path) -> None:
    """
    The console script runs with a data path separate from the run directory.

    Parameters
    ----------
    project : Path
        Synthetic project fixture.
    """
    out = project / "cli_out"
    assert (
        gis.cli(
            [
                str(project),
                "--data-path",
                str(project),
                "--output-path",
                str(out),
                "--no-bootstrap",
                "--n-boot",
                "5",
                "--fudge-factors",
                "3",
                "--reduction",
                "mean",
                "--acf-threshold",
                "0.1",
            ]
        )
        == 0
    )
    summary = pd.read_csv(out / "importance_sampling_summary.csv")
    assert "best_rmse_uq_id" not in summary.columns
    assert list(summary["fudge_factor"].unique()) == [3.0]
    assert set(summary["reduction"]) == {"mean"}
    assert (summary["block_size"].dropna() >= 1).all()
    assert (summary["acf_threshold"].dropna() == 0.1).all()


def test_parse_variable_defaults_the_uncertainty() -> None:
    """A two-field spec gets ``<obs>_err``; a bad spec raises."""
    assert gis.parse_variable("usurf:dh") == ("usurf", "dh", "dh_err")
    assert gis.parse_variable("v:speed:speed_sigma") == ("v", "speed", "speed_sigma")
    with pytest.raises(ValueError):
        gis.parse_variable("usurf")
