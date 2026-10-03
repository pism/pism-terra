"""Sensitivity indices of ensemble time series, and the project-level CLI."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pism_terra import sensitivity
from pism_terra.glacier import sensitivity_indices as gsi

PARAMS = ["surface.pdd.factor_ice", "surface.pdd.factor_snow", "surface.pdd.refreeze"]


def _ensemble(n: int = 120, seed: int = 0) -> tuple[xr.DataArray, pd.DataFrame]:
    """
    An ensemble whose response depends mostly on the first parameter, and on the second late on.

    Parameters
    ----------
    n : int, optional
        Members.
    seed : int, optional
        Random seed.

    Returns
    -------
    tuple
        ``(response, uq_df)``: response on ``(uq_id, time, glacier)`` with two glaciers.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n, 3))
    time = pd.date_range("2000-01-01", periods=6, freq="YS")
    t = np.linspace(0, 1, time.size)
    # glacier A: x0 dominates throughout; glacier B: x1 takes over with time
    ya = 5 * X[:, [0]] + 0.3 * rng.normal(size=(n, 1)) + 0 * t
    yb = 5 * (1 - t) * X[:, [0]] + 5 * t * X[:, [1]] + 0.3 * rng.normal(size=(n, 1))
    da = xr.DataArray(
        np.stack([ya, yb], axis=-1),
        dims=["uq_id", "time", "glacier"],
        coords={"uq_id": [str(i) for i in range(n)], "time": time, "glacier": ["A", "B"]},
        name="ice_mass",
    )
    uq = pd.DataFrame(X, columns=PARAMS, index=pd.Index([str(i) for i in range(n)], name="uq_id"))
    return da, uq


def test_indices_identify_the_driving_parameter() -> None:
    """S1 and delta rank the true driver first, and the ranking follows the drift in time."""
    da, uq = _ensemble()
    ds = sensitivity.sensitivity_indices(da, uq, num_resamples=20, n_jobs=1)
    assert ds.S1.dims == ("glacier", "time", "uq_var")
    assert list(ds.uq_var.values) == PARAMS
    a = ds.sel(glacier="A")
    assert (a.S1.sel(uq_var=PARAMS[0]) > 0.6).all()
    assert (a.S1.sel(uq_var=PARAMS[2]) < 0.2).all()
    assert (a.delta.sel(uq_var=PARAMS[0]) > a.delta.sel(uq_var=PARAMS[2])).all()
    b = ds.sel(glacier="B")
    assert float(b.S1.sel(uq_var=PARAMS[0]).isel(time=0)) > float(b.S1.sel(uq_var=PARAMS[1]).isel(time=0))
    assert float(b.S1.sel(uq_var=PARAMS[1]).isel(time=-1)) > float(b.S1.sel(uq_var=PARAMS[0]).isel(time=-1))
    assert (ds.S1_conf >= 0).all()
    assert ds.attrs["n_members"] == 120


def test_indices_are_nan_where_the_analysis_is_impossible() -> None:
    """Too few members, a constant response, or NaN members give NaN rather than an error."""
    da, uq = _ensemble(n=4)
    ds = sensitivity.sensitivity_indices(da, uq, num_resamples=5, n_jobs=1)
    assert ds.S1.isnull().all()
    da, uq = _ensemble(n=60)
    flat = da.copy(data=np.ones(da.shape))
    assert sensitivity.sensitivity_indices(flat, uq, num_resamples=5, n_jobs=1).delta.isnull().all()
    holed = da.where(da.uq_id != "3")
    out = sensitivity.sensitivity_indices(holed, uq, num_resamples=5, n_jobs=1)
    assert not out.S1.isnull().any()
    with pytest.raises(ValueError, match="no member"):
        sensitivity.sensitivity_indices(da.assign_coords(uq_id=[f"x{i}" for i in range(60)]), uq, n_jobs=1)


def test_process_pool_matches_serial() -> None:
    """The pooled computation reproduces the serial one bit for bit."""
    da, uq = _ensemble(n=40)
    serial = sensitivity.sensitivity_indices(da, uq, num_resamples=5, n_jobs=1)
    pooled = sensitivity.sensitivity_indices(da, uq, num_resamples=5, n_jobs=2)
    xr.testing.assert_identical(serial, pooled)


def test_resample_and_summary() -> None:
    """Monthly input averages to yearly; the summary has one row per glacier and parameter."""
    time = pd.date_range("2000-01-01", periods=24, freq="MS")
    da = xr.DataArray(np.arange(24.0)[None, :, None], dims=["uq_id", "time", "glacier"], coords={"time": time})
    yearly = sensitivity.resample_response(da, "yearly")
    assert yearly.sizes["time"] == 2 and float(yearly.isel(time=0, uq_id=0, glacier=0)) == pytest.approx(5.5)
    assert sensitivity.resample_response(da, "none") is da
    with pytest.raises(ValueError, match="freq"):
        sensitivity.resample_response(da, "weekly")
    ens, uq = _ensemble(n=40)
    ds = sensitivity.sensitivity_indices(ens, uq, num_resamples=5, n_jobs=1)
    table = sensitivity.summarize_indices(ds, "glacier", last_years=2)
    assert len(table) == 6 and set(table.columns) >= {
        "glacier",
        "parameter",
        "S1_mean",
        "S1_last",
        "delta_mean",
        "delta_last",
    }


@pytest.fixture(name="project")
def fixture_project(tmp_path: Path) -> Path:
    """
    A project with one complex: 12 members of scalar_C and scalar_G files plus uq.csv.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Project directory.
    """
    rgi = "RGI2000-v7.0-C-01-00001"
    da, uq = _ensemble(n=12, seed=3)
    out = tmp_path / rgi / "output"
    (out / "processed_scalar").mkdir(parents=True)
    uq.reset_index().rename(columns={"uq_id": "uq"}).to_csv(out / "uq.csv", index=False)
    time = pd.date_range("2000-01-16", periods=6, freq="MS")  # monthly, like the real files
    for m in da.uq_id.values:
        for kind, names in (("C", [rgi]), ("G", ["RGI2000-v7.0-G-01-00010", "RGI2000-v7.0-G-01-00011"])):
            member = da.sel(uq_id=m).drop_vars("uq_id").isel(glacier=slice(0, len(names)))
            ds = xr.Dataset(
                {"ice_mass": (("time", "glacier_id"), member.values.astype(float), {"units": "kg"})},
                coords={"time": time, "glacier_id": np.arange(len(names)), "glacier_id_name": ("glacier_id", names)},
            )
            ds.to_netcdf(out / "processed_scalar" / f"scalar_{kind}_g500m_{rgi}_id_0_uq_{m}_2000-01-01_2001-01-01.nc")
    return tmp_path


def test_project_pipeline_writes_the_usgs_like_layout(project: Path) -> None:
    """
    Both kinds are analyzed per complex, figures are per glacier, and the summary is written.

    Parameters
    ----------
    project : Path
        Synthetic project fixture.
    """
    out = project / "sens"
    summary = gsi.run_pipeline(project, output_path=out, num_resamples=5, n_jobs=1, min_members=8, freq="none")
    rgi = "RGI2000-v7.0-C-01-00001"
    assert set(summary["kind"]) == {"C", "G"}
    assert set(summary["glacier"]) == {rgi, "RGI2000-v7.0-G-01-00010", "RGI2000-v7.0-G-01-00011"}
    assert (out / rgi / "sensitivity_C_ice_mass.nc").is_file()
    assert (out / rgi / f"sensitivity_C_ice_mass_{rgi}.png").is_file()
    assert (out / rgi / "sensitivity_G_ice_mass_RGI2000-v7.0-G-01-00011.png").is_file()
    assert (out / "sensitivity_indices_summary.csv").is_file()
    ds = xr.open_dataset(out / rgi / "sensitivity_G_ice_mass.nc")
    assert ds.attrs["target"] == "ice_mass" and ds.sizes["glacier"] == 2

    with pytest.raises(FileNotFoundError, match="processed members"):
        gsi.run_pipeline(project, output_path=project / "none", min_members=50)


def test_file_cli_finds_uq_csv_above_the_files(project: Path) -> None:
    """
    The generic CLI locates uq.csv by walking up from the files and names outputs by kind.

    Parameters
    ----------
    project : Path
        Synthetic project fixture.
    """
    rgi = "RGI2000-v7.0-C-01-00001"
    files = sorted((project / rgi / "output" / "processed_scalar").glob("scalar_C_*.nc"))
    out = project / "cli"
    assert (
        sensitivity.cli(
            [str(f) for f in files]
            + ["--output-path", str(out), "--n-resamples", "5", "--n-jobs", "1", "--freq", "none"]
        )
        == 0
    )
    assert (out / "sensitivity_C_ice_mass.nc").is_file() and (out / "sensitivity_C_ice_mass.csv").is_file()
