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
Tests for the ``pism-kitp-calibrate`` command line.

The driver used to read one hard-wired observation file and one hard-wired
dEBM ensemble under ``--data-path``. It now takes the observations with
``--obs-path`` and the ensemble's files as arguments, tells the surface model
from the members' config, and sizes grid cells from the grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pism_terra.kitp import calibrate

#: Scale of each member's balance against the observed one; member 3 is exact.
SCALES = (0.6, 0.8, 0.95, 1.0, 1.05, 1.3, 1.6, 2.0)


def _fields(scale: float) -> dict:
    """
    Build a year of monthly balance fields, scaled.

    Parameters
    ----------
    scale : float
        Factor on the spatial pattern of the balance.

    Returns
    -------
    dict
        Data variables for an :class:`xarray.Dataset`.
    """
    x = np.arange(40) * 1200.0
    yy, xx = np.meshgrid(x, x, indexing="ij")
    base = 500 * np.sin(xx / 8000) * np.cos(yy / 9000) - 200
    cmb = np.stack([base * scale + 10 * month for month in range(12)])
    melt = np.abs(cmb) * 0.5 + 100
    attrs = {"units": "kg m^-2 year^-1"}
    return {
        "climatic_mass_balance": (("time", "y", "x"), cmb, attrs),
        "surface_melt_flux": (("time", "y", "x"), melt, attrs),
        "surface_runoff_flux": (("time", "y", "x"), 0.8 * melt, attrs),
        "surface_accumulation_flux": (("time", "y", "x"), cmb - melt, attrs),
    }


def _coords(n_time: int = 12) -> dict:
    """
    Build the coordinates of the stand-in files.

    Parameters
    ----------
    n_time : int, optional
        Number of monthly steps.

    Returns
    -------
    dict
        ``time``, ``x`` and ``y``.
    """
    x = np.arange(40) * 1200.0
    time = ("time", np.arange(n_time) * 30.0, {"units": "days since 0001-01-01", "calendar": "365_day"})
    return {"time": time, "x": x, "y": x}


@pytest.fixture(name="ensemble")
def fixture_ensemble(tmp_path: Path) -> tuple[Path, list[Path]]:
    """
    Write an observation file and an eight-member dEBM ensemble.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.

    Returns
    -------
    tuple
        The observation file and the members' files.
    """
    obs = tmp_path / "spatial_GIS_HIRHAM5-ERA5_YMM_1990_2019_v4.nc"
    fields = _fields(1.0)
    fields.pop("surface_accumulation_flux")
    xr.Dataset(fields, coords=_coords()).to_netcdf(obs)
    files = []
    for i, scale in enumerate(SCALES):
        ds = xr.Dataset(_fields(scale), coords=_coords())
        ds["pism_config"] = xr.DataArray(
            0,
            attrs={
                "surface.models": "debm_simple",
                "surface.debm_simple.c1": 20.0 + 5 * i,
                "surface.debm_simple.c2": -90.0 + i,
                "surface.debm_simple.refreeze": 0.5,
            },
        )
        files.append(tmp_path / f"spatial_GIS_g1200m_id_X_uq_{i}_0001-01-01_0002-01-01.nc")
        ds.to_netcdf(files[-1])
    return obs, files


def test_the_files_on_the_command_line_are_calibrated(ensemble: tuple[Path, list[Path]], tmp_path: Path, capsys):
    """
    Rank the members given as arguments against the file given with ``--obs-path``.

    Parameters
    ----------
    ensemble : tuple
        The observation file and the members' files.
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    capsys : pytest.CaptureFixture
        Pytest fixture capturing the printout.
    """
    obs, files = ensemble
    out = tmp_path / "out"

    calibrate.main(["--obs-path", str(obs), "--output-path", str(out), "--n-samples", "200", *map(str, files)])

    printed = capsys.readouterr().out
    assert "8 files, 8 members, surface model debm: c1, c2, refreeze" in printed
    rmse = pd.read_csv(out / "debm_climatic_mass_balance_rmse.csv", index_col="exp_id")
    assert rmse.index[0] == 3 and rmse["tied_with_best"].sum() >= 1
    assert {"surface.debm_simple.c1", "surface.debm_simple.c2", "surface.debm_simple.refreeze"} <= set(rmse.columns)
    for name in ("debm_surface_melt_flux_best_rmse.png", "debm_surface_runoff_flux_importance.csv"):
        assert (out / name).is_file()
    # 1200 m cells read off the grid: the exact member matches the observed total.
    totals = [line for line in printed.splitlines() if line.startswith("Obs:")]
    obs_total, sim_total = (float(part.split()[1]) for part in totals[0].split(", "))
    assert sim_total == pytest.approx(obs_total)


def test_the_surface_model_is_read_from_the_members_config():
    """Tell PDD from dEBM, and refuse an ensemble that mixes them."""
    config = pd.DataFrame({"surface.models": ["pdd", "pdd,forcing"]})
    assert calibrate.surface_model_of(config) == "pdd"
    assert calibrate.surface_model_of(pd.DataFrame({"surface.models": ["debm_enhanced"]})) == "debm"
    with pytest.raises(ValueError, match="--surface-model"):
        calibrate.surface_model_of(pd.DataFrame({"surface.models": ["pdd", "debm_simple"]}))


def test_an_ensemble_on_other_time_steps_is_refused(ensemble: tuple[Path, list[Path]], tmp_path: Path):
    """
    Stop when the members and the observations have different numbers of steps.

    Parameters
    ----------
    ensemble : tuple
        The observation file and the members' files.
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    _, files = ensemble
    short = tmp_path / "short_obs.nc"
    fields = {k: (dims, data[:6], attrs) for k, (dims, data, attrs) in _fields(1.0).items()}
    xr.Dataset(fields, coords=_coords(6)).to_netcdf(short)

    with pytest.raises(ValueError, match="12 time steps and the observations 6"):
        calibrate.calibrate(short, files, output_path=tmp_path / "out")
