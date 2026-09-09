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
Tests for :mod:`pism_terra.lcurve`.

Builds a small synthetic inversion ensemble — the variables the L-curve
needs (``vel_misfit_weight``, ``inv_residual``, ``J_design``) plus a
``pism_config`` scalar carrying the regularization settings — and covers:

- ``data_misfit`` and ``model_norm`` against hand-computed values.
- ``corner`` on a curve with a known corner, and its short-curve guard.
- ``collect_lcurve`` skipping incomplete members, and erroring when none is
  usable.
- ``plot_lcurve`` writing a figure and picking one corner per parameter group.
- the ``main`` entry point end to end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from pism_terra import lcurve
from pism_terra.lcurve import (
    collect_lcurve,
    corner,
    data_misfit,
    main,
    model_norm,
    plot_lcurve,
)

PENALTY = "inverse.tikhonov.penalty_weight"
CH1 = "inverse.design.cH1"


def write_member(
    path: Path,
    penalty_weight: float,
    residual: np.ndarray,
    j_design: np.ndarray,
    *,
    weight: np.ndarray | None = None,
    cH1: float = 1.0,
    complete: bool = True,
    design: str = "tauc",
    param: str = "exp",
) -> Path:
    """
    Write a synthetic inversion output file.

    Parameters
    ----------
    path : pathlib.Path
        File to write.
    penalty_weight : float
        Value stored as ``inverse.tikhonov.penalty_weight``.
    residual : numpy.ndarray
        2D ``inv_residual`` field, m/yr.
    j_design : numpy.ndarray
        ``J_design`` iteration history.
    weight : numpy.ndarray or None, optional
        2D ``vel_misfit_weight`` field; ``None`` (default) weights every cell
        equally.
    cH1 : float, optional
        Value stored as ``inverse.design.cH1``.
    complete : bool, optional
        If False, the inversion diagnostics are left out, as in a member that
        is still running.
    design : str, optional
        Design variable the run inverted for; its field and ``<design>_prior``
        are written, as ``pismi`` does.
    param : str, optional
        Value of ``inverse.design.param``. Only ``"ident"`` makes the model
        norm carry the design variable's units.

    Returns
    -------
    pathlib.Path
        The path written, for convenience.
    """
    attrs = {
        PENALTY: np.float64(penalty_weight),
        CH1: np.float64(cH1),
        "inverse.design.param": param,
        "stress_balance.blatter.Glen_exponent": np.float64(3.0),
    }
    data: dict[str, Any] = {"pism_config": ((), np.int8(0), attrs)}
    for name in (design, f"{design}_prior", "zeta_inv"):
        data[name] = (("y", "x"), np.ones_like(residual))
    if complete:
        w = np.ones_like(residual) if weight is None else weight
        data["vel_misfit_weight"] = (("time", "y", "x"), w[np.newaxis, ...])
        data["inv_residual"] = (("time", "y", "x"), residual[np.newaxis, ...])
        data["J_design"] = (("inv_iter",), j_design)
    xr.Dataset(data).to_netcdf(path)
    return path


@pytest.fixture(name="ensemble")
def fixture_ensemble(tmp_path: Path) -> list[Path]:
    """
    Four-member sweep in ``penalty_weight``, plus one incomplete member.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    list of pathlib.Path
        The member files, in an order that is *not* sorted by penalty weight.
    """
    files = []
    # Misfit rises and model norm falls with the penalty weight, with the
    # sharp bend between 1 and 10 that makes an L.
    for penalty, misfit, norm in [(0.1, 10.0, 8.0), (1.0, 11.0, 4.0), (10.0, 40.0, 3.0), (100.0, 70.0, 2.5)]:
        path = tmp_path / f"inv_g100m_uq_{penalty:g}.nc"
        files.append(
            write_member(
                path,
                penalty,
                residual=np.full((4, 5), misfit),
                j_design=np.array([100.0, norm**2]),
            )
        )
    files.insert(
        0, write_member(tmp_path / "inv_g100m_uq_running.nc", 1000.0, np.zeros((4, 5)), np.zeros(2), complete=False)
    )
    return files


def test_data_misfit_weighted() -> None:
    """
    Weight the RMS so cells outside the misfit area do not enter it.

    Returns
    -------
    None
        Asserts only.
    """
    residual = np.array([[3.0, 4.0], [1000.0, 1000.0]])
    weight = np.array([[1.0, 1.0], [0.0, 0.0]])
    ds = xr.Dataset(
        {
            "vel_misfit_weight": (("time", "y", "x"), weight[np.newaxis, ...]),
            "inv_residual": (("time", "y", "x"), residual[np.newaxis, ...]),
        }
    )
    assert data_misfit(ds) == pytest.approx(np.sqrt((9.0 + 16.0) / 2.0))


def test_model_norm_uses_last_iteration() -> None:
    """
    The model norm is the square root of the converged ``J_design``.

    Returns
    -------
    None
        Asserts only.
    """
    ds = xr.Dataset({"J_design": (("inv_iter",), np.array([100.0, 25.0, 9.0]))})
    assert model_norm(ds) == pytest.approx(3.0)


def test_corner_picks_the_bend() -> None:
    """
    ``corner`` finds the maximum-curvature point of an L-shaped curve.

    Returns
    -------
    None
        Asserts only.
    """
    norm = np.array([1.0, 10.0, 100.0, 101.0, 102.0])
    misfit = np.array([100.0, 99.0, 98.0, 10.0, 1.0])
    assert corner(norm, misfit) == 2


def test_corner_needs_three_points() -> None:
    """
    Fewer than three points have no interior point, hence no corner.

    Returns
    -------
    None
        Asserts only.
    """
    assert corner(np.array([1.0, 2.0]), np.array([2.0, 1.0])) is None


def test_collect_lcurve_skips_incomplete(ensemble: list[Path]) -> None:
    """
    Drop members without inversion diagnostics, and sort the rest.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.

    Returns
    -------
    None
        Asserts only.
    """
    df = collect_lcurve(ensemble, [PENALTY])
    assert list(df["penalty_weight"]) == [0.1, 1.0, 10.0, 100.0]
    assert list(df["M"]) == pytest.approx([10.0, 11.0, 40.0, 70.0])
    assert list(df["N"]) == pytest.approx([8.0, 4.0, 3.0, 2.5])
    assert "inv_g100m_uq_running.nc" not in set(df["file"])


def test_collect_lcurve_unknown_parameter(ensemble: list[Path]) -> None:
    """
    A parameter absent from ``pism_config`` leaves no usable member.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.

    Returns
    -------
    None
        Asserts only.
    """
    with pytest.raises(ValueError, match="yielded an L-curve point"):
        collect_lcurve(ensemble, ["surface.pdd.factor_ice"])


def test_plot_lcurve_writes_figure(ensemble: list[Path], tmp_path: Path) -> None:
    """
    The single-parameter plot is written and its corner is the bend.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None
        Asserts only.
    """
    df = collect_lcurve(ensemble, [PENALTY])
    output_file = tmp_path / "figures" / "lcurve.png"
    corners = plot_lcurve(df, [PENALTY], output_file, log=True)
    assert output_file.exists()
    assert list(corners["penalty_weight"]) == [1.0]


def test_plot_lcurve_groups_second_parameter(tmp_path: Path) -> None:
    """
    A second parameter splits the ensemble into one curve, and corner, each.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None
        Asserts only.
    """
    files = []
    for cH1 in (0.5, 1.0):
        for penalty, misfit, norm in [(0.1, 10.0, 8.0), (1.0, 11.0, 4.0), (10.0, 40.0, 3.0)]:
            files.append(
                write_member(
                    tmp_path / f"inv_{cH1:g}_{penalty:g}.nc",
                    penalty,
                    residual=np.full((4, 5), misfit * cH1),
                    j_design=np.array([100.0, norm**2]),
                    cH1=cH1,
                )
            )
    df = collect_lcurve(files, [PENALTY, CH1])
    corners = plot_lcurve(df, [PENALTY, CH1], tmp_path / "lcurve.pdf")
    assert (tmp_path / "lcurve.pdf").exists()
    assert list(corners["cH1"]) == [0.5, 1.0]
    assert list(corners["penalty_weight"]) == [1.0, 1.0]


def test_main_end_to_end(ensemble: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The CLI writes both the figure and the table next to it.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to set ``sys.argv``.

    Returns
    -------
    None
        Asserts only.
    """
    output_file = tmp_path / "out" / "lcurve.png"
    monkeypatch.setattr(
        "sys.argv",
        ["pism-inverse-lcurve", "--parameters", PENALTY, "-o", str(output_file)] + [str(f) for f in ensemble],
    )
    main()
    assert output_file.exists()
    assert output_file.with_suffix(".csv").exists()


def test_design_variable_detection(tmp_path: Path) -> None:
    """
    Read the inverted field off the variables the run wrote.

    A ``tauc`` inversion of a Blatter forward problem can carry a prescribed
    ``hardav`` alongside, so the plain field is only the last resort; an
    alternating run names both phases and is reported as neither.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None
        Asserts only.
    """
    residual = np.full((3, 3), 5.0)
    j_design = np.array([10.0, 4.0])
    tauc = write_member(tmp_path / "tauc.nc", 1.0, residual, j_design)
    hardav = write_member(tmp_path / "hardav.nc", 1.0, residual, j_design, design="hardav")
    with xr.open_dataset(tauc) as ds:
        assert lcurve.design_variable(ds) == "tauc"
    with xr.open_dataset(hardav) as ds:
        assert lcurve.design_variable(ds) == "hardav"

    # A tauc run that also carries a prescribed hardav field.
    with xr.open_dataset(tauc) as ds:
        mixed = ds.load()
    mixed["hardav"] = mixed["tauc"]
    assert lcurve.design_variable(mixed) == "tauc"

    # An alternating co-inversion names both phases.
    alternating = mixed.rename({"zeta_inv": "zeta_inv_tauc"})
    alternating["zeta_inv_hardav"] = alternating["zeta_inv_tauc"]
    assert lcurve.design_variable(alternating) is None


def test_model_norm_units_follow_the_parameterization(tmp_path: Path) -> None:
    """
    The norm carries the field's units only under ``param = "ident"``.

    ``J_design`` is evaluated on the parameterized zeta, so the default
    ``exp`` parameterization leaves the norm dimensionless. ``hardav``'s units
    carry the Glen exponent.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None
        Asserts only.
    """
    residual, j_design = np.full((3, 3), 5.0), np.array([10.0, 4.0])
    expectations = [
        ("exp_tauc", "tauc", "exp", None),
        ("exp_hardav", "hardav", "exp", None),
        ("ident_tauc", "tauc", "ident", "Pa"),
        ("ident_hardav", "hardav", "ident", "Pa s^(1/3)"),
    ]
    for name, design, param, expected in expectations:
        path = write_member(tmp_path / f"{name}.nc", 1.0, residual, j_design, design=design, param=param)
        with xr.open_dataset(path) as ds:
            assert lcurve.model_norm_units(ds, lcurve.design_variable(ds)) == expected, name


def test_norm_axis_label_names_the_design_variable(ensemble: list[Path]) -> None:
    """
    The abscissa names the inverted field and how the norm is scaled.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.

    Returns
    -------
    None
        Asserts only.
    """
    df = collect_lcurve(ensemble, [PENALTY])
    label = lcurve.norm_axis_label(df)
    assert "tauc" in label and "dimensionless" in label

    # A mixed ensemble cannot claim either field.
    mixed = df.copy()
    mixed.loc[0, "design"] = "hardav"
    assert "tauc" not in lcurve.norm_axis_label(mixed)
