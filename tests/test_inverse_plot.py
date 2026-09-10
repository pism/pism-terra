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
Tests for :mod:`pism_terra.inverse_plot`.

Builds a small synthetic inversion ensemble carrying the fields the maps
need, and covers:

- ``read_member`` masking the design variable and zeta to the free cells and
  the residual to the misfit area, and skipping unusable files.
- ``shared_limits`` pooling across members, honouring overrides, and its
  guards for log and diverging scales and a fully-masked row.
- ``plot_field`` writing one file per field, named after the variable.
- the ``main`` entry point end to end, including its exit on an empty
  ensemble.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from pism_terra.inverse_plot import (
    FIELDS,
    completed_phases,
    field_items,
    item_key,
    main,
    plot_field,
    read_member,
    shared_limits,
)

PENALTY = "inverse.tikhonov.penalty_weight"


def write_member(
    path: Path,
    penalty_weight: float,
    *,
    design: str = "tauc",
    tauc_free: float = 1.0e5,
    tauc_fixed: float = 1.4e5,
    residual: float = 20.0,
    zeta: float = 1.5,
    complete: bool = True,
    alternating: bool = False,
    completed: str = "c1_hardav",
) -> Path:
    """
    Write a synthetic inversion output with distinguishable masked regions.

    The left half of the domain is the inverted region (``zeta_fixed_mask``
    0, ``vel_misfit_weight`` 1); the right half is fixed and unobserved, and
    carries different values so masking is detectable.

    Parameters
    ----------
    path : pathlib.Path
        File to write.
    penalty_weight : float
        Value stored as ``inverse.tikhonov.penalty_weight``.
    design : str, optional
        Design variable to write, with its ``<design>_prior``.
    tauc_free : float, optional
        Design-variable value on the free (inverted) cells.
    tauc_fixed : float, optional
        Design-variable value on the fixed cells, which masking must drop.
    residual : float, optional
        ``inv_residual`` value on the misfit area; the rest gets twice this,
        which masking must drop.
    zeta : float, optional
        ``zeta_inv`` value on the free cells; the fixed cells get its
        negation, so a sign-preserving diverging scale is testable.
    complete : bool, optional
        If False, the residual is left out, as in a member still running.
    alternating : bool, optional
        Write both phases of a tauc/hardav co-inversion — both fields, both
        priors and a per-phase ``zeta_inv_<design>`` — instead of a
        single-design run's plain ``zeta_inv``.
    completed : str, optional
        Value of the ``pismi_alternation_completed`` stamp, written only for
        an alternating run.

    Returns
    -------
    pathlib.Path
        The path written, for convenience.
    """
    ny, nx = 4, 6
    free = np.zeros((ny, nx), dtype=bool)
    free[:, : nx // 2] = True

    field = np.where(free, tauc_free, tauc_fixed)[np.newaxis, ...]
    zeta_field = np.where(free, zeta, -zeta)[np.newaxis, ...]
    data: dict[str, Any] = {
        "pism_config": ((), np.int8(0), {PENALTY: np.float64(penalty_weight)}),
        "zeta_fixed_mask": (("time", "y", "x"), np.where(free, 0.0, 1.0)[np.newaxis, ...]),
        "vel_misfit_weight": (("time", "y", "x"), free.astype(float)[np.newaxis, ...]),
    }
    designs = ("tauc", "hardav") if alternating else (design,)
    for name in designs:
        data[name] = (("time", "y", "x"), field)
        data[f"{name}_prior"] = (("time", "y", "x"), np.full((1, ny, nx), tauc_fixed))
        if alternating:
            data[f"zeta_inv_{name}"] = (("time", "y", "x"), zeta_field)
    if not alternating:
        data["zeta_inv"] = (("time", "y", "x"), zeta_field)
    if complete:
        data["inv_residual"] = (("time", "y", "x"), np.where(free, residual, 2 * residual)[np.newaxis, ...])
    coords = {"x": np.arange(nx, dtype=float) * 100.0, "y": np.arange(ny, dtype=float) * 100.0}
    ds = xr.Dataset(data, coords=coords)
    if alternating:
        ds.attrs["pismi_alternation_completed"] = completed
    ds.to_netcdf(path)
    return path


@pytest.fixture(name="ensemble")
def fixture_ensemble(tmp_path: Path) -> list[Path]:
    """
    Three-member sweep plus one member that has not written its residual.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    list of pathlib.Path
        The member files, deliberately not ordered by penalty weight.
    """
    files = [
        write_member(tmp_path / f"inv_{p:g}.nc", p, tauc_free=t, residual=r, zeta=z)
        for p, t, r, z in [(10.0, 3.0e5, 5.0, 3.0), (0.1, 1.0e5, 40.0, 0.5), (1.0, 2.0e5, 12.0, 1.5)]
    ]
    files.append(write_member(tmp_path / "inv_running.nc", 100.0, complete=False))
    return files


def test_read_member_masks_to_the_inverted_region(ensemble: list[Path]) -> None:
    """
    Keep only the free cells, and the residual's fit cells.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.

    Returns
    -------
    None
        Asserts only.
    """
    member = read_member(ensemble[0], [PENALTY], list(FIELDS))
    assert member is not None
    assert member["penalty_weight"] == 10.0
    # Every field keeps the same left half and drops the same right half.
    for key in ("design:tauc", "zeta:tauc", "residual"):
        assert np.isnan(member[key]).sum() == member[key].size // 2, key
    assert np.nanmin(member["design:tauc"]) == np.nanmax(member["design:tauc"]) == 3.0e5
    assert np.nanmin(member["zeta:tauc"]) == np.nanmax(member["zeta:tauc"]) == 3.0
    assert np.nanmin(member["residual"]) == np.nanmax(member["residual"]) == 5.0
    assert member["designs"] == ["tauc"]


def test_read_member_reads_only_what_is_asked(ensemble: list[Path]) -> None:
    """
    Skip the fields that will not be plotted.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.

    Returns
    -------
    None
        Asserts only.
    """
    member = read_member(ensemble[0], [PENALTY], ["zeta"])
    assert member is not None
    assert "zeta:tauc" in member
    assert "design:tauc" not in member and "residual" not in member


def test_read_member_unmasked_keeps_everything(ensemble: list[Path]) -> None:
    """
    Skip masking entirely when asked, so the prior stays visible.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.

    Returns
    -------
    None
        Asserts only.
    """
    member = read_member(ensemble[0], [PENALTY], list(FIELDS), mask=False)
    assert member is not None
    assert not np.isnan(member["design:tauc"]).any()
    assert set(np.unique(member["design:tauc"])) == {3.0e5, 1.4e5}
    assert set(np.unique(member["zeta:tauc"])) == {3.0, -3.0}


def test_read_member_skips_incomplete(ensemble: list[Path]) -> None:
    """
    Return None for a member that has not written its residual.

    The residual is only missed when it is asked for; the same file still
    yields a panel for the fields it does carry.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.

    Returns
    -------
    None
        Asserts only.
    """
    assert read_member(ensemble[-1], [PENALTY], list(FIELDS)) is None
    assert read_member(ensemble[-1], [PENALTY], ["design", "zeta"]) is not None


def test_read_member_skips_unknown_parameter(ensemble: list[Path]) -> None:
    """
    Return None when the swept parameter is absent from ``pism_config``.

    Parameters
    ----------
    ensemble : list of pathlib.Path
        Synthetic ensemble from the fixture.

    Returns
    -------
    None
        Asserts only.
    """
    assert read_member(ensemble[0], ["surface.pdd.factor_ice"], list(FIELDS)) is None


def test_field_items_expand_over_the_design_variables() -> None:
    """
    Give the design variable and zeta a figure per phase, the residual one.

    Returns
    -------
    None
        Asserts only.
    """
    assert field_items(list(FIELDS), ["tauc"]) == [("design", "tauc"), ("zeta", "tauc"), ("residual", None)]
    assert field_items(list(FIELDS), ["tauc", "hardav"]) == [
        ("design", "tauc"),
        ("design", "hardav"),
        ("zeta", "tauc"),
        ("zeta", "hardav"),
        ("residual", None),
    ]
    assert item_key("zeta", "hardav") == "zeta:hardav"
    assert item_key("residual", None) == "residual"


def test_shared_limits_pool_across_members() -> None:
    """
    Span every member, and let explicit limits win over the percentiles.

    Returns
    -------
    None
        Asserts only.
    """
    arrays = [np.array([[1.0, 2.0]]), np.array([[10.0, np.nan]])]
    assert shared_limits(arrays, percentile=0.0) == (1.0, 10.0)
    assert shared_limits(arrays, percentile=0.0, vmin=0.5) == (0.5, 10.0)
    assert shared_limits(arrays, percentile=0.0, vmax=99.0) == (1.0, 99.0)
    # Trimming the tails pulls both ends in.
    low, high = shared_limits(arrays, percentile=25.0)
    assert 1.0 < low and high < 10.0


def test_shared_limits_diverging_is_symmetric() -> None:
    """
    Centre a diverging scale on zero, whichever side is larger.

    Returns
    -------
    None
        Asserts only.
    """
    assert shared_limits([np.array([[-2.0, 7.0]])], percentile=0.0, scale="diverging") == (-7.0, 7.0)
    assert shared_limits([np.array([[-7.0, 2.0]])], percentile=0.0, scale="diverging") == (-7.0, 7.0)
    # An explicit limit still wins, even if that breaks the symmetry.
    assert shared_limits([np.array([[-2.0, 7.0]])], percentile=0.0, scale="diverging", vmin=0.0) == (0.0, 7.0)


def test_shared_limits_guards() -> None:
    """
    Refuse a fully-masked row, and drop non-positives for a log scale.

    Returns
    -------
    None
        Asserts only.
    """
    with pytest.raises(ValueError, match="fully masked"):
        shared_limits([np.array([[np.nan, np.nan]])])
    with pytest.raises(ValueError, match="log color scale"):
        shared_limits([np.array([[0.0, -1.0]])], percentile=0.0, scale="log")
    assert shared_limits([np.array([[0.0, 1.0, 4.0]])], percentile=0.0, scale="log") == (1.0, 4.0)
    # A constant field still yields an increasing pair.
    low, high = shared_limits([np.array([[3.0, 3.0]])], percentile=0.0)
    assert low < high


def test_plot_field_writes_one_file_per_variable(ensemble: list[Path], tmp_path: Path) -> None:
    """
    Name each figure after its variable, and draw a single member too.

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
    members = [m for m in (read_member(f, [PENALTY], list(FIELDS)) for f in ensemble) if m is not None]
    members.sort(key=lambda m: m["penalty_weight"])
    base = tmp_path / "figures" / "maps.png"
    written = [
        plot_field(members, key, design, "penalty_weight", base) for key, design in field_items(list(FIELDS), ["tauc"])
    ]
    assert [p.name for p in written] == ["maps_tauc.png", "maps_zeta_inv.png", "maps_inv_residual.png"]
    assert all(p.exists() for p in written)

    single = plot_field(members[:1], "design", "tauc", "penalty_weight", tmp_path / "one.pdf", scale="linear")
    assert single.name == "one_tauc.pdf" and single.exists()


def test_plot_field_wraps_onto_a_grid(ensemble: list[Path], tmp_path: Path) -> None:
    """
    Wrap the members onto ``ncols`` columns, hiding a partial row's leftovers.

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
    members = [m for m in (read_member(f, [PENALTY], ["zeta"]) for f in ensemble) if m is not None]
    # Four members over two columns: a full grid.
    assert plot_field(members, "zeta", "tauc", "penalty_weight", tmp_path / "full.png", ncols=2).exists()
    # Three over two: the fourth cell must be hidden rather than left empty.
    assert plot_field(members[:3], "zeta", "tauc", "penalty_weight", tmp_path / "partial.png", ncols=2).exists()
    # More columns than members still gives one row.
    assert plot_field(members[:2], "zeta", "tauc", "penalty_weight", tmp_path / "wide.png", ncols=8).exists()


def test_main_end_to_end(ensemble: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Write one figure per field from the command line.

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
    base = tmp_path / "out" / "maps.png"
    monkeypatch.setattr(
        "sys.argv",
        ["pism-inverse-plot", "--parameters", PENALTY, "-o", str(base)] + [str(f) for f in ensemble],
    )
    main()
    assert {p.name for p in base.parent.glob("*.png")} == {
        "maps_tauc.png",
        "maps_zeta_inv.png",
        "maps_inv_residual.png",
    }


def test_main_plots_a_subset(ensemble: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Write only the fields ``--variables`` asks for, and reject unknown ones.

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
    base = tmp_path / "subset" / "maps.png"
    argv = ["pism-inverse-plot", "--variables", "zeta", "-o", str(base)] + [str(f) for f in ensemble]
    monkeypatch.setattr("sys.argv", argv)
    main()
    assert {p.name for p in base.parent.glob("*.png")} == {"maps_zeta_inv.png"}

    monkeypatch.setattr("sys.argv", argv[:1] + ["--variables", "speed"] + argv[3:])
    with pytest.raises(SystemExit):
        main()


def test_main_exits_when_nothing_is_usable(
    ensemble: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Exit with a message rather than an empty figure.

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
    monkeypatch.setattr(
        "sys.argv",
        ["pism-inverse-plot", "-o", str(tmp_path / "none.png"), str(ensemble[-1])],
    )
    with pytest.raises(SystemExit, match="yielded a panel"):
        main()


@pytest.fixture(name="alternating")
def fixture_alternating(tmp_path: Path) -> list[Path]:
    """
    Two-member alternating co-inversion, plus one still in its first phase.

    The unfinished member carries only ``zeta_inv_tauc``, as ``pismi`` writes
    mid-cycle.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    list of pathlib.Path
        The member files.
    """
    files = [
        write_member(tmp_path / f"alt_{p:g}.nc", p, zeta=z, alternating=True) for p, z in [(1.0, 1.5), (10.0, 3.0)]
    ]
    partial = xr.open_dataset(files[0]).load()
    partial = partial.drop_vars(["tauc", "hardav", "tauc_prior", "hardav_prior", "zeta_inv_hardav", "inv_residual"])
    partial["pism_config"].attrs[PENALTY] = np.float64(100.0)
    partial.to_netcdf(tmp_path / "alt_partial.nc")
    files.append(tmp_path / "alt_partial.nc")
    return files


def test_alternating_yields_a_figure_per_phase(alternating: list[Path], tmp_path: Path) -> None:
    """
    Read both phases of a co-inversion and name each figure after its field.

    Parameters
    ----------
    alternating : list of pathlib.Path
        Alternating ensemble from the fixture.
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None
        Asserts only.
    """
    member = read_member(alternating[0], [PENALTY], list(FIELDS))
    assert member is not None
    assert member["designs"] == ["tauc", "hardav"]
    assert member["resolved"]["zeta:tauc"] == "zeta_inv_tauc"
    assert member["resolved"]["zeta:hardav"] == "zeta_inv_hardav"

    members = [m for m in (read_member(f, [PENALTY], list(FIELDS)) for f in alternating[:2]) if m is not None]
    base = tmp_path / "alt.png"
    written = [
        plot_field(members, key, design, "penalty_weight", base)
        for key, design in field_items(list(FIELDS), members[0]["designs"])
    ]
    assert [p.name for p in written] == [
        "alt_tauc.png",
        "alt_hardav.png",
        "alt_zeta_inv_tauc.png",
        "alt_zeta_inv_hardav.png",
        "alt_inv_residual.png",
    ]
    assert all(p.exists() for p in written)


def test_alternating_subset_and_partial_members(
    alternating: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Skip a member that has not reached a phase, but keep it for one it has.

    Parameters
    ----------
    alternating : list of pathlib.Path
        Alternating ensemble from the fixture.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to set ``sys.argv``.

    Returns
    -------
    None
        Asserts only.
    """
    # The half-finished member has no hardav phase yet.
    assert read_member(alternating[-1], [PENALTY], list(FIELDS)) is None
    # ... but its tauc zeta is there, so a zeta-only run keeps all three.
    base = tmp_path / "phase" / "alt.png"
    monkeypatch.setattr(
        "sys.argv",
        ["pism-inverse-plot", "--variables", "zeta", "--design-variable", "tauc", "-o", str(base)]
        + [str(f) for f in alternating],
    )
    main()
    assert {p.name for p in base.parent.glob("*.png")} == {"alt_zeta_inv_tauc.png"}


def test_design_variable_takes_a_list(alternating: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Accept a comma-separated ``--design-variable``, and reject a bad name.

    Parameters
    ----------
    alternating : list of pathlib.Path
        Alternating ensemble from the fixture.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to set ``sys.argv``.

    Returns
    -------
    None
        Asserts only.
    """
    base = tmp_path / "both" / "alt.png"
    argv = [
        "pism-inverse-plot",
        "--variables",
        "design",
        "--design-variable",
        "tauc,hardav",
        "-o",
        str(base),
    ] + [str(f) for f in alternating[:2]]
    monkeypatch.setattr("sys.argv", argv)
    main()
    assert {p.name for p in base.parent.glob("*.png")} == {"alt_tauc.png", "alt_hardav.png"}

    monkeypatch.setattr("sys.argv", argv[:4] + ["tauc,speed"] + argv[5:])
    with pytest.raises(SystemExit):
        main()


def test_completed_phases_reads_the_alternation_stamp(tmp_path: Path) -> None:
    """
    Tell an inverted phase from one still holding its prior.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None
        Asserts only.
    """
    cases = {
        # Each cycle runs tauc then hardav.
        "c0_tauc": {"tauc"},
        "c0_hardav": {"tauc", "hardav"},
        "c1_tauc": {"tauc", "hardav"},
        "c1_hardav": {"tauc", "hardav"},
    }
    for stamp, expected in cases.items():
        path = write_member(tmp_path / f"{stamp}.nc", 1.0, alternating=True, completed=stamp)
        with xr.open_dataset(path) as ds:
            assert completed_phases(ds) == expected, stamp

    # A single-design run carries no stamp.
    plain = write_member(tmp_path / "plain.nc", 1.0)
    with xr.open_dataset(plain) as ds:
        assert completed_phases(ds) is None
