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
Tests for :mod:`pism_terra.ismip7.greenland.observations`.

Covers the parts that are ours rather than the upstream data's: that cell
areas shrink toward the pole, that quantifying leaves the coordinates alone
so a derived array still aligns, that the PO.DAAC header is skipped however
long it is, and that the products end up beside the run — including when one
of the three could not be built.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pism_terra.ismip7.greenland import observations as obs


def test_cell_area_shrinks_toward_the_pole():
    """
    A degree of longitude is shorter at 80N than at the equator.

    Summing equivalent water thickness without this is the difference
    between a plausible Greenland mass loss and a badly wrong one.
    """
    equator = obs.polygon_area(0.0, 1.0, 0.0, 1.0)
    greenland = obs.polygon_area(79.0, 80.0, 0.0, 1.0)
    assert equator > greenland
    # cos(79.5 deg) ~ 0.18, so roughly a fifth.
    assert 0.1 < greenland / equator < 0.25


def _bounded_grid(n_lat: int = 4, n_lon: int = 5) -> xr.Dataset:
    """
    A small lat/lon dataset with bounds, like the GSFC mascon file.

    Parameters
    ----------
    n_lat : int, optional
        Number of latitude cells.
    n_lon : int, optional
        Number of longitude cells.

    Returns
    -------
    xr.Dataset
        Dataset with ``lat_bounds``, ``lon_bounds`` and a data variable.
    """
    lat = 60.0 + np.arange(n_lat) * 0.5
    lon = 300.0 + np.arange(n_lon) * 0.5
    return xr.Dataset(
        {
            "lat_bounds": (("lat", "bounds"), np.stack([lat - 0.25, lat + 0.25], axis=1)),
            "lon_bounds": (("lon", "bounds"), np.stack([lon - 0.25, lon + 0.25], axis=1)),
            "thickness": (("lat", "lon"), np.ones((n_lat, n_lon)), {"units": "cm"}),
        },
        coords={
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
    )


def test_cell_areas_covers_the_grid():
    """
    Every cell gets an area, carrying units.
    """
    ds = _bounded_grid()
    area = obs.cell_areas(ds)
    assert area.dims == ("lat", "lon")
    assert area.shape == (4, 5)
    assert area.attrs["units"] == "m^2"
    assert float(area.min()) > 0


def test_quantify_leaves_coordinates_alone_so_derived_arrays_align():
    """
    Quantified data must still multiply with something built by broadcast.

    pint-xarray >= 0.6 gives a quantified indexed coordinate a ``PintIndex``,
    while ``xr.broadcast``/``xr.apply_ufunc`` hand back a plain
    ``PandasIndex``; mixing the two is an ``AlignmentError``. This is the
    exact product the GSFC preparation forms.
    """
    ds = _bounded_grid()
    quantified = obs.quantify_data_only(ds)
    assert type(quantified.xindexes["lat"]).__name__ == "PandasIndex"

    area = obs.cell_areas(quantified)
    product = quantified["thickness"].pint.to("m") * obs.quantify_data_only(area)
    assert product.shape == (4, 5)
    assert str(product.pint.units) == "meter ** 3"


def test_plain_quantify_is_what_we_are_avoiding():
    """
    Guard the reason ``quantify_data_only`` exists at all.

    A plain ``.pint.quantify()`` wraps ``lat``/``lon`` because they carry a
    ``units`` attribute, and the resulting ``PintIndex`` is what will not
    align with a broadcast-derived array later.
    """
    ds = _bounded_grid()
    assert type(ds.pint.quantify().xindexes["lat"]).__name__ == "PintIndex"
    assert type(obs.quantify_data_only(ds).xindexes["lat"]).__name__ == "PandasIndex"


@pytest.mark.parametrize(
    ("decimal_year", "expected"),
    [
        (2002.0, datetime.datetime(2002, 1, 1)),
        (2002.5, datetime.datetime(2002, 7, 3)),
        (2003.9986, datetime.datetime(2003, 12, 31)),
    ],
)
def test_decimal_year_to_datetime(decimal_year, expected):
    """
    Decimal years land on the right day.

    Parameters
    ----------
    decimal_year : float
        Year with a fractional part.
    expected : datetime.datetime
        The day it should fall on.
    """
    assert obs.decimal_year_to_datetime(decimal_year) == expected


def test_read_mass_time_series_ignores_the_header(tmp_path: Path):
    """
    Take the data rows whatever the header length is.

    The old code skipped exactly 32 lines, which was tuned to a release that
    has since been superseded.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    path = tmp_path / "greenland_mass.txt"
    header = ["# GRACE Greenland mass time series"] + [f"# filler {i}" for i in range(40)]
    rows = [f"{2002.29 + i / 12:10.4f} {-i * 2.4:12.4f} {60.0:10.4f}" for i in range(24)]
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")

    df = obs.read_mass_time_series(path)
    assert len(df) == 24
    assert list(df.columns) == ["year", "cumulative_mass_balance", "mass_balance_uncertainty"]
    assert df["cumulative_mass_balance"].iloc[-1] == pytest.approx(-55.2)


def test_read_mass_time_series_rejects_a_file_with_no_data(tmp_path: Path):
    """
    A changed format is an error, not an empty frame.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    path = tmp_path / "empty.txt"
    path.write_text("# only a header\n# and nothing else\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no three-column data rows"):
        obs.read_mass_time_series(path)


def _fake_product(cache_path: Path, name: str) -> Path:
    """
    Write a stand-in product into the cache.

    Parameters
    ----------
    cache_path : pathlib.Path
        Cache directory.
    name : str
        Product filename.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    path = cache_path / name
    path.write_bytes(b"netcdf-ish")
    return path


def test_prepare_observations_places_all_three_beside_the_run(tmp_path: Path, monkeypatch):
    """
    The products land in ``<output_path>/output/observations``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to stand in for the three downloads.
    """
    for builder, name in (
        ("prepare_grace_gsfc", obs.PRODUCTS[0]),
        ("prepare_grace_tellus", obs.PRODUCTS[1]),
        ("prepare_mankoff", obs.PRODUCTS[2]),
    ):
        monkeypatch.setattr(obs, builder, lambda cache_path, name=name, **_kwargs: _fake_product(cache_path, name))

    placed = obs.prepare_observations(tmp_path / "run", cache_path=tmp_path / "cache")
    destination = tmp_path / "run" / "output" / "observations"
    assert [p.parent for p in placed] == [destination] * 3
    assert sorted(p.name for p in destination.iterdir()) == sorted(obs.PRODUCTS)
    # A copy, not a move: the cache is shared and must survive.
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == sorted(obs.PRODUCTS)


def test_prepare_observations_skips_a_failure_only_when_asked(tmp_path: Path, monkeypatch):
    """
    Staging must not fail over observations; the CLI should say so.

    The GRACE Tellus product needs an Earthdata login, which a machine that
    stages a run may not have. Those are validation data, not run inputs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to stand in for the three downloads.
    """

    def _boom(cache_path, **_kwargs):
        """
        Stand in for a product that cannot be fetched.

        Parameters
        ----------
        cache_path : pathlib.Path
            Cache directory, ignored.
        **_kwargs : dict
            Builder arguments, ignored.
        """
        raise RuntimeError("no Earthdata login")

    monkeypatch.setattr(obs, "prepare_grace_gsfc", lambda cache_path, **_k: _fake_product(cache_path, obs.PRODUCTS[0]))
    monkeypatch.setattr(obs, "prepare_grace_tellus", _boom)
    monkeypatch.setattr(obs, "prepare_mankoff", lambda cache_path, **_k: _fake_product(cache_path, obs.PRODUCTS[2]))

    placed = obs.prepare_observations(tmp_path / "run", cache_path=tmp_path / "cache", skip_errors=True)
    assert sorted(p.name for p in placed) == sorted([obs.PRODUCTS[0], obs.PRODUCTS[2]])

    with pytest.raises(RuntimeError, match="no Earthdata login"):
        obs.prepare_observations(tmp_path / "run2", cache_path=tmp_path / "cache2")
