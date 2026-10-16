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
Tests for reading the temperature spread from the forcing only when it has one.

The CARRA2 and ERA5 forcing carry no ``air_temp_sd``; SNAP's still does. A
surface model is pointed at the forcing for its spread only in the second
case, and a config that asks for a spread file the forcing cannot give stops
at render time rather than in PISM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pism_terra.glacier import run


def _forcing(path: Path, spread: bool) -> Path:
    """
    Write a one-cell forcing file, with or without ``air_temp_sd``.

    Parameters
    ----------
    path : pathlib.Path
        File to write.
    spread : bool
        Whether it holds ``air_temp_sd``.

    Returns
    -------
    pathlib.Path
        The file.
    """
    fields = {"air_temp": (("y", "x"), np.full((1, 1), 270.0))}
    if spread:
        fields["air_temp_sd"] = (("y", "x"), np.full((1, 1), 3.0))
    xr.Dataset(fields).to_netcdf(path)
    return path


def test_only_a_forcing_with_a_spread_is_read_for_it(tmp_path: Path):
    """
    Tell a forcing with ``air_temp_sd`` from one without, and from no file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    assert run._has_spread(_forcing(tmp_path / "snap.nc", spread=True))  # pylint: disable=protected-access
    assert not run._has_spread(_forcing(tmp_path / "carra2.nc", spread=False))  # pylint: disable=protected-access
    assert not run._has_spread(tmp_path / "missing.nc")  # pylint: disable=protected-access
    assert not run._has_spread(None)  # pylint: disable=protected-access


@pytest.mark.parametrize("option", run.SPREAD_FILE_OPTIONS)
def test_a_spread_file_left_unset_is_refused(option: str):
    """
    Stop at render time when a config asks for a spread file nobody filled in.

    Parameters
    ----------
    option : str
        One of :data:`pism_terra.glacier.run.SPREAD_FILE_OPTIONS`.
    """
    with pytest.raises(ValueError, match="air_temp_sd"):
        run._require_spread_file({option: "none"})  # pylint: disable=protected-access
    run._require_spread_file({option: "/in/snap.nc"})  # pylint: disable=protected-access
    run._require_spread_file({"surface.pdd.std_dev.value": 5.0})  # pylint: disable=protected-access
