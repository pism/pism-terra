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
Tests for refusing a temperature-spread file no forcing can give.

No climate forcing carries ``air_temp_sd``; the surface models use a constant
spread. A config that still asks for a spread file stops at render time
rather than in PISM.
"""

from __future__ import annotations

import pytest

from pism_terra.glacier import run


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
    run._require_spread_file({option: "/in/spread.nc"})  # pylint: disable=protected-access
    run._require_spread_file({"surface.pdd.std_dev.value": 5.0})  # pylint: disable=protected-access
