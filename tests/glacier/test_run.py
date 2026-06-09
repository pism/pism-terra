# Copyright (C) 2025 Andy Aschwanden
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
Test glacier.run functions.
"""

import pytest

from pism_terra.glacier.run import _nullable_string


@pytest.mark.parametrize(
    "argument_string,expected",
    [
        ("None", None),
        ("none", None),
        (" NONE ", None),
        ("foobar", "foobar"),
    ],
)
def test_nullable_string(argument_string, expected) -> None:
    """
    Pytest for the glacier.run._nullable_string function.

    Parameters
    ----------
    argument_string : str
        String to be tested.
    expected : str | None
        Expected value to be returned.
    """
    assert _nullable_string(argument_string) == expected
