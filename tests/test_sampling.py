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
Tests for the sampling back end, focused on categorical ("choices") variables.
"""

from __future__ import annotations

import pytest
from pandas.api.types import is_numeric_dtype

from pism_terra.sampling import _make_frozen, create_grid_samples, generate_samples

STATE_FUNC = "inverse.state_func"
SIGMA_MAX = "calving.vonmises_calving.sigma_max"
CHOICES = ["meansquare", "huber"]


def test_choices_lhs_draws_declared_values():
    """Latin Hypercube draws of a categorical variable cover the declared choices."""
    d = {STATE_FUNC: {"distribution": "choices", "choices": CHOICES}}
    df = generate_samples(d, n_samples=10, seed=42)

    assert len(df) == 10
    assert not is_numeric_dtype(df[STATE_FUNC])
    assert set(df[STATE_FUNC]) == set(CHOICES)
    # LHS stratifies the unit interval, so a multiple of len(CHOICES) is balanced
    assert df[STATE_FUNC].value_counts().tolist() == [5, 5]


def test_choices_mixed_with_continuous_keeps_dtypes_and_order():
    """A categorical column coexists with a continuous one without disturbing it."""
    d: dict[str, dict] = {
        STATE_FUNC: {"distribution": "choices", "choices": CHOICES},
        SIGMA_MAX: {"distribution": "uniform", "loc": 300000, "scale": 450000},
    }
    df = generate_samples(d, n_samples=8, seed=0)

    assert list(df.columns) == ["sample", STATE_FUNC, SIGMA_MAX]
    assert is_numeric_dtype(df[SIGMA_MAX])
    assert df[SIGMA_MAX].between(300000, 750000).all()
    assert set(df[STATE_FUNC]) <= set(CHOICES)


def test_choices_factorial_sweeps_categories():
    """The full-factorial grid visits each category once when levels match choices."""
    d = {STATE_FUNC: {"distribution": "choices", "choices": CHOICES}}
    df = create_grid_samples(d, n_levels=len(CHOICES))

    assert df[STATE_FUNC].tolist() == CHOICES


def test_choices_weights_are_honoured():
    """Zero-weighted categories are never drawn."""
    d = {STATE_FUNC: {"distribution": "choices", "choices": CHOICES, "weights": [1, 0]}}
    df = generate_samples(d, n_samples=6, seed=1)

    assert set(df[STATE_FUNC]) == {"meansquare"}


def test_choices_values_kept_verbatim():
    """Non-string choices are returned as declared, not coerced to float."""
    d = {"calving.eigen_calving.K": {"distribution": "choices", "choices": [1, 2, 3]}}
    df = generate_samples(d, n_samples=6, seed=3)

    assert set(df["calving.eigen_calving.K"]) <= {1, 2, 3}


def test_make_frozen_rejects_categorical():
    """`_make_frozen` refuses categorical specs, which have no SciPy counterpart."""
    with pytest.raises(ValueError, match="categorical"):
        _make_frozen("choices", {"distribution": "choices", "choices": CHOICES})
