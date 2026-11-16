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
Tests for derived UQ parameters.

Some PISM options are physically locked to one another. Sampling them
separately decorrelates them, and parallel ``choices`` lists do not help
because Latin Hypercube permutes each column independently. A derived
parameter states the relation instead, and is evaluated after sampling.
"""

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from pism_terra.config import UQConfig, load_uq
from pism_terra.sampling import generate_samples

PACKAGED = Path(__file__).resolve().parents[1] / "pism_terra" / "uq" / "kennicott_elevation_dependent.toml"


def uq_from(body: str) -> UQConfig:
    """
    Validate a UQ spec written as a TOML fragment.

    Parameters
    ----------
    body : str
        TOML source.

    Returns
    -------
    UQConfig
        The validated configuration.
    """
    import toml  # pylint: disable=import-outside-toplevel

    return UQConfig.model_validate(toml.loads(body))


BASE_AND_TWO_DERIVED = """
samples = 7

['a.z_min']
distribution = "choices"
choices = [700, 650, 600, 550, 500, 450, 400]

['a.z_ela']
derived_from = "a.z_min"
offset = 1050

['a.z_max']
derived_from = "a.z_min"
offset = 1892
"""


def test_derived_entries_are_not_sampled():
    """
    Keep derived parameters out of the sampling design.
    """
    uq = uq_from(BASE_AND_TWO_DERIVED)

    assert list(uq.tree) == ["a.z_min"]
    assert sorted(uq.derived) == ["a.z_ela", "a.z_max"]
    # Only the base reaches the sampler.
    assert list(uq.to_flat()) == ["a.z_min"]


def test_derived_values_track_their_base():
    """
    Hold the declared relation on every member.
    """
    uq = uq_from(BASE_AND_TWO_DERIVED)

    df = uq.apply_derived(generate_samples(uq.to_flat(), n_samples=uq.samples, seed=42))

    assert set(df["a.z_ela"] - df["a.z_min"]) == {1050}
    assert set(df["a.z_max"] - df["a.z_min"]) == {1892}


def test_whole_numbers_stay_whole():
    """
    Write an elevation as 1750, not 1750.0.
    """
    uq = uq_from(BASE_AND_TWO_DERIVED)

    df = uq.apply_derived(generate_samples(uq.to_flat(), n_samples=uq.samples, seed=42))

    assert df["a.z_ela"].dtype == "int64"


def test_scale_is_applied_before_the_offset():
    """
    Compute ``base * scale + offset``, in that order.
    """
    uq = uq_from("""
        samples = 3

        ['a.base']
        distribution = "choices"
        choices = [100, 200, 300]

        ['a.scaled']
        derived_from = "a.base"
        scale = 1.5
        offset = 10
        """)

    df = uq.apply_derived(generate_samples(uq.to_flat(), n_samples=3, seed=1))

    pd.testing.assert_series_equal(
        df["a.scaled"], df["a.base"].astype(float) * 1.5 + 10, check_names=False, check_dtype=False
    )


def test_a_derived_parameter_may_follow_another():
    """
    Resolve a chain, base first.
    """
    uq = uq_from("""
        samples = 3

        ['a.base']
        distribution = "choices"
        choices = [10, 20, 30]

        ['a.second']
        derived_from = "a.base"
        offset = 5

        ['a.third']
        derived_from = "a.second"
        offset = 5
        """)

    assert uq.derived_order().index("a.second") < uq.derived_order().index("a.third")
    df = uq.apply_derived(generate_samples(uq.to_flat(), n_samples=3, seed=1))
    assert set(df["a.third"] - df["a.base"]) == {10}


def test_an_unknown_base_is_rejected():
    """
    Fail at load rather than as a missing column mid-ensemble.
    """
    with pytest.raises(ValidationError, match="derived_from names no UQ parameter"):
        uq_from("""
            samples = 2

            ['a.base']
            distribution = "choices"
            choices = [1, 2]

            ['a.other']
            derived_from = "a.typo"
            """)


def test_a_cycle_is_rejected():
    """
    Refuse parameters that derive from each other.
    """
    with pytest.raises(ValidationError, match="cycle"):
        uq_from("""
            samples = 2

            ['a.one']
            derived_from = "a.two"

            ['a.two']
            derived_from = "a.one"
            """)


def test_an_entry_cannot_declare_both():
    """
    Refuse a spec carrying a distribution and a base at once.

    Without this the entry is quietly treated as sampled and ``derived_from``
    is ignored, so the linkage the author asked for never happens.
    """
    with pytest.raises(ValidationError, match="both 'distribution' and 'derived_from'"):
        uq_from("""
            samples = 2

            ['a.one']
            distribution = "choices"
            choices = [1, 2]
            derived_from = "a.two"

            ['a.two']
            distribution = "choices"
            choices = [3, 4]
            """)


def test_overlapping_tree_and_derived_are_rejected():
    """
    Guard the same invariant when the model is built directly, not from TOML.
    """
    with pytest.raises(ValidationError, match="both sampled and derived"):
        UQConfig.model_validate(
            {
                "samples": 2,
                "tree": {"a.one": {"distribution": "choices", "choices": [1, 2]}},
                "derived": {"a.one": {"derived_from": "a.one"}},
            }
        )


def test_an_entry_with_neither_key_is_rejected():
    """
    Name the entries that declare no distribution and no base.
    """
    with pytest.raises(ValidationError, match="neither 'distribution' nor 'derived_from'"):
        uq_from("""
            samples = 2

            ['a.one']
            loc = 3
            scale = 1
            """)


def test_a_misspelled_field_is_rejected():
    """
    Catch ``ofset`` rather than silently applying no offset.
    """
    with pytest.raises(ValidationError):
        uq_from("""
            samples = 2

            ['a.base']
            distribution = "choices"
            choices = [1, 2]

            ['a.other']
            derived_from = "a.base"
            ofset = 10
            """)


def test_packaged_kennicott_profile_shifts_as_a_unit():
    """
    Pin the packaged file: one shift sampled, two elevations following it.

    Its ``method = "factorial"`` with a single sampled parameter walks the
    levels in declaration order, so member N is the Nth listed shift.
    """
    uq = load_uq(PACKAGED)

    df = uq.apply_derived(generate_samples(uq.to_flat(), n_samples=uq.samples, method=uq.method, seed=42))

    assert list(uq.tree) == ["surface.elevation_dependent.z_M_min"]
    assert len(df) == uq.samples
    first = df.iloc[0]
    assert first["surface.elevation_dependent.z_M_min"] == 700
    assert first["surface.elevation_dependent.z_ELA"] == 1750
    assert first["surface.elevation_dependent.z_M_max"] == 2592
    assert set(df["surface.elevation_dependent.z_ELA"] - df["surface.elevation_dependent.z_M_min"]) == {1050}
