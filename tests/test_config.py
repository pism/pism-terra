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

# pylint: disable=unused-import

"""
Tests for UQConfig and DistSpec validation & flattening.

This suite checks:
- Flattening nested TOML-like dicts into dotted keys.
- Acceptance of already-flat (dotted) keys.
- Preservation of `samples` and `mapping`.
- Validation of distribution names and required parameters.
- Consistency between `iter_specs()` and `to_flat()` outputs.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# Adjust this import path to match your project layout:
# Adjust the import path to wherever your models live
from pism_terra.config import (  # noqa: F401  (ensure DistSpec is imported)
    DistSpec,
    UQConfig,
)


def _specs_dict(uq: UQConfig) -> dict:
    """
    Convert a UQConfig into a plain dict via `to_flat()`.

    Parameters
    ----------
    uq : UQConfig
        Validated uncertainty configuration.

    Returns
    -------
    dict
        Mapping from dotted variable name to plain dict specification.
    """
    return uq.to_flat()


def test_flatten_nested_norm_ok():
    """
    Validate nested → dotted flattening with a normal distribution spec.

    Notes
    -----
    Ensures that:
    - `samples` and `mapping` are preserved at the top level.
    - A nested `surface.pdd.factor_ice` leaf is flattened to
      `'surface.pdd.factor_ice'` with correct fields.
    """
    raw = {
        "samples": 10,
        "mapping": {"surface.given.file": "cosipy_file"},
        "surface": {
            "pdd": {
                "factor_ice": {
                    "distribution": "norm",
                    "loc": 0,
                    "scale": 1,
                }
            }
        },
    }
    uq = UQConfig.model_validate(raw)
    assert uq.samples == 10
    assert uq.mapping == {"surface.given.file": "cosipy_file"}

    flat = _specs_dict(uq)
    assert "surface.pdd.factor_ice" in flat
    assert flat["surface.pdd.factor_ice"]["distribution"] == "norm"
    assert flat["surface.pdd.factor_ice"]["loc"] == 0
    assert flat["surface.pdd.factor_ice"]["scale"] == 1


def test_flat_block_already_dotted_ok():
    """
    Validate that already-dotted `tree` entries are kept as-is.

    Notes
    -----
    Ensures top-level `samples` and `mapping` are preserved and a dotted key
    in `tree` is not re-flattened or altered.
    """
    raw = {
        "tree": {"surface.pdd.factor_snow": {"distribution": "norm", "loc": 2, "scale": 0.5}},
        "samples": 5,
        "mapping": {"atmosphere.given.file": "cosipy_file"},
    }
    uq = UQConfig.model_validate(raw)
    assert uq.samples == 5
    assert uq.mapping == {"atmosphere.given.file": "cosipy_file"}
    flat = _specs_dict(uq)
    assert list(flat.keys()) == ["surface.pdd.factor_snow"]
    assert flat["surface.pdd.factor_snow"]["distribution"] == "norm"


def test_distribution_name_normalized_case_insensitive():
    """
    Validate case-insensitive normalization of distribution names.

    Notes
    -----
    The distribution `"Norm"` should be normalized to `"norm"`.
    """
    raw = {"surface": {"pdd": {"factor_ice": {"distribution": "Norm", "loc": 0, "scale": 1}}}}
    uq = UQConfig.model_validate(raw)
    flat = _specs_dict(uq)
    assert flat["surface.pdd.factor_ice"]["distribution"] == "norm"


def test_iter_specs_matches_to_flat_keys():
    """
    Ensure `iter_specs()` yields the same keys as `to_flat()`.

    Notes
    -----
    This checks consistency between iteration and serialization helpers.
    """
    raw = {
        "surface": {
            "pdd": {
                "factor_ice": {"distribution": "norm", "loc": 0, "scale": 1},
                "factor_snow": {"distribution": "norm", "loc": 2, "scale": 1},
            }
        }
    }
    uq = UQConfig.model_validate(raw)
    flat = set(_specs_dict(uq).keys())
    iter_keys = {name for name, _ in uq.iter_specs()}
    assert flat == iter_keys


def test_samples_must_be_positive():
    """
    Fail when `samples` ≤ 0.

    Raises
    ------
    pydantic.ValidationError
        If `samples` is not strictly positive.
    """
    raw = {
        "samples": 0,
        "surface": {"pdd": {"factor_ice": {"distribution": "norm", "loc": 0, "scale": 1}}},
    }
    with pytest.raises(ValidationError):
        UQConfig.model_validate(raw)


def test_unknown_distribution_fails():
    """
    Fail when an unknown SciPy distribution is specified.

    Raises
    ------
    pydantic.ValidationError
        If `distribution` does not correspond to an attribute in `scipy.stats`.
    """
    raw = {"surface": {"pdd": {"factor_ice": {"distribution": "not_a_dist", "loc": 0, "scale": 1}}}}
    with pytest.raises(ValidationError) as excinfo:
        UQConfig.model_validate(raw)
    assert "unknown SciPy distribution" in str(excinfo.value)


def test_truncnorm_missing_shape_params_fails():
    """
    Fail when `truncnorm` is missing required shape parameters.

    Raises
    ------
    pydantic.ValidationError
        If the required shape parameters are not provided.

    Notes
    -----
    `truncnorm` requires either:
    - shape parameters `a` and `b`, or
    - convenience bounds `lower` and `upper` (depending on your DistSpec logic).
    """
    raw = {
        "surface": {
            "pdd": {
                "factor_ice": {
                    "distribution": "truncnorm",
                    "loc": 0,
                    "scale": 1,
                }
            }
        }
    }
    with pytest.raises(ValidationError) as excinfo:
        UQConfig.model_validate(raw)
    msg = str(excinfo.value)
    assert "truncnorm" in msg
    assert ("shape parameter(s)" in msg) or ("requires" in msg)


def test_truncnorm_with_a_b_ok():
    """
    Validate `truncnorm` with explicit shape parameters `a` and `b`.

    Notes
    -----
    Ensures DistSpec accepts shape parameters and preserves them on dump.
    """
    raw = {
        "surface": {
            "pdd": {
                "factor_ice": {
                    "distribution": "truncnorm",
                    "loc": 0,
                    "scale": 1,
                    "a": -1.0,
                    "b": 1.5,
                }
            }
        }
    }
    uq = UQConfig.model_validate(raw)
    flat = _specs_dict(uq)
    spec = flat["surface.pdd.factor_ice"]
    assert spec["distribution"] == "truncnorm"
    assert spec["a"] == -1.0
    assert spec["b"] == 1.5


def test_truncnorm_with_lower_upper_ok():
    """
    Validate `truncnorm` when using `lower`/`upper` convenience bounds.

    Notes
    -----
    If supported by your DistSpec, `lower`/`upper` are accepted instead of `a`/`b`.
    """
    raw = {
        "surface": {
            "pdd": {
                "factor_ice": {
                    "distribution": "truncnorm",
                    "loc": 10,
                    "scale": 2,
                    "lower": 8,
                    "upper": 12,
                }
            }
        }
    }
    uq = UQConfig.model_validate(raw)
    flat = _specs_dict(uq)
    spec = flat["surface.pdd.factor_ice"]
    assert spec["distribution"] == "truncnorm"
    # If DistSpec preserves these keys, assert them; otherwise adjust the test.
    assert spec["lower"] == 8
    assert spec["upper"] == 12
