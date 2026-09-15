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

from pathlib import Path

import pytest

# Adjust this import path to match your project layout:
# Adjust the import path to wherever your models live
import toml
from pydantic import ValidationError

from pism_terra.config import (  # noqa: F401  (ensure DistSpec is imported)
    CampaignConfig,
    DistSpec,
    UQConfig,
    load_config,
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


def test_choices_spec_ok():
    """
    Validate a categorical spec and its round-trip through `to_flat()`.

    Notes
    -----
    The `choices` list (and an optional `weights` list) ride along in
    `model_extra`, so they must survive `DistSpec.model_dump()`.
    """
    raw = {
        "samples": 4,
        "inverse.state_func": {"distribution": "choices", "choices": ["meansquare", "huber"]},
        "calving.eigen_calving.K": {"distribution": "categorical", "choices": [1e15, 1e17], "weights": [3, 1]},
    }
    uq = UQConfig.model_validate(raw)
    flat = _specs_dict(uq)
    assert flat["inverse.state_func"]["distribution"] == "choices"
    assert flat["inverse.state_func"]["choices"] == ["meansquare", "huber"]
    assert flat["calving.eigen_calving.K"]["weights"] == [3, 1]


def test_choices_requires_non_empty_list():
    """
    Fail when a categorical spec has a missing or empty `choices` list.

    Raises
    ------
    pydantic.ValidationError
        If `choices` is absent or empty.
    """
    variants: tuple[dict, ...] = ({}, {"choices": []})
    for choices in variants:
        raw = {"inverse.state_func": {"distribution": "choices", **choices}}
        with pytest.raises(ValidationError) as excinfo:
            UQConfig.model_validate(raw)
        assert "choices" in str(excinfo.value)


def test_choices_weights_must_match():
    """
    Fail when categorical `weights` do not line up with `choices`.

    Raises
    ------
    pydantic.ValidationError
        If `weights` has the wrong length, is negative, or sums to zero.
    """
    bad_weights = ([1.0], [1.0, -1.0], [0.0, 0.0])
    for weights in bad_weights:
        raw = {
            "inverse.state_func": {
                "distribution": "choices",
                "choices": ["meansquare", "huber"],
                "weights": weights,
            }
        }
        with pytest.raises(ValidationError) as excinfo:
            UQConfig.model_validate(raw)
        assert "weights" in str(excinfo.value)


def test_campaign_init_fields():
    """Campaign init_start/init_end are parsed and exported by as_params()."""
    config_file = (
        Path(__file__).resolve().parents[1] / "pism_terra" / "config" / "ismip7_greenland_2007_historical_free.toml"
    )
    cfg = load_config(config_file)
    assert cfg.campaign.init_start == "2006-01-01"
    assert cfg.campaign.init_end == "2007-01-01"
    params = cfg.campaign.as_params()
    assert params["init_start"] == "2006-01-01"
    assert params["init_end"] == "2007-01-01"


def _campaign(name: str) -> CampaignConfig:
    """
    Validate just the ``[campaign]`` table of a packaged config.

    The glacier configs do not satisfy the full ``PismConfig`` schema (they
    predate the required ``bed_deformation`` section), so load the one section
    under test rather than the whole file.

    Parameters
    ----------
    name : str
        File name under ``pism_terra/config``.

    Returns
    -------
    CampaignConfig
        The validated campaign section.
    """
    path = Path(__file__).resolve().parents[1] / "pism_terra" / "config" / name
    return CampaignConfig.model_validate(toml.loads(path.read_text("utf-8"))["campaign"])


def test_campaign_project_directory():
    """Campaign project_directory is parsed and exported by as_params()."""
    s4f = _campaign("s4f_carra2_maffezzoli.toml")
    assert s4f.prefix == "glacier/input"
    assert s4f.project_directory == "s4f"
    assert s4f.as_params()["project_directory"] == "s4f"
    # The outlines depend on the project's CRS overrides, so they are named for it.
    assert s4f.rgi_complex_file == "s4f_c.gpkg"

    assert _campaign("rgi_era5_frank.toml").project_directory == "rgi"

    # Campaigns whose input tree is not split by project leave it unset, and
    # as_params() drops empty fields, so stage's .get() falls back to None.
    kitp = _campaign("kitp_greenland.toml")
    assert kitp.project_directory is None
    assert kitp.as_params().get("project_directory") is None


def _config_without(section: str, tmp_path: Path) -> Path:
    """
    Copy a packaged config with one whole section removed.

    Parameters
    ----------
    section : str
        Section name, e.g. ``"bed_deformation"``.
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.

    Returns
    -------
    pathlib.Path
        The stripped config.
    """
    source = Path(__file__).resolve().parents[1] / "pism_terra" / "config" / "s4f_carra2_maffezzoli.toml"
    kept, dropping = [], False
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            dropping = stripped.strip("[]'\"").split(".")[0] == section
        if not dropping:
            kept.append(line)
    out = tmp_path / f"no_{section}.toml"
    out.write_text("\n".join(kept), encoding="utf-8")
    return out


@pytest.mark.parametrize("section", ["bed_deformation", "frontal_melt"])
def test_optional_model_sections_may_be_omitted(section, tmp_path):
    """
    Validate a config that omits an optional model section.

    Parameters
    ----------
    section : str
        Section name to remove before loading.
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    cfg = load_config(_config_without(section, tmp_path))

    assert getattr(cfg, section).model == "none"
    # An omitted section must contribute nothing to the PISM command line.
    assert getattr(cfg, section).selected() == {}


def test_declared_model_sections_are_untouched():
    """Keep the options of a config that does declare the sections."""
    cfg = load_config(Path(__file__).resolve().parents[1] / "pism_terra" / "config" / "ismip7_greenland_c001.toml")

    assert cfg.bed_deformation.selected() == {"bed_deformation.model": "lc"}
    assert cfg.frontal_melt.selected()["frontal_melt.models"] == "routing"
