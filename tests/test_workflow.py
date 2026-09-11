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
Tests for ``pism_terra.workflow`` helpers used by the run-script generators.
"""

from __future__ import annotations

from pism_terra.workflow import filter_overrides_by_config


def test_filter_drops_keys_not_in_config():
    """
    Override keys absent from the config are dropped and returned as skipped.

    Mirrors the surface-model selection case: with ``surface.model == "pdd"``,
    the config exposes ``surface.force_to_thickness.file`` (declared in
    ``[surface.options.pdd]``) but not ``surface.debm_simple.std_dev.file``,
    so the debm override is filtered out.
    """
    overrides = {
        "surface.force_to_thickness.file": "/tmp/boot.nc",
        "surface.debm_simple.std_dev.file": "/tmp/clim.nc",
        "surface.pdd.std_dev.file": "/tmp/clim.nc",
    }
    allowed = {
        "surface.force_to_thickness.file",
        "input.file",
    }

    kept, skipped = filter_overrides_by_config(overrides, allowed)

    assert kept == {"surface.force_to_thickness.file": "/tmp/boot.nc"}
    assert skipped == [
        "surface.debm_simple.std_dev.file",
        "surface.pdd.std_dev.file",
    ]


def test_filter_keeps_all_when_all_present():
    """All override keys present in ``allowed_keys`` are kept; skipped is empty."""
    overrides = {"input.file": "/tmp/boot.nc", "grid.file": "/tmp/grid.nc"}
    allowed = {"input.file", "grid.file", "atmosphere.given.file"}

    kept, skipped = filter_overrides_by_config(overrides, allowed)

    assert kept == overrides
    assert skipped == []


def test_filter_empty_overrides():
    """An empty overrides dict yields an empty kept dict and no skipped keys."""
    kept, skipped = filter_overrides_by_config({}, {"input.file"})
    assert kept == {}
    assert skipped == []


def test_filter_empty_allowed_drops_everything():
    """If no keys are allowed, every override is skipped (sorted)."""
    overrides = {"b": 2, "a": 1}
    kept, skipped = filter_overrides_by_config(overrides, set())
    assert kept == {}
    assert skipped == ["a", "b"]


def test_filter_accepts_iterable_allowed_keys():
    """``allowed_keys`` may be any iterable, not just a set."""
    overrides = {"a": 1, "b": 2, "c": 3}
    kept, skipped = filter_overrides_by_config(overrides, ["a", "c"])
    assert kept == {"a": 1, "c": 3}
    assert skipped == ["b"]


def test_filter_preserves_override_values():
    """Filtering should not coerce or transform override values."""
    sentinel = object()
    overrides = {"input.file": sentinel}
    kept, _ = filter_overrides_by_config(overrides, {"input.file"})
    assert kept["input.file"] is sentinel


def test_filter_skipped_is_sorted():
    """``skipped`` is sorted so callers can rely on deterministic logging order."""
    overrides = {"z": 1, "a": 2, "m": 3}
    _, skipped = filter_overrides_by_config(overrides, set())
    assert skipped == ["a", "m", "z"]
