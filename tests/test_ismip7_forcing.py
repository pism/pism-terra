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

# pylint: disable=protected-access

"""
Tests for the ISMIP7 GrIS forcing prep against the source.coop mirror.

Config expansion (per-GCM ``source``/``version`` with pathway overrides),
the source-spec split, remote paths, and the ETag-based cache-refresh
decision. Everything runs offline.
"""

from pathlib import Path

import pytest
import toml

from pism_terra.ismip7.greenland import forcing

CONFIG_DIR = Path(__file__).parents[1] / "pism_terra" / "config"


def test_split_source_spec():
    """
    Split plain, slash-form, and absent specs correctly.
    """
    assert forcing._split_source_spec("SDBN1-1000m") == (None, "SDBN1-1000m")
    assert forcing._split_source_spec("RACMO2.3p2-ERA/SDBN1-1000m") == ("RACMO2.3p2-ERA", "SDBN1-1000m")
    assert forcing._split_source_spec("none") == (None, "none")
    assert forcing._split_source_spec(None) == (None, "none")
    assert forcing._split_source_spec("") == (None, "none")


def test_remote_var_dir():
    """
    Standard trees keep the pathway segment; OCX-style trees replace it.
    """
    assert (
        forcing._remote_var_dir("CESM2-WACCM", "historical", "SDBN1-1000m", "acabf")
        == f"{forcing.SOURCE_COOP_PREFIX}/CESM2-WACCM/historical/SDBN1-1000m/acabf"
    )
    assert (
        forcing._remote_var_dir("OCX", "historical", "SDBN1-1000m", "acabf", source="RACMO2.3p2-ERA")
        == f"{forcing.SOURCE_COOP_PREFIX}/OCX/RACMO2.3p2-ERA/SDBN1-1000m/acabf"
    )


def test_forcing_tasks_from_shipped_config():
    """
    The shipped setup TOML expands into per-forcing tasks with split specs.
    """
    config = toml.loads((CONFIG_DIR / "setup_ismip7_greenland.toml").read_text("utf-8"))
    tasks = forcing._forcing_tasks(config)
    by_key = {
        (gcm, pathway, fc): (version, start, end, short_hand, source)
        for (_, gcm, fc, version, pathway, start, end, short_hand, _, source) in tasks
    }
    fields_by_key = {(gcm, pathway, fc): fields for (_, gcm, fc, _, pathway, _, _, _, fields, _) in tasks}

    # 2 standard GCMs x 5 pathways x 2 forcings + OCX historical x 2 forcings
    assert len(tasks) == 2 * 5 * 2 + 2

    # Climate and ocean are published on independent version tracks (per-
    # forcing ``version`` inside the source spec), matching the newest
    # upstream tags on source.coop as of 2026-09-03.
    assert by_key[("CESM2-WACCM", "historical", "climate")] == ("v3", 1900, 2014, "SDBN1-1000m", None)
    assert by_key[("CESM2-WACCM", "ssp585", "ocean")] == ("v2", 2015, 2300, "ocean-1000m", None)
    assert by_key[("MRI-ESM2-0", "ssp126", "climate")] == ("v2", 2015, 2300, "GEMB-SDBN1-1000m", None)
    assert by_key[("MRI-ESM2-0", "historical", "ocean")] == ("v1", 1900, 2014, "ocean-1000m", None)
    assert by_key[("OCX", "historical", "climate")] == ("v1", 1958, 2024, "SDBN1-1000m", "RACMO2.3p2-ERA")
    assert by_key[("OCX", "historical", "ocean")] == ("v1", 1958, 2024, "ocean-1000m", "EN4")

    # CTRL2015 (C009/C010) runs 2015-2300 off the same subtrees and version
    # tags as the projections. ``mrro`` appeared under ctrl upstream on
    # 2026-09-11, so ctrl now takes the same climate field list as the rest.
    assert by_key[("CESM2-WACCM", "ctrl", "climate")] == ("v3", 2015, 2300, "SDBN1-1000m", None)
    assert by_key[("MRI-ESM2-0", "ctrl", "ocean")] == ("v1", 2015, 2300, "ocean-1000m", None)
    for gcm in ("CESM2-WACCM", "MRI-ESM2-0"):
        assert fields_by_key[(gcm, "ctrl", "climate")] == fields_by_key[(gcm, "ssp585", "climate")]
        assert "mrro" in fields_by_key[(gcm, "ctrl", "climate")]
    assert fields_by_key[("MRI-ESM2-0", "ctrl", "ocean")] == ["tf", "so"]

    # The GCM-level ``source``/``version`` keys must not be mistaken for pathways.
    assert not [t for t in tasks if t[4] in ("source", "version")]


def test_select_forcing_tasks_narrows_to_one_corner():
    """
    Select a single (pathway, forcing) pair out of the shipped expansion.

    This is what a rerun uses after a variable appears upstream: regenerate
    the two ctrl climate files and leave the other twenty tasks alone.
    """
    config = toml.loads((CONFIG_DIR / "setup_ismip7_greenland.toml").read_text("utf-8"))
    tasks = forcing._forcing_tasks(config)

    selected = forcing.select_forcing_tasks(tasks, pathways="ctrl", forcings="climate")
    assert [(t[1], t[4], t[2]) for t in selected] == [
        ("CESM2-WACCM", "ctrl", "climate"),
        ("MRI-ESM2-0", "ctrl", "climate"),
    ]

    # Case-insensitive, and comma-separated lists combine.
    assert len(forcing.select_forcing_tasks(tasks, gcms="cesm2-waccm", pathways="CTRL,ssp585")) == 4

    # An unset selector is not a filter at all.
    assert forcing.select_forcing_tasks(tasks) == tasks
    assert forcing.select_forcing_tasks(tasks, gcms="", pathways=None) == tasks


def test_select_forcing_tasks_rejects_a_selector_that_matches_nothing():
    """
    Fail loudly on a typo rather than quietly doing no work.
    """
    config = toml.loads((CONFIG_DIR / "setup_ismip7_greenland.toml").read_text("utf-8"))
    tasks = forcing._forcing_tasks(config)
    with pytest.raises(SystemExit, match="no forcing task matches"):
        forcing.select_forcing_tasks(tasks, gcms="CESM2-WACM")


def test_forcing_tasks_pathway_overrides_and_legacy_short_hand():
    """
    Pathway-level source/version override GCM-level; old TOMLs still expand.
    """
    config = {
        "ice_sheet": "GrIS",
        "gcms": {
            "MRI-ESM2-0": {
                "historical": {"start": 1900, "end": 2014},
                "ssp585": {"start": 2015, "end": 2300, "source": {"climate": "SDBN1-1000m"}, "version": 1},
                "ctrl": {"start": 2015, "end": 2300, "fields": {"climate": ["acabf"]}},
                "source": {"climate": "GEMB-SDBN1-1000m", "ocean": "ocean-1000m"},
                "version": 2,
            },
            # Legacy shape: version per pathway, short_hand from [forcing].
            "CESM2-WACCM": {"historical": {"start": 1980, "end": 1990, "version": 2}},
        },
        "forcing": {
            "climate": {"fields": ["acabf"], "short_hand": "SDBN1-1000m"},
            "ocean": {"fields": ["tf"]},
        },
    }
    tasks = forcing._forcing_tasks(config)
    by_key = {
        (gcm, pathway, fc): (version, short_hand, source)
        for (_, gcm, fc, version, pathway, _, _, short_hand, _, source) in tasks
    }
    fields_by_key = {(gcm, pathway, fc): fields for (_, gcm, fc, _, pathway, _, _, _, fields, _) in tasks}

    # A pathway-level ``fields`` entry replaces the ``[forcing]`` list for that
    # forcing only; every other pathway/forcing keeps the global one.
    assert fields_by_key[("MRI-ESM2-0", "ctrl", "climate")] == ["acabf"]
    assert fields_by_key[("MRI-ESM2-0", "ctrl", "ocean")] == ["tf"]
    assert fields_by_key[("MRI-ESM2-0", "ssp585", "climate")] == ["acabf"]

    # The override replaces climate but the GCM-level ocean entry survives.
    assert by_key[("MRI-ESM2-0", "ssp585", "climate")] == ("v1", "SDBN1-1000m", None)
    assert by_key[("MRI-ESM2-0", "ssp585", "ocean")] == ("v1", "ocean-1000m", None)
    assert by_key[("MRI-ESM2-0", "historical", "climate")] == ("v2", "GEMB-SDBN1-1000m", None)
    assert by_key[("CESM2-WACCM", "historical", "climate")] == ("v2", "SDBN1-1000m", None)
    # No source entry and no legacy short_hand: the segment is absent.
    assert by_key[("CESM2-WACCM", "historical", "ocean")] == ("v2", "none", None)


def test_fetch_one_retries_transient_failures(tmp_path, monkeypatch):
    """
    A flaky download is retried with backoff and still lands atomically.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    monkeypatch : pytest.MonkeyPatch
        Used to silence the backoff sleep.
    """
    monkeypatch.setattr(forcing.time, "sleep", lambda _s: None)
    local = tmp_path / "acabf_GrIS_CESM2-WACCM_historical_SDBN1-1000m_v3_1917.nc"
    remote = {"ETag": '"abc"', "size": 4, "LastModified": "2026-08-29"}
    calls = {"n": 0}

    class _FlakyFS:
        """Fake filesystem that times out twice before succeeding."""

        def get_file(self, _rkey, lpath):
            """
            Write the file on the third attempt only.

            Parameters
            ----------
            _rkey : str
                Ignored remote key.
            lpath : str
                Local destination path.

            Raises
            ------
            TimeoutError
                On the first two attempts.
            """
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("socket starved")
            Path(lpath).write_bytes(b"data")

    result = forcing._fetch_one(_FlakyFS(), "bucket/key.nc", local, remote)
    assert result == local
    assert calls["n"] == 3
    assert local.read_bytes() == b"data"
    assert not local.with_suffix(".nc.part").exists()
    assert forcing._meta_path(local).exists()

    # A permanently failing download re-raises after the last attempt.
    calls["n"] = -100
    local2 = tmp_path / "always_fails.nc"
    with pytest.raises(TimeoutError):
        forcing._fetch_one(_FlakyFS(), "bucket/key2.nc", local2, remote, attempts=2)
    assert not local2.exists()


def test_needs_download_decisions(tmp_path):
    """
    Match by sidecar ETag; adopt sidecar-less files on a size match.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    remote = {"ETag": '"abc123"', "size": 10, "LastModified": "2026-08-29"}
    local = tmp_path / "tf_GrIS_EN4_OCX_ocean-1000m_v1_1958.nc"

    # Missing locally.
    assert forcing._needs_download(local, remote)

    # Present with matching size but no sidecar: adopted, sidecar written.
    local.write_bytes(b"0123456789")
    assert not forcing._needs_download(local, remote)
    assert forcing._meta_path(local).exists()

    # Sidecar now matches: still current.
    assert not forcing._needs_download(local, remote)

    # Upstream changed (new ETag): must re-download.
    assert forcing._needs_download(local, {**remote, "ETag": '"def456"'})

    # Upstream changed (new size): must re-download.
    assert forcing._needs_download(local, {**remote, "size": 11})

    # Present with wrong size and no sidecar: must re-download.
    forcing._meta_path(local).unlink()
    assert forcing._needs_download(local, {**remote, "size": 11})


def test_unit_overrides_are_udunits_parsable():
    """
    Every normalized unit is a single UDUNITS token PISM can parse.

    The published ``tf`` units alternate between ``deg_C`` and ``deg C``
    across GCM/pathway trees, and UDUNITS reads the space in the latter as
    multiplication by an undefined ``deg``. Guard against a replacement that
    reintroduces whitespace, and against dropping either variable from the
    map.

    Returns
    -------
    None
        Asserts only.
    """
    assert forcing.UNIT_OVERRIDES["tf"] == "deg_C"
    assert forcing.UNIT_OVERRIDES["so"] == "g/kg"
    for m_var, units in forcing.UNIT_OVERRIDES.items():
        assert " " not in units, f"{m_var}: {units!r} is not a single UDUNITS token"
