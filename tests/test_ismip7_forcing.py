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

    # 2 standard GCMs x 4 pathways x 2 forcings + OCX historical x 2 forcings
    assert len(tasks) == 2 * 4 * 2 + 2

    assert by_key[("CESM2-WACCM", "historical", "climate")] == ("v3", 1900, 2014, "SDBN1-1000m", None)
    assert by_key[("CESM2-WACCM", "ssp585", "ocean")] == ("v3", 2015, 2300, "ocean-1000m", None)
    assert by_key[("MRI-ESM2-0", "ssp126", "climate")] == ("v2", 2015, 2300, "GEMB-SDBN1-1000m", None)
    assert by_key[("OCX", "historical", "climate")] == ("v1", 1958, 2024, "SDBN1-1000m", "RACMO2.3p2-ERA")
    assert by_key[("OCX", "historical", "ocean")] == ("v1", 1958, 2024, "ocean-1000m", "EN4")

    # The GCM-level ``source``/``version`` keys must not be mistaken for pathways.
    assert not [t for t in tasks if t[4] in ("source", "version")]


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
