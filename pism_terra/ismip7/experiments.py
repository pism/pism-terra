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
ISMIP7 Core Experiment definitions keyed by ``counter_id``.

A single ISMIP7 ``counter_id`` (``C001`` … ``C008``) uniquely determines the
experiment identity: the ISMIP7 ``experiment_id`` of the submitted product, the
forcing ``pathway`` used to stage the climate/ocean inputs, the ESM (``gcm``),
and the projection end year. Encoding that mapping here lets one config field
(``run_info.counter``) drive both staging (``pism_terra/ismip7/greenland/stage.py``)
and running (``pism_terra/ismip7/greenland/run.py``), removing the previous
redundancy between ``run_info.experiment`` and ``[campaign].pathway`` / ``gcms``.

Source: the "Core Experiment Overview" table in the ISMIP7 Protocol Overview
spreadsheet (updated 2026-06-29), rows C001–C008.

Notes on the two Historical entries (C001/C002): the protocol asks for the
historical run only, but we still continue into a projection (ssp585 forcing,
ending 2100). For those counters the **historical** leg is the ISMIP7 product and
the projection leg is an internal continuation (flat filenames). For C003–C008 the
**projection** leg is the ISMIP7 product and the historical spin-up leg is kept but
written with flat filenames.

C009–C011 (CTRL2015 control run and OCX observationally-constrained experiment)
need distinct forcing/time handling and are intentionally omitted for now.

This module is deliberately dependency-free (it must not import
:mod:`pism_terra.config`) so it can be imported from the config resolver without
creating an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoreExperiment:
    """
    Resolved identity of one ISMIP7 Core experiment.

    Attributes
    ----------
    experiment_id : str
        ISMIP7 ``experiment_id`` of the submitted product leg, one of
        ``"historical"``, ``"ssp126"``, ``"ssp370"``, ``"ssp585"``.
    pathway : str
        Forcing pathway used to build the staged input filenames
        ``ismip7_greenland_{forcing}_{pathway}_{gcm}_{version}.nc``. For the
        Historical counters this is ``"ssp585"`` (the pathway whose forcing the
        internal continuation uses); the historical climate is embedded in that
        merged forcing file.
    esm_id : str
        The Earth System Model (GCM), e.g. ``"CESM2-WACCM"`` or ``"MRI-ESM2-0"``.
    proj_end_year : int
        Calendar year the projection leg ends (``2100`` or ``2300``).
    product_leg : str
        Which of the two forward legs is the ISMIP7 submission product, either
        ``"historical"`` or ``"projection"``. The other leg is written with flat
        (non-ISMIP7) filenames.
    """

    experiment_id: str
    pathway: str
    esm_id: str
    proj_end_year: int
    product_leg: str


# Core Experiment Overview (ISMIP7 Protocol Overview, updated 2026-06-29), C001–C008.
CORE_EXPERIMENTS: dict[str, CoreExperiment] = {
    "C001": CoreExperiment("historical", "ssp585", "CESM2-WACCM", 2100, "historical"),
    "C002": CoreExperiment("historical", "ssp585", "MRI-ESM2-0", 2100, "historical"),
    "C003": CoreExperiment("ssp370", "ssp370", "CESM2-WACCM", 2100, "projection"),
    "C004": CoreExperiment("ssp370", "ssp370", "MRI-ESM2-0", 2100, "projection"),
    "C005": CoreExperiment("ssp126", "ssp126", "CESM2-WACCM", 2300, "projection"),
    "C006": CoreExperiment("ssp126", "ssp126", "MRI-ESM2-0", 2300, "projection"),
    "C007": CoreExperiment("ssp585", "ssp585", "CESM2-WACCM", 2300, "projection"),
    "C008": CoreExperiment("ssp585", "ssp585", "MRI-ESM2-0", 2300, "projection"),
}


def resolve_counter(counter: str) -> CoreExperiment:
    """
    Look up the :class:`CoreExperiment` for an ISMIP7 ``counter_id``.

    Parameters
    ----------
    counter : str
        ISMIP7 counter id, e.g. ``"C003"`` (case-insensitive).

    Returns
    -------
    CoreExperiment
        The resolved experiment definition.

    Raises
    ------
    ValueError
        If ``counter`` is not a supported Core experiment id.
    """
    key = str(counter).strip().upper()
    try:
        return CORE_EXPERIMENTS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(CORE_EXPERIMENTS))
        raise ValueError(f"unknown ISMIP7 counter {counter!r}; supported: {supported}") from exc
