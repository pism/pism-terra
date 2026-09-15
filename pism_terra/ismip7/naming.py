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
ISMIP7 submission directory and file naming (see the conventions doc, section 8).

Directory structure::

    <domain_id>/<source_id>/<ism_id>/<set_id>/<set_counter>/

File name::

    <variable_id>_<domain_id>_<source_id>_<ism_id>_<ISM_member_id>_<ESM_id>
    _<forcing_member_id>_<experiment_id>_<set_counter>_<time_range>.nc
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# First letter of ``set_counter`` for each ISMIP7 set type.
_SET_LETTER = {"CORE": "C", "ESM": "E", "PPE": "P"}


def member_ids(set_id: str, sample: int) -> tuple[str, str, str]:
    """
    Derive ``(set_counter, ISM_member_id, forcing_member_id)`` from a sample index.

    The ``set_counter`` always increments with the 0-based ``sample`` (it is a
    unique index within the set). Which member id varies depends on the set type
    (see the ISMIP7 conventions, section 8):

    * ``CORE`` -- both members fixed at ``m001`` / ``f001``.
    * ``PPE``  -- ICE member ``mNNN`` varies (perturbed ICE physics), forcing
      fixed at ``f001``.
    * ``ESM``  -- forcing member ``fNNN`` varies (different ESM/forcing), ICE
      fixed at ``m001``.

    A single (non-ensemble) run yields the first counter, e.g.
    ``("C001", "m001", "f001")`` for a CORE set.

    Parameters
    ----------
    set_id : str
        ISMIP7 set type, one of ``"CORE"``, ``"ESM"``, ``"PPE"``.
    sample : int
        0-based ensemble member index.

    Returns
    -------
    tuple of str
        ``(set_counter, ism_member_id, forcing_member_id)``, e.g.
        ``("C001", "m001", "f001")``.

    Raises
    ------
    ValueError
        If ``set_id`` is not a recognized ISMIP7 set type.
    """
    key = set_id.upper()
    try:
        letter = _SET_LETTER[key]
    except KeyError as exc:
        raise ValueError(f"set_id must be one of {sorted(_SET_LETTER)}, got {set_id!r}") from exc
    n = int(sample) + 1
    set_counter = f"{letter}{n:03d}"
    ism_member = f"m{n:03d}" if key == "PPE" else "m001"
    forcing_member = f"f{n:03d}" if key == "ESM" else "f001"
    return set_counter, ism_member, forcing_member


@dataclass(frozen=True)
class ISMIP7Names:  # pylint: disable=too-many-instance-attributes
    """
    Resolved ISMIP7 naming components for one model run/member.

    Attributes
    ----------
    domain_id : str
        ``"GrIS"`` or ``"AIS"``.
    source_id : str
        Modelling group (no underscores/dots/special characters).
    ism_id : str
        Ice-sheet model name and version (no underscores/dots/special characters).
    ism_member_id : str
        ICE member variant, ``mNNN``.
    esm_id : str
        Forcing CMIP/ESM model name, e.g. ``"CESM2-WACCM"``.
    forcing_member_id : str
        Forcing variant, ``fNNN``.
    experiment_id : str
        e.g. ``"historical"``, ``"ssp370"``, ``"ctrl"``.
    set_id : str
        ``"CORE"``/``"ESM"``/``"PPE"``.
    set_counter : str
        Unique index within the set, ``CNNN``/``ENNN``/``PNNN``.
    time_range : str
        Year or year range, e.g. ``"1980-2014"``.
    """

    domain_id: str
    source_id: str
    ism_id: str
    ism_member_id: str
    esm_id: str
    forcing_member_id: str
    experiment_id: str
    set_id: str
    set_counter: str
    time_range: str

    def directory(self, root: Path | str = ".") -> Path:
        """
        Return the ISMIP7 submission directory for this run.

        Parameters
        ----------
        root : pathlib.Path or str, default ``"."``
            Base directory the ISMIP7 tree is created under.

        Returns
        -------
        pathlib.Path
            ``<root>/<domain_id>/<source_id>/<ism_id>/<set_id>/<set_counter>``.
        """
        return Path(root) / self.domain_id / self.source_id / self.ism_id / self.set_id / self.set_counter

    def stem(self) -> str:
        """
        Return the filename stem shared by all variables of this run.

        Returns
        -------
        str
            Everything after ``<variable_id>_`` in the file name, i.e.
            ``<domain>_<source>_<ism>_<ISM_member>_<ESM>_<forcing_member>_
            <experiment>_<set_counter>_<time_range>``.
        """
        return "_".join(
            (
                self.domain_id,
                self.source_id,
                self.ism_id,
                self.ism_member_id,
                self.esm_id,
                self.forcing_member_id,
                self.experiment_id,
                self.set_counter,
                self.time_range,
            )
        )

    def filename(self, variable_id: str) -> str:
        """
        Return the full ISMIP7 file name for one variable.

        Parameters
        ----------
        variable_id : str
            Variable short name (as defined in the ISMIP7 data request).

        Returns
        -------
        str
            ``<variable_id>_<stem>.nc``.
        """
        return f"{variable_id}_{self.stem()}.nc"
