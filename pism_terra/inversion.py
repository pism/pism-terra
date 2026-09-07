"""Helpers shared by the inverse-modelling run generators."""

from __future__ import annotations

from typing import Any


def inversion_uses_hardav(inv: dict[str, Any]) -> bool:
    """
    Return True if a ``pismi`` option dict inverts for the ice hardness.

    That is the case for an alternating tauc/hardav inversion
    (``inverse.alternating_cycles > 0``) or a plain hardness inversion
    (``-inv_design hardav``). Forward legs then need to regrid ``hardav`` from
    the inversion output and enable ``stress_balance.averaged_hardness.enabled``.

    Parameters
    ----------
    inv : dict
        Options passed to ``pismi`` (keys as on the command line, without the
        leading dash).

    Returns
    -------
    bool
        True if the inversion produces a ``hardav`` field.
    """
    try:
        cycles = int(inv.get("inverse.alternating_cycles", 0))
    except (TypeError, ValueError):
        cycles = 0
    design = str(inv.get("inv_design", inv.get("inverse.design", "tauc"))).lower()
    return cycles > 0 or design == "hardav"
