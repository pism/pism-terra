"""Helpers shared by the inverse-modelling run generators."""

from __future__ import annotations

from typing import Any

#: Design variables ``pismi`` can invert for on their own.
DESIGN_VARIABLES = ("tauc", "hardav")


def inverted_variables(inv: dict[str, Any]) -> tuple[str, ...]:
    """
    Return the physical fields a ``pismi`` option dict inverts for.

    Follows ``inverse.design.variable`` (or its short option ``inv_design``)
    exactly as ``pismi`` does: ``tauc`` or ``hardav`` for a single inversion,
    and a pair joined by an underscore (``tauc_hardav``, ``hardav_tauc``) for
    an alternating one, in that order. ``inverse.alternating_cycles`` plays
    no part: ``pismi`` ignores it for a single variable, and the forward leg
    must regrid only the fields the inversion actually wrote.

    Parameters
    ----------
    inv : dict
        Options passed to ``pismi`` (keys as on the command line, without the
        leading dash).

    Returns
    -------
    tuple of str
        The inverted fields in phase order; ``("tauc",)`` when unset.

    Raises
    ------
    ValueError
        If the design variable is neither a known variable nor a pair of them.
    """
    design = str(inv.get("inverse.design.variable", inv.get("inv_design", "tauc"))).strip().lower()
    parts = tuple(design.split("_"))
    if len(parts) == 2 and sorted(parts) == sorted(DESIGN_VARIABLES):
        return parts
    if design in DESIGN_VARIABLES:
        return (design,)
    raise ValueError(
        f"inverse.design.variable must be one of {', '.join(DESIGN_VARIABLES)}, "
        f"{DESIGN_VARIABLES[0]}_{DESIGN_VARIABLES[1]} or {DESIGN_VARIABLES[1]}_{DESIGN_VARIABLES[0]}; got {design!r}"
    )


def inversion_uses_hardav(inv: dict[str, Any]) -> bool:
    """
    Return True if a ``pismi`` option dict inverts for the ice hardness.

    Forward legs then need to regrid ``hardav`` from the inversion output and
    enable ``stress_balance.averaged_hardness.enabled``.

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
    return "hardav" in inverted_variables(inv)


def forward_leg_from_inversion(run: dict[str, Any], inv: dict[str, Any], inv_file: Any) -> None:
    """
    Wire a forward run to hold the inverted fields fixed.

    Regrids exactly the fields the inversion wrote (:func:`inverted_variables`)
    from ``inv_file``. ``tauc`` is held fixed by the constant yield-stress
    model whether or not it was inverted: a hardness inversion saw the
    ``tauc`` of its input state, which the constant model reads back from the
    same state, so the forward leg matches what the inversion assumed. When
    ``hardav`` was inverted the Blatter solver uses it instead of the
    enthalpy-derived hardness.

    Parameters
    ----------
    run : dict
        Forward-leg PISM options, modified in place.
    inv : dict
        The ``pismi`` options of the inversion leg.
    inv_file : pathlib.Path or str
        The inversion output file.
    """
    variables = inverted_variables(inv)
    run.update({"input.regrid.file": inv_file, "input.regrid.vars": ",".join(variables)})
    run["basal_yield_stress.model"] = "constant"
    for key in [k for k in run if k.startswith("basal_yield_stress.mohr_coulomb.")]:
        run.pop(key)
    if "hardav" in variables:
        run["stress_balance.averaged_hardness.enabled"] = "yes"
