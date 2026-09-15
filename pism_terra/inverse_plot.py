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
Side-by-side maps of the fields of a PISM inversion ensemble.

The L-curve reduces each member to two numbers; these maps are the half its
corner cannot show — whether weak regularization is printing observational
noise onto the design variable, or strong regularization is smoothing real
sticky spots away.

One figure per field, each a grid of panels (four per row by default)
ordered by the swept parameter, every panel on one shared color scale
computed across all members. That
shared scale is the whole point: scaling each panel to its own data would
make every member look alike. Three fields are available, written to separate
files:

- ``design`` — the field the run inverted for, ``tauc`` or ``hardav``, on a
  logarithmic scale, since a penalty sweep moves it over several decades.
- ``zeta`` — the parameterized design variable the inversion actually
  optimizes (``tauc = tauc_scale * exp(zeta)`` under the default ``exp``
  parameterization), so it is signed and centered on zero: Crameri's
  ``broc``, symmetric about zero, showing where the inversion pushed the
  field above (positive) and below (negative) ``tauc_scale``.
- ``residual`` — ``inv_residual``, the velocity misfit in m/yr, linear
  because it reaches zero where the model fits the observations.

```bash
pism-inverse-plot --parameters inverse.tikhonov.penalty_weight \
    -o maps.png inv_g*.nc
```

writes ``maps_tauc.png``, ``maps_zeta_inv.png`` and ``maps_inv_residual.png``.

An alternating ``tauc``/``hardav`` co-inversion solves for both, so the
design variable and zeta each get a figure per phase — ``maps_tauc.png``,
``maps_hardav.png``, ``maps_zeta_inv_tauc.png``, ``maps_zeta_inv_hardav.png``
— while the residual stays shared. ``--design-variable`` takes a
comma-separated subset when only one phase is wanted. A field no member has
reached yet is skipped rather than half-drawn, so asking for ``--variables
zeta --design-variable tauc`` will map a sweep that is still in its first
cycle. A phase the run has not reached is warned about rather than passed off
as a result: until ``pismi`` has inverted for ``hardav``, the ``hardav`` in
the file is the prior it computed from enthalpy.

The design variable and zeta are masked to the cells the inversion was free
to change (``zeta_fixed_mask == 0``) and the residual to the misfit area PISM
actually fit (``vel_misfit_weight > 0``) — elsewhere the field is just the
prior, and plotting it would dominate the shared color scale. Members that
are still running, or that crashed before writing the fields, are skipped
with a warning.
"""

from __future__ import annotations

import logging
import math
import re
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Imported for its side effect: registering Crameri's "cmc.*" colormaps.
import cmcrameri.cm  # noqa: F401  pylint: disable=unused-import
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm, Normalize

from pism_terra.lcurve import (
    DEFAULT_PARAMETERS,
    DESIGN_VARIABLES,
    design_units,
    design_variables,
    fontsize,
    mathtext_units,
    rc_params,
    short_name,
)
from pism_terra.log import setup_logging

xr.set_options(keep_attrs=True)
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)

logger = logging.getLogger("pism_terra.inverse_plot")

# Mask candidates, as (mask variable, value kept), tried in order — the first
# one the file carries wins; ``inf`` means "every positive value".
#
# The design variable and its parameterization are only a result where the
# inversion was free to change them: outside ``zeta_fixed_mask`` they still
# hold the prior, which would otherwise dominate the color scale. Ice
# thickness is the fallback, and a poor one on a bootstrapped domain where
# every cell carries a sliver of ice. The residual is only defined on the
# cells PISM actually fit.
FREE_CELL_MASKS = (("zeta_fixed_mask", 0.0), ("thk", np.inf))
MISFIT_AREA_MASKS = (("vel_misfit_weight", np.inf),)


@dataclass(frozen=True)
class FieldSpec:
    """
    How one plotted field is read, masked, scaled and labelled.

    Attributes
    ----------
    variables : tuple of str
        Candidate variable names, tried in order; the literal ``{design}`` is
        replaced with the design variable the run inverted for. Plain
        substitution, not ``str.format``: the labels carry mathtext braces.
    label : str
        Axis and colorbar label; ``{design}`` is replaced as above.
    units : str
        Units for the colorbar; ``{design_units}`` is replaced with the
        design variable's units.
    cmap : str
        Default colormap.
    scale : str
        ``"log"``, ``"linear"`` or ``"diverging"`` (symmetric about zero).
    masks : tuple
        Mask candidates as in :data:`FREE_CELL_MASKS`.
    """

    variables: tuple[str, ...]
    label: str
    units: str
    cmap: str
    scale: str
    masks: tuple[tuple[str, float], ...]


FIELDS = {
    "design": FieldSpec(("{design}",), "{design}", "{design_units}", "viridis", "log", FREE_CELL_MASKS),
    # An alternating co-inversion names zeta per phase; a single-design run
    # writes the plain name.
    "zeta": FieldSpec(
        ("zeta_inv", "zeta_inv_{design}"),
        r"$\zeta$ ({design})",
        "1",
        "cmc.broc",
        "diverging",
        FREE_CELL_MASKS,
    ),
    "residual": FieldSpec(("inv_residual",), "inversion residual", "m yr$^{-1}$", "magma", "linear", MISFIT_AREA_MASKS),
}


def field_items(fields: list[str], designs: list[str]) -> list[tuple[str, str | None]]:
    """
    Expand field keys over the design variables they depend on.

    An alternating co-inversion solves for both ``tauc`` and ``hardav``, so
    the design variable and zeta each become one figure per phase; the
    residual is shared and stays single.

    Parameters
    ----------
    fields : list of str
        Keys of :data:`FIELDS`.
    designs : list of str
        Design variables the run solved for.

    Returns
    -------
    list of tuple
        ``(field key, design variable or None)`` pairs, one per figure.
    """
    items: list[tuple[str, str | None]] = []
    for key in fields:
        spec = FIELDS[key]
        if any("{design}" in variable for variable in spec.variables):
            items.extend((key, design) for design in designs)
        else:
            items.append((key, None))
    return items


def item_key(key: str, design: str | None) -> str:
    """
    Key one expanded field takes in a member dict.

    Parameters
    ----------
    key : str
        Field key, one of :data:`FIELDS`.
    design : str or None
        Design variable the field belongs to, or ``None`` when it is shared.

    Returns
    -------
    str
        ``"zeta:hardav"`` for a per-design field, ``"residual"`` otherwise.
    """
    return f"{key}:{design}" if design else key


def _slice2d(ds: xr.Dataset, name: str) -> np.ndarray:
    """
    Read one 2D field, dropping the degenerate time axis PISM writes.

    Parameters
    ----------
    ds : xarray.Dataset
        Inversion output.
    name : str
        Variable to read.

    Returns
    -------
    numpy.ndarray
        The field as a 2D float array.
    """
    return ds[name].squeeze().values.astype(float)


def _resolve(spec: FieldSpec, ds: xr.Dataset, design: str | None) -> str | None:
    """
    Find the first of a field's candidate variables that the file carries.

    Parameters
    ----------
    spec : FieldSpec
        Field being read.
    ds : xarray.Dataset
        Inversion output.
    design : str or None
        Design variable this figure belongs to; ``None`` for a shared field.

    Returns
    -------
    str or None
        Variable name, or ``None`` when the file carries none of them.
    """
    for candidate in spec.variables:
        name = candidate.replace("{design}", design or "")
        if name in ds:
            return name
    return None


def _apply_mask(ds: xr.Dataset, values: np.ndarray, spec: FieldSpec, filename: str, key: str) -> np.ndarray:
    """
    Blank out the cells where a field is not a result.

    Parameters
    ----------
    ds : xarray.Dataset
        Inversion output holding the mask variables.
    values : numpy.ndarray
        Field to mask.
    spec : FieldSpec
        Field being masked; supplies the mask candidates.
    filename : str
        File name, for the warning when no mask is available.
    key : str
        Field key, for the same warning.

    Returns
    -------
    numpy.ndarray
        The field with masked-out cells set to NaN, or unchanged when the
        file carries none of the mask variables.
    """
    for mask_var, keep in spec.masks:
        if mask_var not in ds:
            continue
        mask_values = _slice2d(ds, mask_var)
        selected = mask_values > 0 if np.isinf(keep) else mask_values == keep
        return np.where(selected, values, np.nan)
    logger.warning("%s: none of %s present, %s left unmasked", filename, ", ".join(v for v, _ in spec.masks), key)
    return values


def read_member(
    path: Path,
    parameters: list[str],
    fields: list[str],
    *,
    designs: list[str] | None = None,
    mask: bool = True,
) -> dict[str, Any] | None:
    """
    Read one member's fields and swept parameters.

    Parameters
    ----------
    path : pathlib.Path
        Inversion output file.
    parameters : list of str
        Dotted ``pism_config`` keys to read; they key the returned dict by
        their last dotted component.
    fields : list of str
        Keys of :data:`FIELDS` to read. A file missing any of them is
        skipped, so ask only for what will be plotted.
    designs : list of str or None, optional
        Design variables to read, or ``None`` to detect them per file with
        :func:`pism_terra.lcurve.design_variables`. An alternating
        co-inversion yields both, and each gets its own entry.
    mask : bool, optional
        Mask each field to where it is meaningful.

    Returns
    -------
    dict or None
        One entry per expanded field, keyed by :func:`item_key`, plus
        ``designs``, ``resolved`` (the variable each entry came from), ``x``,
        ``y``, ``file`` and one entry per parameter — or ``None`` when the
        file cannot contribute a panel, which is logged.
    """
    try:
        with xr.open_dataset(path) as ds:
            found = designs or design_variables(ds)
            if not found:
                logger.warning("%s: skipped, cannot tell tauc from hardav; pass --design-variable", path.name)
                return None
            config = ds["pism_config"].attrs
            absent = [p for p in parameters if p not in config]
            if absent:
                logger.warning("%s: skipped, pism_config has no %s", path.name, ", ".join(absent))
                return None

            member: dict[str, Any] = {short_name(p): float(config[p]) for p in parameters}
            resolved: dict[str, str] = {}
            for key, design in field_items(fields, found):
                spec = FIELDS[key]
                name = _resolve(spec, ds, design)
                if name is None:
                    logger.warning(
                        "%s: skipped, no %s field — none of %s present, so this phase is not written yet",
                        path.name,
                        item_key(key, design),
                        ", ".join(v.replace("{design}", design or "") for v in spec.variables),
                    )
                    return None
                values = _slice2d(ds, name)
                entry = item_key(key, design)
                member[entry] = _apply_mask(ds, values, spec, path.name, entry) if mask else values
                resolved[entry] = name
            member["designs"] = found
            member["resolved"] = resolved
            member["units"] = {d: design_units(ds, d) for d in found}
            _warn_unfinished_phases(ds, found, path.name)
            member["x"] = ds["x"].values
            member["y"] = ds["y"].values
            member["file"] = path.name
            return member
    except (OSError, KeyError, ValueError) as error:
        logger.warning("%s: skipped, %s", path.name, error)
        return None


def completed_phases(ds: xr.Dataset) -> set[str] | None:
    """
    Which phases of an alternating co-inversion have finished.

    ``pismi`` stamps the output with ``pismi_alternation_completed``, the last
    phase it got through, as ``c<cycle>_<design>``. Each cycle runs ``tauc``
    then ``hardav``, so finishing ``c0_hardav`` means both are done, while
    finishing ``c0_tauc`` means ``hardav`` has never been inverted — the
    ``hardav`` in the file is then still the prior computed from enthalpy, not
    a result.

    Parameters
    ----------
    ds : xarray.Dataset
        Inversion output.

    Returns
    -------
    set of str or None
        The design variables that have been inverted at least once, or
        ``None`` when the file carries no alternation stamp (a single-design
        run, or one that has not finished a phase).
    """
    stamp = str(ds.attrs.get("pismi_alternation_completed", ""))
    match = re.fullmatch(r"c(\d+)_(\w+)", stamp)
    if match is None:
        return None
    cycle, design = int(match.group(1)), match.group(2)
    if design == "hardav":
        return {"tauc", "hardav"}
    return {"tauc", "hardav"} if cycle >= 1 else {"tauc"}


def _warn_unfinished_phases(ds: xr.Dataset, designs: list[str], filename: str) -> None:
    """
    Warn when a design variable in the file was never actually inverted.

    Parameters
    ----------
    ds : xarray.Dataset
        Inversion output.
    designs : list of str
        Design variables being read.
    filename : str
        File name, for the warning.
    """
    done = completed_phases(ds)
    if done is None:
        return
    for design in designs:
        if design not in done:
            logger.warning(
                "%s: %s has not been inverted yet (alternation reached %s); its field is still the prior",
                filename,
                design,
                ds.attrs.get("pismi_alternation_completed"),
            )


def shared_limits(
    arrays: list[np.ndarray],
    *,
    percentile: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    scale: str = "linear",
) -> tuple[float, float]:
    """
    Compute color limits spanning every member, so the panels are comparable.

    Taken as percentiles over the pooled values rather than the outright
    min/max, so a handful of extreme cells in one member does not flatten the
    whole row.

    Parameters
    ----------
    arrays : list of numpy.ndarray
        One field per member; NaNs (masked cells) are ignored.
    percentile : float, optional
        Lower percentile; the upper is its complement. ``0`` gives the full
        range.
    vmin, vmax : float or None, optional
        Explicit overrides, each taking precedence over the percentile.
    scale : str, optional
        ``"log"`` drops non-positive values, which it cannot show;
        ``"diverging"`` returns limits symmetric about zero, so the sign is
        read off the colormap's midpoint.

    Returns
    -------
    tuple of float
        ``(vmin, vmax)``, guaranteed to be an increasing pair.

    Raises
    ------
    ValueError
        If every member is entirely masked, or a log scale is asked for a
        field with no positive values.
    """
    pooled = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    if pooled.size == 0:
        raise ValueError("every member is fully masked; nothing to scale the colors by")
    if scale == "log":
        pooled = pooled[pooled > 0]
        if pooled.size == 0:
            raise ValueError("no positive values; a log color scale is not possible")

    if scale == "diverging":
        extent = float(np.percentile(np.abs(pooled), 100.0 - percentile))
        low = -extent if vmin is None else vmin
        high = extent if vmax is None else vmax
    else:
        low = float(np.percentile(pooled, percentile)) if vmin is None else vmin
        high = float(np.percentile(pooled, 100.0 - percentile)) if vmax is None else vmax

    if low >= high:
        # A constant field (every member identical) still deserves a panel.
        low, high = (low - 0.5, high + 0.5) if low == high else (high, low)
    return low, high


def plot_field(
    members: list[dict[str, Any]],
    key: str,
    design: str | None,
    sweep: str,
    output_file: Path,
    *,
    cmap: str | None = None,
    scale: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    percentile: float = 1.0,
    ncols: int = 4,
    panel_width: float = 1.6,
    dpi: int = 300,
) -> Path:
    """
    Draw one field across the ensemble as a grid of panels, and write it.

    Parameters
    ----------
    members : list of dict
        Member dicts from :func:`read_member`, in plotting order.
    key : str
        Field to draw, a key of :data:`FIELDS`.
    design : str or None
        Design variable this figure belongs to; ``None`` for a shared field
        such as the residual.
    sweep : str
        Column naming the swept parameter; its value titles each panel.
    output_file : pathlib.Path
        Base path; the name of the variable actually plotted is appended to
        its stem, so ``maps.png`` becomes e.g. ``maps_tauc.png`` or
        ``maps_zeta_inv_hardav.png``.
    cmap : str or None, optional
        Colormap, defaulting to the field's own.
    scale : str or None, optional
        Color scale, defaulting to the field's own.
    vmin, vmax : float or None, optional
        Color limits, each defaulting to the percentile over all members.
    percentile : float, optional
        Percentile for the shared limits (see :func:`shared_limits`).
    ncols : int, optional
        Panels per row; the members wrap onto as many rows as they need.
    panel_width : float, optional
        Width of one panel, inches.
    dpi : int, optional
        Resolution of raster output.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    spec = FIELDS[key]
    entry = item_key(key, design)
    # The design variable names the field for a "design" figure; for zeta it
    # names the phase. Either way the resolved variable names the file, so an
    # alternating run's two zeta figures do not collide.
    label = spec.label.replace("{design}", design or "")
    units = mathtext_units(spec.units.replace("{design_units}", members[0]["units"].get(design or "", "")))
    variable = members[0]["resolved"][entry]
    scale = scale or spec.scale
    low, high = shared_limits([m[entry] for m in members], percentile=percentile, vmin=vmin, vmax=vmax, scale=scale)
    norm = LogNorm(vmin=low, vmax=high) if scale == "log" else Normalize(vmin=low, vmax=high)

    with mpl.rc_context(rc=rc_params):
        extent = [members[0]["x"][0], members[0]["x"][-1], members[0]["y"][0], members[0]["y"][-1]]
        aspect = abs((extent[3] - extent[2]) / (extent[1] - extent[0]))
        columns = max(1, min(ncols, len(members)))
        rows = math.ceil(len(members) / columns)
        fig, axs = plt.subplots(
            rows,
            columns,
            figsize=(panel_width * columns + 0.9, panel_width * aspect * rows + 0.6),
            squeeze=False,
            layout="constrained",
        )
        flat = axs.ravel()
        for ax, member in zip(flat, members):
            image = ax.imshow(
                member[entry],
                origin="lower",
                extent=extent,
                cmap=cmap or spec.cmap,
                norm=norm,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{sweep} = {member[sweep]:g}", fontsize=fontsize)
        # A partly-filled last row would otherwise show empty framed panels.
        for ax in flat[len(members) :]:
            ax.set_visible(False)
        for row in range(rows):
            axs[row][0].set_ylabel(label, fontsize=fontsize)
        colorbar = fig.colorbar(image, ax=list(flat), fraction=0.02, pad=0.01, extend="both")
        colorbar.set_label(f"{label} ({units})" if units else label, fontsize=fontsize)
        colorbar.ax.tick_params(labelsize=fontsize)

        path = output_file.with_name(f"{output_file.stem}_{variable}{output_file.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    logger.info("wrote %s", path)
    return path


def _in_every(item: tuple[str, str | None], members: list[dict[str, Any]]) -> bool:
    """
    Check that every member carries one expanded field.

    Parameters
    ----------
    item : tuple
        ``(field key, design variable or None)`` from :func:`field_items`.
    members : list of dict
        Member dicts from :func:`read_member`.

    Returns
    -------
    bool
        True when every member has the field, so the figure would be
        complete.
    """
    return all(item_key(*item) in member for member in members)


def main() -> None:
    """
    Run the ``pism-inverse-plot`` command line tool.

    Returns
    -------
    None
        One figure per requested field is written to disk.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = (
        "Maps of an inversion ensemble side by side, one figure per field: the inverted "
        "field (tauc or hardav), the parameterized design variable zeta, and the velocity "
        "residual. Each figure puts every member on one shared color scale so they can be "
        "compared by eye."
    )
    parser.add_argument(
        "--parameters",
        help="Comma-separated pism_config keys varied across the ensemble. The first one "
        "orders the panels and titles them.",
        type=str,
        default=",".join(DEFAULT_PARAMETERS),
    )
    parser.add_argument(
        "-o",
        "--output-file",
        help="Base path for the figures; each field's variable name is appended to the stem, "
        "so maps.png becomes maps_tauc.png, maps_zeta_inv.png and maps_inv_residual.png. "
        "The suffix picks the format.",
        type=str,
        default="inverse_maps.png",
    )
    parser.add_argument(
        "--variables",
        help=f"Comma-separated fields to plot, one figure each: {', '.join(FIELDS)}.",
        type=str,
        default=",".join(FIELDS),
    )
    parser.add_argument(
        "--design-variable",
        help=f"Comma-separated design variables to plot ({', '.join(DESIGN_VARIABLES)}), each getting its "
        "own figure. The default reads them from each file, which yields both phases of an "
        "alternating co-inversion.",
        type=str,
        default=None,
    )
    for key, spec in FIELDS.items():
        parser.add_argument(f"--{key}-cmap", help=f"Colormap for the {key} figure.", type=str, default=spec.cmap)
        parser.add_argument(f"--{key}-vmin", help=f"Lower color limit of the {key} figure.", type=float, default=None)
        parser.add_argument(f"--{key}-vmax", help=f"Upper color limit of the {key} figure.", type=float, default=None)
    parser.add_argument(
        "--linear",
        help="Linear color scale for the design variable instead of the default logarithmic one.",
        action="store_true",
    )
    parser.add_argument(
        "--percentile",
        help="Percentile trimmed off each end when the color limits are computed from the data.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--no-mask",
        help="Plot the full domain instead of masking to the inverted cells and the misfit area.",
        action="store_true",
    )
    parser.add_argument(
        "--ncols",
        help="Panels per row; the members wrap onto as many rows as they need.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--panel-width",
        help="Width of one panel, inches.",
        type=float,
        default=1.6,
    )
    parser.add_argument(
        "--dpi",
        help="Resolution of raster output.",
        type=int,
        default=300,
    )
    parser.add_argument(
        "INFILES",
        help="Inversion output files, one per ensemble member, e.g. inv_g*.nc.",
        nargs="+",
    )

    options = parser.parse_args()
    parameters = [p.strip() for p in options.parameters.split(",") if p.strip()]
    if not parameters:
        parser.error("--parameters needs at least one pism_config key")
    fields = [f.strip() for f in options.variables.split(",") if f.strip()]
    unknown = [f for f in fields if f not in FIELDS]
    if unknown or not fields:
        parser.error(f"--variables takes any of {', '.join(FIELDS)}; got {', '.join(unknown) or 'nothing'}")
    designs = None
    if options.design_variable:
        designs = [d.strip() for d in options.design_variable.split(",") if d.strip()]
        unknown = [d for d in designs if d not in DESIGN_VARIABLES]
        if unknown or not designs:
            parser.error(
                f"--design-variable takes any of {', '.join(DESIGN_VARIABLES)}; "
                f"got {', '.join(unknown) or 'nothing'}"
            )

    output_file = Path(options.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(output_file.parent / "inverse_plot.log")

    members = [
        member
        for member in (
            read_member(Path(f), parameters, fields, designs=designs, mask=not options.no_mask) for f in options.INFILES
        )
        if member is not None
    ]
    if not members:
        raise SystemExit(
            f"none of the {len(options.INFILES)} given files yielded a panel; "
            f"they need {', '.join(fields)} and {', '.join(parameters)}"
        )
    sweep = short_name(parameters[0])
    members.sort(key=lambda m: tuple(m[short_name(p)] for p in parameters))
    logger.info("plotting %d of %d files", len(members), len(options.INFILES))

    # Members may disagree on which phases they have reached, so plot only
    # the fields every one of them carries.
    items = [item for item in field_items(fields, members[0]["designs"]) if _in_every(item, members)]
    for key, design in items:
        path = plot_field(
            members,
            key,
            design,
            sweep,
            output_file,
            cmap=getattr(options, f"{key}_cmap"),
            scale="linear" if key == "design" and options.linear else None,
            vmin=getattr(options, f"{key}_vmin"),
            vmax=getattr(options, f"{key}_vmax"),
            percentile=options.percentile,
            ncols=options.ncols,
            panel_width=options.panel_width,
            dpi=options.dpi,
        )
        print(f"wrote {path} ({len(members)} members)")


if __name__ == "__main__":
    main()
