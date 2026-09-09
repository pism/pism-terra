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
Side-by-side maps of an inversion ensemble's design variable and residual.

The L-curve reduces each member to two numbers; this is the other half of
choosing a regularization parameter, which the L-curve's corner cannot tell
you: what the inverted field actually looks like. Weak regularization prints
observational noise onto ``tauc`` as speckle, strong regularization smooths
away real sticky spots, and only the maps show which is happening.

One row per field — the design variable (``tauc`` or ``hardav``, whichever
the run inverted for) on top, ``inv_residual`` below — with one panel per
ensemble member, ordered by the swept parameter. Every panel in a row shares
one color scale and colormap, so the panels are comparable by eye; that
shared scale is the whole point, and it is computed across all members rather
than per panel.

```bash
pism-inverse-plot --parameters inverse.tikhonov.penalty_weight \
    -o inverse_maps.png inv_g*.nc
```

By default the design variable is masked to the cells the inversion was free
to change (``zeta_fixed_mask == 0``) and the residual to the misfit area PISM
actually fit (``vel_misfit_weight > 0``) — elsewhere the field is just the
prior, and plotting it would dominate the shared color scale. The design
variable also gets a logarithmic scale, since a penalty sweep moves ``tauc``
over several decades. Members that are still running, or that crashed before
writing the fields, are skipped with a warning.
"""

from __future__ import annotations

import logging
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm, Normalize

from pism_terra.lcurve import (
    DEFAULT_PARAMETERS,
    DESIGN_VARIABLES,
    design_variable,
    fontsize,
    rc_params,
    short_name,
)
from pism_terra.log import setup_logging

xr.set_options(keep_attrs=True)
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)

logger = logging.getLogger("pism_terra.inverse_plot")

RESIDUAL_VAR = "inv_residual"

# Mask each field to where it means something, as (mask variable, value kept)
# candidates tried in order — the first one the file carries wins.
#
# The design variable is only a result where the inversion was free to change
# it: outside ``zeta_fixed_mask`` it still holds the prior, which would
# otherwise dominate the color scale. Ice thickness is the fallback, and a
# poor one on a bootstrapped domain where every cell carries a sliver of ice.
# The residual is only defined on the cells PISM actually fit.
MASKS: dict[str, tuple[tuple[str, float], ...]] = {
    "design": (("zeta_fixed_mask", 0.0), ("thk", np.inf)),
    RESIDUAL_VAR: (("vel_misfit_weight", np.inf),),
}


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


def read_member(path: Path, parameters: list[str], variable: str | None, mask: bool = True) -> dict[str, Any] | None:
    """
    Read one member's design variable, residual and swept parameters.

    Parameters
    ----------
    path : pathlib.Path
        Inversion output file.
    parameters : list of str
        Dotted ``pism_config`` keys to read; they key the returned dict by
        their last dotted component.
    variable : str or None
        Design variable to plot, or ``None`` to detect it per file with
        :func:`pism_terra.lcurve.design_variable`.
    mask : bool, optional
        Mask each field to where it is meaningful (see :data:`MASKS`).

    Returns
    -------
    dict or None
        Keys ``design`` (the field), ``design_name``, ``inv_residual``,
        ``x``, ``y``, ``file``, plus one entry per parameter — or ``None``
        when the file cannot contribute a panel, which is logged.
    """
    try:
        with xr.open_dataset(path) as ds:
            name = variable or design_variable(ds)
            if name is None:
                logger.warning("%s: skipped, cannot tell tauc from hardav; pass --variable", path.name)
                return None
            missing = [v for v in (name, RESIDUAL_VAR) if v not in ds]
            if missing:
                logger.warning("%s: skipped, missing %s", path.name, ", ".join(missing))
                return None
            config = ds["pism_config"].attrs
            absent = [p for p in parameters if p not in config]
            if absent:
                logger.warning("%s: skipped, pism_config has no %s", path.name, ", ".join(absent))
                return None

            member: dict[str, Any] = {short_name(p): float(config[p]) for p in parameters}
            fields = {"design": _slice2d(ds, name), RESIDUAL_VAR: _slice2d(ds, RESIDUAL_VAR)}
            if mask:
                for key, candidates in MASKS.items():
                    for mask_var, keep in candidates:
                        if mask_var not in ds:
                            continue
                        values = _slice2d(ds, mask_var)
                        # ``keep`` is either an exact flag value to keep
                        # (zeta_fixed_mask == 0, the free cells) or inf,
                        # meaning "every positive value".
                        selected = values > 0 if np.isinf(keep) else values == keep
                        fields[key] = np.where(selected, fields[key], np.nan)
                        break
                    else:
                        logger.warning(
                            "%s: none of %s present, %s left unmasked",
                            path.name,
                            ", ".join(v for v, _ in candidates),
                            key,
                        )
            member.update(fields)
            member["design_name"] = name
            member["x"] = ds["x"].values
            member["y"] = ds["y"].values
            member["file"] = path.name
            return member
    except (OSError, KeyError, ValueError) as error:
        logger.warning("%s: skipped, %s", path.name, error)
        return None


def shared_limits(
    arrays: list[np.ndarray],
    *,
    percentile: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    positive: bool = False,
) -> tuple[float, float]:
    """
    Color limits spanning every member, so the panels are comparable.

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
    positive : bool, optional
        Clip the lower limit to the smallest positive value, for a log scale
        that cannot show zero or negative values.

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
    if positive:
        pooled = pooled[pooled > 0]
        if pooled.size == 0:
            raise ValueError("no positive values; a log color scale is not possible")
    low = float(np.percentile(pooled, percentile)) if vmin is None else vmin
    high = float(np.percentile(pooled, 100.0 - percentile)) if vmax is None else vmax
    if low >= high:
        # A constant field (every member identical) still deserves a panel.
        low, high = (low - 0.5, high + 0.5) if low == high else (high, low)
    return low, high


def plot_members(
    members: list[dict[str, Any]],
    sweep: str,
    output_file: Path,
    *,
    cmaps: tuple[str, str] = ("viridis", "magma"),
    log: bool = True,
    limits: tuple[float | None, float | None, float | None, float | None] = (None, None, None, None),
    percentile: float = 1.0,
    panel_width: float = 1.6,
    dpi: int = 300,
) -> None:
    """
    Draw the design-variable and residual rows and write the figure.

    Parameters
    ----------
    members : list of dict
        Member dicts from :func:`read_member`, in plotting order.
    sweep : str
        Column naming the swept parameter; its value titles each panel.
    output_file : pathlib.Path
        Where to write the figure; the suffix picks the format.
    cmaps : tuple of str, optional
        Colormaps for the design-variable and residual rows.
    log : bool, optional
        Logarithmic color scale for the design variable. On by default: a
        penalty sweep moves ``tauc`` over several decades, and a linear scale
        then shows the weakly-regularized members as uniformly dark. The
        residual stays linear — it reaches zero where the model fits the
        observations, which a log scale cannot show.
    limits : tuple, optional
        ``(vmin, vmax, residual_vmin, residual_vmax)``; each ``None`` entry
        falls back to the percentile over all members.
    percentile : float, optional
        Percentile for the shared limits (see :func:`shared_limits`).
    panel_width : float, optional
        Width of one panel, inches.
    dpi : int, optional
        Resolution of raster output.
    """
    design_name = members[0]["design_name"]
    names = {m["design_name"] for m in members}
    if len(names) > 1:
        logger.warning(
            "members disagree on the design variable (%s); labelling as %s", ", ".join(sorted(names)), design_name
        )

    vmin, vmax, res_vmin, res_vmax = limits
    design_limits = shared_limits(
        [m["design"] for m in members], percentile=percentile, vmin=vmin, vmax=vmax, positive=log
    )
    residual_limits = shared_limits(
        [m[RESIDUAL_VAR] for m in members], percentile=percentile, vmin=res_vmin, vmax=res_vmax
    )
    rows = (
        ("design", design_name, DESIGN_VARIABLES.get(design_name, ""), cmaps[0], design_limits, log),
        (RESIDUAL_VAR, "inversion residual", "m yr$^{-1}$", cmaps[1], residual_limits, False),
    )

    with mpl.rc_context(rc=rc_params):
        extent = [members[0]["x"][0], members[0]["x"][-1], members[0]["y"][0], members[0]["y"][-1]]
        aspect = abs((extent[3] - extent[2]) / (extent[1] - extent[0]))
        fig, axs = plt.subplots(
            len(rows),
            len(members),
            figsize=(panel_width * len(members) + 0.9, panel_width * aspect * len(rows) + 0.5),
            squeeze=False,
            layout="constrained",
        )
        for row, (key, title, units, cmap, (low, high), row_log) in enumerate(rows):
            norm = LogNorm(vmin=low, vmax=high) if row_log else Normalize(vmin=low, vmax=high)
            for col, member in enumerate(members):
                ax = axs[row][col]
                image = ax.imshow(
                    member[key], origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest"
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f"{sweep} = {member[sweep]:g}", fontsize=fontsize)
                if col == 0:
                    ax.set_ylabel(title, fontsize=fontsize)
            colorbar = fig.colorbar(image, ax=list(axs[row]), fraction=0.02, pad=0.01, extend="both")
            colorbar.set_label(f"{title} ({units})" if units else title, fontsize=fontsize)
            colorbar.ax.tick_params(labelsize=fontsize)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=dpi)
        plt.close(fig)
    logger.info("wrote %s", output_file)


def main() -> None:
    """
    Run the ``pism-inverse-plot`` command line tool.

    Returns
    -------
    None
        The figure is written to disk.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = (
        "Maps of an inversion ensemble side by side: the inverted field (tauc or hardav) "
        "and the velocity residual, one panel per member, sharing one color scale per row "
        "so the members can be compared by eye."
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
        help="Figure to write; the suffix picks the format.",
        type=str,
        default="inverse_maps.png",
    )
    parser.add_argument(
        "--variable",
        help="Design variable to plot. The default reads it from each file.",
        choices=sorted(DESIGN_VARIABLES),
        default=None,
    )
    parser.add_argument(
        "--cmap",
        help="Colormap for the design variable.",
        type=str,
        default="viridis",
    )
    parser.add_argument(
        "--residual-cmap",
        help="Colormap for the residual.",
        type=str,
        default="magma",
    )
    parser.add_argument(
        "--linear",
        help="Linear color scale for the design variable instead of the default logarithmic one.",
        action="store_true",
    )
    parser.add_argument("--vmin", help="Lower color limit of the design variable.", type=float, default=None)
    parser.add_argument("--vmax", help="Upper color limit of the design variable.", type=float, default=None)
    parser.add_argument("--residual-vmin", help="Lower color limit of the residual.", type=float, default=None)
    parser.add_argument("--residual-vmax", help="Upper color limit of the residual.", type=float, default=None)
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

    output_file = Path(options.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(output_file.parent / "inverse_plot.log")

    members = [
        member
        for member in (
            read_member(Path(f), parameters, options.variable, mask=not options.no_mask) for f in options.INFILES
        )
        if member is not None
    ]
    if not members:
        raise SystemExit(
            f"none of the {len(options.INFILES)} given files yielded a panel; "
            f"they need a design variable, {RESIDUAL_VAR} and {', '.join(parameters)}"
        )
    sweep = short_name(parameters[0])
    members.sort(key=lambda m: tuple(m[short_name(p)] for p in parameters))
    logger.info("plotting %d of %d files", len(members), len(options.INFILES))

    plot_members(
        members,
        sweep,
        output_file,
        cmaps=(options.cmap, options.residual_cmap),
        log=not options.linear,
        limits=(options.vmin, options.vmax, options.residual_vmin, options.residual_vmax),
        percentile=options.percentile,
        panel_width=options.panel_width,
        dpi=options.dpi,
    )
    print(f"wrote {output_file} ({len(members)} members)")


if __name__ == "__main__":
    main()
