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

# pylint: disable=unused-import,unused-variable

"""
Postprocessing.
"""

import logging
import re
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import cftime
import dask
import matplotlib as mpl
import matplotlib.pylab as plt
import nc_time_axis
import numpy as np
import pint_xarray
import seaborn as sns
import xarray as xr
from cmap import Colormap
from cycler import cycler
from dask.diagnostics import ProgressBar
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pism_terra.kitp.adjust_kitp_timeseries import SPINUP_END_YEAR, trim_spinup
from pism_terra.processing import preprocess_netcdf as preprocess

cm = Colormap("tol:bright").to_matplotlib()
cm_cycler = cycler(color=cm.colors)
cm_precip = Colormap("crameri:navia").to_matplotlib()
cm_rdbu = Colormap("crameri:vik_r").to_matplotlib()

xr.set_options(keep_attrs=True)
warnings.filterwarnings("ignore", message="Increasing number of chunks")
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

logger = logging.getLogger("pism_terra.kitp.analyze")


fontsize = 6
rc_params = {
    "axes.linewidth": 0.15,
    "xtick.major.size": 2.0,
    "xtick.major.width": 0.15,
    "ytick.major.size": 2.0,
    "ytick.major.width": 0.15,
    "hatch.linewidth": 0.15,
    "font.size": fontsize,
    "font.family": "DejaVu Sans",
}

single_model_gcms = ["CESM1-WACCM-SC"]
multi_model_gcms = [
    "CESM1-WACCM-SC",
    "AWI-CM-1-1-MR",
    "CESM2",
    "CNRM-CM6-1",
    "CanESM5",
    "HadGEM3-GC31-MM",
    "IPSL-CM6A-LR",
]

BASELINE_OPTS = {"short_hand": "baseline", "color": (0, 0, 0), "ls": "dashed", "lw": 1.5, "title": "Baseline"}

EXPS_OPTS: dict[str, dict[str, Any]] = {
    "pdSST-futArcSICSIT_pdSST-pdSICSIT": {
        "color": (0.0660, 0.4430, 0.7450),
        "ls": "solid",
        "lw": 1.25,
        "title": "Arctic sea ice loss (AGCM)",
    },
    "pa-futArcSIC-ext_pa-pdSIC-ext": {
        "color": (0.8660, 0.3290, 0),
        "ls": "solid",
        "lw": 1.25,
        "title": "Arctic sea ice loss (AOGCM)",
    },
    "futSST-pdSIC_pdSST-pdSIC": {
        "color": (0.9290, 0.6940, 0.1250),
        "ls": "solid",
        "lw": 0.75,
        "title": "Global SST warming",
    },
    "pdSST-futArcSIC_pdSST-pdSIC": {
        "color": (0.5210, 0.0860, 0.8190),
        "ls": "solid",
        "lw": 0.75,
        "title": "Arctic sea ice loss (AGCM + 2m ice)",
    },
    "futSST-futArcSIC-SUM_pdSST-pdSIC": {
        "color": (0.2310, 0.6660, 0.1960),
        "ls": "solid",
        "lw": 0.75,
        "title": "Global SST warming + SIC loss (AGCM + 2m ice)",
    },
}


REGION_DIM = "glacier_id"
# Region dimensions this module has produced over time, newest first. Files
# written before the post-processing modules were unified use ``basin`` (or
# ``RGIid`` for glacier runs) with string labels; newer ones use an integer
# ``glacier_id`` index whose labels live in ``glacier_id_name``.
REGION_DIMS = (REGION_DIM, "basin", "RGIid")


def with_region_labels(ds: xr.Dataset, default: str = "GIS") -> xr.Dataset:
    """
    Put string region labels back on the region dimension.

    Post-processing writes the region dimension as a positional integer index
    so that CDO can read the file, keeping the labels in a companion
    ``<dim>_name`` coordinate. Selecting by name means restoring that
    coordinate as the index. Older files that already carry string labels, and
    single-region files with no region dimension at all, are brought to the
    same shape.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset as read from a scalar post-processing file.
    default : str, default "GIS"
        Label given to a dataset that has no region dimension.

    Returns
    -------
    xarray.Dataset
        Dataset whose :data:`REGION_DIM` dimension is indexed by region name.
    """
    dim = next((d for d in REGION_DIMS if d in ds.dims), None)
    if dim is None:
        return ds.expand_dims({REGION_DIM: [default]})

    name_var = f"{dim}_name"
    if name_var in ds.coords:
        # Swap the positional index for the labels; ``set_index`` consumes the
        # name coordinate, leaving one region coordinate rather than two.
        ds = ds.set_index({dim: name_var})
    if dim != REGION_DIM:
        ds = ds.rename({dim: REGION_DIM})
    return ds


def load_dataset(filename_or_obj: Sequence[str | Path], join: str | None = "outer", **kwargs) -> xr.Dataset:
    """
    Load and preprocess multiple NetCDF files into a single dataset.

    Parameters
    ----------
    filename_or_obj : list of str or Path
        NetCDF files to open.
    join : str, optional
        How to combine datasets along any shared dimensions, by default "outer".
    **kwargs
        Forwarded to :func:`preprocess_netcdf`.

    Returns
    -------
    xr.Dataset
        The merged dataset.
    """
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    delta_coder = xr.coders.CFTimedeltaCoder()

    kwargs.setdefault("process_config", False)
    with ProgressBar():
        _dss: list[xr.Dataset] = []
        for f in filename_or_obj:
            _ds = xr.open_dataset(
                f,
                engine="netcdf4",
                decode_times=time_coder,
                decode_timedelta=delta_coder,
            )
            if _ds.time.size > 0:
                _ds = preprocess(_ds, **kwargs)
                _dss.append(_ds)
        ds = xr.concat(_dss, dim="time", join=join).sortby("time")
        ds = ds.sel({"time": slice(cftime.DatetimeNoLeap(11, 1, 1), cftime.DatetimeNoLeap(311, 1, 1))})
    return ds


def plot_scalar_timeseries(infiles: list[str | Path], output_dir: str | Path):
    """
    Plot ice-mass change timeseries from scalar output files.

    Parameters
    ----------
    infiles : list of str or Path
        Scalar NetCDF files (must include one baseline HIRHAM5 file).
    output_dir : str or Path
        Directory where to save figures.
    """

    output_dir = Path(output_dir)

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    delta_coder = xr.coders.CFTimedeltaCoder()

    gt2mmsle = xr.DataArray(-1 / 361.8).pint.quantify("mm/Gt")
    pctls = [0.05, 0.5, 0.95]
    cumulative_vars = ["ice_mass"]
    flux_vars = [
        "tendency_of_ice_mass",
        "tendency_of_ice_mass_due_to_surface_mass_flux",
        "tendency_of_ice_mass_due_to_discharge",
    ]

    baseline_file = next(f for f in infiles if "HIRHAM5" in Path(f).name)
    baseline = xr.open_dataset(baseline_file, chunks=None, decode_times=time_coder, decode_timedelta=delta_coder)
    pism_config = baseline.pism_config
    dx = float(pism_config.attrs["grid.dx"])
    res = f"{int(dx)}m"
    # Per-cell area of *this* run, used to turn summed per-area fluxes into
    # mass fluxes. It was hard-coded to 900 m, which silently rescaled every
    # run at another resolution by (dx / 900)**2.
    cell_area = xr.DataArray(dx).pint.quantify("m") ** 2
    baseline = with_region_labels(baseline)
    # Discard the settling years and renumber from year 1, so every run on the
    # figure starts from a comparable state at the same place on the axis.
    baseline = trim_spinup(baseline, window_end_year=None)
    baseline = baseline.pint.quantify()

    exp_files = [Path(f) for f in infiles if "HIRHAM5" not in Path(f).name]

    # Group files by experiment, then classify experiments as single- or multi-GCM
    exp_regexp = re.compile(r"_exp_((?:futSST|pdSST|pa)-\S+?)_(?:uq_\d+_)?\d{4}-\d{2}-\d{2}")
    gcm_regexp = re.compile(r"_gcm_(.+?)_exp_")
    files_by_exp: dict[str, list[Path]] = {}
    for f in exp_files:
        m = exp_regexp.search(f.name)
        if m:
            files_by_exp.setdefault(m.group(1), []).append(f)

    single_model_files = []
    multi_model_files = []
    for exp_name, files in files_by_exp.items():
        gcms = {m.group(1) for f in files if (m := gcm_regexp.search(f.name)) is not None}
        if gcms - set(single_model_gcms):
            multi_model_files.extend(files)
        else:
            single_model_files.extend(files)
        logger.info("  %s: %d GCMs -> %s", exp_name, len(gcms), "multi" if gcms - set(single_model_gcms) else "single")

    load_kwargs: dict[str, Any] = {
        "exp_regexp": r"_exp_((?:futSST|pdSST|pa)-\S+?)_(?:uq_\d+_)?\d{4}-\d{2}-\d{2}",
        "uq_regexp": None,
        "uq_dim": None,
        "exp_dim": "exp_id",
    }

    def _prepare(ds: xr.Dataset) -> xr.Dataset:
        """
        Drop object-dtype vars, restore region labels, trim spin-up, and quantify.

        Parameters
        ----------
        ds : xr.Dataset
            Input dataset.

        Returns
        -------
        xr.Dataset
            Prepared dataset with yearly-mean values and pint-quantified variables.
        """
        obj_vars = [v for v in ds.data_vars if ds[v].dtype == object]
        ds = ds.drop_vars(obj_vars)
        ds = with_region_labels(ds)
        # Runs differ in length, so keep everything after the spin-up.
        ds = trim_spinup(ds, window_end_year=None)
        return ds.pint.quantify()

    model_files = single_model_files + multi_model_files
    logger.info("Loading GCM experiments")
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        _dss: list[xr.Dataset] = []
        gcm = _prepare(load_dataset(model_files, **load_kwargs))

    baseline_cumulative_computed = baseline[cumulative_vars].pint.to("Gt").pint.dequantify().compute()
    baseline_fluxes_computed = baseline[flux_vars].pint.to("Gt/yr").pint.dequantify().compute()
    baseline_computed = xr.merge([baseline_cumulative_computed, baseline_fluxes_computed]).pint.quantify()

    gcm_cumulative_computed = gcm[cumulative_vars].pint.to("Gt").pint.dequantify().compute()
    gcm_fluxes_computed = gcm[flux_vars].pint.to("Gt/yr").pint.dequantify().compute()
    gcm_computed = xr.merge([gcm_cumulative_computed, gcm_fluxes_computed]).pint.quantify()

    gcm_sub_baseline = gcm_computed - baseline_computed

    with mpl.rc_context(rc=rc_params):
        for region_name in baseline[REGION_DIM].values:
            basin_gcm = gcm_computed.sel({REGION_DIM: region_name})
            basin_slc = ((basin_gcm["ice_mass"] - basin_gcm["ice_mass"].isel({"time": 0})) * gt2mmsle).pint.dequantify()
            time_vals = basin_slc.time.values
            basin_baseline = baseline_computed.sel({REGION_DIM: region_name})
            basin_baseline_slc = (
                (basin_baseline["ice_mass"] - basin_baseline["ice_mass"].isel({"time": 0})) * gt2mmsle
            ).pint.dequantify()

            fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.2))

            l = []

            for exp_name, exp in EXPS_OPTS.items():
                _gcm_slc = basin_slc.sel({"exp_id": exp_name})
                _ = _gcm_slc.plot(ax=ax, hue="gcm_id", color=exp["color"], ls=exp["ls"], lw=exp["lw"], add_legend=False)
                _l = _gcm_slc.isel({"gcm_id": 0}).plot(
                    ax=ax, color=exp["color"], ls=exp["ls"], label=exp["title"], lw=exp["lw"], add_legend=False
                )
                l.append(_l)

            _l = basin_baseline_slc.plot(
                ax=ax,
                color=BASELINE_OPTS["color"],
                ls=BASELINE_OPTS["ls"],
                label=BASELINE_OPTS["title"],
                lw=BASELINE_OPTS["lw"],
            )
            l.append(_l)

            ax.axhline(y=0, color="k", ls="dotted", lw=0.5)
            ax.set_ylabel("Contribution to sea-level (mm)")
            ax.set_xlabel("Time")
            ax.set_title(region_name)
            ax.set_xlim(time_vals[0], time_vals[-1])
            year_ticks = [t for t in time_vals if t.year % 50 == 0]
            ax.set_xticks(year_ticks)
            ax.set_xticklabels([f"{int(t.year)}" for t in year_ticks])

            leg_line = fig.legend(
                handles=[h for item in l for h in (item if isinstance(item, list) else [item])],
                loc="upper left",
                bbox_to_anchor=(0.09, 0.93),
                ncol=1,
            )
            leg_line.get_frame().set_linewidth(0.0)
            leg_line.get_frame().set_alpha(0.0)

            fig.tight_layout()
            fig.savefig(output_dir / f"pism_kitp_gcm_{region_name}_{res}.png", dpi=300)
            fig.savefig(output_dir / f"pism_kitp_gcm_{region_name}_{res}.pdf")
            fig.savefig(output_dir / f"pism_kitp_gcm_{region_name}_{res}.svg")
            plt.close(fig)

    with mpl.rc_context(rc=rc_params):
        for region_name in baseline[REGION_DIM].values:
            basin_gcm = gcm_sub_baseline.sel({REGION_DIM: region_name})
            # ``grounding_line_flux_nonneg`` is derived by the scalar
            # post-processing, which masks the *spatial* field before summing.
            # Raw PISM scalar files only carry the total, from which it cannot
            # be recovered, so the curve is dropped rather than approximated.
            region_gcm = gcm.sel({REGION_DIM: region_name})
            basin_glf = None
            if "grounding_line_flux_nonneg" in region_gcm:
                basin_glf = (region_gcm["grounding_line_flux_nonneg"] * cell_area).pint.to("Gt/yr")
            else:
                logger.info("No grounding_line_flux_nonneg in the inputs; omitting it from the flux panel")
            basin_smb = (region_gcm["tendency_of_ice_mass_due_to_surface_mass_flux"]).pint.to("Gt/yr")
            basin_slc = (basin_gcm["ice_mass"] * gt2mmsle).pint.dequantify()
            time_vals = basin_slc.time.values

            exps_palette = {k: v["color"] for k, v in EXPS_OPTS.items()}
            gcm_palette = [v["color"] for v in EXPS_OPTS.values()]

            fig, axs = plt.subplots(2, 1, figsize=(4.8, 3.2))

            l = []

            for exp_name, exp in EXPS_OPTS.items():
                _gcm_slc = basin_slc.sel({"exp_id": exp_name})
                _ = _gcm_slc.plot(
                    ax=axs[0], hue="gcm_id", color=exp["color"], ls=exp["ls"], lw=exp["lw"], add_legend=False
                )
                _gcm_smb = basin_smb.sel({"exp_id": exp_name})
                _ = _gcm_smb.plot(
                    ax=axs[1], hue="gcm_id", color=exp["color"], ls=exp["ls"], lw=exp["lw"], add_legend=False
                )
                if basin_glf is not None:
                    _gcm_glf = basin_glf.sel({"exp_id": exp_name})
                    _ = _gcm_glf.plot(
                        ax=axs[1], hue="gcm_id", color=exp["color"], ls=exp["ls"], lw=exp["lw"], add_legend=False
                    )
                _l = _gcm_slc.isel({"gcm_id": 0}).plot(
                    ax=axs[0], color=exp["color"], ls=exp["ls"], label=exp["title"], lw=exp["lw"], add_legend=False
                )
                l.append(_l)

            axs[0].axhline(y=0, color="k", ls="dotted", lw=0.5)
            axs[0].set_ylabel("Contribution to sea-level (mm)")
            axs[1].set_xlabel("Time")
            axs[0].set_title(region_name)
            axs[0].set_xlim(time_vals[0], time_vals[-1])
            year_ticks = [t for t in time_vals if t.year % 50 == 0]
            axs[1].set_xticks(year_ticks)
            axs[1].set_xticklabels([f"{int(t.year)}" for t in year_ticks])

            leg_line = fig.legend(
                handles=[h for item in l for h in (item if isinstance(item, list) else [item])],
                loc="upper left",
                bbox_to_anchor=(0.08, 0.93),
                ncol=1,
            )
            leg_line.get_frame().set_linewidth(0.0)
            leg_line.get_frame().set_alpha(0.0)

            fig.tight_layout()
            fig.savefig(output_dir / f"pism_kitp_norm_gcm_with_flux_{region_name}_{res}.png", dpi=300)
            fig.savefig(output_dir / f"pism_kitp_norm_gcm_with_flux_{region_name}_{res}.pdf")
            fig.savefig(output_dir / f"pism_kitp_norm_gcm_with_flux_{region_name}_{res}.svg")
            plt.close(fig)

    with mpl.rc_context(rc=rc_params):
        for region_name in baseline[REGION_DIM].values:
            basin_gcm = gcm_sub_baseline.sel({REGION_DIM: region_name})
            basin_slc = (basin_gcm["ice_mass"] * gt2mmsle).pint.dequantify()
            time_vals = basin_slc.time.values

            _slices = []
            for y in [100, 200, 300]:
                start = y - 10
                end = y
                _slice = (
                    basin_slc.sel({"time": slice(cftime.DatetimeNoLeap(start, 1, 1), cftime.DatetimeNoLeap(end, 1, 1))})
                    .mean(dim="time")
                    .squeeze()
                    .expand_dims(time=[f"{start}-{end}"])
                )
                _slice.name = "SLC"
                _slices.append(_slice)
            basin_slice = xr.concat(_slices, dim="time")
            _slice_df = basin_slice.to_dataframe()

            exps_palette = {k: v["color"] for k, v in EXPS_OPTS.items()}
            gcm_palette = [v["color"] for v in EXPS_OPTS.values()]

            g = sns.catplot(
                data=_slice_df,
                kind="swarm",
                x="time",
                y="SLC",
                hue="exp_id",
                palette=exps_palette,
                alpha=1,
                height=1.5,
                aspect=1.5,
                legend=False,
                linewidth=0.25,
                size=2.5,
            )
            g.despine(left=True)
            g.set_axis_labels("Year", "$\\Delta_{\\mathrm{sea-level}}$ (mm)")
            g.ax.axhline(y=0, color="k", ls="dotted", lw=0.5)
            g.fig.suptitle(region_name)
            g.fig.savefig(output_dir / f"pism_kitp_slice_xy_gcm_{region_name}_{res}.png", dpi=300)
            g.fig.savefig(output_dir / f"pism_kitp_slice_xy_gcm_{region_name}_{res}.pdf")
            g.fig.savefig(output_dir / f"pism_kitp_slice_xy_gcm_{region_name}_{res}.svg")

            fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.6))

            l = []

            for exp_name, exp in EXPS_OPTS.items():
                _gcm_slc = basin_slc.sel({"exp_id": exp_name})
                _ = _gcm_slc.plot(ax=ax, hue="gcm_id", color=exp["color"], ls=exp["ls"], lw=exp["lw"], add_legend=False)
                _l = _gcm_slc.isel({"gcm_id": 0}).plot(
                    ax=ax, color=exp["color"], ls=exp["ls"], label=exp["title"], lw=exp["lw"], add_legend=False
                )
                l.append(_l)

            ax.axhline(y=0, color="k", ls="dotted", lw=0.5)
            ax.set_ylabel("Contribution to sea-level (mm)")
            ax.set_xlabel("Time")
            ax.set_title(region_name)
            ax.set_xlim(time_vals[0], time_vals[-1])
            year_ticks = [t for t in time_vals if t.year % 50 == 0]
            ax.set_xticks(year_ticks)
            ax.set_xticklabels([f"{int(t.year)}" for t in year_ticks])

            leg_line = fig.legend(
                handles=[h for item in l for h in (item if isinstance(item, list) else [item])],
                loc="upper left",
                bbox_to_anchor=(0.08, 0.93),
                ncol=1,
            )
            leg_line.get_frame().set_linewidth(0.0)
            leg_line.get_frame().set_alpha(0.0)

            fig.tight_layout()
            fig.savefig(output_dir / f"pism_kitp_norm_gcm_{region_name}_{res}.png", dpi=300)
            fig.savefig(output_dir / f"pism_kitp_norm_gcm_{region_name}_{res}.pdf")
            fig.savefig(output_dir / f"pism_kitp_norm_gcm_{region_name}_{res}.svg")
            plt.close(fig)


def main():
    """Run main script."""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Postprocess KITP Greenland."
    parser.add_argument("--output-path", help="Directory to save figures", default=".")
    parser.add_argument("FILES", help="netCDF files to process.", nargs="*")

    options, _ = parser.parse_known_args()
    output_dir = Path(options.output_path)
    infiles = options.FILES

    output_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.WARNING, format=log_format)

    pism_logger = logging.getLogger("pism_terra")
    pism_logger.setLevel(logging.INFO)
    pism_logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    pism_logger.addHandler(console_handler)

    file_handler = logging.FileHandler("analyze.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    pism_logger.addHandler(file_handler)

    plot_scalar_timeseries(infiles, output_dir)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
