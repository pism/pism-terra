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

import dask
import matplotlib as mpl
import matplotlib.pylab as plt
import nc_time_axis
import numpy as np
import pint_xarray
import xarray as xr
from cmap import Colormap
from cycler import cycler
from dask.diagnostics import ProgressBar
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

BASELINE_OPTS = {"short_hand": "baseline", "color": (0, 0, 0), "ls": "dashed", "title": "Baseline"}

EXPS_OPTS: dict[str, dict[str, Any]] = {
    "pdSST-futArcSICSIT_pdSST-pdSICSIT": {
        "color": (0.0660, 0.4430, 0.7450),
        "ls": "solid",
        "title": "Arctic sea ice loss (AGCM)",
    },
    "pa-futArcSIC-ext_pa-pdSIC-ext": {
        "color": (0.8660, 0.3290, 0),
        "ls": "solid",
        "title": "Arctic sea ice loss (AOGCM)",
    },
    "futSST-pdSIC_pdSST-pdSIC": {
        "color": (0.9290, 0.6940, 0.1250),
        "ls": "solid",
        "title": "Global SST warming",
    },
    "pdSST-futArcSIC_pdSST-pdSIC": {
        "color": (0.5210, 0.0860, 0.8190),
        "ls": "solid",
        "title": "Arctic sea ice loss (AGCM + 2m ice)",
    },
    "futSST-futArcSIC-SUM_pdSST-pdSIC": {
        "color": (0.2310, 0.6660, 0.1960),
        "ls": "solid",
        "title": "Global SST warming + SIC loss (AGCM + 2m ice)",
    },
}


def load_dataset(filename_or_obj: Sequence[str | Path], join: str = "outer", **kwargs) -> xr.Dataset:
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
        ds = xr.open_mfdataset(
            filename_or_obj,
            preprocess=partial(preprocess, **kwargs),
            engine="netcdf4",
            parallel=True,
            chunks=None,
            data_vars="minimal",
            coords="minimal",
            join=join,
            decode_times=time_coder,
            decode_timedelta=delta_coder,
        )
    return ds


def plot_scalar_timeseries(infiles: list[str | Path]):
    """
    Plot ice-mass change timeseries from scalar output files.

    Parameters
    ----------
    infiles : list of str or Path
        Scalar NetCDF files (must include one baseline HIRHAM5 file).
    """

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
    if "basin" not in baseline.dims:
        baseline = baseline.expand_dims({"basin": ["GIS"]})
    baseline = baseline.resample({"time": "YE"}).mean()
    baseline = baseline.assign_coords(time=[t.replace(year=t.year - 1) for t in baseline.time.values])
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
        Drop object-dtype vars, ensure ``basin`` dim, resample to yearly means, and quantify.

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
        if "basin" not in ds.dims:
            ds = ds.expand_dims({"basin": ["GIS"]})
        ds = ds.resample({"time": "YE"}).mean()
        # Shift time coordinate by -1 year so the series starts at year 0
        ds = ds.assign_coords(time=[t.replace(year=t.year - 1) for t in ds.time.values])
        return ds.pint.quantify()

    def _compute_pctls(ds: xr.Dataset) -> xr.Dataset:
        """
        Compute percentiles across ``gcm_id`` for cumulative and flux variables.

        Parameters
        ----------
        ds : xr.Dataset
            Pint-quantified dataset with a ``gcm_id`` dimension.

        Returns
        -------
        xr.Dataset
            Dataset with a ``pctl`` dimension, cumulative variables in Gt and fluxes in Gt/yr.
        """
        with ProgressBar():
            cumulative = (
                ds[cumulative_vars]
                .pint.to("Gt")
                .pint.dequantify()
                .quantile(pctls, dim="gcm_id")
                .rename({"quantile": "pctl"})
                .compute()
            )
            fluxes = (
                ds[flux_vars]
                .pint.to("Gt/yr")
                .pint.dequantify()
                .quantile(pctls, dim="gcm_id")
                .rename({"quantile": "pctl"})
                .compute()
            )
        return xr.merge([cumulative, fluxes])

    logger.info("Loading multi-GCM experiments")
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        multi_gcm_ds = _prepare(load_dataset(multi_model_files, **load_kwargs))
    logger.info("Computing multi-GCM percentiles")
    multi_gcm_pctls = _compute_pctls(multi_gcm_ds)

    logger.info("Loading single-GCM experiments")
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        single_gcm_ds = _prepare(load_dataset(single_model_files, **load_kwargs))
    logger.info("Computing single-GCM percentiles")
    single_gcm_pctls = _compute_pctls(single_gcm_ds)

    experiments = xr.concat(
        [multi_gcm_ds, single_gcm_ds],
        dim="exp_id",
    )
    single_gcm = experiments.sel({"gcm_id": single_model_gcms})
    experiments_pctls = xr.concat(
        [multi_gcm_pctls, single_gcm_pctls],
        dim="exp_id",
    )
    baseline_cumulative_computed = baseline[cumulative_vars].pint.to("Gt").pint.dequantify().compute()
    baseline_fluxes_computed = baseline[flux_vars].pint.to("Gt/yr").pint.dequantify().compute()
    baseline_computed = xr.merge([baseline_cumulative_computed, baseline_fluxes_computed])

    res = "2400m"
    #    for basin_name in experiments_pctls.basin.values:
    with mpl.rc_context(rc=rc_params):
        for basin_name in baseline.basin.values:
            basin_single = single_gcm_pctls.sel(basin=basin_name)
            basin_multi = multi_gcm_pctls.sel(basin=basin_name)
            basin_baseline = baseline_computed.sel(basin=basin_name)
            basin_single_gcm = single_gcm.sel(basin=basin_name)

            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6.4, 3.6), height_ratios=[1.618, 1])

            l = []
            ci = []

            ice_mass = basin_baseline["ice_mass"]
            ice_mass = ice_mass - ice_mass.isel(time=0)
            slc = ice_mass * gt2mmsle
            _l = slc.plot(
                ax=axs[0], color=BASELINE_OPTS["color"], ls=BASELINE_OPTS["ls"], label=BASELINE_OPTS["title"], lw=1
            )
            l.append(_l)

            smb = basin_baseline["tendency_of_ice_mass_due_to_surface_mass_flux"]
            smb.plot(ax=axs[1], color=BASELINE_OPTS["color"], ls=BASELINE_OPTS["ls"], lw=1)

            for exp_name, exp in EXPS_OPTS.items():
                in_multi = exp_name in basin_multi.exp_id.values
                in_single = exp_name in basin_single.exp_id.values
                if not in_multi and not in_single:
                    continue

                if in_multi:
                    multi_ds = basin_multi.sel(exp_id=exp_name)
                    multi_ice_mass = multi_ds["ice_mass"]
                    multi_ice_mass = multi_ice_mass - multi_ice_mass.isel(time=0)
                    multi_slc = multi_ice_mass * gt2mmsle
                    multi_smb = multi_ds["tendency_of_ice_mass_due_to_surface_mass_flux"]
                    multi_glf = multi_ds["tendency_of_ice_mass_due_to_discharge"]
                    multi_time_vals = multi_slc.sel(pctl=0.5).time.values

                if in_single:
                    single_ds = basin_single.sel(exp_id=exp_name)
                    single_ice_mass = single_ds["ice_mass"]
                    single_ice_mass = single_ice_mass - single_ice_mass.isel(time=0)
                    single_slc = single_ice_mass * gt2mmsle
                    single_smb = single_ds["tendency_of_ice_mass_due_to_surface_mass_flux"]
                    single_glf = single_ds["tendency_of_ice_mass_due_to_discharge"]
                    single_time_vals = single_slc.sel(pctl=0.5).time.values

                if in_multi:
                    _ci = axs[0].fill_between(
                        multi_time_vals,
                        multi_slc.sel(pctl=pctls[0]),
                        multi_slc.sel(pctl=pctls[-1]),
                        color=exp["color"],
                        lw=0,
                        alpha=0.25,
                    )
                    ci.append(_ci)

                    _l = multi_slc.sel(pctl=0.5).plot(
                        ax=axs[0], color=exp["color"], ls=exp["ls"], label=exp["title"], lw=0.75
                    )
                    l.append(_l)
                if in_single:
                    _l = single_slc.sel(pctl=0.5).plot(
                        ax=axs[0],
                        color=exp["color"],
                        ls=exp["ls"],
                        label=exp["title"] if not in_multi else None,
                        lw=0.75,
                    )
                    l.append(_l)
                    single_smb.sel(pctl=0.5).plot(ax=axs[1], color=exp["color"], ls=exp["ls"], lw=0.75)
                if in_multi:
                    axs[1].fill_between(
                        multi_time_vals,
                        multi_smb.sel(pctl=pctls[0]),
                        multi_smb.sel(pctl=pctls[-1]),
                        color=exp["color"],
                        lw=0,
                        alpha=0.25,
                    )
                    multi_smb.sel(pctl=0.5).plot(ax=axs[1], color=exp["color"], ls=exp["ls"], lw=0.75)

            axs[0].set_ylabel("Contribution to sea-level (mm)")
            axs[1].set_ylabel("Surface mass balance (Gt/yr)")
            axs[0].set_xlabel(None)
            axs[0].set_title(basin_name)
            axs[1].set_title(None)
            axs[1].axhline(y=0, color="k", ls="dotted", lw=0.5)
            axs[-1].set_xlim(multi_time_vals[0], multi_time_vals[-1])
            # cftime axis: pick every 50th year from the actual time values
            year_ticks = [t for t in multi_time_vals if t.year % 50 == 0]
            axs[-1].set_xticks(year_ticks)
            axs[-1].set_xticklabels([f"{int(t.year)}" for t in year_ticks])
            # Legend 1: multi-GCM confidence intervals

            # leg_ci = fig.legend(
            #     handles=ci,
            #     loc="upper left",
            #     bbox_to_anchor=(0.08, 0.93),
            #     ncol=1,
            #     title="5-95% c.i.",
            # )
            # leg_ci.get_frame().set_linewidth(0.0)
            # leg_ci.get_frame().set_alpha(0.0)
            # fig.add_artist(leg_ci)

            leg_line = fig.legend(
                handles=[h for item in l for h in (item if isinstance(item, list) else [item])],
                loc="upper left",
                bbox_to_anchor=(0.08, 0.93),
                ncol=1,
            )
            leg_line.get_frame().set_linewidth(0.0)
            leg_line.get_frame().set_alpha(0.0)

            fig.tight_layout()
            fig.savefig(f"pism_kitp_multi_gcm_{basin_name}_{res}.png", dpi=300)
            fig.savefig(f"pism_kitp_multi_gcm_{basin_name}_{res}.pdf")
            plt.close(fig)

            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6.4, 3.6), height_ratios=[1.618, 1])

            ice_mass_bl = basin_baseline["ice_mass"]
            ice_mass_bl = ice_mass_bl - ice_mass_bl.isel(time=0)
            slc_bl = ice_mass_bl * gt2mmsle
            slc_bl.plot(
                ax=axs[0], color=BASELINE_OPTS["color"], ls=BASELINE_OPTS["ls"], label=BASELINE_OPTS["title"], lw=1
            )

            smb_bl = basin_baseline["tendency_of_ice_mass_due_to_surface_mass_flux"]
            smb_bl.plot(ax=axs[1], color=BASELINE_OPTS["color"], ls=BASELINE_OPTS["ls"], lw=1)

            ice_mass_gcm = basin_single_gcm["ice_mass"].pint.to("Gt").pint.dequantify()
            ice_mass_gcm = ice_mass_gcm - ice_mass_gcm.isel(time=0)
            slc_gcm = ice_mass_gcm * gt2mmsle.pint.dequantify()
            smb_gcm = (
                basin_single_gcm["tendency_of_ice_mass_due_to_surface_mass_flux"].pint.to("Gt/yr").pint.dequantify()
            )

            for exp_name, exp in EXPS_OPTS.items():

                slc_gcm.sel({"exp_id": exp_name}).plot(
                    ax=axs[0], color=exp["color"], ls=exp["ls"], label=exp["title"], lw=0.75
                )
                smb_gcm.sel({"exp_id": exp_name}).plot(
                    ax=axs[1], color=exp["color"], ls=exp["ls"], add_legend=False, lw=0.75
                )

            handles, labels = axs[0].get_legend_handles_labels()
            legend_main = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.1, 0.9), ncol=1)
            legend_main.set_title(None)
            legend_main.get_frame().set_linewidth(0.0)
            legend_main.get_frame().set_alpha(0.0)
            axs[0].set_ylabel("Contribution to sea-level (mm)")
            axs[1].set_ylabel("Surface mass balance (Gt/yr)")
            axs[0].set_xlabel(None)
            axs[0].set_title(basin_name)
            axs[1].set_title(None)
            axs[1].axhline(y=0, color="k", ls="dotted", lw=0.5)
            axs[-1].set_xlim(multi_time_vals[0], multi_time_vals[-1])
            # cftime axis: pick every 50th year from the actual time values
            year_ticks = [t for t in multi_time_vals if t.year % 50 == 0]
            axs[-1].set_xticks(year_ticks)
            axs[-1].set_xticklabels([f"{int(t.year)}" for t in year_ticks])
            # Legend 1: multi-GCM confidence intervals
            fig.tight_layout()
            fig.savefig(f"pism_kitp_cesm1_gcm_{basin_name}_{res}.png", dpi=300)
            fig.savefig(f"pism_kitp_cesm1_gcm_{basin_name}_{res}.pdf")
            plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))

        for exp_name, exp in EXPS_OPTS.items():

            ice_mass = baseline["ice_mass"]
            ice_mass = ice_mass - ice_mass.isel(time=0)
            slc = ice_mass * gt2mmsle
            slc.plot(
                ax=ax,
                hue="basin",
                color=BASELINE_OPTS["color"],
                ls=BASELINE_OPTS["ls"],
                label=BASELINE_OPTS["title"],
                lw=1,
            )
            fig.savefig(f"pism_kitp_{res}.png", dpi=300)
            fig.savefig(f"pism_kitp_{res}.pdf")
            plt.close(fig)


def main():
    """Run main script."""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Postprocess KITP Greenland."
    parser.add_argument("--ntasks", help="Sets number of tasks.", type=int, default=4)
    parser.add_argument("FILES", help="netCDF files to process.", nargs="*")

    options, _ = parser.parse_known_args()
    infiles = options.FILES

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

    plot_scalar_timeseries(infiles)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
