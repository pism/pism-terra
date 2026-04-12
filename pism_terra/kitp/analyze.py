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


def load_dataset(filename_or_obj: Sequence[str | Path], **kwargs) -> xr.Dataset:
    """
    Load and preprocess multiple NetCDF files into a single dataset.

    Parameters
    ----------
    filename_or_obj : list of str or Path
        NetCDF files to open.
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
            join="outer",
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
    pctls = [0.16, 0.5, 0.84]
    cumulative_vars = ["ice_mass"]
    flux_vars = ["tendency_of_ice_mass", "tendency_of_ice_mass_due_to_surface_mass_flux", "grounding_line_flux"]

    baseline_file = next(f for f in infiles if "HIRHAM5" in Path(f).name)
    baseline = xr.open_dataset(baseline_file, chunks=None, decode_times=time_coder, decode_timedelta=delta_coder)
    if "basin" not in baseline.dims:
        baseline = baseline.expand_dims({"basin": ["GIS"]})
    baseline = baseline.resample({"time": "YE"}).mean().pint.quantify()

    exp_files = [Path(f) for f in infiles if "HIRHAM5" not in Path(f).name]

    logger.info("Loading experiments")
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        exps_ds = load_dataset(
            exp_files,
            exp_regexp=r"_exp_((?:futSST|pdSST|pa)-\S+?)_(?:uq_\d+_)?\d{4}-\d{2}-\d{2}",
            uq_regexp=None,
            uq_dim=None,
            exp_dim="exp_id",
        )

    obj_vars = [v for v in exps_ds.data_vars if exps_ds[v].dtype == object]
    experiments = exps_ds.drop_vars(obj_vars)
    if "basin" not in experiments.dims:
        experiments = experiments.expand_dims({"basin": ["GIS"]})
    experiments = experiments.resample({"time": "YE"}).mean().pint.quantify()

    # Compute percentiles and load into memory once
    logger.info("Computing percentiles")
    with ProgressBar():
        experiments_cumulative_pctls = (
            experiments[cumulative_vars]
            .quantile(pctls, dim="gcm_id", skipna=True)
            .rename({"quantile": "pctl"})
            .pint.to("Gt")
            .pint.dequantify()
            .compute()
        )
        experiments_fluxes_pctls = (
            experiments[flux_vars]
            .quantile(pctls, dim="gcm_id", skipna=False)
            .rename({"quantile": "pctl"})
            .pint.to("Gt/yr")
            .pint.dequantify()
            .compute()
        )
        experiments_pctls = xr.merge([experiments_cumulative_pctls, experiments_fluxes_pctls])
    baseline_cumulative_computed = baseline[cumulative_vars].pint.to("Gt").pint.dequantify().compute()
    baseline_fluxes_computed = baseline[flux_vars].pint.to("Gt/yr").pint.dequantify().compute()
    baseline_computed = xr.merge([baseline_cumulative_computed, baseline_fluxes_computed])

    res = "2400m"
    for basin_name in experiments_pctls.basin.values:
        basin = experiments_pctls.sel(basin=basin_name)
        basin_baseline = baseline_computed.sel(basin=basin_name)
        with mpl.rc_context(rc=rc_params):
            fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6.4, 4.8), height_ratios=[2, 1, 1])

            ice_mass = basin_baseline["ice_mass"]
            ice_mass = ice_mass - ice_mass.isel(time=0)
            slc = ice_mass * gt2mmsle
            slc.plot(
                ax=axs[0], color=BASELINE_OPTS["color"], ls=BASELINE_OPTS["ls"], label=BASELINE_OPTS["title"], lw=1
            )
            smb = basin_baseline["tendency_of_ice_mass_due_to_surface_mass_flux"]
            smb.plot(ax=axs[1], color=BASELINE_OPTS["color"], ls=BASELINE_OPTS["ls"], lw=1)

            for exp_name, exp in EXPS_OPTS.items():
                if exp_name not in basin.exp_id.values:
                    continue
                _ds = basin.sel(exp_id=exp_name)
                ice_mass = _ds["ice_mass"]
                ice_mass = ice_mass - ice_mass.isel(time=0)
                slc = ice_mass * gt2mmsle
                smb = _ds["tendency_of_ice_mass_due_to_surface_mass_flux"]
                glf = _ds["grounding_line_flux"]
                time_vals = slc.sel(pctl=0.5).time.values
                axs[0].fill_between(
                    time_vals,
                    slc.sel(pctl=pctls[0]),
                    slc.sel(pctl=pctls[-1]),
                    color=exp["color"],
                    lw=0,
                    alpha=0.25,
                )
                slc.sel(pctl=0.5).plot(ax=axs[0], color=exp["color"], ls=exp["ls"], label=exp["title"], lw=0.75)
                axs[1].fill_between(
                    time_vals,
                    smb.sel(pctl=pctls[0]),
                    smb.sel(pctl=pctls[-1]),
                    color=exp["color"],
                    lw=0,
                    alpha=0.25,
                )
                glf.sel(pctl=0.5).plot(ax=axs[2], color=exp["color"], ls=exp["ls"], label=exp["title"], lw=0.75)
                axs[2].fill_between(
                    time_vals,
                    glf.sel(pctl=pctls[0]),
                    glf.sel(pctl=pctls[-1]),
                    color=exp["color"],
                    lw=0,
                    alpha=0.25,
                )
                glf.sel(pctl=0.5).plot(ax=axs[2], color=exp["color"], ls=exp["ls"], label=exp["title"], lw=0.75)

            axs[0].set_ylabel("Contribution to sea-level (mm)")
            axs[1].set_ylabel("Surface mass balance (Gt/yr)")
            axs[2].set_ylabel("Grounding line flux (Gt/yr)")
            axs[0].set_xlabel(None)
            axs[0].set_title(basin_name)
            axs[1].set_title(None)
            axs[2].set_title(None)
            axs[0].axhline(y=0, ls="dotted", lw=0.5)
            axs[1].axhline(y=0, ls="dotted", lw=0.5)
            axs[0].set_ylim(-10, 100)
            axs[1].set_ylim(-100, 500)
            axs[2].set_ylim(-500, 0)
            axs[-1].set_xlim(time_vals[0], time_vals[-1])
            handles, labels = axs[0].get_legend_handles_labels()
            legend_main = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.1, 0.9), ncol=1)
            legend_main.set_title(None)
            legend_main.get_frame().set_linewidth(0.0)
            legend_main.get_frame().set_alpha(0.0)
            fig.tight_layout()
            fig.savefig(f"pism_kitp_{basin_name}_{res}.png", dpi=300)
            fig.savefig(f"pism_kitp_{basin_name}_{res}.pdf")
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
