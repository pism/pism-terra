"""
KITP calibration driver.

Ranks PISM-KITP UQ ensemble members against observed surface mass-balance
fields with a spatially-aware metric: pixel-wise RMSE is replaced by a
block-bootstrap RMSE whose block size matches the field's decorrelation
length. The best-RMSE experiment and every experiment whose 5-95 % CI
overlaps the leader's are reported as the "tied" calibration set.

Alongside the ranking, every field is importance-sampled: each member gets a
Gaussian likelihood weight from its misfit against the observations (with a
relative error plus a floor standing in for the missing observational
uncertainty), for a handful of fudge factors on that error. The weights, their
effective sample size and the resampled parameter histograms show how
strongly the field constrains the parameters, which a single winner cannot.
"""

import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
from pathlib import Path

import matplotlib.pylab as plt
import pandas as pd
import pint_xarray  # pylint: disable=unused-import
import xarray as xr
import xarray_regrid.methods.conservative  # pylint: disable=unused-import
from dask.diagnostics import ProgressBar

# The metric and the importance sampling live in pism_terra.calibration so the
# glacier driver can share them; the names stay importable from here.
from pism_terra.calibration import (  # pylint: disable=unused-import
    block_bootstrap_rmse,
    bootstrap_rmse_from_blocks,
    decorrelation_length,
    importance_weights,
    observation_uncertainty,
    plot_parameter_histograms,
    posterior_table,
    rank_by_bootstrap_rmse,
    squared_error_blocks,
)
from pism_terra.processing import preprocess_netcdf as preprocess

debm_uq_vars = {
    "surface.debm_simple.c1": "c1",
    "surface.debm_simple.c2": "c2",
    "surface.debm_simple.air_temp_all_precip_as_snow": "as_snow",
    "surface.debm_simple.air_temp_all_precip_as_rain": "as_rain",
    "surface.debm_simple.refreeze": "refreeze",
}


DEFAULT_DATA_DIR = "~/base/pism-terra"
DEFAULT_FUDGE_FACTORS = (1.0, 3.0, 10.0)
DEFAULT_N_SAMPLES = 10_000

pctls = [0.05, 0.95]
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

debm_uq_vars = {
    "surface.debm_simple.c1": "c1",
    "surface.debm_simple.c2": "c2",
    "surface.debm_simple.air_temp_all_precip_as_snow": "as_snow",
    "surface.debm_simple.air_temp_all_precip_as_rain": "as_rain",
    "surface.debm_simple.refreeze": "refreeze",
}

pdd_uq_vars = {"surface.pdd.factor_ice": "fice", "surface.pdd.factor_snow": "fsnow", "surface.pdd.refreeze": "refreeze"}

m_vars = ["surface_accumulation_flux", "surface_melt_flux", "surface_runoff_flux", "climatic_mass_balance"]


def calibrate(
    data_dir,
    fudge_factors=DEFAULT_FUDGE_FACTORS,
    n_samples=DEFAULT_N_SAMPLES,
    relative_error=0.10,
    error_floor=50.0,
):
    """
    Rank KITP UQ ensemble members against observed surface mass balance.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Root directory holding the KITP calibration inputs and outputs (the
        ``2026_08_kitp_*_calib`` trees). ``~`` is expanded.
    fudge_factors : sequence of float, optional
        Multipliers on the observed error for the importance sampling.
    n_samples : int, optional
        Draws with replacement per fudge factor; see :func:`importance_weights`.
    relative_error : float, optional
        Relative observational error; see :func:`observation_uncertainty`.
    error_floor : float, optional
        Absolute error floor in kg m^-2 yr^-1; see :func:`observation_uncertainty`.
    """
    data_dir = Path(data_dir).expanduser()

    obs = xr.open_dataset(
        f"{data_dir}/2026_08_kitp_debm_calib/kitp/input/v4/spatial_GIS_HIRHAM5-ERA5_YMM_1990_2019_v4.nc",
        engine="netcdf4",
        decode_times=False,
        decode_timedelta=False,
        chunks=None,
    ).drop_dims("nv", errors="ignore")

    # Keep only what the metric needs: everything else (air_temp, albedo,
    # precipitation, ...) would otherwise be carried through the conservative
    # regridding below, which is the second-most expensive step here.
    obs = obs[["climatic_mass_balance", "surface_melt_flux", "surface_runoff_flux"]].pint.quantify()
    obs["surface_accumulation_flux"] = obs["climatic_mass_balance"] - obs["surface_melt_flux"]
    for v in m_vars:
        obs[v] = obs[v].pint.to("kg m^-2 yr^-1")
    obs = obs[m_vars].pint.dequantify()

    for (
        ebm,
        ebm_uq_vars,
    ) in zip(["debm"], [debm_uq_vars]):

        ds = (
            xr.open_mfdataset(
                f"{data_dir}/2026_08_kitp_{ebm}_calib/output/basin/"
                "spatial_GIS_g1200m_id_HIRHAM5-ERA5_YMM_1990_2019_uq_*_0001-01-01_0002-01-01.nc",
                preprocess=partial(preprocess, uq_regexp=None, exp_regexp="uq_(.+?)_"),
                engine="netcdf4",
                join="outer",
                compat="no_conflicts",
                parallel=True,
                chunks="auto",
                decode_times=False,
                decode_timedelta=False,
            )
            .drop_dims("nv", errors="ignore")
            .pint.quantify()
        )
        ds["exp_id"] = ds["exp_id"].astype("int")
        for v in m_vars:
            ds[v] = ds[v].pint.to("kg m^-2 yr^-1")
        ds = ds.pint.dequantify()

        ebm_uq_df = ds.pism_config.to_series().apply(json.loads).apply(pd.Series)[ebm_uq_vars.keys()]
        ds["time"] = obs["time"]

        # Regrid onto the simulation grid. Only the grid of the target
        # matters, so hand the regridder bare coordinates instead of the
        # ensemble itself.
        target_grid = xr.Dataset(coords={"y": ds["y"], "x": ds["x"]}).reset_coords(drop=True)
        _obs = observation_uncertainty(
            obs.regrid.conservative(target_grid).squeeze(), relative=relative_error, floor=error_floor
        )
        _ds = ds[m_vars]

        cmb_obs = (
            (_obs["climatic_mass_balance"].pint.quantify() * xr.DataArray(1200).pint.quantify("m") ** 2)
            .pint.to("Gt/yr")
            .mean(dim="time")
            .sum()
            .pint.dequantify()
            .compute()
            .values
        )

        for v in ["climatic_mass_balance", "surface_accumulation_flux", "surface_melt_flux", "surface_runoff_flux"]:

            with ProgressBar():

                # 0) Importance sampling: likelihood weights per member for a
                # few fudge factors on the observed error. The effective sample
                # size says how many members the field really distinguishes.
                weighted = importance_weights(_ds, _obs, v, fudge_factors=fudge_factors, n_samples=n_samples)
                for fudge_factor in weighted.fudge_factor.values:
                    w = weighted["weights"].sel(fudge_factor=fudge_factor)
                    top_id = w.idxmax(dim="exp_id").values
                    print(
                        f"{ebm}/{v}: fudge {fudge_factor:g}: ESS = {float(weighted['ess'].sel(fudge_factor=fudge_factor)):.1f} "
                        f"of {w.sizes['exp_id']}, top exp_id = {top_id} (weight {float(w.max()):.3f})"
                    )
                    plot_parameter_histograms(
                        ebm_uq_df,
                        ebm_uq_vars,
                        weighted["counts"].sel(fudge_factor=fudge_factor).to_pandas(),
                        f"{ebm}_{v}_ff_{fudge_factor:g}.png",
                    )
                posterior_table(weighted, ebm_uq_df).to_csv(f"{ebm}_{v}_importance.csv")

                # 1-3) Block-bootstrap RMSE per exp_id with the block side set by the
                # observed field's decorrelation length; members whose 5-95 % CI
                # overlaps the leader's are statistically tied with the best.
                # ``sim_mean_all`` stays lazy: the ensemble is reduced to per-block
                # sums as it streams off disk.
                sim_mean_all = _ds[v].mean(dim="time")
                obs_mean = _obs[v].mean(dim="time").squeeze().compute()
                ranking = rank_by_bootstrap_rmse(sim_mean_all, obs_mean, n_boot=500)
                L, block_size = ranking.attrs["decorrelation_length"], ranking.attrs["block_size"]
                print(f"{ebm}/{v}: decorrelation length ≈ {L:.0f} m, block_size = {block_size} px")
                rmse_mean, rmse_lo, rmse_hi = ranking["rmse_mean"], ranking["rmse_lo"], ranking["rmse_hi"]
                tied_mask = ranking["tied_with_best"]
                best_id = rmse_mean.idxmin(dim="exp_id").values
                tied_ids = list(rmse_mean.exp_id.where(tied_mask, drop=True).values)
                print(f"{ebm}/{v}: best exp_id = {best_id}, n tied within 5-95% CI = {len(tied_ids)}")

                # Per-experiment weight for the parameter histograms: 1 if the
                # exp_id is in the statistically-tied set, 0 otherwise. This is
                # what ``np.repeat`` consumes below so each parameter value
                # contributes to the histogram only if its experiment passed the
                # bootstrap tie test.
                ebm_counts = pd.Series(
                    tied_mask.values.astype(int),
                    index=pd.Index(rmse_mean.exp_id.values, name="exp_id"),
                )

                plot_parameter_histograms(ebm_uq_df, ebm_uq_vars, ebm_counts, f"{ebm}_{v}.png")

                # Write per-experiment stats to CSV so the user can inspect ties.
                rmse_df = (
                    pd.DataFrame(
                        {
                            "rmse_mean": rmse_mean.values,
                            "rmse_lo": rmse_lo.values,
                            "rmse_hi": rmse_hi.values,
                            "tied_with_best": tied_mask.values,
                        },
                        index=pd.Index(rmse_mean.exp_id.values, name="exp_id"),
                    )
                    .join(ebm_uq_df, how="left")
                    .sort_values("rmse_mean")
                )
                rmse_df.to_csv(f"{ebm}_{v}_rmse.csv")

                # Read the winner once; it is plotted three times below.
                sim_best = _ds[v].sel(exp_id=best_id).mean(dim="time").squeeze().compute()
                vmin = min(float(obs_mean.min()), float(sim_best.min()))
                vmax = max(float(obs_mean.max()), float(sim_best.max()))
                best_params = ebm_uq_df.loc[best_id]
                fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
                obs_mean.plot(ax=axes[0], vmin=vmin, vmax=vmax)
                axes[0].set_title("Observed")
                sim_best.plot(ax=axes[1], vmin=vmin, vmax=vmax)
                param_str = ", ".join(f"{name}={best_params[k]:.4g}" for k, name in ebm_uq_vars.items())
                rmse_best_mean = float(rmse_mean.sel(exp_id=best_id))
                rmse_best_lo = float(rmse_lo.sel(exp_id=best_id))
                rmse_best_hi = float(rmse_hi.sel(exp_id=best_id))
                axes[1].set_title(
                    f"Best (id={best_id}, RMSE={rmse_best_mean:.1f} "
                    f"[{rmse_best_lo:.1f}-{rmse_best_hi:.1f}], n_tied={len(tied_ids)})\n{param_str}"
                )
                (sim_best - obs_mean).plot(ax=axes[2], cmap="RdBu", vmin=-1000, vmax=1000)
                axes[2].set_title("Difference")
                fig.tight_layout()
                fig.savefig(f"{ebm}_{v}_best_rmse.png", dpi=300)
                plt.close()
                del fig

            cmb_sim = (
                (
                    _ds.sel(exp_id=best_id)["climatic_mass_balance"].pint.quantify()
                    * xr.DataArray(1200).pint.quantify("m") ** 2
                )
                .pint.to("Gt/yr")
                .mean(dim="time")
                .sum()
                .pint.dequantify()
                .compute()
                .values
            )
            print(f"Obs: {cmb_obs} Gt/yr, Sim: {cmb_sim} Gt/yr")


def main():
    """
    Run main script.
    """

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Calibrate KITP surface mass balance ensembles."
    parser.add_argument(
        "--data-path",
        help="Root directory holding the KITP calibration inputs and outputs.",
        type=str,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--fudge-factors",
        help="Comma-separated multipliers on the observed error for the importance sampling.",
        type=lambda s: tuple(float(x) for x in s.split(",")),
        default=DEFAULT_FUDGE_FACTORS,
    )
    parser.add_argument(
        "--n-samples",
        help="Draws with replacement per fudge factor; sets the resolution of the resampled histograms.",
        type=int,
        default=DEFAULT_N_SAMPLES,
    )
    parser.add_argument(
        "--relative-error",
        help="Relative observational error of the RCM fields.",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--error-floor",
        help="Absolute floor on the observational error, kg m^-2 yr^-1.",
        type=float,
        default=50.0,
    )

    options = parser.parse_args()

    calibrate(
        options.data_path,
        fudge_factors=options.fudge_factors,
        n_samples=options.n_samples,
        relative_error=options.relative_error,
        error_floor=options.error_floor,
    )


if __name__ == "__main__":
    main()
