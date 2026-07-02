"""
KITP calibration driver.

Ranks PISM-KITP UQ ensemble members against observed surface mass-balance
fields with a spatially-aware metric: pixel-wise RMSE is replaced by a
block-bootstrap RMSE whose block size matches the field's decorrelation
length. The best-RMSE experiment and every experiment whose 5-95 % CI
overlaps the leader's are reported as the "tied" calibration set.
"""

import json
from functools import partial

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pint_xarray  # pylint: disable=unused-import
import xarray as xr
import xarray_regrid.methods.conservative  # pylint: disable=unused-import
from dask.diagnostics import ProgressBar

from pism_terra.filtering import importance_sampling
from pism_terra.processing import preprocess_netcdf as preprocess

debm_uq_vars = {
    "surface.debm_simple.c1": "c1",
    "surface.debm_simple.c2": "c2",
    "surface.debm_simple.air_temp_all_precip_as_snow": "as_snow",
    "surface.debm_simple.air_temp_all_precip_as_rain": "as_rain",
    "surface.debm_simple.refreeze": "refreeze",
}


def decorrelation_length(field_2d, pixel_size, threshold=1.0 / np.e):
    """
    Radially-averaged spatial-ACF decorrelation length for a 2D field.

    Pixel-wise RMSE treats every cell as independent, but glaciological
    fields are smooth on scales of many cells. The lag at which the
    radially-averaged autocorrelation first falls below ``threshold`` is a
    practical block side for bootstrap resampling: blocks of that side are
    statistically (approximately) independent.

    Parameters
    ----------
    field_2d : numpy.ndarray
        The two-dimensional field to analyse. Non-finite values are filled
        with the field's mean before the FFT; if every entry is non-finite
        the function returns ``nan``.
    pixel_size : float
        Side length of one cell in physical units (typically metres). The
        returned decorrelation length is in the same units.
    threshold : float, default ``1 / e``
        ACF level at which the decorrelation length is read off. Common
        alternatives are ``0.1`` (longer block) or ``0.5`` (shorter block).

    Returns
    -------
    float
        Decorrelation length in the units of ``pixel_size``. Returns
        ``nan`` when the input has no finite values.
    """
    a = np.asarray(field_2d, dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        return float("nan")
    a = np.where(finite, a, np.nanmean(a))
    a = a - a.mean()
    fft = np.fft.fft2(a)
    acf = np.fft.fftshift(np.fft.ifft2(fft * np.conj(fft)).real)
    acf = acf / acf.max()
    ny, nx = a.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.indices(a.shape)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    counts = np.maximum(np.bincount(r.ravel()), 1)
    radial = np.bincount(r.ravel(), weights=acf.ravel()) / counts
    rmax = min(cy, cx)
    radial = radial[: rmax + 1]
    below = np.where(radial < threshold)[0]
    lag_pixels = below[0] if below.size else rmax
    return float(lag_pixels) * float(pixel_size)


def block_bootstrap_rmse(sim, obs, block_size, n_boot=500, seed=0):
    """
    Block-bootstrap spatial RMSE per experiment.

    The domain is tiled into non-overlapping square blocks of side
    ``block_size`` pixels. For each bootstrap iteration, blocks are drawn
    with replacement and a single global RMSE is computed across the
    resampled blocks for every experiment in ``sim``. Choosing
    ``block_size`` ≳ ``decorrelation_length(obs) / pixel_size`` makes the
    resampled blocks (approximately) independent, so the spread of
    bootstrap RMSEs reflects sampling uncertainty under spatial
    autocorrelation.

    Parameters
    ----------
    sim : xarray.DataArray
        Per-experiment simulated field with dims ``(exp_id, y, x)``.
    obs : xarray.DataArray
        Observed field with dims ``(y, x)`` aligned with ``sim``.
    block_size : int
        Block side in pixels. Must be ≥ 1; typically chosen as
        ``ceil(L / pixel_size)`` where ``L`` is the decorrelation length.
    n_boot : int, default ``500``
        Number of bootstrap resamples.
    seed : int, default ``0``
        Seed for :class:`numpy.random.Generator`. Use a fixed value to
        make the bootstrap deterministic.

    Returns
    -------
    xarray.DataArray
        RMSE distribution with dims ``(exp_id, boot)``, where ``boot``
        ranges over the bootstrap resamples. Aggregate with
        ``.mean(dim="boot")`` for the central RMSE and
        ``.quantile([0.05, 0.95], dim="boot")`` for confidence bands.
    """
    sim_v = np.asarray(sim.values, dtype=float)
    obs_v = np.asarray(obs.values, dtype=float)
    sq_err = (sim_v - obs_v[None, :, :]) ** 2
    valid = np.isfinite(sq_err).all(axis=0)
    ny, nx = obs_v.shape
    by = max(1, ny // block_size)
    bx = max(1, nx // block_size)
    block_sums = np.zeros((sim_v.shape[0], by, bx))
    block_counts = np.zeros((by, bx), dtype=int)
    for i in range(by):
        for j in range(bx):
            ys = slice(i * block_size, (i + 1) * block_size)
            xs = slice(j * block_size, (j + 1) * block_size)
            v = valid[ys, xs]
            block_counts[i, j] = int(v.sum())
            chunk = np.where(v, sq_err[:, ys, xs], 0.0)
            block_sums[:, i, j] = chunk.sum(axis=(1, 2))
    block_sums = block_sums.reshape(sim_v.shape[0], -1)
    block_counts = block_counts.reshape(-1)
    valid_blocks = np.where(block_counts > 0)[0]
    rng = np.random.default_rng(seed)
    rmses = np.empty((sim_v.shape[0], n_boot))
    for b in range(n_boot):
        idx = rng.choice(valid_blocks, size=valid_blocks.size, replace=True)
        s = block_sums[:, idx].sum(axis=1)
        c = block_counts[idx].sum()
        rmses[:, b] = np.sqrt(s / max(c, 1))
    return xr.DataArray(
        rmses,
        dims=["exp_id", "boot"],
        coords={"exp_id": sim.exp_id, "boot": np.arange(n_boot)},
    )


data_dir = "~/base/pism-terra"

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

m_vars = ["surface_melt_flux", "surface_runoff_flux", "climatic_mass_balance"]

obs = xr.open_dataset(
    f"{data_dir}/2026_06_kitp_debm_calib/kitp/input/v4/HIRHAM5-ERA5_YMM_1990_2019_v4.nc",
    engine="netcdf4",
    decode_times=False,
    decode_timedelta=False,
    chunks=None,
).drop_dims("nv", errors="ignore")

obs = obs.pint.quantify()
for v in ["surface_melt_flux", "surface_runoff_flux", "climatic_mass_balance"]:
    obs[v] = obs[v].pint.to("kg m^-2 yr^-1")
obs = obs.pint.dequantify()
for v in ["surface_melt_flux", "surface_runoff_flux", "climatic_mass_balance"]:
    obs[f"{v}_error"] = xr.where(obs[v] != 0, 0.10 * obs[v], 1e-8)

for (
    ebm,
    ebm_uq_vars,
) in zip(["debm"], [debm_uq_vars]):

    ds = (
        xr.open_mfdataset(
            f"{data_dir}/2026_06_kitp_{ebm}_calib/output/processed_spatial/clipped_spatial_g4800m_id_HIRHAM5-ERA5_YMM_1990_2019_uq_*.nc",
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
    for v in ["surface_melt_flux", "surface_runoff_flux", "climatic_mass_balance"]:
        ds[v] = ds[v].pint.to("kg m^-2 yr^-1")
    ds = ds.pint.dequantify()

    ebm_uq_df = ds.pism_config.to_series().apply(json.loads).apply(pd.Series)[ebm_uq_vars.keys()]
    ds["time"] = obs["time"]

    _obs = obs.regrid.conservative(ds.drop_vars("pism_config")).squeeze()
    mask = ds[m_vars].isel(exp_id=0).notnull()
    _obs[m_vars] = _obs[m_vars].where(mask)
    melt_mask = _obs["climatic_mass_balance"].mean(dim="time") < 1e36
    _obs[m_vars] = _obs[m_vars].where(melt_mask)
    _ds = ds[m_vars].where(melt_mask)

    for v in ["climatic_mass_balance"]:

        for fudge_factor in [1, 10, 100, 1000, 10_000, 100_000]:
            with ProgressBar():
                ebm_filtered = importance_sampling(
                    _ds,
                    _obs,
                    sim_var=v,
                    obs_mean_var=v,
                    obs_std_var=f"{v}_error",
                    sum_dims=["time", "x", "y"],
                    n_samples=ds.exp_id.size,
                    fudge_factor=fudge_factor,
                )

                ebm_sampled_ids = ebm_filtered.exp_id_sampled.values
                ebm_counts = pd.Series(ebm_sampled_ids).value_counts()

                # Reindex config_df to the sampled IDs and plot histograms
                ds_sampled_configs = ebm_uq_df.loc[ebm_counts.index].reindex(ebm_counts.index)
                most_sampled_id = ebm_counts.idxmax()
                most_sampled_params = ebm_uq_df.loc[most_sampled_id]
                print(f"\n{ebm} / {v} — {fudge_factor} — most sampled id={most_sampled_id} (count={ebm_counts.max()})")
                for k, short in ebm_uq_vars.items():
                    print(f"  {short}: {most_sampled_params[k]:.6g}")

                fig, axes = plt.subplots(1, len(ebm_uq_vars), sharey=True, figsize=(6.4, 1.8))
                for ax, (key, value) in zip(axes.flat, ebm_uq_vars.items()):
                    # Repeat each parameter value by its sample count
                    values = np.repeat(
                        ebm_uq_df[key].values, ebm_counts.reindex(ebm_uq_df.index, fill_value=0).values.astype(int)
                    )
                    print(key, np.median(values))
                    ax.hist(values, bins=15)
                    ax.set_xlabel(value)
                    ax.set_xlim(ebm_uq_df[key].min(), ebm_uq_df[key].max())
                    # ax.set_ylabel("Count")

                fig.tight_layout()
                fig.savefig(f"{ebm}_{v}_ff_{fudge_factor}.png", dpi=300)
                plt.close()
                del fig

        with ProgressBar():

            # 1) Decorrelation length from the observed time-mean field.
            sim_mean_all = _ds[v].mean(dim="time").compute()
            obs_mean = _obs[v].mean(dim="time").squeeze().compute()
            pixel_size = float(abs(_obs.x.diff("x").mean()))
            L = decorrelation_length(obs_mean.values, pixel_size)
            block_size = max(1, int(np.ceil(L / pixel_size)))
            print(f"{ebm}/{v}: decorrelation length ≈ {L:.0f} m, block_size = {block_size} px")

            # 2) Block-bootstrap RMSE per exp_id (honors spatial correlation).
            rmse_boot = block_bootstrap_rmse(sim_mean_all, obs_mean, block_size, n_boot=500)
            rmse_mean = rmse_boot.mean(dim="boot")
            rmse_lo = rmse_boot.quantile(0.05, dim="boot")
            rmse_hi = rmse_boot.quantile(0.95, dim="boot")

            # 3) Rank by bootstrap-mean RMSE; treat exp_ids whose CI overlaps
            # the leader's upper bound as statistically tied with the best.
            best_id = rmse_mean.idxmin(dim="exp_id").values
            best_hi = float(rmse_hi.sel(exp_id=best_id))
            tied_mask = rmse_lo <= best_hi
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

            fig, axes = plt.subplots(1, len(ebm_uq_vars), sharey=True, figsize=(6.4, 1.8))
            for ax, (key, value) in zip(axes.flat, ebm_uq_vars.items()):
                # Repeat each parameter value by its sample count (= 1 if
                # the experiment tied with the best, 0 otherwise).
                values = np.repeat(
                    ebm_uq_df[key].values, ebm_counts.reindex(ebm_uq_df.index, fill_value=0).values.astype(int)
                )
                ax.hist(values, bins=15)
                ax.set_xlabel(value)
                ax.set_xlim(ebm_uq_df[key].min(), ebm_uq_df[key].max())
                # ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(f"{ebm}_{v}.png", dpi=300)
            plt.close()
            del fig

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

            sim_best = _ds[v].sel(exp_id=best_id).mean(dim="time").squeeze()
            vmin = min(float(obs_mean.min()), float(sim_best.min()))
            vmax = max(float(obs_mean.max()), float(sim_best.max()))
            best_params = ebm_uq_df.loc[best_id]
            fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
            obs_mean.plot(ax=axes[0], vmin=vmin, vmax=vmax)
            axes[0].set_title("Observed")
            sim_best.plot(ax=axes[1], vmin=vmin, vmax=vmax)
            param_str = ", ".join(f"{v}={best_params[k]:.4g}" for k, v in ebm_uq_vars.items())
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
