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

import dask
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
    block_sums, block_counts = squared_error_blocks(sim, obs, block_size)
    return bootstrap_rmse_from_blocks(block_sums, block_counts, sim.exp_id, n_boot=n_boot, seed=seed)


def squared_error_blocks(sim, obs, block_size):
    """
    Sum the squared simulation error over non-overlapping square blocks.

    The reduction is expressed with :meth:`xarray.DataArray.coarsen`, so a
    lazy (dask-backed) ``sim`` is streamed block by block: only the
    per-block sums are materialised, never the full ``(exp_id, y, x)``
    error field. Blocks that do not fit a whole ``block_size`` at the top
    or right edge are trimmed, matching a plain tiling of the domain.

    Parameters
    ----------
    sim : xarray.DataArray
        Per-experiment simulated field with dims ``(exp_id, y, x)``. May be
        dask-backed; it is computed exactly once.
    obs : xarray.DataArray
        Observed field with dims ``(y, x)`` aligned with ``sim``.
    block_size : int
        Block side in pixels; clamped to the domain so that at least one
        block is produced.

    Returns
    -------
    block_sums : numpy.ndarray
        Summed squared error, shape ``(n_exp, n_blocks)``. Cells that are
        non-finite in any experiment contribute zero.
    block_counts : numpy.ndarray
        Number of contributing cells per block, shape ``(n_blocks,)``.
    """
    block_y = max(1, min(block_size, sim.sizes["y"]))
    block_x = max(1, min(block_size, sim.sizes["x"]))
    sq_err = (sim - obs) ** 2
    valid = np.isfinite(sq_err).all(dim="exp_id")
    windows = {"y": block_y, "x": block_x, "boundary": "trim"}
    sums = sq_err.where(valid, 0.0).coarsen(**windows).sum().stack(block=("y", "x"))
    counts = valid.astype("int64").coarsen(**windows).sum().stack(block=("y", "x"))
    # One compute: `sums` and `counts` share the `sq_err` sub-graph, so the
    # inputs are read from disk a single time.
    sums, counts = dask.compute(sums, counts)
    return np.asarray(sums.transpose("exp_id", "block").values, dtype=float), np.asarray(counts.values, dtype=int)


def bootstrap_rmse_from_blocks(block_sums, block_counts, exp_id, n_boot=500, seed=0):
    """
    Bootstrap RMSE from pre-computed per-block squared-error sums.

    Parameters
    ----------
    block_sums : numpy.ndarray
        Summed squared error per experiment and block, shape
        ``(n_exp, n_blocks)``, as returned by :func:`squared_error_blocks`.
    block_counts : numpy.ndarray
        Contributing cells per block, shape ``(n_blocks,)``.
    exp_id : array-like
        Experiment labels used as the ``exp_id`` coordinate of the result.
    n_boot : int, default ``500``
        Number of bootstrap resamples.
    seed : int, default ``0``
        Seed for :class:`numpy.random.Generator`. Use a fixed value to
        make the bootstrap deterministic.

    Returns
    -------
    xarray.DataArray
        RMSE distribution with dims ``(exp_id, boot)``.
    """
    valid_blocks = np.where(block_counts > 0)[0]
    rng = np.random.default_rng(seed)
    rmses = np.empty((block_sums.shape[0], n_boot))
    for b in range(n_boot):
        idx = rng.choice(valid_blocks, size=valid_blocks.size, replace=True)
        s = block_sums[:, idx].sum(axis=1)
        c = block_counts[idx].sum()
        rmses[:, b] = np.sqrt(s / max(c, 1))
    return xr.DataArray(
        rmses,
        dims=["exp_id", "boot"],
        coords={"exp_id": exp_id, "boot": np.arange(n_boot)},
    )


def observation_uncertainty(obs, relative=0.10, floor=50.0):
    """
    Attach a ``<var>_error`` field to every variable of an observation set.

    The RCM fields carry no uncertainty of their own, so the likelihood uses
    a relative error with an absolute floor: ``max(relative * |obs|, floor)``.
    The floor keeps near-zero cells (the accumulation zone in the melt
    fields, the equilibrium line in the balance) from becoming infinitely
    informative, which is what a purely relative error does there.

    Parameters
    ----------
    obs : xarray.Dataset
        Observed fields, all in the same units.
    relative : float, default ``0.10``
        Relative error as a fraction of the absolute value.
    floor : float, default ``50.0``
        Smallest error, in the units of ``obs`` (kg m^-2 yr^-1 for the
        mass-balance fields: 5 cm w.e. per year).

    Returns
    -------
    xarray.Dataset
        ``obs`` with one ``<var>_error`` per data variable.
    """
    errors = {f"{v}_error": np.maximum(relative * abs(obs[v]), floor) for v in obs.data_vars}
    return obs.assign(**errors)


def importance_weights(sim, obs, var, *, fudge_factors=(1.0, 3.0, 10.0), n_samples=10_000, seed=0, dim="exp_id"):
    """
    Importance-sample one field for several fudge factors on its error.

    Wraps :func:`pism_terra.filtering.importance_sampling`; the weights are
    the Gaussian log-likelihood of ``var`` averaged over ``time``, ``y`` and
    ``x``, with the observed error scaled by the fudge factor. Resampling
    resolves the weights into counts; ``n_samples`` only sets that
    resolution (the share of a member with weight *w* has standard error
    ``sqrt(w (1 - w) / n_samples)``), so the default resolves shares to well
    under one percent whatever the ensemble size.

    Parameters
    ----------
    sim : xarray.Dataset
        Ensemble with dims ``(dim, time, y, x)`` holding ``var``.
    obs : xarray.Dataset
        Observations on the same grid holding ``var`` and ``<var>_error``.
    var : str
        Field to compare.
    fudge_factors : sequence of float, default ``(1, 3, 10)``
        Multipliers on the observed error, one filter per value.
    n_samples : int, default ``10_000``
        Draws with replacement per fudge factor.
    seed : int, default ``0``
        Seed of the resampler.
    dim : str, default ``"exp_id"``
        Ensemble dimension.

    Returns
    -------
    xarray.Dataset
        ``weights`` and ``counts`` on ``(fudge_factor, dim)`` and ``ess`` on
        ``fudge_factor``: the effective sample size ``1 / sum(w**2)``, which
        equals the ensemble size when the field cannot tell the members apart
        and 1 when a single member carries all the weight.
    """
    weights, counts = [], []
    for fudge_factor in fudge_factors:
        filtered = importance_sampling(
            sim[[var]],
            obs[[var, f"{var}_error"]],
            sim_var=var,
            obs_mean_var=var,
            obs_std_var=f"{var}_error",
            sum_dims=["time", "x", "y"],
            fudge_factor=fudge_factor,
            n_samples=n_samples,
            seed=seed,
            dim=dim,
        )
        w = filtered["weights"].squeeze(drop=True)
        weights.append(w.transpose(dim))
        sampled = filtered[f"{dim}_sampled"].values.ravel()
        counts.append(xr.DataArray([(sampled == i).sum() for i in w[dim].values], dims=[dim], coords={dim: w[dim]}))
    weights_da = xr.concat(weights, dim="fudge_factor").assign_coords(fudge_factor=list(fudge_factors))
    counts_da = xr.concat(counts, dim="fudge_factor").assign_coords(fudge_factor=list(fudge_factors))
    ess = 1.0 / (weights_da**2).sum(dim=dim)
    return xr.Dataset({"weights": weights_da, "counts": counts_da, "ess": ess})


def plot_parameter_histograms(uq_df, uq_vars, counts, filename):
    """
    Histogram each UQ parameter with every member repeated ``counts`` times.

    Parameters
    ----------
    uq_df : pandas.DataFrame
        Parameter values per member, indexed by experiment id.
    uq_vars : dict
        Mapping of parameter column to short label.
    counts : pandas.Series
        Repetitions per member (importance-sampling counts, or 0/1 for the
        bootstrap tie test), indexed by experiment id.
    filename : str or pathlib.Path
        Output figure.
    """
    fig, axes = plt.subplots(1, len(uq_vars), sharey=True, figsize=(6.4, 1.8))
    repeats = counts.reindex(uq_df.index, fill_value=0).values.astype(int)
    for ax, (key, value) in zip(axes.flat, uq_vars.items()):
        ax.hist(np.repeat(uq_df[key].values, repeats), bins=15)
        ax.set_xlabel(value)
        ax.set_xlim(uq_df[key].min(), uq_df[key].max())
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


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
                importance_df = pd.concat(
                    {
                        f"{name}_ff_{fudge_factor:g}": weighted[name].sel(fudge_factor=fudge_factor).to_pandas()
                        for fudge_factor in weighted.fudge_factor.values
                        for name in ("weights", "counts")
                    },
                    axis=1,
                )
                importance_df.index.name = "exp_id"
                importance_df.join(ebm_uq_df, how="left").to_csv(f"{ebm}_{v}_importance.csv")

                # 1) Decorrelation length from the observed time-mean field.
                sim_mean_all = _ds[v].mean(dim="time")
                obs_mean = _obs[v].mean(dim="time").squeeze().compute()
                pixel_size = float(abs(_obs.x.diff("x").mean()))
                L = decorrelation_length(obs_mean.values, pixel_size)
                block_size = max(1, int(np.ceil(L / pixel_size)))
                print(f"{ebm}/{v}: decorrelation length ≈ {L:.0f} m, block_size = {block_size} px")

                # 2) Block-bootstrap RMSE per exp_id (honors spatial correlation).
                # ``sim_mean_all`` stays lazy: the ensemble is reduced to
                # per-block sums as it streams off disk, so peak memory does
                # not grow with the number of experiments.
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
        "--data-dir",
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
        options.data_dir,
        fudge_factors=options.fudge_factors,
        n_samples=options.n_samples,
        relative_error=options.relative_error,
        error_floor=options.error_floor,
    )


if __name__ == "__main__":
    main()
