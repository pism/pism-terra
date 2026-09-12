"""
Ensemble calibration building blocks shared by the KITP and glacier drivers.

Two ways of confronting an ensemble with a gridded observation live here:

* **Importance sampling.** Every member gets a Gaussian likelihood weight from
  its misfit, with the observed error widened by a *fudge factor*; the weights
  are resolved into resampling counts, an effective sample size says how many
  members the field really distinguishes, and the counts turn the ensemble's
  parameter draws into posterior histograms.
* **Block-bootstrap RMSE ranking.** Pixel-wise RMSE treats every cell as
  independent, which glaciological fields are not. The domain is tiled into
  blocks of the field's decorrelation length, blocks are resampled with
  replacement, and every member whose 5-95 % RMSE interval overlaps the
  leader's is reported as statistically tied with the best.

Both work on ``(member, y, x)`` fields whatever the member dimension is called
(``exp_id`` for the ice-sheet ensembles, ``uq_id`` for the glacier ones).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import dask
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import xarray as xr

from pism_terra.filtering import importance_sampling
from pism_terra.likelihood import REDUCTIONS


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


def squared_error_blocks(sim, obs, block_size, dim="exp_id"):
    """
    Sum the squared simulation error over non-overlapping square blocks.

    The reduction is expressed with :meth:`xarray.DataArray.coarsen`, so a
    lazy (dask-backed) ``sim`` is streamed block by block: only the
    per-block sums are materialised, never the full ``(dim, y, x)``
    error field. Blocks that do not fit a whole ``block_size`` at the top
    or right edge are trimmed, matching a plain tiling of the domain.

    Parameters
    ----------
    sim : xarray.DataArray
        Per-member simulated field with dims ``(dim, y, x)``. May be
        dask-backed; it is computed exactly once.
    obs : xarray.DataArray
        Observed field with dims ``(y, x)`` aligned with ``sim``.
    block_size : int
        Block side in pixels; clamped to the domain so that at least one
        block is produced.
    dim : str, default ``"exp_id"``
        Member dimension of ``sim``.

    Returns
    -------
    block_sums : numpy.ndarray
        Summed squared error, shape ``(n_members, n_blocks)``. Cells that are
        non-finite in any member contribute zero.
    block_counts : numpy.ndarray
        Number of contributing cells per block, shape ``(n_blocks,)``.
    """
    block_y = max(1, min(block_size, sim.sizes["y"]))
    block_x = max(1, min(block_size, sim.sizes["x"]))
    sq_err = (sim - obs) ** 2
    valid = np.isfinite(sq_err).all(dim=dim)
    windows = {"y": block_y, "x": block_x, "boundary": "trim"}
    sums = sq_err.where(valid, 0.0).coarsen(**windows).sum().stack(block=("y", "x"))
    counts = valid.astype("int64").coarsen(**windows).sum().stack(block=("y", "x"))
    # One compute: `sums` and `counts` share the `sq_err` sub-graph, so the
    # inputs are read from disk a single time.
    sums, counts = dask.compute(sums, counts)
    return np.asarray(sums.transpose(dim, "block").values, dtype=float), np.asarray(counts.values, dtype=int)


def bootstrap_rmse_from_blocks(block_sums, block_counts, members, n_boot=500, seed=0, *, dim="exp_id"):
    """
    Bootstrap RMSE from pre-computed per-block squared-error sums.

    Parameters
    ----------
    block_sums : numpy.ndarray
        Summed squared error per member and block, shape
        ``(n_members, n_blocks)``, as returned by :func:`squared_error_blocks`.
    block_counts : numpy.ndarray
        Contributing cells per block, shape ``(n_blocks,)``.
    members : array-like
        Member labels used as the ``dim`` coordinate of the result.
    n_boot : int, default ``500``
        Number of bootstrap resamples.
    seed : int, default ``0``
        Seed for :class:`numpy.random.Generator`. Use a fixed value to
        make the bootstrap deterministic.
    dim : str, default ``"exp_id"``
        Name of the member dimension.

    Returns
    -------
    xarray.DataArray
        RMSE distribution with dims ``(dim, boot)``.
    """
    valid_blocks = np.where(block_counts > 0)[0]
    rng = np.random.default_rng(seed)
    rmses = np.empty((block_sums.shape[0], n_boot))
    for b in range(n_boot):
        idx = rng.choice(valid_blocks, size=valid_blocks.size, replace=True)
        s = block_sums[:, idx].sum(axis=1)
        c = block_counts[idx].sum()
        rmses[:, b] = np.sqrt(s / max(c, 1))
    return xr.DataArray(rmses, dims=[dim, "boot"], coords={dim: np.asarray(members), "boot": np.arange(n_boot)})


def block_bootstrap_rmse(sim, obs, block_size, n_boot=500, seed=0, *, dim="exp_id"):
    """
    Block-bootstrap spatial RMSE per member.

    The domain is tiled into non-overlapping square blocks of side
    ``block_size`` pixels. For each bootstrap iteration, blocks are drawn
    with replacement and a single global RMSE is computed across the
    resampled blocks for every member in ``sim``. Choosing
    ``block_size`` ≳ ``decorrelation_length(obs) / pixel_size`` makes the
    resampled blocks (approximately) independent, so the spread of
    bootstrap RMSEs reflects sampling uncertainty under spatial
    autocorrelation.

    Parameters
    ----------
    sim : xarray.DataArray
        Per-member simulated field with dims ``(dim, y, x)``.
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
    dim : str, default ``"exp_id"``
        Member dimension of ``sim``.

    Returns
    -------
    xarray.DataArray
        RMSE distribution with dims ``(dim, boot)``, where ``boot``
        ranges over the bootstrap resamples. Aggregate with
        ``.mean(dim="boot")`` for the central RMSE and
        ``.quantile([0.05, 0.95], dim="boot")`` for confidence bands.
    """
    block_sums, block_counts = squared_error_blocks(sim, obs, block_size, dim=dim)
    return bootstrap_rmse_from_blocks(block_sums, block_counts, sim[dim].values, n_boot=n_boot, seed=seed, dim=dim)


def block_size_from_field(obs, threshold=1.0 / np.e):
    """
    Block side, in cells, that makes blocks of a field approximately independent.

    Parameters
    ----------
    obs : xarray.DataArray
        Observed field with dims ``(y, x)``; a ``time`` dimension is averaged out.
    threshold : float, default ``1 / e``
        ACF level defining the decorrelation length.

    Returns
    -------
    float
        Decorrelation length in the grid's units (NaN when the field is empty).
    int
        Block side in cells, at least 1.
    """
    if "time" in obs.dims:
        obs = obs.mean(dim="time")
    obs = obs.compute()
    pixel_size = float(abs(obs.x.diff("x").mean()))
    length = decorrelation_length(obs.values, pixel_size, threshold=threshold)
    block_size = max(1, int(np.ceil(length / pixel_size))) if np.isfinite(length) else 1
    return float(length), block_size


def rank_by_bootstrap_rmse(
    sim, obs, *, n_boot=500, seed=0, dim="exp_id", pctls=(0.05, 0.95), threshold=1.0 / np.e, block_size=None
):
    """
    Rank members by block-bootstrap RMSE and flag the ones tied with the best.

    The block side is the observed field's decorrelation length in pixels
    (:func:`decorrelation_length`); the bootstrap spread then reflects the
    sampling uncertainty under spatial autocorrelation. A member is *tied*
    with the leader when the lower bound of its RMSE interval does not exceed
    the leader's upper bound.

    Parameters
    ----------
    sim : xarray.DataArray
        Per-member time-mean field with dims ``(dim, y, x)``; may be lazy.
    obs : xarray.DataArray
        Observed time-mean field with dims ``(y, x)`` on the same grid.
    n_boot : int, default ``500``
        Number of bootstrap resamples.
    seed : int, default ``0``
        Seed of the bootstrap.
    dim : str, default ``"exp_id"``
        Member dimension of ``sim``.
    pctls : tuple of float, default ``(0.05, 0.95)``
        Lower and upper percentiles of the RMSE interval.
    threshold : float, default ``1 / e``
        ACF level defining the decorrelation length.
    block_size : int or None, optional
        Block side in cells; computed from ``obs`` with :func:`block_size_from_field` when None.

    Returns
    -------
    xarray.Dataset
        ``rmse_mean``, ``rmse_lo``, ``rmse_hi`` and boolean ``tied_with_best``
        on ``dim``; attrs ``decorrelation_length`` (grid units),
        ``block_size`` (pixels) and ``best`` (label of the leader).
    """
    obs = obs.compute()
    length, auto_block = block_size_from_field(obs, threshold=threshold)
    block_size = auto_block if block_size is None else int(block_size)
    rmse_boot = block_bootstrap_rmse(sim, obs, block_size, n_boot=n_boot, seed=seed, dim=dim)
    rmse_mean = rmse_boot.mean(dim="boot")
    rmse_lo = rmse_boot.quantile(pctls[0], dim="boot").drop_vars("quantile")
    rmse_hi = rmse_boot.quantile(pctls[1], dim="boot").drop_vars("quantile")
    best = rmse_mean.idxmin(dim=dim).values
    tied = rmse_lo <= float(rmse_hi.sel({dim: best}))
    ranking = xr.Dataset({"rmse_mean": rmse_mean, "rmse_lo": rmse_lo, "rmse_hi": rmse_hi, "tied_with_best": tied})
    ranking.attrs.update({"decorrelation_length": float(length), "block_size": int(block_size), "best": str(best)})
    return ranking


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


def _resample_counts(weights: np.ndarray, rng: np.random.Generator, n_samples: int) -> np.ndarray:
    """
    Draw members with replacement according to their weights and count the draws.

    Parameters
    ----------
    weights : numpy.ndarray
        Normalised weights of one member vector; NaNs count as zero.
    rng : numpy.random.Generator
        Random generator shared across calls so the draws are reproducible.
    n_samples : int
        Number of draws.

    Returns
    -------
    numpy.ndarray
        Draws per member, summing to ``n_samples``.
    """
    p = np.where(np.isfinite(weights), weights, 0.0)
    p = p / p.sum() if p.sum() > 0 else np.full_like(p, 1.0 / p.size)
    drawn = rng.choice(p.size, size=n_samples, replace=True, p=p)
    return np.bincount(drawn, minlength=p.size)


def weights_from_log_likelihood(log_likelihood, *, dim="exp_id", n_samples=10_000, seed=0):
    """
    Turn per-member log-likelihoods into weights, resampling counts and the ESS.

    Parameters
    ----------
    log_likelihood : xarray.DataArray
        Log-likelihood per member on ``dim``; any other dimension (typically
        ``fudge_factor``) is handled independently.
    dim : str, default ``"exp_id"``
        Member dimension.
    n_samples : int, default ``10_000``
        Draws with replacement that resolve the weights into counts. This only
        sets the resolution: the share of a member with weight *w* has
        standard error ``sqrt(w (1 - w) / n_samples)``.
    seed : int, default ``0``
        Seed of the resampler.

    Returns
    -------
    xarray.Dataset
        ``log_likelihood``, ``weights`` and ``counts`` on the input dims, and
        ``ess`` on the remaining dims: the effective sample size
        ``1 / sum(w**2)``, equal to the ensemble size when the field cannot
        tell the members apart and 1 when a single member carries all the
        weight.
    """
    scaled = log_likelihood - log_likelihood.max(dim=dim)
    weights = np.exp(scaled)
    weights = weights / weights.sum(dim=dim)
    weights.name = "weights"
    rng = np.random.default_rng(seed)
    members = log_likelihood[dim].values
    counts = xr.apply_ufunc(
        _resample_counts,
        weights,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        output_dtypes=[int],
        kwargs={"rng": rng, "n_samples": n_samples},
    ).assign_coords({dim: members})
    counts.name = "counts"
    ess = 1.0 / (weights**2).sum(dim=dim)
    ess.name = "ess"
    return xr.Dataset({"log_likelihood": log_likelihood, "weights": weights, "counts": counts, "ess": ess})


def importance_weights(
    sim,
    obs,
    var,
    *,
    obs_var=None,
    obs_std_var=None,
    fudge_factors=(1.0, 3.0, 10.0),
    n_samples=10_000,
    seed=0,
    dim="exp_id",
    sum_dims=("time", "x", "y"),
    reduction="blocks",
    block_size=None,
    threshold=1.0 / np.e,
):
    """
    Importance-sample one field for several fudge factors on its error.

    Wraps :func:`pism_terra.filtering.importance_sampling` (Gaussian
    log-likelihood of ``var`` averaged over ``sum_dims``, with the observed
    error scaled by the fudge factor) and resolves the weights into counts
    and an effective sample size with :func:`weights_from_log_likelihood`.
    ``sim`` and ``obs`` must already share their grid.

    Parameters
    ----------
    sim : xarray.Dataset
        Ensemble with dims ``(dim, ...)`` holding ``var``.
    obs : xarray.Dataset
        Observations on the same grid holding ``obs_var`` and ``obs_std_var``.
    var : str
        Simulated field to compare.
    obs_var : str or None, optional
        Observed mean; defaults to ``var``.
    obs_std_var : str or None, optional
        Observed uncertainty; defaults to ``<obs_var>_error``.
    fudge_factors : sequence of float, default ``(1, 3, 10)``
        Multipliers on the observed error, one filter per value.
    n_samples : int, default ``10_000``
        Draws with replacement per fudge factor.
    seed : int, default ``0``
        Seed of the resampler.
    dim : str, default ``"exp_id"``
        Ensemble dimension.
    sum_dims : sequence of str, default ``("time", "x", "y")``
        Dimensions the log-likelihood is reduced over; missing ones are skipped.
    reduction : {"blocks", "mean", "sum"}, default ``"blocks"``
        How the cells are collapsed (:func:`pism_terra.likelihood.reduce_log_likelihood`).
        ``"blocks"`` sums one independent sample per decorrelation-length block,
        ``"mean"`` averages (the former behaviour, a strongly tempered posterior)
        and ``"sum"`` treats every cell as independent.
    block_size : int or None, optional
        Block side in cells for ``"blocks"``; computed from the observed field
        with :func:`block_size_from_field` when None.
    threshold : float, default ``1 / e``
        ACF level defining the decorrelation length when ``block_size`` is
        computed; lower values give longer blocks and a softer posterior.

    Returns
    -------
    xarray.Dataset
        ``log_likelihood``, ``weights`` and ``counts`` on ``(fudge_factor, dim)``
        and ``ess`` on ``fudge_factor``; attrs ``reduction`` and ``block_size``.

    Raises
    ------
    ValueError
        If ``reduction`` is unknown.
    """
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
    obs_var = var if obs_var is None else obs_var
    obs_std_var = f"{obs_var}_error" if obs_std_var is None else obs_std_var
    dims = [d for d in sum_dims if d in sim[var].dims]
    if reduction == "blocks" and block_size is None:
        _, block_size = block_size_from_field(obs[obs_var], threshold=threshold)
    likelihood_kwargs = {"reduction": reduction, "block_size": 1 if block_size is None else int(block_size)}
    log_likes = []
    for fudge_factor in fudge_factors:
        filtered = importance_sampling(
            sim[[var]],
            obs[[obs_var, obs_std_var]],
            sim_var=var,
            obs_mean_var=obs_var,
            obs_std_var=obs_std_var,
            sum_dims=dims,
            likelihood_kwargs=likelihood_kwargs,
            fudge_factor=fudge_factor,
            n_samples=1,
            seed=seed,
            dim=dim,
        )
        ll = filtered["log_likelihood"]
        ll = ll.squeeze([d for d in ll.dims if d != dim and ll.sizes[d] == 1], drop=True)
        log_likes.append(ll.transpose(dim))
    log_likelihood = xr.concat(log_likes, dim="fudge_factor").assign_coords(fudge_factor=list(fudge_factors))
    weighted = weights_from_log_likelihood(log_likelihood, dim=dim, n_samples=n_samples, seed=seed)
    weighted.attrs.update({"reduction": reduction, "block_size": likelihood_kwargs["block_size"]})
    return weighted


def joint_log_likelihood(per_glacier: Mapping[str, xr.DataArray], dim="uq_id") -> tuple[xr.DataArray, list]:
    """
    Sum log-likelihoods over glaciers for the members every glacier has.

    The ensemble members are shared parameter draws, so their evidence adds
    up across glaciers. How much each glacier counts follows from the
    reduction used per glacier: with ``"blocks"`` a glacier contributes one
    independent sample per decorrelation-length block, so larger glaciers
    weigh more; with ``"mean"`` every glacier weighs the same.

    Parameters
    ----------
    per_glacier : mapping of str to xarray.DataArray
        Log-likelihood per member (and fudge factor) keyed by glacier.
    dim : str, default ``"uq_id"``
        Member dimension.

    Returns
    -------
    xarray.DataArray
        Summed log-likelihood over the common members.
    list
        Labels of the common members, in the order they appear in the sum.
    """
    common = None
    for da in per_glacier.values():
        labels = set(np.asarray(da[dim].values).tolist())
        common = labels if common is None else common & labels
    members = sorted(common or [], key=lambda v: (len(str(v)), str(v)))
    if not members:
        raise ValueError("no ensemble member is present for every glacier; nothing to combine")
    total = sum(da.sel({dim: members}) for da in per_glacier.values())
    total.name = "log_likelihood"
    return total, members


def posterior_table(weighted: xr.Dataset, uq_df: pd.DataFrame, dim="exp_id") -> pd.DataFrame:
    """
    Tabulate weights and counts per member next to the member's parameters.

    Parameters
    ----------
    weighted : xarray.Dataset
        Output of :func:`importance_weights` or :func:`weights_from_log_likelihood`.
    uq_df : pandas.DataFrame
        Parameter values per member, indexed like ``weighted[dim]``.
    dim : str, default ``"exp_id"``
        Member dimension.

    Returns
    -------
    pandas.DataFrame
        One row per member: the parameters, then ``log_likelihood_ff_<f>``,
        ``weights_ff_<f>`` and ``counts_ff_<f>`` per fudge factor.
    """
    columns = {}
    for fudge_factor in weighted.fudge_factor.values:
        for name in ("log_likelihood", "weights", "counts"):
            columns[f"{name}_ff_{fudge_factor:g}"] = weighted[name].sel(fudge_factor=fudge_factor).to_pandas()
    table = pd.concat(columns, axis=1)
    table.index.name = dim
    return uq_df.reindex(table.index).join(table, how="right")


def plot_parameter_histograms(uq_df, uq_vars, counts, filename, *, prior=False, title=None, bins=15):
    """
    Histogram each UQ parameter with every member repeated ``counts`` times.

    Parameters
    ----------
    uq_df : pandas.DataFrame
        Parameter values per member, indexed by member id.
    uq_vars : dict
        Mapping of parameter column to short label.
    counts : pandas.Series
        Repetitions per member (importance-sampling counts, or 0/1 for the
        bootstrap tie test), indexed by member id.
    filename : str or pathlib.Path
        Output figure.
    prior : bool, default ``False``
        Draw the prior (every member once) as an outline behind the posterior.
    title : str or None, optional
        Figure title.
    bins : int, default ``15``
        Histogram bins per parameter.
    """
    fig, axes = plt.subplots(1, len(uq_vars), sharey=False, figsize=(1.6 * len(uq_vars) + 0.8, 1.9))
    repeats = counts.reindex(uq_df.index, fill_value=0).values.astype(int)
    for ax, (key, value) in zip(np.atleast_1d(axes).flat, uq_vars.items()):
        lo, hi = float(uq_df[key].min()), float(uq_df[key].max())
        edges = np.linspace(lo, hi, bins + 1) if hi > lo else bins
        if prior:
            ax.hist(uq_df[key].values, bins=edges, density=True, histtype="step", color="0.4", label="prior")
        ax.hist(np.repeat(uq_df[key].values, repeats), bins=edges, density=True, alpha=0.7, label="posterior")
        ax.set_xlabel(value)
        if hi > lo:
            ax.set_xlim(lo, hi)
        ax.set_yticks([])
    if prior:
        np.atleast_1d(axes).flat[0].legend(fontsize=6, frameon=False)
    if title:
        fig.suptitle(title, fontsize=7)
    fig.tight_layout()
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def short_labels(columns: Iterable[str]) -> dict[str, str]:
    """
    Abbreviate dotted PISM parameter names to their last component.

    Parameters
    ----------
    columns : iterable of str
        Parameter names such as ``surface.pdd.factor_ice``.

    Returns
    -------
    dict
        ``{"surface.pdd.factor_ice": "factor_ice", ...}``; a suffix that is
        not unique keeps one more component.
    """
    columns = list(columns)
    labels = {c: c.split(".")[-1] for c in columns}
    seen: dict[str, int] = {}
    for c in columns:
        seen[labels[c]] = seen.get(labels[c], 0) + 1
    for c in columns:
        if seen[labels[c]] > 1:
            labels[c] = ".".join(c.split(".")[-2:])
    return labels


__all__: Sequence[str] = (
    "block_bootstrap_rmse",
    "block_size_from_field",
    "bootstrap_rmse_from_blocks",
    "decorrelation_length",
    "importance_weights",
    "joint_log_likelihood",
    "observation_uncertainty",
    "plot_parameter_histograms",
    "posterior_table",
    "rank_by_bootstrap_rmse",
    "short_labels",
    "squared_error_blocks",
    "weights_from_log_likelihood",
)
