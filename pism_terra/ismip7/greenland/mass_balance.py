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
Regional mass balance of an ISMIP7 Greenland submission, against Mankoff.

An ISMIP7 submission tree holds one file per variable and Core Experiment
counter, each a ``(time, y, x)`` flux per unit area. This module opens the
tree as one lazy ensemble on ``(gcm_id, ssp_id)``, integrates the surface,
basal and grounding-line fluxes over the Mouginot basins in a single pass
over the data, adds them up into a mass balance, accumulates it from a
reference year, and plots the result per basin next to the Mankoff et al.
(2021) input-output estimate.

The tree can sit in a bucket (``s3://pism-cloud-data/<name>/<project>/output``,
read anonymously) or on disk; everything goes through :mod:`fsspec`, so the
same code serves the cloud runs and a run on chinook. A cloud project keeps
one directory per job, and the counters land in different jobs, so the root
may carry a wildcard where the job id goes:
``s3://pism-cloud-data/<name>/<project>/*/output`` collects every job's
tree (and finds the observations in whichever job staged them). The pieces are meant
to be used one at a time from a notebook as much as from the console script:

>>> from pism_terra.ismip7.greenland import mass_balance as mb   # doctest: +SKIP
>>> ds = mb.open_submission(mb.find_files(root, ["acabf", "ligroundf", "libmassbfgr"]))
>>> regions = mb.compute_regions(ds, outline)                    # one Dask pass
>>> mb.plot_regions(regions, mb.load_mankoff(obs_url), "fluxes.png")

Progress is shown on a terminal: a bar over the files while they are
checked, and the Dask progress display while the basins are integrated
(the distributed one when a :class:`dask.distributed.Client` is active).
"""

from __future__ import annotations

import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence
from functools import partial
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import cf_xarray.units  # pylint: disable=unused-import  # noqa: F401  (teaches pint UDUNITS: "kg m-2 s-1")
import fsspec
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pint
import pint_xarray  # pylint: disable=unused-import  # noqa: F401  (registers the .pint accessor)
import rioxarray  # pylint: disable=unused-import  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from tqdm.contrib.logging import logging_redirect_tqdm

from pism_terra.ismip7.postprocess_flux import submission_crs
from pism_terra.log import setup_logging
from pism_terra.plotting import rc_params
from pism_terra.postprocess_scalar import basin_masks
from pism_terra.processing import (
    integrate_rate,
    normalize_timeseries,
    preprocess_netcdf,
)
from pism_terra.progress import compute, distributed_client, progress_bar
from pism_terra.workflow import drop_grid_mapping

# Named after the module even under ``python -m``, where ``__name__`` is
# ``__main__`` and a logger of that name would sit outside the ``pism_terra``
# tree that :func:`pism_terra.log.setup_logging` writes to the log file.
logger = logging.getLogger("pism_terra.ismip7.greenland.mass_balance" if __name__ == "__main__" else __name__)

#: Fluxes summed into the mass balance: surface mass balance, basal mass
#: balance of grounded ice and the flux across the grounding line, all per
#: unit area in the submission files.
DEFAULT_VARIABLES = ("acabf", "libmassbfgr", "ligroundf")

#: Where a submission tree sits below a run's ``output`` directory.
DEFAULT_TREE = ("GrIS", "UAF", "PISM", "CORE")

#: Reference year the cumulative series is zeroed at.
DEFAULT_REFERENCE_YEAR = "1985"

#: Outline the basins are read from, looked for in the run's observations
#: directory first and in the package's data otherwise.
DEFAULT_OUTLINE = "mouginot_basins_w_shelves.gpkg"

#: The staged Mankoff product, in the run's observations directory.
DEFAULT_MANKOFF = "mankoff_greenland_mass_balance.nc"

#: Name of the whole-ice-sheet total added to the basin sums.
TOTAL_REGION = "GIS_GIS"

#: Units the regional fluxes are reported in.
FLUX_UNITS = "Gt year-1"

#: How the files are opened: h5netcdf reads over fsspec, and each chunk is
#: a few time steps of the whole grid, so the basin integration streams
#: through a file in ~100 MB pieces at 1200 m.
OPEN_KWARGS: dict[str, Any] = {"engine": "h5netcdf"}
DEFAULT_CHUNKS: dict[str, int] = {"time": 5, "y": -1, "x": -1}

#: Line colours per pathway and styles per GCM in the figures.
SSP_COLORS = {
    "historical": "k",
    "OCX": "k",
    "ctrl": "0.5",
    "ssp119": "#00ADCF",
    "ssp126": "#173C66",
    "ssp245": "#F79420",
    "ssp370": "#E71D25",
    "ssp585": "#951B1E",
}
GCM_STYLES = {"MRI-ESM2-0": "dashed", "CESM2-WACCM": "solid"}

#: File-name parsing for the submission tree: the GCM sits between the
#: model and forcing counters, the experiment between the forcing counter
#: and the set counter. There is no RGI id and no UQ draw to read.
preprocess_ismip7 = partial(
    preprocess_netcdf,
    gcm_dim="gcm_id",
    gcm_regexp=r"_m\d+_(.+?)_f\d+_",
    exp_dim="ssp_id",
    exp_regexp=r"_f\d+_(.+?)_[CEP]\d{3}_",
    rgi_dim=None,
    uq_dim=None,
    process_config=False,
)


# --- Files -----------------------------------------------------------------------


def filesystem(root: str) -> tuple[fsspec.AbstractFileSystem, str]:
    """
    Resolve a root that is either an S3 URL or a local path.

    Parameters
    ----------
    root : str
        ``s3://bucket/prefix`` or a directory.

    Returns
    -------
    fsspec.AbstractFileSystem
        The filesystem, anonymous for S3 (the production bucket is public).
    str
        The root path on that filesystem, without the protocol.
    """
    options = {"anon": True} if root.startswith("s3://") else {}
    fs, path = fsspec.core.url_to_fs(root, **options)
    return fs, path.rstrip("/")


def with_protocol(fs: fsspec.AbstractFileSystem, path: str) -> str:
    """
    Put a filesystem's protocol back in front of a path, as xarray wants it.

    Parameters
    ----------
    fs : fsspec.AbstractFileSystem
        Filesystem the path belongs to.
    path : str
        Path without protocol, as :meth:`fs.glob` returns it.

    Returns
    -------
    str
        ``s3://path`` for S3, the path itself for local files.
    """
    protocol = fs.protocol[0] if isinstance(fs.protocol, (tuple, list)) else fs.protocol
    return path if protocol in ("file", "local") else f"{protocol}://{path}"


def first_match(root: str, relative: str) -> str | None:
    """
    The first file matching a path below the root, wildcards in the root included.

    Parameters
    ----------
    root : str
        A run's ``output`` directory, possibly with a wildcard for the job id.
    relative : str
        Path below it, e.g. ``observations/mankoff_greenland_mass_balance.nc``.

    Returns
    -------
    str or None
        URL of the first match in sorted order, or None when there is none.
    """
    fs, path = filesystem(root)
    matches = sorted(fs.glob(f"{path}/{relative}"))
    return with_protocol(fs, matches[0]) if matches else None


def find_files(
    root: str, variables: Sequence[str] = DEFAULT_VARIABLES, tree: Sequence[str] = DEFAULT_TREE
) -> list[str]:
    """
    List the per-counter files of some variables in a submission tree.

    Parameters
    ----------
    root : str
        A run's ``output`` directory, S3 URL or local path.
    variables : sequence of str, optional
        ISMIP7 variable names.
    tree : sequence of str, optional
        Path components from ``root`` to the counter directories.

    Returns
    -------
    list of str
        Sorted file URLs, ready for :func:`open_submission`.
    """
    fs, path = filesystem(root)
    fs.invalidate_cache()
    base = "/".join([path, *tree])
    found = [with_protocol(fs, p) for v in variables for p in fs.glob(f"{base}/*/{v}_*.nc")]
    if not found:
        raise FileNotFoundError(f"no {'/'.join(variables)} files matching {base}/*/<variable>_*.nc")
    logger.info("%d file(s) for %s below %s", len(found), ", ".join(variables), base)
    return sorted(found)


def storage_options(url: str) -> dict[str, Any]:
    """
    The ``storage_options`` xarray needs to open a URL.

    Parameters
    ----------
    url : str
        File URL or path.

    Returns
    -------
    dict
        ``{"storage_options": {"anon": True}}`` for S3, empty otherwise.
    """
    return {"storage_options": {"anon": True}} if str(url).startswith("s3://") else {}


def nonempty(paths: Sequence[str]) -> list[str]:
    """
    Keep the files whose time axis has records.

    A run still in flight, or an rsync still copying, leaves files whose
    ``time`` is empty; opening them into one ensemble fails on the concat.
    Each file's header is read once, which over S3 is one request per file.

    Parameters
    ----------
    paths : sequence of str
        File URLs.

    Returns
    -------
    list of str
        The ones with at least one record, in the given order.
    """
    keep = []
    # Log lines go through tqdm while the bar is up, so they do not tear it.
    with logging_redirect_tqdm():
        for path in progress_bar(paths, desc="Checking files", unit="file"):
            with xr.open_dataset(
                path, decode_times=False, decode_timedelta=False, **OPEN_KWARGS, **storage_options(path)
            ) as ds:
                if ds.sizes.get("time", 1) > 0:
                    keep.append(path)
                else:
                    logger.warning("%s has no records yet; skipped", path)
    return keep


def open_submission(paths: Sequence[str], chunks: dict[str, int] | None = None, **kwargs: Any) -> xr.Dataset:
    """
    Open submission files as one lazy ensemble on ``(gcm_id, ssp_id)``.

    Parameters
    ----------
    paths : sequence of str
        File URLs of one or more variables, one file per variable and
        counter (:func:`find_files`).
    chunks : dict or None, optional
        Dask chunks; :data:`DEFAULT_CHUNKS` when None.
    **kwargs : Any
        Further :func:`xarray.open_mfdataset` options.

    Returns
    -------
    xarray.Dataset
        Lazy dataset with dims ``(gcm_id, ssp_id, time, y, x)``. A pathway a
        GCM never ran is NaN, and the different experiments' time axes are
        joined into one.
    """
    paths = nonempty(paths)
    if not paths:
        raise FileNotFoundError("no submission file with records")
    options = storage_options(paths[0])
    ds = xr.open_mfdataset(
        list(paths),
        preprocess=preprocess_ismip7,
        combine="by_coords",
        chunks=DEFAULT_CHUNKS if chunks is None else chunks,
        decode_timedelta=True,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
        join="outer",
        parallel=True,
        **OPEN_KWARGS,
        **options,
        **kwargs,
    )
    logger.info(
        "Ensemble: GCMs %s, pathways %s, %d time step(s)",
        list(ds["gcm_id"].values),
        list(ds["ssp_id"].values),
        ds.sizes["time"],
    )
    return ds


def splice_historical(ds: xr.Dataset, historical: str = "historical") -> xr.Dataset:
    """
    Prepend a GCM's historical run to each of its pathways.

    Parameters
    ----------
    ds : xarray.Dataset
        Ensemble on ``(gcm_id, ssp_id, time, ...)`` with a historical pathway.
    historical : str, optional
        Name of the historical pathway.

    Returns
    -------
    xarray.Dataset
        The other pathways, each filled with the historical values where it
        has none (before 2015). Still lazy.
    """
    hist = ds.sel(ssp_id=historical, drop=True)
    return ds.drop_sel(ssp_id=historical).combine_first(hist)


# --- Basins ----------------------------------------------------------------------


def read_outline(path: str) -> gpd.GeoDataFrame:
    """
    Read a basin outline from disk or S3.

    GDAL's own ``/vsis3/`` reader wants credentials even for a public
    bucket, so an S3 file is opened through fsspec instead.

    Parameters
    ----------
    path : str
        GeoPackage path or URL.

    Returns
    -------
    geopandas.GeoDataFrame
        The outlines.
    """
    if str(path).startswith("s3://"):
        fs, _ = filesystem(path)
        with fs.open(path, "rb") as handle:
            return gpd.read_file(handle)
    return gpd.read_file(path)


def resolve_outline(root: str, outline: str | None) -> str:
    """
    Decide which outline file to read.

    Parameters
    ----------
    root : str
        A run's ``output`` directory.
    outline : str or None
        A path or URL, a bare file name looked for in ``<root>/observations``,
        or None for :data:`DEFAULT_OUTLINE`.

    Returns
    -------
    str
        Path or URL of the outline. A bare name missing from the run's
        observations falls back to the copy shipped with the package.
    """
    name = outline or DEFAULT_OUTLINE
    if "/" in name or Path(name).is_file():
        return name
    found = first_match(root, f"observations/{name}")
    if found is not None:
        return found
    packaged = files("pism_terra") / "data" / name
    logger.info("%s not in %s/observations; using the packaged %s", name, root, packaged)
    return str(packaged)


def regional_sums(
    ds: xr.Dataset,
    outline: gpd.GeoDataFrame | str,
    *,
    column: str | None = None,
    dim_name: str = "region",
    crs: str | None = None,
    all_touched: bool = False,
    total: str | None = TOTAL_REGION,
    client=None,
) -> xr.Dataset:
    """
    Integrate every gridded flux over each outline, lazily and in one pass.

    Each basin mask is rasterized once, the masks are stacked on
    ``dim_name`` and the fluxes are contracted with them in a single
    ``xr.dot``, so every chunk of the data is read once however many basins
    there are; masking and summing basin by basin would read it once per
    basin. Missing values count as zero, which is right for a flux over
    cells without ice.

    Parameters
    ----------
    ds : xarray.Dataset
        Ensemble with ``(y, x)`` fields in per-area units.
    outline : geopandas.GeoDataFrame or str
        Basin outlines, or where to read them from (:func:`read_outline`).
    column : str or None, optional
        Outline column naming the basins; resolved from the file when None.
    dim_name : str, optional
        Name of the basin dimension.
    crs : str or None, optional
        CRS of the grid, when the files do not say.
    all_touched : bool, optional
        Count cells the outline merely touches, not only those whose centre
        is inside.
    total : str or None, optional
        Name of the whole-domain sum appended to the basins; None adds none.
    client : dask.distributed.Client or None, optional
        Client to scatter the masks to; the active one when None.

    Returns
    -------
    xarray.Dataset
        Lazy per-basin integrals (per-area units times m^2), the outline
        ``area`` in m^2, and no grid mapping.
    """
    dst_crs = submission_crs(ds, crs)
    if not isinstance(outline, gpd.GeoDataFrame):
        outline = read_outline(outline)
    outline = gpd.GeoDataFrame(outline.to_crs(dst_crs), crs=dst_crs)

    spatial_vars = [v for v in ds.data_vars if {"x", "y"} <= set(ds[v].dims)]
    grid = ds[spatial_vars].rio.write_crs(dst_crs).rio.set_spatial_dims(x_dim="x", y_dim="y")
    cell_area = abs(float(ds["x"][1] - ds["x"][0]) * float(ds["y"][1] - ds["y"][0]))

    masks = basin_masks(grid, outline, column=column, all_touched=all_touched, client=client or distributed_client())
    names = [name for name, _ in masks]
    weights = xr.concat([mask for _, mask in masks], dim=dim_name).assign_coords({dim_name: names})
    out = xr.Dataset()
    for var in spatial_vars:
        field = grid[var]
        w = weights.astype(field.dtype)
        # Missing cells count as zero in the integral, but a basin with no
        # data at all -- a pathway this GCM never ran, a year outside a run's
        # span -- must stay missing rather than sum to zero, so the valid
        # cells are counted in the same pass.
        valid = xr.dot(field.notnull().astype(field.dtype), w, dim=("y", "x"))
        out[var] = (xr.dot(field.fillna(0.0), w, dim=("y", "x")) * cell_area).where(valid > 0)
    # kg m-2 s-1 summed over cells x m^2 = kg s-1; pint does the bookkeeping,
    # because the files spell the units several ways ("kg m^-2 s^-1",
    # "kg m^-2 second^-1").
    ureg = pint.application_registry
    for var in spatial_vars:
        if units := ds[var].attrs.get("units"):
            out[var].attrs = dict(ds[var].attrs, units=f"{(ureg(units) * ureg('m^2')).units:~}")
    out["area"] = xr.DataArray(
        [geom.area for geom in outline.geometry],
        dims=(dim_name,),
        coords={dim_name: names},
        attrs={"units": "m^2", "long_name": "outline area"},
    )
    if total is not None:
        # ``min_count`` keeps a step every basin lacks as missing rather than zero.
        with xr.set_options(keep_attrs=True):
            whole = out.sum(dim_name, skipna=True, min_count=1).expand_dims({dim_name: [total]})
            out = xr.concat([out, whole], dim_name)
    return drop_grid_mapping(out)


def to_units(ds: xr.Dataset, target: str = FLUX_UNITS) -> xr.Dataset:
    """
    Convert every variable whose units allow it; leave the rest alone.

    Parameters
    ----------
    ds : xarray.Dataset
        Variables with ``units`` attributes.
    target : str, optional
        Target units.

    Returns
    -------
    xarray.Dataset
        ``ds`` with the compatible variables converted.
    """
    ureg = pint.application_registry
    convertible = [
        v for v in ds.data_vars if ds[v].attrs.get("units") and ureg(ds[v].attrs["units"]).is_compatible_with(target)
    ]
    out = ds[convertible].pint.quantify().pint.to({v: target for v in convertible}).pint.dequantify()
    for v in convertible:
        # pint writes its own spelling of the unit; keep the one asked for.
        out[v].attrs["units"] = target
    return out.assign({v: ds[v] for v in ds.data_vars if v not in convertible})


def compute_regions(
    ds: xr.Dataset,
    outline: gpd.GeoDataFrame | str,
    *,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    reference_year: str = DEFAULT_REFERENCE_YEAR,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Integrate the fluxes over the basins and build the mass balance series.

    This is the one expensive step: the whole ensemble is read once. The
    Dask progress display follows it on a terminal.

    Parameters
    ----------
    ds : xarray.Dataset
        Lazy ensemble (:func:`open_submission`).
    outline : geopandas.GeoDataFrame or str
        Basin outlines.
    variables : sequence of str, optional
        The fluxes summed into ``mass_balance``, when all are present.
    reference_year : str, optional
        Year the cumulative mass balance is zeroed at.
    **kwargs : Any
        Passed to :func:`regional_sums`.

    Returns
    -------
    xarray.Dataset
        In memory: the fluxes in Gt/yr per basin, ``mass_balance`` and
        ``cumulative_mass_balance`` (Gt, zero at ``reference_year``).
    """
    (sums,) = compute(regional_sums(ds, outline, **kwargs), desc="Integrating the fluxes over the basins")
    regions = to_units(sums)
    present = [v for v in variables if v in regions]
    if len(present) < len(variables):
        logger.warning("mass_balance is the sum of %s only; %s missing", present, sorted(set(variables) - set(present)))
    regions["mass_balance"] = sum(regions[v] for v in present)
    regions["mass_balance"].attrs = {
        "units": FLUX_UNITS,
        "long_name": "mass balance",
        "components": " + ".join(present),
    }
    # ``integrate_rate`` hands back a pint quantity when given target units;
    # the series is stored as plain numbers with the units in its attributes.
    cumulative = integrate_rate(regions["mass_balance"], to="Gt").pint.dequantify()
    cumulative = normalize_timeseries(cumulative, reference_date=reference_year)
    regions["cumulative_mass_balance"] = cast(xr.DataArray, cumulative)
    regions["cumulative_mass_balance"].attrs = {
        "units": "Gt",
        "long_name": f"cumulative mass balance since {reference_year}",
    }
    return regions


# --- Observations ----------------------------------------------------------------


#: The Mankoff series kept by default: the cumulative mass balance and its
#: uncertainty, which the overview figure draws.
MANKOFF_CUMULATIVE = ("cumulative_mass_balance", "cumulative_mass_balance_uncertainty")

#: The Mankoff flux series, with their uncertainties, for per-basin flux panels.
MANKOFF_FLUXES = (
    "mass_balance",
    "mass_balance_uncertainty",
    "surface_mass_balance",
    "surface_mass_balance_uncertainty",
    "grounding_line_flux",
    "grounding_line_flux_uncertainty",
)


def load_mankoff(
    url: str,
    reference_year: str = DEFAULT_REFERENCE_YEAR,
    start: str = "1985",
    end: str = "2025",
    variables: Sequence[str] = MANKOFF_CUMULATIVE,
) -> xr.Dataset:
    """
    Load the staged Mankoff mass balance on annual bins, zeroed at the reference year.

    Parameters
    ----------
    url : str
        ``mankoff_greenland_mass_balance.nc`` from ``pism-ismip7-greenland-observations``.
    reference_year : str, optional
        Year the cumulative series are zeroed at.
    start, end : str, optional
        Years kept.
    variables : sequence of str, optional
        Series to keep: :data:`MANKOFF_CUMULATIVE` by default, plus
        :data:`MANKOFF_FLUXES` for the flux panels. Cumulative ones come
        out in Gt and are zeroed at ``reference_year``; fluxes in Gt/yr.

    Returns
    -------
    xarray.Dataset
        The series, regions named like the model's (``GIS_NW``).
    """
    obs = xr.open_dataset(url, decode_timedelta=True, chunks={"time": -1}, **OPEN_KWARGS, **storage_options(url))
    keep = [v for v in variables if v in obs]
    cumulative = [v for v in keep if v.startswith("cumulative")]
    targets = {v: ("Gt" if v in cumulative else FLUX_UNITS) for v in keep}
    obs = obs[keep].pint.quantify().pint.to(targets).pint.dequantify().load()
    for v, units in targets.items():
        obs[v].attrs["units"] = units
    obs = obs.assign_coords(region=np.char.add("GIS_", obs["region"].values.astype(str)))
    obs = obs.sel(time=slice(start, end)).resample(time="YS").mean("time")
    if cumulative:
        obs = cast(xr.Dataset, normalize_timeseries(obs, variables=cumulative, reference_date=reference_year))
    return obs


# --- Figure ----------------------------------------------------------------------


def plot_regions(
    regions: xr.Dataset,
    mankoff: xr.Dataset | None,
    filename: str | Path,
    *,
    sigma: float = 1.0,
    xlim: tuple[str, str] = ("1985", "2100"),
    variable: str = "cumulative_mass_balance",
) -> None:
    """
    One panel per basin: every (GCM, pathway) series over the observed band.

    Parameters
    ----------
    regions : xarray.Dataset
        Output of :func:`compute_regions`.
    mankoff : xarray.Dataset or None
        Output of :func:`load_mankoff`; None draws the model alone.
    filename : str or Path
        Output figure.
    sigma : float, optional
        Half-width of the observed band, in standard deviations.
    xlim : tuple of str, optional
        Years shown.
    variable : str, optional
        Series to plot.
    """
    names = [str(r) for r in regions["region"].values]
    ncols = 4
    nrows = -(-len(names) // ncols)
    rc = dict(rc_params, **{"font.size": 4})
    with mpl.rc_context(rc=rc):
        fig, axs = plt.subplots(nrows, ncols, figsize=(6.4, 1.9 * nrows), sharex=True, squeeze=False)
        for ax in axs.flat[len(names) :]:
            ax.set_visible(False)
        for ax, region in zip(axs.flat, names):
            if mankoff is not None and region in mankoff["region"].values:
                obs = mankoff.sel(region=region)
                ax.fill_between(
                    obs["time"].values,
                    obs["cumulative_mass_balance"] - sigma * obs["cumulative_mass_balance_uncertainty"],
                    obs["cumulative_mass_balance"] + sigma * obs["cumulative_mass_balance_uncertainty"],
                    lw=0,
                    color="0.75",
                    alpha=0.5,
                )
                ax.plot(obs["time"].values, obs["cumulative_mass_balance"], lw=0.75, color="k")
            series = regions[variable].sel(region=region)
            for gcm in series["gcm_id"].values:
                for ssp in series["ssp_id"].values:
                    line = series.sel(gcm_id=gcm, ssp_id=ssp).dropna("time")
                    if line.size:
                        ax.plot(
                            line["time"].values,
                            line,
                            ls=GCM_STYLES.get(str(gcm), "solid"),
                            color=SSP_COLORS.get(str(ssp), "0.3"),
                            lw=0.5,
                            label=f"{gcm} {ssp}",
                        )
            ax.set_xlim(np.datetime64(xlim[0]), np.datetime64(xlim[1]))
            ax.set_title(region)
        axs.flat[0].legend(fontsize=3, frameon=False)
        fig.supylabel(f"{variable.replace('_', ' ')} ({regions[variable].attrs.get('units', '')})", fontsize=5)
        fig.tight_layout()
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=300)
        plt.close(fig)
    logger.info("Wrote %s", filename)


# --- Driver ----------------------------------------------------------------------


def run(
    root: str,
    output_path: str | Path,
    *,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    tree: Sequence[str] = DEFAULT_TREE,
    outline: str | None = None,
    mankoff: str | None = None,
    reference_year: str = DEFAULT_REFERENCE_YEAR,
    sigma: float = 1.0,
    xlim: tuple[str, str] = ("1985", "2100"),
    regions_file: str | Path | None = None,
) -> xr.Dataset:
    """
    Integrate a submission over the basins, save the series and plot them.

    Parameters
    ----------
    root : str
        A run's ``output`` directory, S3 URL or local path.
    output_path : str or Path
        Directory for the figure and the regional series.
    variables : sequence of str, optional
        Fluxes to integrate and sum.
    tree : sequence of str, optional
        Path components from ``root`` to the counter directories.
    outline : str or None, optional
        Basin outline (:func:`resolve_outline`).
    mankoff : str or None, optional
        The staged Mankoff file; ``<root>/observations/`` by default, and
        the figure goes without it when there is none.
    reference_year : str, optional
        Year the cumulative series is zeroed at.
    sigma : float, optional
        Half-width of the observed band.
    xlim : tuple of str, optional
        Years shown.
    regions_file : str or Path or None, optional
        A ``regional_mass_balance.nc`` from an earlier run to plot again
        without touching the ensemble.

    Returns
    -------
    xarray.Dataset
        The regional series (:func:`compute_regions`).
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if regions_file is not None:
        regions = xr.open_dataset(regions_file).load()
        logger.info("Read the regional series from %s", regions_file)
    else:
        ds = open_submission(find_files(root, variables, tree))
        regions = compute_regions(
            ds, resolve_outline(root, outline), variables=variables, reference_year=reference_year
        )
        regions.to_netcdf(output_path / "regional_mass_balance.nc")
        regions[["mass_balance", "cumulative_mass_balance"]].to_dataframe().to_csv(
            output_path / "regional_mass_balance.csv"
        )

    obs_url = mankoff or first_match(root, f"observations/{DEFAULT_MANKOFF}")
    observed = None
    if obs_url is None:
        logger.warning("No %s below %s/observations; plotting the model alone", DEFAULT_MANKOFF, root)
    else:
        try:
            observed = load_mankoff(obs_url, reference_year)
        except (FileNotFoundError, OSError) as err:
            logger.warning("No Mankoff product at %s (%s); plotting the model alone", obs_url, err)
    plot_regions(regions, observed, output_path / "regional_mass_balance.png", sigma=sigma, xlim=xlim)
    return regions


def main(argv: Sequence[str] | None = None) -> int:
    """
    Console entry point.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments without the program name.

    Returns
    -------
    int
        Exit code, ``0`` on success.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Regional mass balance of an ISMIP7 Greenland submission, against Mankoff et al. (2021)."
    parser.add_argument(
        "--root",
        default=None,
        help="A run's output directory, S3 URL or local path; a cloud project's jobs are collected with a "
        "wildcard for the job id, e.g. 's3://pism-cloud-data/ismip7_production/2026_09_core/*/output' (quoted). "
        "Default: s3://BUCKET/NAME/PROJECT/output.",
    )
    parser.add_argument("--bucket", default="pism-cloud-data", help="Bucket the cloud runs write to.")
    parser.add_argument("--name", default="ismip7_production", help="Job name of the cloud runs.")
    parser.add_argument("--project", default="2026_10_ismip7_core_ctrl", help="Project of the cloud runs.")
    parser.add_argument(
        "--variables",
        default=",".join(DEFAULT_VARIABLES),
        help="Comma-separated fluxes to integrate over the basins and sum into the mass balance.",
    )
    parser.add_argument(
        "--tree",
        default="/".join(DEFAULT_TREE),
        help="Path from the output directory to the counter directories (domain/group/model/set).",
    )
    parser.add_argument(
        "--outline",
        default=None,
        help=f"Basin outline: a path or URL, or a file name in <root>/observations. Default: {DEFAULT_OUTLINE}, "
        "falling back to the packaged copy.",
    )
    parser.add_argument(
        "--mankoff", default=None, help=f"The staged Mankoff file. Default: <root>/observations/{DEFAULT_MANKOFF}."
    )
    parser.add_argument(
        "--reference-year", default=DEFAULT_REFERENCE_YEAR, help="Year the cumulative series is zeroed at."
    )
    parser.add_argument(
        "--sigma", type=float, default=1.0, help="Half-width of the observed band, standard deviations."
    )
    parser.add_argument("--xlim", default="1985,2100", help="Years shown, comma-separated.")
    parser.add_argument(
        "--regions-file",
        default=None,
        help="Plot a regional_mass_balance.nc from an earlier run instead of recomputing.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Start a local Dask cluster with this many workers; the default runs on the threaded scheduler.",
    )
    parser.add_argument("OUTPUT_PATH", nargs=1, help="Directory for the figure and the regional series.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_path = Path(args.OUTPUT_PATH[0])
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(output_path / "mass_balance.log")
    root = args.root or f"s3://{args.bucket}/{args.name}/{args.project}/output"
    xlim = args.xlim.split(",")
    if len(xlim) != 2:
        parser.error("--xlim takes two years, e.g. 1985,2100")

    client = None
    if args.n_workers:
        from dask.distributed import Client  # pylint: disable=import-outside-toplevel

        client = Client(n_workers=args.n_workers)
        logger.info("Dask dashboard at %s", client.dashboard_link)
    try:
        run(
            root,
            output_path,
            variables=[v.strip() for v in args.variables.split(",") if v.strip()],
            tree=[p for p in args.tree.split("/") if p],
            outline=args.outline,
            mankoff=args.mankoff,
            reference_year=args.reference_year,
            sigma=args.sigma,
            xlim=(xlim[0].strip(), xlim[1].strip()),
            regions_file=args.regions_file,
        )
    finally:
        if client is not None:
            client.close()
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(main())
