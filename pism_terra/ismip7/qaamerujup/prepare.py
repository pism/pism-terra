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
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Prepare the Qaamerujup regional domain.

Qaamerujup is run with the ISMIP7 Greenland machinery
(``pism-ismip7-greenland-run-*``) on a small domain around the glacier,
starting from its 1931 geometry. This module builds the three inputs that
differ from the ice-sheet runs; the forcing, heat flux and regrid state are
the Greenland-wide files, which PISM interpolates onto the domain.

- **Grid** (``pism_qaamerujup_grid.nc``): the bounding box of the domain
  polygon, grown to a whole number of cells at every resolution of the setup.
  Only the extent is stored; the run's ``grid.resolution`` sets the spacing.
- **Boot file** (``boot_1931_qaamerujup.nc``), on the 32 m ArcticDEM grid: the
  surface is the reconstructed 1931 DEM inside the buffered 1931 outline and
  ArcticDEM outside it, with the gaps between the two filled by a Laplace
  solve. The bed is the ISMIP7 observation dataset's (BedMachine). Inside the
  1931 outline the thickness is surface minus bed; elsewhere it follows
  BedMachine's mask (grounded ice surface minus bed, floating ice from
  flotation, none on land and in the ocean), so differences between the two
  DEMs never become ice on land. The exclusion polygon becomes deep ocean.
  ``ftt_mask`` is 1 outside the buffered outline, where force-to-thickness
  holds the geometry to this file.
- **Observation file** (``obs_1931_qaamerujup.nc``): the velocity mosaic and
  the inversion masks on the same grid, built like the ice-sheet one
  (:func:`pism_terra.ismip7.greenland.forcing.inverse_observations`).

The outputs go into ``<OUTPUT_PATH>/input``, to be uploaded next to the
ISMIP7 Greenland inputs.
"""

import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Sequence

import fsspec
import geopandas as gpd
import numpy as np
import rioxarray  # pylint: disable=unused-import
import toml
import xarray as xr
from pyfiglet import Figlet
from rasterio.enums import Resampling

import pism_terra.interpolation  # pylint: disable=unused-import  # registers the .utils accessor
from pism_terra.aws import download_from_s3
from pism_terra.domain import create_domain, get_bounds
from pism_terra.ismip7.greenland.forcing import (
    basin_mask,
    inverse_observations,
    write_inverse_observations,
)
from pism_terra.log import setup_logging
from pism_terra.prepare_select import add_include_argument, select_datasets
from pism_terra.workflow import (
    check_xr_fully,
    check_xr_lazy,
    compressed_encoding,
    drop_geotransform_attr,
    stamp_grid_mapping,
)

xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)

#: Datasets this command can prepare, in execution order.
QAAMERUJUP_DATASETS = ["grid", "boot", "observations"]

#: Densities PISM uses for flotation (``constants.ice.density``,
#: ``constants.sea_water.density``).
RHO_ICE = 910.0
RHO_SEA_WATER = 1028.0

#: Bed elevation (m) of the excluded area: deep enough that no ice grounds.
EXCLUDED_BED = -2000.0


def product_names(config: dict) -> dict[str, str]:
    """
    Name the files this setup produces.

    Parameters
    ----------
    config : dict
        Setup configuration (``name`` and ``year``).

    Returns
    -------
    dict of str to str
        ``grid_file``, ``boot_file`` and ``obs_file``, as the campaign config
        names them.
    """
    name, year = config["name"], config["year"]
    return {
        "grid_file": f"pism_{name}_grid.nc",
        "boot_file": f"boot_{year}_{name}.nc",
        "obs_file": f"obs_{year}_{name}.nc",
    }


def fetch(name: str, config: dict, cache_path: Path, data_path: Path | None = None) -> Path:
    """
    Find one raw input of the domain, downloading it once.

    Parameters
    ----------
    name : str
        File name.
    config : dict
        Setup configuration; the file lives in
        ``s3://{bucket}/{source_prefix}/``.
    cache_path : pathlib.Path
        Where downloads are kept between runs.
    data_path : pathlib.Path or None, optional
        Local directory with the raw inputs; used instead of the bucket when
        it holds the file.

    Returns
    -------
    pathlib.Path
        The local file.
    """
    if data_path is not None and (data_path / name).exists():
        return data_path / name
    local = cache_path / name
    if not local.exists():
        download_from_s3(f"s3://{config['bucket']}/{config['source_prefix']}/{name}", local)
    return local


def read_outline(path: Path | str, crs: str) -> gpd.GeoDataFrame:
    """
    Read an outline in ``crs``, mending a geographic label on projected coordinates.

    Parameters
    ----------
    path : pathlib.Path or str
        Vector file.
    crs : str
        CRS of the domain.

    Returns
    -------
    geopandas.GeoDataFrame
        The outline in ``crs``.
    """
    gdf = gpd.read_file(path)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    labelled_geographic = gdf.crs is None or gdf.crs.is_geographic
    if labelled_geographic and (min(xmin, ymin) < -180 or max(xmax, ymax) > 180):
        logger.warning("%s is labelled %s but its coordinates are projected; reading it in %s", path, gdf.crs, crs)
        gdf = gdf.set_crs(crs, allow_override=True)
    return gdf.to_crs(crs)


def domain_bounds(
    outline: gpd.GeoDataFrame, base_resolution: int, multipliers: Sequence[int]
) -> tuple[list[float], list[float]]:
    """
    Bounding box of an outline that every resolution of the setup tiles.

    Parameters
    ----------
    outline : geopandas.GeoDataFrame
        Domain polygon(s), in a projected CRS.
    base_resolution : int
        Finest resolution (m).
    multipliers : sequence of int
        The resolutions are ``base_resolution * multipliers``.

    Returns
    -------
    tuple of list of float
        ``([x_min, x_max], [y_min, y_max])``, centred on the outline's box and
        a whole multiple of the least common multiple of the resolutions wide.
    """
    xmin, ymin, xmax, ymax = outline.total_bounds
    half = base_resolution / 2
    cells = xr.Dataset(
        coords={
            "x": np.arange(xmin + half, xmax, base_resolution),
            "y": np.arange(ymin + half, ymax, base_resolution),
        }
    )
    x_bnds, y_bnds = get_bounds(cells, base_resolution=base_resolution, multipliers=list(multipliers))
    return [float(v) for v in x_bnds], [float(v) for v in y_bnds]


def inside(template: xr.DataArray, outline: gpd.GeoDataFrame) -> xr.DataArray:
    """
    Mark the cells of ``template`` that ``outline`` touches.

    Parameters
    ----------
    template : xarray.DataArray
        Grid with x/y coordinates and a CRS.
    outline : geopandas.GeoDataFrame
        Polygons in the grid's CRS.

    Returns
    -------
    xarray.DataArray
        Boolean mask, True inside.
    """
    ones = xr.ones_like(template, dtype="float32").rio.write_crs(template.rio.crs)
    try:
        return ones.rio.clip(outline.geometry, drop=False, all_touched=True).notnull()
    except Exception:  # pylint: disable=broad-exception-caught  # no overlap with the grid
        return xr.zeros_like(template, dtype=bool)


def read_raster(path: Path | str, bounds: tuple[float, float, float, float] | None = None) -> xr.DataArray:
    """
    Read a single-band GeoTIFF with its no-data cells as NaN.

    Parameters
    ----------
    path : pathlib.Path or str
        GeoTIFF.
    bounds : tuple of float or None, optional
        ``(xmin, ymin, xmax, ymax)`` window to keep.

    Returns
    -------
    xarray.DataArray
        The band, float.
    """
    da = rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True)
    if bounds is not None:
        da = da.rio.clip_box(*bounds)
    return da.astype("float64").load()


def merge_surface(reference: xr.DataArray, reconstruction: xr.DataArray, buffered: gpd.GeoDataFrame) -> xr.DataArray:
    """
    The reconstructed surface inside the buffered outline, the reference outside.

    Parameters
    ----------
    reference : xarray.DataArray
        Present-day DEM; its grid is the result's.
    reconstruction : xarray.DataArray
        Reconstructed DEM, on any grid; values at or below 0 are no data.
    buffered : geopandas.GeoDataFrame
        Where the reconstruction replaces the reference.

    Returns
    -------
    xarray.DataArray
        Surface elevation (m). Cells inside the outline the reconstruction
        does not cover are filled by a Laplace solve between the two DEMs,
        so the seam has no step.
    """
    reference = reference.where(reference > 0.1, 0.0)
    reconstruction = reconstruction.rio.reproject_match(reference, resampling=Resampling.bilinear)
    reconstruction = reconstruction.where(reconstruction > 0)
    in_buffer = inside(reference, buffered)
    surface = xr.where(in_buffer, reconstruction, reference)
    n_gaps = int(surface.isnull().sum())
    if n_gaps:
        logger.info("Filling %d cells inside the outline the reconstruction does not cover", n_gaps)
        surface = surface.utils.fillna()
    return surface.rio.write_crs(reference.rio.crs)


def read_observations(url: str, bounds: tuple[float, float, float, float], pad: float = 1000.0) -> xr.Dataset:
    """
    Read the window of the ISMIP7 observation dataset around the domain.

    Parameters
    ----------
    url : str
        ``s3://`` URL (read anonymously, only the window is transferred) or path.
    bounds : tuple of float
        ``(xmin, ymin, xmax, ymax)``.
    pad : float, default 1000
        Extra margin (m), so interpolation has neighbours at the edges.

    Returns
    -------
    xarray.Dataset
        ``bed``, ``thickness``, ``mask``, ``icemask_promice``, ``vx_mosaic``
        and ``vy_mosaic``, loaded, with ascending y.
    """
    xmin, ymin, xmax, ymax = bounds
    names = ["bed", "thickness", "mask", "icemask_promice", "vx_mosaic", "vy_mosaic"]
    options = {"anon": True} if str(url).startswith("s3://") else {}
    fs, path = fsspec.core.url_to_fs(str(url), **options)
    with fs.open(path, "rb", block_size=8 * 2**20) as handle:
        with xr.open_dataset(handle, engine="h5netcdf") as ds:
            ds = ds.sortby("y")
            window = ds[names].sel(x=slice(xmin - pad, xmax + pad), y=slice(ymin - pad, ymax + pad)).load()
    return window.drop_vars(["mapping", "spatial_ref"], errors="ignore")


def on_grid(obs: xr.Dataset, grid: xr.DataArray) -> xr.Dataset:
    """
    Interpolate the observations onto a grid: continuous fields bilinearly, masks by nearest cell.

    Parameters
    ----------
    obs : xarray.Dataset
        Output of :func:`read_observations`.
    grid : xarray.DataArray
        Target grid (x/y coordinates).

    Returns
    -------
    xarray.Dataset
        The observations on ``grid``.
    """
    coords = {"x": grid["x"], "y": grid["y"]}
    continuous = obs[["bed", "thickness", "vx_mosaic", "vy_mosaic"]].astype("float64").interp(coords).astype("float32")
    categorical = obs[["mask", "icemask_promice"]].interp(coords, method="nearest")
    return xr.merge([continuous, categorical])


def boot_geometry(
    surface: xr.DataArray,
    obs: xr.Dataset,
    glacier: xr.DataArray,
    buffer: xr.DataArray,
    excluded: xr.DataArray,
) -> xr.Dataset:
    """
    Bed, surface and thickness of the boot file, with its masks.

    Parameters
    ----------
    surface : xarray.DataArray
        Merged surface (:func:`merge_surface`).
    obs : xarray.Dataset
        Observations on the same grid (:func:`on_grid`).
    glacier : xarray.DataArray
        True inside the reconstructed glacier outline.
    buffer : xarray.DataArray
        True inside the buffered outline.
    excluded : xarray.DataArray
        True in the area turned into deep ocean.

    Returns
    -------
    xarray.Dataset
        ``bed``, ``surface``, ``thickness``, ``ftt_mask`` and
        ``land_ice_area_fraction_retreat``.
    """
    bed = obs["bed"]
    mask = obs["mask"]
    alpha = 1.0 - RHO_ICE / RHO_SEA_WATER
    grounded = (surface - bed).clip(min=0)
    # Freeboard alpha * H, capped by the water depth a column of that
    # thickness can float in (see prepare_observations for Greenland).
    floating = np.minimum(surface / alpha, (-bed * RHO_SEA_WATER / RHO_ICE).clip(min=0))
    present = xr.where(mask == 2, grounded, xr.where(mask == 3, floating, 0.0))
    thickness = xr.where(glacier, xr.where(surface > 0.1, grounded, 0.0), present)

    bed = bed.where(~excluded, EXCLUDED_BED)
    surface = surface.where(~excluded, 0.0)
    thickness = thickness.where(~excluded, 0.0)
    mask = mask.where(~excluded, 0)

    boot = xr.Dataset(
        {
            "bed": bed.astype("float32"),
            "surface": surface.astype("float32"),
            "thickness": thickness.astype("float32"),
            # Force-to-thickness holds the geometry outside the buffered outline.
            "ftt_mask": (~buffer).astype("int8"),
            "land_ice_area_fraction_retreat": (mask != 0).astype("int8"),
        }
    ).fillna(0)
    boot["bed"].attrs = {"standard_name": "bedrock_altitude", "long_name": "bed elevation", "units": "m"}
    boot["surface"].attrs = {"standard_name": "surface_altitude", "long_name": "ice surface elevation", "units": "m"}
    boot["thickness"].attrs = {"standard_name": "land_ice_thickness", "long_name": "ice thickness", "units": "m"}
    boot["ftt_mask"].attrs = {"units": "1", "long_name": "force-to-thickness mask (1 = held to the boot geometry)"}
    boot["land_ice_area_fraction_retreat"].attrs = {"units": "1"}
    return boot


def with_grid_mapping(ds: xr.Dataset, crs: str) -> xr.Dataset:
    """
    Give a dataset CF coordinates and one ``mapping`` grid-mapping variable.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset on x/y.
    crs : str
        Its CRS.

    Returns
    -------
    xarray.Dataset
        The dataset, ready to write without fill values.
    """
    ds = ds.drop_vars(["spatial_ref", "mapping", "crs"], errors="ignore")
    ds = ds.rio.write_crs(crs, grid_mapping_name="mapping").rio.write_coordinate_system()
    for name, axis in (("x", "X"), ("y", "Y")):
        ds[name].attrs.update(
            {
                "standard_name": f"projection_{name}_coordinate",
                "long_name": f"{name} coordinate of projection",
                "units": "m",
                "axis": axis,
            }
        )
    for name in ds.data_vars:
        ds[name].attrs.pop("grid_mapping", None)
        ds[name].attrs.pop("coordinates", None)
    drop_geotransform_attr(ds)
    ds = stamp_grid_mapping(ds, name="mapping")
    for name in list(ds.data_vars) + list(ds.coords):
        ds[name].encoding["_FillValue"] = None
    return ds


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Prepare the Qaamerujup grid, boot and observation files.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments without the program name; ``None`` reads
        ``sys.argv``. Pass ``[]``-style lists from notebooks.

    Returns
    -------
    dict of str to Any
        ``config`` and the written ``grid_file``, ``boot_file`` and
        ``obs_file`` (``None`` for steps not run).
    """
    parser = ArgumentParser(description="Prepare the Qaamerujup regional domain for pism-ismip7-greenland-run.")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Local directory with the raw inputs (DEMs and outlines); files not found there are "
        "downloaded from s3://{bucket}/{source_prefix}/.",
    )
    parser.add_argument(
        "--obs-file",
        default=None,
        help="Local copy of GreenlandObsISMIP7-v1.3.nc; by default only the domain's window is read from obs_url.",
    )
    parser.add_argument("--force-overwrite", action="store_true", help="Download the raw inputs again.")
    add_include_argument(parser, QAAMERUJUP_DATASETS)
    parser.add_argument("CONFIG_FILE", help="Setup TOML, e.g. pism_terra/config/setup_qaamerujup.toml.")
    parser.add_argument("OUTPUT_PATH", help="Output directory; the products go into OUTPUT_PATH/input.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_path = Path(args.OUTPUT_PATH)
    input_path = output_path / "input"
    cache_path = output_path / "source"
    for p in (input_path, cache_path):
        p.mkdir(parents=True, exist_ok=True)
    if args.force_overwrite:
        for f in cache_path.iterdir():
            f.unlink()
    setup_logging(output_path / "prepare.log")
    selected = select_datasets(args.include, QAAMERUJUP_DATASETS)
    data_path = Path(args.data_path) if args.data_path else None

    logger.info("=" * 120)
    logger.info("\n%s", Figlet(font="standard").renderText("pism-terra"))
    logger.info("=" * 120)

    config = toml.loads(Path(args.CONFIG_FILE).read_text("utf-8"))
    logger.info("Preparing the %s domain (%s)", config["name"], config["year"])
    names = product_names(config)
    domain, dem = config["domain"], config["dem"]
    crs = domain["crs"]

    def outline(key: str) -> gpd.GeoDataFrame:
        """
        Read one of the setup's outlines in the domain CRS.

        Parameters
        ----------
        key : str
            File name of the outline.

        Returns
        -------
        geopandas.GeoDataFrame
            The outline.
        """
        return read_outline(fetch(key, config, cache_path, data_path), crs)

    x_bnds, y_bnds = domain_bounds(outline(domain["file"]), domain["base_resolution"], domain["multipliers"])
    resolutions = [domain["base_resolution"] * m for m in domain["multipliers"]]
    logger.info(
        "Domain x %s, y %s: %.0f m x %.0f m, tiled by %s m",
        x_bnds,
        y_bnds,
        x_bnds[1] - x_bnds[0],
        y_bnds[1] - y_bnds[0],
        resolutions,
    )

    result: dict[str, Any] = {"config": config, "grid_file": None, "boot_file": None, "obs_file": None}

    if "grid" in selected:
        grid_file = input_path / names["grid_file"]
        grid = create_domain(x_bnds, y_bnds, crs=crs)
        grid.attrs.update({"domain": config["name"]})
        grid.to_netcdf(grid_file)
        check_xr_fully(grid_file)
        logger.info("Grid: %s", grid_file)
        result["grid_file"] = grid_file

    margin = float(domain["margin"])
    bounds = (x_bnds[0] - margin, y_bnds[0] - margin, x_bnds[1] + margin, y_bnds[1] + margin)
    boot_file = input_path / names["boot_file"]
    boot: xr.Dataset | None = None
    obs: xr.Dataset | None = None
    if {"boot", "observations"} & set(selected):
        # Raster operations keep the GeoTIFF's descending y; the files are
        # written with ascending y.
        reference = read_raster(fetch(dem["reference"], config, cache_path, data_path), bounds)
        obs_source = args.obs_file or config["obs_url"]
        logger.info("Reading the observations around the domain from %s", obs_source)
        obs = on_grid(read_observations(obs_source, bounds), reference)

    if "boot" in selected:
        assert obs is not None
        reconstruction = read_raster(fetch(dem["reconstruction"], config, cache_path, data_path))
        buffered = outline(dem["buffered_outline"])
        surface = merge_surface(reference, reconstruction, buffered)
        boot = boot_geometry(
            surface,
            obs,
            glacier=inside(reference, outline(dem["outline"])),
            buffer=inside(reference, buffered),
            excluded=inside(reference, outline(dem["exclude"])),
        )
        boot.attrs.update({"title": f"{config['name']} {config['year']} boot file", "Conventions": "CF-1.8"})
        boot = with_grid_mapping(boot.sortby("y"), crs)
        boot.to_netcdf(boot_file, encoding=compressed_encoding(boot), engine="h5netcdf")
        check_xr_lazy(boot_file)
        logger.info("Boot file: %s", boot_file)
        result["boot_file"] = boot_file

    if "observations" in selected:
        assert obs is not None
        if boot is None:
            boot = xr.open_dataset(boot_file).load()
        obs = obs.sortby("y")
        vel = inverse_observations(
            obs["vx_mosaic"],
            obs["vy_mosaic"],
            boot["bed"],
            boot["thickness"],
            obs["icemask_promice"] > 0.5,
            basin_mask(boot),
        )
        obs_file = write_inverse_observations(vel, input_path / names["obs_file"])
        check_xr_lazy(obs_file)
        logger.info("Observation file: %s", obs_file)
        result["obs_file"] = obs_file

    logger.info("-" * 120)
    logger.info("Now run")
    logger.info(
        "aws s3 sync %s s3://%s/%s/%s", input_path.resolve(), config["bucket"], config["prefix"], config["version"]
    )
    logger.info("-" * 120)
    return result


def cli(argv: Sequence[str] | None = None) -> int:
    """
    Console entry point (``pism-qaamerujup-prepare``).

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments without the program name; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    main(argv=argv)
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(cli())
