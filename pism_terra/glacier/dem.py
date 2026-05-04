# Copyright (C) 2025 Andy Aschwanden
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

# mypy: disable-error-code="call-overload"
# pylint: disable=too-many-positional-arguments

"""
Prepare DEM.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Literal, Sequence

import rasterio
import rioxarray as rxr
import xarray as xr
from dem_stitcher import stitch_dem
from pyproj import Transformer

from pism_terra.glacier.ice_thickness import get_ice_thickness
from pism_terra.glacier.observations import glacier_velocities_from_grid
from pism_terra.workflow import check_rio

xr.set_options(keep_attrs=True)


def get_surface_dem_by_bounds(
    bounds: Sequence[float],
    dataset: Literal["glo_30", "arcticdem"],
    path: str | Path = "input",
    force_overwrite: bool = False,
) -> Path:
    """
    Create (or reuse) a surface DEM GeoTIFF for a geographic bounding box.

    Mosaics/exports a DEM over ``bounds`` via :func:`stitch_dem`, writes a
    single-band GeoTIFF with appropriate tags, and returns the output path.
    If a file with the expected name already exists under ``path`` and opens
    successfully via :func:`check_rio`, that file is reused unless
    ``force_overwrite=True``.

    Parameters
    ----------
    bounds : tuple of float
        Geographic bounding box ``(minx, miny, maxx, maxy)`` in degrees (WGS84).
    dataset : str, default ``"glo_30"``
        DEM source identifier recognized by :func:`stitch_dem`
        (e.g., ``"glo_30"``, ``"arcticdem"``).
    path : str or pathlib.Path, default ``"input"``
        Output directory for the GeoTIFF. Created if it does not exist.
    force_overwrite : bool, default ``False``
        If ``True``, skip cache reuse and regenerate the DEM even if a readable
        file already exists at the target location.

    Returns
    -------
    pathlib.Path
        Path to the GeoTIFF file containing the DEM.

    Raises
    ------
    FileNotFoundError
        If required DEM tiles cannot be fetched/assembled.
    ValueError
        If the stitched DEM/profile is invalid or incompatible with GeoTIFF.
    rasterio.errors.RasterioIOError
        On errors writing the GeoTIFF.
    Exception
        Any other error propagated by :func:`stitch_dem` or I/O routines.

    See Also
    --------
    stitch_dem
        Assemble a DEM mosaic and return the array and raster profile.
    check_rio
        Lightweight validity check for raster files readable by rioxarray.

    Notes
    -----
    - Output is a **single-band** GeoTIFF tagged with ``AREA_OR_POINT="Point"``.
    - Heights follow the profile from :func:`stitch_dem`
      (here ``dst_ellipsoidal_height=False`` typically implies orthometric heights).
    - The file is **not** deleted automatically; callers manage lifecycle.

    Examples
    --------
    >>> tif = get_surface_dem_by_bounds((214.1, 59.0, 219.7, 63.9),
    ...                                 dem_name="glo_30", path="input")
    >>> tif.exists()
    True
    """
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    geo_file = out_dir / f"{dataset}.tif"
    # Reuse if present and readable
    if (not check_rio(geo_file)) or force_overwrite:
        X, p = stitch_dem(
            bounds,
            dem_name=dataset,
            dst_ellipsoidal_height=False,
            dst_area_or_point="Point",
        )

        # Ensure the profile matches what we're writing
        p = p.copy()
        p.update(
            {
                "driver": "GTiff",
                "count": 1,
                "dtype": X.dtype,  # e.g., 'float32'
                "BIGTIFF": "YES",  # allow files >4 GB
            }
        )

        with rasterio.open(geo_file, "w", **p) as src:
            src.write(X, 1)  # write band 1
            src.update_tags(AREA_OR_POINT="Point")

    return geo_file


def prepare_surface(
    target_grid: xr.Dataset,
    dataset: Literal["glo_30", "arcticdem"],
    path: str | Path = "input_files",
    **kwargs,
) -> xr.DataArray:
    """
    Prepare a surface DEM aligned to a target grid.

    Workflow:
    (1) Derive a geographic (WGS84) bounding box from ``target_grid``.
    (2) Download/mosaic a DEM over that bounding box.
    (3) Reproject/resample it to match ``target_grid`` (CRS, extent, resolution).
    (4) Write the result to ``surface.nc`` under ``path`` and return it.

    Parameters
    ----------
    target_grid : xarray.Dataset
        Target grid dataset providing the destination CRS (via ``spatial_ref``)
        and the cell-edge bounds (``x_bnds``/``y_bnds``) used to derive both the
        geographic query bounds and the destination grid for reprojection.
    dataset : str, default ``"glo_30"``
        DEM source identifier passed to :func:`get_surface_dem_by_bounds`.
    path : str or pathlib.Path, default ``"input_files"``
        Output directory for generated files. Created if missing.
    **kwargs
        Additional keyword arguments forwarded to :func:`get_surface_dem_by_bounds`
        (e.g., ``force_overwrite=True``). These do not affect the reprojection
        step directly.

    Returns
    -------
    xarray.DataArray
        Surface elevation (meters) on ``target_grid``, with
        ``standard_name="land_ice_elevation"``. Also written to
        ``Path(path) / "surface.nc"``.

    Raises
    ------
    FileNotFoundError
        If DEM retrieval fails to produce an output file.
    ValueError
        On invalid bounds/CRS or reprojection issues.
    Exception
        Any other I/O or decoding errors propagated by helper functions.

    Notes
    -----
    - The surface variable is named ``"surface"`` with
      ``standard_name="land_ice_elevation"`` and ``units="m"``.
    - After reprojection to the target grid, missing values are filled with 0.
      Adjust if downstream tooling expects masked NaNs instead.
    """

    bounds = [
        target_grid.x_bnds.values[0][0],
        target_grid.y_bnds.values[0][0],
        target_grid.x_bnds.values[-1][-1],
        target_grid.y_bnds.values[-1][-1],
    ]
    dst_crs = target_grid.spatial_ref.attrs["crs_wkt"]
    t = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)
    geo_bounds = t.transform_bounds(*bounds)

    geo_file = get_surface_dem_by_bounds(geo_bounds, dataset=dataset, path=path, **kwargs)
    surface = rxr.open_rasterio(geo_file).squeeze().drop_vars("band", errors="ignore")
    surface.name = "surface"
    surface.attrs.update({"standard_name": "land_ice_elevation", "units": "m"})

    surface_reprojected = surface.rio.reproject_match(target_grid).fillna(0)

    surface_file = Path(path) / "surface.nc"
    surface_reprojected.to_netcdf(surface_file)

    return surface_reprojected


def boot_file_from_grid(
    target_grid: xr.Dataset,
    rgi_id: str,
    geometries: collections.abc.Iterable,
    dem_dataset: Literal["glo_30", "arcticdem"],
    ice_thickness_dataset: Literal["maffezzoli", "millan"],
    velocity_dataset: Literal["none", "its_live"] | None,
    path: str | Path = "input_files",
    **kwargs,
) -> xr.Dataset:
    """
    Build a glacier “boot” dataset (surface, thickness, bed, masks, aux vars) from an RGI ID.

    Steps:
    (1) Locate the glacier geometry by ``rgi_id`` (GeoDataFrame or file-based RGI).
    (2) Buffer the glacier polygon in its native projected CRS (meters).
    (3) Mosaic/reproject a DEM over the buffered extent at ``resolution``.
    (4) Interpolate glacier ice thickness onto the target grid.
    (5) Derive bed elevation and boolean masks from DEM clipping.
    (6) Optionally fetch observed velocities and create a simple ``tillwat`` field.
    Returns a regular 2-D xarray Dataset in the glacier’s projected CRS.

    Parameters
    ----------
    target_grid : xarray.Dataset
        Target grid dataset (with ``x``/``y`` coords, ``x_bnds``/``y_bnds`` cell-edge
        bounds, and ``spatial_ref`` carrying the destination CRS) onto which all
        derived fields are aligned.
    rgi_id : str
        Glacier identifier, e.g., ``"RGI2000-v7.0-C-06-00014"``.
    geometries : iterable of shapely geometries
        Glacier outline(s) in ``target_grid``'s CRS, used for clipping the DEM
        and constructing masks.
    dem_dataset : str, default ``"glo_30"``
        DEM source for surface preparation (e.g., ``"glo_30"``, ``"arcticdem"``).
    ice_thickness_dataset : str, default ``"maffezzoli"``
        Source for ice thickness (e.g., ``"glo_30"``, ``"arcticdem"``).
    velocity_dataset : str, default ``"its_live"``
        Source for velocities (e.g., ``"none"``, ``"its_live"``).
    path : str or pathlib.Path, default ``"input_files"``
        Working directory used by helper routines to cache/write intermediate rasters/grids.
    **kwargs
        Forwarded to :func:`prepare_surface` (e.g., ``force_overwrite=True``) and any
        downstream helpers it calls. Does not alter variable naming/semantics.

    Returns
    -------
    xarray.Dataset
        Regular 2-D dataset (dims typically ``y``, ``x``) in the glacier CRS with at least:
        - ``surface`` : float32, m — DEM surface elevation.
        - ``thickness`` : float32, m — ice thickness on the target grid.
        - ``bed`` : float32, m — bedrock elevation (``surface - thickness``).
        - ``land_ice_area_fraction_retreat`` : bool — 1 where DEM is ice-free after clipping.
        - ``ftt_mask`` : bool — complementary outside-footprint mask (1 outside).
        - ``tillwat`` : float32, m — simple basal water proxy (here ``0`` or ``2`` m based on speed).
        - ``v`` and related velocity fields (from :func:`glacier_velocities_from_rgi_id`), reprojected
          to the surface grid, if available.

        CRS is recorded via the rioxarray accessor (``.rio.crs``); spatial dims are set with
        ``.rio.set_spatial_dims(x_dim="x", y_dim="y")``.

    Raises
    ------
    FileNotFoundError
        If the supplied RGI path does not exist or required inputs are missing.
    ValueError
        If ``rgi_id`` is not found or CRS information is invalid.
    Exception
        Propagated I/O/projection/decoding errors from DEM/thickness/velocity preparation.

    See Also
    --------
    get_glacier_from_rgi_id
        Extract a glacier feature by RGI ID from an RGI table.
    prepare_surface
        Mosaic/reproject a DEM over a geographic bounding box and build the target grid.
    get_ice_thickness
        Interpolate glacier ice thickness onto a target grid.
    glacier_velocities_from_rgi_id
        Retrieve observed surface velocities for the glacier domain.

    Notes
    -----
    - Buffering is done in the glacier’s projected CRS (meters). The buffered geometry is
      converted to WGS84 only to derive geographic bounds for DEM staging.
    - ``land_ice_area_fraction_retreat`` and ``ftt_mask`` are derived from DEM clipping and
      are intentionally simple; refine as needed for your application.
    - ``tillwat`` is a coarse proxy here: set to ``2 m`` where reprojected speed ``v >= 100 m/yr``,
      else ``0 m``. Adjust the threshold and values per your physics.
    - The function returns an **in-memory** dataset; it does not write to disk.
    """

    print("")
    print("Generate DEM")
    print("-" * 120)

    dst_crs = target_grid.spatial_ref.attrs["crs_wkt"]

    surface = prepare_surface(target_grid, dataset=dem_dataset, path=path)
    surface = surface.where(surface > 0.0, 0.0)

    ice_thickness = get_ice_thickness(
        rgi_id, dataset=ice_thickness_dataset, path=path, target_grid=target_grid, target_crs=dst_crs, **kwargs
    )
    ice_thickness = ice_thickness.rio.reproject_match(target_grid)
    ice_thickness = ice_thickness.rio.clip(geometries, drop=False).fillna(0)
    ice_thickness = ice_thickness.where(ice_thickness > 0.0, 0.0)

    bed = surface - ice_thickness
    bed.name = "bed"
    bed.attrs.update({"standard_name": "bedrock_altitude", "units": "m"})

    liafr = surface.rio.clip(geometries, drop=False)
    liafr = xr.where(liafr.isnull(), 0, 1)
    liafr.name = "land_ice_area_fraction_retreat"
    liafr.attrs.update({"units": "1"})
    liafr = liafr.astype("bool")

    # ftt_mask = surface.rio.clip(glacier_projected.geometry, drop=False)
    ftt_mask = xr.ones_like(liafr)
    ftt_mask.name = "ftt_mask"
    ftt_mask.attrs.update({"units": "1"})
    ftt_mask = ftt_mask.astype("bool")

    tillwat = xr.zeros_like(bed)
    tillwat.name = "tillwat"
    tillwat.attrs.update({"units": "m"})

    ds = xr.merge([bed, surface, ice_thickness, liafr, ftt_mask, tillwat], compat="no_conflicts")
    if velocity_dataset not in ("none", None):
        v_filename = path / Path(f"obs_{rgi_id}.nc")
        v = glacier_velocities_from_grid(target_grid, geometries, path=v_filename)
        _v = v["v"].fillna(0)
        ds["tillwat"] = xr.where(_v < 100, 0, xr.where(_v > 250, 2, 1 + (_v - 100) / (250 - 100)))
        ds["tillwat"].attrs.update({"units": "m"})
        ds["v"] = v["v"]

    ds = ds.fillna(0)
    if hasattr(ds["spatial_ref"], "GeoTransform"):
        del ds["spatial_ref"].attrs["GeoTransform"]
    for name in ("x", "y", "thickness", "bed", "surface", "tillwat", "ftt_mask", "land_ice_area_fraction_retreat"):
        if name in ds:
            ds[name].encoding.update({"_FillValue": None})
    return ds
