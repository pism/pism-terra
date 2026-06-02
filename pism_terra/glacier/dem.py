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

import numpy as np
import rioxarray as rxr  # noqa: F401  -- registers the .rio accessor
import xarray as xr
from dem_stitcher import stitch_dem
from pyproj import Transformer
from rasterio.enums import Resampling

from pism_terra.glacier.ice_thickness import get_ice_thickness
from pism_terra.glacier.observations import (
    bathymetry_from_grid,
    glacier_velocities_from_grid,
)
from pism_terra.workflow import check_xr_lazy

xr.set_options(keep_attrs=True)


def get_surface_dem_by_bounds(
    bounds: Sequence[float],
    dataset: Literal["glo_30", "arcticdem"],
    path: str | Path = "input",
    force_overwrite: bool = False,
) -> Path:
    """
    Create (or reuse) a surface DEM NetCDF for a geographic bounding box.

    Mosaics/exports a DEM over ``bounds`` via :func:`stitch_dem` and writes the
    result as a CF-compliant NetCDF (``netcdf4`` engine) with the CRS encoded
    via rioxarray's ``grid_mapping``. If a file with the expected name already
    exists under ``path`` and opens successfully via :func:`check_xr_lazy`,
    that file is reused unless ``force_overwrite=True``.

    Parameters
    ----------
    bounds : tuple of float
        Geographic bounding box ``(minx, miny, maxx, maxy)`` in degrees (WGS84).
    dataset : {"glo_30", "arcticdem"}
        DEM source identifier recognized by :func:`stitch_dem`.
    path : str or pathlib.Path, default ``"input"``
        Output directory for the NetCDF cache. Created if it does not exist.
    force_overwrite : bool, default ``False``
        If ``True``, regenerate the DEM even if a readable cache already exists.

    Returns
    -------
    pathlib.Path
        Path to the NetCDF cache containing the DEM (``<dataset>.nc`` under
        ``path``).
    """
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    geo_file = out_dir / f"{dataset}.nc"

    if check_xr_lazy(geo_file, verbose=False) and not force_overwrite:
        return geo_file

    X, p = stitch_dem(
        bounds,
        dem_name=dataset,
        dst_ellipsoidal_height=False,
        dst_area_or_point="Point",
    )

    transform = p["transform"]
    height, width = X.shape
    # Pixel-center coordinates derived from the affine transform.
    xs = transform.c + transform.a * (np.arange(width) + 0.5)
    ys = transform.f + transform.e * (np.arange(height) + 0.5)

    da = xr.DataArray(
        np.asarray(X, dtype="float32"),
        dims=("y", "x"),
        coords={"x": xs, "y": ys},
        name="surface",
        attrs={"AREA_OR_POINT": "Point"},
    )
    da = da.rio.write_crs(p["crs"]).rio.write_coordinate_system()
    ds = da.to_dataset()
    ds.attrs["Conventions"] = "CF-1.8"

    geo_file.unlink(missing_ok=True)
    ds.to_netcdf(geo_file, engine="h5netcdf")
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
        Target grid dataset providing the destination CRS
        and the cell-edge bounds (``x_bnds``/``y_bnds``) used to derive both the
        geographic query bounds and the destination grid for reprojection.
    dataset : {"glo_30", "arcticdem"}
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
        ``standard_name="surface_altitude"``. Also written to
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
      ``standard_name="surface_altitude"`` and ``units="m"``.
    - After reprojection to the target grid, missing values are filled with 0.
      Adjust if downstream tooling expects masked NaNs instead.
    """

    bounds = [
        target_grid.x_bnds.values[0][0],
        target_grid.y_bnds.values[0][0],
        target_grid.x_bnds.values[-1][-1],
        target_grid.y_bnds.values[-1][-1],
    ]
    mapping_var = target_grid.rio.grid_mapping
    dst_crs = target_grid[mapping_var].attrs["crs_wkt"]
    t = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)
    geo_bounds = t.transform_bounds(*bounds)

    geo_file = get_surface_dem_by_bounds(geo_bounds, dataset=dataset, path=path, **kwargs)
    # rxr.open_rasterio reliably propagates CRS on a netCDF read; xr.open_dataset
    # + manual write_crs misses grid_mapping unless the variable has the attribute.
    surface = rxr.open_rasterio(geo_file).squeeze().drop_vars("band", errors="ignore")
    surface.name = "surface"
    surface.attrs.update({"standard_name": "surface_altitude", "units": "m"})

    surface_reprojected = surface.rio.reproject_match(target_grid, resampling=Resampling.bilinear).fillna(0)

    surface_file = Path(path) / "surface.nc"
    surface_reprojected.to_netcdf(surface_file)

    return surface_reprojected


def boot_file_from_grid(
    target_grid: xr.Dataset,
    rgi_id: str,
    geometries: collections.abc.Iterable,
    dem_dataset: Literal["glo_30", "arcticdem"],
    ice_thickness_dataset: Literal["maffezzoli", "millan"],
    bathymetry_dataset: Literal["none", "gebco"] | None,
    velocity_dataset: Literal["none", "its_live"] | None,
    forcing_mask: Literal["none", "all", "glacier"] | None,
    path: str | Path = "input_files",
    **kwargs,
) -> xr.Dataset:
    """
    Build a glacier “boot” dataset (surface, thickness, bed, masks, aux vars) from an RGI ID.

    Returns a regular 2-D xarray Dataset in the glacier’s projected CRS.

    Parameters
    ----------
    target_grid : xarray.Dataset
        Target grid dataset (with ``x``/``y`` coords, ``x_bnds``/``y_bnds`` cell-edge
        bounds onto which all
        derived fields are aligned.
    rgi_id : str
        Glacier identifier, e.g., ``"RGI2000-v7.0-C-06-00014"``.
    geometries : iterable of shapely geometries
        Glacier outline(s) in ``target_grid``'s CRS, used for clipping the DEM
        and constructing masks.
    dem_dataset : {"glo_30", "arcticdem"}
        DEM source for surface preparation.
    ice_thickness_dataset : {"maffezzoli", "millan"}
        Source for ice thickness.
    bathymetry_dataset : {"none", "gebco"} or None
        Source for ocean bathymetry. When set, a cloud raster (e.g.
        ``s3://.../<bathymetry_dataset>/bathymetry.tif``) is fetched, clipped
        to the target grid, and used to fill ``bed`` where ``surface <= 0``.
        ``"none"`` or ``None`` disables bathymetry merging.
    velocity_dataset : {"none", "its_live"} or None
        Source for velocities.
    forcing_mask : {"none", "glacier", "all"} or None
        FTT mask.
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
        - ``v`` and related velocity fields (from :func:`glacier_velocities_from_grid`), reprojected
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
    glacier_velocities_from_grid
        Retrieve observed surface velocities for the glacier domain.
    """

    print("")
    print("Generate DEM")
    print("-" * 120)

    bucket: str = kwargs.pop("bucket", "pism-cloud-data")
    prefix: str = kwargs.pop("prefix", "rgi")

    mapping_var = target_grid.rio.grid_mapping
    dst_crs = target_grid[mapping_var].attrs["crs_wkt"]

    surface = prepare_surface(target_grid, dataset=dem_dataset, path=path)
    surface = surface.where(surface > 0.0, 0.0)

    ice_thickness = get_ice_thickness(
        rgi_id,
        dataset=ice_thickness_dataset,
        path=path,
        target_grid=target_grid,
        target_crs=dst_crs,
        bucket=bucket,
        prefix=prefix,
        **kwargs,
    )
    ice_thickness = ice_thickness.rio.reproject_match(target_grid, resampling=Resampling.bilinear)
    ice_thickness = ice_thickness.rio.clip(geometries, drop=False).fillna(0)
    ice_thickness = ice_thickness.where(ice_thickness > 0.0, 0.0)

    bed = surface - ice_thickness
    if bathymetry_dataset not in ("none", None):
        bathymetry_uri = f"/vsis3/{bucket}/{prefix}/{bathymetry_dataset}/bathymetry.tif"
        bathymetry_p = path / Path("bathymetry.nc")
        bathymetry = bathymetry_from_grid(target_grid, uri=bathymetry_uri, path=bathymetry_p)
        bed = bed.where(surface > 0.0, bathymetry)
        bed.name = "bed"
    bed.attrs.update({"standard_name": "bedrock_altitude", "units": "m"})

    liafr = surface.rio.clip(geometries, drop=False)
    liafr = xr.where(liafr.isnull(), 0, 1)
    liafr.name = "land_ice_area_fraction_retreat"
    liafr.attrs.update({"units": "1"})
    liafr = liafr.astype("bool")

    if forcing_mask == "glacier":
        print("Forcing mask: 1 outside 0 inside glacier")
        # rio.clip leaves the surface elevation inside the polygon and NaN
        # outside. Without the xr.where, the subsequent .astype("bool") would
        # turn both real elevations AND NaN into True (1 everywhere). Map
        # NaN -> 1 (outside) and any value -> 0 (inside) explicitly.
        ftt_mask = surface.rio.clip(geometries, drop=False)
        ftt_mask = xr.where(ftt_mask.isnull(), 1, 0)
    elif forcing_mask == "all":
        print("Forcing mask: 1 everywhere")
        ftt_mask = xr.ones_like(liafr)
    else:
        print("Forcing mask: 0 everywhere")
        ftt_mask = xr.zeros_like(liafr)
    ftt_mask.name = "ftt_mask"
    ftt_mask.attrs.update({"units": "1"})
    if "standard_name" in ftt_mask.attrs:
        del ftt_mask.attrs["standard_name"]

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

    # Drop the leftover `band` scalar coord that rxr.open_rasterio leaves on
    # some inputs (e.g. ice thickness, velocity). If it survives in the merged
    # Dataset, xarray auto-emits ``coordinates = "band"`` on every variable on
    # write, which trips up QGIS.
    ds = ds.drop_vars("band", errors="ignore")
    for var in ds.data_vars:
        ds[var].attrs.pop("coordinates", None)
        ds[var].encoding.pop("coordinates", None)

    # Re-attach the CRS as a CF grid_mapping link on every data variable
    # and stamp standard_name/units onto x/y so GDAL recognises them as
    # projected eastings/northings (otherwise the geotransform is dropped).
    ds = ds.rio.write_crs(dst_crs).rio.write_grid_mapping().rio.write_coordinate_system()

    # Mark the file as CF so GDAL picks the netCDF driver (and reads
    # grid_mapping → CRS) instead of falling through to the HDF5 driver.
    ds.attrs["Conventions"] = "CF-1.8"

    for name in ("x", "y", "thickness", "bed", "surface", "tillwat", "ftt_mask", "land_ice_area_fraction_retreat"):
        if name in ds:
            ds[name].encoding.update({"_FillValue": None})
    return ds
