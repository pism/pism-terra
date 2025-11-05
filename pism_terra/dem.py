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

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr
from dem_stitcher import stitch_dem
from rasterio.merge import merge

from pism_terra.aws import s3_to_local
from pism_terra.domain import create_domain
from pism_terra.download import download_archive, extract_archive
from pism_terra.observations import glacier_velocities_from_rgi_id
from pism_terra.raster import check_overlap, reproject_file
from pism_terra.vector import get_glacier_from_rgi_id
from pism_terra.workflow import check_rio, check_xr_sampled

xr.set_options(keep_attrs=True)


def get_surface_dem_by_bounds(
    bounds: tuple[float, float, float, float],
    dem_name: str = "glo_30",
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
    dem_name : str, default ``"glo_30"``
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
    geoid_file = out_dir / f"{dem_name}.tif"
    # Reuse if present and readable
    if (not check_rio(geoid_file)) or force_overwrite:
        X, p = stitch_dem(
            bounds,
            dem_name=dem_name,
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

        with rasterio.open(geoid_file, "w", **p) as src:
            src.write(X, 1)  # write band 1
            src.update_tags(AREA_OR_POINT="Point")

    return geoid_file


def prepare_ice_thickness(
    glacier, target_grid: xr.Dataset | xr.DataArray, dataset: str = "millan", path: str | Path = "input_files", **kwargs
):
    """
    Prepare ice thickness data for a given glacier and target grid.

    This function dispatches to a dataset-specific loader to prepare an
    ice thickness field interpolated to the resolution and bounds of a
    specified target grid. Currently only the "millan" dataset is supported.

    Parameters
    ----------
    glacier : geopandas.GeoDataFrame or geopandas.GeoSeries
        Glacier geometry to match against ice thickness tiles.
    target_grid : xarray.Dataset or xarray.DataArray
        Grid to which the output ice thickness will be interpolated.
    dataset : str, optional
        The name of the ice thickness dataset to use. Currently only "millan" is implemented.
        Default is "millan".
    path : str or pathlib.Path, default ``"input_files"``
        Working directory used by helper routines to cache/write intermediate rasters/grids.
    **kwargs
        Additional keyword arguments passed to the dataset-specific function,
        e.g., `target_crs="EPSG:32641"`.

    Returns
    -------
    xarray.DataArray
        Ice thickness interpolated to the target grid.

    Raises
    ------
    NotImplementedError
        If the specified dataset is not supported.
    """
    if dataset == "millan":
        thickness = prepare_ice_thickness_millan(glacier, target_grid, path=path, **kwargs)
    else:
        raise NotImplementedError(f"Ice thickness dataset '{dataset}' not implemented.")
    thickness = thickness.where(thickness > 0, 0)
    thickness.attrs.update({"standard_name": "land_ice_thickness", "units": "m"})

    return thickness


def prepare_ice_thickness_millan(
    glacier, target_grid: xr.Dataset | xr.DataArray, path: str | Path = "input_files", **kwargs
):
    """
    Load and interpolate Millan et al. (2022) ice thickness data to a target grid.

    This function identifies all Millan ice thickness raster files that overlap
    the input glacier geometry, reprojects them to the specified CRS and resolution,
    interpolates them onto the target grid, and returns the summed result.

    Parameters
    ----------
    glacier : geopandas.GeoDataFrame or geopandas.GeoSeries
        Geometry of the glacier to extract overlapping thickness rasters.
    target_grid : xarray.Dataset or xarray.DataArray
        Target grid to which ice thickness should be interpolated.
    path : str or pathlib.Path, default ``"input_files"``
        Working directory used by helper routines to cache/write intermediate rasters/grids.
    **kwargs
        Additional keyword arguments. Must include:
        - target_crs : str
            CRS to reproject rasters to (e.g., "EPSG:32641").

    Returns
    -------
    xarray.DataArray
        Interpolated and summed ice thickness field on the target grid.

    Notes
    -----
    - Uses `rioxarray` to load and project raster files.
    - All overlapping rasters are summed to produce the final thickness field.
    - Assumes a fixed reprojected resolution of 50 meters.
    """

    force_overwrite: bool = bool(kwargs.pop("force_overwrite", False))

    bucket: str = "pism-cloud-data"
    out_dir = Path(path)
    thickness_file = out_dir / "thickness.nc"
    ice_thickness_files = list(Path("data/ice_thickness/millan").rglob("THICKNESS_*.tif"))

    if (not check_xr_sampled(thickness_file)) or force_overwrite:

        thickness_file.unlink()
        s3_to_local(bucket, prefix="millan", dest_dir="data/ice_thickness")

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(check_overlap, path, glacier) for path in ice_thickness_files]

            overlapping_rasters = [f.result() for f in as_completed(futures) if f.result() is not None]

        thicknesses = []
        for k, p in enumerate(overlapping_rasters):
            if p is not None:
                projected_file = reproject_file(p, dst_crs=kwargs["target_crs"], resolution=50)
                da = rxr.open_rasterio(projected_file).squeeze().drop_vars("band", errors="ignore")
                thickness = da.interp_like(target_grid)
                thickness.rio.write_nodata(None, inplace=True)
                thickness.name = "thickness"
                thickness["raster"] = k
                thicknesses.append(thickness)

        thickness = xr.concat(thicknesses, dim="raster").sum(dim="raster")
        thickness.to_netcdf(thickness_file)
    else:
        thickness = xr.open_dataset(thickness_file)

    return thickness


def prepare_ice_thickness_farinotti(glacier):
    """
    Load and interpolate Farniotti et al (2019) ice thickness data to a target grid.

    This function identifies all Millan ice thickness raster files that overlap
    the input glacier geometry, reprojects them to the specified CRS and resolution,
    interpolates them onto the target grid, and returns the summed result.

    Parameters
    ----------
    glacier : geopandas.GeoDataFrame or geopandas.GeoSeries
        Geometry of the glacier to extract overlapping thickness rasters.

    Returns
    -------
    xarray.DataArray
        Interpolated and summed ice thickness field on the target grid.

    Notes
    -----
    - Uses `rioxarray` to load and project raster files.
    - All overlapping rasters are summed to produce the final thickness field.
    - Assumes a fixed reprojected resolution of 50 meters.
    """

    path = Path("data/ice_thickness")
    path.mkdir(parents=True, exist_ok=True)

    region = glacier["o1region"]
    url = f"https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/315707/composite_thickness_RGI60-{region}.zip"
    archive = download_archive(url)
    extract_archive(archive, extract_to=path)

    ice_thickness_files = list(Path(f"data/ice_thickness/RGI60-{region}").rglob("*.tif"))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(check_overlap, path, glacier) for path in ice_thickness_files]

        overlapping_rasters = [f.result() for f in as_completed(futures) if f.result() is not None]

    # Step 1: List all .tif files
    tif_files = overlapping_rasters

    # Step 2: Open all files as datasets
    src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files]

    # Step 3: Merge them
    mosaic, out_transform = merge(src_files_to_mosaic)

    # Step 4: Get metadata from first file, update with new shape and transform
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update(
        {"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_transform}
    )

    # Step 5: Write the result to disk
    with rasterio.open("merged.tif", "w", **out_meta) as dest:
        dest.write(mosaic)


def add_malaspina_bed(
    ds: xr.Dataset,
    target_crs: str,
    bed_file: str | Path = "data/ice_thickness/malaspina/malaspina_bed_3338.tif",
    outline_file: str | Path = "data/rgi/rgi-malaspina.shp",
) -> xr.Dataset:
    """
    Replace bed topography in a dataset using the Malaspina Glacier bed dataset.

    This function reads a GeoTIFF file containing bed topography data for the Malaspina Glacier,
    clips it to the glacier outline, reprojects it to match the target dataset's CRS, and
    replaces the corresponding region in the input dataset. It also updates the `thickness`
    and `surface` fields accordingly.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing at least the variables 'bed', 'thickness', and 'surface'.
    target_crs : str
        The target coordinate reference system (CRS) to use for reprojection (e.g., "EPSG:3413").
    bed_file : str or Path, optional
        Path to the GeoTIFF file containing the Malaspina bed topography.
        Default is "data/ice_thickness/malaspina/malaspina_bed_3338.tif".
    outline_file : str or Path, optional
        Path to the glacier outline shapefile used to clip the bed topography.
        Default is "data/rgi/rgi-malaspina.shp".

    Returns
    -------
    xr.Dataset
        Modified dataset with updated 'bed', 'thickness', and 'surface' fields within the Malaspina region.

    Notes
    -----
    - Bed values of -9999.0 are treated as nodata and replaced with NaN.
    - Replaces `bed` where new values are available and recalculates `thickness = surface - bed`.
    - Ensures that thickness and surface are non-negative.
    - Updates CF-convention attributes and CRS metadata.
    """

    outline = gpd.read_file(outline_file).to_crs(target_crs)
    da = (
        rxr.open_rasterio(bed_file, mask=True)
        .squeeze()
        .drop_vars("band", errors="ignore")
        .rio.reproject_match(ds["bed"])
    )
    clipped_da = da.rio.clip(outline.geometry, drop=False)
    clipped_da = clipped_da.where(clipped_da != -9999.0, other=np.nan).drop_vars("spatial_ref")
    ds["bed"] = xr.where(~np.isnan(clipped_da), clipped_da, ds["bed"], keep_attrs=True)
    ds["thickness"] = xr.where(~np.isnan(clipped_da), ds["surface"] - clipped_da, ds["thickness"], keep_attrs=True)

    ds["thickness"] = ds["thickness"].where(ds["thickness"] > 0.0, 0.0)
    ds["surface"] = ds["surface"].where(ds["thickness"] > 0.0, 0.0)
    ds["surface"].attrs.update({"standard_name": "land_ice_elevation", "units": "m"})

    ds["thickness"].attrs.update({"standard_name": "land_ice_thickness", "units": "m"})

    ds["bed"].attrs.update({"standard_name": "bedrock_altitude", "units": "m"})
    return ds


def prepare_surface(
    bounds: tuple[float, float, float, float],
    crs: str,
    dem_name: str = "glo_30",
    path: str | Path = "input_files",
    resolution: float = 50.0,
    **kwargs,
) -> tuple[Path, Path]:
    """
    Prepare a surface DEM and matching target grid over a glacier extent.

    Workflow:
    (1) Download/mosaic a DEM over ``bounds`` (geographic),
    (2) reproject/resample it to ``crs`` at ``resolution``,
    (3) crop to a 100 m–aligned box in projected coordinates,
    (4) generate a regular target grid covering that box, and
    (5) write both the surface (NetCDF) and the target grid (NetCDF) to ``path``.

    Parameters
    ----------
    bounds : tuple of float
        Geographic bounding box as ``(minx, miny, maxx, maxy)`` in WGS84 degrees.
    crs : str
        Target coordinate reference system (e.g., ``"EPSG:32606"``).
    dem_name : str, default ``"glo_30"``
        DEM source identifier passed to :func:`get_surface_dem_by_bounds`.
    path : str or pathlib.Path, default ``"input_files"``
        Output directory for generated files. Created if missing.
    resolution : float, default ``50.0``
        Target grid spacing (meters) used during reprojection and grid creation.
    **kwargs
        Additional keyword arguments forwarded to :func:`get_surface_dem_by_bounds`
        (e.g., ``force_overwrite=True``). These do not affect the reprojection
        or grid creation steps directly.

    Returns
    -------
    tuple of pathlib.Path
        ``(surface_file, target_grid_file)``:
        - ``surface_file``: NetCDF with variable ``surface`` (meters).
        - ``target_grid_file``: NetCDF with the regular grid (coords/bounds).

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
    - Cropping aligns the domain to multiples of 100 m in projected x/y:
      ``x_min = ceil(min(x)/100)*100`` and ``x_max = floor(max(x)/100)*100``
      (similarly for y).
    - After interpolation to the target grid, missing values are filled with 0.
      Adjust if downstream tooling expects masked NaNs instead.

    Examples
    --------
    >>> surf_nc, grid_nc = prepare_surface(
    ...     bounds=(214.1, 59.0, 219.7, 63.9),
    ...     crs="EPSG:32606",
    ...     dem_name="glo_30",
    ...     path="input_files",
    ...     resolution=50.0,
    ...     force_overwrite=True,  # forwarded via **kwargs
    ... )
    """

    geoid_file = get_surface_dem_by_bounds(bounds, dem_name=dem_name, path=path, **kwargs)
    projected_file = reproject_file(geoid_file, crs, resolution)

    surface = rxr.open_rasterio(projected_file).squeeze().drop_vars("band", errors="ignore")
    surface.name = "surface"
    surface.attrs.update({"standard_name": "land_ice_elevation", "units": "m"})

    # Round in projected meters to a 100 m grid
    x_min = np.ceil((surface.x.min()) / 100) * 100
    x_max = np.floor((surface.x.max()) / 100) * 100
    y_min = np.ceil((surface.y.min()) / 100) * 100
    y_max = np.floor((surface.y.max()) / 100) * 100

    target_grid = create_domain([x_min, x_max], [y_min, y_max], resolution=resolution, crs=crs)

    surface = surface.interp_like(target_grid).fillna(0)

    surface_file = Path(path) / "surface.nc"
    surface.to_netcdf(surface_file)

    target_grid_file = Path(path) / "target_grid.nc"
    target_grid.to_netcdf(target_grid_file)

    return surface_file, target_grid_file


def boot_file_from_rgi_id(
    rgi_id: str,
    rgi: gpd.GeoDataFrame | str | Path = "rgi/rgi.gpkg",
    dem_name: str = "glo_30",
    buffer_distance: float = 5000.0,
    path: str | Path = "input_files",
    resolution: float = 50.0,
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
    rgi_id : str
        Glacier identifier, e.g., ``"RGI2000-v7.0-C-06-00014"``.
    rgi : geopandas.GeoDataFrame or str or pathlib.Path, default ``"rgi/rgi.gpkg"``
        In-memory RGI table or a path to a GeoPackage/shape readable by
        :func:`geopandas.read_file`. Must contain a row with ``rgi_id`` and an
        ``epsg`` column specifying the glacier CRS.
    dem_name : str, default ``"glo_30"``
        DEM source for surface preparation (e.g., ``"glo_30"``, ``"arcticdem"``).
    buffer_distance : float, default ``5000.0``
        Buffer distance **in meters** applied to the glacier polygon in the projected CRS
        to define the working extent for DEM/thickness.
    path : str or pathlib.Path, default ``"input_files"``
        Working directory used by helper routines to cache/write intermediate rasters/grids.
    resolution : float, default ``50.0``
        Target grid spacing (meters) for reprojection/resampling.
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
    prepare_ice_thickness
        Interpolate glacier ice thickness onto a target grid.
    create_domain
        Create a regular xarray grid with specified bounds and resolution.
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
    print("-" * 80)

    # Accept GeoDataFrame or a path to the RGI layer
    if isinstance(rgi, (str, Path)):
        rgi = gpd.read_file(rgi)

    glacier = get_glacier_from_rgi_id(rgi, rgi_id)
    glacier_series = glacier.iloc[0]
    dst_crs = glacier_series["epsg"]

    glacier_projected = glacier.to_crs(dst_crs)
    geometry_buffered_projected = glacier_projected.geometry.buffer(buffer_distance)
    geometry_buffered_geoid = geometry_buffered_projected.to_crs("EPSG:4326").iloc[0]

    bounds_geoid_buffered = geometry_buffered_geoid.bounds
    surface_file, target_grid_file = prepare_surface(
        bounds_geoid_buffered, dst_crs, dem_name=dem_name, path=path, resolution=resolution, **kwargs
    )
    surface = xr.open_dataarray(surface_file)
    surface = surface.where(surface > 0.0, 0.0)
    target_grid = xr.open_dataset(target_grid_file)

    ice_thickness = prepare_ice_thickness(glacier, path=path, target_grid=target_grid, target_crs=dst_crs)
    ice_thickness = ice_thickness.rio.clip(glacier_projected.geometry, drop=False).fillna(0)
    ice_thickness = ice_thickness.where(ice_thickness > 0.0, 0.0)

    bed = surface - ice_thickness
    bed.name = "bed"
    bed.attrs.update({"standard_name": "bedrock_altitude", "units": "m"})

    liafr = surface.rio.clip(glacier_projected.geometry, drop=False)
    liafr = xr.where(liafr.isnull(), 0, 1)
    liafr.name = "land_ice_area_fraction_retreat"
    liafr.attrs.update({"units": "1"})
    liafr = liafr.astype("bool")

    ftt_mask = surface.rio.clip(glacier_projected.geometry, drop=False)
    ftt_mask = xr.where(ftt_mask.isnull(), 1, 0)
    ftt_mask.name = "ftt_mask"
    ftt_mask.attrs.update({"units": "1"})
    ftt_mask = ftt_mask.astype("bool")

    v_filename = path / Path(f"obs_{rgi_id}.nc")
    v = glacier_velocities_from_rgi_id(rgi_id, rgi, buffer_distance=5000.0, path=v_filename, **kwargs)
    v = v.rio.reproject_match(surface)
    _v = v["v"].fillna(0)

    tillwat = xr.where(_v < 100, 0, xr.where(_v > 500, 2, 1 + (_v - 100) / (500 - 100)))
    tillwat.name = "tillwat"
    tillwat.attrs.update({"units": "m"})

    ds = xr.merge([bed, surface, ice_thickness, liafr, ftt_mask, tillwat, _v])
    return ds.fillna(0)
