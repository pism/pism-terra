# S4F planning products

Rasters for Snow 4 Flow mission planning, written by `pism-s4f-planning`
(pism-terra $version, $generated). There is one folder per glacier complex or
aggregate, `<RGI_ID>/input/`, and the complex outlines of the whole project
are in `$outlines`.

## Grid

All rasters of one complex share a grid:

- $resolution m cells, in the complex's projected CRS (the `crs` column of
  the RGI table, normally its UTM zone);
- covering the complex outline plus a margin of 1 to 2 km on every side
  (a 2 km margin, with the edges then moved inward onto a 1 km lattice).

The files are Cloud Optimized GeoTIFFs (ZSTD compression, with overviews and
statistics). Where a product is undefined (outside the outlines, or not
observed) the cells are NaN, which is the declared nodata value of the
observation products.

## Products

`<RGI_ID>` stands for the complex, e.g. `RGI2000-v7.0-C-01-04374`.

| File | Field | Units | Source |
|---|---|---|---|
| `<RGI_ID>_surface.tif` | surface elevation; below sea level set to 0 | m | DEM `$dem` |
| `<RGI_ID>_surface_clipped.tif` | surface elevation inside the outlines, NaN outside | m | DEM `$dem` |
| `<RGI_ID>_thickness.tif` | ice thickness, 0 outside the outlines | m | `$ice_thickness` |
| `<RGI_ID>_bed.tif`, `<RGI_ID>_bed.nc` | bed elevation: surface minus thickness, and the bathymetry where the surface is at sea level | m | the two above, bathymetry `$bathymetry` |
| `<RGI_ID>_dh.tif` | surface elevation change 2000-2020, the total over the 20 years (not per year) | m | Hugonnet et al. (2021), doi:10.1038/s41586-021-03436-z |
| `<RGI_ID>_dh_err.tif` | 1-sigma error of `dh` | m | Hugonnet et al. (2021) |
| `<RGI_ID>_${velocity}_v.tif` | surface speed | m yr-1 | `$velocity` |
| `<RGI_ID>_${velocity}_vx.tif`, `<RGI_ID>_${velocity}_vy.tif` | velocity components along the grid's x and y axes | m yr-1 | `$velocity` |
| `<RGI_ID>_${velocity}_vx_error.tif`, `<RGI_ID>_${velocity}_vy_error.tif` | errors of the components | m yr-1 | `$velocity` |
| `<RGI_ID>_carra2_snowfall_${snowfall_first}_${snowfall_last}.tif` | mean annual snowfall $snowfall_first-$snowfall_last, water equivalent | m yr-1 | CARRA2 |

`dh`, `dh_err` and the velocity fields are clipped to the outlines and NaN
where there is no observation. The snowfall covers the whole grid.

### Snowfall

CARRA2 gives monthly snowfall as a daily rate (kg m-2 day-1). Each month's
rate is multiplied by the number of days in that month, leap-year Februaries
included, and the months are summed to an annual total. The totals of
$snowfall_first-$snowfall_last are averaged and divided by 1000 kg m-3 to give metres
of water equivalent per year. This is done on CARRA2's 2.5 km grid, and the
result is interpolated bilinearly onto the planning grid, so it carries no
detail finer than CARRA2's.

### Model inputs

The folders can also hold fields that are inputs of the PISM runs rather than
planning products:

| File | Field |
|---|---|
| `<RGI_ID>_land_ice_area_fraction_retreat.tif` | 1 inside the outlines, 0 outside |
| `<RGI_ID>_ftt_mask.tif` | where the model's force-to-thickness mechanism applies (`$forcing_mask`) |
| `<RGI_ID>_v.tif` | surface speed with 0 where unobserved |
| `<RGI_ID>_tillwat.tif` | initial till water, 0 to 2 m, from the surface speed |

A product that does not apply to a complex is missing from its folder: a
complex outside the elevation-change tiles has no `dh`, one outside CARRA2's
domain has no snowfall.
