# Climate forcing

Climate-forcing preparation lives in {py:mod}`pism_terra.glacier.climate` and
supports several backends keyed off the run config's `climate` block.

## Supported backends

| Backend | Description | Function |
|---|---|---|
| `era5` | ERA5 monthly means, cut from a regional store on S3 (or downloaded per glacier from CDS) | {py:func}`~pism_terra.glacier.climate.era5` |
| `carra2` | Pan-Arctic CARRA2 reanalysis (Zarr on S3; per-S4F-group caches) | {py:func}`~pism_terra.glacier.climate.carra2` |
| `carra2-monthly-mean` | CARRA2 1990–2019 monthly climatology, 12 periodic steps | {py:func}`~pism_terra.glacier.climate.carra2_monthly_mean` |
| `pmip4` | PMIP4 paleo simulations | {py:func}`~pism_terra.glacier.climate.pmip4` |
| `snap` | SNAP downscaled climate (GeoTIFFs) | {py:func}`~pism_terra.glacier.climate.snap` |

## Monthly climatologies

`carra2-monthly-mean` is the CARRA2 counterpart of `era5-monthly-mean`: twelve
fields, one per calendar month, on a 365-day `days since 0001-01-01` axis with
`time_bounds` tiling the year — a periodic forcing PISM cycles for the length of
the run.

The averaging happens once, in `pism-glacier-prepare`, over a **fixed**
1990–2019 reference period
({py:data}`~pism_terra.glacier.climate.CARRA2_CLIMATOLOGY_YEARS`). Fixing the
period keeps the climatology stable when the CARRA2 download is later extended,
and comparable with the ERA5/RACMO/MAR means over the same years. The result is
`climate/carra2_monthly_mean.zarr`, shared across projects like the store it
comes from.

`air_temp_sd` is averaged like every other field, so it stays the typical
*within-month* temperature variability — what PISM's positive-degree-day scheme
reads. It is deliberately not the year-to-year spread of the monthly means,
which is much smaller and would understate melt.

## ERA5 regional stores

Asking CDS for one glacier's ERA5 forcing takes hours: the standard deviation
of the daily means, `air_temp_sd`, needs a year of daily statistics per request.
`pism-glacier-prepare --include era5` therefore builds the forcing once per
region of the setup file and writes it to `climate/era5_<region>.zarr` under the
project subtree, over the box of that region's outlines and for
{py:data}`~pism_terra.glacier.climate.ERA5_STORE_YEARS` (1986–2025). See
{py:func}`~pism_terra.glacier.climate.prepare_era5`.

A region that names a `crs` in `[regions]` is stored in it, on a 5 km grid; one
that does not stays on ERA5-Land's latitude/longitude grid. ERA5 is requested
over the box of the glaciers plus one degree. A projected store is a rectangle
in its own CRS and wider than that box at its corners -- Alaska's reach past
180° E -- so the cells there, away from every glacier, take the value of the
nearest cell with data:

```toml
[regions]
1 = {name = "alaska", crs = "EPSG:5936"}   # era5_01_alaska.zarr in EPSG:5936
6 = {name = "iceland"}                      # era5_06_iceland.zarr in EPSG:4326
```

Staging with `climate = "era5"` looks for a store that covers the glacier's grid
and the years of the run and crops it
({py:func}`~pism_terra.glacier.climate.era5_from_store`): to `era5_<rgi_id>.nc`
in the store's CRS, or to `era5_wgs84_<rgi_id>.nc` in latitude/longitude. PISM
interpolates either onto the model grid. When no store covers the glacier, or
one lacks a year, the forcing is downloaded from CDS for that glacier as before.

## CARRA2 caching

`pism-glacier-prepare` pre-reprojects CARRA2 once per aggregate group and
uploads `carra2_<rgi_id>.nc` to S3 under the project subtree
(`<prefix>/<project_directory>/climate/`), since the result depends on that
project's CRS. The merged `carra2.zarr` store itself is global and is shared
across projects at `<prefix>/climate/`. The per-glacier
{py:func}`~pism_terra.glacier.climate.carra2` call then downloads that single
file instead of streaming the full Zarr — see
{py:func}`~pism_terra.glacier.climate.prepare_carra2_for_group`.

The runtime also fills missing years from the nearest available source year
and attaches monthly `time_bnds` so PISM can interpret the data as monthly
means
({py:func}`pism_terra.glacier.climate._carra2_fill_years_and_bounds`).

```{admonition} TODO
- Document the expected variable names per backend.
- Describe how to add a new backend.
- Cross-link to the PISM atmosphere/surface model docs.
```
