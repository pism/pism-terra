# Climate forcing

Climate-forcing preparation lives in {py:mod}`pism_terra.glacier.climate` and
supports several backends keyed off the run config's `climate` block.

## Supported backends

| Backend | Description | Function |
|---|---|---|
| `era5` | ERA5 monthly means, cut from a regional store on S3 (or downloaded per glacier from CDS) | {py:func}`~pism_terra.glacier.climate.era5` |
| `era5-mean` | ERA5 1990–2019 mean, one constant step (`era5_mean_<rgi_id>.nc`), downloaded per glacier from CDS | {py:func}`~pism_terra.glacier.climate.era5_mean` |
| `era5-monthly-mean` | ERA5 1990–2019 monthly climatology, 12 periodic steps | {py:func}`~pism_terra.glacier.climate.era5_monthly_mean` |
| `carra2` | Pan-Arctic CARRA2 reanalysis (Zarr on S3; per-S4F-group caches) | {py:func}`~pism_terra.glacier.climate.carra2` |
| `carra2-monthly-mean` | CARRA2 1990–2019 monthly climatology, 12 periodic steps | {py:func}`~pism_terra.glacier.climate.carra2_monthly_mean` |
| `pmip4` | PMIP4 paleo simulations | {py:func}`~pism_terra.glacier.climate.pmip4` |
| `snap` | SNAP downscaled climate (GeoTIFFs) | {py:func}`~pism_terra.glacier.climate.snap` |

## Monthly climatologies

`era5-monthly-mean` and `carra2-monthly-mean` are twelve fields, one per
calendar month, on a 365-day `days since 0001-01-01` axis with `time_bounds`
tiling the year — a periodic forcing PISM cycles for the length of the run,
typically the init leg (`campaign.init_climate`).

The averaging happens once, in `pism-glacier-prepare`, over a **fixed**
1990–2019 reference period
({py:data}`~pism_terra.glacier.climate.CARRA2_CLIMATOLOGY_YEARS`,
{py:data}`~pism_terra.glacier.climate.ERA5_CLIMATOLOGY_YEARS`). Fixing the
period keeps the climatology stable when the download is later extended, and
comparable with the RACMO/MAR means over the same years.

- CARRA2: `climate/carra2_monthly_mean.zarr`, shared across projects like the
  store it comes from
  ({py:func}`~pism_terra.glacier.climate.prepare_carra2_monthly_mean`).
- ERA5: `climate/era5_monthly_mean_<region>.zarr` under the project subtree,
  one per regional store and on its latitude/longitude grid
  ({py:func}`~pism_terra.glacier.climate.prepare_era5_monthly_mean`).

Staging crops the glacier's box out of it and writes
`carra2_monthly_mean_<rgi_id>.nc` or `era5_monthly_mean_<rgi_id>.nc`, next to
and never in place of the transient forcing. There is no per-glacier download
of the ERA5 climatology: the reference period is thirty years of CDS
requests. Staging a glacier that no regional climatology covers fails and names
the prepare step that builds it.

## Temperature spread

No climate forcing -- CARRA2, ERA5, SNAP or KITP's -- carries `air_temp_sd`,
the spread of the daily temperatures within a month. The positive-degree-day and dEBM surface models use
a constant spread instead (`surface.pdd.std_dev.value`,
`surface.debm_simple.std_dev.value`, 5 K: PISM's default, set explicitly in most
packaged configs), the value their
degree-day and melt coefficients were calibrated with. CARRA2's own spread was
about 1.5 K in summer, which with the classic degree-day factors cut the PDD
melt by a factor of several near 0 °C. A config that still asks for a spread
file (`surface.*.std_dev.file`) stops when the run script is generated.

## ERA5 regional stores

Asking CDS for every glacier's ERA5 forcing would repeat the same requests for
each glacier of a region. `pism-glacier-prepare --include era5` therefore builds
the forcing once per
region of the setup file and writes it to `climate/era5_<region>.zarr` under the
project subtree, over the box of that region's outlines and for
{py:data}`~pism_terra.glacier.climate.ERA5_STORE_YEARS` (1986–2025). See
{py:func}`~pism_terra.glacier.climate.prepare_era5`.

A store stays on ERA5-Land's own latitude/longitude grid, whatever CRS the
region is modelled in, and is bounded by the box of the region's glaciers plus
one degree. Nothing is resampled; PISM regrids the forcing onto the model grid,
as it does for CARRA2.

Staging with `climate = "era5"` looks for a store that covers the glacier's grid
and the years of the run and crops it to `era5_wgs84_<rgi_id>.nc`
({py:func}`~pism_terra.glacier.climate.era5_from_store`), the same file the
per-glacier download writes. When no store covers the glacier, or one lacks a
year, the forcing is downloaded from CDS for that glacier as before.

## CARRA2 on its own grid

`pism-glacier-prepare` builds the merged `carra2.zarr` store and its
`carra2_monthly_mean.zarr` climatology once, on CARRA2's own 2.5 km
polar-stereographic grid, shared across projects at `<prefix>/climate/`.
Staging a glacier ({py:func}`~pism_terra.glacier.climate.carra2`,
{py:func}`~pism_terra.glacier.climate.carra2_monthly_mean`) cuts the box of
its model domain out of the store, with three CARRA2 cells of margin on every
side ({py:func}`~pism_terra.glacier.climate.carra2_cutout`), and writes it as
it is, with CARRA2's projection in its grid mapping. Nothing is reprojected:
PISM reads the projection from the file and interpolates the forcing onto the
model grid, as it does for ERA5's latitude/longitude forcing.

The runtime also fills missing years from the nearest available source year
and attaches monthly `time_bnds` so PISM can interpret the data as monthly
means
({py:func}`pism_terra.glacier.climate._carra2_fill_years_and_bounds`).

```{admonition} TODO
- Document the expected variable names per backend.
- Describe how to add a new backend.
- Cross-link to the PISM atmosphere/surface model docs.
```
