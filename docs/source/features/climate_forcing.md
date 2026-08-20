# Climate forcing

Climate-forcing preparation lives in {py:mod}`pism_terra.glacier.climate` and
supports several backends keyed off the run config's `climate` block.

## Supported backends

| Backend | Description | Function |
|---|---|---|
| `era5` | ERA5 monthly means via the ECMWF data stores client | {py:func}`~pism_terra.glacier.climate.era5` |
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
