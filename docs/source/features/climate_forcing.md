# Climate forcing

Climate-forcing preparation lives in {py:mod}`pism_terra.glacier.climate` and
supports several backends keyed off the run config's `climate` block.

## Supported backends

| Backend | Description | Function |
|---|---|---|
| `era5` | ERA5 monthly means via the ECMWF data stores client | {py:func}`~pism_terra.glacier.climate.era5` |
| `carra2` | Pan-Arctic CARRA2 reanalysis (Zarr on S3; per-S4F-group caches) | {py:func}`~pism_terra.glacier.climate.carra2` |
| `pmip4` | PMIP4 paleo simulations | {py:func}`~pism_terra.glacier.climate.pmip4` |
| `snap` | SNAP downscaled climate (GeoTIFFs) | {py:func}`~pism_terra.glacier.climate.snap` |

## CARRA2 caching

`pism-glacier-prepare` pre-reprojects CARRA2 once per S4F aggregate group and
uploads `carra2_<rgi_id>.nc` to S3. The per-glacier
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
