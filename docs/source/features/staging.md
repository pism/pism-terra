# Staging inputs

Staging is the step that turns "an RGI ID + a config TOML" into the
PISM-ready NetCDF inputs a run needs. The work is driven by
{py:func}`pism_terra.glacier.stage.main` and the
`pism-glacier-stage` CLI.

## What gets staged

| Input | Source | Module |
|---|---|---|
| Surface DEM | Copernicus GLO-30 or ArcticDEM | {py:mod}`pism_terra.glacier.dem` |
| Bathymetry | GEBCO | {py:func}`pism_terra.glacier.observations.bathymetry_from_grid` |
| Ice thickness | Maffezzoli (Zenodo) or Frank (Figshare) | {py:mod}`pism_terra.glacier.ice_thickness` |
| Climate forcing | ERA5 / PMIP4 / CARRA2 / SNAP | {py:mod}`pism_terra.glacier.climate` |
| Velocities | ITS_LIVE v2.1 per-region COG | {py:mod}`pism_terra.glacier.observations` |

## Output layout

`pism-glacier-stage` writes one tree per glacier:

```text
<RGI_ID>/
├── input/                 # final, PISM-ready NetCDFs (boot file, climate, obs, …)
└── staging/               # intermediates (cached so reruns are cheap)
```

## Prepared-input layout

`pism-glacier-prepare` writes the tree those runs draw from. Its `input/`
directory is what gets synced to S3 and addressed by the campaign config's
`prefix`:

```text
<OUTPUT_PATH>/
├── input/
│   ├── gebco/, heatflux/               # global — shared by every project
│   ├── climate/carra2.zarr, carra2_monthly_mean.zarr, snap_cru_TS40_*.nc
│   └── <project_directory>/            # e.g. rgi, s4f
│       ├── rgi/<project>_{c,g}.gpkg
│       ├── ice_thickness/{frank,maffezzoli}/
│       └── climate/carra2_<group>.nc, carra2_monthly_mean_<group>.nc
└── staging/                            # intermediates, never uploaded
```

The split follows the `[regions]` CRS overrides: anything whose contents depend
on them lives under `<project_directory>` (addressed as
`{prefix}/{project_directory}`), and everything else is stored once. Campaigns
that do not set `project_directory` see the two prefixes collapse into one.

```{admonition} TODO
- Document per-input cache invalidation rules (`force_overwrite`).
- Cross-link to each backend page once written.
- Add a sequence diagram of the staging pipeline.
```
