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

```text
<RGI_ID>/
├── input/                 # final, PISM-ready NetCDFs (boot file, climate, obs, …)
└── staging/               # intermediates (cached so reruns are cheap)
```

```{admonition} TODO
- Document per-input cache invalidation rules (`force_overwrite`).
- Cross-link to each backend page once written.
- Add a sequence diagram of the staging pipeline.
```
