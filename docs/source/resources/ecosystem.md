# Ecosystem

pism-terra sits at the seam of several glaciological projects.

## Upstream

- **[PISM](https://www.pism.io)** — the ice-sheet model that does the science.
  pism-terra writes its inputs and reads its outputs but does not modify it.
- **[RGI v7](https://www.glims.org/RGI/)** — Randolph Glacier Inventory v7,
  source of every glacier and complex outline used in the package.

## Input data

- **DEMs**: Copernicus GLO-30, ArcticDEM (via `dem_stitcher`).
- **Ice thickness**: Maffezzoli (Zenodo), Frank (Figshare).
- **Climate**: ECMWF ERA5 (via ECMWF Data Stores), CARRA2 (S3), PMIP4, SNAP.
- **Velocities**: ITS_LIVE v2.1 per-region COGs.
- **Bathymetry**: GEBCO.

## Related tooling

- **GLAMBIE** — glacier mass-balance intercomparison; pism-terra outputs can be
  formatted for GLAMBIE submission (TODO).
- **ISMIP7** — community ice-sheet model intercomparison; pism-terra ships
  ISMIP7 Greenland-specific CLIs.

```{admonition} TODO
- Add canonical citations for each upstream data product.
- Diagram the data flow across the ecosystem.
```
