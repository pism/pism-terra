# Ice thickness

Two ice-thickness products are wired up:

| Backend | Source | Function |
|---|---|---|
| `maffezzoli` | Zenodo per-region zip of RGI v7 thickness rasters | {py:func}`pism_terra.glacier.ice_thickness.prepare_ice_thickness_maffezzoli` |
| `frank` | Figshare global per-RGI v6 thickness tifs | {py:func}`pism_terra.glacier.ice_thickness.prepare_ice_thickness_frank` |

## Maffezzoli

RGI v7-keyed; the per-region archive maps directly onto pism-terra's complex
outlines.

## Frank

RGI v6-keyed and only published for some glaciers (about 12,000 globally).
pism-terra resolves the v6 ⇄ v7 mismatch by **spatial intersection** rather
than ID matching — see
{py:func}`pism_terra.glacier.ice_thickness.prepare_ice_thickness_frank` and the
auxiliary footprint indexer
{py:func}`pism_terra.glacier.ice_thickness._frank_footprint_4326`.

Per-complex merging uses `rasterio.merge` with `method="max"` and
`nodata=0` so the literal-zero "outside outline" pixels in each per-glacier
tif don't overwrite real thickness from a neighbouring tif's edge.

```{admonition} TODO
- Discuss how the two products compare in coverage.
- Document the spatial intersection algorithm with a figure.
- Describe how to add a third backend (e.g. Millan).
```
