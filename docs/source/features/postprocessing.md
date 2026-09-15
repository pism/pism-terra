# Post-processing

Post-processing happens after a PISM run finishes and turns the raw spatial
output into per-region diagnostics. One pair of campaign-neutral CLIs serves
Greenland drainage basins and RGI glaciers alike — the regions are simply
whatever the outline file contains.

## CLIs

- `pism-postprocess-scalar` — per-region sums over `x`/`y`, one row per outline.
- `pism-postprocess-spatial` — per-region masked fields, one NetCDF per outline.
- `pism-glacier-postprocess` — legacy TOML-driven glacier variant. The glacier
  run scripts no longer generate its input TOML; keep using it only against
  TOMLs written by an earlier version.

A glacier run is reduced twice, once per outline type: `-C` (the complex, one
region covering the whole glacier system) and `-G` (one region per glacier).
Both read the same spatial file, so `pism-glacier-run-*` names their outputs
`processed_scalar/scalar_C_<stem>.nc` and `scalar_G_<stem>.nc` rather than
letting both derive the same default name.

```bash
pism-postprocess-scalar SPATIAL.nc OUTDIR/ OUTLINES.gpkg
```

`OUTDIR/` may be a directory (the output is named `basin_<input>.nc`) or an
explicit file path.

## What it does

1. Reads the CRS from the file's grid-mapping variable and reprojects the
   outlines onto it, so outlines stored in EPSG:4326 (as RGI ships them) work
   against a projected model grid.
2. Drops `x_bnds`/`y_bnds` and the grid-mapping variables (their post-clip
   values would be stale), and splits off the non-spatial variables so they can
   ride along untouched.
3. Rasterizes every outline onto the model grid once, then reduces (or masks)
   all regions in a single Dask pass, so each input chunk is read once.
4. Writes the result with `time` leading and the region dimension as an integer
   index, which is what CDO needs to open the file at all.

## Options worth knowing

- `--column` — outline column holding the region name. Tried in order
  `glacier_id`, `rgi_id`, `SUBREGION1` when unset.
- `--dim-name` — name of the region dimension, `glacier_id` by default. Every
  campaign uses that default; pass something else only for one-off output.
- `--total-name` — append a whole-domain region summing every outline (the
  Greenland campaigns pass `GIS`). Off by default.
- `--crs` — override the CRS when a file carries no usable grid mapping.

Region labels live in a companion `<dim>_name` coordinate, so select by name
with `ds.set_index(glacier_id="glacier_id_name").sel(glacier_id="GIS_CE")`.
With the packaged Mouginot outlines the Greenland basins are labelled
`GIS_CE`, `GIS_CW`, … and the appended total is `GIS`.

See {py:func}`pism_terra.postprocess_scalar.process_file` for the details.

```{admonition} TODO
- Cross-link to the analysis notebooks in the gallery.
- Cover how to add custom basin masks.
```
