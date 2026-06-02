# Post-processing

Post-processing happens after a PISM run finishes and turns the raw spatial
output into basin-clipped NetCDFs and scalar diagnostics.

## CLIs

- `pism-glacier-postprocess` — glacier-domain runs.
- `pism-kitp-postprocess` — KITP campaigns.
- `pism-ismip7-greenland-postprocess` — ISMIP7 Greenland.

## What it does

1. Reads the spatial output described in the post-processing TOML.
2. Drops `x_bnds`/`y_bnds` (their post-clip values would be stale).
3. Splits into spatial and non-spatial variables, clips the spatial set to
   the basin polygon (`rio.clip(..., drop=False)`), and merges the
   non-spatial vars back in.
4. Writes the clipped spatial NetCDF and a per-basin scalar field-sum NetCDF.

See {py:func}`pism_terra.kitp.postprocess.process_file` for the KITP variant.

```{admonition} TODO
- Document the expected TOML schema.
- Cross-link to the analysis notebooks in the gallery.
- Cover how to add custom basin masks.
```
