# Grids and domains

The PISM target grid is computed once from the requested RGI complex outline
and reused as the destination for every staged input.

The two relevant modules are:

- {py:mod}`pism_terra.domain` — generic grid construction:
  {py:func}`~pism_terra.domain.create_domain`,
  {py:func}`~pism_terra.domain.get_bounds_from_geometry`,
  {py:func}`~pism_terra.domain.new_range`.
- {py:mod}`pism_terra.grids` — packaged CDO grid descriptors for source data
  (e.g. CARRA2).

## Domain construction

`create_domain` returns an {py:class}`xarray.Dataset` with `x`, `y`, `x_bnds`,
`y_bnds`, and a CF-compliant `mapping` grid-mapping variable. Downstream
staging always reprojects with `rio.reproject_match(target_grid, …)` so every
variable lands on the same cells.

## CRS handling

- Aggregate complexes (S4F: `S4F_AK`, `S4F_CA`, `S4F_SV`) carry an explicit
  CRS in `rgi_c.gpkg`.
- Single-glacier complexes inherit a per-region UTM (or polar-stereo) CRS.

```{admonition} TODO
- Add an example showing how to build a domain from a custom polygon.
- Document expected grid resolutions per study area.
```
