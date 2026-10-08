# Velocities

Observed surface velocities come from ITS_LIVE v2.1 per-region COGs hosted on
S3 — handled in {py:mod}`pism_terra.glacier.observations`.

## Per-region COG selection

The v2 "global" mosaic is in EPSG:3857, which means its scale factor varies
with latitude and breaks the finite-difference projection round-trip used to
align vectors to the model CRS. pism-terra avoids that by using the per-RGI
v2.1 COGs (native polar-stereographic) and selecting the right one
automatically:

- {py:func}`~pism_terra.glacier.observations.region_code_from_rgi_id` reads the
  o1 region straight out of the glacier's ID (`RGI2000-v7.0-C-03-01124` →
  `03`). This is the authoritative answer and is tried first.
- {py:func}`~pism_terra.glacier.observations.region_code_from_bounds` is the
  fallback when no ID is available: it probes each published region COG header
  (one cached `/vsicurl/` open per region) and returns the code covering most
  of the requested bbox.
- {py:func}`~pism_terra.glacier.observations.get_itslive_velocities_by_region_code`
  loads the components for that region.

`13`, `15`, and `16` are not published as standalone regions — they're rolled
into the High-Mountain-Asia mosaic (`14`).

```{admonition} Why the RGI ID wins over geometry
:class: note

The per-region footprints overlap generously in polar stereographic. A domain
over northern Ellesmere Island falls inside the Arctic Canada (`03`) *and* the
Greenland (`05`) footprints by identical amounts, so geometry alone cannot
choose — but only `03` holds data there.
```

Full coverage of the requested bbox is preferred, not required. A staged domain
is the glacier plus a buffer plus a geographic pad, so it routinely overhangs a
tile edge by a few kilometres while the glacier itself sits well inside; the
overhang is interpolated to NaN and zeroed like any other ITS_LIVE gap. The
shortfall is logged, so a domain that genuinely falls outside coverage is still
visible.

## Boot-file velocity field

{py:func}`~pism_terra.glacier.observations.glacier_velocities_from_grid`
computes `vx`/`vy` on the target grid by finite-differencing the trajectory
of points advected through ITS_LIVE in the source CRS — this preserves vector
orientation across the source→target reprojection.

```{admonition} TODO
- Cover the `zeta_fixed_mask` and `vel_misfit_weight` outputs.
- Explain the `landice` mask interpretation.
```
