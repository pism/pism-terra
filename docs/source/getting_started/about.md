# About pism-terra

pism-terra is a Python package that automates running
[PISM](https://www.pism.io) — the Parallel Ice Sheet Model — for any glacier
complex in the world. It is keyed on
[Randolph Glacier Inventory v7](https://www.glims.org/RGI/) IDs and orchestrates
the steps that would otherwise be hand-rolled per study:

1. **Staging** — downloads and conditions input data (DEMs, ice thickness,
   climate, velocities, bathymetry) for the requested glacier or aggregate.
2. **Configuration** — Pydantic-validated TOML configs combined with Jinja2
   HPC job templates produce ready-to-submit run scripts.
3. **Ensembles** — Latin Hypercube sampling over SciPy distributions generates
   parameter ensembles for uncertainty quantification.
4. **Post-processing** — clipping to glacier outlines, aggregating to scalar
   diagnostics, and exporting tidy NetCDFs for downstream analysis.

## What problem it solves

PISM is a powerful but ungentle tool: it expects pre-projected NetCDF inputs
on a specific grid, with the right CF metadata, the right CRS, the right
units, in the right order. Building those inputs for hundreds of glaciers, or
across a handful of campaigns, is repetitive and error-prone. pism-terra
turns that whole pipeline into a sequence of CLI calls driven by
human-readable TOML.

## Where it is used

- Single-glacier studies (e.g. Wrangell, Kaskawulsh, Storstrømmen).
- Multi-glacier "Snow 4 Flow" (S4F) campaigns across AK, CA, SV.
- ISMIP7 Greenland community simulations.
- KITP (Kaskawulsh / Tuktut / Paxson) calibration and forecasting.

```{admonition} TODO
- Add a "Scope and non-goals" subsection.
- Cite at least one paper that used pism-terra.
- Cross-link to {doc}`../resources/workflows` once it's written.
```
