# Citing and method overview

If pism-terra contributes to a publication, please cite both the package and
the underlying PISM model.

## Citing pism-terra

```{admonition} TODO
- Add a Zenodo DOI for the package.
- Provide a BibTeX entry.
```

## Citing PISM

PISM should be cited following the guidance on
[the PISM project website](https://www.pism.io/docs/index.html#citing-pism).

## Method overview

pism-terra is a thin orchestration layer; the science is done by PISM. The
package's contribution is in:

- **Reproducible input staging** — every NetCDF written has a documented
  source, CRS, and units provenance.
- **Configuration-as-data** — runs are fully described by TOML files committed
  alongside results.
- **Ensemble generation** — Latin Hypercube sampling over user-declared
  parameter distributions enables consistent uncertainty studies.
- **Post-processing on the model grid** — basin clips and scalar aggregations
  are computed against PISM's native grid, avoiding regrid-and-lose-volume
  artefacts.

```{admonition} TODO
- Diagram of the staging → run → post-process pipeline.
- Link to the relevant PISM physics chapters per subsystem.
```
