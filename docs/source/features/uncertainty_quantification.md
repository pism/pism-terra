# Uncertainty quantification

Ensemble runs are driven by a separate UQ TOML that names parameters to vary
and the SciPy distributions they're drawn from.

## UQ config schema

The top-level model is {py:class}`pism_terra.config.UQConfig`. Each entry is
a {py:class}`~pism_terra.config.DistSpec` with `distribution` plus distribution
parameters (`loc`, `scale`, `a`, `b`, …):

```toml
['surface.pdd.factor_ice']
loc = 0.008
scale = 0.004
distribution = "truncnorm"
a = -2
b = 2
```

## Sampling

{py:func}`pism_terra.sampling.lhs_sample` performs Latin Hypercube sampling and
inverse-transforms each unit-cube column through its declared distribution.

## Generating an ensemble

```bash
pism-glacier-run-ensemble \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/rgi_g.gpkg \
    pism_terra/config/rgi_init_maffezzoli.toml \
    pism_terra/templates/debug.j2 \
    pism_terra/uq/debm.toml
```

This produces one run script per ensemble member.

```{admonition} TODO
- Document discrete-distribution support (`randint`, …).
- Cover correlated/conditioned distributions if/when supported.
- Show how to inspect the realised sample with pandas.
```
