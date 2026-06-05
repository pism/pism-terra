---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
mystnb:
  execution_mode: force
---

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
```

# Inverse Modeling

## Staging the data

Stage `RGI2000-v7.0-C-01-04374` with the Frank ice thickness

```bash
pism-glacier-stage \
    --output-path frank_inverse \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/rgi_inverse_gpbld_frank.toml
```

and with the Maffezzoli ice thickness

```bash
pism-glacier-stage \
    --output-path maffezzoli_inverse \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/rgi_inverse_gpbld_maffezzoli.toml
```

## Running the inverse model

Prepare the run script for the Frank dataset

```bash
pism-glacier-inverse \
    --output-path frank_inverse \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/rgi_inverse_gpbld_frank.toml \
    pism_terra/templates/debug-inverse.j2
```

and then for the Maffezzoli dataset:

```bash
pism-glacier-inverse \
    --output-path maffezzoli_inverse \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/rgi_inverse_gpbld_maffezzoli.toml \
    pism_terra/templates/debug-inverse.j2
```

Now you can run both models. It will first to a forward simulation to prepare an initial state and then run the inverse model.

```bash
. frank_inverse/RGI2000-v7.0-C-01-04374/run_scripts/submit_g500m_RGI2000-v7.0-C-01-04374_id_0_0001-01-01_0021-02-01.sh
```

```bash
. maffezzoli_inverse/RGI2000-v7.0-C-01-04374/run_scripts/submit_g500m_RGI2000-v7.0-C-01-04374_id_0_0001-01-01_0021-02-01.sh
```



