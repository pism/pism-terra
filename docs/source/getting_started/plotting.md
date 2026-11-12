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


## Staging the data

Stage `RGI2000-v7.0-C-01-04374` with the Frank ice thickness

```bash
pism-glacier-stage \
    --output-path frank \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/docs_rgi_era5-mean_frank.toml
```
