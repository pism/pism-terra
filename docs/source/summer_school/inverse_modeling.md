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

and with the Millan ice thickness

```bash
pism-glacier-stage \
    --output-path millan_inverse \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/rgi_inverse_gpbld_millan.toml
```

## Running the inverse model

`pism-glacier-run-inverse` renders one script holding three PISM calls:

1. an **init** run, which bootstraps and spins the glacier up over
   `campaign.init_start`..`campaign.init_end`;
2. the **inversion** (`pismi`), which restarts from that state and solves for
   the basal yield stress `tauc`, writing it to `output/inverse/`;
3. the **main run** over the `[time]` span, restarting from the init state
   with the inverted `tauc` regridded in and held fixed
   (`basal_yield_stress.model = "constant"`).

The init bounds are what leg 2 inverts on, so the config has to declare them.
The three configs below spin up for five model years, invert on that state, and
then run one year with the inverted `tauc`:

```toml
[campaign]
init_start = "0001-01-01"
init_end = "0006-01-01"

[time]
'time.start' = "0006-01-01"
'time.end' =  "0007-01-01"
```

Prepare the run script for the Frank dataset

```bash
pism-glacier-run-inverse \
    --output-path frank_inverse \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/rgi_inverse_gpbld_frank.toml \
    pism_terra/templates/debug-inverse.j2
```

and then for the Maffezzoli dataset:

```bash
pism-glacier-run-inverse \
    --output-path maffezzoli_inverse \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/rgi_inverse_gpbld_maffezzoli.toml \
    pism_terra/templates/debug-inverse.j2
```

and finally for the Millan dataset:

```bash
pism-glacier-run-inverse \
    --output-path millan_inverse \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/rgi_inverse_gpbld_millan.toml \
    pism_terra/templates/debug-inverse.j2
```

Now you can run the models. Each script spins the glacier up, inverts for
`tauc`, and then runs forward with the inverted field. The script's name
carries the resolution, the RGI ID and the `[time]` span, so let the shell find
it rather than typing it out:

```bash
. frank_inverse/RGI2000-v7.0-C-01-04374/run_scripts/submit_*.sh
```

```bash
. maffezzoli_inverse/RGI2000-v7.0-C-01-04374/run_scripts/submit_*.sh
```

```bash
. millan_inverse/RGI2000-v7.0-C-01-04374/run_scripts/submit_*.sh
```

The inversion writes one `output/inverse/inv_*.nc` per glacier; that file is
what the three thickness datasets are compared through.
