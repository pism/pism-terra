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

# Forward Modeling

For the Wrangell Mountain Glacier Complex (RGI2000-v7.0-C-01-04374), we will
assess the role of ice thickness. Ice thickness exerts a first order control
on ice flow. The complex flow patterns in Greenland's outlet glaciers cannot
be reproduced *for the right reason* without accurate ice thickness {cite}`Aschwanden2016`.
Two global ice thickness products are currently supported: Frank et al and
Maffezzoli et al.; see {cite}`Frank2026` and {cite}`Maffezzoli2025`.


## Staging the data

Stage `RGI2000-v7.0-C-01-04374` with the Frank ice thickness

```bash
pism-glacier-stage \
    --output-path frank \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/rgi_era5-mean_frank.toml
```

and with the Maffezzoli ice thickness

```bash
pism-glacier-stage \
    --output-path maffezzoli \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/rgi_era5-mean_maffezzoli.toml
```

## Compare input ice thicknesses

The version of the cell below is what you would run on your *own* checkout
after staging both datasets. ``skip-execution`` tells myst-nb to render the
source verbatim but never run it, so the docs build doesn't depend on having
the staged data on disk.

```{code-cell} ipython3
:tags: [skip-execution]

import xarray as xr
import matplotlib.pyplot as plt

from pism_terra.colormaps import register_colormaps
register_colormaps()

frank_ds = xr.open_dataset("frank/RGI2000-v7.0-C-01-04374/input/bootfile_RGI2000-v7.0-C-01-04374.nc")
frank_thickness = frank_ds.thickness.where(frank_ds.thickness > 0)
maffezzoli_ds = xr.open_dataset("maffezzoli/RGI2000-v7.0-C-01-04374/input/bootfile_RGI2000-v7.0-C-01-04374.nc")
maffezzoli_thickness = maffezzoli_ds.thickness.where(maffezzoli_ds.thickness > 0)

diff_thickness = frank_thickness - maffezzoli_thickness

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 16))
frank_thickness.plot(ax=axs[0], vmin=0, vmax=1000)
maffezzoli_thickness.plot(ax=axs[1], vmin=0, vmax=1000)
diff_thickness.plot(ax=axs[2], cmap="RdBu", vmin=-250, vmax=250)
axs[0].set_title("Frank ice thickness")
axs[1].set_title("Maffezzoli ice thickness")
axs[2].set_title("Ice Thickness (EFrank - Maffezzoli)")
```

```{code-cell} ipython3
:tags: [remove-input]

# Hidden twin of the cell above — runs against the small bundled fixtures
# under ``docs/source/_data/`` so the build produces a real figure without
# requiring the user's full staged dataset on disk. myst-nb's cwd is the
# page's directory (``docs/source/summer_school/``), so ``_data/`` is one
# level up.
import xarray as xr
import matplotlib.pyplot as plt

from pism_terra.colormaps import register_colormaps
register_colormaps()

frank_ds = xr.open_dataset("../_data/frank_thickness.nc")
frank_thickness = frank_ds.thickness.where(frank_ds.thickness > 0)
maffezzoli_ds = xr.open_dataset("../_data/maffezzoli_thickness.nc")
maffezzoli_thickness = maffezzoli_ds.thickness.where(maffezzoli_ds.thickness > 0)

diff_thickness = frank_thickness - maffezzoli_thickness

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 16))
frank_thickness.plot(ax=axs[0], vmin=0, vmax=1000)
maffezzoli_thickness.plot(ax=axs[1], vmin=0, vmax=1000)
diff_thickness.plot(ax=axs[2], cmap="RdBu", vmin=-250, vmax=250)
axs[0].set_title("Frank ice thickness")
axs[1].set_title("Maffezzoli ice thickness")
axs[2].set_title("Ice Thickness (Frank - Maffezzoli)")
```

Differences between the two ice thickenss datasets are substantial. Let's get to work.

## Generate run scripts

Generate the run scripts for your local machine using the `debug.j2` template, first for the Frank dataset

```bash
pism-glacier-run \
    --output-path frank \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/rgi_era5-mean_frank.toml \
    pism_terra/templates/debug.j2
```

and then for the Maffezzoli dataset:

```bash
pism-glacier-run \
    --output-path maffezzoli \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/rgi_era5-mean_maffezzoli.toml \
    pism_terra/templates/debug.j2
```

## Run the scripts

Now you can execute the run script

```bash
. frank/RGI2000-v7.0-C-01-04374/run_scripts/submit_g400m_RGI2000-v7.0-C-01-04374_id_0_1980-01-01_1985-01-01.sh
```

This takes about 30min on a M2 Macbook Pro with 8 cores. Now postprocess with

```bash
pism-glacier-postprocess \
    frank/RGI2000-v7.0-C-01-04374/output/post_processing/g500m_RGI2000-v7.0-C-01-04374_id_0_0001-01-01_0006-01-01.toml
```

Do the same with the Maffezzoli dataset:

```bash
. maffezzoli/RGI2000-v7.0-C-01-04374/run_scripts/submit_g400m_RGI2000-v7.0-C-01-04374_id_0_1980-01-01_1985-01-01.sh

pism-glacier-postprocess \
    maffezzoli/RGI2000-v7.0-C-01-04374/output/post_processing/g500m_RGI2000-v7.0-C-01-04374_id_0_0001-01-01_0006-01-01.toml
```

## Compare surface speeds

```{code-cell} ipython3
:tags: [skip-execution]

time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
delta_coder = xr.coders.CFTimedeltaCoder()

frank_state = xr.open_dataset("frank/RGI2000-v7.0-C-01-04374/output/state/clipped_state_g500m_RGI2000-v7.0-C-01-04374_id_0_0001-01-01_0006-01-01.nc",
                              decode_times=time_coder,
                              decode_timedelta=delta_coder,
                             ).squeeze()
frank_speed = frank_state.velsurf_mag
maffezzoli_state = xr.open_dataset("maffezzoli/RGI2000-v7.0-C-01-04374/output/state/clipped_state_g500m_RGI2000-v7.0-C-01-04374_id_0_0001-01-01_0006-01-01.nc",
                              decode_times=time_coder,
                              decode_timedelta=delta_coder,
                                  ).squeeze()
maffezzoli_speed = maffezzoli_state.velsurf_mag

diff_speed = frank_speed - maffezzoli_speed

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 16))
frank_speed.plot(ax=axs[0], cmap="speed_colorblind", vmin=0, vmax=1000)
maffezzoli_speed.plot(ax=axs[1], cmap="speed_colorblind" , vmin=0, vmax=1000)
diff_speed.plot(ax=axs[2], cmap="RdBu", vmin=-250, vmax=250)
axs[0].set_title("Frank surface speed")
axs[1].set_title("Maffezzoli surface speed")
axs[2].set_title("Speed (Frank - Maffezzoli)")
```

```{code-cell} ipython3
:tags: [remove-input]

# Hidden twin of the cell above — runs against the small bundled fixtures
# under ``docs/source/_data/`` so the build produces a real figure without
# requiring the user's full staged dataset on disk. myst-nb's cwd is the
# page's directory (``docs/source/summer_school/``), so ``_data/`` is one
# level up.
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
delta_coder = xr.coders.CFTimedeltaCoder()

frank_state = xr.open_dataset("../_data/frank_speed.nc",
                              decode_times=time_coder,
                              decode_timedelta=delta_coder,
                             ).squeeze()
frank_speed = frank_state.velsurf_mag
maffezzoli_state = xr.open_dataset("../_data/maffezzoli_speed.nc",
                              decode_times=time_coder,
                              decode_timedelta=delta_coder,
                                  ).squeeze()
maffezzoli_speed = maffezzoli_state.velsurf_mag

diff_speed = frank_speed - maffezzoli_speed

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 16))
frank_speed.plot(ax=axs[0], cmap="speed_colorblind", vmin=0, vmax=1000)
maffezzoli_speed.plot(ax=axs[1], cmap="speed_colorblind" , vmin=0, vmax=1000)
diff_speed.plot(ax=axs[2], cmap="RdBu", vmin=-250, vmax=250)
axs[0].set_title("Frank surface speed")
axs[1].set_title("Maffezzoli surface speed")
axs[2].set_title("Speed(Frank - Maffezzoli)")
```

Unsurprisingly, the differences in simulated surface speeds are huge.

```{admonition} TODO
- Verify the exact commands against a fresh checkout.
- Add expected runtimes per step.
- Screenshot the produced figures.
```
