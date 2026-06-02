```{image} _static/pism_logo_transp.png
:alt: pism-terra
:width: 240px
:align: left
:class: only-light
```

**pism-terra** automates the staging, configuration, and execution of glacier
simulations with [PISM](https://www.pism.io) (the Parallel Ice Sheet Model) for
any glacier complex in the world, keyed off
[RGI v7](https://www.glims.org/RGI/) glacier IDs. It wraps the messy parts:
DEM stitching, ice-thickness retrieval, climate forcing, ITS_LIVE velocity
mosaics, Pydantic-validated TOML configuration, Latin Hypercube uncertainty
quantification, Jinja2 HPC job templates, and post-processing.

The structure of documentation was inspired by Romain Hugonnet [xDEM](https://xdem.readthedocs.io) 

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Quick start
:link: getting_started/quick_start
:link-type: doc

Stage and run one glacier end to end.
:::

:::{grid-item-card} {octicon}`book` Features
:link: features/staging
:link-type: doc

Tour the subsystems: staging, climate, thickness, UQ.
:::

:::{grid-item-card} {octicon}`code-square` API reference
:link: reference/api
:link-type: doc

Every public function, grouped by subpackage.
:::
::::

```{toctree}
:caption: Getting started
:hidden:

getting_started/about
getting_started/installation
getting_started/quick_start
getting_started/citing
```

```{toctree}
:caption: Features
:hidden:

features/staging
features/grids_domain
features/climate_forcing
features/ice_thickness
features/velocities
features/run_configuration
features/uncertainty_quantification
features/postprocessing
```


```{toctree}
:caption: Summer School in Glaciology 2026
:hidden:

summer_school/getting_started
summer_school/forward_modeling
summer_school/inverse_modeling
```

```{toctree}
:caption: Resources
:hidden:

resources/workflows
resources/cheatsheet
resources/ecosystem
```

```{toctree}
:caption: Gallery of examples
:hidden:

examples/index
```

```{toctree}
:caption: Reference
:hidden:

reference/api
reference/cli
reference/configuration
reference/release_notes
```

```{toctree}
:caption: Project information
:hidden:

project/publications
project/credits
references
```
