# Model description paper

This section holds the case studies of the pism-terra model description paper,
written for [Geoscientific Model Development](https://www.geoscientific-model-development.net/).
Every figure in the paper is made by a page in this section, so the paper and
the documentation cannot drift apart.

## How the case studies are reproduced

Each case study comes in two tiers.

Run
: The simulations themselves: the configuration file, checked into the package
  under `pism_terra/config/`, and the commands that stage the inputs, render the
  run script and run it. A run takes hours to days on many cores, so it is done
  once, on an HPC system or PISM-Cloud, and its small postprocessed products are
  archived.

Figure
: The analysis that turns those products (time series, basin sums, the change of
  a field between two dates) into the paper's figures and numbers. It runs on a
  laptop in seconds to minutes, and the pages below show its code.

The figure tier reads its data from the directory named by the environment
variable `PISM_TERRA_PAPER_DATA`, one subdirectory per case study.

```{admonition} To do before submission
:class: warning

- Archive the products of every case study with a DOI (Zenodo), and fetch them
  in the pages with `pooch` instead of `PISM_TERRA_PAPER_DATA`.
- Turn on execution for this section in `conf.py`, so that every documentation
  build makes the figures.
- Tag the pism-terra release the paper describes and archive it (Zenodo), and
  record the PISM commit and container image the runs used.
```

## Case studies

```{toctree}
:maxdepth: 1

case_study_1_glacier
```
