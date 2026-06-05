# Quick start

A minimal end-to-end run for a single RGI v7 complex.

## 1. Pick a glacier

Use any RGI v7 complex ID. Wrangell glacier (Alaska) is a good first target:

```text
RGI2000-v7.0-C-01-04374
```

## 2. Stage inputs

Staging downloads the DEM, ice thickness, climate forcing, and observed
velocities, then projects everything onto the run's target grid.

```bash
pism-glacier-stage \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/rgi_era5_maffezzoli_1year.toml
```

You'll get a `<RGI_ID>/input/` directory with one NetCDF per input.

## 3. Generate the run script

```bash
pism-glacier-run \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/rgi_era5_maffezzoli_1year.toml \
    pism_terra/templates/debug.j2
```

This produces a shell script under `<RGI_ID>/run/` that you can launch
directly (debug template) or submit to a scheduler (Slurm / PBS templates).

## 4. Post-process

After the run finishes:

```bash
pism-glacier-postprocess \
    <RGI_ID>/output/post_processing/g500m_RGI2000-v7.0-C-01-04374_id_0_1978-01-01_1979-01-01.toml
```

You end up with basin-clipped spatial NetCDFs and scalar-aggregated time
series.

## What's next

- {doc}`../features/staging` — what staging actually does and how to customise it.
- {doc}`../features/uncertainty_quantification` — turn the single run into a
  Latin-Hypercube ensemble.
- {doc}`../examples/index` — runnable example notebooks.

```{admonition} TODO
- Verify the exact commands against a fresh checkout.
- Add expected runtimes per step.
- Screenshot the produced figures.
```
