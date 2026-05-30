# Workflow guides

End-to-end recipes for the common pism-terra workflows.

## Single glacier

See {doc}`../getting_started/quick_start` for the minimal version. Inputs and
config are independent of any other glacier or campaign.

## Ensemble (Latin Hypercube)

Combine a UQ TOML (see {doc}`../features/uncertainty_quantification`) with
`pism-glacier-run-ensemble`. The output is one rendered run script per
member; HPC submission is up to the user.

## Snow 4 Flow (S4F) aggregates

S4F campaigns (`S4F_AK`, `S4F_CA`, `S4F_SV`) bundle many RGI v7 complexes
into a single aggregate run. CSV target files under `pism_terra/config/`
list the constituent glaciers; `pism-s4f-prepare` builds the union outlines,
per-group CARRA2 caches, and aggregate ice-thickness mosaics.

## ISMIP7 Greenland

`pism-ismip7-greenland-{prepare,stage,run,run-ensemble,postprocess}` drive
the ISMIP7 community simulations. Inputs come from BedMachine and the
GreenlandObsISMIP7 dataset.

## KITP

`pism-kitp-{prepare,stage,run,run-ensemble,postprocess,writer}` cover the
Kaskawulsh / Tuktut / Paxson studies, including the
`pism-kitp-writer` async-writer side process.

```{admonition} TODO
- Walk each workflow end-to-end with timings.
- Cite the relevant papers per study area.
```
