---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Case study 1: a glacier complex

A hindcast of one RGI v7 glacier complex, the Wrangell Mountains in Alaska
(`RGI2000-v7.0-C-01-04374`, 4129 km²), from 1986 to 2025. The modelled
change in surface elevation over 2000 to 2020 is compared with the observed
change of {cite:t}`hugonnet2021`.

The case study shows the single-glacier workflow end to end: inputs are
staged from nothing but the RGI ID, the basal shear stress is inverted from
observed surface velocities, and the forward run is forced with ERA5.

## Run

The configuration is
[`pism_terra/config/gmd_case_study_1_glacier.toml`](https://github.com/pism/pism-terra/blob/main/pism_terra/config/gmd_case_study_1_glacier.toml):
enthalpy, a Blatter–Pattyn stress balance, a positive-degree-day surface mass
balance driven by ERA5, on a 200 m grid.

Stage the inputs (surface elevation, ice thickness, climate, velocities,
observed elevation change) for the glacier complex:

```bash
pism-glacier-stage RGI2000-v7.0-C-01-04374 pism_terra/config/gmd_case_study_1_glacier.toml
```

Render the run script, which inverts for the basal shear stress and then runs
forward from 1986 to 2025, and postprocesses the output:

```bash
pism-glacier-run-inverse \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/gmd_case_study_1_glacier.toml \
    pism_terra/templates/debug.j2
```

The script ends with the two postprocessing steps the figures need:
`pism-postprocess-scalar` reduces the spatial output to time series over the
glacier complex, and `pism-glacier-postprocess-dh --vars usurf --start
2000-01-01 --end 2020-01-01` writes the modelled change in surface elevation.

The figure tier needs three files, kept in `$PISM_TERRA_PAPER_DATA/case_study_1/`
with the run's directory layout:

| File | From |
|---|---|
| `output/dh/dh_<RGI_ID>_id_0_2000-01-01_2020-01-01.nc` | `pism-glacier-postprocess-dh` |
| `output/processed_scalar/scalar_C_g200m_<RGI_ID>_id_0_1986-01-01_2025-01-01.nc` | `pism-postprocess-scalar` |
| `input/obs_<RGI_ID>.nc` | `pism-glacier-stage` (observed elevation change) |

## Figures

```{code-cell} ipython3
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pism_terra.plotting import rc_params

RGI_ID = "RGI2000-v7.0-C-01-04374"
DATA = Path(os.environ.get("PISM_TERRA_PAPER_DATA", "paper_data")).expanduser() / "case_study_1"
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

#: Density used to turn a volume change into a mass change (Huss, 2013).
VOLUME_TO_MASS_DENSITY = 850.0  # kg m-3
YEARS = 20.0  # 2000 to 2020

modelled = xr.open_dataset(DATA / f"output/dh/dh_{RGI_ID}_id_0_2000-01-01_2020-01-01.nc")["usurf"].squeeze("time", drop=True)
observed = xr.open_dataset(DATA / f"input/obs_{RGI_ID}.nc")[["dh", "dh_err"]]
```

The observations come on the 100 m grid of the staged inputs, the model on its
200 m grid, whose cells are made of 2 × 2 observation cells; the observations
are averaged onto the model grid.

```{code-cell} ipython3
observed = observed.coarsen(x=2, y=2).mean().assign_coords(x=modelled["x"], y=modelled["y"])
on_ice = observed["dh"].notnull()
difference = (modelled - observed["dh"]).where(on_ice)


def geodetic_mass_balance(dh: xr.DataArray) -> float:
    """Area-averaged elevation change as a mass balance, m w.e. per year."""
    return float(dh.where(on_ice).mean()) / YEARS * VOLUME_TO_MASS_DENSITY / 1000.0


print(f"observed {geodetic_mass_balance(observed['dh']):+.2f} m w.e./yr, "
      f"modelled {geodetic_mass_balance(modelled):+.2f} m w.e./yr (2000-2020)")
```

```{code-cell} ipython3
panels = [
    (observed["dh"], "Observed (Hugonnet et al., 2021)", "RdBu"),
    (modelled.where(on_ice), "Modelled", "RdBu"),
    (difference, "Modelled − observed", "PuOr"),
]
with plt.rc_context(rc_params):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True, sharey=True, layout="constrained")
    images = []
    for ax, (field, title, cmap) in zip(axs, panels):
        images.append(field.plot.imshow(ax=ax, cmap=cmap, vmin=-60, vmax=60, add_colorbar=False, rasterized=True))
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_axis_off()
    # One colorbar per colour scale: elevation change, and the model's misfit.
    fig.colorbar(images[0], ax=axs[:2], shrink=0.8, orientation="horizontal", label="Elevation change 2000–2020 (m)")
    fig.colorbar(images[2], ax=axs[2], shrink=0.8, orientation="horizontal", label="Difference (m)")
    fig.savefig(FIGURES / "case_study_1_dh.png", dpi=300)
```

The mass budget of the glacier complex over the hindcast, from the time series
of the scalar postprocessing:

```{code-cell} ipython3
from pism_terra.processing import integrate_rate

scalar = xr.open_dataset(DATA / f"output/processed_scalar/scalar_C_g200m_{RGI_ID}_id_0_1986-01-01_2025-01-01.nc")
scalar = scalar.squeeze("glacier_id", drop=True)
terms = {
    "tendency_of_ice_mass_due_to_surface_mass_flux": "Surface mass balance",
    "tendency_of_ice_mass_due_to_discharge": "Discharge",
    "tendency_of_ice_mass": "Total",
}
with plt.rc_context(rc_params):
    fig, ax = plt.subplots(figsize=(4.8, 2.6), layout="constrained")
    for name, label in terms.items():
        cumulative = integrate_rate(scalar[name], to="Gt").pint.dequantify()
        ax.plot(cumulative["time"], cumulative, label=label, lw=1.0)
    ax.axhline(0.0, color="k", lw=0.4, ls="dotted")
    ax.set_ylabel("Cumulative mass change since 1986 (Gt)")
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "case_study_1_mass_budget.png", dpi=300)
```
