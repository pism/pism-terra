# Glacial Cycle Simulations

`pism-greenland-paleo-{prepare,stage,run}` set up simulations of the Greenland Ice Sheet
over the last glacial cycle, from 125,000 years before present to today, with the age of
the ice and isochrones tracked along the way. The workflow follows the
[`run_glacial_cycle.py`](https://github.com/pism/pism-greenland/blob/master/paleo/run_glacial_cycle.py)
script of pism-greenland and reuses the ISMIP7 Greenland inputs: the same domain, the boot
and heat-flux files derived from GreenlandObsISMIP7, and the same initial state.

## Forcing

Base climate
: The 1960-1989 monthly mean of the OCX air temperature (`tas`) and precipitation (`pr`),
  cycled for the whole run. The surface mass balance is computed with the positive
  degree-day model.

Temperature anomaly
: The GRIP ice-core series of the SeaRISE Greenland data set (`pism_dT.nc`). It shifts the
  air temperature (`delta_T`) and scales precipitation (`precip_scaling`); air temperature
  also follows the changing surface elevation (`elevation_change`).

Sea level
: The SPECMAP series of the same data set (`pism_dSL.nc`), applied with
  `-sea_level constant,delta_sl`.

Ocean
: The `th` model on the 1960-1989 monthly mean of the OCX thermal forcing and salinity.
  `pism_ocean_dT.nc` holds the air-temperature anomaly scaled by `ocean_delta_T_scale` and
  is applied with the `delta_T` ocean modifier.

```{note}
PISM's ocean `delta_T` modifier offsets the temperature at the base of the ice shelves. It
does not change the ocean temperature the `th` model computes melt from, so sub-shelf melt
follows the present-day climatology through the whole cycle. Select the `th` option table
(`[ocean] model = "th"`) to leave the modifier out.
```

## Prepare

```bash
pism-greenland-paleo-prepare pism_terra/config/setup_greenland_paleo.toml paleo_input
```

This writes the SeaRISE series and the two climatologies to `paleo_input/input/` and
prints the `aws s3 sync` command that uploads them. The base period, the OCX fields and
the ocean scaling are set in `setup_greenland_paleo.toml`. Restrict the work with
`--include searise` or `--include climatology`; the grid, boot and heat-flux files are
taken from the ISMIP7 inputs and are only rebuilt with `--include grid,observations`.

## Stage and run

```bash
pism-greenland-paleo-run --output-path 2026_10_paleo --resolution 4500m \
    pism_terra/config/greenland_paleo.toml pism_terra/templates/chinook-ismip7.j2
```

Staging is part of the run command; `pism-greenland-paleo-stage` does it alone. The ISMIP7
files come from `campaign.shared_prefix`, the paleo files from
`campaign.prefix`/`campaign.version`.

A run is one PISM invocation that bootstraps from the boot file and takes the thermal
state from the initial-state file. `--start` and `--end` (years, negative before present)
shorten it, and a UQ file as a third argument renders an ensemble:

```bash
pism-greenland-paleo-run --output-path 2026_10_paleo_uq --resolution 9000m \
    pism_terra/config/greenland_paleo.toml pism_terra/templates/chinook-ismip7.j2 \
    pism_terra/uq/greenland_paleo.toml
```

## Age and isochrones

`greenland_paleo.toml` turns on the age model and deposits a new isochronal layer every 200
years (`[age]` and `[isochrones]`). `isochrone_depth` and `isochronal_layer_thickness` are
written to the spatial file every 100 years, and full snapshots are saved at selected times
between 120 ka and present (`output.snapshot.times`).
