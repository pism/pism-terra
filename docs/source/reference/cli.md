# Command-line interface

Every CLI is installed as a console script by `pip install -e .` and is also
runnable as `python -m <module>`. The full list lives in
`pyproject.toml`'s `[project.scripts]`.

## Glacier

```{list-table}
:header-rows: 1
:widths: 35 65

* - Command
  - Purpose
* - `pism-glacier-prepare`
  - Bootstrap a glacier study (RGI download, complex IDs, base inputs). Takes
    an optional list of glacier-ID CSVs to scope the run to a study area.
* - `pism-glacier-stage`
  - Stage all PISM inputs for one RGI ID.
* - `pism-glacier-run-forward`
  - Stage inputs and render a forward run script.
* - `pism-glacier-run-inverse`
  - Stage inputs and render an inverse run script (init → inversion → main
    run, see below).
* - `pism-glacier-execute`
  - Execute a pre-rendered run script.
* - `pism-glacier-postprocess`
  - Clip and aggregate spatial output (TOML-driven, legacy — the run scripts
    now emit `pism-postprocess-scalar` instead).
* - `pism-glacier-mip4`
  - Build the per-region `RGI7_NN` aggregate complexes and shared inputs for
    GlacierMIP4.
```

Both run commands take `RGI_ID CONFIG_FILE TEMPLATE_FILE` plus an *optional*
`UQ_FILE` positional: omit it for a single run, supply it to render one script
per ensemble member. `--data-path` points at a shared staging tree so several
experiments can reuse one staged copy of the inputs, while `--output-path`
keeps their outputs apart.

`pism-glacier-run-inverse` renders three PISM calls into one script:

1. the **init/prior** leg over `campaign.init_start`..`campaign.init_end` —
   required for an inverse run, since the inversion restarts from its state;
2. the **inversion** (`pismi`), writing the inverted `tauc` to
   `output/inverse/`;
3. the **main run** over the `[time]` span, restarting from the init state with
   that `tauc` regridded in and held fixed
   (`basal_yield_stress.model = "constant"`).

## Post-processing (all campaigns)

```{list-table}
:header-rows: 1
:widths: 35 65

* - Command
  - Purpose
* - `pism-postprocess-scalar`
  - Reduce a spatial file to per-region scalar sums.
* - `pism-postprocess-spatial`
  - Extract per-region masked spatial fields, one file per region.
* - `pism-validate`
  - Sanity-check a staging directory's NetCDFs.
```

## ISMIP7 Greenland

```{list-table}
:header-rows: 1
:widths: 35 65

* - Command
  - Purpose
* - `pism-ismip7-greenland-prepare`
  - One-time prep for ISMIP7 Greenland inputs (BedMachine, observations).
* - `pism-ismip7-greenland-add-basins`
  - Stamp the GrIS basin mask onto existing ocean forcing files without
    regenerating them.
* - `pism-ismip7-greenland-stage`
  - Stage inputs for a Greenland sub-domain.
* - `pism-ismip7-greenland-run-forward`
  - Render a forward run script (init leg + historical/projection legs).
* - `pism-ismip7-greenland-run-inverse`
  - Render an inverse run script (init → inversion → forward legs). See
    {doc}`../greenland/inversion` for choosing the Tikhonov penalty weight.
```

## KITP

```{list-table}
:header-rows: 1
:widths: 35 65

* - Command
  - Purpose
* - `pism-kitp-prepare`
  - Build the grid and stage common inputs for KITP.
* - `pism-kitp-stage`
  - Stage per-run inputs.
* - `pism-kitp-run`
  - Render the run script.
* - `pism-kitp-run-ensemble`
  - Render ensemble run scripts.
* - `pism-kitp-calibrate`
  - Calibrate KITP surface mass balance ensembles.
* - `pism-kitp-analyze`
  - Post-process KITP Greenland output into figures.
* - `pism-kitp-adjust-timeseries`
  - Trim spin-up from a KITP scalar timeseries and normalise it to its first
    year.
```

## Snow 4 Flow (S4F)

```{list-table}
:header-rows: 1
:widths: 35 65

* - Command
  - Purpose
* - `pism-s4f-planning`
  - Stage candidate S4F glaciers for mission planning.
```

## Tools

```{list-table}
:header-rows: 1
:widths: 35 65

* - Command
  - Purpose
* - `combine-crameri-colormaps`
  - Bundle Crameri colormaps for plotting.
```

```{admonition} TODO
- Add `--help` output verbatim for each command.
- Cross-link to the relevant Feature page per CLI.
```
