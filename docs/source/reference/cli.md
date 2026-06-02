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
  - Bootstrap a glacier study (RGI download, complex IDs, base inputs).
* - `pism-glacier-stage`
  - Stage all PISM inputs for one RGI ID.
* - `pism-glacier-run`
  - Render the run script for a single glacier.
* - `pism-glacier-run-ensemble`
  - Render run scripts for an ensemble of UQ samples.
* - `pism-glacier-execute`
  - Execute a pre-rendered run script.
* - `pism-glacier-postprocess`
  - Clip and aggregate spatial output.
```

## ISMIP7 Greenland

```{list-table}
:header-rows: 1
:widths: 35 65

* - Command
  - Purpose
* - `pism-ismip7-greenland-prepare`
  - One-time prep for ISMIP7 Greenland inputs (BedMachine, observations).
* - `pism-ismip7-greenland-stage`
  - Stage inputs for a Greenland sub-domain.
* - `pism-ismip7-greenland-run`
  - Render the run script.
* - `pism-ismip7-greenland-run-ensemble`
  - Render ensemble run scripts.
* - `pism-ismip7-greenland-postprocess`
  - Post-process Greenland output.
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
* - `pism-kitp-postprocess`
  - Clip to basin and aggregate scalar diagnostics.
* - `pism-kitp-writer`
  - Async writer side process for KITP.
```

## Snow 4 Flow (S4F)

```{list-table}
:header-rows: 1
:widths: 35 65

* - Command
  - Purpose
* - `pism-s4f-prepare`
  - Build aggregate complexes (S4F_AK, S4F_CA, S4F_SV) and per-group inputs.
* - `pism-s4f-planning`
  - Plan S4F campaigns.
```

## Tools

```{list-table}
:header-rows: 1
:widths: 35 65

* - Command
  - Purpose
* - `pism-validate`
  - Sanity-check a staging directory's NetCDFs.
* - `combine-crameri-colormaps`
  - Bundle Crameri colormaps for plotting.
```

```{admonition} TODO
- Add `--help` output verbatim for each command.
- Cross-link to the relevant Feature page per CLI.
```
