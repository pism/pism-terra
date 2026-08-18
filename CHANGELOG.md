# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Glacier input preparation is one command again. `pism_terra/glacier/prepare.py` carried two near-identical ~200-line functions, `rgi()` and `s4f()`, that had already drifted apart (different step order, different outline loading, a lost comment); they are now a single `prepare()`. **Removed:** `pism-s4f-prepare` — use `pism-glacier-prepare` and pass the glacier-ID CSVs as trailing arguments:

  ```bash
  pism-glacier-prepare pism_terra/config/setup_rgi.toml glacier_input
  pism-glacier-prepare pism_terra/config/setup_s4f.toml glacier_input \
      pism_terra/config/S4F_target_*.csv
  ```

  With no CSVs the run covers whole regions, exactly as `pism-glacier-prepare` did before. `--include glaciermip4` is now accepted for every project (it used to be an S4F-only dataset name).
- Prepared inputs are split by project instead of sharing one `glacier/` tree. The setup TOML declares `[staging] project_directory = "rgi"` (or `"s4f"`), and the output layout becomes:

  ```text
  <OUTPUT_PATH>/input/{gebco,heatflux,climate}/     # global — written once
  <OUTPUT_PATH>/input/<project>/rgi/<project>_{c,g}.gpkg
  <OUTPUT_PATH>/input/<project>/ice_thickness/{frank,maffezzoli}/
  <OUTPUT_PATH>/input/<project>/climate/carra2_<group>.nc
  ```

  The split follows the `[regions]` `crs` overrides. Those change the *contents* of the RGI GeoPackages and the ice-thickness rasters without changing their paths, so an S4F run and an RGI run previously overwrote each other's `ice_thickness/frank/RGI2000-v7.0-C-01/*_thickness.tif` and `rgi/rgi_c.gpkg`. Everything that does not depend on a CRS override — GEBCO, heat flux, SNAP, the merged `carra2.zarr` — is now stored once rather than per project, and `staging/` stays shared so the raw downloads are not fetched twice.
  **Migration:** campaign configs gain `project_directory` alongside `prefix`, which now names the shared root: `prefix = "rgi/glacier"` becomes `prefix = "glacier/input"` + `project_directory = "rgi"`, and the S4F configs also switch to `rgi_complex_file = "s4f_c.gpkg"` / `rgi_glacier_file = "s4f_g.gpkg"`. Readers of project-specific data (outlines, ice thickness, per-group CARRA2) address `{prefix}/{project_directory}/…`; readers of global data keep using `{prefix}/…`. Campaigns that leave `project_directory` unset — ISMIP7, KITP, ML — are unaffected, since the two prefixes then coincide. The S3 tree has to be rearranged to match before the new configs resolve; the old `rgi/glacier/…` and `s4f/glacier/…` trees can stay in place until the cutover is confirmed.
- Post-processing is now two campaign-neutral modules, `pism_terra/postprocess_scalar.py` and `pism_terra/postprocess_spatial.py`, exposed as `pism-postprocess-scalar` and `pism-postprocess-spatial`. They serve Greenland basins and RGI glaciers alike: the CRS is read from the file's grid mapping instead of assuming EPSG:3413, outlines are reprojected onto it (RGI ships EPSG:4326), the label column is resolved from `glacier_id`/`rgi_id`/`SUBREGION1`, the region dimension is `--dim-name` (default `glacier_id`), the whole-domain total is opt-in via `--total-name`, and each region carries its outline `area`. **Removed:** `pism-ismip7-greenland-postprocess`, `pism-ismip7-greenland-postprocess-spatial` and `pism-kitp-postprocess` — regenerate run scripts, or call `pism-postprocess-scalar ... --total-name GIS`. `pism-glacier-postprocess` is unchanged.
  **Migration:** every campaign, Greenland included, now writes the region dimension as `glacier_id` and labels basins from the outline's `glacier_id` column (`GIS_CE`, `GIS_CW`, …) rather than `basin`/`CE`. Selection becomes `ds.set_index(glacier_id="glacier_id_name").sel(glacier_id="GIS_CE")`; the appended whole-ice-sheet total is still `GIS`. Analysis code that hard-codes the old names — `pism_terra/analyze_scalar.py`, `pism_terra/kitp/analyze.py` — has not been updated and will need it. Pass `--column SUBREGION1 --dim-name basin` to reproduce the previous naming.
- `glacier.execute` no longer accepts a `--job-id` parameter and now stages files based on the full S3 URI of `RUN_SCIPT`, if needed.
- Per-region scalar output (ISMIP7 Greenland, KITP, and glacier post-processing) is now readable by CDO: `time` is written as the first dimension, and the region dimension (`basin` / `RGIid`) is a positional integer index whose labels live in a companion `basin_name` / `RGIid_name` coordinate. **Migration:** label selection now needs `ds.set_index(basin="basin_name").sel(basin="GIS")`. Previously CDO refused these files outright with `Time must be the first dimension!` followed by `Unsupported file structure`.

### Fixed

- `pism-glacier-prepare` exits 0 on success. Its entry point pointed at a function returning a `dict`, which setuptools passed to `sys.exit`, so a completely successful run printed the dict to stderr and exited 1.
- Per-region scalar output no longer carries a CF grid mapping. A summed timeseries has no grid, but the dimensionless `mapping` (or `spatial_ref`) variable survived the reduction over `x`/`y` and every variable pointed at it through `coordinates = "glacier_id_name mapping"`. Dropping it before the reduction was not enough — `rio.write_crs` recreates it from the dataset's own grid-mapping name — so it is now stripped from the result. Affects `pism-postprocess-scalar` and `pism-glacier-postprocess`; the spatial post-processing keeps its grid mapping, which it needs.
- `normalize_timeseries` no longer fails on a single-element variable list. `ds[["a"]] -= ...` takes a different `__setitem__` branch than a longer list and raised `cannot directly convert an xarray.Dataset into a numpy array`, so a bare string and a two-variable list worked while `["a"]` did not.
- `pism-kitp-analyze` reads the current post-processing output again. It assumed a `basin` dimension carrying string labels; scalar files now use the `glacier_id` integer index with labels in `glacier_id_name`. `with_region_labels` normalises all three layouts it has produced over time — `glacier_id`, `basin`/`RGIid`, and raw PISM scalar files with no region dimension — so selection by region name works regardless of file age. Two long-standing bugs fell out: the per-cell area used to convert grounding-line flux was hard-coded to 900 m (rescaling every other resolution by `(dx/900)²`, e.g. 1.78× at 1200 m) and is now read from the run's `grid.dx`; and `grounding_line_flux_nonneg`, which only post-processed files carry, is omitted with a log line instead of raising `KeyError` on raw scalar input.
- Region labels (`glacier_id_name`, and `basin_name` / `RGIid_name` when those dimension names are used) are written as a NetCDF char array instead of the netCDF-4-only variable-length `NC_STRING`. ncview rejected the whole file with `netcdf_fi_get_data: error on nc_get_vara_float call ... NetCDF: Not a valid data type or _FillValue type mismatch`. The labels still read back as strings, so label-based selection is unchanged; re-run post-processing on files that need to open in ncview.
- `pism-ismip7-greenland-postprocess` / `pism-kitp-postprocess` accept a directory as `OUTFILE` and name the file `basin_<input>.nc`, following the run scripts' convention. Passing a directory previously ran the whole reduction and then died in `to_netcdf` with `PermissionError: ... /output/basins`, losing the work. An unwritable destination is now reported before the reduction starts, and the log is written next to the output file.
- glacier post-processing now derives `ice_mass_glacierized` too, so the per-RGI scalar output carries the same diagnostic as the basin post-processors (only when the input has both `ice_mass` and `thk`).
- `ice_mass_glacierized` (ISMIP7 Greenland and KITP basin post-processing) now takes its ice-free threshold from the run's own `output.ice_free_thickness_standard` instead of a hard-wired 10 m, so a config that overrides the standard is summarised with the value the simulation used. Falls back to 10 m — PISM's default — when the file carries no `pism_config`.
- `pism-kitp-calibrate` no longer materialises the whole ensemble to rank it. The block-bootstrap RMSE is now a streaming `coarsen` reduction (`squared_error_blocks`), so peak memory is set by the dask chunk size rather than by `n_exp × ny × nx`, and only the metric's four variables are carried through the conservative regridding. Measured on an 8-member ensemble: 68 s → 27 s, with bit-identical RMSE.
- basin post-processing no longer fails with `Object has inconsistent chunks along dimension time` on files whose spatial variables differ in dtype
- basin rasters are scattered to the Dask workers instead of being embedded in the task graph, removing the `Sending large graph of size ...` warning on fine grids
- `spatial_ref` and other CF grid-mapping variables are dropped before the per-basin merge, which previously failed with `unable to determine if these variables should be coordinates or not`
- missing force_to_thickness.file
- runtime environment is now default, for dev work use environment-dev.yml.
- merged missing commits from summer school
- updated Docker image to pull to fix build bug in pism/pism
- improved postprocessing of RGI
- fixed missing output filename for inverse state
- fixed missing resolve()
- fixed initialization
- fixed CLI for stress balance
- fixed SNAP climate
- fixed environment-dev.yml

### Added

- `pism-kitp-adjust-timeseries`: trims spin-up from a KITP scalar timeseries, restamps the surviving years as 1..N and normalises the cumulative variables to zero at year 1, writing both a NetCDF and a CSV. Promoted from a notebook cell; `--variables` (default `ice_mass,ice_mass_glacierized`), `--spinup-end-year` and `--window-end-year` were hard-coded there.
- Generated `submit_*.sh` scripts now carry their own provenance: every `run.py` (glacier, KITP, ISMIP7 Greenland) stamps the git commit of the running code, whether its working tree was clean, and the command line that produced the script as a comment block below the scheduler directives. Untracked files (model output, scratch data) do not count as dirty, and a state that cannot be determined is left blank rather than reported as clean.
- `pism-kitp-calibrate`: the KITP surface-mass-balance calibration driver is now a CLI. Its data root is the `--data-dir` option (default `~/base/pism-terra`, `~` expanded) instead of a hard-coded module-level constant, and the analysis runs from `main()` rather than at import time.
- `pism-ismip7-greenland-postprocess-spatial`: per-basin **spatial** extraction (masked fields, no summing over x/y) writing one NetCDF per basin. Streams end-to-end, so >50 GB inputs never materialize; `--crop bbox|full` chooses bounding-box-cropped (default) or full-grid output, `--vars` restricts variables (default: all, including 3D), and `--method netcdf|zarr|shards` selects the write strategy — the default was picked by `benchmarks/bench_postprocess_spatial.py`, which stays in the repo.
- no-ops `elevation-dependent` climate for glaciers
- compliance checker run after simulation
- notebooks/pism_cloud_app.ipynb, a `voila` app.
- pism_terra/glacier/render_terrain_3.py

## [0.1.3]

### Added
- Applied UQ logic from KITP.

### Changed
- Upgraded the base PISM image to v2.3.0.

### Fixed
- Added the awscrt optional dependency to boto3 we need, but is not provided by the conda-forge recipe.
- Added the missing `campaign` section from `config/era5_ec2_1year.toml`.
- Ensures the `spatial` directory exists before executing PISM simulations.
- Fixed bugs in UQ assignment and ordering.

## [0.1.2]

### Added

- Support for ISMIP7 Greenland
- Support for KITP Greenland (sea-ice experiments)

## [0.1.1]

### Added

- New ice thickness dataset from Maffezzoli (in review).

## [Unreleased]

### Changed

- Use bucket and prefix to get RGI file with name stored in config file
- Refactor to isolate prepare stage to allow syncing with pism-cloud-data
- Switched output.{extra,timeseries} to output.{spatial,scalar}
- This requires PISM dev >= commit dev@f4a9668ef5204145820c8ce091dcabd0174b57b4
- Refactor {climate,dem}.py -> pism_terra/glacier/
- Fixed caculation of bounds

## [0.1.0]

### Added
- `Dockerfile` to build a pism-terra container image
- `pism_terra/etc/entrypoint.sh` script to serve as the container entrypoint
- `pism_terra.__main__` package entrypoint to support containerized runs in PISM-Cloud
- GitHub Actions workflows to support, building container images, ensuring this changelog is updated, testing, and releasing pism-terra.

### Changed
- `pism-terra` now uses `setuptools_scm` to dynamically compute a version number from the git history

## [0.0.2]

- Work on SNAP climate

## [0.0.1]

- PISM Terra project created
