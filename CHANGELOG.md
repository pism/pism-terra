# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `pism-glacier-usgs-stakes OUTPUT.gpkg` writes the point measurements behind the USGS benchmark-glacier balances — seasonal `bw`/`ba` per stake and year, and the sub-seasonal `db` between visits where a glacier has them — to one GeoPackage (`sites`, `stakes`, `subseasonal` layers), in the release's m w.e. with the two date notations it mixes parsed into real date fields. Shares the download cache with `pism-glacier-usgs-benchmark`.
- `pism-glacier-usgs-benchmark` plots modelled winter and summer balances when the run has monthly output. A balance year runs from one annual minimum to the next and splits at the winter maximum, and the USGS release dates both per year (`Ba_Date`, `Bw_Date`), so the model is integrated over exactly the interval each observation covers — a month straddling a boundary is apportioned by days rather than snapped, since the measured dates fall mid-month more often than not. The integrated variable is `tendency_of_ice_mass_due_to_surface_mass_flux`, which is what a stake network measures; the existing annual line still shows total `tendency_of_ice_mass`. Years the release leaves undated, and seasons the model record does not fully span, are omitted rather than partially summed. Annual-only output is detected and leaves the plot unchanged. The seasonal series are saved alongside the observations as `Bw_model`/`Bs_model`/`Ba_model`.
- `--samples N` on `pism-glacier-run-forward`, `pism-glacier-run-inverse`, the ISMIP7 Greenland runners and `pism-kitp-run-ensemble` overrides the `samples` entry of the UQ file, so a large ensemble spec can be dry-run or trimmed without editing the TOML. Ensemble mode only; ignored without a `UQ_FILE`.
- `pism-glacier-usgs-benchmark` compares PISM with the USGS benchmark-glacier record. It downloads the ScienceBase release *Compiled Input Data and Glacier-Wide Mass Balances* (Gulkana, Wolverine, Lemon Creek, South Cascade, Sperry, Taku), assigns each glacier the RGI v7 `-G-` outline holding most of its stakes (the CSVs carry names, not IDs), converts the winter/summer/annual balances from m w.e. to Gt/yr with `pint_xarray` and the release's own time-varying area, and plots one figure per glacier. Given a run directory it overlays `tendency_of_ice_mass` from every `scalar_G_*.nc` below it — one line for a single run, median and 5-95 % band for several. Error bars are drawn when the CSV has an uncertainty column (none in v10) or `--uncertainty` supplies a constant.
- `[campaign] init_surface_model` lets the init leg run a different surface model from the main leg — e.g. `init_surface_model = "debm_enhanced_forcing"` to spin up with `surface.models = "debm_enhanced,forcing"` holding the geometry, then continue on plain `debm_enhanced`. It names one of the config's `[surface.options.*]` tables; option values of `"none"` are the placeholders staging fills in, so forcing files the main leg already resolved are carried over rather than reset to unset. An unknown name is rejected at render time with the available tables listed. Complements `init_climate`, which does the same for the forcing.
- A worked L-curve example for choosing the Tikhonov penalty weight, at `docs/source/greenland/inversion.md`. It runs against real output from a nine-decade sweep, locates the corner by maximum Menger curvature (eta = 100 for Greenland at 1500 m), and shows how to check that every member actually converged before trusting the curve. `docs/make_data/lcurve_fixture.py` reduces a sweep to the diagnostics the analysis needs — nine 2 GB inversion files become a 98 KB fixture under `docs/source/_data/`, so the page builds real figures without large files in the repository.
- UQ parameters can be **derived** from another instead of sampled: `derived_from` names the base and `scale`/`offset` give the relation (`value = base * scale + offset`). Physically linked options — a mass-balance profile whose melt limits and ELA shift together — stay linked in every ensemble member. Writing them as parallel `choices` lists could not do this: Latin Hypercube permutes each column independently, so the lists came out misaligned. Derived values are computed after sampling, so the base may be categorical or continuous and a derived parameter may itself be another's base; unresolvable bases, cycles and specs declaring both a `distribution` and a `derived_from` are rejected at load. Wired into the glacier, ISMIP7 Greenland and KITP ensemble builders. `pism_terra/uq/kennicott_elevation_dependent.toml` now samples only `z_M_min` and derives the other two.
- `[bed_deformation]` and `[frontal_melt]` are optional. `PismConfig` required both, so 29 of the packaged glacier configs — every `rgi_*`, `kennicott_*` and `s4f_carra2*` file — failed `load_config` outright with `bed_deformation Field required`, taking `pism-glacier-stage` and `pism-glacier-run-*` down with them. An omitted section now behaves exactly like `model = "none"` with an empty option table: it contributes no flags and PISM keeps its defaults. Configs that do declare the sections are unaffected; only the ISMIP7 and KITP runners read them.
- Glacier runs can prepend an init leg, the capability the ISMIP7 Greenland runner already had. `[campaign] init_start` / `init_end` make `pism-glacier-run-forward` and `-run-inverse` emit a bootstrap run over that span first; the main leg then restarts from its state file instead of bootstrapping itself. Init products are named for the init span, so they do not collide with the main leg's. Without the two keys nothing changes.
  A new `[campaign] init_climate` names a second climate builder used by the init leg alone — e.g. `init_climate = "carra2-monthly-mean"` to spin up on a climatology and then continue on transient CARRA2. `stage` builds both forcings and the run generator substitutes the init one into `atmosphere.given.file`, `surface.debm_simple.{std_dev,albedo_input}.file` and `surface.pdd.std_dev.file` for that leg. The glacier templates gained a guarded `run_init_str` slot.
- `carra2-monthly-mean`, a CARRA2 monthly climatology, as the counterpart of `era5-monthly-mean`. Twelve fields, one per calendar month, on the periodic 365-day `days since 0001-01-01` axis with `time_bounds` tiling the year, so PISM cycles it for the length of the run. `pism-glacier-prepare` does the averaging once into `climate/carra2_monthly_mean.zarr` (shared, like the store it comes from) and pre-reprojects it per aggregate group as `climate/carra2_monthly_mean_<group>.nc`; `stage` fetches the group file when it exists and otherwise clips the store directly. Select it with `climate = "carra2-monthly-mean"`.
  The reference period is **fixed** at 1990-2019 (`CARRA2_CLIMATOLOGY_YEARS`) rather than "everything in the store", so extending the CARRA2 download later does not silently change the climatology, and it stays comparable with the ERA5/RACMO/MAR means over the same years. `air_temp_sd` is averaged like every other field and so remains the typical *within-month* variability that PISM's PDD scheme reads — not the year-to-year spread of the monthly means.

### Changed
- S4F planning COGs (`pism-s4f-planning`) are written with lossless ZSTD plus a predictor (floating-point for elevations, horizontal for masks), embedded statistics, and cubic/nearest overviews via `s4f.cog_profile`. Plain DEFLATE barely compressed the float32 surfaces — a data-rich tile came out 13 % *larger* than raw once overviews were added — whereas ZSTD+predictor is 24 % smaller and decodes about twice as fast, which is what QGIS spends per tile when streaming from S3. The statistics let QGIS skip its first-open stretch computation.
- Glacier run scripts post-process with `pism-postprocess-scalar` instead of `pism-glacier-postprocess`. They emit one command per outline type — the complex (`-C`, one region) and the per-glacier outlines (`-G`, one region each) — writing `output/processed_scalar/scalar_{C,G}_<stem>.nc`. Both reductions read the same spatial file, so the output names are given explicitly rather than letting each derive the same default and overwrite the other. The `output/post_processing/*.toml` the old command consumed is **no longer written**; the run configuration is already recorded in the generated script's provenance header. `pism-glacier-postprocess` itself is untouched for existing TOMLs. Also moves `postprocess_ntasks` (which clamps a wide PISM decomposition down to a workable Dask worker count) into `pism_terra.workflow`, replacing the copies the ISMIP7 and KITP runners each carried.
- Every config with an `[inverse]` block now sets `inverse.tikhonov.rtol = 0.15` and `inverse.max_iterations = 500`. At PISM's default `rtol`, neither the Greenland (1500 m) nor the Kennicott (500 m) inversion could ever satisfy the stopping test: both reach a genuine stationary point — the state and design gradients balancing to 0.1 % and 2.6 % respectively — but `grad_sum` then floors 1.56x and 2.75x *above* `rtol * max(grad_state, grad_design)`. The optimizer ran on until TAO's line search collapsed with `DIVERGED_LS_FAILURE` (Greenland at iteration 1986 of 2500, Kennicott at 99), which still wrote usable results but only after hundreds of iterations that changed `J_misfit` in the sixth significant figure. 0.15 is the smallest round value that clears both floors; it stops Greenland at iteration 328 for 0.95 % in RMS velocity misfit against the 1986-iteration result, and Kennicott at 99 for 0.00 %. `atol` is left as an unreachable backstop. Raise `rtol` to 0.2 if a glacier still ends in a line-search failure — the Kennicott margin is only 9 %.
  `max_iterations` was 50, 250 or 1000 depending on the file; with `rtol` now reachable nothing approaches 500, so it serves purely as a cap.
- Glacier input preparation is one command again. `pism_terra/glacier/prepare.py` carried two near-identical ~200-line functions, `rgi()` and `s4f()`, that had already drifted apart (different step order, different outline loading, a lost comment); they are now a single `prepare()`. **Removed:** `pism-s4f-prepare` — use `pism-glacier-prepare` and pass the glacier-ID CSVs as trailing arguments:

  ```bash
  pism-glacier-prepare pism_terra/config/setup_rgi.toml glacier_input
  pism-glacier-prepare pism_terra/config/setup_s4f.toml glacier_input \
      pism_terra/config/S4F_target_*.csv
  ```

  With no CSVs the run covers whole regions, exactly as `pism-glacier-prepare` did before. **Removed:** the `glaciermip4` step — its archive was only mirrored into staging and nothing published it, so `--include glaciermip4` now exits with an unknown-dataset error. `prepare_glaciermip4` remains in `pism_terra.glacier.climate` for direct use. SNAP is unchanged: `--include snap` still builds `snap_cru_TS40_*.nc` into the shared `input/climate/`.
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
- `prepare_carra2` no longer fails with `NetCDF: HDF error` on workstations whose `/tmp` is a RAM-backed tmpfs. The yearly CARRA2 batches were merged with `cdo mergetime` into a temporary NetCDF in `$TMPDIR` — the whole 1986-2025 record, ~57 GB compressed — which overflowed a 62 GB tmpfs, and the merged file was then read back and written a second time into `carra2.zarr`. The batches are now concatenated lazily with `xr.open_mfdataset` and written straight into the store, one band of rows at a time (`write_zarr_in_bands`), so the record is written once and peak memory is one band rather than the rechunk of the whole domain. CDO scratch for the remaining small steps goes next to the batches instead of `/tmp`.

- The project's RGI GeoPackages are fetched once per data path, not once per glacier. `stage_glacier` and `s4f_glacier` cached them under each glacier's own `staging/` directory, so `pism-s4f-planning` — which walks every glacier in the project — re-downloaded the 309 MB S4F pair for each one and kept a copy: 4.6 GB of identical files after only 16 glaciers. Both now take an `rgi_cache_path`, and the CLIs pass the shared `--data-path` root. The outlines describe the whole project, so one copy serves every glacier. Existing per-glacier copies are harmless but can be deleted.
- Staging no longer aborts when a glacier's domain overhangs an ITS_LIVE tile edge. `region_code_from_bounds` required a per-region COG to *fully contain* the requested bbox, and a staged domain is the glacier plus a buffer plus a 0.25°/0.1° geographic pad — so `RGI2000-v7.0-C-03-01124` (northern Ellesmere) failed with "No ITS_LIVE per-region COG fully contains bounds" even though the glacier itself is 100 % inside region `03`; the domain reached 17 km past the tile's northern edge. The region is now taken from the RGI ID when available and the containment requirement is a preference: partial coverage is used and the shortfall logged. The overhang interpolates to NaN and is zeroed, which the velocity code already did for ITS_LIVE's own nodata gaps.
  Deriving the region from the ID also removes a genuine ambiguity — the footprints overlap in polar stereographic, and that Ellesmere domain sits inside the Arctic Canada (`03`) and Greenland (`05`) footprints by *identical* amounts, so the old geometry-only search picked whichever came first in the probe order.
- The apptainer job scripts honour a container passed in from the submitting environment. `container=/path/to/other.sif sbatch <script>` reached the job — SLURM's default `--export=ALL` carries it — but the script then discarded it on line 20 with an unconditional `export container=$HOME/pism-ismip7.sif`, so every run silently used the default image. It is now `export container="${container:-$HOME/pism-ismip7.sif}"`, and the not-found message shows the override syntax. Applies to all five apptainer templates; re-render existing run scripts to pick it up.
- CARRA2 forcing files carry a CF `grid_mapping` attribute again. `_finalize_pism_crs` stamps it, but rioxarray records `grid_mapping` in each variable's *encoding*, and handing `to_netcdf` a per-variable `encoding` dict for compression **replaces** that encoding rather than merging into it — so the attribute never reached the file and the written NetCDF had no discoverable projection. It went unnoticed because the per-glacier file is reprojected onto the model grid, where PISM does not need the projection; a per-group cache, which PISM must reproject itself, would have been rejected with "computational domain is not a subset". A new `pism_terra.workflow.compressed_encoding` builds the encoding while carrying `grid_mapping` through, and the three CARRA2 write sites use it. Re-run staging to fix existing files; nothing else changes in them.
- CARRA2 staging no longer stacks the time-invariant `orography` field once per output year. `_carra2_fill_years_and_bounds` concatenated its per-year pieces with xarray's default `data_vars="all"`, which broadcast orography across the whole time axis only for the next block to drop the copies; it now passes `data_vars="minimal"` explicitly. Output is unchanged, and xarray's `FutureWarning` about the changing default is gone.
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
