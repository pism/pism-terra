# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `glacier.execute` no longer accepts a `--job-id` parameter and now stages files based on the full S3 URI of `RUN_SCIPT`, if needed.
- Per-region scalar output (ISMIP7 Greenland, KITP, and glacier post-processing) is now readable by CDO: `time` is written as the first dimension, and the region dimension (`basin` / `RGIid`) is a positional integer index whose labels live in a companion `basin_name` / `RGIid_name` coordinate. **Migration:** label selection now needs `ds.set_index(basin="basin_name").sel(basin="GIS")`. Previously CDO refused these files outright with `Time must be the first dimension!` followed by `Unsupported file structure`.

### Fixed

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
