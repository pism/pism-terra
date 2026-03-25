# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Fixed a bug in UQ assignment
- Applied UQ logic from KITP

## [0.1.3]

### Changed
- Upgraded the base PISM image to v2.3.0.

### Fixed
- Added the awscrt optional dependency to boto3 we need, but is not provided by the conda-forge recipe.
- Added the missing `campaign` section from `config/era5_ec2_1year.toml`.
- Ensures the `spatial` directory exists before executing PISM simulations.

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
