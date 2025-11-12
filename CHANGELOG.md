# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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
