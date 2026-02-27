# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pism-terra is a Python package for simulating any glacier complex in the world using PISM (Parallel Ice Sheet Model) based on RGI7 (Randolph Glacier Inventory v7) IDs. It automates the workflow of staging input data, configuring simulations, and running ensemble experiments.

## Development Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate pism-terra

# Install in development mode
python -m pip install -e .
```

## Commands

### Running Tests
```bash
pytest                          # Run all tests
pytest tests/test_config.py     # Run specific test file
pytest -k "test_flatten"        # Run tests matching pattern
```

### Code Quality
```bash
black pism_terra/               # Format code (line-length 120)
pylint pism_terra/              # Lint code
mypy pism_terra/                # Type checking
```

### CLI Entry Points
```bash
pism-glacier-stage RGI_ID RGI_FILE CONFIG_FILE    # Stage glacier inputs
pism-glacier-run RGI_ID RGI_FILE CONFIG_FILE TEMPLATE_FILE  # Generate single run
pism-glacier-run-ensemble RGI_ID RGI_FILE CONFIG_FILE TEMPLATE_FILE UQ_FILE  # Generate ensemble
pism-glacier-postprocess TOML_FILE                # Post-process results
```

## Architecture

### Core Workflow Pipeline
1. **Staging** (`pism_terra/glacier/stage.py`): Downloads and prepares input data (DEM, ice thickness, climate forcing) for a glacier identified by RGI ID
2. **Run Configuration** (`pism_terra/glacier/run.py`): Generates PISM run scripts from TOML configs and Jinja2 templates
3. **Post-processing** (`pism_terra/glacier/postprocess.py`): Processes simulation outputs

### Configuration System
- **PISM configs** (`config/*.toml`): Define run parameters, grid settings, physics models
- **UQ configs** (`uq/*.toml`): Define uncertainty quantification with SciPy distributions for ensemble runs
- **Templates** (`templates/*.j2`): Jinja2 templates for HPC job submission scripts

Configuration is validated using Pydantic models in `pism_terra/config.py`:
- `PismConfig`: Top-level config aggregating run, job, time, grid, physics sections
- `UQConfig`: Uncertainty specification with distribution specs (supports nested or dotted TOML keys)
- `DistSpec`: SciPy distribution parameters (norm, truncnorm, randint, etc.)

### Key Modules
- `config.py`: Pydantic configuration models with TOML parsing and validation
- `domain.py`: Grid creation and coordinate bounds calculation
- `dem.py`: DEM/thickness/bed data preparation from various sources
- `climate.py`: Climate forcing generation (ERA5, PMIP4, SNAP)
- `sampling.py`: Latin Hypercube sampling with SciPy distribution transforms
- `workflow.py`: Dataset validation utilities

### Data Flow
```
RGI ID + Config TOML
    -> stage_glacier() produces boot file, grid file, climate file
    -> run_glacier() generates submission script with PISM flags
    -> For ensembles: UQ config + sampling creates parameter variations
```

## Configuration Patterns

TOML configs use dotted keys that map to PISM command-line flags:
```toml
['surface.pdd.factor_ice']
loc = 0.008
scale = 0.004
distribution = "truncnorm"
```

Config models accept both nested tables and dotted keys interchangeably.
