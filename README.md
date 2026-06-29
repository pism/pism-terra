[![Documentation Status](https://readthedocs.org/projects/pism-terra/badge/?version=latest)](https://pism-terra.readthedocs.io/en/latest/?badge=latest)
[![License: GPL-3.0](https://img.shields.io:/github/license/pism/pypac)](https://opensource.org/licenses/GPL-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B%20%20%E2%97%8F%20%20%E2%97%8B-orange)](https://fair-software.eu)

# pism-terra

Simulate any glacier complex in the world based on their RGI7 ID.

## Installation

Get pism-terra source from GitHub:

    git clone git@github.com:pism/pism-terra.git
    cd pism-terra

Optionally create Conda environment named *pism-terra*:

    conda env create -f environment.yml
    conda activate pism-terra

or using Mamba instead:

    mamba env create -f environment.yml
    mamba activate pism-terra

Install pism-terra:

    python -m pip install .


## Documentation

Full documentation — installation, tutorials and workflows — is published on **Read the Docs**  <https://pism-terra.readthedocs.io/>

To build the docs locally:

    python -m pip install -e ".[docs]"
    cd docs
    make html
    open _build/html/index.html

Live-reload while editing:

    make livehtml

## Development

For development work and actual use of the package install the `dev` environment instead:

    conda env create -f environment-dev.yml

or

    mamba env create -f environment-dev.yml

Install it *editabble*:

    python -m pip install -e .

For rapid development and without internet, you can also run

    python -m pip install -e . --no-deps --no-build-isolation
