[![License: GPL-3.0](https://img.shields.io:/github/license/pism/pypac)](https://opensource.org/licenses/GPL-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

# pism-terra

Simulate any glacier complex in the world based on their RGI7 ID.

## Installation

Get pism-terra source from GitHub:

    $ git clone git@github.com:pism/pism-terra.git
    $ cd pism-terra

Optionally create Conda environment named *pism-terra*:

    $ conda env create -f environment.yml
    $ conda activate pism-terra

or using Mamba instead:

    $ mamba env create -f environment.yml
    $ mamba activate pism-terra

Install pism-terra:

    $ python -m pip install .

To install the dev version, replace the previous command with

    $ python -m pip install -e .
