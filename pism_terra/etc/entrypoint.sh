#!/bin/bash --login
set -e
conda activate pism-terra
exec python -um pism_terra "$@"
