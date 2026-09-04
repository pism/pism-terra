#!/bin/bash

# ISMIP7 post-processing script for scalar diagnostics
#
# Expects the file with scalar diagnostics to be named
#
# scalar_DOMAIN_GROUP_PISM_XXX_XXX_XXX_XXX_XXX_YYYY-YYYY.nc, e.g.
#
# scalar_DOMAIN_GROUP_PISM_m001_CESM2-WACCM_f001_historical_C001_1985-2014.nc
#
# when splitting this file, "scalar" will be replaced by the name of a scalar diagnostic.

set -u
set -e

input=$1

# fix global attributes:
ncatted -O \
        -a crs,global,c,c,"EPSG:3413" \
        -a command,global,d,c,"" \
        -a source,global,d,c,"" \
        ${input} tmp.nc

# tell bash to remove tmp.nc when done:
trap 'rm -f tmp.nc' EXIT

# fix time units:
script='
time=float(time/86400);
time@units="days since 1850-01-01";
time_bounds=float(time_bounds/86400);
time_bounds@units="days since 1850-01-01"
'

ncap2 -O -s "${script}" tmp.nc tmp.nc

snapshot_variables="
lim
limnsw
iareagr
iareafl
"

flux_variables="
tendacabf
tendlibmassbfgr
tendlibmassbffl
tendlicalvf
tendlifmassbf
tendligroundf
"

fill_value=9.9692099683868690e+36

for var in ${snapshot_variables};
do
  output=${input/scalar/${var}}
  # extract the variable
  ncks -v ${var} -O tmp.nc ${output}
  # convert from double to float
  ncap2 -s "${var}=float(${var})" -O ${output} ${output}
  # set _FillValue
  ncatted -a _FillValue,${var},c,f,${fill_value} -O ${output} ${output}
done

for var in ${flux_variables};
do
  output=${input/scalar/${var}}
  # extract the variable
  ncks -v ${var} -O tmp.nc ${output}
  # convert from double to float
  ncap2 -s "${var}=float(${var})" -O ${output} ${output}
  # set _FillValue
  ncatted -a _FillValue,${var},c,f,${fill_value} \
          -a units,${var},m,c,"kg s-1" \
          -O ${output} ${output}
  # correct the time dimension
  ./fix-time-flux-variables.py ${output}
done
