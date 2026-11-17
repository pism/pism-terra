# CARRA2 Near-Surface Temperature Bias

A forcing built from CARRA2's free atmosphere and one built from its 2 m air
temperature disagree at the glacier surface. Over melting ice the 2 m air is
held near 0 °C, while the free air above it keeps warming.
`pism-carra2-bias-correction`
({py:mod}`pism_terra.glacier.carra2_bias_correction`) measures it. PISM's
`carra2_bias` atmosphere modifier adds it back to the 2 m temperature.

## What the bias is

A daily CARRA2 store written by `pism-s4f-carra2` holds the air temperature
and the geopotential height on twelve pressure levels (1000–500 hPa), the 2 m
air temperature, and the terrain (`orography`) that the 2 m temperature refers
to. For every grid cell and every day, a straight line `T = a + b z` is fitted
through the levels and evaluated at the terrain
({py:func}`~pism_terra.glacier.carra2_bias_correction.fit_to_ground`):

$$
b = T_\text{fit}(z_\text{terrain}) - T_\text{2m}.
$$

CARRA2 also reports temperatures on levels below its terrain, but those values
are extrapolated, not air. The fit uses every level above the terrain plus
`--below` levels under it (one by default). With only one level above the
terrain, a high site could not be fitted otherwise.

The figure below shows the fit for two CARRA2 cells on S4F target 01. Both
sit on 1000 m terrain; one is on glacier and one is off. The top two rows show
a typical summer day and a typical winter day for each cell, picked as the day
in June–August or December–February whose bias is the median of the season.
Filled points are levels above the terrain. Hollow points lie below it, and
grey ones are left out of the fit. The line is carried down to the terrain
(diamond) and compared with the 2 m air temperature (star). The inset zooms in
on the ground. The bottom row shows each cell's daily bias over 2000–2009.

```{image} ../_static/s4f/carra2_bias_profile.png
:alt: Pressure-level temperatures fitted and carried down to the terrain, compared with the 2 m air temperature, for an on-glacier and an off-glacier CARRA2 cell in summer and winter.
:width: 100%
```

The two cells behave very differently:

- **On glacier**, the bias peaks every summer. On the typical summer day the
  free air carried down gives 7.2 °C, but the 2 m air over the melting surface
  is 1.4 °C, a bias of +5.7 K. Individual melt days reach 10–15 K.
- **Off glacier**, the 2 m air over bare ground warms with the free air, so the
  summer bias is under 1 K. The bias peaks mildly in winter instead, when the
  surface cools below the air above it. On the typical winter day the levels
  also bend near the ground, so the straight fit (−4.6 K km⁻¹) misses the
  lowest levels.

## Fitting it by elevation and season

`pism-carra2-bias-correction` runs the per-cell fit over the whole store. It
then splits cells into on and off glacier by whether their centre lies inside
an outline, and bins them by terrain elevation:

```bash
pism-carra2-bias-correction --outlines glacier_s4f_input/s4f_g.gpkg \
    --output-path s4f_01_mean_split -- \
    ~/storstrommen/terra/s4f_carra2_3d/carra2_daily_S4F_target_01.zarr
```

The store is read one chunk at a time, and each bias is counted into
histograms per day, category and elevation bin
({py:func}`~pism_terra.glacier.carra2_bias_correction.bias_histograms`).
Memory therefore depends on the chunk size, not on the size of the domain or
the length of the record. Cells below `--min-elevation` (500 m) are taken for
sea and left out.

The first figure shows the distribution of the bias per 250 m elevation bin,
for every cell and day. The numbers on the right count the cells in each bin.
The lines are the annual-mean profiles of the fit described next.

```{image} ../_static/s4f/carra2_bias_by_elevation.png
:alt: Violin plots of the bias by elevation bin, on and off glacier, with the fitted annual-mean profiles.
:width: 80%
```

For each category, the daily median bias of each bin is fitted with an
annual-mean elevation profile plus one yearly cycle. The cycle's amplitude and
phase may change linearly with elevation
({py:func}`~pism_terra.glacier.carra2_bias_correction.fit_seasonal`). Two
profiles are fitted and compared by BIC:

- a straight line in elevation;
- a bias that decays exponentially with elevation toward a constant, which
  follows the large excess over the lower glacier that levels off higher up.

By default the fit follows the mean (least squares). The reason is that PISM
adds the bias to a daily-mean temperature that the PDD and dEBM schemes
integrate around, so the warm tail of melt days has to count. `--fit median`
follows the typical day instead.

```{image} ../_static/s4f/carra2_bias_seasonal_fit.png
:alt: Daily median bias per elevation bin against day of the year, on and off glacier, with the fitted yearly cycles coloured by elevation.
:width: 100%
```

For S4F target 01 the exponential profile wins in both categories. Off
glacier, the cycle peaks in winter at the lowest bins and flattens with
elevation. On glacier, it peaks in late July at the lowest bins and turns to a
peak in late autumn to winter above about 3000 m, where the surface rarely melts.

The tool writes the medians, both fits and the fitted cycle per bin as CSV
files (`bias_seasonal_fit.csv` and others) into `--output-path`, together with
three diagnostics:

- `r2_ceiling`: the best R² that any bias depending on elevation and season
  alone can reach. Whatever remains is weather.
- `sigma_extra`: the day-to-day variance that the corrected forcing still
  lacks. It can be added in quadrature to a PDD or dEBM `std_dev`.
- A half-record comparison
  ({py:func}`~pism_terra.glacier.carra2_bias_correction.half_record_stability`),
  which checks that the fit belongs to the region rather than to the years it
  saw.

## Using it in PISM

The `given_carra2bias` atmosphere model applies the fitted bias on top of the
CARRA2 2 m temperature, after the elevation-change lapse-rate correction:

```toml
[atmosphere]

model = "given_carra2bias"

[atmosphere.options.given_carra2bias]

'atmosphere.models' = "given,elevation_change,carra2_bias"
```

See `pism_terra/config/s4f_calib_carra2_debm_enhanced_bias.toml` for a full
configuration.

## Regenerating the figures

The two figures from the bias tool are copied unchanged from its
`--output-path`. The profile figure is made by
{download}`docs/make_data/carra2_bias_profile_figure.py <../../make_data/carra2_bias_profile_figure.py>`.
It calls the same
{py:func}`~pism_terra.glacier.carra2_bias_correction.fit_to_ground`, so the
fit it draws is exactly the one the bias tool uses. It reads only the store
chunk around each cell, so it takes seconds:

```bash
python docs/make_data/carra2_bias_profile_figure.py \
    --outlines glacier_s4f_input/s4f_g.gpkg --elevation 1000 \
    -o docs/source/_static/s4f/carra2_bias_profile.png \
    ~/storstrommen/terra/s4f_carra2_3d/carra2_daily_S4F_target_01.zarr
```

Use `--elevation` to choose the terrain height of the two cells, and
`--cells on` or `--cells off` for a single column. `--x`/`--y` picks one
particular cell, `--summer-day`/`--winter-day` fix the dates shown, and
`--below` matches the bias tool's option.
