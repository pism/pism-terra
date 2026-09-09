# Snow4Flow Inverse Modeling

{doc}`../greenland/inversion` sweeps the Tikhonov penalty weight for one design
variable, the basal yield stress `tauc`, and reads the corner of the resulting
L-curve. Everything there about what the two axes mean applies here and is not
repeated — read it first.

What is different on a valley glacier is *which field to invert for*. On the
ice sheet the velocity misfit is dominated by sliding, so `tauc` is the obvious
knob. On a debris-covered complex like Kaskawulsh the ice is thinner, colder in
places, and a good part of the surface speed is internal deformation, which
`tauc` cannot touch at all: no yield stress makes stiff ice flow faster. The
vertically-averaged hardness `hardav` is the knob for that half of the problem.

So there are three sweeps rather than one, over the same nine penalty weights:

| sweep | inverts for | config |
| --- | --- | --- |
| `tauc` | basal yield stress | `s4f_inverse_calib_tauc.toml` |
| `hardav` | vertically-averaged ice hardness | `s4f_inverse_calib_hardav.toml` |
| `alt` | both, in alternation | `s4f_inverse_calib_alt.toml` |

The third is a co-inversion: each cycle inverts for `tauc` with the hardness
held fixed, then for `hardav` with the yield stress held fixed, handing the
result of each phase to the next. `inverse.alternating_cycles = 2` runs that
twice.

## Running the sweeps

`pism_terra/uq/inverse_penalty.toml` walks η over nine decades, the same ladder
the Greenland sweep uses:

```toml
samples = 9

['inverse.tikhonov.penalty_weight']

choices = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
distribution = "choices"
```

Adjust `--ntasks` / `--tasks` and the template to match your system; the
commands below use `debug.j2`, which runs the members in the foreground.

```bash
pism-glacier-run-inverse --resolution 200m --ntasks 48 --tasks 24 \
    --data-path glacier_s4f_input --output-path 2026_09_s4f_inverse_tauc_penalty \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/s4f_inverse_calib_tauc.toml \
    pism_terra/templates/debug.j2 \
    pism_terra/uq/inverse_penalty.toml
```

```bash
pism-glacier-run-inverse --resolution 200m --ntasks 48 --tasks 24 \
    --data-path glacier_s4f_input --output-path 2026_09_s4f_inverse_hardav_penalty \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/s4f_inverse_calib_hardav.toml \
    pism_terra/templates/debug.j2 \
    pism_terra/uq/inverse_penalty.toml
```

```bash
pism-glacier-run-inverse --resolution 200m --ntasks 48 --tasks 24 \
    --data-path glacier_s4f_input --output-path 2026_09_s4f_inverse_alt_penalty \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/s4f_inverse_calib_alt.toml \
    pism_terra/templates/debug.j2 \
    pism_terra/uq/inverse_penalty.toml
```

Each writes one `output/inverse/inv_g200m_*_uq_*.nc` per member.

### What the three configs actually differ in

They are the same file bar the inversion block, so the comparison is clean:

| option | `tauc` | `hardav` | `alt` |
| --- | --- | --- | --- |
| `inverse.design.variable` | `tauc` | `hardav` | — (set per phase) |
| `inverse.alternating_cycles` | — | — | 2 |
| `inverse.adjoint.method` | `exact` | `exact` | `approximate` |
| `inverse.stress_balance.tauc_max` | 1e8 Pa | 1e8 Pa | 5e7 Pa |

`inverse.stress_balance.length_scale` is 1 km here rather than Greenland's
50 km — it sets the scale below which structure counts as roughness, and a
valley glacier's sticky spots are two orders of magnitude smaller than an ice
stream's.

```{admonition} The alternating run uses the approximate adjoint
:class: note

`inverse.adjoint.method = "approximate"` solves the symmetrized Newton
Jacobian rather than transposing it, which is faster and works with any
preconditioner but is not the exact gradient. It is what makes two cycles of
co-inversion affordable; if the alternating result looks off, that is the
first thing to vary.
```

## Reading the L-curves

`pism-inverse-lcurve` takes the members straight from the command line, so no
fixture is needed — the paths below are relative to wherever the sweeps were
written:

```bash
pism-inverse-lcurve -o tauc_lcurve.png \
    2026_09_s4f_inverse_tauc_penalty/RGI2000-v7.0-C-01-04374/output/inverse/inv_g200m_RGI2000-v7.0-C-01-04374_id_0_uq_*.nc
```

```bash
pism-inverse-lcurve -o hardav_lcurve.png \
    2026_09_s4f_inverse_hardav_penalty/RGI2000-v7.0-C-01-04374/output/inverse/inv_g200m_RGI2000-v7.0-C-01-04374_id_0_uq_*.nc
```

```bash
pism-inverse-lcurve -o alt_lcurve.png \
    2026_09_s4f_inverse_alt_penalty/RGI2000-v7.0-C-01-04374/output/inverse/inv_g200m_RGI2000-v7.0-C-01-04374_id_0_uq_*.nc
```

Each writes the figure, the table behind it as `.csv`, and prints the corner it
picks. The tool reads the swept parameter out of each file's `pism_config`, and
works out which field the run inverted for the same way — so the third command
needs no extra flags despite being a co-inversion.

::::{tab-set}
:::{tab-item} tauc
```{image} ../_static/s4f/tauc_lcurve.png
:alt: L-curve of the tauc penalty sweep
:width: 100%
```
:::
:::{tab-item} hardav
```{image} ../_static/s4f/hardav_lcurve.png
:alt: L-curve of the hardav penalty sweep
:width: 100%
```
:::
:::{tab-item} alternating
```{image} ../_static/s4f/alt_lcurve_combined.png
:alt: L-curves of both phases of the alternating co-inversion
:width: 100%
```
:::
::::

The alternating sweep produces three figures rather than one:
`alt_lcurve_tauc.png` and `alt_lcurve_hardav.png` for the phases separately,
and `alt_lcurve_combined.png` — shown above — overlaying them. The phases share
a misfit, since one residual comes out of both design variables together, so
the two curves differ only along the model-norm axis. That the axis is
comparable at all is a consequence of the parameterization: `J_design` is
evaluated on the dimensionless ζ, not on the field, so a `tauc` norm and a
`hardav` norm are the same kind of number. Under
`inverse.design.param = "ident"` they would carry Pa and Pa s^(1/3) and the
combined figure is refused rather than drawn.

## Looking at the fields

The corner is a scalar summary and cannot tell you whether the inverted field
is glaciologically plausible. `pism-inverse-plot` maps every member of a sweep
side by side on one shared color scale, which is what makes over-fitting
visible: weak regularization prints observational noise onto the field as
speckle, strong regularization smooths real sticky spots away.

```bash
pism-inverse-plot --design-variable tauc -o tauc_maps.png \
    2026_09_s4f_inverse_tauc_penalty/RGI2000-v7.0-C-01-04374/output/inverse/inv_g200m_RGI2000-v7.0-C-01-04374_id_0_uq_*.nc
```

```bash
pism-inverse-plot --design-variable hardav -o hardav_maps.png \
    2026_09_s4f_inverse_hardav_penalty/RGI2000-v7.0-C-01-04374/output/inverse/inv_g200m_RGI2000-v7.0-C-01-04374_id_0_uq_*.nc
```

```bash
pism-inverse-plot --design-variable tauc,hardav -o alt_maps.png \
    2026_09_s4f_inverse_alt_penalty/RGI2000-v7.0-C-01-04374/output/inverse/inv_g200m_RGI2000-v7.0-C-01-04374_id_0_uq_*.nc
```

Each field becomes its own figure, named after the variable it plots — so the
last command writes `alt_maps_tauc.png`, `alt_maps_hardav.png`,
`alt_maps_zeta_inv_tauc.png`, `alt_maps_zeta_inv_hardav.png` and
`alt_maps_inv_residual.png`. Three fields are drawn per design variable:

- the **design variable** itself, on a logarithmic scale, since a penalty
  sweep moves it over several decades;
- **ζ**, the parameterized design variable the optimizer actually works on
  (`tauc = tauc_scale · e^ζ`), signed and centred on zero, so Crameri's `broc`
  symmetric about zero shows which way the inversion pushed the field;
- the **velocity residual**, linear, because it reaches zero where the model
  fits the observations.

Both the design variable and ζ are masked to `zeta_fixed_mask == 0`, the cells
the inversion was free to change — elsewhere the field still holds the prior
and would dominate the shared color scale. The residual is masked to the
misfit area PISM actually fit.

::::{tab-set}
:::{tab-item} tauc
```{image} ../_static/s4f/tauc_maps_tauc.png
:alt: inverted tauc across the penalty sweep
:width: 100%
```
```{image} ../_static/s4f/tauc_maps_inv_residual.png
:alt: velocity residual across the tauc penalty sweep
:width: 100%
```
:::
:::{tab-item} hardav
```{image} ../_static/s4f/hardav_maps_hardav.png
:alt: inverted hardav across the penalty sweep
:width: 100%
```
```{image} ../_static/s4f/hardav_maps_inv_residual.png
:alt: velocity residual across the hardav penalty sweep
:width: 100%
```
:::
:::{tab-item} alternating
```{image} ../_static/s4f/alt_maps_tauc.png
:alt: inverted tauc from the alternating co-inversion
:width: 100%
```
```{image} ../_static/s4f/alt_maps_hardav.png
:alt: inverted hardav from the alternating co-inversion
:width: 100%
```
```{image} ../_static/s4f/alt_maps_inv_residual.png
:alt: velocity residual across the alternating co-inversion
:width: 100%
```
:::
::::

The ζ figures are not shown here but are written alongside; they are often the
easier read, since the same colour means the same multiple of `tauc_scale` in
every panel.

```{admonition} A phase that has not run yet is not a result
:class: warning

An alternating member that has only got through its `tauc` phase still carries
a `hardav` field — the prior computed from enthalpy, not an inversion of
anything. `pism-inverse-plot` reads `pismi`'s `pismi_alternation_completed`
stamp and warns when it maps such a member, rather than passing it off as a
result. Watch for that line in the output; it is the difference between a flat
panel that means "the ice is uniformly hard" and one that means "nothing has
happened here yet".
```

## Regenerating the figures

The sweeps are long-running, and a member contributes nothing to either tool
until it has written its inversion diagnostics. The figures on this page are
therefore a snapshot. Rebuild them from whatever has finished with:

```bash
python docs/make_data/s4f_inversion_figures.py --root /mnt/storstrommen/pism/terra
```

It runs exactly the commands above for all three sweeps, writes the PNGs into
`docs/source/_static/s4f/`, and skips any sweep whose output directory is not
there yet — so it is safe to run mid-campaign. `--rgi-id`, `--resolution` and
`--dpi` are there for a different glacier or a different-looking page.

```{admonition} These sweeps are incomplete
:class: warning

As of writing, roughly half of each nine-member sweep had finished: the
figures above are built from 5 (`tauc`), 3 (`hardav`) and 4 (`alt`) members.
Both tools print how many members went into each figure and warn by name about
the ones they skipped — read those counts before drawing conclusions, and in
particular do not trust a corner from a sweep with three points, where only
one interior point exists to have curvature at all.
```

## Using the result

Set the chosen weight, and the design variable it belongs to, in the campaign
config:

```toml
[inverse]

'inverse.design.variable' = "tauc"
'inverse.tikhonov.penalty_weight' = 100
```

`pism-glacier-run-inverse` chains the init, inversion and forward legs
together, regridding the inverted field into the forward run and holding it
fixed — `basal_yield_stress.model = "constant"` for `tauc`, and
`stress_balance.averaged_hardness.enabled` for `hardav`. See
{doc}`../features/run_configuration` for how the legs fit together.
