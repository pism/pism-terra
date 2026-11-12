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

So the inversion is run with four *strategies*, the four values of
`inverse.design.variable`:

| strategy | inverts for |
| --- | --- |
| `tauc` | basal yield stress |
| `hardav` | vertically-averaged ice hardness |
| `tauc_hardav` | both, alternating: `tauc` first, then `hardav` |
| `hardav_tauc` | both, alternating: `hardav` first, then `tauc` |

The last two are co-inversions: each phase inverts for one field with the
other held at its latest value, and hands its result to the next phase.
`inverse.alternating_cycles` sets how many times the pair is run, one by
default; a single field ignores it.

## Running the ensemble

Because the strategy is an ordinary parameter, one ensemble samples it
together with the penalty weight. `pism_terra/uq/inverse_penalty.toml` walks
η over nine decades, the same ladder the Greenland sweep uses, and asks for
the full factorial design, so every weight is run with every strategy — 36
members:

```toml
samples = 1
method = "factorial"

['inverse.tikhonov.penalty_weight']

choices = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
distribution = "choices"

['inverse.design.variable']

choices = ['tauc', 'hardav', 'tauc_hardav', 'hardav_tauc']
distribution = "choices"
```

Under `method = "factorial"` a variable with `choices` contributes all of
them, so `samples` does not change the count. The default Latin Hypercube
would not do here: it draws each variable separately and does not guarantee
every pairing.

All members share one config, `pism_terra/config/s4f_inverse_calib.toml`, and
land in one project directory. Adjust `--ntasks` / `--tasks` and the template
to match your system:

```bash
pism-glacier-run-inverse --resolution 200m --ntasks 48 --tasks 24 \
    --data-path glacier_s4f_input --output-path 2026_10_inverse_lcurve \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/s4f_inverse_calib.toml \
    pism_terra/templates/chinook-apptainer.j2 \
    pism_terra/uq/inverse_penalty.toml
```

Each member writes one `output/inverse/inv_g200m_*_uq_*.nc`, and
`output/uq.csv` lists the weight and strategy of every `uq` index.

## Reading the L-curves

`pism-inverse-lcurve` takes the members straight from the command line and
reads the penalty weight and the strategy out of each file's `pism_config`, so
the whole project goes in at once:

```bash
pism-inverse-lcurve -o lcurve.png \
    2026_10_inverse_lcurve/RGI2000-v7.0-C-01-04374/output/inverse/inv_g200m_RGI2000-v7.0-C-01-04374_id_0_uq_*.nc
```

It draws one L-curve per strategy on one pair of axes, each with its corner
starred, writes the table behind it as `lcurve.csv` (one row per member) and
prints the corners. `--strategy` keeps a subset, e.g. `--strategy tauc,hardav`.

The four curves share an axis because of what the model norm measures.
`J_design` is the regularization functional of ζ − ζ₀, the departure of the
parameterized field from its prior, and a single-field run leaves the other
field at its prior. So the norm of every strategy is

N = √(J_tauc + J_hardav)

with each pair phase taken from its last cycle, and a single field contributing
only its own term. Under the default `inverse.design.param = "exp"` every ζ is
dimensionless, so the terms add. Reading across the curves at the same N
compares how well each strategy fits the observed speeds for the same amount
of structure put into the fields.

Each pair also gets a figure of its phases, `lcurve_tauc_hardav.png` and
`lcurve_hardav_tauc.png`, one curve per phase against the shared misfit — the
phases produce one residual together, so the curves differ only along the
norm axis. They show which field the regularization is biting on, and whether
the two phases' corners agree on a weight. Under `inverse.design.param =
"ident"` the phase norms carry Pa and Pa s^(1/3), cannot be added or compared,
and the tool refuses those figures rather than draw them.

::::{tab-set}
:::{tab-item} all strategies
```{image} ../_static/s4f/lcurve.png
:alt: L-curves of the four inversion strategies
:width: 100%
```
:::
:::{tab-item} tauc_hardav phases
```{image} ../_static/s4f/lcurve_tauc_hardav.png
:alt: L-curves of both phases of the tauc_hardav co-inversion
:width: 100%
```
:::
:::{tab-item} hardav_tauc phases
```{image} ../_static/s4f/lcurve_hardav_tauc.png
:alt: L-curves of both phases of the hardav_tauc co-inversion
:width: 100%
```
:::
::::

## Looking at the fields

The corner is a scalar summary and cannot tell you whether the inverted field
is glaciologically plausible. `pism-inverse-plot` maps every member side by
side on one shared color scale, which is what makes over-fitting visible: weak
regularization prints observational noise onto the field as speckle, strong
regularization smooths real sticky spots away.

```bash
pism-inverse-plot -o maps.png \
    2026_10_inverse_lcurve/RGI2000-v7.0-C-01-04374/output/inverse/inv_g200m_RGI2000-v7.0-C-01-04374_id_0_uq_*.nc
```

Each field becomes its own figure, named after the variable it plots:
`maps_tauc.png`, `maps_hardav.png`, `maps_zeta_inv_tauc.png`,
`maps_zeta_inv_hardav.png` and `maps_inv_residual.png`. Every figure has one
row per strategy and one column per penalty weight, so reading down a column
compares the strategies at the same weight. A design-variable figure has rows
only for the strategies that inverted for that field — the `tauc` of a
`hardav`-only run is its prior, not a result — so `maps_tauc.png` has three
rows and `maps_inv_residual.png` four. Three fields are drawn per design
variable:

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
misfit area PISM actually fit. `--strategy` and `--design-variable` narrow the
figures down to a subset.

::::{tab-set}
:::{tab-item} tauc
```{image} ../_static/s4f/maps_tauc.png
:alt: inverted tauc for every strategy that inverts for it, across the penalty sweep
:width: 100%
```
:::
:::{tab-item} hardav
```{image} ../_static/s4f/maps_hardav.png
:alt: inverted hardav for every strategy that inverts for it, across the penalty sweep
:width: 100%
```
:::
:::{tab-item} residual
```{image} ../_static/s4f/maps_inv_residual.png
:alt: velocity residual of every strategy across the penalty sweep
:width: 100%
```
:::
::::

The ζ figures are not shown here but are written alongside; they are often the
easier read, since the same colour means the same multiple of `tauc_scale` in
every panel.

```{admonition} A phase that has not run yet is not a result
:class: warning

A pair member that has only got through its first phase still carries the
other field — for `tauc_hardav` a `hardav` that is the prior computed from
enthalpy, not an inversion of anything. `pism-inverse-plot` reads `pismi`'s
`pismi_alternation_completed` stamp, in the phase order of the strategy, and
warns when it maps such a member rather than passing it off as a result.
`pism-inverse-lcurve` leaves such a member out, since its total norm is not
known yet. Watch for those lines in the output; they are the difference
between a flat panel that means "the ice is uniformly hard" and one that means
"nothing has happened here yet".
```

## Regenerating the figures

The members are long-running, and one contributes nothing to either tool until
it has written its inversion diagnostics. The figures on this page are
therefore a snapshot. Rebuild them from whatever has finished with:

```bash
python docs/make_data/s4f_inversion_figures.py --root /mnt/storstrommen/pism/terra
```

It runs exactly the two commands above, writes the PNGs into
`docs/source/_static/s4f/`, and is safe to run while members are still
queued. `--project` names the project directory under `--root`
(`2026_10_inverse_lcurve` by default); `--rgi-id`, `--resolution` and `--dpi`
are there for a different glacier or a different-looking page.

## Using the result

Set the chosen strategy and weight in the campaign config:

```toml
[inverse]

'inverse.design.variable' = "tauc_hardav"
'inverse.tikhonov.penalty_weight' = 1000
```

`pism-glacier-run-inverse` chains the init, inversion and forward legs
together, regridding exactly the fields the inversion wrote into the forward
run and holding them fixed — `basal_yield_stress.model = "constant"` for
`tauc`, `stress_balance.averaged_hardness.enabled` for `hardav`, and both for
a pair. See {doc}`../features/run_configuration` for how the legs fit
together.
