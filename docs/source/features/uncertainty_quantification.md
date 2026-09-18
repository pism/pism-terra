# Uncertainty quantification

Ensemble runs are driven by a separate UQ TOML that names parameters to vary
and the SciPy distributions they're drawn from.

## UQ config schema

The top-level model is {py:class}`pism_terra.config.UQConfig`. Each entry is
a {py:class}`~pism_terra.config.DistSpec` with `distribution` plus distribution
parameters (`loc`, `scale`, `a`, `b`, …):

```toml
['surface.pdd.factor_ice']
loc = 0.008
scale = 0.004
distribution = "truncnorm"
a = -2
b = 2
```

### Categorical parameters

A parameter that takes one of a fixed set of values (a model name, a cost
function, …) uses the `choices` pseudo-distribution instead of a SciPy one:

```toml
['inverse.state_func']
distribution = "choices"
choices = ["meansquare", "huber"]
```

Values are passed to PISM verbatim, so strings stay strings and numbers stay
numbers. `weights` (one non-negative number per choice) makes the draw
non-uniform; omitting it means uniform. `"choice"` and `"categorical"` are
accepted spellings of the same thing.

Categories are laid out on the unit interval in declaration order, so Latin
Hypercube sampling draws them (near-)equally often when `samples` is a multiple
of the number of choices, and `method = "factorial"` sweeps them
deterministically.

This is distinct from the `mapping` block, which stays the way to point a flag
at *staged* files: there, an integer `randint` draw indexes a column of the
staging table (see {py:func}`pism_terra.workflow.apply_choice_mapping`).

### Derived parameters

Some PISM options are not independent. A mass-balance profile, for instance,
may shift as a unit, so its melt limits and ELA move together. Sampling them
separately decorrelates values that are physically locked — and writing
parallel `choices` lists does not help, because Latin Hypercube permutes each
column independently, so member *N* gets an arbitrary combination.

Declare the relation instead. A derived parameter names its base and how it
follows it, `value = base * scale + offset`:

```toml
samples = 7

['surface.elevation_dependent.z_M_min']
distribution = "choices"
choices = [700, 650, 600, 550, 500, 450, 400]

['surface.elevation_dependent.z_ELA']
derived_from = "surface.elevation_dependent.z_M_min"
offset = 1050

['surface.elevation_dependent.z_M_max']
derived_from = "surface.elevation_dependent.z_M_min"
offset = 1892
```

Only `z_M_min` is sampled; the other two are computed per member, so the
1050 m and 1892 m separations hold in every run. `scale` defaults to `1` and
`offset` to `0`, so either alone is enough. Because the values are computed
*after* sampling, the base can be categorical as above or continuous, and a
derived parameter may itself serve as another's base.

Unresolvable bases, cycles, and specs that declare both a `distribution` and a
`derived_from` are rejected when the file is loaded, rather than surfacing as a
missing column once the ensemble is already staged.

```{admonition} Member order
:class: note

Deriving fixes the *linkage*, not the order of members: Latin Hypercube still
decides which shift member 0 gets. When exactly one parameter is sampled,
`method = "factorial"` walks its levels in declaration order instead, so
member *N* is the *N*-th listed value. With more than one sampled parameter
that becomes the Cartesian product.
```

## Sampling

{py:func}`pism_terra.sampling.lhs_sample` performs Latin Hypercube sampling and
inverse-transforms each unit-cube column through its declared distribution.
Derived parameters are added afterwards by
{py:meth}`pism_terra.config.UQConfig.apply_derived`, which every run generator
calls once the posterior (if any) has been folded in.

## Generating an ensemble

```bash
pism-glacier-run-ensemble \
    RGI2000-v7.0-C-01-04374 \
    pism_terra/config/rgi_g.gpkg \
    pism_terra/config/rgi_init_maffezzoli.toml \
    pism_terra/templates/debug.j2 \
    pism_terra/uq/debm.toml
```

This produces one run script per ensemble member.

```{admonition} TODO
- Document discrete-distribution support (`randint`, …).
- Cover correlated/conditioned distributions if/when supported.
- Show how to inspect the realised sample with pandas.
```
