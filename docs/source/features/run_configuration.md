# Run configuration

Runs are described entirely by two artefacts:

1. A **TOML config** (e.g. `pism_terra/config/rgi_init_maffezzoli.toml`),
   validated by the Pydantic models in {py:mod}`pism_terra.config`.
2. A **Jinja2 template** (e.g. `pism_terra/templates/debug.j2`) that the run
   generator renders into a shell script.

## Pydantic config model

The top-level model is {py:class}`pism_terra.config.PismConfig`, which
aggregates per-section models (`run`, `job`, `time`, `grid`, `physics`, …).
Dotted TOML keys are flattened transparently, so either nesting works:

```toml
[surface.pdd]
factor_ice = 0.008
```

```toml
['surface.pdd.factor_ice'] = 0.008
```

## Init leg

A campaign can prepend a spin-up to the run of interest by declaring bounds in
its `[campaign]` section:

```toml
[campaign]
climate = "carra2"
init_climate = "carra2-monthly-mean"
init_surface_model = "debm_enhanced_forcing"
init_start = "0001-01-01"
init_end = "0051-01-01"
```

The generated script then holds two PISM calls. The first bootstraps and runs
over `init_start`..`init_end`; the second restarts from its state file — it no
longer sets `input.bootstrap` — and runs the `[time]` span. Init products are
named for the init span (`state_g500m_<rgi_id>_id_0_0001-01-01_0051-01-01.nc`),
so they never collide with the main leg's.

`init_climate` is optional and applies to the init leg alone: it names a second
{data}`pism_terra.glacier.stage.CLIMATE` key, staged alongside the main forcing
and substituted into `atmosphere.given.file`, `surface.debm_simple.std_dev.file`,
`surface.debm_simple.albedo_input.file` and `surface.pdd.std_dev.file`. A
monthly climatology is the usual choice: PISM cycles it, so the spin-up is not
pinned to a particular stretch of the historical record and its length is free.
Omit `init_climate` to spin up on the same forcing as the main leg.

`init_surface_model` is the same idea for the surface scheme: it names one of
the config's `[surface.options.*]` tables and applies to the init leg alone, so
the spin-up can run a different model — typically one adding `forcing` to hold
the geometry — before the main leg continues with `[surface] model`. Option
values of `"none"` are the placeholders staging fills in, so a file the main leg
already resolved is carried over rather than reset; everything else comes from
the named table. An unknown name fails at render time, listing what the config
offers.

Without `init_start`/`init_end` the run bootstraps directly, exactly as before.
The mechanism mirrors the ISMIP7 Greenland runner, which uses the same two
campaign keys.

## Inverse runs

`pism-glacier-run-inverse` (and its ISMIP7 counterpart) chains a third call
between the two: the init leg produces a prior state, `pismi` inverts on it,
and the main run restarts from that same prior state with the inverted result
regridded in.

1. **Init/prior** — bootstraps over `init_start`..`init_end` with the yield
   stress the config declares (usually `basal_yield_stress.model =
   "mohr_coulomb"`).
2. **Inversion** — `pismi` restarts from the init state and writes `tauc` to
   `output/inverse/inv_<init tag>.nc`. It is driven by the `[inverse]` section
   and the `[solver.inverse]` PETSc knobs.
3. **Main run** — restarts from the init state (no bootstrap), regrids `tauc`
   from the inversion output, and switches to `basal_yield_stress.model =
   "constant"` so that field is held fixed; the
   `basal_yield_stress.mohr_coulomb.*` options are dropped, having done their
   job in legs 1 and 2. The `[solver.forward]` knobs drive this call.

The init leg is therefore *required* for an inverse run — a config without
`campaign.init_start`/`init_end` exits with an error, since `pismi` would have
no state to invert on.

## Jinja2 templates

Templates expose the rendered `run_str` (PISM command-line flags) plus any
HPC scheduler scaffolding. `run_init_str` carries the init leg's command line
and is empty when no init bounds are configured, so templates guard it with
`{% raw %}{% if run_init_str %}{% endraw %}`. Inverse templates additionally
render `inv_str`, the `pismi` command line, **between** the two — the
`*-inverse.j2` templates are laid out in that order. Bundled templates live in
`pism_terra/templates/`. The `debug.j2` template is for interactive runs;
Slurm/PBS variants are provided per cluster.

The ISMIP7 runners split the forward span into `run_hist_str` and
`run_proj_str` instead of a single `run_str`, so their templates
(`*-ismip7*.j2`) are not interchangeable with the glacier ones.

```{admonition} TODO
- Document every variable available to templates.
- Show a minimal example of writing a custom template.
- Cross-link to the {doc}`../reference/configuration` page for the full schema.
```
