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

## Jinja2 templates

Templates expose the rendered `run_str` (PISM command-line flags) plus any
HPC scheduler scaffolding. Bundled templates live in
`pism_terra/templates/`. The `debug.j2` template is for interactive runs;
Slurm/PBS variants are provided per cluster.

```{admonition} TODO
- Document every variable available to templates.
- Show a minimal example of writing a custom template.
- Cross-link to the {doc}`../reference/configuration` page for the full schema.
```
