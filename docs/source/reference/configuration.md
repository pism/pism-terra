# Configuration reference

pism-terra configs are TOML files validated by Pydantic models in
{py:mod}`pism_terra.config`. Sections accept either nested tables or dotted
keys interchangeably:

```toml
[surface.pdd]
factor_ice = 0.008
```

is equivalent to

```toml
['surface.pdd.factor_ice'] = 0.008
```

## Top-level models

The full set of fields is rendered automatically:

```{eval-rst}
.. currentmodule:: pism_terra.config

.. autoclass:: PismConfig
    :members:
    :inherited-members: BaseModel
    :exclude-members: model_config, model_fields, model_computed_fields

.. autoclass:: UQConfig
    :members:
    :inherited-members: BaseModel
    :exclude-members: model_config, model_fields, model_computed_fields

.. autoclass:: DistSpec
    :members:
    :inherited-members: BaseModel
    :exclude-members: model_config, model_fields, model_computed_fields
```

## Bundled example configs

Look under `pism_terra/config/` for ready-made TOMLs covering common
scenarios:

- `rgi_init_maffezzoli.toml` — RGI v7 initialisation with Maffezzoli thickness.
- `kitp_greenland.toml` — KITP 1200 m Greenland.
- `setup_kitp_greenland.toml` — KITP shared-input prep.
- `era5_*.toml` — ERA5 forcing examples.

```{admonition} TODO
- Hand-write tables for each section (run, job, time, grid, physics, ...).
- Document the SciPy distributions supported by `DistSpec`.
```
