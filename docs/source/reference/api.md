# API reference

Hand-grouped by subpackage. Each table is built with `autosummary` and
generates one page per symbol under `generated/`.

## Core domain & grids

```{eval-rst}
.. currentmodule:: pism_terra.domain

.. autosummary::
    :toctree: generated/

    create_domain
    create_grid
    get_bounds
    get_bounds_from_geometry
    new_range
```

```{eval-rst}
.. currentmodule:: pism_terra.grids

.. autosummary::
    :toctree: generated/

    load_grid
```

## Configuration

The TOML schema is documented in detail under {doc}`./configuration` — that
page is the canonical detail view. The summary table below intentionally has
no `:toctree:` so the field descriptions are registered in exactly one place
(otherwise autodoc + Pydantic's `model_fields` produces duplicate
``ref.python`` warnings).

```{eval-rst}
.. currentmodule:: pism_terra.config

.. autosummary::

    PismConfig
    UQConfig
    DistSpec
```

## Sampling

```{eval-rst}
.. currentmodule:: pism_terra.sampling

.. autosummary::
    :toctree: generated/

    create_samples
```

## Glacier subpackage

### Staging entrypoints

```{note}
``pism_terra.glacier.stage`` is documented as a CLI in {doc}`./cli`. It's
excluded from autosummary because it currently has an import-time error
(references `pmip4` from `pism_terra.glacier.climate`, which doesn't exist).
Re-add ``main`` and ``stage_glacier`` here when the upstream import is fixed.
```

### DEM

```{eval-rst}
.. currentmodule:: pism_terra.glacier.dem

.. autosummary::
    :toctree: generated/

    boot_file_from_grid
    prepare_surface
    get_surface_dem_by_bounds
```

### Climate forcing

```{eval-rst}
.. currentmodule:: pism_terra.glacier.climate

.. autosummary::
    :toctree: generated/

    era5
    carra2
    snap
    prepare_carra2
    prepare_carra2_for_group
```

### Ice thickness

```{eval-rst}
.. currentmodule:: pism_terra.glacier.ice_thickness

.. autosummary::
    :toctree: generated/

    prepare_ice_thickness_maffezzoli
    prepare_ice_thickness_frank
    get_ice_thickness
```

### Velocity & observations

```{eval-rst}
.. currentmodule:: pism_terra.glacier.observations

.. autosummary::
    :toctree: generated/

    region_code_from_bounds
    get_itslive_velocities_by_region_code
    get_velocities_by_bounds
    glacier_velocities_from_grid
    bathymetry_from_grid
```

### Run generation & post-processing

```{note}
``pism_terra.glacier.run`` is documented as a CLI in {doc}`./cli`. It's
excluded from autosummary because of an outstanding import-time error
(`pmip4` missing from `pism_terra.glacier.climate`); add it back here when
that's resolved.
```

```{eval-rst}
.. currentmodule:: pism_terra.glacier.postprocess

.. autosummary::
    :toctree: generated/

    main
```

## ISMIP7 Greenland

```{eval-rst}
.. currentmodule:: pism_terra.ismip7.greenland

.. autosummary::
    :toctree: generated/

    forcing
    prepare
    stage
    run
    postprocess
```

## KITP

```{eval-rst}
.. currentmodule:: pism_terra.kitp

.. autosummary::
    :toctree: generated/

    prepare
    stage
    run
    postprocess
    analyze
    forcing
```

```{note}
``pism_terra.kitp.writer`` is documented as a CLI in
{doc}`./cli` but excluded from autosummary because it imports the optional
`yac` dependency, which is not installed in the docs build environment.
```

## Infrastructure

```{eval-rst}
.. currentmodule:: pism_terra.aws

.. autosummary::
    :toctree: generated/

    s3_to_local
    local_to_s3
    download_from_s3
```

```{eval-rst}
.. currentmodule:: pism_terra.download

.. autosummary::
    :toctree: generated/

    download_archive
    download_file
    extract_archive
    download_request
    carra_download_request
```

```{eval-rst}
.. currentmodule:: pism_terra.workflow

.. autosummary::
    :toctree: generated/

    check_xr_lazy
    check_xr_fully
    check_rio
```

## Tools

```{eval-rst}
.. currentmodule:: pism_terra.tools.combine_crameri_colormaps

.. autosummary::
    :toctree: generated/

    main
```
