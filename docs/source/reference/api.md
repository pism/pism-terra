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

```{eval-rst}
.. currentmodule:: pism_terra.config

.. autosummary::
    :toctree: generated/

    PismConfig
    UQConfig
    DistSpec
```

## Sampling

```{eval-rst}
.. currentmodule:: pism_terra.sampling

.. autosummary::
    :toctree: generated/

    lhs_sample
```

## Glacier subpackage

### Staging entrypoints

```{eval-rst}
.. currentmodule:: pism_terra.glacier.stage

.. autosummary::
    :toctree: generated/

    main
    stage_glacier
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
    pmip4
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

```{eval-rst}
.. currentmodule:: pism_terra.glacier.run

.. autosummary::
    :toctree: generated/

    run_single
    run_ensemble
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
    writer
    analyze
    forcing
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
