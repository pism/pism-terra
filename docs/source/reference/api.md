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

```{eval-rst}
.. currentmodule:: pism_terra.glacier.stage

.. autosummary::
    :toctree: generated/

    stage_glacier
    main
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
    carra2_monthly_mean
    snap
    prepare_carra2
    prepare_carra2_for_group
    prepare_carra2_monthly_mean
    write_zarr_in_bands
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

### USGS benchmark glaciers

```{eval-rst}
.. currentmodule:: pism_terra.glacier.usgs

.. autosummary::
    :toctree: generated/

    load_sites
    load_glacier_wide
    parse_dates
    load_measurements
    load_rgi_glaciers
    match_rgi_ids
    rgi_output_dir
    find_model_files
    open_pism
    grid_spacing
    run_label
    interval_edges
    is_monthly
    integrate_rate
    score
    format_skill
```

```{eval-rst}
.. currentmodule:: pism_terra.glacier.usgs_benchmark_glaciers

.. autosummary::
    :toctree: generated/

    to_mass_rate
    load_model_series
    model_seasonal_balances
    ice_area
    specific_balance
    to_specific_balances
    skill_scores
    plot_glacier
    run_pipeline
```

```{eval-rst}
.. currentmodule:: pism_terra.glacier.usgs_benchmark_stakes

.. autosummary::
    :toctree: generated/

    find_spatial_files
    stake_points
    sample_points
    point_smb_rate
    stake_balances
    stake_skill
    plot_stakes
    plot_scatter
    fit_gradient
    gradient_fits
    plot_gradient
    to_dataset
    run_pipeline
```

```{eval-rst}
.. currentmodule:: pism_terra.glacier.usgs_generate_geopackage

.. autosummary::
    :toctree: generated/

    build_stake_layers
    write_stake_geopackage
```

### Run generation & post-processing

```{eval-rst}
.. currentmodule:: pism_terra.glacier.run

.. autosummary::
    :toctree: generated/

    run_forward
    run_inverse
```

```{eval-rst}
.. currentmodule:: pism_terra.glacier.postprocess

.. autosummary::
    :toctree: generated/

    main
```

## Post-processing (all campaigns)

```{eval-rst}
.. currentmodule:: pism_terra

.. autosummary::
    :toctree: generated/

    postprocess_scalar
    postprocess_spatial
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
```

## KITP

```{eval-rst}
.. currentmodule:: pism_terra.kitp

.. autosummary::
    :toctree: generated/

    prepare
    stage
    run
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
    download_usgs_benchmark
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
