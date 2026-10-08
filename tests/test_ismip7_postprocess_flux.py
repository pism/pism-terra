# Copyright (C) 2026 Andy Aschwanden
#
# This file is part of pism-terra.
#
# PISM-TERRA is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PISM-TERRA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software

"""
Tests for :mod:`pism_terra.ismip7.postprocess_flux`.

Builds a miniature ISMIP7 submission directory — one file per flux variable,
sharing a grid and an undecoded time axis — and covers:

- finding the per-variable files without mistaking ``tendacabf`` for
  ``acabf``, and naming the output after the experiment.
- the cell area and the units the integral carries, including that a
  thickness rate becomes a volume rate.
- reading the projection from ``proj_params``, as submission files state it.
- an end-to-end reduction whose per-basin numbers are ``flux * area * cells``
  and whose time axis comes out byte-for-byte as it went in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pism_terra.ismip7.postprocess_flux import (
    FLUX_VARS,
    cell_area,
    find_flux_files,
    integrated_units,
    output_name,
    process_experiment,
    submission_crs,
)

#: Cell size of the synthetic grid, metres.
SPACING = 1000.0
STEM = "GrIS_UAF_PISM_m001_CESM2-WACCM_f001_ssp126_C005_2015-2299"


def write_submission(directory: Path, n_time: int = 3, n_xy: int = 6, value: float = 2.0) -> Path:
    """
    Write a miniature ISMIP7 submission directory.

    One file per flux variable plus a ``tendacabf`` decoy, all sharing a
    regular grid, an undecoded time axis and a CF grid mapping that states
    its projection the way real submissions do.

    Parameters
    ----------
    directory : pathlib.Path
        Directory to create and fill.
    n_time : int, optional
        Length of the time axis.
    n_xy : int, optional
        Grid size, both dimensions.
    value : float, optional
        Constant every flux variable is filled with.

    Returns
    -------
    pathlib.Path
        The directory written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    x = np.arange(n_xy, dtype="float64") * SPACING
    y = np.arange(n_xy, dtype="float64") * SPACING
    # Raw day numbers with CF attributes and no decoding, as the submission
    # files carry them after pism-ismip7-fix-time-flux-variables.
    time = np.array([60446.0 + 365 * i for i in range(n_time)], dtype="float32")
    bounds = np.stack([time - 182.0, time + 183.0], axis=1)

    units = {"hfgeoubed": "W m^-2", "dlithkdt": "m s^-1"}
    for variable in sorted(FLUX_VARS) + ["tendacabf"]:
        data = np.full((n_time, n_xy, n_xy), value, dtype="float32")
        ds = xr.Dataset(
            {
                variable: (("time", "y", "x"), data, {"units": units.get(variable, "kg m^-2 s^-1")}),
                "time_bounds": (("time", "nv"), bounds),
                "mapping": ((), np.int8(0), {"grid_mapping_name": "polar_stereographic", "proj_params": "EPSG:3413"}),
            },
            coords={
                "time": (
                    "time",
                    time,
                    {"units": "days since 1850-01-01", "calendar": "standard", "bounds": "time_bounds"},
                ),
                "x": ("x", x),
                "y": ("y", y),
            },
        )
        ds.to_netcdf(directory / f"{variable}_{STEM}.nc", encoding={"time": {"dtype": "float32"}})
    return directory


@pytest.fixture(name="submission")
def fixture_submission(tmp_path: Path) -> Path:
    """
    A miniature submission directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    pathlib.Path
        The directory.
    """
    return write_submission(tmp_path / "C005")


def test_find_flux_files_ignores_the_tend_variables(submission: Path):
    """
    Match on the variable name, not a substring.

    ``tendacabf`` sits in the same directory and starts with the same letters
    once a glob is involved; it is a different quantity (already integrated)
    and must not be picked up as ``acabf``.

    Parameters
    ----------
    submission : pathlib.Path
        Miniature submission directory.
    """
    files = find_flux_files(submission)
    assert set(files) == FLUX_VARS
    assert files["acabf"].name.startswith("acabf_")
    assert not any("tend" in f.name for f in files.values())


def test_find_flux_files_tolerates_a_missing_variable(submission: Path):
    """
    Leave out what a run did not report, rather than failing.

    Parameters
    ----------
    submission : pathlib.Path
        Miniature submission directory.
    """
    (submission / f"dlithkdt_{STEM}.nc").unlink()
    files = find_flux_files(submission)
    assert "dlithkdt" not in files
    assert len(files) == len(FLUX_VARS) - 1


def test_output_name_drops_the_variable_prefix(submission: Path):
    """
    Name the result after the experiment, not after whichever file was first.

    Parameters
    ----------
    submission : pathlib.Path
        Miniature submission directory.
    """
    assert output_name(find_flux_files(submission)) == f"basin_flux_{STEM}.nc"


def test_output_name_needs_files():
    """
    Refuse to invent a name with nothing to name it after.
    """
    with pytest.raises(ValueError, match="no flux files"):
        output_name({})


def test_cell_area_reads_the_grid(submission: Path):
    """
    Take the cell area from the coordinates, and reject an uneven grid.

    The submission grid is not the grid the simulation ran on, so the area
    cannot be assumed.

    Parameters
    ----------
    submission : pathlib.Path
        Miniature submission directory.
    """
    with xr.open_dataset(submission / f"acabf_{STEM}.nc", decode_times=False) as ds:
        assert cell_area(ds) == SPACING * SPACING

        uneven = ds.assign_coords(x=np.array([0.0, 1000.0, 3000.0, 4000.0, 5000.0, 6000.0]))
        with pytest.raises(ValueError, match="not evenly spaced"):
            cell_area(uneven)


@pytest.mark.parametrize(
    ("units", "expected"),
    [
        ("kg m^-2 s^-1", "kg s^-1"),
        ("kg m^-2 second^-1", "kg second^-1"),
        ("kg m-2 s-1", "kg s-1"),
        ("W m^-2", "W"),
        # A thickness rate integrates to a volume rate: the metre exponent
        # gains two rather than the factor being cancelled.
        ("m s^-1", "m^3 s^-1"),
        ("", ""),
        (None, None),
    ],
)
def test_integrated_units(units, expected):
    """
    Multiply the units by an area.

    Parameters
    ----------
    units : str or None
        Units as a submission file states them.
    expected : str or None
        Units of the integral.
    """
    assert integrated_units(units) == expected


def test_submission_crs_falls_back_to_proj_params(submission: Path):
    """
    Read the projection the way submission files state it.

    They carry a CF grid mapping without ``crs_wkt``, which is what
    :func:`pism_terra.workflow.dataset_crs` alone requires.

    Parameters
    ----------
    submission : pathlib.Path
        Miniature submission directory.
    """
    with xr.open_dataset(submission / f"acabf_{STEM}.nc", decode_times=False) as ds:
        assert submission_crs(ds) == "EPSG:3413"
        # An explicit override wins.
        assert submission_crs(ds, "EPSG:3031") == "EPSG:3031"


def test_process_experiment_integrates_and_keeps_time(tmp_path: Path, submission: Path):
    """
    Integrate over the basin and leave the time axis exactly as it was.

    The per-basin number must be ``flux * cell_area * cells``, not a sum of
    per-area values, and the time coordinate must come back with the same
    values, dtype and attributes — those stamps were placed inside their
    averaging interval by an earlier step.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    submission : pathlib.Path
        Miniature submission directory.
    """
    import geopandas as gpd  # pylint: disable=import-outside-toplevel
    from dask.distributed import Client  # pylint: disable=import-outside-toplevel
    from shapely.geometry import box  # pylint: disable=import-outside-toplevel

    # One basin over the lower-left 3x3 cells, whose centres are at 0..2 km.
    outline = tmp_path / "basin.gpkg"
    gpd.GeoDataFrame(
        {"basin": ["B1"]}, geometry=[box(-SPACING / 2, -SPACING / 2, 2.5 * SPACING, 2.5 * SPACING)], crs="EPSG:3413"
    ).to_file(outline)

    out = tmp_path / "basins"
    with Client(processes=False, n_workers=1, threads_per_worker=2, dashboard_address=None) as client:
        written = process_experiment(submission, out, outline, client, column="basin")

    assert written is not None and written.name == f"basin_flux_{STEM}.nc"
    with xr.open_dataset(written, decode_times=False) as result:
        # 9 cells x 1e6 m^2 x 2.0 = 1.8e7 in the integrated units.
        expected = 2.0 * SPACING * SPACING * 9
        for variable in FLUX_VARS:
            assert float(result[variable].isel(time=0).squeeze()) == pytest.approx(expected), variable
        assert result["acabf"].attrs["units"] == "kg s^-1"
        assert result["dlithkdt"].attrs["units"] == "m^3 s^-1"
        assert result["hfgeoubed"].attrs["units"] == "W"

        with xr.open_dataset(submission / f"acabf_{STEM}.nc", decode_times=False) as original:
            np.testing.assert_array_equal(result["time"].values, original["time"].values)
            assert result["time"].dtype == original["time"].dtype
            for key in ("units", "calendar"):
                assert result["time"].attrs[key] == original["time"].attrs[key]
            assert "time_bounds" in result
            np.testing.assert_array_equal(result["time_bounds"].values, original["time_bounds"].values)
