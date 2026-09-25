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
Tests for :mod:`pism_terra.ismip7.greenland.mass_balance`.

A miniature submission tree on disk -- two GCMs, a historical and one
pathway each, one of the files empty as a run in flight leaves it -- and two
square basins whose integrals are known, covering:

- listing the tree and skipping the empty file;
- opening it as one ensemble on ``(gcm_id, ssp_id)``;
- the one-pass basin integration against sums done by hand, its units,
  the outline area and the whole-domain total;
- the mass balance and its cumulative series through the driver, and the
  files it writes.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from pism_terra.ismip7.greenland import mass_balance as mb

#: Grid: 10 x 8 cells of 1 km, EPSG:3413.
NX, NY, DX = 10, 8, 1000.0
X = -100_000.0 + DX * np.arange(NX)
Y = -1_000_000.0 - DX * np.arange(NY)

#: Two basins: the left 4 columns and the right 6 columns, in cells.
BASINS = {"GIS_W": (0, 4), "GIS_E": (4, 10)}

#: Flux per cell, kg m-2 s-1, keyed by (gcm, pathway, variable).
LEVELS = {
    ("CESM2-WACCM", "historical", "acabf"): 1.0,
    ("CESM2-WACCM", "historical", "ligroundf"): -0.5,
    ("MRI-ESM2-0", "historical", "acabf"): 2.0,
    ("MRI-ESM2-0", "historical", "ligroundf"): -1.0,
    ("CESM2-WACCM", "ssp585", "acabf"): 3.0,
    ("CESM2-WACCM", "ssp585", "ligroundf"): -2.0,
}
COUNTERS = {
    ("CESM2-WACCM", "historical"): "C001",
    ("MRI-ESM2-0", "historical"): "C002",
    ("CESM2-WACCM", "ssp585"): "C007",
}
YEARS = {"historical": (2013, 2014), "ssp585": (2015, 2016)}


def write_tree(root: Path) -> Path:
    """
    Write the miniature submission tree.

    Parameters
    ----------
    root : pathlib.Path
        The run's ``output`` directory to create.

    Returns
    -------
    pathlib.Path
        ``root``.
    """
    for (gcm, ssp, var), level in LEVELS.items():
        counter = COUNTERS[(gcm, ssp)]
        first, last = YEARS[ssp]
        time = pd.to_datetime([f"{year}-07-01" for year in range(first, last + 1)])
        data = np.full((time.size, NY, NX), level, dtype="float32")
        ds = xr.Dataset(
            {var: (("time", "y", "x"), data, {"units": "kg m-2 s-1"})},
            coords={"time": time, "y": Y, "x": X},
        )
        ds["mapping"] = ((), np.int8(0), {"grid_mapping_name": "polar_stereographic", "proj_params": "EPSG:3413"})
        ds[var].attrs["grid_mapping"] = "mapping"
        directory = root.joinpath(*mb.DEFAULT_TREE, counter)
        directory.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(directory / f"{var}_GrIS_UAF_PISM_m001_{gcm}_f001_{ssp}_{counter}_{first}-{last}.nc")
    # A run in flight: the file exists but has no records yet.
    empty = xr.Dataset({"acabf": (("time", "y", "x"), np.zeros((0, NY, NX), "float32"))}, coords={"y": Y, "x": X})
    empty.to_netcdf(
        root.joinpath(*mb.DEFAULT_TREE, "C007") / "acabf_GrIS_UAF_PISM_m001_MRI-ESM2-0_f001_ssp585_C008_2015-2299.nc"
    )
    return root


def outline() -> gpd.GeoDataFrame:
    """
    The two square basins as an outline in the grid's CRS.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per basin with a ``glacier_id`` column.
    """
    half = DX / 2
    boxes = [box(X[c0] - half, Y[-1] - half, X[c1 - 1] + half, Y[0] + half) for c0, c1 in BASINS.values()]
    return gpd.GeoDataFrame({"glacier_id": list(BASINS)}, geometry=boxes, crs="EPSG:3413")


@pytest.fixture(name="tree")
def fixture_tree(tmp_path: Path) -> Path:
    """
    The miniature submission tree.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    pathlib.Path
        The run's ``output`` directory.
    """
    return write_tree(tmp_path / "output")


def test_wildcard_root_collects_several_job_directories(tmp_path: Path):
    """
    A cloud project keeps one directory per job; a wildcard root gathers them all and finds the observations.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    write_tree(tmp_path / "job-a" / "output")
    # Move the second GCM into its own job directory, as the cloud lays it out.
    c002 = tmp_path / "job-a" / "output" / Path(*mb.DEFAULT_TREE) / "C002"
    target = tmp_path / "job-b" / "output" / Path(*mb.DEFAULT_TREE) / "C002"
    target.parent.mkdir(parents=True)
    c002.rename(target)
    (tmp_path / "job-b" / "output" / "observations").mkdir()
    outline().to_file(tmp_path / "job-b" / "output" / "observations" / mb.DEFAULT_OUTLINE)

    root = str(tmp_path / "*" / "output")
    found = mb.find_files(root, ["acabf"])
    # <job>/output/GrIS/UAF/PISM/CORE/<counter>/<file>: the job is six levels up.
    assert {Path(f).parents[6].name for f in found} == {"job-a", "job-b"}
    assert mb.resolve_outline(root, None) == str(tmp_path / "job-b" / "output" / "observations" / mb.DEFAULT_OUTLINE)
    assert mb.first_match(root, "observations/nothing.nc") is None
    ds = mb.open_submission(found)
    assert sorted(ds["gcm_id"].values) == ["CESM2-WACCM", "MRI-ESM2-0"]
    with pytest.raises(FileNotFoundError, match="no lithk files"):
        mb.find_files(root, ["lithk"])


def test_find_files_lists_the_tree(tree: Path):
    """
    Every file of the requested variables is found, in name order, as a plain path.

    Parameters
    ----------
    tree : pathlib.Path
        The run's ``output`` directory.
    """
    found = mb.find_files(str(tree), ["acabf", "ligroundf"])
    assert len(found) == 7
    assert all(not f.startswith("file://") for f in found)
    with pytest.raises(FileNotFoundError):
        mb.find_files(str(tree), ["lithk"])


def test_open_submission_skips_empty_files_and_stacks_gcm_and_pathway(tree: Path):
    """
    The empty file is dropped and the rest become one lazy ensemble on (gcm_id, ssp_id).

    Parameters
    ----------
    tree : pathlib.Path
        The run's ``output`` directory.
    """
    ds = mb.open_submission(mb.find_files(str(tree), ["acabf", "ligroundf"]))
    assert set(ds.data_vars) >= {"acabf", "ligroundf"}
    assert sorted(ds["gcm_id"].values) == ["CESM2-WACCM", "MRI-ESM2-0"]
    assert sorted(ds["ssp_id"].values) == ["historical", "ssp585"]
    assert ds.sizes["time"] == 4
    assert ds["acabf"].chunks is not None, "still lazy"
    # A pathway a GCM never ran is missing, not zero.
    assert np.isnan(ds["acabf"].sel(gcm_id="MRI-ESM2-0", ssp_id="ssp585").values).all()
    with pytest.raises(FileNotFoundError):
        mb.open_submission(
            [
                str(
                    tree.joinpath(*mb.DEFAULT_TREE, "C007")
                    / "acabf_GrIS_UAF_PISM_m001_MRI-ESM2-0_f001_ssp585_C008_2015-2299.nc"
                )
            ]
        )


def test_regional_sums_match_sums_done_by_hand(tree: Path):
    """
    Each basin's integral is level x cells x cell area, with the area and the total appended.

    Parameters
    ----------
    tree : pathlib.Path
        The run's ``output`` directory.
    """
    ds = mb.open_submission(mb.find_files(str(tree), ["acabf", "ligroundf"]))
    sums = mb.regional_sums(ds, outline()).compute()
    assert list(sums["region"].values) == ["GIS_W", "GIS_E", mb.TOTAL_REGION]
    assert sums["acabf"].attrs["units"] == "kg / s"
    for (gcm, ssp, var), level in LEVELS.items():
        for name, (c0, c1) in BASINS.items():
            expected = level * (c1 - c0) * NY * DX * DX
            np.testing.assert_allclose(sums[var].sel(gcm_id=gcm, ssp_id=ssp, region=name).dropna("time"), expected)
        total = sums[var].sel(gcm_id=gcm, ssp_id=ssp, region=mb.TOTAL_REGION).dropna("time")
        np.testing.assert_allclose(total, level * NX * NY * DX * DX)
    np.testing.assert_allclose(sums["area"].sel(region="GIS_W"), 4 * NY * DX * DX)
    assert "mapping" not in sums.variables


def test_compute_regions_builds_the_mass_balance(tree: Path):
    """
    The fluxes come out in Gt/yr, the mass balance is their sum, the cumulative series is zero at the reference year.

    Parameters
    ----------
    tree : pathlib.Path
        The run's ``output`` directory.
    """
    ds = mb.open_submission(mb.find_files(str(tree), ["acabf", "ligroundf"]))
    regions = mb.compute_regions(ds, outline(), variables=["acabf", "ligroundf"], reference_year="2013")
    assert regions["acabf"].attrs["units"] == mb.FLUX_UNITS
    cesm = regions.sel(gcm_id="CESM2-WACCM", ssp_id="historical", region="GIS_W").isel(time=0)
    # 1 kg m-2 s-1 over 4 x 8 km^2 = 32e6 kg/s = 32e6 * 3.15576e7 s/yr / 1e12 kg/Gt
    np.testing.assert_allclose(cesm["acabf"], 32e6 * 3.15576e7 / 1e12, rtol=1e-3)
    np.testing.assert_allclose(regions["mass_balance"], regions["acabf"] + regions["ligroundf"])
    zero = regions["cumulative_mass_balance"].sel(time="2013", gcm_id="CESM2-WACCM", ssp_id="historical")
    np.testing.assert_allclose(zero, 0.0, atol=1e-9)


def test_run_writes_the_series_and_the_figure(tree: Path, tmp_path: Path):
    """
    The driver works on a local tree without observations and can plot again from its own output.

    Parameters
    ----------
    tree : pathlib.Path
        The run's ``output`` directory.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    outline_file = tmp_path / "basins.gpkg"
    outline().to_file(outline_file)
    out = tmp_path / "figures"
    regions = mb.run(str(tree), out, variables=["acabf", "ligroundf"], outline=str(outline_file), reference_year="2013")
    assert (out / "regional_mass_balance.nc").is_file()
    assert (out / "regional_mass_balance.csv").is_file()
    assert (out / "regional_mass_balance.png").is_file()
    again = mb.run(str(tree), out, regions_file=out / "regional_mass_balance.nc")
    xr.testing.assert_allclose(again, regions)


def test_resolve_outline_falls_back_to_the_packaged_file(tree: Path):
    """
    A bare name missing from the run's observations resolves to the package's copy.

    Parameters
    ----------
    tree : pathlib.Path
        The run's ``output`` directory.
    """
    resolved = mb.resolve_outline(str(tree), None)
    assert resolved.endswith(mb.DEFAULT_OUTLINE) and Path(resolved).is_file()
    (tree / "observations").mkdir()
    outline().to_file(tree / "observations" / mb.DEFAULT_OUTLINE)
    assert mb.resolve_outline(str(tree), None) == str(tree / "observations" / mb.DEFAULT_OUTLINE)


def test_splice_historical_fills_the_pathways(tree: Path):
    """
    Each pathway gets its GCM's historical values where it has none.

    Parameters
    ----------
    tree : pathlib.Path
        The run's ``output`` directory.
    """
    ds = mb.open_submission(mb.find_files(str(tree), ["acabf"]))
    spliced = mb.splice_historical(ds)
    assert list(spliced["ssp_id"].values) == ["ssp585"]
    cesm = spliced["acabf"].sel(gcm_id="CESM2-WACCM", ssp_id="ssp585").isel(y=0, x=0).compute()
    np.testing.assert_allclose(cesm.values, [1.0, 1.0, 3.0, 3.0])
