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
Tests for the observed thickness-change product.

The Smith et al. (2020) archive states a mean dH/dt over 2003-2019 with a
per-cell RMSE; a run reports cumulative thickness change. Turning one into
the other hinges on the interval, which the rasters do not state, and on the
two fields staying consistent with each other -- which is what these test,
from a miniature stand-in for the archive.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import cftime
import numpy as np
import pytest
import rioxarray  # pylint: disable=unused-import
import xarray as xr

from pism_terra.ismip7.greenland.forcing import (
    DAYS_PER_YEAR,
    DH_SMITH_CRS,
    DH_SMITH_END,
    DH_SMITH_NAME,
    DH_SMITH_RATE,
    DH_SMITH_RMSE,
    DH_SMITH_START,
    OBS_CALENDAR,
    OBS_TIME_UNITS,
    cumulative_dh,
    prepare_dh_observations,
    smith_rasters,
)

#: Cell size of the synthetic archive, metres -- the archive's own 5 km.
SPACING = 5000.0
N = 6


def _days(year: int, month: int = 1, day: int = 1) -> float:
    """
    Encode a date in the product's time units.

    Parameters
    ----------
    year, month, day : int
        Calendar date.

    Returns
    -------
    float
        Days since the reference epoch.
    """
    return float(
        cftime.date2num(cftime.datetime(year, month, day, calendar=OBS_CALENDAR), OBS_TIME_UNITS, OBS_CALENDAR)
    )


def write_raster(path: Path, values: np.ndarray) -> None:
    """
    Write one GeoTIFF the way the archive ships them.

    On the ISMIP7 Greenland projection with ``y`` descending and NaN for the
    cells the survey never saw.

    Parameters
    ----------
    path : pathlib.Path
        File to write.
    values : numpy.ndarray
        Field of shape ``(N, N)``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(
        values.astype("float32"),
        dims=("y", "x"),
        coords={
            "y": -676440.0 - SPACING * np.arange(N),
            "x": -626302.0 + SPACING * np.arange(N),
        },
    )
    da.rio.write_crs(DH_SMITH_CRS).rio.write_nodata(np.nan).rio.to_raster(path)


def write_archive(directory: Path, *, zipped: bool = False, rmse_holes: bool = False) -> Path:
    """
    Build a miniature stand-in for the Smith archive.

    A rate that thins toward one side, a hole the survey never saw, and an
    RMSE beside it -- laid out in the ``dhdt`` subdirectory the real archive
    uses, with the decoys that must not be picked up.

    Parameters
    ----------
    directory : pathlib.Path
        Directory to build under.
    zipped : bool, optional
        Return a zip archive rather than the directory.
    rmse_holes : bool, optional
        Leave a cell that has a rate but no RMSE.

    Returns
    -------
    pathlib.Path
        The archive directory, or the zip file.
    """
    root = directory / "ICESat1_ICESat2_mass_change_updated_2_2021"
    rate = np.tile(np.linspace(-2.0, 0.0, N), (N, 1))
    rate[0, 0] = np.nan  # a cell the survey never saw
    rmse = np.full((N, N), 0.25, dtype="float32")
    rmse[0, 0] = np.nan
    if rmse_holes:
        rmse[1, 1] = np.nan  # a rate with no uncertainty beside it

    write_raster(root / "dhdt" / DH_SMITH_RATE, rate)
    write_raster(root / "dhdt" / DH_SMITH_RMSE, rmse)
    # Decoys: the smoothed display grid, and the Antarctic pair.
    write_raster(root / "dhdt" / "gris_dhdt_filt.tif", np.zeros((N, N)))
    write_raster(root / "dhdt" / "ais_dhdt_grounded.tif", np.zeros((N, N)))
    (root / "README.txt").write_text("stand-in\n")

    if not zipped:
        return root
    archive = directory / "smith.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for item in sorted(root.rglob("*")):
            if item.is_file():
                zf.write(item, item.relative_to(directory))
    return archive


@pytest.fixture(name="archive")
def fixture_archive(tmp_path: Path) -> Path:
    """
    A miniature archive directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    pathlib.Path
        The archive directory.
    """
    return write_archive(tmp_path / "src")


def test_cumulative_dh_accumulates_and_keeps_the_start_in_bounds():
    """
    Accumulate from the first interval's start, and say so in the bounds.

    A run reports change since its own start, so every record's lower bound
    is that same first epoch rather than the previous record's.
    """
    times = np.array([_days(y, 7, 1) for y in (2003, 2004, 2005)])
    bounds = np.stack([[_days(y) for y in (2003, 2004, 2005)], [_days(y) for y in (2004, 2005, 2006)]], axis=1)
    rate = xr.DataArray(np.full((3, 2, 2), -0.5, dtype="float32"), dims=("t", "y", "x"), coords={"t": times})
    out = cumulative_dh(rate, bounds, "t")

    spans = (bounds[:, 1] - bounds[:, 0]) / DAYS_PER_YEAR
    np.testing.assert_allclose(out["dh"].values[:, 0, 0], -0.5 * np.cumsum(spans), rtol=1e-6)
    # Every lower bound is the first interval's start.
    assert (out["time_bnds"].values[:, 0] == bounds[0, 0]).all()
    np.testing.assert_array_equal(out["time_bnds"].values[:, 1], bounds[:, 1])


def test_cumulative_dh_does_not_turn_gaps_into_zeros():
    """
    Keep a cell the survey never saw missing.

    ``cumsum`` reads NaN as zero, which would quietly report "no change"
    for a cell that was never measured -- the one value a calibration must
    not be handed.
    """
    times = np.array([_days(y, 7, 1) for y in (2003, 2004)])
    bounds = np.stack([[_days(2003), _days(2004)], [_days(2004), _days(2005)]], axis=1)
    values = np.full((2, 2, 2), -0.5, dtype="float32")
    values[:, 0, 0] = np.nan
    rate = xr.DataArray(values, dims=("t", "y", "x"), coords={"t": times})

    out = cumulative_dh(rate, bounds, "t")
    assert np.isnan(out["dh"].values[:, 0, 0]).all()
    assert np.isfinite(out["dh"].values[:, 1, 1]).all()


def test_smith_rasters_picks_the_unfiltered_greenland_pair(archive: Path):
    """
    Take the raw Greenland rate, not the smoothed one and not Antarctica.

    The archive ships ``gris_dhdt_filt.tif`` beside the real thing; the
    README marks it as smoothed for display only, so integrating it would
    quietly report a different ice sheet.

    Parameters
    ----------
    archive : pathlib.Path
        Miniature archive directory.
    """
    rate, rmse = smith_rasters(archive)
    assert rate.name == DH_SMITH_RATE
    assert rmse.name == DH_SMITH_RMSE
    assert "filt" not in rate.name and "ais" not in rate.name


def test_smith_rasters_reads_a_zip(tmp_path: Path):
    """
    Accept the archive as downloaded, not only unpacked.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    zipped = write_archive(tmp_path / "src", zipped=True)
    rate, rmse = smith_rasters(zipped)
    assert rate.name == DH_SMITH_RATE and rmse.name == DH_SMITH_RMSE


def test_smith_rasters_rejects_what_is_neither(tmp_path: Path):
    """
    Say so rather than searching a file that is not an archive.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    stray = tmp_path / "notes.txt"
    stray.write_text("not an archive")
    with pytest.raises(FileNotFoundError, match="neither a directory nor a zip"):
        smith_rasters(stray)


def test_smith_rasters_needs_both_fields(tmp_path: Path):
    """
    Refuse an archive missing the RMSE.

    The uncertainty is the reason for using this archive; preparing the rate
    alone would silently produce a product the likelihood cannot weight.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    root = write_archive(tmp_path / "src")
    (root / "dhdt" / DH_SMITH_RMSE).unlink()
    with pytest.raises(FileNotFoundError, match=DH_SMITH_RMSE):
        smith_rasters(root)


def test_prepare_integrates_the_rate_over_the_stated_period(tmp_path: Path, archive: Path):
    """
    Integrate over 2003-2019 and carry the RMSE through the same interval.

    The rasters state no time at all, so the interval comes from the paper;
    getting it wrong scales every value. The RMSE describes the one 16-year
    rate rather than sixteen annual ones, so it scales with the interval, not
    with its square root.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    archive : pathlib.Path
        Miniature archive directory.
    """
    products = prepare_dh_observations(archive, tmp_path / "out")
    assert set(products) == {"smith"}
    assert products["smith"].name == DH_SMITH_NAME

    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(products["smith"], decode_times=coder) as ds:
        assert ds.sizes["time"] == 1
        assert f"{ds['time_bnds'].values[0][0]:%Y-%m-%d}" == DH_SMITH_START
        assert f"{ds['time_bnds'].values[0][1]:%Y-%m-%d}" == DH_SMITH_END

        span = (_days(2019) - _days(2003)) / DAYS_PER_YEAR
        rate = np.tile(np.linspace(-2.0, 0.0, N), (N, 1))
        expected = rate * span
        actual = ds["dh"].isel(time=0).values
        finite = np.isfinite(expected) & np.isfinite(actual)
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-5)

        np.testing.assert_allclose(ds["dh_error"].isel(time=0).values[1:, 1:], 0.25 * span, rtol=1e-5)
        assert ds["dh"].attrs["units"] == ds["dh_error"].attrs["units"] == "m"


def test_prepare_keeps_the_archives_own_grid(tmp_path: Path, archive: Path):
    """
    Leave the product on the 5 km grid it arrived on.

    Upsampling to the 1 km submission grid would invent a resolution the
    survey does not have and make neighbouring uncertainties perfectly
    correlated; which grid a comparison happens on is the comparison's
    business. The projection is already the ISMIP7 one, so nothing is
    reprojected either.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    archive : pathlib.Path
        Miniature archive directory.
    """
    products = prepare_dh_observations(archive, tmp_path / "out")
    with xr.open_dataset(products["smith"], decode_times=False) as ds:
        assert list(ds["dh"].dims) == ["time", "y", "x"]
        assert abs(float(ds["x"][1] - ds["x"][0])) == SPACING
        assert abs(float(ds["y"][1] - ds["y"][0])) == SPACING
        assert "mapping" in ds
        assert ds["dh"].attrs["grid_mapping"] == "mapping"
        assert "3413" in ds["mapping"].attrs["crs_wkt"]


def test_prepare_drops_cells_without_an_uncertainty(tmp_path: Path):
    """
    Never leave a value whose uncertainty is missing.

    A NaN uncertainty takes the log-likelihood of every member to NaN, which
    destroys the whole posterior rather than one cell: the cell has to go.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    root = write_archive(tmp_path / "src", rmse_holes=True)
    products = prepare_dh_observations(root, tmp_path / "out")
    with xr.open_dataset(products["smith"], decode_times=False) as ds:
        dh = ds["dh"].isel(time=0).values
        err = ds["dh_error"].isel(time=0).values
        assert np.isnan(dh[1, 1]), "a rate with no RMSE must not survive"
        # Nowhere is there a value without an uncertainty beside it.
        assert not (np.isfinite(dh) & ~np.isfinite(err)).any()


def test_prepare_does_not_stamp_the_tiff_metadata_onto_the_field(tmp_path: Path, archive: Path):
    """
    Describe the field, not the file it came out of.

    GDAL hands back the TIFF's own tags and an identity ``scale_factor`` /
    ``add_offset``; left in place the last of those would be applied a second
    time by anything that decodes the file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    archive : pathlib.Path
        Miniature archive directory.
    """
    products = prepare_dh_observations(archive, tmp_path / "out")
    with xr.open_dataset(products["smith"], decode_times=False) as ds:
        for name in ("dh", "dh_error"):
            assert not [k for k in ds[name].attrs if "TIFF" in k or k in ("scale_factor", "add_offset")]


def test_prepare_writes_a_decodable_time_axis(tmp_path: Path, archive: Path):
    """
    Keep the CF time attributes through to the file.

    Assigning a variable carrying a bare ``time`` replaces the coordinate,
    attributes and all; without them the axis reads back as plain integers
    and nothing aligns against a run.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    archive : pathlib.Path
        Miniature archive directory.
    """
    products = prepare_dh_observations(archive, tmp_path / "out")
    with xr.open_dataset(products["smith"], decode_times=False) as raw:
        assert raw["time"].attrs["units"] == OBS_TIME_UNITS
        assert raw["time"].attrs["calendar"] == OBS_CALENDAR
        assert raw["time"].attrs["bounds"] == "time_bnds"

    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(products["smith"], decode_times=coder) as ds:
        assert f"{ds['time'].values[0]:%Y-%m-%d}" == DH_SMITH_END


def test_prepare_reuses_existing_products(tmp_path: Path, archive: Path):
    """
    Do not rebuild what is already there unless asked.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    archive : pathlib.Path
        Miniature archive directory.
    """
    first = prepare_dh_observations(archive, tmp_path / "out")
    stamp = first["smith"].stat().st_mtime_ns
    prepare_dh_observations(archive, tmp_path / "out")
    assert first["smith"].stat().st_mtime_ns == stamp

    prepare_dh_observations(archive, tmp_path / "out", force_overwrite=True)
    assert first["smith"].stat().st_mtime_ns != stamp
