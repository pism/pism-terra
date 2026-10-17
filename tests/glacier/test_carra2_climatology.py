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
Tests for the CARRA2 monthly-mean climatology.

The climatology is a 12-step periodic forcing PISM cycles, so two things have
to hold: the averaging is over a fixed reference period (not "whatever is in
the store"), and the written file carries a time axis and a grid mapping PISM
can actually read.
"""

import cftime
import netCDF4
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS, Transformer
from rioxarray.raster_array import RasterArray

from pism_terra.glacier import climate
from pism_terra.glacier.climate import (
    CARRA2_CLIMATOLOGY_YEARS,
    _carra2_fill_years_and_bounds,
    _promote_grid_mapping,
    files_by_year,
    has_variables,
    open_carra2_zarr,
    prepare_carra2_monthly_mean,
    stamp_monthly_climatology_axis,
)
from pism_terra.workflow import compressed_encoding, stamp_grid_mapping


def carra2_store(path, first_year: int = 1986, last_year: int = 2026):
    """
    Write a CARRA2-shaped Zarr store with a value we can predict exactly.

    Every time-varying field is ``month * 100 + (year - first_year)`` times a
    per-variable scale, so the mean over any set of years is known in closed
    form.

    Parameters
    ----------
    path : pathlib.Path
        Destination Zarr store.
    first_year : int, default 1986
        First year in the store.
    last_year : int, default 2026
        One past the last year in the store.

    Returns
    -------
    pathlib.Path
        The written store.
    """
    time = xr.date_range(f"{first_year}-01-01", periods=(last_year - first_year) * 12, freq="MS")
    stamps = pd.to_datetime(time)
    base = (np.array([t.month for t in stamps]) * 100 + (np.array([t.year for t in stamps]) - first_year)).astype(
        "float64"
    )

    def field(scale):
        """
        Broadcast the predictable signal over a small spatial grid.

        Parameters
        ----------
        scale : float
            Per-variable multiplier.

        Returns
        -------
        numpy.ndarray
            Array shaped ``(time, y, x)``.
        """
        return base[:, None, None] * scale * np.ones((1, 4, 5))

    ds = xr.Dataset(
        {
            "air_temp": (("time", "y", "x"), field(1.0)),
            "precipitation": (("time", "y", "x"), field(0.1)),
            "snowfall": (("time", "y", "x"), field(0.05)),
            "orography": (("y", "x"), np.arange(20.0).reshape(4, 5)),
            "time_bnds": (("time", "bnds"), np.zeros((len(time), 2))),
        },
        coords={"time": time, "y": np.linspace(0, 3000, 4), "x": np.linspace(0, 4000, 5)},
    )
    ds = ds.rio.write_crs("EPSG:3413").rio.write_grid_mapping()
    ds.to_zarr(path, mode="w", consolidated=True)
    return path


@pytest.fixture(name="climatology")
def fixture_climatology(tmp_path):
    """
    Build the climatology from a synthetic store.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.

    Returns
    -------
    xarray.Dataset
        The written climatology, reopened.
    """
    source = carra2_store(tmp_path / "carra2.zarr")
    out = prepare_carra2_monthly_mean(source, tmp_path / "carra2_monthly_mean.zarr")
    return xr.open_zarr(out, consolidated=True)


def test_averages_over_the_fixed_reference_period(climatology):
    """
    Average 1990-2019, not whatever the store happens to hold.

    Parameters
    ----------
    climatology : xarray.Dataset
        Climatology built from the synthetic store.
    """
    years = list(CARRA2_CLIMATOLOGY_YEARS)
    assert (years[0], years[-1]) == (1990, 2019)
    # Source value is month*100 + (year - 1986), so the mean is known exactly.
    offset = np.mean(np.array(years) - 1986)

    assert climatology.sizes["time"] == 12
    for month in (1, 7, 12):
        assert float(climatology["air_temp"].sel(time=month).mean()) == pytest.approx(month * 100 + offset)
    assert climatology.attrs["climatology_period"] == "1990-2019"


def test_time_invariant_fields_survive(climatology):
    """
    Leave orography two-dimensional and drop the stale source bounds.

    Parameters
    ----------
    climatology : xarray.Dataset
        Climatology built from the synthetic store.
    """
    assert climatology["orography"].dims == ("y", "x")
    assert "time_bnds" not in climatology.variables


def test_month_axis_is_unambiguous(climatology):
    """
    Label the intermediate by calendar month, 1 to 12.

    The CF climatological axis is stamped when a per-glacier file is written;
    keeping the intermediate as plain month numbers means it survives the
    per-group reprojection round trip without being reinterpreted.

    Parameters
    ----------
    climatology : xarray.Dataset
        Climatology built from the synthetic store.
    """
    assert list(climatology["time"].values) == list(range(1, 13))


def test_crs_is_recoverable(tmp_path, climatology):
    """
    Keep the store self-describing after the groupby.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    climatology : xarray.Dataset
        Climatology built from the synthetic store.
    """
    _ = climatology
    assert open_carra2_zarr(tmp_path / "carra2_monthly_mean.zarr").rio.crs is not None


def test_a_period_outside_the_store_is_rejected(tmp_path):
    """
    Fail loudly rather than writing an empty climatology.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    source = carra2_store(tmp_path / "short.zarr", first_year=2020, last_year=2024)

    with pytest.raises(ValueError, match="no time steps in 1990-2019"):
        prepare_carra2_monthly_mean(source, tmp_path / "out.zarr")


def test_cached_output_is_reused(tmp_path, climatology):
    """
    Skip the work when the store is already there.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    climatology : xarray.Dataset
        Climatology built from the synthetic store.
    """
    _ = climatology
    # A missing source would raise if the function did not short-circuit.
    again = prepare_carra2_monthly_mean(tmp_path / "does-not-exist.zarr", tmp_path / "carra2_monthly_mean.zarr")

    assert again == tmp_path / "carra2_monthly_mean.zarr"


def test_climatological_axis_matches_era5(climatology):
    """
    Place the 12 steps mid-month on a 365-day year spanning exactly one year.

    Parameters
    ----------
    climatology : xarray.Dataset
        Climatology built from the synthetic store.
    """
    stamped = stamp_monthly_climatology_axis(climatology.drop_vars("spatial_ref", errors="ignore"))

    assert stamped["time"].attrs["units"] == "days since 0001-01-01"
    assert stamped["time"].attrs["calendar"] == "365_day"
    assert stamped["time"].attrs["bounds"] == "time_bounds"
    np.testing.assert_allclose(stamped["time"].values[:2], [15.5, 45.0])
    bounds = stamped["time_bounds"].values
    assert (bounds[0][0], bounds[-1][1]) == (0.0, 365.0)
    # Bounds tile the year with no gaps.
    np.testing.assert_array_equal(bounds[1:, 0], bounds[:-1, 1])


def test_a_series_that_is_not_twelve_months_is_rejected(climatology):
    """
    Refuse to stamp a climatological axis on something that is not one.

    Parameters
    ----------
    climatology : xarray.Dataset
        Climatology built from the synthetic store.
    """
    with pytest.raises(ValueError, match="needs 12 time steps"):
        stamp_monthly_climatology_axis(climatology.isel(time=slice(0, 6)))


def test_compressed_encoding_keeps_the_grid_mapping(tmp_path):
    """
    Carry ``grid_mapping`` through into the written file.

    Handing ``to_netcdf`` an encoding dict replaces a variable's encoding, so
    asking for compression the obvious way drops the pointer to the CRS
    variable and PISM can no longer find the projection.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    ds = xr.Dataset(
        {"air_temp": (("y", "x"), np.ones((3, 4), "float32"))},
        coords={"y": np.arange(3.0), "x": np.arange(4.0)},
    )
    ds = stamp_grid_mapping(ds.rio.write_crs("EPSG:3338").rio.write_grid_mapping())

    naive, safe = tmp_path / "naive.nc", tmp_path / "safe.nc"
    ds.to_netcdf(naive, encoding={"air_temp": {"zlib": True, "complevel": 2}}, engine="h5netcdf")
    ds.to_netcdf(safe, encoding=compressed_encoding(ds), engine="h5netcdf")

    with netCDF4.Dataset(naive) as nc:  # pylint: disable=no-member
        assert not hasattr(nc.variables["air_temp"], "grid_mapping")
    with netCDF4.Dataset(safe) as nc:  # pylint: disable=no-member
        assert nc.variables["air_temp"].grid_mapping == "spatial_ref"
    assert xr.open_dataset(safe, decode_coords="all").rio.crs is not None


def test_snowfall_is_carried_into_the_climatology(climatology):
    """
    Average snowfall like every other time-varying field of the store.

    Nothing downstream of the store names its variables, so a field added to
    the download reaches the climatology -- and the per-glacier files cut
    from it -- without being listed anywhere.

    Parameters
    ----------
    climatology : xarray.Dataset
        Climatology built from the synthetic store.
    """
    offset = np.mean(np.array(list(CARRA2_CLIMATOLOGY_YEARS)) - 1986)
    assert float(climatology["snowfall"].sel(time=7).mean()) == pytest.approx((7 * 100 + offset) * 0.05)
    # Half the precipitation in the synthetic store, as it was put in.
    ratio = climatology["snowfall"] / climatology["precipitation"]
    assert float(ratio.min()) == pytest.approx(0.5) and float(ratio.max()) == pytest.approx(0.5)


def test_a_cache_that_predates_snowfall_is_not_current(tmp_path):
    """
    Tell a store or batch written before a variable existed from a current one.

    Both open and pass the health check, so without this the yearly batches
    and the store of an existing staging tree would be reused as they are and
    snowfall would never be built.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    store = carra2_store(tmp_path / "carra2.zarr", last_year=1988)
    assert has_variables(store, ["snowfall"])
    assert has_variables(store, ["precipitation", "snowfall", "air_temp"])
    assert not has_variables(store, ["snowfall", "rainfall"])

    batch = tmp_path / "batch_1986.nc"
    xr.open_zarr(store).drop_vars("snowfall").load().to_netcdf(batch)
    assert has_variables(batch, ["precipitation"])
    assert not has_variables(batch, ["snowfall"])

    assert not has_variables(tmp_path / "missing.nc", ["snowfall"])


@pytest.mark.parametrize("calendar", ["numpy", "cftime"])
def test_carra2_bounds_are_those_of_the_calendar_month(calendar: str):
    """
    Give a mid-month stamp the bounds of its month, and keep every step.

    CARRA2 is stamped on the 15th. Pairing each stamp with the same day of
    the following month gave January the bounds 15 January to 15 February,
    which applies every monthly mean half a month late.

    Parameters
    ----------
    calendar : str
        Whether the stamps are ``numpy.datetime64`` or cftime objects.
    """
    stamps = [pd.Timestamp(year=2024, month=month, day=15) for month in range(1, 13)]
    if calendar == "cftime":
        time = [cftime.DatetimeGregorian(stamp.year, stamp.month, stamp.day) for stamp in stamps]
    else:
        time = stamps
    ds = xr.Dataset(
        {"air_temp": (("time", "y", "x"), np.zeros((12, 2, 2)))}, coords={"time": time, "y": [0, 1], "x": [0, 1]}
    )

    out = _carra2_fill_years_and_bounds(ds, [2024, 2025])

    assert out.sizes["time"] == 24
    bounds = out["time_bnds"].values
    as_dates = [[(value.year, value.month, value.day) for value in map(pd.Timestamp, map(str, row))] for row in bounds]
    assert as_dates[0] == [(2024, 1, 1), (2024, 2, 1)]
    assert as_dates[1] == [(2024, 2, 1), (2024, 3, 1)]  # a leap February
    assert as_dates[11] == [(2024, 12, 1), (2025, 1, 1)]
    assert as_dates[-1] == [(2025, 12, 1), (2026, 1, 1)]
    # The months tile the two years, and each stamp lies inside its own bounds.
    assert all(as_dates[i][1] == as_dates[i + 1][0] for i in range(23))
    assert out["time"].attrs["bounds"] == "time_bnds"


def test_yearly_downloads_are_paired_by_year_not_by_position(tmp_path):
    """
    Refuse to go on when a year failed to download, and never shift the others.

    CDS failed two of forty yearly temperature requests in one run. The files
    that did arrive, taken in order, would have put 1988's temperature under
    1987 and so on down the line, each restamped with the wrong year.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    years = [1986, 1987, 1988, 1989]
    complete = [tmp_path / f"_cds_carra_2m_temperature_abc123_{year}.nc" for year in years]

    by_year = files_by_year(reversed(complete), years, "2 m temperature")
    assert [by_year[year].name for year in years] == [file.name for file in complete]

    arrived = [file for file in complete if "_1987" not in file.name]
    with pytest.raises(RuntimeError, match="2 m temperature: no download for 1987"):
        files_by_year(arrived, years, "2 m temperature")
    # A file of a year that was not asked for is ignored rather than paired with anything.
    assert list(files_by_year(complete, [1988], "2 m temperature")) == [1988]


def _grid_file(tmp_path, crs_wkt=None):
    """
    Write a PISM grid file laid out like pism-terra's ``grid_<rgi_id>.nc``.

    One ``domain`` variable pointing at a 0-d ``spatial_ref``, with ``x_bnds`` and
    ``y_bnds`` cell bounds: 120 x 120 cells of 100 m around Storglaciaren.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory to write ``grid.nc`` into.
    crs_wkt : str or None, optional
        WKT stored as the ``crs_wkt`` of ``spatial_ref``; ``None`` writes a grid
        mapping without one.

    Returns
    -------
    pathlib.Path
        Path to the written grid file.
    """
    x = np.arange(392050.0, 404000.0, 100.0)
    y = np.arange(7529050.0, 7541000.0, 100.0)
    ds = xr.Dataset(
        {
            "domain": (("y", "x"), np.zeros((y.size, x.size), dtype="int64"), {"grid_mapping": "spatial_ref"}),
            "x_bnds": (("x", "nv2"), np.stack([x - 50, x + 50], axis=1)),
            "y_bnds": (("y", "nv2"), np.stack([y - 50, y + 50], axis=1)),
            "spatial_ref": ((), 0, {} if crs_wkt is None else {"crs_wkt": crs_wkt}),
        },
        coords={"x": ("x", x, {"bounds": "x_bnds"}), "y": ("y", y, {"bounds": "y_bnds"})},
    )
    path = tmp_path / "grid.nc"
    ds.to_netcdf(path)
    return path


def test_a_grid_opened_without_decode_coords_gets_its_crs(tmp_path):
    """
    Recover the CRS of a grid whose grid mapping xarray left as a data variable.

    Plain ``xr.open_dataset`` keeps ``spatial_ref`` out of the coordinates, so
    rioxarray finds no CRS and ``reproject_match`` fails with "Missing dst_crs".

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    path = _grid_file(tmp_path, CRS("EPSG:32634").to_wkt())
    with xr.open_dataset(path) as grid:
        assert grid.rio.crs is None  # the failure mode: spatial_ref is a data variable
        promoted = _promote_grid_mapping(grid)
        assert promoted.rio.crs.to_epsg() == 32634
        assert "spatial_ref" in promoted.coords
        assert promoted.rio.shape == grid.rio.shape


def test_a_grid_that_already_has_a_crs_is_untouched(tmp_path):
    """
    Return a grid that already has a usable CRS as it is.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    path = _grid_file(tmp_path, CRS("EPSG:32634").to_wkt())
    with xr.open_dataset(path, decode_coords="all") as grid:
        assert _promote_grid_mapping(grid) is grid


def test_a_grid_without_a_crs_is_rejected(tmp_path):
    """
    Fail early, and say why, on a grid with no CRS to recover.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    """
    with xr.open_dataset(_grid_file(tmp_path)) as grid:
        with pytest.raises(ValueError, match="CRS of the target grid"):
            _promote_grid_mapping(grid)


def _fake_carra2_store(years):
    """
    Build a small CARRA2-like store on EPSG:3413 around the test grid.

    Every cell of ``air_temp`` and ``precipitation`` holds ``100 * year + month``,
    so the output shows which source month it came from.

    Parameters
    ----------
    years : sequence of int
        Years in the store, twelve mid-month steps each.

    Returns
    -------
    xarray.Dataset
        The store, with its CRS and spatial dims set as ``open_carra2_zarr`` would.
    """
    cx, cy = Transformer.from_crs("EPSG:32634", "EPSG:3413", always_xy=True).transform(398000.0, 7535000.0)
    x = cx + 2500.0 * np.arange(-6, 7)
    y = cy + 2500.0 * np.arange(-6, 7)
    time = pd.to_datetime([f"{yr}-{m:02d}-15" for yr in years for m in range(1, 13)])
    code = np.array([100.0 * t.year + t.month for t in time])[:, None, None]
    field = np.broadcast_to(code, (time.size, y.size, x.size)).astype("float32")
    ds = xr.Dataset(
        {
            "air_temp": (("time", "y", "x"), field.copy()),
            "precipitation": (("time", "y", "x"), field.copy()),
            "orography": (("y", "x"), np.full((y.size, x.size), 1500.0, dtype="float32")),
        },
        coords={"time": time, "x": x, "y": y},
    )
    return ds.rio.write_crs("EPSG:3413").rio.set_spatial_dims(x_dim="x", y_dim="y")


class _NoPrebuiltS3:
    """
    Stand in for ``s3fs.S3FileSystem`` with no pre-reprojected CARRA2 file.

    Parameters
    ----------
    *args, **kwargs
        Accepted and ignored, like the real constructor's options.
    """

    def __init__(self, *args, **kwargs):
        """
        Accept the real constructor's options and ignore them.

        Parameters
        ----------
        *args, **kwargs
            Ignored.
        """
        _ = args, kwargs

    def exists(self, uri):
        """
        Report that nothing exists, so ``carra2`` takes the reprojection path.

        Parameters
        ----------
        uri : str
            Ignored.

        Returns
        -------
        bool
            Always ``False``.
        """
        _ = uri
        return False


@pytest.mark.parametrize(
    "store_years, requested, source",
    [((1985, 1986, 1987, 1988), 1986, 1986), ((1985, 1990), 1987, 1985)],
)
def test_carra2_reprojects_only_the_source_years_it_uses(
    tmp_path, monkeypatch, capsys, store_years, requested, source
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """
    Reproject only the months the requested years are filled from.

    ``carra2`` used to reproject every step in the store and pick the requested
    years afterwards: a one-year run read and warped the whole forty-year record.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided scratch directory.
    monkeypatch : pytest.MonkeyPatch
        Replaces the S3 access and counts the reprojected steps.
    capsys : pytest.CaptureFixture
        Captures the log naming the source years used.
    store_years : tuple of int
        Years in the fake CARRA2 store.
    requested : int
        The one year asked for.
    source : int
        The store year that ``requested`` is filled from.
    """
    reprojected_steps = []
    real_reproject_match = RasterArray.reproject_match

    def counting_reproject_match(self, *args, **kwargs):
        """
        Record the number of time steps, then reproject as usual.

        Parameters
        ----------
        self : rioxarray.raster_array.RasterArray
            The ``.rio`` accessor being reprojected.
        *args, **kwargs
            Passed on to the real ``reproject_match``.

        Returns
        -------
        xarray.DataArray
            The reprojected array.
        """
        reprojected_steps.append(self._obj.sizes.get("time", 0))  # pylint: disable=protected-access
        return real_reproject_match(self, *args, **kwargs)

    monkeypatch.setattr(climate.s3fs, "S3FileSystem", _NoPrebuiltS3)
    monkeypatch.setattr(climate, "open_carra2_zarr", lambda uri: _fake_carra2_store(store_years))
    monkeypatch.setattr(RasterArray, "reproject_match", counting_reproject_match)

    with xr.open_dataset(_grid_file(tmp_path, CRS("EPSG:32634").to_wkt())) as grid:
        out_file = climate.carra2(grid, "RGI-test", years=[requested], path=tmp_path / "out")

    assert f"source years used: [{source}]" in capsys.readouterr().out
    assert sum(reprojected_steps) == 12 * 2  # 12 months of air_temp and precipitation, nothing more
    with xr.open_dataset(out_file) as out:
        assert out.sizes["time"] == 12
        assert set(out["time"].dt.year.values) == {requested}
        expected = 100.0 * source + np.arange(1, 13)
        np.testing.assert_allclose(out["air_temp"].mean(dim=("x", "y")).values, expected)
