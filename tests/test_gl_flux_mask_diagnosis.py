"""
Tests for :mod:`pism_terra.tools.gl_flux_mask_diagnosis`.

Two synthetic one-year runs on a 900 m grid differ only in their retreat mask:
one exact on the grid, one at 450 m and offset, as the CalFin mask was.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from pism_terra.tools import gl_flux_mask_diagnosis as diag

NX, NY, DX = 40, 30, 900.0
X0, Y0 = -200_000.0, -2_300_000.0
FRONT = 20  # first ocean column


def grid() -> tuple[np.ndarray, np.ndarray]:
    """
    Cell centres of the run grid.

    Returns
    -------
    tuple of numpy.ndarray
        The x and y centres, both ascending.
    """
    return X0 + DX * np.arange(NX), Y0 + DX * np.arange(NY)


def write_retreat(path: Path, x: np.ndarray, y: np.ndarray, edge: float) -> None:
    """
    Write a monthly 0/1 retreat mask, 1 west of ``edge``.

    Parameters
    ----------
    path : pathlib.Path
        File to write.
    x, y : numpy.ndarray
        Cell centres of the mask's own grid.
    edge : float
        The x coordinate of the front.
    """
    values = (x[None, :] < edge).astype("float32") * np.ones((y.size, 1), "float32")
    time = pd.date_range("1980-01-01", periods=12, freq="MS")
    xr.Dataset(
        {diag.RETREAT_VAR: (("time", "y", "x"), np.repeat(values[None], time.size, axis=0))},
        coords={"time": time, "x": x, "y": y},
    ).to_netcdf(path)


def write_run(path: Path, retreat: Path, gl_per_cell: float) -> None:
    """
    Write a spatial file: grounded ice up to ``FRONT``, the flux booked in the first ocean column.

    Parameters
    ----------
    path : pathlib.Path
        File to write.
    retreat : pathlib.Path
        Retreat file named in the ``command`` attribute.
    gl_per_cell : float
        Grounding-line flux per ocean cell, Gt/yr (negative: into the ocean).
    """
    x, y = grid()
    starts = pd.date_range("1980-01-01", periods=12, freq="MS")
    ends = pd.date_range("1980-02-01", periods=12, freq="MS")
    mask = np.full((NY, NX), diag.ICE_FREE_OCEAN, "int8")
    mask[:, :FRONT] = diag.GROUNDED
    thk = np.where(mask == diag.GROUNDED, 500.0, 0.0)
    gl = np.zeros((NY, NX))
    gl[:, FRONT] = gl_per_cell

    def stack(a):
        """
        Repeat a field for every record.

        Parameters
        ----------
        a : numpy.ndarray
            The field.

        Returns
        -------
        numpy.ndarray
            The field with a leading time axis.
        """
        return np.repeat(a[None], starts.size, axis=0)

    ds = xr.Dataset(
        {
            diag.GL_VAR: (("time", "y", "x"), stack(gl), {"units": "Gt year^-1"}),
            "tendency_of_ice_mass_due_to_discharge": (("time", "y", "x"), stack(gl), {"units": "Gt year^-1"}),
            "thk": (("time", "y", "x"), stack(thk), {"units": "m"}),
            "mask": (("time", "y", "x"), stack(mask)),
            "velsurf_mag": (("time", "y", "x"), stack(np.where(thk > 0, 1000.0, np.nan)), {"units": "m year^-1"}),
            "time_bounds": (("time", "nv"), np.stack([starts.values, ends.values], axis=1)),
        },
        coords={"time": ("time", starts + (ends - starts) / 2, {"bounds": "time_bounds"}), "x": x, "y": y},
        attrs={"command": f"pism -geometry.front_retreat.prescribed.file {retreat} -time.start 1980-01-01"},
    )
    ds.to_netcdf(path)


@pytest.fixture(name="runs")
def fixture_runs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """
    Two runs, a 450 m offset mask and an exact one, and a two-region outline.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    tuple of pathlib.Path
        The 450 m run, the 900 m run and the outline.
    """
    x, y = grid()
    fine_x = X0 - 225.0 + 450.0 * np.arange(2 * NX + 1)
    fine_y = Y0 - 225.0 + 450.0 * np.arange(2 * NY + 1)
    # The exact mask's front is a cell edge of the run grid; the 450 m one
    # falls between two fine cells that straddle a run-grid centre, so
    # interpolation puts 0.5 on that column.
    write_retreat(tmp_path / "mask_450m.nc", fine_x, fine_y, X0 + DX * FRONT)
    write_retreat(tmp_path / "mask_900m.nc", x, y, X0 + DX * (FRONT - 0.5))
    write_run(tmp_path / "run_450m.nc", tmp_path / "mask_450m.nc", -0.5)
    write_run(tmp_path / "run_900m.nc", tmp_path / "mask_900m.nc", -0.3)
    mid = Y0 + DX * NY / 2
    gpd.GeoDataFrame(
        {"SUBREGION1": ["SW", "CW"]},
        geometry=[box(X0 - DX, Y0 - DX, X0 + DX * FRONT, mid), box(X0 - DX, mid, X0 + DX * FRONT, Y0 + DX * NY)],
        crs="EPSG:3413",
    ).to_file(tmp_path / "outline.gpkg")
    return tmp_path / "run_450m.nc", tmp_path / "run_900m.nc", tmp_path / "outline.gpkg"


def test_retreat_fraction_of_the_offset_mask_is_partial_at_the_front(runs):
    """
    The 450 m mask interpolated to the 900 m grid puts a fraction on the front column, the exact one does not.

    Parameters
    ----------
    runs : tuple of pathlib.Path
        The fixture's runs and outline.
    """
    x, y = grid()
    start, end = pd.Timestamp("1980-01-01"), pd.Timestamp("1981-01-01")
    coarse, n = diag.retreat_fraction(str(runs[0].parent / "mask_450m.nc"), x, y, start, end)
    exact, _ = diag.retreat_fraction(str(runs[0].parent / "mask_900m.nc"), x, y, start, end)
    assert n == 12
    assert set(np.unique(exact)) == {0.0, 1.0}
    assert (diag.classify(coarse) == 1).any()
    assert (diag.classify(exact) == 1).sum() == 0


def test_main_writes_the_tables_and_splits_the_difference(runs, tmp_path):
    """
    The driver reads the retreat files from 'command', totals the flux per region and ranks the tiles.

    Parameters
    ----------
    runs : tuple of pathlib.Path
        The fixture's runs and outline.
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    run_a, run_b, outline = runs
    out = tmp_path / "diag"
    assert diag.main([str(run_a), str(run_b), "--labels", "450m,900m", "--outline", str(outline),
                      "--output-dir", str(out), "--tile", "5", "--plot-top", "2"]) == 0  # fmt: skip
    for name in ("summary.txt", "regions.csv", "tiles.csv", "timeseries.csv", "fields.nc", "timeseries.png"):
        assert (out / name).is_file()
    assert (out / "tile_01.png").is_file()
    regions = pd.read_csv(out / "regions.csv", index_col=0)
    np.testing.assert_allclose(regions.loc["GIS", "gl_flux_450m"], -0.5 * NY)
    np.testing.assert_allclose(regions.loc["GIS", "gl_flux_900m"], -0.3 * NY)
    np.testing.assert_allclose(regions.loc[["SW", "CW"], "gl_flux_900m"].sum(), -0.3 * NY)
    assert regions.loc["GIS", "gl_cells_450m"] == NY
    np.testing.assert_allclose(regions.loc["GIS", "gl_proxy_flux_900m"], -NY * 910 * 500 * 1000 * DX / 1e12)
    tiles = pd.read_csv(out / "tiles.csv")
    np.testing.assert_allclose(tiles["gl_flux_900m-450m"].abs().max(), 0.2 * 5)
    assert "partial" in (out / "summary.txt").read_text()
