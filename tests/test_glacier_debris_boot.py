"""
Tests for putting the staged debris thickness into the glacier boot file.

PISM's debris transport model regrids ``debris_thickness`` from the file it
bootstraps from, so the boot file has to carry it as a plain 2-D field in m.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pism_terra.domain import create_domain
from pism_terra.glacier.debris import _add_static_time, add_debris_to_boot
from pism_terra.workflow import stamp_grid_mapping


def domain() -> xr.Dataset:
    """
    Build a small UTM domain like the staging's.

    Returns
    -------
    xarray.Dataset
        The grid dataset.
    """
    return create_domain([400_000.0, 410_000.0], [6_700_000.0, 6_706_000.0], 1_000, crs="EPSG:32606")


def debris_on(grid: xr.Dataset) -> xr.Dataset:
    """
    Build a debris dataset shaped like :func:`debris_from_grid`'s output.

    Parameters
    ----------
    grid : xarray.Dataset
        Grid the fields sit on.

    Returns
    -------
    xarray.Dataset
        ``debris_thickness`` ramping from 0 to 1 m and a unit melt factor, with the static time axis.
    """
    shape = (grid.sizes["y"], grid.sizes["x"])
    thickness = np.linspace(0.0, 1.0, shape[0] * shape[1]).reshape(shape)
    ds = xr.Dataset(
        {
            "debris_thickness": (("y", "x"), thickness, {"units": "m", "long_name": "supraglacial debris thickness"}),
            "debris_melt_factor": (("y", "x"), np.ones(shape), {"units": "1"}),
        },
        coords={"x": grid["x"], "y": grid["y"]},
    )
    return _add_static_time(ds)


def test_add_debris_to_boot_writes_a_2d_field(tmp_path: Path):
    """
    Thickness and melt factor land in the boot file without a time axis, with units and the grid mapping.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    grid = domain()
    boot = xr.Dataset(
        {"thickness": (("y", "x"), np.full((grid.sizes["y"], grid.sizes["x"]), 100.0), {"units": "m"})},
        coords={"x": grid["x"], "y": grid["y"]},
    ).rio.write_crs("EPSG:32606")
    debris = debris_on(grid)

    out = add_debris_to_boot(boot, debris)
    for name in ("debris_thickness", "debris_melt_factor"):
        assert out[name].dims == ("y", "x")
        np.testing.assert_allclose(out[name], debris[name].isel(time=0))

    path = tmp_path / "bootfile.nc"
    stamp_grid_mapping(out).to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path) as ds:
        assert "time" not in ds.dims
        for name, units in (("debris_thickness", "m"), ("debris_melt_factor", "1")):
            assert ds[name].dims == ("y", "x")
            assert ds[name].attrs["units"] == units
            assert ds[name].attrs.get("grid_mapping") == ds["thickness"].attrs.get("grid_mapping")
            assert not ds[name].isnull().any()


def test_add_debris_to_boot_refuses_another_grid():
    """
    A debris field on a different grid is an error rather than a silent misplacement.
    """
    grid = domain()
    boot = xr.Dataset(coords={"x": grid["x"], "y": grid["y"]})
    shifted = debris_on(grid).assign_coords(x=grid["x"] + 500.0)
    with pytest.raises(ValueError, match="not on the boot file's x grid"):
        add_debris_to_boot(boot, shifted)


def test_debris_file_carries_a_zero_input_rate(tmp_path: Path, monkeypatch):
    """
    The staged debris file holds ``debris_input_rate`` = 0 on its static time axis; the boot file does not.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture replacing the granule download.
    """
    from shapely.geometry import box  # pylint: disable=import-outside-toplevel

    from pism_terra.glacier import debris as debris_module  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(debris_module, "download_debris_tifs", lambda *args, **kwargs: {})
    grid = domain()
    outline = [box(402_000.0, 6_701_000.0, 408_000.0, 6_705_000.0)]
    path = tmp_path / "debris_RGI.nc"
    debris_module.debris_from_grid(grid, outline, rgi_id="RGI", path=path, staging_path=tmp_path)

    with xr.open_dataset(path) as ds:
        rate = ds["debris_input_rate"]
        assert rate.dims == ("time", "y", "x")
        assert rate.attrs["units"] == "m year^-1"
        assert float(np.abs(rate).max()) == 0.0
        assert "time_bnds" in ds
        boot = add_debris_to_boot(xr.Dataset(coords={"x": grid["x"], "y": grid["y"]}), ds.load())
    assert "debris_input_rate" not in boot
    assert "debris_thickness" in boot
