"""Generate synthetic PISM-like ensemble output in several file layouts.

Layout knobs
------------
member_dim : None | "char" | "int"
    None   -> (time, y, x)                              current PISM default
    "char" -> (exp_id, time, y, x) + char exp_id(exp_id, nc)   PISM -exp_id today
    "int"  -> (uq_id, time, y, x) + int32 uq_id(uq_id)  proposed
time_unlimited : bool
chunking : "record" (1, ny, nx) | "netcdf_default" (let netCDF-C choose) | "contiguous"
config : "attrs" (|S1 var with ~2330 attrs, as PISM does) | "json" (scalar string var, 0 attrs) | "none"
dtype : "f8" | "f4"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import netCDF4
import numpy as np

REAL_CFG = "/Users/andy/base/pism-terra/test_uq_saturday/spatial/spatial_g500m_RGI2000-v7.0-C-01-04374_id_0_uq_0_0001-01-01_0004-01-01.nc"

VARS = [
    "climatic_mass_balance",
    "effective_air_temp",
    "effective_precipitation",
    "ice_mass",
    "ice_surface_temp",
    "tendency_of_ice_mass",
    "tendency_of_ice_mass_due_to_basal_mass_flux",
    "tendency_of_ice_mass_due_to_conservation_error",
    "tendency_of_ice_mass_due_to_discharge",
    "tendency_of_ice_mass_due_to_flow",
    "tendency_of_ice_mass_due_to_surface_mass_flux",
    "thk",
    "usurf",
    "velbase_mag",
    "velsurf_mag",
    "surface_melt_flux",
]
FILL = -2e9


def load_real_config():
    nc = netCDF4.Dataset(REAL_CFG)
    v = nc.variables["pism_config"]
    attrs = {a: v.getncattr(a) for a in v.ncattrs()}
    blob = v[:].tobytes().rstrip(b"\x00").decode()
    nc.close()
    return attrs, blob


def glacier_field(ny, nx, rng, member):
    """Mostly fill value with a glacier-shaped blob; cheap to compress like real output."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = ny * 0.5, nx * 0.5
    r = np.sqrt(((yy - cy) / (ny * 0.15)) ** 2 + ((xx - cx) / (nx * 0.25)) ** 2)
    field = np.full((ny, nx), FILL)
    m = r < 1
    field[m] = np.round(100 * (1 - r[m]) * (1 + 0.1 * member) + rng.normal(0, 1, m.sum()), 1)
    return field


def write_member(
    path,
    member,
    *,
    ny,
    nx,
    nt,
    nvars,
    member_dim,
    time_unlimited,
    chunking,
    config,
    dtype,
    cfg_attrs,
    cfg_blob,
    complevel=2,
    exp_id_max_length=10,
):
    rng = np.random.default_rng(member)
    nc = netCDF4.Dataset(path, "w", format="NETCDF4")
    nc.createDimension("time", None if time_unlimited else nt)
    nc.createDimension("nv", 2)
    nc.createDimension("y", ny)
    nc.createDimension("x", nx)

    lead: tuple[str, ...] = ()
    if member_dim == "char":
        nc.createDimension("exp_id", 1)
        nc.createDimension("nc", exp_id_max_length)
        v = nc.createVariable("exp_id", "S1", ("exp_id", "nc"))
        v.long_name = "experiment ID"
        v[0] = netCDF4.stringtochar(np.array([str(member).ljust(exp_id_max_length - 1)], "S%d" % exp_id_max_length))[0]
        lead = ("exp_id",)
    elif member_dim == "int":
        nc.createDimension("uq_id", 1)
        v = nc.createVariable("uq_id", "i4", ("uq_id",))
        v.long_name = "ensemble member index"
        v[:] = member
        lead = ("uq_id",)

    t = nc.createVariable("time", "f8", ("time",))
    t.units = "seconds since 2000-01-01"
    t.calendar = "standard"
    t.axis = "T"
    t.long_name = "time"
    t.bounds = "time_bounds"
    tb = nc.createVariable("time_bounds", "f8", ("time", "nv"))
    day = 86400.0
    edges = np.arange(nt + 1) * 30.4375 * day
    t[:] = 0.5 * (edges[:-1] + edges[1:])
    tb[:, 0] = edges[:-1]
    tb[:, 1] = edges[1:]
    for name, n in (("y", ny), ("x", nx)):
        c = nc.createVariable(name, "f8", (name,))
        c.units = "m"
        c.axis = name.upper()
        c.standard_name = f"projection_{name}_coordinate"
        c.long_name = f"{name}-coordinate in projected coordinate system"
        c[:] = np.arange(n) * 500.0
    sr = nc.createVariable("spatial_ref", "i4", ())
    sr.grid_mapping_name = "transverse_mercator"
    sr.crs_wkt = 'PROJCS["dummy",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]]]]'
    for k in (
        "false_easting",
        "false_northing",
        "latitude_of_projection_origin",
        "longitude_of_central_meridian",
        "scale_factor_at_central_meridian",
        "semi_major_axis",
        "inverse_flattening",
    ):
        sr.setncattr(k, 1.0)

    if config == "attrs":
        nc.createDimension("cfg", 32768)
        v = nc.createVariable("pism_config", "S1", ("cfg",))
        v.setncatts(cfg_attrs)
        b = np.frombuffer(cfg_blob.encode().ljust(32768, b"\x00"), dtype="S1")
        v[:] = b
    elif config == "json":
        v = nc.createVariable("pism_config", str, ())
        v.long_name = "PISM configuration as JSON"
        v[()] = cfg_blob

    dims = lead + ("time", "y", "x")
    if chunking == "record":
        ck = {"chunksizes": (1,) * len(lead) + (1, ny, nx), "contiguous": False}
    elif chunking == "netcdf_default":
        ck = {"contiguous": False}
    elif chunking == "contiguous":
        ck = {"contiguous": True}
    else:
        raise ValueError(chunking)
    base = glacier_field(ny, nx, rng, member).astype(dtype)
    for i, name in enumerate(VARS[:nvars]):
        var = nc.createVariable(
            name, dtype, dims, zlib=complevel > 0, complevel=complevel, shuffle=True, fill_value=FILL, **ck
        )
        var.units = "m"
        var.long_name = name.replace("_", " ")
        var.standard_name = name
        var.grid_mapping = "spatial_ref"
        data = np.empty((nt, ny, nx), dtype)
        for it in range(nt):
            data[it] = base
            data[it][base != FILL] += i + 0.01 * it
        var[...] = data[None] if lead else data
    nc.history = "synthetic"
    nc.Conventions = "CF-1.6"
    nc.command = f"pism -exp_id {member} synthetic"
    nc.source = "synthetic PISM"
    nc.title = "layout benchmark"
    nc.close()


LAYOUTS = {
    # name: (member_dim, time_unlimited, chunking, config)
    "A_pism_today": (None, True, "record", "attrs"),
    "B_postproc_default": (None, True, "netcdf_default", "attrs"),
    "C_exp_id_char": ("char", True, "record", "attrs"),
    "D_uq_id_int": ("int", True, "record", "attrs"),
    "E_uq_id_fixed_time": ("int", False, "record", "attrs"),
    "F_uq_id_json_cfg": ("int", False, "record", "json"),
    "G_uq_id_no_cfg": ("int", False, "record", "none"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--members", type=int, default=100)
    p.add_argument("--ny", type=int, default=512)
    p.add_argument("--nx", type=int, default=512)
    p.add_argument("--nt", type=int, default=12)
    p.add_argument("--nvars", type=int, default=16)
    p.add_argument("--dtype", default="f8")
    p.add_argument("--layouts", nargs="*", default=list(LAYOUTS))
    a = p.parse_args()
    cfg_attrs, cfg_blob = load_real_config()
    for name in a.layouts:
        member_dim, unl, ck, cfg = LAYOUTS[name]
        d = a.out / name
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        for m in range(a.members):
            f = d / f"spatial_g500m_RGI2000-v7.0-C-01-04374_id_0_uq_{m}_2000-01-01_2001-01-01.nc"
            if f.exists():
                continue
            write_member(
                f,
                m,
                ny=a.ny,
                nx=a.nx,
                nt=a.nt,
                nvars=a.nvars,
                member_dim=member_dim,
                time_unlimited=unl,
                chunking=ck,
                config=cfg,
                dtype=a.dtype,
                cfg_attrs=cfg_attrs,
                cfg_blob=cfg_blob,
            )
        size = sum(f.stat().st_size for f in d.glob("*.nc")) / 1e6
        print(f"{name:22s} {a.members} files, {size:8.1f} MB total, {time.perf_counter()-t0:6.1f} s")


if __name__ == "__main__":
    main()
