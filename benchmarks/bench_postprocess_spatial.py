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
Benchmark the write strategies of ``postprocess_spatial``.

Generates a PISM-like synthetic input of ``--target-gb`` (uncompressed,
random data — the worst case for zlib, so wall times are conservative), then
runs each ``--methods`` entry of
:func:`pism_terra.ismip7.greenland.postprocess_spatial.process_file_spatial`
in a fresh subprocess with a processes-based Dask cluster, exactly as
production runs. The parent samples the subprocess tree's RSS and the scratch
directory size, checks that all methods produce value-identical output
(per-basin ``thk`` checksums), and writes a CSV plus a markdown table.

Run from the repo root::

    python benchmarks/bench_postprocess_spatial.py --target-gb 2 --workers 4

The winner sets ``DEFAULT_METHOD`` in ``postprocess_spatial.py``; see
``benchmarks/README.md`` for recorded results.
"""

import argparse
import csv
import subprocess
import sys
import tempfile
import time
from importlib.resources import files
from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
import psutil
import rioxarray  # noqa: F401  pylint: disable=unused-import
import xarray as xr

N_TIME = 24
Z_LEVELS = 32
# Bytes per (y, x) cell per time step: thk f32 + ice_mass f64 + mask i32 + enthalpy f32*z.
BYTES_PER_CELL_STEP = 4 + 8 + 4 + 4 * Z_LEVELS
DEFAULT_OUTLINE = str(files("pism_terra.data").joinpath("mouginot_basins_w_shelves.gpkg"))


def make_input(path: Path, target_gb: float, outline: str) -> float:
    """
    Stream a PISM-like synthetic input of roughly ``target_gb`` to ``path``.

    The grid is derived from the real basin-outline bounds so the basin
    rasterization is realistic; the resolution is chosen so the uncompressed
    in-memory size hits the target. Values are random (deterministic seed via
    dask's default), i.e. essentially incompressible — a worst case that
    makes the timings conservative.

    Parameters
    ----------
    path : pathlib.Path
        Output NetCDF path.
    target_gb : float
        Uncompressed target size in GiB.
    outline : str
        Basin outline file whose bounds size the grid.

    Returns
    -------
    float
        The chosen grid resolution in metres.
    """
    basins = gpd.read_file(outline)
    x_min, y_min, x_max, y_max = basins.total_bounds
    width, height = x_max - x_min, y_max - y_min

    cells = target_gb * 2**30 / (N_TIME * BYTES_PER_CELL_STEP)
    resolution = float(np.sqrt(width * height / cells))
    x = np.arange(x_min, x_max, resolution) + resolution / 2
    y = (np.arange(y_min, y_max, resolution) + resolution / 2)[::-1]
    shape = (N_TIME, y.size, x.size)
    chunks = (1, 512, 512)

    ds = xr.Dataset(
        {
            "thk": (("time", "y", "x"), da.random.random(shape, chunks=chunks).astype("float32") * 3000.0),
            "ice_mass": (("time", "y", "x"), da.random.random(shape, chunks=chunks) * 1e12),
            "mask": (("time", "y", "x"), da.random.randint(0, 4, shape, chunks=chunks, dtype="int32")),
            "enthalpy": (
                ("time", "y", "x", "z"),
                da.random.random((*shape, Z_LEVELS), chunks=(*chunks, Z_LEVELS)).astype("float32") * 1e5,
            ),
        },
        coords={"time": np.arange(N_TIME, dtype="float64"), "y": y, "x": x, "z": np.linspace(0, 4000, Z_LEVELS)},
    )
    ds["pism_config"] = xr.DataArray(np.int8(0), attrs={"note": "synthetic benchmark input"})
    ds = ds.rio.write_crs("EPSG:3413").rio.set_spatial_dims(x_dim="x", y_dim="y")

    comp = {"zlib": True, "complevel": 2, "shuffle": True}
    encoding = {v: comp for v in ("thk", "ice_mass", "mask", "enthalpy")}
    print(f"generating {target_gb} GiB input at {resolution:.0f} m ({y.size} x {x.size} cells) -> {path}")
    ds.to_netcdf(path, engine="h5netcdf", encoding=encoding, unlimited_dims=["time"])
    return resolution


def tree_rss(proc: psutil.Process) -> int:
    """
    Resident set size of a process and all its descendants, in bytes.

    Parameters
    ----------
    proc : psutil.Process
        Root of the tree (the benchmark subprocess; Dask workers are its
        children).

    Returns
    -------
    int
        Total RSS in bytes; 0 if the tree is already gone.
    """
    try:
        procs = [proc, *proc.children(recursive=True)]
        return sum(p.memory_info().rss for p in procs if p.is_running())
    except psutil.NoSuchProcess:
        return 0


def dir_bytes(path: Path) -> int:
    """
    Total size of the files under ``path``, in bytes.

    Parameters
    ----------
    path : pathlib.Path
        Directory to walk; may not exist yet.

    Returns
    -------
    int
        Sum of file sizes, 0 when the directory is missing.
    """
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def run_method(method: str, infile: Path, outdir: Path, outline: str, workers: int) -> dict:
    """
    Run one write method in a fresh subprocess and measure it.

    Parameters
    ----------
    method : str
        Write strategy (``netcdf``/``zarr``/``shards``).
    infile : pathlib.Path
        Benchmark input file.
    outdir : pathlib.Path
        Per-method output directory (also hosts the scratch subdirectory).
    outline : str
        Basin outline file.
    workers : int
        Dask worker count for the subprocess.

    Returns
    -------
    dict
        ``method``, ``wall_s``, ``peak_rss_gb``, ``out_gb``,
        ``peak_scratch_gb``, and ``returncode``.
    """
    scratch = outdir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    code = (
        "from pism_terra.ismip7.greenland.postprocess_spatial import postprocess_glacier_spatial; "
        f"postprocess_glacier_spatial({str(infile)!r}, {str(outdir)!r}, {outline!r}, "
        f"n_workers={workers}, local_directory={str(scratch)!r}, method={method!r})"
    )
    start = time.time()
    with subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL) as proc:
        ps = psutil.Process(proc.pid)
        peak_rss = peak_scratch = 0
        while proc.poll() is None:
            peak_rss = max(peak_rss, tree_rss(ps))
            peak_scratch = max(peak_scratch, dir_bytes(scratch))
            time.sleep(0.5)
    wall = time.time() - start

    out_bytes = sum(f.stat().st_size for f in outdir.glob("spatial_*.nc"))
    return {
        "method": method,
        "wall_s": round(wall, 1),
        "peak_rss_gb": round(peak_rss / 2**30, 2),
        "out_gb": round(out_bytes / 2**30, 2),
        "peak_scratch_gb": round(peak_scratch / 2**30, 2),
        "returncode": proc.returncode,
    }


def checksums(outdir: Path) -> dict[str, float]:
    """
    Per-basin ``thk`` nansum fingerprints of one method's output.

    Parameters
    ----------
    outdir : pathlib.Path
        Directory holding the per-basin files.

    Returns
    -------
    dict
        Basin file name to checksum.
    """
    sums = {}
    for f in sorted(outdir.glob("spatial_*.nc")):
        with xr.open_dataset(f, decode_times=False, decode_timedelta=False, chunks="auto") as ds:
            sums[f.name] = float(ds["thk"].sum())
    return sums


def main():
    """
    Run the benchmark.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--target-gb", type=float, default=2.0, help="Uncompressed input size to generate.")
    parser.add_argument("--methods", type=str, default="netcdf,zarr,shards", help="Comma-separated methods.")
    parser.add_argument("--workers", type=int, default=4, help="Dask workers per run.")
    parser.add_argument("--outdir", type=str, default="benchmarks/results", help="Where to put CSV and outputs.")
    parser.add_argument("--input", type=str, default=None, help="Reuse an existing input file instead of generating.")
    parser.add_argument("--outline", type=str, default=DEFAULT_OUTLINE, help="Basin outline file.")
    parser.add_argument("--keep", action="store_true", help="Keep the generated input and per-method outputs.")
    options = parser.parse_args()

    results_dir = Path(options.outdir)
    results_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="bench_spatial_", dir=results_dir if options.keep else None))

    if options.input:
        infile = Path(options.input)
    else:
        infile = work / "bench_input.nc"
        make_input(infile, options.target_gb, options.outline)
    print(f"input: {infile} ({infile.stat().st_size / 2**30:.2f} GiB on disk)")

    rows, sums = [], {}
    for method in options.methods.split(","):
        outdir = work / method
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {method} ===")
        row = run_method(method, infile, outdir, options.outline, options.workers)
        rows.append(row)
        sums[method] = checksums(outdir)
        print(row)

    # Cross-method value check: identical per-basin checksums across the
    # methods that succeeded. A failed method (returncode != 0, or no output)
    # is marked False — it must never look like it agreed.
    succeeded = [row["method"] for row in rows if row["returncode"] == 0 and sums[row["method"]]]
    agree = bool(succeeded) and all(
        np.allclose(sorted(sums[succeeded[0]].values()), sorted(sums[method].values()), rtol=1e-6)
        for method in succeeded
    )
    for row in rows:
        row["checksums_agree"] = agree and row["method"] in succeeded
        if row["method"] not in succeeded:
            print(f"WARNING: method {row['method']} FAILED (returncode {row['returncode']})")

    csv_path = results_dir / "spatial_write_bench.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nresults -> {csv_path}\n")
    keys = list(rows[0].keys())
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for row in rows:
        print("| " + " | ".join(str(row[k]) for k in keys) + " |")

    if not options.keep:
        import shutil  # pylint: disable=import-outside-toplevel

        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
