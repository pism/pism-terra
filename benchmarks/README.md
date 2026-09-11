# Benchmarks

Standalone performance harnesses. These are deliberately **not** pytest
tests — they generate multi-GB inputs and run for minutes, so they never
slow down CI. Run them manually when a relevant code path changes.

## `bench_postprocess_spatial.py`

Compares the write strategies of
`pism_terra.ismip7.greenland.postprocess_spatial` (per-basin spatial
extraction, no summing) and backs the module's `DEFAULT_METHOD` choice.

```bash
python benchmarks/bench_postprocess_spatial.py --target-gb 2 --workers 4
```

What it does:

1. Generates a PISM-like synthetic input (`thk` f32, `ice_mass` f64,
   `mask` i32, 3D `enthalpy` f32 with 32 levels; 24 monthly steps) on an
   EPSG:3413 grid sized from the real Mouginot outlines, at whatever
   resolution hits `--target-gb` uncompressed. Values are random, i.e.
   nearly incompressible — wall times are therefore conservative; real PISM
   fields compress better and write faster.
2. Runs each `--methods` entry in a **fresh subprocess** with a
   processes-based Dask cluster (matching production), sampling the process
   tree's peak RSS and the scratch directory's peak size.
3. Verifies all methods produce value-identical output (per-basin `thk`
   checksums) and writes `results/spatial_write_bench.csv` plus a markdown
   table.

To reuse a previously generated input (e.g. for a quick re-run or to test at
50 GB): `--input path/to/file.nc`. To probe a different scale:
`--target-gb 50` (needs ~2× that in free disk with `--keep`).

### Recorded results

2 GiB (uncompressed) input at 2422 m, 24 monthly steps, 4 workers, Apple
M-series (12 cores, 64 GB), local SSD, 2026-08-13 — all checksums agree:

| method | wall_s | peak_rss_gb | out_gb | peak_scratch_gb |
|---|---|---|---|---|
| netcdf | 17.6 | 4.9 | 0.80 | 0.0 |
| zarr   | 38.0 | 5.4 | 0.80 | 0.19 |
| shards | 37.9 | 8.0 | 0.79 | 0.39 |

`DEFAULT_METHOD = "netcdf"` in `postprocess_spatial.py` reflects this run:
2.2× faster than either alternative, lowest memory, no scratch I/O. Peak RSS
is bounded by in-flight chunks (workers × chunk × pipeline depth), not by
file size, so the ranking should hold at 50 GB — but if you re-run at
production scale on Lustre and a different method wins there, change the
default (one constant) and update this table.

Method notes:

- **netcdf** — all basins' delayed `to_netcdf` writes computed as one graph.
  Separate target files, so no write-lock contention between basins, and
  shared input chunks are read once. No scratch use.
- **zarr** — per-basin scratch Zarr store (lock-free parallel write), then a
  streamed Zarr→NetCDF conversion. Pays one extra output copy of scratch I/O.
- **shards** — serial time-slab writes per basin, then concat + streamed
  final write. Slowest, but memory and file handles are strictly bounded;
  the fallback if the distributed netcdf write misbehaves on a cluster.
