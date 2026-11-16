# `xr.open_mfdataset` layout benchmark

Synthetic PISM-like ensembles in several NetCDF-4 layouts, plus scripts that
time `xr.open_mfdataset` over them and count the POSIX reads one open needs.
Results and conclusions are summarised in
`docs/source/developer/ensemble_output_store.md` ("Measured: where the
per-file open cost goes").

```bash
# 1. generate 7 layouts x 200 members (~6.5 GB, ~15 min); see LAYOUTS in gen_ensemble.py
python gen_ensemble.py data --members 200 --nt 2 --ny 512 --nx 512

# 2. optional: paged-aggregation variants (needs h5repack on PATH)
mkdir -p data/H_pism_today_paged1M
for f in data/A_pism_today/*.nc; do h5repack -S PAGE -G 1048576 "$f" data/H_pism_today_paged1M/$(basename "$f"); done

# 3. reads/seeks per open, per layout (h5netcdf engine through a counting file object)
python count_io.py data

# 4. one configuration in a fresh process (prints a JSON line)
python bench_one.py data/A_pism_today --engine netcdf4 --parallel distributed --preprocess full

# 5. a matrix of configurations -> results.jsonl
python run_matrix.py results.jsonl matrix2.json

# 6. emulate a latency-bound file system: add 1 ms per read()
python latency_model.py 0.001
```

`gen_ensemble.py` copies the real ~2330 `pism_config` attributes from
`test_uq_saturday/spatial/...uq_0...nc`; point `REAL_CFG` at any PISM output
file if that one is gone. `results.jsonl` / `results2.jsonl` are the runs
behind the tables in the design note (M-series laptop, warm page cache,
xarray 2024.11, netCDF-C 4.9.3, HDF5 1.12.2).
