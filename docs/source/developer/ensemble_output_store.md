# Single-store ensemble output (design note)

Feasibility assessment and design for writing every ensemble member into one
Zarr store — `output/ensemble/ensemble_<tag>.zarr` — instead of one NetCDF
file per member, including whether icechunk or a coordinating `pism-server`
is needed. Status: **design only; nothing below is implemented.**

## Problem

Ensembles run as one Slurm job per member (`pism-glacier-run-forward
--samples N ...`, then `sbatch` each `submit_*_id_<row>_uq_<n>_*.sh`). Each
member writes its own NetCDF files through PISM's asynchronous writer
(`pism_async_writer`, a 1-rank Python MPI process coupled to PISM via YAC,
launched MPMD from `pism_terra/templates/chinook-apptainer-async.j2`).

Analysis then reopens hundreds of files. Measured on a 500-member synthetic
ensemble (one 234×292 glacier, `usurf` only):

| stage | wall time | peak RSS |
|---|---|---|
| `xr.open_mfdataset` over 500 files | **258 s** | 0.27 GB |
| `importance_sampling` over the whole ensemble | 7 s | 0.48 GB |

`combine="by_coords"` versus `combine="nested"` makes no difference: the
cost is the per-file open, so it scales with the file count and only a
single-store layout removes it. Ensembles have to run both on Lustre-type
HPC (chinook, `/import/c1/...`) and on AWS compute writing directly to S3.

## Measured: where the per-file open cost goes

Benchmark in `benchmarks/open_mfdataset/` (2026-09-04): synthetic ensembles
of 200 members, 16 two-dimensional `f8` variables on a 512×512 grid, zlib-2,
about 4.5 MB per file, opened on an M-series laptop with a warm page cache
(xarray 2024.11, netCDF-C 4.9.3, HDF5 1.12.2). Layouts:

| layout | member dim in file | `time` | chunks | `pism_config` |
|---|---|---|---|---|
| A `pism_today` | none | unlimited | `(1, ny, nx)` | `char(cfg)` + ~2330 attributes |
| B `postproc_default` | none | unlimited | netCDF-C default `(1, 30, 73)`-like | as A |
| C `exp_id_char` | `-exp_id`: `char exp_id(exp_id, nc)` | unlimited | as A | as A |
| D `uq_id_int` | `int32 uq_id(uq_id)` | unlimited | as A | as A |
| E `uq_id_fixed_time` | as D | fixed | as A | as A |
| F `uq_id_json_cfg` | as D | fixed | as A | scalar string variable, no attributes |
| G `uq_id_no_cfg` | as D | fixed | as A | absent |
| H, I | A and F after `h5repack -S PAGE -G 1048576` | | | |

**Reads per open** (h5netcdf engine through a counting file object, one file):

| layout | `read()` calls | bytes read |
|---|---|---|
| A, B | 215 | 326 KB |
| C, D, E | 232–235 | 330 KB |
| F | 109 | 69 KB |
| G | 103 | 41 KB |
| H or I with `page_buf_size=4 MiB` | 8 | 3 MB |

Of a file's 394 KB of HDF5 metadata, ~320 KB is the dense attribute storage
of `pism_config`. Per file, `netCDF4.Dataset()` takes 16 ms and reading all
attributes another 13 ms, of which `pism_config` is essentially all; with
h5py the file open is 0.1 ms and the `pism_config` attributes 31 ms.

**`open_mfdataset` over 200 files** (seconds, netcdf4 engine unless noted;
`preprocess` = `preprocess_netcdf`):

| call | A | D | F |
|---|---|---|---|
| preprocess, serial | 12.6 | 13.2 | 4.2 |
| no preprocess, serial | — | 23.8 | 5.4 |
| preprocess, `parallel=True` (default threaded scheduler) | 17.8 | | |
| preprocess, `parallel=True`, dask `processes` scheduler, 8 workers | 5.7 | | |
| preprocess, `parallel=True`, `distributed.LocalCluster` 8 procs | 2.9 | | |
| no preprocess, `combine="nested"` + `data_vars/coords="minimal"`, `compat/join="override"` | | | 4.1 (3.6 with `decode_cf=False`) |
| the same with the 8-process `LocalCluster` | | | 2.6 |
| preprocess, serial, h5netcdf engine | 28.8 | | 12.3 (no preprocess) |

B equals A, E equals D, G equals F and H equals A within noise. Loading
`usurf` for all 200 members afterwards takes 2–4 s regardless of layout.

**Latency model.** Re-running the single-file open with a 1 ms sleep per
`read()` gives A 517 ms/file, F 186 ms/file, I with a 4 MiB page buffer
53 ms/file. The A figure reproduces the 258 s / 500 files measured on chinook
(516 ms/file): on Lustre the open is bound by the number of small metadata
reads, not by CPU.

### Conclusions

1. **The ~2330 attributes on `pism_config` are the dominant cost**: half the
   reads, 80 % of the bytes, half the CPU per file, a 3× difference in
   `open_mfdataset` (D → F) and 2.8× in the latency model. PISM already
   writes the same content as a JSON string in the variable's data
   (`Config::json()`), so the attributes written by `config_metadata()` in
   `~/pism/src/util/Config.cc` are redundant. Proposed PISM change: a flag
   (e.g. `output.pism_config.attributes = false`) that skips them. On the
   pism-terra side `preprocess_netcdf` now decodes the JSON blob through
   `decode_pism_config` (`pism_terra/processing.py`) and only falls back to
   the attributes when the data holds no JSON; the JSON's `[value, units]`
   pairs and booleans are mapped onto the attribute convention, so the
   `pism_config` coordinate is unchanged for consumers. Still reading the
   attributes directly: `pism_config_value` (`pism_terra/workflow.py`) and
   `kitp/analyze.py` (`pism_config.attrs["grid.dx"]`).
2. **Recording the member id in PISM does not make the open faster** (D ≈ A);
   it only removes the regex + `expand_dims` part of `preprocess_netcdf`
   (~1 s per 200 files). Today's `-exp_id` is the wrong shape for it: xarray
   decodes `char exp_id(exp_id, nc)` to an `|S10` coordinate that sorts
   lexicographically (`'0', '1', '10', '11', '2', …`), the label is capped
   at 10 characters, and `preprocess_netcdf` raises
   `Dimension exp_id already exists` on such files. If PISM records the
   member, it should be an integer coordinate (`int uq_id(uq_id)`, name
   configurable) on every non-time variable, as the leading dimension — which
   is exactly the `(uq_id, time, y, x)` layout of the Zarr schema above.
3. **Chunking and the unlimited `time` dimension do not matter** for open
   (A = B, D = E) or for a full-slice read.
4. **HDF5 paged aggregation only pays off when read with a page buffer**
   (h5netcdf engine, `driver_kwds={"page_buf_size": 4 << 20}`): reads drop
   from 215 to 8 at the price of reading 3 MB instead of 0.3 MB. netCDF-C
   has no API for either the file-space strategy or the page buffer, so the
   netcdf4 engine cannot benefit and PISM cannot write such files; it needs a
   post-run `h5repack` (7 s per file with the attributes, 1 s without).
   Worth it only on Lustre or S3, and only together with item 1.
5. **Client side**, independent of the file layout: `parallel=True` with the
   default threaded scheduler is *slower* than serial (the netCDF4/HDF5
   global lock serialises the opens); a process-based scheduler
   (`LocalCluster(processes=True)`) gives 4×. The nested/minimal/override
   combine saves another 25 %. h5netcdf is 2–3× slower than netcdf4 on a
   local disk (per-attribute Python calls) and is only worth using for paged
   files on a latency-bound file system.

With items 1 and 5 the 500-file open on chinook is estimated at roughly
186 ms × 500 / 8 ≈ 12–40 s instead of 258 s; the single store below still
removes the per-file cost entirely and remains the target, but the NetCDF
path — and the Phase 0 consolidation that has to read the files once — gets
much cheaper on the way.

## Verdict

Possible and worthwhile — but not with icechunk on Lustre, and without a
server.

1. **The writer seam is clean.** PISM's client side
   (`~/pism/src/util/io/YacOutputWriter.{hh,cc}`, ~720 lines) is
   format-agnostic: one JSON `{"action": <int>, "info": {...}}` message per
   action over an MPI intercommunicator, bulk data via `MPI_Isend`
   (non-gridded) or `yac_cput` (gridded). Everything NetCDF-specific on the
   server is `class OutputFile` (`~/pism/util/pism_async_writer:338-448`,
   ~110 lines) plus `get_file()` in `main()`. `pism_ismip7_writer` and
   `pism_kitp_writer` are existing forks that speak the same protocol.
2. **icechunk is the wrong tool on Lustre.** Its local-filesystem storage is
   documented as *"not safe in the presence of concurrent commits … one of
   the commits can be lost"* ([storage guide][ic-storage];
   [earth-mover/icechunk #804][ic-804], open since March 2025). Commit
   needs a compare-and-swap on the branch ref, which object stores provide
   (conditional PUT) and POSIX/NFS/Lustre do not. Five hundred independent
   committers on `/import/c1` would silently drop members — the retry loop
   never fires because both commits "succeed".
3. **On S3 icechunk works, but every member must commit with a rebase-retry
   loop** ([parallel writes][ic-parallel]); rebase is O(intervening
   commits) ([#1871][ic-1871]); and any resize or metadata edit is an
   unresolvable `ZarrMetadataDoubleUpdate` ([conflicts][ic-conflicts],
   [version control][ic-vc]). Arrays therefore have to be pre-allocated
   anyway — which removes most of the reason to want transactions *during*
   the run.
4. **Plain Zarr v3 with pre-allocation and disjoint per-member chunks needs
   no server on either platform.** If (a) every array is created once
   before any job starts, (b) each member only writes chunks whose leading
   `uq_id` index is its own, and (c) shared keys (`zarr.json`, consolidated
   metadata, 1-D `time`, `x`, `y`) are written by exactly one process, the
   store itself is the coordinator: distinct files on Lustre, distinct
   objects on S3 ([`Dataset.to_zarr` region writes][xr-to-zarr]).
   Coordination collapses to two one-shot steps, **init** before submission
   and **finalize** after the last job. A `pism-server` would only earn its
   keep for dynamic membership or for brokering icechunk commits; neither
   applies.
5. **Store-agnostic by construction.** zarr-python 3's `Store` API covers
   `file://` (`LocalStore`, wrapped for atomic temp-file + `os.replace` on
   Lustre), `s3://` (`FsspecStore` / `ObjectStore`), and icechunk
   (`session.store` *is* a zarr Store). The writer codes against arrays, not
   backends; `--icechunk` is an optional commit-at-shard-boundary layer for
   S3 only, to be scale-tested before it is relied on.
6. **Scalar time series, final state and checkpoints stay NetCDF.** Only
   `output.spatial.file` and snapshots pass through the async writer
   (`~/pism/src/icemodel/IceModel.cc:141-157`); scalars, `-o` and
   checkpoints are synchronous C++ I/O. Scalars are tiny, so a post-run
   consolidation into `ensemble_<tag>_scalar.zarr` takes seconds.

### Platform matrix

| | Lustre (chinook) | S3 (AWS compute) |
|---|---|---|
| Plain Zarr v3, pre-allocated, disjoint shards | **yes** — atomic rename per shard | **yes** — atomic PUT per object |
| icechunk, one session per member, commit per shard | **no** — lost commits ([#804][ic-804]) | possible; rebase cost unproven at 500 writers ([#1871][ic-1871]) |
| icechunk as single-writer post-run consolidation (virtual refs into the NetCDFs) | yes ([virtual datasets][ic-virtual], `LocalFileSystemAccess`) | yes |
| `pism-server` needed | no | no |

## Store schema

Path: `<out>/<RGI>/output/ensemble/ensemble_g<res>_<RGI>_<start>_<end>.zarr`
— one store per glacier per campaign; the init leg keeps NetCDF. Defined once
in a future `pism_terra/ensemble/schema.py` so that post-run consolidation
and the live writer produce identical layouts.

| array | dims | chunk | shard | written by |
|---|---|---|---|---|
| 2-D data vars | `(uq_id, time, y, x)` | `(1, 1, ny, nx)` | `(1, T_shard, ny, nx)` | member |
| 3-D data vars | `(uq_id, time, z, y, x)` | `(1, 1, nz, ny, nx)` | `(1, T_shard, nz, ny, nx)` | member (PISM delivers `(z, y, x)`; no transpose) |
| `time_member`, `time_bounds_member` | `(uq_id, time[, nv])` | `(1, T_max[, 2])` | — | member |
| `time_length`, `status` (`i4`) | `(uq_id,)` | `(1,)` | — | member |
| `x`, `y`, `z`, `mapping`, `uq_id`, `exp_id(uq_id)`, `sample(uq_id)` | 1-D | full | — | init |
| `time`, `time_bounds` (1-D, CF) | `(time,)` | full | — | finalize |
| `pism_config(uq_id)` JSON, `history(uq_id)`, run stats `(uq_id,)` | 1-D | full | — | finalize, from sidecars |
| consolidated metadata, attrs | — | — | — | finalize |

Per-member sidecar `store/_members/<uq_id:05d>.json` (atomic replace) holds
variable and global attributes, history, the `pism_config` dict, run stats,
`time_length` and `status`. The JSON-string `pism_config(uq_id)` coordinate
is what `preprocess_netcdf` (`pism_terra/processing.py`) produces today and
`kitp/calibrate.py` consumes, so downstream code carries over unchanged.

### Sizes

For a 234×292 grid, 468 monthly steps, 500 members, 16 variables:

- one 2-D slice: 68 328 cells = 0.52 MiB `f8` / 0.26 MiB `f4`;
- whole ensemble: 1.86 TiB `f8` / 0.93 TiB `f4` uncompressed, roughly
  200–400 GB with zstd (glacier fields are mostly fill);
- unsharded `(1, 1, ny, nx)` chunks: **3.74 M files/objects** — unacceptable
  on Lustre and 3.7 M PUTs on S3;
- sharded with `T_shard = 24`: **160 k** files/objects of ~12.5 MiB, writer
  buffer 200 MB per member (default; a CLI knob). `T_shard = 12` → 312 k
  files, 100 MB; `T_shard = 48` → 80 k files, 400 MB.

Store `f4` by default (`--dtype f8` to override); PISM sends `f8` and the
writer casts. `T_max` is PISM's schedule count (`Time::compute_times_monthly`,
inclusive ends, first record at `t == start` skipped) rounded up by one shard;
the actual count lands in `time_length`. Zarr only materialises chunks that
are written (`write_empty_chunks=False`), so the slack costs nothing.

## Phased path

### Phase 0 — post-run consolidation (pism-terra only, no PISM change)

`pism-ensemble-consolidate` builds the schema above from the existing
per-member NetCDF files:

1. glob `output/spatial/spatial_*_uq_*.nc` (or `output/dh/`), parse ids with
   the regexes already in `preprocess_netcdf`;
2. pre-allocate with the `to_zarr(compute=False)` idiom of
   `write_zarr_in_bands` (`pism_terra/glacier/climate.py`);
3. per member, in a process pool: open, `preprocess_netcdf`,
   `_zarr_safe_chunks` (`pism_terra/postprocess_spatial.py`),
   `to_zarr(mode="r+", region={"uq_id": slice(i, i + 1)}, consolidated=False)`,
   write the sidecar;
4. `pism-ensemble-finalize`: verify all `time_member` rows agree, collapse to
   1-D `time`/`time_bounds`, merge sidecars, `zarr.consolidate_metadata`
   **once**.

A `--scalar` variant does the same for `output/scalar/*.nc`. This validates
the schema before any writer work and removes the reopen cost immediately.
Optional 0b: `pism-ensemble-write-member`, appended to each job's
post-processing block, writes the member into a pre-initialised store as
the job ends — the live single store with zero PISM changes, at the cost of
double disk during the transition.

### Phase 1 — `pism_zarr_writer` (PISM `util/`) plus init/finalize

- Fork `pism_async_writer`; keep `YacWrapper` and `ServerActions` verbatim;
  replace `OutputFile` with `ZarrMemberWriter` (same duck-typed methods plus
  `open_for_append() -> (time_length, last_time)` for `OPEN_FILE`). Files
  that do not match the spatial pattern (init leg, snapshots) fall through
  to the existing NetCDF backend.
- CLI: `--store file://…|s3://… --member N [--t-shard 24] [--dtype f4]
  [--flush-interval-min 30] [--icechunk]`. The member index comes from
  `--member` (the row position pism-terra already knows in
  `_build_ensemble_df`), not from `-exp_id`: that is a 10-character label
  which also prepends a length-1 `exp_id` axis to every variable — an axis
  the current server does not index (`write_gridded_variable` builds its
  hyperslab from `ndims` and `time_dependent` only). Fix that one-liner in
  `pism_async_writer` regardless.
- Backends: `AtomicLocalStore(LocalStore)` (temp file + `os.replace`, ~40
  lines) for Lustre; `FsspecStore`/`ObjectStore` for S3; icechunk
  `session.store` with `commit(rebase_with=ConflictDetector())` in a retry
  loop at shard boundaries, S3 only, behind a scale test.
- Semantics: buffer `T_shard` steps per variable and write whole shards (no
  read-modify-write); partial flush at `close`, on `SIGTERM`
  (`#SBATCH --signal=B:TERM@120`) and every `--flush-interval-min`;
  `_FillValue` is PISM's `-2e9`, set as the array `fill_value` at init;
  `x`/`y`/`z` are checked with `allclose`, never written; text variables
  (`pism_config` blob, `exp_id`) go to the sidecar; restart reads
  `time_length[member]`.
- pism-terra: `pism_terra/ensemble/{schema,variables,init,consolidate,finalize}.py`
  and console scripts `pism-ensemble-{init,consolidate,write-member,finalize}`;
  `_build_ensemble_df` gains a `member` column; `_render_forward_run` gains
  Jinja keys `writer`, `ensemble_store`, `member`, `t_shard`; the MPMD line
  in `chinook-apptainer-async.j2` (and `ec2.j2` for AWS) becomes
  `{{ writer }} --store … --member … : -n ${pismtasks} … pism …`; the
  template's `pismtasks=$(( SLURM_NTASKS - 4 ))` becomes `- 1` (only one
  writer rank is launched); finalize runs as a `--dependency=afterany` job.
  `zarr>=3` (plus `s3fs`/`obstore`, optionally `icechunk`) is added to the
  PISM container image; `pyproject.toml` pins `zarr>=3`.
- Tests: driver-free unit tests of `ZarrMemberWriter` against a recorded
  action stream (lazy `yac`/`mpi4py` imports); a 2-member MPMD run modelled
  on `~/pism/test/regression/async_io/run_test.sh`, compared with
  `pism_async_writer` output; a 32-process concurrency test with random
  `SIGTERM` (no shared key modified, no `.tmp` leftovers); the same on
  `/import/c1` and on an S3 bucket; an icechunk scale test with 500 fake
  committers before `--icechunk` is enabled.

### Phase 2 — consumers open the store

`kitp/analyze.py:load_dataset`, `kitp/calibrate.py`, `postprocess_dh.py`
and `postprocess_scalar.py` accept a `.zarr` path through
`schema.open_ensemble`; the NetCDF path stays. `importance_sampling`
(`pism_terra/filtering.py`) needs nothing: it already works on
`(uq_id, time, y, x)` arrays.

### Phase 3 (optional, C++) — scalars through the async writer

Deferred; Phase 0's `--scalar` consolidation is seconds and adequate. State
and checkpoint files must remain NetCDF because they are PISM restart inputs.

## Risks and open questions

- The variable list and ranks must be known at init (`output.spatial.vars`
  plus a rank table, cross-checked with `pism -list_diagnostics json`). An
  unknown variable should either fail fast or fall back to per-member
  NetCDF — to be decided.
- The exact record-count rule for custom `-output.spatial.times` lists and
  for restarts from the init leg.
- Crash gap: steps buffered since the last flush are lost; keep
  `--flush-interval-min` below the checkpoint interval.
- icechunk with 500 live committers is unbenchmarked ([#1871][ic-1871]);
  treat it as opt-in on S3 only.
- S3 egress when analysing from chinook (~$0.09/GB → $30–40 per full
  400 GB read); AWS-side analysis avoids it.
- `(z, y, x)` layout for 3-D variables differs from PISM NetCDF `(y, x, z)`;
  `f4` storage loses precision relative to on-wire `f8` (fine for
  diagnostics; state stays NetCDF).

## Sources

- icechunk: [storage guide][ic-storage] · [parallel writes][ic-parallel] ·
  [version control][ic-vc] · [conflicts API][ic-conflicts] ·
  [performance / manifest splitting][ic-perf] · [virtual datasets][ic-virtual]
  · [issue #804][ic-804] · [issue #1871][ic-1871]
- xarray: [`Dataset.to_zarr`][xr-to-zarr]
- PISM: `~/pism/util/pism_async_writer`, `~/pism/src/util/io/YacOutputWriter.cc`,
  `~/pism/src/icemodel/IceModel.cc:141-157`,
  `~/pism/doc/sphinx/manual/practical-usage/input-output.rst` (async I/O section)

[ic-storage]: https://icechunk.io/en/latest/guides/storage/
[ic-parallel]: https://icechunk.io/en/latest/understanding/parallel/
[ic-vc]: https://icechunk.io/en/latest/understanding/version-control/
[ic-conflicts]: https://icechunk.io/en/stable/reference/conflicts/
[ic-perf]: https://icechunk.io/en/latest/guides/performance/
[ic-virtual]: https://icechunk.io/en/latest/guides/virtual/
[ic-804]: https://github.com/earth-mover/icechunk/issues/804
[ic-1871]: https://github.com/earth-mover/icechunk/issues/1871
[xr-to-zarr]: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_zarr.html
