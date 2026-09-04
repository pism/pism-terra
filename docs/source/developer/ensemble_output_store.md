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
