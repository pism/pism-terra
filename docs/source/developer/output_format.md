# Output format and wall time on chinook

In September 2026 the members of an S4F UQ ensemble finished between 1 and
4 hours apart although every member did the same solver work: the same
number of monthly steps, the same Newton and Krylov iteration counts per
step, one Blatter solve per month. This page records how the spread was
traced to the output path, what was measured, and what follows for the
configs. The measurements can be repeated with
`python -m pism_terra.tools.chinook_node_io_test` (see the end).

## Symptoms

- `sacct` showed the forward leg keeping **2 to 5 of 24 cores busy**
  (`TotalCPU / Elapsed`) while the PETSc-bound inversion in the same job
  kept 21 to 22. The forward leg was waiting, not computing, and the slow
  node had run the compute-bound inversion *faster* than the fast one.
- Jobs cancelled mid-run froze a snapshot: the scalar file, which rank 0
  writes synchronously every month, tracked the model on every node; the
  spatial file, written by `pism_async_writer`, had not received a single
  record after 32 minutes on 9 of 15 nodes, and those were the nodes where
  the model itself crawled.
- Even RGI2000-v7.0-C-01-14094, whose Blatter problem has a handful of
  elements, took 1 to 3.4 hours for 469 monthly steps: 7 to 26 seconds per
  step of overhead unrelated to the grid.

## The per-node test

`pism_terra.tools.chinook_node_io_test` pins a shortened copy of a
production run script (four model years, same flags, same container) to
named nodes in several *variants* that differ only in how output is
written. Round one, glacier RGI2000-v7.0-C-01-05881, 48 monthly steps on
eight nodes:

| variant | what it measures | 48 steps took |
| --- | --- | --- |
| `noout` | the model alone, no spatial output | 44 to 94 s on every node |
| `sync` | PISM writes the spatial file itself, `netcdf4_parallel` | 270 to 1513 s |
| `async` | production: `pism_async_writer` writes it | 231 to 1930 s |

Writing 48 half-megabyte records cost **4 to 40 s per record**, with both
paths, while `dd` moved 512 MB in 2 to 6 s and a netCDF-4 append benchmark
of the same shape ran in under a second on every node. The same node was
fast in one variant and slow in the other (n83: 231 s async, 850 s sync;
n31 the reverse), so the stall depends on the moment, not the hardware.
Three nodes paid an identical 1341 s in the async variant: 27.9 s per
record to the second, the signature of a timeout rather than of bandwidth.

Round two changed only the file path of PISM's *own* files (state and
scalar, plus the spatial file in the `sync` variants), on four nodes:

| variant | change | seconds per step |
| --- | --- | --- |
| `sync_nolock`, `async_nolock` | `HDF5_USE_FILE_LOCKING=FALSE` | 3.7 to 11.6 |
| `sync_nc4s` | `output.format = netcdf4_serial` | 0.7 to 1.0 |
| `sync_nc3` | `output.format = netcdf3` | 0.6 to 0.9 |
| `async_nc3` | production writer, PISM's own files `netcdf3` | 0.6 to 0.9 |

With PISM's own files in serial NetCDF-4 or classic NetCDF-3 the entire
overhead disappears on every node, and the async writer runs at model
speed. Serial NetCDF-4 is as fast as classic, so HDF5 itself is fine: it is
the **parallel HDF5 path through MPI-IO on Lustre** that stalls, on every
monthly scalar flush (which every spatial record triggers) and on every
spatial record the model writes itself. Disabling file locking helps a
little and does not fix it. `pism_async_writer` was never the bottleneck.

The `noout` runs of round one were themselves paying for one parallel-HDF5
flush at the end of the run, which is why the round-two variants beat them.

## What changed

The 20 `s4f_*.toml` configs write PISM's own files with
`output.format = "netcdf4_serial"`. Glacier files are a few megabytes and
gain nothing from parallel I/O; serial keeps compression and the format
every downstream tool reads. The chinook templates still launch
`pism_async_writer`.

## ISMIP7 at 1200 m is unaffected

The same test on `2026_09_ismip7_core_1200m` (96 ranks over two nodes,
`pism_ismip7_writer`, yearly output, a 2.2 GB end state) gave 1424 to 1906 s
for 224 steps in **every** variant, `noout` included; the spread between
node pairs is as large as the spread between variants. At one record per
year the stall is paid four times in four years and vanishes behind 6 to
8 s of compute per step. The ISMIP7 configs keep `netcdf4_parallel`.

Monthly ISMIP7 output would change that: 48 records per four years, each
about 375 MB. The writer would keep up (it compresses about 50 MB/s, the
model needs about 32 s per model month), but PISM's own monthly scalar
flush would go through the parallel path 48 times per four years, 4 to 40 s
each. Switch PISM's own files to `netcdf4_serial` first, the `async_nc4s`
variant, which was also the fastest ISMIP7 variant on average.

## Repeating the test

```bash
# glacier: one node per job; ISMIP7: two nodes joined with +, or let Slurm choose
python -m pism_terra.tools.chinook_node_io_test generate RUN_SCRIPT \
    --nodes n31,n34 --variants async,sync,noout,async_nc4s \
    --test-dir /import/c1/ICESHEET/ICESHEET/pism-terra/node_io_test --submit
python -m pism_terra.tools.chinook_node_io_test analyze /import/c1/ICESHEET/ICESHEET/pism-terra/node_io_test
```

`generate` reads the writer launch, task counts and partition from the
production script; `analyze` prints seconds per model step by node and
variant and the `sacct` command for the jobs. Nodes with fewer cores than
`--tasks-per-node` are refused by Slurm; omit `--nodes` to let it choose and
the results record the nodes it used.
