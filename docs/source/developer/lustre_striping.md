# Tuning Lustre striping on chinook

PISM output on chinook lands on Lustre, where a file is spread over some
number of object storage targets (OSTs). How many, and in how large a chunk,
is the *stripe layout*, and the default is conservative: **1 OST, 1 MB
stripes** on a filesystem with **15 OSTs**
([UAF RCS striping guide](https://uaf-rcs.gitbook.io/uaf-rcs-storage-docs/lustre_striping_guide)).
A 4 GB spatial file written that way is one OST's problem, and every rank
waiting on it waits on that one disk.

Two rules matter more than any number below:

- **A layout is a directory default.** Files created inside a striped
  directory inherit it. It cannot be changed on a file that already exists —
  restriping means recreating the file with `cp` (not `mv`, which preserves
  the layout).
- **Stripe wide only for large files.** A high stripe count on small files
  costs more in OST round-trips than it buys in bandwidth.

## The two setups write differently

This is the reason one set of numbers will not do. The S4F configs set
`output.format = "netcdf4_parallel"` with `output.compression_level = 2`, but
the two chinook templates route the writing very differently:

| template | who writes | pattern |
| --- | --- | --- |
| `chinook-apptainer.j2` | every MPI rank, collectively through parallel HDF5 | one shared file, many writers |
| `chinook-apptainer-async.j2` | `pism_async_writer`, one process | one large file, a single writer |

The async template starts `mpiexec -n 1 pism_async_writer : -n $((ntasks-4)) pism … -output.asynchronous`.
`pism_async_writer` is a serial Python program built on `netCDF4`, and it
refuses to start on more than one rank. So however many ranks the simulation
uses, exactly one process writes.

That splits the guidance:

- **Collective (`chinook-apptainer.j2`).** A shared file with many writers is
  the case the guide says should have "stripe count ideally equal to the
  number of processes" — capped by the 15 OSTs available, so `-c 15` is the
  candidate to beat. Large files also want a larger stripe size, `-S 32m`
  being the guide's suggestion.
- **Async (`chinook-apptainer-async.j2`).** One writer cannot use 15 OSTs the
  way 48 ranks can; a single client saturates somewhere lower. This is *not*
  the "file-per-process, use `-c 1`" case either — that advice is about many
  small files, and these are hundreds of MB to several GB. Expect the sweet
  spot in the middle, and measure it.

Both are hypotheses. The point of the harness is that they are cheap to test.

```{admonition} Compression interacts with all of this
:class: note

`output.compression_level = 2` means the parallel writes are *filtered*
collective HDF5 writes, which are far more sensitive to chunk and stripe
alignment than raw ones. If the collective numbers come out disappointing at
every stripe setting, compare against `output.compression_level = 0` before
concluding that striping does not help — the bottleneck may not be the
filesystem at all.
```

## Measuring

The harness lives in `benchmarks/lustre_striping/` and has its own
[README](https://github.com/pism/pism-terra/tree/main/benchmarks/lustre_striping)
with the exact commands. In outline:

1. **`stripe_probe.sh`** — `dd` into directories striped every which way, on
   a compute node, a couple of minutes. Reports one-stream and eight-stream
   bandwidth per setting: the first is the ceiling for the async writer, the
   second the aggregate the collective writes cannot exceed. Use it to shrink
   the matrix.
2. **`run_stripe_matrix.sh`** — one short forward run per (template, stripe
   count, stripe size), each into its own directory striped before the run
   creates anything inside it. `--dry-run` shows the matrix first.
3. **`collect_stripes.py`** — joins the manifest to `sacct` and to the bytes
   on disk, and prints MB/s per candidate.

Keep the runs short. `--end 1990-01-01` on a 1986 start is four model years,
enough spatial output to be dominated by writing rather than by the Blatter
solver.

## Applying the answer

Striping has to be set on the output directory *before* PISM writes into it,
which means in the run script, before the `mpiexec` lines:

```bash
lfs setstripe -c 8 -S 1m "{{ output_path }}"
```

Both templates already create the output tree through
`pism-glacier-run-forward`, so the `lfs setstripe` belongs alongside that.
On the 2026-09-10 measurement the count is there for consistency rather than
speed, and the size may as well stay at the filesystem default — nothing
separated 1m from 32m. Re-measure before treating either as settled: the
answer depends on how PISM writes and on how loaded the filesystem is, and
both change.

```{admonition} Measured 2026-09-10: striping does not make writes faster here
:class: important

Five shuffled repeats per setting could not separate any two layouts. The
fastest and slowest single-stream cells still overlap by ~150 MB/s, and so
does every pair between them — whatever striping does on `/import/c1` is
smaller than what the filesystem's other users do.

One thing did survive: **a single writer on one stripe is wildly
inconsistent**, ranging over a factor of 15 across five identical writes
(77–1128 MB/s), while at 8–15 stripes the same write varies by under a factor
of two. So a wide stripe buys *predictable* wall times for the async
template, not faster ones. Stripe size made no difference at all.

The numbers, and the reasoning behind reading them that way, are in the
harness README.
```

```{admonition} A full filesystem is its own problem
:class: note

Lustre's allocator degrades and starts steering away from the fullest OSTs
well before 100 %. At 90 % this is a plausible reason wide stripes may not
pay off, and it means a layout tuned today is tuned against today's fill
level. Worth re-checking after any large cleanup.
```
