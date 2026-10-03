# Lustre striping

Finds the stripe count and stripe size that PISM output should be written
with on chinook, separately for the two write strategies the run templates
use. The reasoning behind the candidates, and where the answer gets applied, is
in `docs/source/developer/lustre_striping.md` — read that first. This
directory is the machinery.

Everything here has to run on chinook: `lfs` only exists on a Lustre client,
and the numbers only mean anything on the real filesystem.

## 1. Screen the candidates

```bash
srun -n1 -p t2small --time=00:20:00 --pty \
    benchmarks/lustre_striping/stripe_probe.sh --path /import/c1/ICESHEET/ICESHEET/pism-terra
```

Writes with `dd` into directories striped every which way and reports two
rates per setting: one stream (what the async writer, a single serial
process, can reach) and eight concurrent streams (roughly what the OSTs can
absorb in aggregate, an upper bound for the collective writes). Takes a
couple of minutes and costs one core. Use it to cut the matrix below down
from 16 jobs to the four or five worth running.

## 2. Run the matrix

```bash
benchmarks/lustre_striping/run_stripe_matrix.sh \
    --base /import/c1/ICESHEET/ICESHEET/pism-terra/2026_09_striping \
    --regrid-file /import/c1/ICESHEET/ICESHEET/pism-terra/2026_09_s4f_iceflow_calib_ho_tw/RGI2000-v7.0-C-01-04374/output/state/state_g500m_RGI2000-v7.0-C-01-04374_id_0_0001-01-01_0501-01-01.nc \
    --container /import/c1/ICESHEET/ICESHEET/pism-terra/pism-insolation.sif \
    --counts "1 8 15" --sizes "1m 32m"
```

One short forward run per (template, stripe count, stripe size). Each gets
its own output tree, striped **before** anything is written into it — a
Lustre layout is a directory default that new files inherit, and it cannot be
changed on a file that already exists.

`--dry-run` prints the matrix without creating or submitting anything; start
there to check the paths.

## 3. Read the result

```bash
python benchmarks/lustre_striping/collect_stripes.py --base /import/c1/ICESHEET/ICESHEET/pism-terra/2026_09_striping
```

Joins the manifest to `sacct` and to what actually landed on disk:

```
template   count  size      state  elapsed_s  written_MB     MB/s
parallel       1   32m  COMPLETED       1830      4210.0      2.3
parallel      15   32m  COMPLETED        980      4210.0      4.3
async          1   32m  COMPLETED       1150      4210.0      3.7
...
```

`MB/s` is bytes over *total* wall time, compute included, so it ranks
candidates rather than benchmarking the filesystem. Keep `--end` short so
writing is a large share of the run.

## Recorded results

### Probe, 2026-09-10, `/import/c1` (lustre2, 15 OSTs, 90 % full)

Medians of 5 shuffled repeats per setting, 2 GiB per stream, MB/s:

| count | size | single | (range) | 8-stream | (range) |
|---|---|---|---|---|---|
| 1 | 1m | 507 | 77–1128 | 911 | 577–1282 |
| 1 | 32m | 467 | 107–1033 | 1093 | 577–1175 |
| 2 | 1m | 243 | 131–661 | 794 | 592–971 |
| 2 | 32m | 748 | 198–830 | 747 | 612–1693 |
| 4 | 1m | 228 | 159–346 | 722 | 523–935 |
| 4 | 32m | 283 | 193–406 | 1277 | 784–1379 |
| 8 | 1m | 407 | 326–462 | 1221 | 679–1460 |
| 8 | 32m | 417 | 334–495 | 883 | 711–1192 |
| 15 | 1m | 524 | 279–545 | 878 | 753–1204 |
| 15 | 32m | 467 | 396–545 | 1046 | 751–1322 |

**No setting is faster than any other, to the precision available.** The
best and worst single-stream cells (c=2/32m at 748, c=4/1m at 228) have
ranges that still overlap by 148 MB/s; the 8-stream extremes overlap by 151.
Every pair in between overlaps more. Whatever stripe count and stripe size do
on this filesystem is smaller than what the other users on it do, and five
repeats cannot see past that.

**What does survive is the scatter.** Relative spread of the single-stream
measurement, range width over median:

| count | 1m | 32m |
|---|---|---|
| 1 | 2.07 | 1.98 |
| 2 | 2.18 | 0.84 |
| 4 | 0.82 | 0.75 |
| 8 | 0.33 | 0.39 |
| 15 | 0.51 | 0.32 |

A single writer on one stripe is hostage to whichever OST it landed on and to
whatever else is hitting that OST: throughput ranged over a factor of 15
(77–1128 MB/s) across five identical writes. Spread over 8–15 OSTs the same
write varies by well under a factor of two. The effect is consistent down
both stripe-size columns, which is more than any of the medians manage —
though with five samples the min–max is a crude estimate and this is a
tendency, not a measurement.

So, for this filesystem in this state:

- **Striping will not make PISM write faster.** Do not expect a win.
- **A wide stripe makes the async path predictable.** `-c 8` or `-c 15` is
  worth setting on output directories for the single-writer template, for
  wall times that vary less between otherwise identical jobs. It also
  satisfies the UAF guide's "avoid leaving very large files on 1–2 OSTs".
- **Stripe size does not separate.** 1m and 32m are interchangeable here.
- **The 8-stream aggregate (~0.7–1.3 GB/s) is roughly twice the single-writer
  figure**, unchanged from the first pass: the async writer forfeits about
  half the available bandwidth by being one process, which is a bigger lever
  than any layout.

### Matrix

Not run, and on this evidence not worth 16 node-hours: the probe cannot
separate the settings at the filesystem level, and an end-to-end PISM run
adds compute noise on top of the same I/O noise. Worth revisiting if the
filesystem empties out or if a run turns out to be write-bound in a way the
probe does not capture (filtered collective HDF5 is not modelled by `dd`).

Fill this in from `results.csv` if it does get run, then set the layout in
`pism_terra/templates/chinook-apptainer.j2` and `chinook-apptainer-async.j2`
and note the date and the PISM version here — the answer depends on both the
filesystem's state and on how PISM writes.

| template | stripe count | stripe size | elapsed (s) | written (MB) | MB/s |
|---|---|---|---|---|---|
| parallel | | | | | |
| async | | | | | |
