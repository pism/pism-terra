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

A first pass with one measurement per setting, before `--repeats` existed:

| count | size | single MB/s | 8-stream MB/s |
|---|---|---|---|
| 1 | 1m | 154.8 | 649.9 |
| 1 | 32m | 795.6 | 575.3 |
| 2 | 1m | 653.9 | 1122.2 |
| 2 | 32m | 598.0 | 1513.0 |
| 4 | 1m | 282.8 | 940.2 |
| 4 | 32m | 294.0 | 1758.5 |
| 8 | 1m | 420.1 | 752.0 |
| 8 | 32m | 375.8 | 1064.2 |
| 15 | 1m | 351.6 | 1101.9 |
| 15 | 32m | 252.2 | 1060.0 |

**These do not rank the settings.** They are non-monotonic in ways no stripe
effect explains — `c=1/32m` is the best single-stream figure and `c=1/1m` the
worst, `c=2/1m` beats `c=4/1m`, and the 8-stream column peaks at `c=4` then
falls back at `c=8` and `c=15`. One shot per setting, run in a fixed order on
a filesystem shared with everyone else, cannot separate a layout from a busy
neighbour; the first cell measured (154.8) is very likely cold-start cost.
`--repeats` with shuffled trial order exists because of this run.

Two things it does establish:

- **The aggregate ceiling is around 1.7 GB/s** and a **single writer tops out
  near 0.8 GB/s** — so the async setup's one serial process gives up roughly
  half the bandwidth the OSTs can absorb, whatever the striping. That is a
  point about the two templates, not about layout.
- **The filesystem is 90 % full** (OSTs 88–93 %). Lustre allocation degrades
  at that fill level and steers away from the fullest OSTs, which is a
  plausible reason wide stripes did not win. Any tuning done now is tuning
  against that state.

Re-run with repeats before drawing conclusions:

```bash
srun -n1 -p t2small --time=01:00:00 --pty \
    benchmarks/lustre_striping/stripe_probe.sh \
    --path /import/c1/ICESHEET/ICESHEET/pism-terra --repeats 5
```

### Matrix

Not yet run. Fill this in from `results.csv`, then set the winning layout in
`pism_terra/templates/chinook-apptainer.j2` and `chinook-apptainer-async.j2`
and note the date and the PISM version here — the answer depends on both the
filesystem's state and on how PISM writes.

| template | stripe count | stripe size | elapsed (s) | written (MB) | MB/s |
|---|---|---|---|---|---|
| parallel | | | | | |
| async | | | | | |
