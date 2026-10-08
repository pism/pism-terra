#!/usr/bin/env python3
# Copyright (C) 2026 Andy Aschwanden
#
# This file is part of pism-terra.
#
# PISM-TERRA is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# Turn a stripe matrix into a table: how long each job took and how much it
# wrote, per (template, stripe count, stripe size).

"""
Collect the results of a Lustre stripe matrix into a comparison table.

Reads the manifest ``run_stripe_matrix.sh`` wrote, asks ``sacct`` how long
each job ran, measures what landed on disk, and prints the effective write
rate per candidate. The winner per template is what the run templates should
stripe to.

Effective MB/s is bytes-written over *total* wall time, so it includes
compute — it ranks candidates, it is not a filesystem benchmark. Keep the run
short (``--end`` a few years) so writes are a large share of the total, and
read ``stripe_probe.sh`` for the raw bandwidth numbers.

    python benchmarks/lustre_striping/collect_stripes.py --base /path/to/striping
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def sacct_elapsed(jobid: str) -> tuple[str, float | None]:
    """
    Ask sacct for one job's state and elapsed seconds.

    Parameters
    ----------
    jobid : str
        SLURM job id.

    Returns
    -------
    tuple
        ``(state, seconds)``; ``seconds`` is None when the job has not
        finished or sacct knows nothing about it.
    """
    try:
        out = subprocess.run(
            ["sacct", "-j", jobid, "--noheader", "--parsable2", "--format=State,ElapsedRaw"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout
    except FileNotFoundError:
        return "no-sacct", None

    for line in out.splitlines():
        # The first line is the job allocation itself; later ones are steps.
        state, _, elapsed = line.partition("|")
        state = state.strip()
        if not state:
            continue
        try:
            return state, float(elapsed)
        except ValueError:
            return state, None
    return "unknown", None


def written_bytes(outdir: Path) -> int:
    """
    Total size of the NetCDF output under one candidate's tree.

    Only ``.nc`` files count: the run scripts, logs and staged inputs are the
    same for every candidate and would dilute the comparison.

    Parameters
    ----------
    outdir : pathlib.Path
        The candidate's output directory.

    Returns
    -------
    int
        Bytes.
    """
    return sum(f.stat().st_size for f in outdir.rglob("*.nc") if f.is_file())


def main() -> int:
    """
    Print the stripe matrix as a table, and write it beside the manifest.

    Returns
    -------
    int
        0 on success, 1 when the manifest is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="Directory run_stripe_matrix.sh wrote to.")
    parser.add_argument("--csv", type=Path, default=None, help="Where to write the table (default <base>/results.csv).")
    options = parser.parse_args()

    manifest = options.base / "manifest.tsv"
    if not manifest.exists():
        print(f"no manifest at {manifest}; run run_stripe_matrix.sh first", file=sys.stderr)
        return 1

    rows = []
    with manifest.open() as handle:
        for entry in csv.DictReader(handle, delimiter="\t"):
            outdir = Path(entry["outdir"])
            state, elapsed = sacct_elapsed(entry["jobid"])
            written = written_bytes(outdir) if outdir.exists() else 0
            mb = written / 1e6
            rows.append(
                {
                    "template": entry["template"],
                    "count": int(entry["count"]),
                    "size": entry["size"],
                    "jobid": entry["jobid"],
                    "state": state,
                    "elapsed_s": elapsed,
                    "written_mb": round(mb, 1),
                    "effective_mb_s": round(mb / elapsed, 1) if elapsed else None,
                }
            )

    out_csv = options.csv or options.base / "results.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["template"])
        writer.writeheader()
        writer.writerows(rows)

    header = f"{'template':<10} {'count':>5} {'size':>5} {'state':>10} {'elapsed_s':>10} {'written_MB':>11} {'MB/s':>8}"
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: (r["template"], r["count"], r["size"])):
        # ``is not None`` rather than truthiness: a job that wrote almost
        # nothing has a real rate of 0.0, which is a result, not a gap.
        elapsed_text = f"{row['elapsed_s']:.0f}" if row["elapsed_s"] is not None else "-"
        rate_text = f"{row['effective_mb_s']:.1f}" if row["effective_mb_s"] is not None else "-"
        print(
            f"{row['template']:<10} {row['count']:>5} {row['size']:>5} {row['state']:>10} "
            f"{elapsed_text:>10} {row['written_mb']:>11.1f} {rate_text:>8}"
        )

    incomplete = [r for r in rows if r["elapsed_s"] is None]
    if incomplete:
        print(f"\n{len(incomplete)} job(s) have no elapsed time yet — still queued or running.")
    print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
