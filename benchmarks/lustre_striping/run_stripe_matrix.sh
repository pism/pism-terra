#!/bin/bash
# Copyright (C) 2026 Andy Aschwanden
#
# This file is part of pism-terra.
#
# PISM-TERRA is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# Run one short PISM job per (template, stripe count, stripe size) and record
# what was submitted, so collect_stripes.py can turn it into a table.
#
# The two templates write very differently, so they are tuned separately:
#
#   chinook-apptainer.j2        every rank writes one shared file collectively
#                               (output.format = netcdf4_parallel)
#   chinook-apptainer-async.j2  pism_async_writer, a single serial netCDF4
#                               process, writes the whole file
#
# Striping is a directory default that new files inherit and that cannot be
# changed afterwards, so each candidate gets its own output directory, striped
# before the run creates anything inside it.

set -euo pipefail

RGI_ID="RGI2000-v7.0-C-01-04374"
CONFIG="pism_terra/config/s4f_historical_baseline_era5_pdd.toml"
REGRID_FILE=""
DATA_PATH="glacier_s4f_input"
BASE=""
CONTAINER=""
END="1990-01-01"
RESOLUTION="200m"
NTASKS=48
TASKS=24
STRESS_BALANCE="blatter"
TEMPLATES="chinook-apptainer.j2 chinook-apptainer-async.j2"
COUNTS="1 4 8 15"
SIZES="1m 32m"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: run_stripe_matrix.sh --base DIR --regrid-file FILE [options]

  --base DIR         Directory on Lustre the striped output trees go under
                     (required). One subdirectory per candidate.
  --regrid-file FILE Spin-up state to regrid from (required).
  --container FILE   Passed to sbatch as container=FILE, as the templates expect.
  --rgi-id ID        Default RGI2000-v7.0-C-01-04374.
  --config FILE      Default pism_terra/config/s4f_historical_baseline_era5_pdd.toml.
  --data-path DIR    Default glacier_s4f_input.
  --end DATE         Run end; keep it short so writes dominate. Default 1990-01-01.
  --resolution R     Default 200m.
  --ntasks N         Default 48.
  --tasks N          Default 24.
  --templates "..."  Default "chinook-apptainer.j2 chinook-apptainer-async.j2".
  --counts "..."     Stripe counts. Default "1 4 8 15".
  --sizes "..."      Stripe sizes. Default "1m 32m".
  --dry-run          Print what would happen; create and stripe nothing.
  -h, --help         This message.

The default matrix is 2 templates x 4 counts x 2 sizes = 16 jobs. Screen with
stripe_probe.sh first and cut --counts/--sizes down to what looks promising.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --base)           BASE="$2";           shift 2 ;;
        --regrid-file)    REGRID_FILE="$2";    shift 2 ;;
        --container)      CONTAINER="$2";      shift 2 ;;
        --rgi-id)         RGI_ID="$2";         shift 2 ;;
        --config)         CONFIG="$2";         shift 2 ;;
        --data-path)      DATA_PATH="$2";      shift 2 ;;
        --end)            END="$2";            shift 2 ;;
        --resolution)     RESOLUTION="$2";     shift 2 ;;
        --ntasks)         NTASKS="$2";         shift 2 ;;
        --tasks)          TASKS="$2";          shift 2 ;;
        --stress-balance) STRESS_BALANCE="$2"; shift 2 ;;
        --templates)      TEMPLATES="$2";      shift 2 ;;
        --counts)         COUNTS="$2";         shift 2 ;;
        --sizes)          SIZES="$2";          shift 2 ;;
        --dry-run)        DRY_RUN=1;           shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[ -n "$BASE" ]        || { echo "--base is required" >&2; usage >&2; exit 2; }
[ -n "$REGRID_FILE" ] || { echo "--regrid-file is required" >&2; usage >&2; exit 2; }

if [ "$DRY_RUN" -eq 0 ]; then
    command -v lfs >/dev/null 2>&1 || {
        echo "lfs not found: this host has no Lustre client. Use --dry-run to inspect the matrix." >&2
        exit 1
    }
    command -v sbatch >/dev/null 2>&1 || { echo "sbatch not found; run this on the cluster." >&2; exit 1; }
    mkdir -p "$BASE"
    lfs getstripe -d "$BASE" >/dev/null 2>&1 || {
        echo "$BASE is not on Lustre; striping does not apply there." >&2
        exit 1
    }
fi

MANIFEST="${BASE}/manifest.tsv"
if [ "$DRY_RUN" -eq 0 ]; then
    printf 'template\tcount\tsize\toutdir\tscript\tjobid\n' > "$MANIFEST"
fi

say() { echo "== $*"; }

for template in $TEMPLATES; do
    label="${template%.j2}"
    label="${label#chinook-apptainer}"
    label="${label#-}"
    [ -n "$label" ] || label="parallel"

    for count in $COUNTS; do
        for size in $SIZES; do
            out="${BASE}/${label}_c${count}_s${size}"
            say "$label  stripe count=$count size=$size  ->  $out"

            if [ "$DRY_RUN" -eq 1 ]; then
                echo "   lfs setstripe -c $count -S $size $out"
                echo "   pism-glacier-run-forward --end $END --stress-balance $STRESS_BALANCE \\"
                echo "       --regrid-file $REGRID_FILE --resolution $RESOLUTION \\"
                echo "       --ntasks $NTASKS --tasks $TASKS --data-path $DATA_PATH \\"
                echo "       --output-path $out $RGI_ID $CONFIG pism_terra/templates/$template"
                echo "   sbatch \$(ls -t $out/run_scripts/submit_*.sh | head -1)"
                continue
            fi

            # Stripe the tree root before anything is written into it: the
            # layout is inherited by the subdirectories and files PISM creates,
            # and cannot be applied to a file after the fact.
            mkdir -p "$out"
            lfs setstripe -c "$count" -S "$size" "$out"

            pism-glacier-run-forward \
                --end "$END" \
                --stress-balance "$STRESS_BALANCE" \
                --regrid-file "$REGRID_FILE" \
                --resolution "$RESOLUTION" \
                --ntasks "$NTASKS" \
                --tasks "$TASKS" \
                --data-path "$DATA_PATH" \
                --output-path "$out" \
                "$RGI_ID" \
                "$CONFIG" \
                "pism_terra/templates/$template"

            script=$(ls -t "$out"/run_scripts/submit_*.sh 2>/dev/null | head -1 || true)
            if [ -z "$script" ]; then
                echo "   no run script was generated; skipping" >&2
                continue
            fi

            if [ -n "$CONTAINER" ]; then
                jobid=$(container="$CONTAINER" sbatch --parsable "$script")
            else
                jobid=$(sbatch --parsable "$script")
            fi
            echo "   submitted $jobid  ($(basename "$script"))"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$label" "$count" "$size" "$out" "$script" "$jobid" >> "$MANIFEST"
        done
    done
done

if [ "$DRY_RUN" -eq 0 ]; then
    echo
    echo "manifest: $MANIFEST"
    echo "when the jobs finish:  python benchmarks/lustre_striping/collect_stripes.py --base $BASE"
fi
