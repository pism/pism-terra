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
# Measure raw Lustre write bandwidth per stripe setting, without running PISM.
#
# A screening step: it takes a couple of minutes and narrows the candidates
# worth spending node hours on in run_stripe_matrix.sh. Two numbers per
# setting:
#
#   single   one stream writing one file      -> models the async setup, where
#                                                pism_async_writer is a single
#                                                serial netCDF4 process
#   striped  N streams writing N files        -> the aggregate the OSTs can
#                                                take, an upper bound on what
#                                                the collective (parallel)
#                                                setup can reach
#
# Neither models collective HDF5 exactly; that is what the end-to-end matrix
# is for. Run this on a compute node, not the login node.

set -euo pipefail

PATH_BASE=""
SIZE_GB=2
COUNTS="1 2 4 8 15"
SIZES="1m 32m"
STREAMS=8
REPEATS=3
KEEP=0

usage() {
    cat <<'EOF'
Usage: stripe_probe.sh --path DIR [options]

  --path DIR       Directory on Lustre to probe in (required). A scratch
                   subdirectory is created and removed.
  --size-gb N      Bytes written per stream, GiB (default 2).
  --counts "..."   Stripe counts to try (default "1 2 4 8 15").
  --sizes "..."    Stripe sizes to try, lfs -S syntax (default "1m 32m").
  --streams N      Concurrent streams for the striped measurement (default 8).
  --repeats N      Measurements per setting (default 3). The filesystem is
                   shared, so one shot per setting cannot be told apart from
                   the neighbours' load; the median of several can.
  --keep           Leave the scratch directory in place.
  -h, --help       This message.

Example, on a compute node:
  srun -n1 -p t2small --time=00:20:00 --pty \
      benchmarks/lustre_striping/stripe_probe.sh --path /import/c1/ICESHEET/ICESHEET/pism-terra
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --path)    PATH_BASE="$2"; shift 2 ;;
        --size-gb) SIZE_GB="$2";   shift 2 ;;
        --counts)  COUNTS="$2";    shift 2 ;;
        --sizes)   SIZES="$2";     shift 2 ;;
        --streams) STREAMS="$2";   shift 2 ;;
        --repeats) REPEATS="$2";   shift 2 ;;
        --keep)    KEEP=1;         shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[ -n "$PATH_BASE" ] || { echo "--path is required" >&2; usage >&2; exit 2; }

command -v lfs >/dev/null 2>&1 || {
    echo "lfs not found: this host has no Lustre client, so there is nothing to tune." >&2
    exit 1
}
[ -d "$PATH_BASE" ] || { echo "no such directory: $PATH_BASE" >&2; exit 1; }
lfs getstripe -d "$PATH_BASE" >/dev/null 2>&1 || {
    echo "$PATH_BASE is not on Lustre (lfs getstripe failed); striping does not apply there." >&2
    exit 1
}

echo "# filesystem"
lfs df -h "$PATH_BASE" | tail -n +1 | sed 's/^/  /'
echo
echo "# current default layout of $PATH_BASE"
lfs getstripe -d "$PATH_BASE" | sed 's/^/  /'
echo

SCRATCH="$PATH_BASE/.stripe_probe.$$"
mkdir -p "$SCRATCH"
cleanup() { [ "$KEEP" -eq 1 ] || rm -rf "$SCRATCH"; }
trap cleanup EXIT

# dd writes with conv=fsync so the timing includes the flush to the OSTs
# rather than stopping at the client's page cache.
write_one() {
    local target="$1" mb="$2"
    dd if=/dev/zero of="$target" bs=1M count="$mb" conv=fsync status=none
}

elapsed() {  # start_ns end_ns -> seconds, 3 dp
    awk -v a="$1" -v b="$2" 'BEGIN { printf "%.3f", (b - a) / 1e9 }'
}

rate() {     # megabytes seconds -> MB/s, 1 dp
    awk -v m="$1" -v s="$2" 'BEGIN { printf "%.1f", (s > 0) ? m / s : 0 }'
}

MB=$((SIZE_GB * 1024))

# One directory per layout, created and striped up front: a Lustre layout is a
# directory default that files inherit, so this has to exist before any write.
for count in $COUNTS; do
    for size in $SIZES; do
        dir="$SCRATCH/c${count}_s${size}"
        mkdir -p "$dir"
        lfs setstripe -c "$count" -S "$size" "$dir"
    done
done

# Build the full list of trials, then shuffle it. Run in nested-loop order,
# whatever the filesystem happens to be doing over the ten minutes this takes
# is charged to whichever setting is running at the time; shuffling spreads
# that load across every setting instead of aliasing it onto one.
TRIALS="$SCRATCH/trials.txt"
: > "$TRIALS"
for count in $COUNTS; do
    for size in $SIZES; do
        for kind in single multi; do
            for _ in $(seq 1 "$REPEATS"); do
                echo "$count $size $kind" >> "$TRIALS"
            done
        done
    done
done
if command -v shuf >/dev/null 2>&1; then
    # Not ``shuf "$TRIALS" -o "$TRIALS"``: unlike sort, shuf has no in-place
    # guarantee and truncates the file it is reading.
    shuf "$TRIALS" > "${TRIALS}.shuffled"
    mv "${TRIALS}.shuffled" "$TRIALS"
fi

# A throwaway write first: the first measurement of a run pays for cache
# warming and directory metadata, and would otherwise be charged to whichever
# setting drew the short straw.
warmup_dir=$(dirname "$(head -1 "$TRIALS" | awk -v s="$SCRATCH" '{print s "/c" $1 "_s" $2 "/x"}')")
write_one "$warmup_dir/warmup.dat" "$MB"
rm -f "$warmup_dir/warmup.dat"

RESULTS="$SCRATCH/results.txt"
: > "$RESULTS"
total=$(wc -l < "$TRIALS")
n=0
while read -r count size kind; do
    n=$((n + 1))
    dir="$SCRATCH/c${count}_s${size}"
    # Only when stderr is a terminal: the carriage return is meant to
    # overwrite in place, and turns a captured log into one long line.
    [ -t 2 ] && printf '\r  trial %d/%d (c=%s S=%s %s)   ' "$n" "$total" "$count" "$size" "$kind" >&2

    start=$(date +%s%N)
    if [ "$kind" = "single" ]; then
        write_one "$dir/single.dat" "$MB"
        end=$(date +%s%N)
        mb=$MB
    else
        for i in $(seq 1 "$STREAMS"); do
            write_one "$dir/stream_$i.dat" "$MB" &
        done
        wait
        end=$(date +%s%N)
        mb=$((MB * STREAMS))
    fi
    echo "$count $size $kind $(rate "$mb" "$(elapsed "$start" "$end")")" >> "$RESULTS"
    rm -f "$dir"/*.dat
done < "$TRIALS"
[ -t 2 ] && printf '\r%*s\r' 60 "" >&2
true  # the [ -t 2 ] test above must not decide the script's exit status

# median and range of the repeats, so a difference can be judged against the
# scatter that produced it
stat_of() {  # count size kind field -> median or "min-max"
    local values
    values=$(awk -v c="$1" -v s="$2" -v k="$3" '$1==c && $2==s && $3==k {print $4}' "$RESULTS" | sort -n)
    awk -v what="$4" '
        { v[NR] = $1 }
        END {
            if (NR == 0) { print "-"; exit }
            if (what == "median") {
                printf "%.0f", (NR % 2) ? v[(NR+1)/2] : (v[NR/2] + v[NR/2+1]) / 2
            } else {
                printf "%.0f-%.0f", v[1], v[NR]
            }
        }' <<< "$values"
}

printf '%-7s %-6s %12s %13s %14s %15s\n' count size "single" "(range)" "${STREAMS}-stream" "(range)"
printf '%-7s %-6s %12s %13s %14s %15s\n' ----- ---- ----------- ------------ ------------- --------------
for count in $COUNTS; do
    for size in $SIZES; do
        printf '%-7s %-6s %12s %13s %14s %15s\n' \
            "$count" "$size" \
            "$(stat_of "$count" "$size" single median)" \
            "$(stat_of "$count" "$size" single range)" \
            "$(stat_of "$count" "$size" multi median)" \
            "$(stat_of "$count" "$size" multi range)"
    done
done
echo
echo "medians of $REPEATS shuffled repeats, MB/s; compare differences against the ranges"

echo
echo "single  -> the async setup's ceiling (one serial writer process)"
echo "${STREAMS}-stream -> aggregate OST bandwidth; the collective setup cannot beat it"
