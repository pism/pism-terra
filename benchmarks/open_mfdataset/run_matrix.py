"""Drive bench_one.py over a matrix of configs in fresh subprocesses; append JSON lines to results.jsonl."""

import json
import os
import subprocess
import sys

root = "data"
out = open(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl", "a")
matrix = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else []


def run(layout, **kw):
    cmd = [sys.executable, "bench_one.py", os.path.join(root, layout)]
    for k, v in kw.items():
        if v is True:
            cmd.append(f"--{k}")
        elif v is False or v is None:
            continue
        else:
            cmd += [f"--{k}", str(v)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if not line:
            rec = dict(
                layout=layout, **kw, error=r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else "no output"
            )
        else:
            rec = json.loads(line[-1])
    except subprocess.TimeoutExpired:
        rec = dict(layout=layout, **kw, error="timeout")
    print(json.dumps(rec), flush=True)
    out.write(json.dumps(rec) + "\n")
    out.flush()


for cfg in matrix:
    layouts = cfg.pop("layouts")
    for lay in layouts:
        run(lay, **cfg)
