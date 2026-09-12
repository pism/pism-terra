"""
Time the same short forward run on chosen chinook nodes, with and without the async writer.

The UQ ensembles of September 2026 showed a 3 to 6 times spread in wall time
between members with identical solver work, depending on the node. ``sacct``
put the forward leg at 2 to 5 busy cores of 24 while the PETSc-bound
inversion used 22, and the cancelled jobs showed the asynchronous writer
lagging the model by hundreds of records on the slow nodes. This tool pins
the same job to named nodes and times, per node:

1. importing the writer's Python stack from the container (cold, then warm);
2. a 512 MB ``dd`` to the output directory;
3. a netCDF append benchmark shaped like the spatial output;
4. a short forward leg (default four years) in three variants:
   ``async`` (production: separate ``pism_async_writer`` process),
   ``sync`` (PISM writes its own output in the production format),
   ``noout`` (no spatial output), ``sync_nc3`` (PISM writes classic
   NetCDF-3 through rank 0, no HDF5), ``sync_nc4s`` (serial NetCDF-4) and
   ``async_nc3`` (the production writer, but PISM's own scalar and state
   files in classic NetCDF-3). The monthly scalar flush is the step the
   two slow variants share and the fast one lacks, so the format variants
   separate HDF5-on-Lustre from the writer. ``async_nolock`` and
   ``sync_nolock`` export ``HDF5_USE_FILE_LOCKING=FALSE`` before the
   forward leg, the usual remedy for HDF5 stalls on Lustre.

Usage on chinook::

    python -m pism_terra.tools.chinook_node_io_test generate RUN_SCRIPT \\
        --nodes n31,n34,n112,n115 --test-dir /import/c1/.../node_io_test [--submit]
    python -m pism_terra.tools.chinook_node_io_test analyze /import/c1/.../node_io_test

``RUN_SCRIPT`` is a production ``submit_*.sh``; its environment block and the
main-leg PISM command are reused verbatim, so the test measures exactly the
production configuration. Each variant appends one line to
``<test-dir>/results.csv`` and ``analyze`` tabulates them next to the number
of model steps found in the logs.
"""

from __future__ import annotations

import re
import subprocess
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence
from pathlib import Path

VARIANTS = ("async", "sync", "noout", "sync_nc3", "sync_nc4s", "async_nc3", "async_nolock", "sync_nolock")

BENCH_NC = '''"""Append records to a netCDF-4 file the way the async writer does."""
import sys
import time

import netCDF4
import numpy as np

path, n_records, ny, nx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
nc = netCDF4.Dataset(path, "w", format="NETCDF4")
nc.createDimension("time", None)
nc.createDimension("y", ny)
nc.createDimension("x", nx)
variables = [nc.createVariable(f"v{i}", "f8", ("time", "y", "x"), zlib=True, complevel=1) for i in range(9)]
rng = np.random.default_rng(0)
t0 = time.time()
for k in range(n_records):
    for v in variables:
        v[k, :, :] = rng.random((ny, nx))
    nc.sync()
nc.close()
print(f"IOTEST ncbench_s {time.time() - t0:.2f}")
'''


def split_run_script(text: str) -> tuple[str, str]:
    """
    Split a production run script into its environment block and its main-leg command.

    Parameters
    ----------
    text : str
        Contents of a ``submit_*.sh`` rendered by pism-glacier-run(-inverse).

    Returns
    -------
    str
        Everything after the ``#SBATCH`` header up to and including the
        ``pismtasks=`` line: module loads, container check, ulimits.
    str
        The last ``mpiexec ... pism_async_writer : ... pism`` command, all
        continuation lines included.

    Raises
    ------
    ValueError
        If the script has no ``pismtasks=`` line or no async main leg.
    """
    lines = text.split("\n")
    try:
        end_env = next(i for i, l in enumerate(lines) if l.startswith("pismtasks="))
    except StopIteration as err:
        raise ValueError("no 'pismtasks=' line in the run script") from err
    env = "\n".join(l for l in lines[1 : end_env + 1] if not l.startswith("#SBATCH"))
    starts = [i for i, l in enumerate(lines) if l.startswith("mpiexec") and "pism_async_writer" in l]
    if not starts:
        raise ValueError("no 'mpiexec ... pism_async_writer' main leg in the run script")
    start = starts[-1]
    end = start
    while lines[end].rstrip().endswith("\\"):
        end += 1
    return env, "\n".join(lines[start : end + 1])


def variant_command(command: str, variant: str, end: str, outputs: dict[str, Path]) -> str:
    """
    Adapt the production main-leg command to a test variant.

    Parameters
    ----------
    command : str
        Main-leg command from :func:`split_run_script`.
    variant : str
        One of :data:`VARIANTS`.
    end : str
        New ``-time.end`` date.
    outputs : dict of str to Path
        Test paths for the ``state``, ``scalar`` and ``spatial`` files.

    Returns
    -------
    str
        The modified command.

    Raises
    ------
    ValueError
        If ``variant`` is unknown.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    cmd = re.sub(r"-time\.end \S+", f"-time.end {end}", command)
    cmd = re.sub(r"-output\.file \S+", f"-output.file {outputs['state']}", cmd)
    cmd = re.sub(r"-output\.scalar\.file \S+", f"-output.scalar.file {outputs['scalar']}", cmd)
    cmd = re.sub(r"-output\.spatial\.file \S+", f"-output.spatial.file {outputs['spatial']}", cmd)
    cmd = re.sub(r"\s*-output\.checkpoint\.interval \S+\s*\\\n", "\n", cmd)
    if variant not in ("async", "async_nc3", "async_nolock"):
        cmd = re.sub(r"^mpiexec -n 1\s+apptainer run \$\{container\} pism_async_writer : -n", "mpiexec -n", cmd)
        cmd = cmd.replace(" -output.asynchronous", "")
    if variant in ("sync_nc3", "async_nc3"):
        cmd = re.sub(r"-output\.format \S+", "-output.format netcdf3", cmd)
    if variant == "sync_nc4s":
        cmd = re.sub(r"-output\.format \S+", "-output.format netcdf4_serial", cmd)
    if variant == "noout":
        cmd = re.sub(r"^\s*-output\.spatial\.\S+ \S+\s*\\\n", "", cmd, flags=re.M)
    return cmd


def job_script(
    env: str, command: str, *, node: str, variant: str, test_dir: Path, partition: str, ntasks: int, walltime: str
) -> str:
    """
    Render one test job for a node and a variant.

    Parameters
    ----------
    env : str
        Environment block from the production script.
    command : str
        Main-leg command already adapted by :func:`variant_command`.
    node : str
        Node the job is pinned to with ``--nodelist``.
    variant : str
        Output variant, used in file and job names.
    test_dir : Path
        Directory holding ``logs/``, ``output/``, ``bench_nc.py`` and ``results.csv``.
    partition : str
        Slurm partition.
    ntasks : int
        Tasks per job (one node).
    walltime : str
        Slurm time limit.

    Returns
    -------
    str
        The sbatch script.
    """
    name = f"iotest_{node}_{variant}"
    out = test_dir / "output"
    nolock = "export HDF5_USE_FILE_LOCKING=FALSE\n" if variant.endswith("_nolock") else ""
    return f"""#!/bin/sh
#SBATCH --partition={partition}
#SBATCH --nodelist={node}
#SBATCH --ntasks={ntasks}
#SBATCH --tasks-per-node={ntasks}
#SBATCH --time={walltime}
#SBATCH --job-name={name}
#SBATCH --output={test_dir}/logs/{name}.%j
{env}

stamp() {{ echo "IOTEST $1 $(date +%s.%N)"; }}
secs() {{ awk -v a="$1" -v b="$2" 'BEGIN {{ printf "%.2f", b - a }}'; }}

echo "IOTEST node $(hostname) job $SLURM_JOB_ID variant {variant}"
lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread' | sed 's/^/IOTEST lscpu /'
echo "IOTEST loadavg $(cat /proc/loadavg)"
command -v lfs >/dev/null 2>&1 && lfs setstripe -c 8 {out} 2>/dev/null || true
command -v lfs >/dev/null 2>&1 && lfs getstripe -d {out} 2>/dev/null | sed 's/^/IOTEST stripe /' || true

# 1. importing the writer's stack from the container: cold, then warm
t0=$(date +%s.%N); apptainer run ${{container}} python3 -c "import yac, netCDF4, numpy, pyproj, mpi4py"; t1=$(date +%s.%N)
apptainer run ${{container}} python3 -c "import yac, netCDF4, numpy, pyproj, mpi4py"; t2=$(date +%s.%N)
import_cold=$(secs $t0 $t1); import_warm=$(secs $t1 $t2)
echo "IOTEST import_cold_s $import_cold import_warm_s $import_warm"

# 2. raw write throughput to the output directory
t0=$(date +%s.%N); dd if=/dev/zero of={out}/dd_{name}_$SLURM_JOB_ID.bin bs=1M count=512 conv=fsync 2>/dev/null; t1=$(date +%s.%N)
rm -f {out}/dd_{name}_$SLURM_JOB_ID.bin
dd_s=$(secs $t0 $t1); echo "IOTEST dd_512MB_s $dd_s"

# 3. netCDF append benchmark shaped like the spatial output (48 records, 9 variables)
ncbench=$(apptainer run ${{container}} python3 {test_dir}/bench_nc.py {out}/ncbench_{name}_$SLURM_JOB_ID.nc 48 112 84 | tee -a /dev/stderr | awk '/IOTEST ncbench_s/ {{print $3}}')
rm -f {out}/ncbench_{name}_$SLURM_JOB_ID.nc

# 4. the forward leg, variant {variant}
{nolock}rm -f {out}/{name}_state.nc {out}/{name}_scalar.nc {out}/{name}_spatial.nc
t0=$(date +%s.%N); stamp forward_start
{command}
t1=$(date +%s.%N); stamp forward_end
forward_s=$(secs $t0 $t1)
echo "IOTEST forward_s $forward_s"
echo "{node},{variant},$SLURM_JOB_ID,$import_cold,$import_warm,$dd_s,$ncbench,$forward_s" >> {test_dir}/results.csv
"""


def generate(args) -> int:
    """
    Write the test jobs and, optionally, submit them.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``generate`` arguments.

    Returns
    -------
    int
        Exit status.
    """
    text = Path(args.RUN_SCRIPT).read_text(encoding="utf-8")
    env, command = split_run_script(text)
    test_dir = Path(args.test_dir).expanduser().resolve()
    for sub in ("logs", "output", "scripts"):
        (test_dir / sub).mkdir(parents=True, exist_ok=True)
    (test_dir / "bench_nc.py").write_text(BENCH_NC, encoding="utf-8")
    results = test_dir / "results.csv"
    if not results.exists():
        results.write_text(
            "node,variant,jobid,import_cold_s,import_warm_s,dd_512MB_s,ncbench_s,forward_s\n", encoding="utf-8"
        )
    scripts = []
    for node in args.nodes.split(","):
        for variant in args.variants.split(","):
            name = f"iotest_{node}_{variant}"
            outputs = {k: test_dir / "output" / f"{name}_{k}.nc" for k in ("state", "scalar", "spatial")}
            cmd = variant_command(command, variant, args.end, outputs)
            script = test_dir / "scripts" / f"{name}.sh"
            script.write_text(
                job_script(
                    env,
                    cmd,
                    node=node,
                    variant=variant,
                    test_dir=test_dir,
                    partition=args.partition,
                    ntasks=args.ntasks,
                    walltime=args.walltime,
                ),
                encoding="utf-8",
            )
            scripts.append(script)
    for script in scripts:
        if args.submit:
            subprocess.run(["sbatch", str(script)], check=True)
        else:
            print(f"sbatch {script}")
    print(f"{len(scripts)} job scripts in {test_dir / 'scripts'}", file=sys.stderr)
    return 0


def analyze(args) -> int:
    """
    Tabulate the results of the test jobs.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``analyze`` arguments.

    Returns
    -------
    int
        Exit status.
    """
    import pandas as pd  # pylint: disable=import-outside-toplevel

    test_dir = Path(args.TEST_DIR).expanduser()
    results = pd.read_csv(test_dir / "results.csv")
    steps, cpu_model = {}, {}
    for log in (test_dir / "logs").glob("iotest_*"):
        text = log.read_text(encoding="utf-8", errors="replace")
        try:
            jobid = int(log.suffix.lstrip("."))
        except ValueError:
            continue
        main = text[text.rfind("PISM (basic evolution run mode)") :]
        steps[jobid] = len(re.findall(r"^S \d{4}-\d{2}-\d{2}", main, re.M))
        model = re.search(r"IOTEST lscpu Model name:\s*(.*)", text)
        cpu_model[jobid] = model.group(1).strip() if model else ""
    results["steps"] = results["jobid"].map(steps)
    results["s_per_step"] = results["forward_s"] / results["steps"]
    results["cpu"] = results["jobid"].map(cpu_model)
    pd.set_option("display.width", 200)
    cols = [
        "node",
        "variant",
        "jobid",
        "import_cold_s",
        "import_warm_s",
        "dd_512MB_s",
        "ncbench_s",
        "forward_s",
        "steps",
        "s_per_step",
        "cpu",
    ]
    print(results[cols].sort_values(["variant", "s_per_step"]).round(2).to_string(index=False))
    pivot = results.pivot_table(index="node", columns="variant", values="s_per_step")
    print("\nseconds per model step by node and variant:")
    print(pivot.round(1).to_string())
    print("\nsacct for the forward legs (cores busy = TotalCPU / Elapsed):")
    print("  sacct -j " + ",".join(str(j) for j in results["jobid"]) + " -o JobID,NodeList,Elapsed,TotalCPU,MaxRSS")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """
    Command-line entry point.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Exit status.
    """
    parser = ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0], formatter_class=ArgumentDefaultsHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Write (and optionally submit) one test job per node and variant.")
    gen.add_argument("RUN_SCRIPT", help="Production submit_*.sh whose environment and main leg are reused.")
    gen.add_argument("--nodes", required=True, help="Comma-separated node names, e.g. n31,n34,n112,n115.")
    gen.add_argument(
        "--variants", default="async,sync,noout", help="Comma-separated subset of " + ",".join(VARIANTS) + "."
    )
    gen.add_argument("--test-dir", required=True, help="Directory for scripts, logs, outputs and results.csv.")
    gen.add_argument("--end", default="1990-01-01", help="time.end of the shortened forward leg.")
    gen.add_argument("--partition", default="t2small", help="Slurm partition.")
    gen.add_argument("--ntasks", type=int, default=24, help="Tasks per job.")
    gen.add_argument("--walltime", default="03:00:00", help="Slurm time limit per job.")
    gen.add_argument("--submit", action="store_true", default=False, help="Run sbatch on each script.")
    gen.set_defaults(func=generate)
    ana = sub.add_parser("analyze", help="Tabulate results.csv and the logs of a finished test.")
    ana.add_argument("TEST_DIR", help="The --test-dir given to generate.")
    ana.set_defaults(func=analyze)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
