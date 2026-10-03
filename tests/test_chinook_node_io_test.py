"""Parsing and rewriting of production run scripts by the chinook node I/O test."""

from pathlib import Path

import pytest

from pism_terra.tools import chinook_node_io_test as t

ENV = "set -e\nmodule purge\nexport container=x.sif\n\npismtasks=$(( SLURM_NTASKS - 4 ))\n\n"
PISMI = "mpiexec -n  ${SLURM_NTASKS} apptainer run  ${container} pismi -inverse.file obs.nc\n"


def _script(writer: str, ntasks: int, per_node: int, spatial: str, start: str = "1986-01-01") -> str:
    """
    Render a minimal production-like run script.

    Parameters
    ----------
    writer : str
        Writer launch prefix, e.g. ``pism_async_writer`` or ``pism_ismip7_writer -r 1``.
    ntasks : int
        ``--ntasks`` of the header.
    per_node : int
        ``--tasks-per-node`` of the header.
    spatial : str
        Value of ``-output.spatial.file``.
    start : str, optional
        ``-time.start``.

    Returns
    -------
    str
        The script.
    """
    header = f"#!/bin/sh\n#SBATCH --partition=t2small\n#SBATCH --ntasks={ntasks}\n#SBATCH --tasks-per-node={per_node}\n"
    launch = f"mpiexec -n 1  apptainer run ${{container}} {writer} : -n ${{pismtasks}} apptainer run ${{container}} pism   -atmosphere.models given  \\\n"
    flags = (
        "  -output.checkpoint.interval 24  \\\n  -output.file /p/state.nc  \\\n  -output.format netcdf4_parallel  \\\n"
        "  -output.scalar.file /p/scalar.nc  \\\n"
        f"  -output.spatial.file {spatial}  \\\n  -output.spatial.times yearly  \\\n  -output.spatial.vars ismip  \\\n"
        f"  -time.end 2025-01-01  \\\n  -time.start {start}  \\\n  -time_stepping.skip.max 100 -output.asynchronous\n"
    )
    return header + ENV + PISMI + launch + flags + "\npism-postprocess-scalar /p/spatial.nc\n"


GLACIER = _script("pism_async_writer", 24, 24, "/p/spatial.nc")
ISMIP7 = _script("pism_ismip7_writer -r 1", 96, 48, "/p/{var}_GrIS.nc", start="1985-01-01")
OUTPUTS = {k: Path(f"/t/{k}.nc") for k in ("state", "scalar", "spatial")}


@pytest.mark.parametrize("script", [GLACIER, ISMIP7], ids=["glacier", "ismip7"])
def test_split_and_header(script: str) -> None:
    """
    The env block ends with pismtasks, the main leg is the pism launch, and the header is read.

    Parameters
    ----------
    script : str
        Rendered run script.
    """
    env, command = t.split_run_script(script)
    assert env.rstrip().endswith("pismtasks=$(( SLURM_NTASKS - 4 ))") and "#SBATCH" not in env
    assert (
        command.startswith("mpiexec -n 1")
        and "pismi" not in command
        and command.rstrip().endswith("-output.asynchronous")
    )
    header = t.sbatch_options(script)
    assert header["partition"] == "t2small" and int(header["ntasks"]) in (24, 96)


def test_variants_for_the_glacier_script() -> None:
    """Async keeps the writer, sync and noout drop it, format variants rewrite output.format."""
    _, command = t.split_run_script(GLACIER)
    for variant in t.VARIANTS:
        cmd = t.variant_command(command, variant, "1990-01-01", OUTPUTS)
        assert "-time.end 1990-01-01" in cmd and "-output.file /t/state.nc" in cmd and "checkpoint" not in cmd
        if variant in t.ASYNC_VARIANTS:
            assert cmd.startswith("mpiexec -n 1  apptainer run ${container} pism_async_writer : -n ${pismtasks}")
            assert cmd.count("-output.asynchronous") == 1
        else:
            assert (
                cmd.startswith("mpiexec -n ${pismtasks} apptainer run ${container} pism ") and "asynchronous" not in cmd
            )
        expected = {
            "sync_nc3": "netcdf3",
            "async_nc3": "netcdf3",
            "sync_nc4s": "netcdf4_serial",
            "async_nc4s": "netcdf4_serial",
        }
        assert f"-output.format {expected.get(variant, 'netcdf4_parallel')}" in cmd
        assert ("output.spatial" not in cmd) == (variant == "noout")


def test_ismip7_writer_flags_and_per_variable_files_survive() -> None:
    """The ISMIP7 writer keeps its ``-r 1``, and the ``{var}`` placeholder stays in the spatial path."""
    _, command = t.split_run_script(ISMIP7)
    start = t.start_date(command)
    assert start is not None and start.isoformat() == "1985-01-01"
    cmd = t.variant_command(command, "async_nc4s", "1989-01-01", OUTPUTS)
    assert cmd.startswith("mpiexec -n 1  apptainer run ${container} pism_ismip7_writer -r 1 : -n ${pismtasks}")
    assert "-output.spatial.file /t/spatial_{var}.nc" in cmd and "-output.format netcdf4_serial" in cmd
    sync = t.variant_command(command, "sync", "1989-01-01", OUTPUTS)
    assert sync.startswith("mpiexec -n ${pismtasks} apptainer run ${container} pism ") and "ismip7_writer" not in sync


def test_job_script_pins_node_groups() -> None:
    """A ``+``-joined node group becomes a comma nodelist and an underscore job name."""
    env, command = t.split_run_script(ISMIP7)
    cmd = t.variant_command(command, "async", "1989-01-01", OUTPUTS)
    script = t.job_script(
        env,
        cmd,
        node="n31+n34",
        variant="async",
        test_dir=Path("/t"),
        partition="t2small",
        ntasks=96,
        tasks_per_node=48,
        walltime="03:00:00",
        bench_grid=(1000, 1000),
    )
    assert "#SBATCH --nodelist=n31,n34\n" in script and "#SBATCH --tasks-per-node=48\n" in script
    assert "--job-name=iotest_n31_n34_async" in script and "48 1000 1000" in script
    assert "tee -a /dev/stderr" in script


def test_script_without_a_writer_leg_is_rejected() -> None:
    """A script with only the inversion raises."""
    with pytest.raises(ValueError, match="main leg"):
        t.split_run_script("#!/bin/sh\n" + ENV + PISMI)
