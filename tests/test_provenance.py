"""
Tests for the provenance block stamped into generated submission scripts.

Every ``run.py`` writes the generating command line and the git commit of the
running code into the script it renders, so a job script found months later
can be traced back to the code and invocation that produced it.
"""

import subprocess

from pism_terra.workflow import add_provenance, git_provenance, provenance_comment

SLURM_SCRIPT = """#!/bin/sh
#SBATCH --partition=t2standard
#SBATCH --ntasks=240
#SBATCH --output="/logs/pism.%j"

module purge

mpirun -np ${SLURM_NTASKS} pism -y 1
"""

ARGV = ["/import/home/aaschwanden/envs/bin/pism-ismip7-greenland-run-forward", "--ntasks", "240", "config.toml"]


def test_provenance_comment_records_the_command():
    """
    The command block keeps the arguments but not the interpreter path.
    """
    block = provenance_comment(command=ARGV)

    assert "# Command" in block
    assert "# pism-ismip7-greenland-run-forward --ntasks 240 config.toml" in block
    assert "/import/home" not in block
    assert block.endswith("\n")
    assert all(line == "" or line.startswith("#") for line in block.splitlines())


def test_add_provenance_keeps_sbatch_directives_contiguous():
    """
    The block lands below the header, never between two ``#SBATCH`` lines.
    """
    out = add_provenance(SLURM_SCRIPT, command=ARGV)
    lines = out.splitlines()

    directives = [i for i, line in enumerate(lines) if line.startswith("#SBATCH")]
    command_line = next(i for i, line in enumerate(lines) if line.startswith("# pism-ismip7"))
    body = next(i for i, line in enumerate(lines) if line.startswith("module purge"))

    assert lines[0] == "#!/bin/sh"
    assert directives == list(range(1, 1 + len(directives)))
    assert max(directives) < command_line < body
    # Nothing from the original script is lost or reordered.
    assert [line for line in SLURM_SCRIPT.splitlines() if line] == [
        line for line in lines if line and not line.startswith("# ") and line != "# Git" and line != "# Command"
    ]


def test_add_provenance_without_a_scheduler_header():
    """
    A bare script gets the block at the very top.
    """
    out = add_provenance("echo hello\n", command=ARGV)

    assert out.splitlines()[0] == ""
    assert out.strip().startswith("#")
    assert out.endswith("echo hello\n")


def test_add_provenance_leaves_an_empty_script_alone():
    """
    Debug mode renders an empty script, which must stay empty.
    """
    assert add_provenance("", command=ARGV) == ""
    assert add_provenance("   \n", command=ARGV) == "   \n"


def test_git_provenance_outside_a_repository(tmp_path):
    """
    Code installed outside a checkout degrades to no git block.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided directory that is not a git repository.
    """
    assert git_provenance(tmp_path) == ""
    block = provenance_comment(command=ARGV, repo=tmp_path)
    assert "# Git" not in block
    assert "# Command" in block


def test_git_provenance_reports_the_head_commit(tmp_path):
    """
    Inside a checkout, the block names the commit that is checked out.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided directory used for a throwaway repository.
    """
    env = {"GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"), "GIT_CONFIG_SYSTEM": "/dev/null", "PATH": "/usr/bin:/bin"}
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, env=env)
    (tmp_path / "f.txt").write_text("hello\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "f.txt"], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.name=T", "-c", "user.email=t@e", "commit", "-qm", "init"],
        check=True,
        env=env,
    )
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, env=env
    ).stdout.strip()

    block = git_provenance(tmp_path)

    assert block.splitlines()[0].startswith(f"commit {head}")
    assert block.splitlines()[1].startswith("Author: T <t@e>")
    assert block.splitlines()[2].startswith("Date:")
