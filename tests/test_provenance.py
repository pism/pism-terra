"""
Tests for the provenance block stamped into generated submission scripts.

Every ``run.py`` writes the generating command line and the git commit of the
running code into the script it renders, so a job script found months later
can be traced back to the code and invocation that produced it.
"""

import subprocess

from pism_terra.workflow import (
    add_provenance,
    git_provenance,
    git_tree_state,
    provenance_comment,
)

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


def make_repo(path):
    """
    Create a throwaway repository with one committed and one ignored file.

    Parameters
    ----------
    path : pathlib.Path
        Directory to initialise as a repository.

    Returns
    -------
    str
        The full SHA of the single commit.
    """
    env = {"GIT_CONFIG_GLOBAL": str(path / "gitconfig"), "GIT_CONFIG_SYSTEM": "/dev/null", "PATH": "/usr/bin:/bin"}
    subprocess.run(["git", "init", "-q", str(path)], check=True, env=env)
    (path / "f.txt").write_text("hello\n")
    subprocess.run(["git", "-C", str(path), "add", "f.txt"], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=T", "-c", "user.email=t@e", "commit", "-qm", "init"],
        check=True,
        env=env,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, env=env
    ).stdout.strip()


def test_git_provenance_reports_the_head_commit(tmp_path):
    """
    Inside a clean checkout, the block names the commit and says so.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided directory used for a throwaway repository.
    """
    head = make_repo(tmp_path)

    block = git_provenance(tmp_path)

    assert block.splitlines()[0].startswith(f"commit {head}")
    assert block.splitlines()[1].startswith("Author: T <t@e>")
    assert block.splitlines()[2].startswith("Date:")
    assert block.splitlines()[3] == "Tree:   clean"


def test_git_tree_state_flags_modified_tracked_files(tmp_path):
    """
    Editing a tracked file marks the tree dirty and counts the edits.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided directory used for a throwaway repository.
    """
    make_repo(tmp_path)
    assert git_tree_state(tmp_path) == "clean"

    (tmp_path / "f.txt").write_text("edited\n")

    assert git_tree_state(tmp_path) == "dirty (1 tracked file modified)"
    assert "Tree:   dirty (1 tracked file modified)" in git_provenance(tmp_path)
    assert "# Tree:   dirty (1 tracked file modified)" in provenance_comment(command=ARGV, repo=tmp_path)


def test_git_tree_state_ignores_untracked_files(tmp_path):
    """
    Model output sitting in the working directory does not count as dirty.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided directory used for a throwaway repository.
    """
    make_repo(tmp_path)
    (tmp_path / "2026_08_run").mkdir()
    (tmp_path / "2026_08_run" / "output.nc").write_text("not real netcdf\n")

    assert git_tree_state(tmp_path) == "clean"


def test_git_tree_state_unknown_is_not_reported_as_clean(tmp_path):
    """
    Outside a repository the state is empty rather than ``"clean"``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided directory that is not a git repository.
    """
    assert git_tree_state(tmp_path) == ""
