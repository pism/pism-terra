import subprocess
import sys
from pathlib import Path


def execute(work_dir: Path = Path.cwd()):
    for run_script in work_dir.glob('**/run_scripts/*.sh'):
        print("Executing script: ", run_script)
        subprocess.run(
            f'bash -ex {run_script.resolve()}',
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=True,
            check=True,
        )
