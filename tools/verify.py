"""Run code-only release checks and CPU-compatible tests without datasets."""
from pathlib import Path
import os
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    env['CUDA_VISIBLE_DEVICES'] = ''
    commands = [
        [sys.executable, 'tools/check_release.py'],
        [sys.executable, '-m', 'pytest', '-q', 'tests/tusz_meta_ttt', 'tests/data',
         'tests/evaluation', 'tests/preprocessing', 'tests/test_contracts.py',
         'tests/test_publication_paths.py', 'tests/test_publication_release.py'],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=root, env=env, check=False)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
