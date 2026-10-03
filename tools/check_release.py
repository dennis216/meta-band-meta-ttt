"""Check syntax and forbidden artifacts in the code-only publication tree."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re


def check(root: Path) -> dict:
    excluded = {'.git', '.venv', '__pycache__', '.pytest_cache', '.ruff_cache', '.local-runtime'}
    forbidden = {'.edf', '.csv_bi', '.parquet', '.npy', '.npz', '.pt', '.pth', '.ckpt',
                 '.pkl', '.pickle', '.pem', '.key', '.zip', '.tar', '.gz', '.log'}
    issues, count = [], 0
    secret_patterns = [r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{35,}',
                       r'AKIA[A-Z0-9]{16}', r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
    manifest = root / 'docs/source-manifest.json'
    if manifest.exists():
        try:
            entries = json.loads(manifest.read_text(encoding='utf-8'))
            if not isinstance(entries, dict):
                raise ValueError('expected a path mapping')
            for name in entries:
                if not (root / name).is_file():
                    issues.append(f'missing source-manifest file: {name}')
        except (ValueError, UnicodeDecodeError) as error:
            issues.append(f'invalid source manifest: {error}')
    for directory, dirs, files in os.walk(root, followlinks=False):
        # Prune before descending: do not traverse .venv or Git object databases.
        dirs[:] = [name for name in dirs if name not in excluded and not name.endswith('.egg-info')]
        for name in list(dirs):
            path = Path(directory) / name
            if path.is_symlink():
                issues.append(f'symlink: {path.relative_to(root)}')
                dirs.remove(name)
        for name in files:
            path = Path(directory) / name
            rel = path.relative_to(root)
            if path.is_symlink():
                issues.append(f'symlink: {rel}')
                continue
            if path.suffix.lower() in forbidden or (name == '.env' or name.startswith('.env.') and name != '.env.example'):
                issues.append(f'forbidden artifact: {rel}')
                continue
            try:
                text = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                issues.append(f'unreviewed binary: {rel}')
                continue
            if any(re.search(pattern, text) for pattern in secret_patterns):
                issues.append(f'credential pattern: {rel}')
            if path.suffix == '.py':
                try:
                    ast.parse(text, filename=str(rel))
                    count += 1
                except SyntaxError as error:
                    issues.append(f'syntax: {rel}:{error.lineno}')
    return {'python_files_parsed': count, 'issues': issues}


def main() -> None:
    report = check(Path(__file__).resolve().parents[1])
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report['issues']))


if __name__ == '__main__':
    main()
