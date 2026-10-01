"""Check syntax and forbidden artifacts in the code-only publication tree."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    excluded = {'.git', '.venv', '__pycache__', '.pytest_cache', '.local-runtime'}
    forbidden = {'.edf', '.csv_bi', '.parquet', '.npy', '.npz', '.pt', '.pth', '.ckpt', '.pkl', '.pem', '.key'}
    issues, count = [], 0
    secret_patterns = [r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{35,}',
                       r'AKIA[A-Z0-9]{16}', r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
    for path in root.rglob('*'):
        rel = path.relative_to(root)
        if any(part in excluded or part.endswith('.egg-info') for part in rel.parts) or not path.is_file():
            continue
        if path.suffix in forbidden or path.name == '.env':
            issues.append(f'forbidden artifact: {rel}')
            continue
        text = path.read_text(encoding='utf-8')
        if any(re.search(pattern, text) for pattern in secret_patterns):
            issues.append(f'credential pattern: {rel}')
        if path.suffix == '.py':
            try:
                ast.parse(text, filename=str(rel))
                count += 1
            except SyntaxError as error:
                issues.append(f'syntax: {rel}:{error.lineno}')
    print(json.dumps({'python_files_parsed': count, 'issues': issues}, indent=2))
    raise SystemExit(bool(issues))


if __name__ == '__main__':
    main()
