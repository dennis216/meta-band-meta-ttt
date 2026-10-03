"""Create a separate Linux runtime copy with historical paths relocated."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def runtime_ignore(source: Path):
    """Ignore generated assets only at root; retain packages named data/cache."""
    root_only = {'outputs', 'runs', 'logs', 'cache', 'data', 'manifests', 'weights',
                 'build', 'dist', '.local-runtime'}
    everywhere = {'.git', '.venv', '__pycache__', '.pytest_cache', '.ruff_cache'}

    def ignore(directory: str, names: list[str]) -> set[str]:
        current = Path(directory)
        skipped = {name for name in names if name in everywhere or name.endswith('.egg-info')}
        if current == source:
            skipped.update(root_only.intersection(names))
        # Never dereference local asset links into the shared code copy.
        skipped.update(name for name in names if (current / name).is_symlink())
        skipped.update(name for name in names if Path(name).suffix.lower() in
                       {'.edf', '.pt', '.pth', '.ckpt', '.npy', '.npz', '.parquet', '.pkl', '.pem', '.key'}
                       or name == '.env' or name.startswith('.env.') and name != '.env.example')
        return skipped

    return ignore


def prepare(source: Path, destination: Path, tusz: str, chb: str, cache: str) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise ValueError('destination must not exist; existing experiments are never overwritten')
    if destination == source or source in destination.parents:
        raise ValueError('destination must be outside the publication checkout')
    for value in (str(destination), tusz, chb, cache):
        if not value.startswith('/') or any(c in value for c in '\n\r\t\"\'`$\\ '):
            raise ValueError('use absolute Linux paths without spaces or shell metacharacters')
    shutil.copytree(source, destination, ignore=runtime_ignore(source))
    replacements = {
        '/mnt/c/Users/User/Documents/ChatGPT/EEG_ZiquanBaoBao/metaTTT_migration_20260905/project/scripts': str(destination / 'scripts'),
        '/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/NeuroTTT/CBraMod': str(destination / 'external/NeuroTTT_CBraMod'),
        '/root/b_false_alarm_atlas': str(destination),
        '/mnt/d/TUH_EEG/TUSZ_v2.0.6': tusz,
        '/mnt/d/EEGData/bfa_cache_v3_official_noclip/cbramod': cache,
        '/mnt/d/EEGData/chbmit-1.0.0': chb,
        '/mnt/d/EEGData/meta_ttt_prefix_v2': str(destination / 'cache/meta_ttt_prefix_v2'),
    }
    changed = []
    for path in destination.rglob('*'):
        relative = path.relative_to(destination)
        if not path.is_file() or relative.parts[0] in ('archive', 'docs', 'tools'):
            continue
        if path.suffix not in ('.py', '.sh', '.yaml', '.yml', '.toml'):
            continue
        before = path.read_text(encoding='utf-8')
        after = before
        for old, new in replacements.items():
            after = after.replace(old, new)
        if before != after:
            path.write_text(after, encoding='utf-8')
            changed.append(relative.as_posix())
    report = {'replacements': replacements, 'changed_files': changed}
    (destination / 'runtime-paths.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--tusz-root', required=True)
    parser.add_argument('--chb-root', required=True)
    parser.add_argument('--chb-cache', required=True)
    args = parser.parse_args()
    report = prepare(Path(__file__).resolve().parents[1], args.destination,
                     args.tusz_root, args.chb_root, args.chb_cache)
    print(json.dumps({'destination': str(args.destination), 'changed_files': len(report['changed_files'])}))


if __name__ == '__main__':
    main()
