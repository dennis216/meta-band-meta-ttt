import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('prepare_runtime', Path(__file__).resolve().parents[1] / 'tools/prepare_runtime.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_relocation_preserves_original_and_refuses_overwrite(tmp_path):
    source = tmp_path / 'source'
    (source / 'scripts').mkdir(parents=True)
    script = source / 'scripts/example.py'
    original = 'ROOT = "/root/b_false_alarm_atlas"\nDATA = "/mnt/d/TUH_EEG/TUSZ_v2.0.6"\n'
    script.write_text(original)
    destination = tmp_path / 'runtime'
    module.prepare(source, destination, '/data/tusz', '/data/chb', '/data/cache')
    assert script.read_text() == original
    assert '/data/tusz' in (destination / 'scripts/example.py').read_text()
    assert str(destination) in (destination / 'scripts/example.py').read_text()
    with pytest.raises(ValueError, match='must not exist'):
        module.prepare(source, destination, '/data/tusz', '/data/chb', '/data/cache')


def test_copy_preserves_data_packages_but_excludes_assets_and_symlinks(tmp_path):
    source = tmp_path / 'source'
    retained = ['src/bfa/data/__init__.py', 'tests/data/test_scan.py',
                'third_party/CBraMod/pretrained_weights/README.md', '.env.example']
    omitted = ['data/private.txt', 'cache/private.txt', 'outputs/result.txt',
               'third_party/CBraMod/pretrained_weights/model.pth', '.env.production',
               'src/bfa/data/__pycache__/data.pyc']
    for name in retained + omitted:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture\n')
    (source / 'linked-data').symlink_to(source / 'data', target_is_directory=True)
    destination = tmp_path / 'runtime'
    module.prepare(source, destination, '/data/tusz', '/data/chb', '/data/cache')
    assert all((destination / name).is_file() for name in retained)
    assert all(not (destination / name).exists() for name in omitted)
    assert not (destination / 'linked-data').exists()
