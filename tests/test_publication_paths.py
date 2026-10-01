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
