import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('check_release', Path(__file__).resolve().parents[1] / 'tools/check_release.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_reports_environment_variants_and_binary_without_crashing(tmp_path):
    (tmp_path / '.env.production').write_text('EXAMPLE=1')
    (tmp_path / '.env.example').write_text('EXAMPLE=')
    (tmp_path / 'unexpected.bin').write_bytes(b'\xff\xfe')
    (tmp_path / 'valid.py').write_text('value = 1\n')
    ignored = tmp_path / '.venv'
    ignored.mkdir()
    (ignored / 'bad.py').write_text('invalid python !')
    report = module.check(tmp_path)
    assert report['python_files_parsed'] == 1
    assert sorted(report['issues']) == ['forbidden artifact: .env.production', 'unreviewed binary: unexpected.bin']
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs/source-manifest.json').write_text('{"src/bfa/data/scan.py": {}}')
    assert 'missing source-manifest file: src/bfa/data/scan.py' in module.check(tmp_path)['issues']


def test_does_not_follow_external_symlinks(tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'sensitive.txt').write_text('never read')
    root = tmp_path / 'publication'
    root.mkdir()
    (root / 'linked').symlink_to(outside, target_is_directory=True)
    assert module.check(root)['issues'] == ['symlink: linked']
