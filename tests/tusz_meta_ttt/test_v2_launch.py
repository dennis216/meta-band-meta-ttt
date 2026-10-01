from __future__ import annotations

import json
import sys

import pytest

from bfa.tusz_meta_ttt_v2.launch import run_monitored


def test_launch_preserves_output_and_saves_resource_evidence(tmp_path):
    log = tmp_path / 'success.log'
    run_monitored([sys.executable, '-c', 'print("finished")'], cwd=tmp_path, log=log)
    assert 'finished' in log.read_text()
    report = json.loads(log.with_suffix('.startup.json').read_text())
    assert report['pid'] > 0
    assert report['returncode_after_startup'] == 0
    assert report['command'][0] == sys.executable


def test_launch_propagates_child_failure(tmp_path):
    log = tmp_path / 'failure.log'
    with pytest.raises(RuntimeError, match='command failed \\(7\\)'):
        run_monitored([sys.executable, '-c', 'raise SystemExit(7)'], cwd=tmp_path, log=log)
    assert json.loads(log.with_suffix('.startup.json').read_text())['returncode_after_startup'] == 7
