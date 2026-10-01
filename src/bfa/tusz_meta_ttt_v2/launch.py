"""Launch diagnostics collected while the workload runs, before a long wait."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def run_monitored(command: list[str], *, cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    started = time.monotonic()
    with log.open('a') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
        # Work proceeds throughout the observation period. GPU totals include
        # other conditions; they must never be attributed to this one process.
        while process.poll() is None and time.monotonic() - started < 30:
            try:
                gpu = subprocess.check_output([
                    'nvidia-smi',
                    '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,power.limit',
                    '--format=csv,noheader,nounits',
                ], text=True, timeout=3).strip()
                samples.append({'elapsed_s': time.monotonic()-started, 'gpu_totals_csv': gpu})
            except (subprocess.SubprocessError, OSError) as error:
                samples.append({'elapsed_s': time.monotonic()-started, 'error': str(error)})
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        report = {
            'created_utc': datetime.now(UTC).isoformat(),
            'pid': process.pid, 'command': command,
            'alive_after_startup': process.poll() is None,
            'returncode_after_startup': process.poll(),
            'samples': samples,
            'scope': 'whole GPU, not per-process; startup sample is not a throughput benchmark',
        }
        destination=log.with_suffix('.startup.json')
        temporary=destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(destination)
        result=process.wait()
    if result:
        raise RuntimeError(f'command failed ({result}); see {log}')
