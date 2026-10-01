#!/usr/bin/env python3
"""Sample real pipeline GPU, CPU, memory, and storage utilization."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil


def gpu() -> dict[str, float]:
    line = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    ).splitlines()[0]
    values = [float(value.strip()) for value in line.split(",")]
    return dict(zip(("gpu", "gpu_memory", "memory_mib", "power_w", "temperature_c"), values, strict=True))


def workers(pattern: str) -> tuple[int, float, int]:
    rows = []
    for process in psutil.process_iter(("cmdline", "memory_info")):
        try:
            command = " ".join(process.info["cmdline"] or ())
            if pattern in command and "monitor_tusz_resources" not in command:
                rows.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    cpu = 0.0
    rss = 0
    for process in rows:
        try:
            cpu += process.cpu_percent(None)
            rss += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return len(rows), cpu, rss


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p10": float(np.quantile(values, 0.1)),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "maximum": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--pattern", default="tusz_meta_ttt_v2.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    psutil.cpu_percent(None)
    workers(args.pattern)
    disk_start = psutil.disk_io_counters()
    started = time.perf_counter()
    samples = []
    while time.perf_counter() - started < args.seconds:
        sample = gpu()
        count, process_cpu, process_rss = workers(args.pattern)
        sample.update(
            system_cpu=psutil.cpu_percent(None),
            process_count=count,
            process_cpu=process_cpu,
            process_rss_gib=process_rss / 2**30,
            system_memory_percent=psutil.virtual_memory().percent,
        )
        samples.append(sample)
        time.sleep(args.interval)
    elapsed = time.perf_counter() - started
    disk_end = psutil.disk_io_counters()
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "samples": len(samples),
        "process_pattern": args.pattern,
        "gpu_util_percent": summary([row["gpu"] for row in samples]),
        "gpu_memory_util_percent": summary([row["gpu_memory"] for row in samples]),
        "gpu_memory_mib": summary([row["memory_mib"] for row in samples]),
        "gpu_power_w": summary([row["power_w"] for row in samples]),
        "gpu_temperature_c": summary([row["temperature_c"] for row in samples]),
        "system_cpu_percent": summary([row["system_cpu"] for row in samples]),
        "matching_process_cpu_percent": summary([row["process_cpu"] for row in samples]),
        "matching_process_rss_gib": summary([row["process_rss_gib"] for row in samples]),
        "system_memory_percent": summary([row["system_memory_percent"] for row in samples]),
        "matching_process_count": summary([row["process_count"] for row in samples]),
        "disk_read_mib_per_s": (disk_end.read_bytes - disk_start.read_bytes) / 2**20 / elapsed,
        "disk_write_mib_per_s": (disk_end.write_bytes - disk_start.write_bytes) / 2**20 / elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
