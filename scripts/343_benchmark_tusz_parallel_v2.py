#!/usr/bin/env python3
"""Measure condition-level GPU concurrency using identical bounded Meta workloads."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from bfa.tusz_meta_ttt.dataset import filter_inventory_records, load_cached_arrays

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"
SOURCE = V1 / "runs/supervised/development/s1_seed3407_check0.25/best.pt"


def gpu_sample() -> dict[str, float] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(command, text=True, timeout=5).splitlines()[0]
        values = [float(value.strip()) for value in line.split(",")]
        return dict(zip(("gpu_util", "memory_util", "memory_mib", "power_w", "temperature_c"), values, strict=True))
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return None


def workload_hours(maximum_records: int) -> tuple[int, float]:
    inventory = json.loads((V1 / "manifests/records.json").read_text())
    split = json.loads((V1 / "manifests/development_split.json").read_text())
    patients = set(split["development_fit"])
    paths = [
        path for path in sorted((V1 / "cache/train").rglob("*.npz"))
        if path.parts[-4] in patients
    ]
    paths = filter_inventory_records(paths, inventory, partition="train")[:maximum_records]
    seconds = 0.0
    for path in paths:
        times = load_cached_arrays(path)["decision_end_s"]
        if len(times):
            seconds += float(times[-1])
    return len(paths), seconds / 3600


def parse_epoch(log: Path) -> dict:
    for line in reversed(log.read_text().splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "outer_weighted_loss" in value:
            return value
    raise RuntimeError(f"no epoch result found in {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective-checkpoint", type=Path, required=True)
    parser.add_argument("--gradient-calibration", type=Path, required=True)
    parser.add_argument("--objective", choices=["band", "mask"], default="band")
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--relative-step", type=float, default=1e-5)
    parser.add_argument("--prefix-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--prefix-microbatch", type=int, default=0)
    parser.add_argument("--prefix-cuda-graph", action="store_true")
    parser.add_argument("--records", type=int, default=48)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--inner-kernel", choices=["reference", "packed"], default="reference")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=OUT / "benchmarks/parallel_scaling.json")
    args = parser.parse_args()
    record_count, eeg_hours = workload_hours(args.records)
    results = []
    for concurrency in args.levels:
        level_root = args.output.parent / (args.output.stem + "_runs") / f"parallel_{concurrency}"
        level_root.mkdir(parents=True, exist_ok=True)
        processes = []
        streams = []
        started = time.perf_counter()
        for index in range(concurrency):
            run = level_root / f"worker_{index}"
            log = level_root / f"worker_{index}.log"
            stream = log.open("w")
            streams.append(stream)
            seed = args.seed
            command = [
                sys.executable,
                str(ROOT / "scripts/320_train_tusz_meta_ttt_v2.py"),
                "--source", str(SOURCE),
                "--objective-checkpoint", str(args.objective_checkpoint.resolve()),
                "--objective", args.objective,
                "--difficulty", str(args.difficulty),
                "--mode", "future",
                "--outer-scope", "ed",
                "--relative-step", str(args.relative_step),
                "--inner-kernel", args.inner_kernel,
                "--prefix-precision", args.prefix_precision,
                "--prefix-microbatch", str(args.prefix_microbatch),
                "--gradient-calibration", str(args.gradient_calibration.resolve()),
                "--epochs", "1",
                "--maximum-records", str(args.records),
                "--seed", str(seed),
                "--run-directory", str(run),
            ]
            if args.prefix_cuda_graph:
                command.append('--prefix-cuda-graph')
            processes.append(subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT))
        samples = []
        while any(process.poll() is None for process in processes):
            sample = gpu_sample()
            if sample is not None:
                samples.append(sample)
            time.sleep(args.sample_interval)
        wall = time.perf_counter() - started
        for stream in streams:
            stream.close()
        failures = [process.returncode for process in processes if process.returncode]
        if failures:
            raise RuntimeError(f"parallel level {concurrency} failed: {failures}")
        epochs = [parse_epoch(level_root / f"worker_{index}.log") for index in range(concurrency)]
        aggregate_updates = sum(int(row["updates"]) for row in epochs)
        aggregate_hours = eeg_hours * concurrency
        results.append({
            "parallel_conditions": concurrency,
            "records_per_condition": record_count,
            "eeg_hours_per_condition": eeg_hours,
            "wall_s": wall,
            "aggregate_inner_updates": aggregate_updates,
            "aggregate_updates_per_s": aggregate_updates / wall,
            "aggregate_eeg_hours_per_wall_hour": aggregate_hours / (wall / 3600),
            "per_condition_elapsed_s": [float(row["elapsed_s"]) for row in epochs],
            "gpu_samples": len(samples),
            "mean_gpu_util_percent": statistics.fmean(row["gpu_util"] for row in samples),
            "p10_gpu_util_percent": float(np.quantile([row["gpu_util"] for row in samples], 0.1)),
            "p90_gpu_util_percent": float(np.quantile([row["gpu_util"] for row in samples], 0.9)),
            "peak_memory_mib": max(row["memory_mib"] for row in samples),
            "mean_power_w": statistics.fmean(row["power_w"] for row in samples),
            "peak_temperature_c": max(row["temperature_c"] for row in samples),
        })
        print(json.dumps(results[-1]), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix('.partial.json').write_text(json.dumps(results, indent=2) + '\n')
    feasible = [row for row in results if row["peak_memory_mib"] <= 28 * 1024]
    selected = max(feasible, key=lambda row: row["aggregate_updates_per_s"])
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "objective": args.objective,
        "inner_kernel": args.inner_kernel,
        "difficulty": args.difficulty,
        "relative_step": args.relative_step,
        "prefix_precision": args.prefix_precision,
        "prefix_microbatch": args.prefix_microbatch,
        "prefix_cuda_graph": args.prefix_cuda_graph,
        "results": results,
        "selected_parallel_conditions": selected["parallel_conditions"],
        "selection_rule": "maximum aggregate updates/s subject to peak GPU memory <= 28 GiB",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "selected": report["selected_parallel_conditions"]}))


if __name__ == "__main__":
    main()
