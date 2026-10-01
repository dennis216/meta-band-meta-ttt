#!/usr/bin/env python3
"""Single resumable entry point for all v2 stages and final report compilation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"


def run(command: list[str], name: str) -> None:
    log = OUT / "logs/pipeline" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        result = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        raise RuntimeError(f"pipeline stage {name} failed; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", choices=[17, 42, 3407], default=[3407])
    args = parser.parse_args()
    full_confirmation = set(args.seeds) == {17, 42, 3407}
    formal_stage = "formal_three_seed" if full_confirmation else "formal_active_seeds"
    ssl_selection = OUT / "selection/ssl_difficulty_seed3407.json"
    development_queue = OUT / "queues/development_v2.json"
    meta_selection = OUT / "selection/meta_development_v2.json"
    stages = [
        "asset_audit",
        "ssl_stage_b",
        "throughput_benchmark",
        "development_24_and_controls",
        "freeze_meta_selection",
        "mechanisms",
        formal_stage,
        "compile_reports",
        "completion_audit",
    ]
    if not full_confirmation:
        stages.remove("completion_audit")
    state_path = OUT / "queues/pipeline_v2.json"
    state = {"created_utc": datetime.now(UTC).isoformat(), "stages": stages, "status": "planned",
             "active_seeds": args.seeds, "deferred_seeds": sorted({17,42,3407}-set(args.seeds))}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    if not args.execute:
        print(json.dumps({"pipeline": str(state_path), "stages": stages}))
        return
    sources = [
        V1 / "runs/supervised/development/s1_seed3407_check0.25/best.pt",
        *(V1 / f"runs/supervised/formal/s1_seed{seed}_check0.25/best.pt" for seed in (17, 42, 3407)),
    ]
    commands = [
        (
            "asset_audit",
            [sys.executable, str(ROOT / "scripts/333_audit_tusz_assets_v2.py"),
             "--sources", *map(str, sources)],
        ),
        (
            "ssl_stage_b",
            [sys.executable, str(ROOT / "scripts/331_run_tusz_ssl_stage_b_v2.py"), "--execute"],
        ),
        (
            "throughput_benchmark",
            [
                sys.executable,
                str(ROOT / "scripts/343_benchmark_tusz_parallel_v2.py"),
                "--objective-checkpoint",
                str(OUT / "runs/ssl/development/band_0.5_seed3407/epoch_02.pt"),
                "--gradient-calibration",
                str(OUT / "calibration/band_0.5_seed3407/gradient_calibration.json"),
                "--objective", "band", "--difficulty", "0.5",
                "--relative-step", "1e-5", "--records", "48",
                "--levels", "1", "2", "3", "4", "5", "6",
            ],
        ),
        (
            "development_24_and_controls",
            [sys.executable, str(ROOT / "scripts/327_run_tusz_meta_development_v2.py"),
             "--selection", str(ssl_selection), "--parallel-conditions", "4", "--execute"],
        ),
        (
            "freeze_meta_selection",
            [sys.executable, str(ROOT / "scripts/334_select_tusz_meta_v2.py"),
             "--queue", str(development_queue), "--output", str(meta_selection)],
        ),
        (
            "mechanisms",
            [sys.executable, str(ROOT / "scripts/336_run_tusz_mechanisms_v2.py"),
             "--selection", str(meta_selection), "--parallel-conditions", "4", "--execute"],
        ),
        (
            formal_stage,
            [sys.executable, str(ROOT / "scripts/344_run_tusz_formal_parallel_v2.py"),
             "--selection", str(meta_selection), "--development-queue", str(development_queue),
             "--execute", "--seeds", *map(str, args.seeds)],
        ),
        (
            "compile_reports",
            [sys.executable, str(ROOT / "scripts/339_compile_tusz_meta_ttt_v2.py")],
        ),
        (
            "completion_audit",
            [
                sys.executable,
                str(ROOT / "scripts/342_verify_tusz_meta_ttt_v2_completion.py"),
            ],
        ),
    ]
    state["status"] = "running"
    state["completed_stages"] = []
    for name, command in commands:
        if name not in stages:
            continue
        if name == "throughput_benchmark" and (
            OUT / "benchmarks/parallel_scaling.json"
        ).is_file():
            state["completed_stages"].append(name)
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            continue
        run(command, name)
        state["completed_stages"].append(name)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    state["status"] = "complete" if full_confirmation else "active_seed_stages_finished_confirmation_deferred"
    state["completed_utc"] = datetime.now(UTC).isoformat()
    state_path.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
