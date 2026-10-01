#!/usr/bin/env python3
"""Evaluate post-TTT SSL/classification alignment on Train, Dev, and Eval.

The host has one GPU, so this is an explicit single-GPU queue.  It calibrates
thresholds on official Dev, evaluates fixed thresholds on all three partitions,
then samples paired pre/post SSL and classification losses for each partition.
Eval is read only for this diagnostic and never controls model selection.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v1"
EVALUATOR = ROOT / "scripts/304_evaluate_tusz_meta_ttt_v1.py"
GRADIENTS = ROOT / "scripts/308_analyze_tusz_gradients_v1.py"
SOURCE = OUT / "runs/supervised/development/s1_seed3407_check0.25/best.pt"
CONFIGS = {
    "band_lambda0.1": (
        OUT
        / "runs/meta/development/online/band_lr0.0001_seed3407_cosine_lambda0.1_cosine_v1/epoch_02.pt",
        "band_lr0.0001",
    ),
    "band_lambda1": (
        OUT
        / "runs/meta/development/online/band_lr0.0001_seed3407_cosine_lambda1_cosine_v1/epoch_02.pt",
        "band_lr0.0001",
    ),
    "learned_lambda0.1": (
        OUT
        / "runs/meta/development/online/learned_lr1e-05_seed3407_cosine_lambda0.1_cosine_v1/epoch_02.pt",
        "learned_lr1e-05",
    ),
    "learned_lambda1": (
        OUT
        / "runs/meta/development/online/learned_lr1e-05_seed3407_cosine_lambda1_cosine_v1/epoch_02.pt",
        "learned_lr1e-05",
    ),
}


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def summary_path(variant: str, mode: str, partition: str) -> Path:
    return OUT / "evaluation" / variant / mode / "seed3407" / partition / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["online", "online_no_carry", "same_window"], default="online"
    )
    parser.add_argument("--samples-per-group", type=int, default=256)
    parser.add_argument("--max-samples-per-patient-group", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--run-tag", default="eval_ssl_alignment_v1")
    args = parser.parse_args()
    while not SOURCE.is_file() or not all(
        checkpoint.is_file() for checkpoint, _ in CONFIGS.values()
    ):
        time.sleep(args.poll_seconds)
    state_path = OUT / "eval_ssl_alignment_queue_state.json"
    state = {"status": "running", "mode": args.mode, "configs": {}, "source": str(SOURCE)}
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    for name, (checkpoint, variant) in CONFIGS.items():
        state["configs"][name] = {
            "checkpoint": str(checkpoint),
            "variant": variant,
            "status": "running",
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        eval_variant = f"{variant}_{name}_{args.run_tag}"
        dev_log = OUT / "logs" / f"eval_alignment_{name}_dev.log"
        run(
            [
                sys.executable,
                str(EVALUATOR),
                "--source",
                str(SOURCE),
                "--objective-checkpoint",
                str(checkpoint),
                "--partition",
                "dev",
                "--mode",
                args.mode,
                "--calibrate",
                "--seed",
                "3407",
                "--variant-suffix",
                f"{name}_{args.run_tag}",
            ],
            dev_log,
        )
        dev_variant = eval_variant
        dev_summary = json.loads(summary_path(dev_variant, args.mode, "dev").read_text())
        frozen = dev_summary["conditions"]["frozen"]
        adapted = dev_summary["conditions"]["adapted"]
        frozen_threshold = frozen.get("threshold") or frozen.get("maximum_sensitivity_threshold")
        adapted_threshold = adapted.get("threshold") or adapted.get("maximum_sensitivity_threshold")
        if frozen_threshold is None or adapted_threshold is None:
            raise RuntimeError(f"no Dev threshold for {name}")
        state["configs"][name]["dev_thresholds"] = {
            "frozen": frozen_threshold,
            "adapted": adapted_threshold,
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        probability_paths = {}
        for partition in ("train", "dev", "eval"):
            log = OUT / "logs" / f"eval_alignment_{name}_{partition}.log"
            run(
                [
                    sys.executable,
                    str(EVALUATOR),
                    "--source",
                    str(SOURCE),
                    "--objective-checkpoint",
                    str(checkpoint),
                    "--partition",
                    partition,
                    "--mode",
                    args.mode,
                    "--seed",
                    "3407",
                    "--variant-suffix",
                    f"{name}_{args.run_tag}",
                    "--frozen-threshold",
                    str(frozen_threshold),
                    "--adapted-threshold",
                    str(adapted_threshold),
                ],
                log,
            )
            probability_paths[partition] = str(
                OUT
                / "evaluation"
                / dev_variant
                / args.mode
                / "seed3407"
                / partition
                / "probabilities.parquet"
            )
            gradient_log = OUT / "logs" / f"gradient_alignment_{name}_{partition}.log"
            run(
                [
                    sys.executable,
                    str(GRADIENTS),
                    "--source",
                    str(SOURCE),
                    "--objective-checkpoint",
                    str(checkpoint),
                    "--partition",
                    partition,
                    "--threshold",
                    str(frozen_threshold),
                    "--probabilities",
                    probability_paths[partition],
                    "--cohort",
                    "all",
                    "--samples-per-group",
                    str(args.samples_per_group),
                    "--max-samples-per-patient-group",
                    str(args.max_samples_per_patient_group),
                    "--seed",
                    "3407",
                ],
                gradient_log,
            )
        state["configs"][name].update({"status": "complete", "probabilities": probability_paths})
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    state["status"] = "complete"
    state_path.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
