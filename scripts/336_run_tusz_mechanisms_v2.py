#!/usr/bin/env python3
"""Run the three preregistered mechanism controls for each F/C winner."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bfa.tusz_meta_ttt_v2.launch import run_monitored

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"
SOURCE = ROOT / "outputs/reports/tusz_meta_ttt_v1/runs/supervised/development/s1_seed3407_check0.25/best.pt"


def call(command: list[str], log: Path) -> None:
    run_monitored(command, cwd=ROOT, log=log)


def run_name(row: dict, suffix: str = "") -> str:
    base = (
        f'{row["objective"]}_{row["difficulty"]:g}_eds_'
        f'rho{row["relative_step"]:g}_seed3407'
    )
    return base + suffix


def mechanism_suffix(item: dict) -> str:
    if item["kind"] == "fixed_sgd":
        return f'_sgd{float(item["extra"][-1]):g}'
    return {
        "detector_open": "_detector_open",
        "stop_encoder_through_inner": "_stop_encoder_through_inner",
    }[item["kind"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--parallel-conditions", type=int, default=4)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    planned = []
    for mode in ("future", "current"):
        winner = selection["modes"][mode]["first"]
        calibration = json.loads(Path(winner["gradient_calibration"]).read_text())
        fixed_lr = statistics.median(
            winner["relative_step"]
            * float(calibration["block_reference_norms"][str(block)])
            / float(calibration["block_gradient_medians"][str(block)])
            for block in (10, 11)
        )
        planned.extend([
            {"mode": mode, "kind": "detector_open", "row": winner, "extra": [
                "--detector-freeze-fraction", "0",
            ]},
            {"mode": mode, "kind": "fixed_sgd", "row": winner, "extra": [
                "--inner-rule", "sgd", "--fixed-inner-lr", str(fixed_lr),
            ]},
            {"mode": mode, "kind": "stop_encoder_through_inner", "row": winner, "extra": [
                "--stop-encoder-through-inner",
            ]},
        ])
    state_path = OUT / "queues/mechanisms_v2.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"runs": planned}
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    if not args.execute:
        print(json.dumps({"queue": str(state_path), "runs": len(planned)}))
        return
    state_lock = threading.Lock()

    def persist() -> None:
        with state_lock:
            state_path.write_text(json.dumps(state, indent=2) + "\n")

    def execute_item(item: dict) -> None:
        row = item["row"]
        mode = item["mode"]
        suffix = mechanism_suffix(item)
        name = run_name(row, suffix)
        run = OUT / "runs/meta/development" / mode / name
        epochs = int(row["best_epoch"])
        checkpoint = run / f"epoch_{epochs:02d}.pt"
        if not checkpoint.is_file():
            command = [
                sys.executable, str(ROOT / "scripts/320_train_tusz_meta_ttt_v2.py"),
                "--source", str(SOURCE),
                "--objective-checkpoint", row["objective_checkpoint"],
                "--objective", row["objective"],
                "--difficulty", str(row["difficulty"]),
                "--mode", mode,
                "--outer-scope", "eds",
                "--relative-step", str(row["relative_step"]),
                "--gradient-calibration", row["gradient_calibration"],
                "--epochs", str(epochs),
                *item["extra"],
            ]
            if (run / "last.pt").is_file():
                command.extend(("--resume", str(run / "last.pt")))
            call(command, OUT / "logs" / f"mechanism_{mode}_{name}.log")
        evaluation = OUT / "evaluation" / mode / f"{name}_final/development_validation/summary.json"
        if not evaluation.is_file():
            call(
                [
                    sys.executable, str(ROOT / "scripts/323_evaluate_tusz_meta_ttt_v2.py"),
                    "--meta-checkpoint", str(checkpoint),
                    "--partition", "train", "--cohort", "development_validation",
                    "--calibrate", "--tag", "final",
                ],
                OUT / "logs" / f"mechanism_eval_{mode}_{name}.log",
            )
        item.update(status="complete", checkpoint=str(checkpoint), evaluation=str(evaluation))
        persist()

    with ThreadPoolExecutor(max_workers=max(1, args.parallel_conditions)) as executor:
        futures = [executor.submit(execute_item, item) for item in planned]
        for future in as_completed(futures):
            future.result()

    gradient_jobs = []
    for mode in ("future", "current"):
        row = selection["modes"][mode]["first"]
        name = run_name(row)
        checkpoint = OUT / "runs/meta/development" / mode / name / f'epoch_{row["best_epoch"]:02d}.pt'
        evaluation_root = OUT / "evaluation" / mode / f'{name}_epoch{row["best_epoch"]:02d}'
        validation = evaluation_root / "development_validation"
        summary = json.loads((validation / "summary.json").read_text())
        if "source_frozen" not in summary.get("conditions", {}):
            call(
                [
                    sys.executable,
                    str(ROOT / "scripts/323_evaluate_tusz_meta_ttt_v2.py"),
                    "--meta-checkpoint", str(checkpoint),
                    "--partition", "train",
                    "--cohort", "development_validation",
                    "--calibrate",
                    "--tag", f'epoch{row["best_epoch"]:02d}',
                ],
                OUT / "logs" / f"full_best_evaluation_{mode}_{name}.log",
            )
            summary = json.loads((validation / "summary.json").read_text())
        threshold = summary["conditions"]["source_frozen"]["threshold"]
        if threshold is None:
            threshold = summary["conditions"]["source_frozen"]["maximum_sensitivity_threshold"]
        for cohort in ("development_fit", "development_validation"):
            gradient_jobs.append((mode, row, name, checkpoint, validation, threshold, cohort))

    def execute_gradient_job(job: tuple) -> None:
        mode, row, name, checkpoint, validation, threshold, cohort = job
        evaluation_root = OUT / "evaluation" / mode / f'{name}_epoch{row["best_epoch"]:02d}'
        evaluation = evaluation_root / cohort
        if not (evaluation / "probabilities.parquet").is_file():
            call(
                [
                    sys.executable, str(ROOT / "scripts/323_evaluate_tusz_meta_ttt_v2.py"),
                    "--meta-checkpoint", str(checkpoint),
                    "--partition", "train", "--cohort", cohort,
                    "--thresholds", str(validation / "summary.json"),
                    "--tag", f'epoch{row["best_epoch"]:02d}',
                ],
                OUT / "logs" / f"gradient_scores_{cohort}_{mode}_{name}.log",
            )
        call(
            [
                sys.executable, str(ROOT / "scripts/328_analyze_tusz_gradients_v2.py"),
                "--meta-checkpoint", str(checkpoint),
                "--probabilities", str(evaluation / "probabilities.parquet"),
                "--source-threshold", str(threshold),
                "--partition", "train", "--cohort", cohort,
            ],
            OUT / "logs" / f"gradient_{cohort}_{mode}_{name}.log",
        )

    with ThreadPoolExecutor(max_workers=max(1, args.parallel_conditions)) as executor:
        futures = [executor.submit(execute_gradient_job, job) for job in gradient_jobs]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
