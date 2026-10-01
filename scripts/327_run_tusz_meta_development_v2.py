#!/usr/bin/env python3
"""Build or execute the 24-condition v2 development queue."""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from bfa.tusz_meta_ttt_v2.launch import run_monitored

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"
SOURCE = ROOT / "outputs/reports/tusz_meta_ttt_v1/runs/supervised/development/s1_seed3407_check0.25/best.pt"


def rank(summary: dict) -> tuple:
    adapted = summary["conditions"]["adapted"]
    frozen = summary["conditions"]["meta_frozen"]
    if adapted["reachable"] and frozen["reachable"]:
        metrics = adapted["metrics"]
        baseline = frozen["metrics"]
        paired_delay = summary.get("paired_delay", {}).get(
            "adapted_vs_meta_frozen", {}
        ).get("median_delay_difference_s")
        promoted = (
            metrics["sensitivity"] - baseline["sensitivity"] >= -0.02
            and metrics["false_alarms_per_hour"]
            <= 0.90 * baseline["false_alarms_per_hour"]
            and metrics["false_alarm_minutes_per_hour"]
            <= baseline["false_alarm_minutes_per_hour"]
            and paired_delay is not None
            and paired_delay <= 2.0
        )
        ratio = (
            metrics["false_alarms_per_hour"] / baseline["false_alarms_per_hour"]
            if baseline["false_alarms_per_hour"] > 0
            else metrics["false_alarms_per_hour"] - baseline["false_alarms_per_hour"]
        )
        return (
            0 if promoted else 1,
            ratio,
            metrics["false_alarms_per_hour"],
            metrics["false_alarm_minutes_per_hour"],
        )
    metrics = adapted["maximum_sensitivity_metrics"]
    return (
        2,
        -metrics["sensitivity"],
        metrics["false_alarms_per_hour"],
        metrics["false_alarm_minutes_per_hour"],
    )


def run_command(command: list[str], log: Path) -> None:
    run_monitored(command, cwd=ROOT, log=log)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--minimum-epochs", type=int, default=2)
    parser.add_argument("--maximum-epochs", type=int, default=2)
    parser.add_argument("--parallel-conditions", type=int, default=4)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())["selected"]
    inventory = json.loads((ROOT / "outputs/reports/tusz_meta_ttt_v1/manifests/records.json").read_text())
    split = json.loads((ROOT / "outputs/reports/tusz_meta_ttt_v1/manifests/development_split.json").read_text())
    fit_patients = set(split["development_fit"])
    expected_patients = len(fit_patients)
    expected_records = sum(
        row["partition"] == "train"
        and row["patient_id"] in fit_patients
        and row["exclusion"] is None
        for row in inventory
    )
    queue = []
    for objective in ("band", "temporal", "mask"):
        chosen = selection.get(objective)
        if chosen is None:
            queue.append({"objective": objective, "status": "health_check_failed"})
            continue
        difficulty = float(chosen["difficulty"])
        ssl_run = Path(chosen["run"])
        objective_checkpoint = max(ssl_run.glob("epoch_*.pt"))
        calibration = OUT / "calibration" / f"{objective}_{difficulty:g}_seed3407/gradient_calibration.json"
        calibration_state = json.loads(calibration.read_text()) if calibration.is_file() else {}
        relative_step = calibration_state.get("selected_relative_step")
        for mode in ("future", "current"):
            for scope in ("e", "ed", "es", "eds"):
                queue.append({
                    "objective": objective,
                    "difficulty": difficulty,
                    "mode": mode,
                    "outer_scope": scope,
                    "relative_step": relative_step,
                    "objective_checkpoint": str(objective_checkpoint),
                    "gradient_calibration": str(calibration),
                    "status": "ready" if relative_step is not None else "inner_health_check_failed",
                })
    queue_path = OUT / "queues/development_v2.json"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"created_utc": datetime.now(UTC).isoformat(), "runs": queue}
    queue_path.write_text(json.dumps(state, indent=2) + "\n")
    if not args.execute:
        print(json.dumps({"queue": str(queue_path), "ready": sum(row["status"] == "ready" for row in queue)}))
        return

    state_lock = threading.Lock()

    def persist() -> None:
        with state_lock:
            queue_path.write_text(json.dumps(state, indent=2) + "\n")

    def execute_meta_row(row: dict) -> None:
        if row["status"] != "ready":
            return
        run_name = (
            f'{row["objective"]}_{row["difficulty"]:g}_{row["outer_scope"]}'
            f'_rho{row["relative_step"]:g}_seed3407'
        )
        run_dir = OUT / "runs/meta/development" / row["mode"] / run_name
        best_rank = None
        stale_epochs = 0
        for epoch in range(1, args.maximum_epochs + 1):
            checkpoint = run_dir / f"epoch_{epoch:02d}.pt"
            if checkpoint.is_file():
                history_path = run_dir / "history.json"
                history = json.loads(history_path.read_text()) if history_path.is_file() else []
                coverage = next((item for item in history if item.get("epoch") == epoch), None)
                if not coverage or (
                    coverage.get("patients") != expected_patients
                    or coverage.get("records") != expected_records
                ):
                    raise RuntimeError(
                        f"formal-coverage check failed for {checkpoint}: "
                        f"expected {expected_patients} patients/{expected_records} records, "
                        f"got {coverage}"
                    )
            if not checkpoint.is_file():
                command = [
                    sys.executable,
                    str(ROOT / "scripts/320_train_tusz_meta_ttt_v2.py"),
                    "--source", str(SOURCE),
                    "--objective-checkpoint", row["objective_checkpoint"],
                    "--objective", row["objective"],
                    "--difficulty", str(row["difficulty"]),
                    "--mode", row["mode"],
                    "--outer-scope", row["outer_scope"],
                    "--relative-step", str(row["relative_step"]),
                    "--gradient-calibration", row["gradient_calibration"],
                    "--epochs", str(epoch),
                ]
                if epoch > 1:
                    command.extend(("--resume", str(run_dir / "last.pt")))
                run_command(command, OUT / "logs" / f"meta_{row['mode']}_{run_name}.log")
            row.update(status="running", completed_epochs=epoch)
            persist()
            # A full online replay is more expensive than a training traversal.
            # The protocol forbids selection before two traversals, so evaluating
            # epoch 1 cannot affect a valid decision and only occupies a GPU slot.
            if epoch < args.minimum_epochs:
                continue
            evaluation = (
                OUT / "evaluation" / row["mode"] / f"{run_name}_epoch{epoch:02d}"
                / "development_validation/summary.json"
            )
            if not evaluation.is_file():
                run_command(
                    [
                        sys.executable,
                        str(ROOT / "scripts/323_evaluate_tusz_meta_ttt_v2.py"),
                        "--meta-checkpoint", str(checkpoint),
                        "--partition", "train",
                        "--cohort", "development_validation",
                        "--conditions", "meta_frozen", "adapted",
                        "--calibrate",
                        "--tag", f"epoch{epoch:02d}",
                    ],
                    OUT / "logs" / f"evaluate_{row['mode']}_{run_name}_epoch{epoch:02d}.log",
                )
            current_rank = rank(json.loads(evaluation.read_text()))
            if best_rank is None or current_rank < best_rank:
                best_rank = current_rank
                stale_epochs = 0
                row.update(best_epoch=epoch, best_rank=current_rank)
            else:
                stale_epochs += 1
            if epoch >= args.minimum_epochs and stale_epochs >= 2:
                break
        row["status"] = "complete"
        persist()

    ready_rows = [row for row in queue if row["status"] == "ready"]
    with ThreadPoolExecutor(max_workers=max(1, args.parallel_conditions)) as executor:
        futures = {executor.submit(execute_meta_row, row): row for row in ready_rows}
        for future in as_completed(futures):
            future.result()

    for scope in ("e", "ed"):
        control_run = OUT / "runs/supervised_controls/development" / f"{scope}_seed3407"
        control_checkpoint = control_run / "epoch_02.pt"
        if not control_checkpoint.is_file():
            command = [
                sys.executable,
                str(ROOT / "scripts/326_train_tusz_supervised_control_v2.py"),
                "--source", str(SOURCE), "--scope", scope, "--epochs", "2",
            ]
            if (control_run / "last.pt").is_file():
                command.extend(("--resume", str(control_run / "last.pt")))
            run_command(
                command,
                OUT / "logs" / f"supervised_control_{scope}.log",
            )
        destination = OUT / "evaluation/detectors" / f"development_{scope}_seed3407" / "development_validation/summary.json"
        if not destination.is_file():
            run_command(
                [
                    sys.executable,
                    str(ROOT / "scripts/335_evaluate_tusz_detector_v2.py"),
                    "--checkpoint", str(control_checkpoint), "--partition", "train",
                    "--cohort", "development_validation", "--calibrate",
                    "--tag", f"development_{scope}_seed3407",
                ],
                OUT / "logs" / f"supervised_control_eval_{scope}.log",
            )

    for row in queue:
        if row.get("status") != "complete" or row["outer_scope"] != "eds":
            continue
        nonmeta = (
            OUT / "runs/nonmeta/development" / row["mode"]
            / f'{row["objective"]}_{row["difficulty"]:g}_seed3407' / "checkpoint.pt"
        )
        if not nonmeta.is_file():
            run_command(
                [
                    sys.executable,
                    str(ROOT / "scripts/329_package_tusz_nonmeta_ttt_v2.py"),
                    "--source", str(SOURCE),
                    "--objective-checkpoint", row["objective_checkpoint"],
                    "--gradient-calibration", row["gradient_calibration"],
                    "--mode", row["mode"], "--output", str(nonmeta),
                ],
                OUT / "logs" / f"nonmeta_package_{row['mode']}_{row['objective']}.log",
            )
        nonmeta_eval = (
            OUT / "evaluation" / row["mode"] / nonmeta.parent.name
            / "development_validation/summary.json"
        )
        if not nonmeta_eval.is_file():
            run_command(
                [
                    sys.executable,
                    str(ROOT / "scripts/323_evaluate_tusz_meta_ttt_v2.py"),
                    "--meta-checkpoint", str(nonmeta), "--partition", "train",
                    "--cohort", "development_validation", "--calibrate",
                ],
                OUT / "logs" / f"nonmeta_eval_{row['mode']}_{row['objective']}.log",
            )
    state["completed_utc"] = datetime.now(UTC).isoformat()
    queue_path.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
