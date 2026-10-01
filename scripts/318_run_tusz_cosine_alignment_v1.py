#!/usr/bin/env python3
"""Run the pre-registered TUSZ cosine-alignment Meta ablation sequentially.

The queue is intentionally single-process because the host has one GPU.  It
keeps the original trainer's ``lambda=0`` path untouched and writes a state
manifest after each objective/lambda pair.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v1"
DEFAULT_SOURCE = OUT / "runs/supervised/development/s1_seed3407_check0.25/best.pt"
TRAINER = ROOT / "scripts/303_train_tusz_meta_ttt_v1.py"
SELECTED = (("band", 1.0e-4), ("learned", 1.0e-5))
LAMBDAS = (0.1, 1.0)


def _write_state(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--stage", choices=["development", "formal"], default="development")
    parser.add_argument("--mode", choices=["online", "same_window"], default="online")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--maximum-records", type=int)
    parser.add_argument("--records-per-patient", type=int)
    parser.add_argument("--same-window-samples-per-record", type=int, default=4)
    parser.add_argument("--objectives", nargs="+", default=[name for name, _ in SELECTED])
    parser.add_argument("--lambdas", nargs="+", type=float, default=list(LAMBDAS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-tag", default="alignment_v1")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    selected_lrs = dict(SELECTED)
    unknown = sorted(set(args.objectives) - set(selected_lrs))
    if unknown:
        parser.error(f"unsupported objective(s): {unknown}")
    if any(value <= 0.0 for value in args.lambdas):
        parser.error("all alignment lambdas must be positive; lambda=0 is the baseline")
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    OUT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = OUT / "cosine_alignment_queue_state.json"
    tasks = [
        {"objective": objective, "inner_lr": selected_lrs[objective], "lambda": value}
        for objective in args.objectives
        for value in args.lambdas
    ]
    payload = {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "source": str(source),
        "source_sha256": None,
        "stage": args.stage,
        "mode": args.mode,
        "seed": args.seed,
        "epochs": args.epochs,
        "tasks": tasks,
        "results": [],
    }
    _write_state(state_path, payload)

    for task in tasks:
        objective = task["objective"]
        inner_lr = float(task["inner_lr"])
        alignment_lambda = float(task["lambda"])
        started = time.monotonic()
        log_path = log_dir / (
            f"meta_{args.mode}_cosine_{objective}_lambda{alignment_lambda:g}.log"
        )
        command = [
            sys.executable,
            str(TRAINER),
            "--objective",
            objective,
            "--mode",
            args.mode,
            "--stage",
            args.stage,
            "--source",
            str(source),
            "--inner-lr",
            str(inner_lr),
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--alignment-lambda",
            str(alignment_lambda),
            "--run-tag",
            args.run_tag,
            "--same-window-samples-per-record",
            str(args.same_window_samples_per_record),
        ]
        if args.maximum_records is not None:
            command.extend(["--maximum-records", str(args.maximum_records)])
        if args.records_per_patient is not None:
            command.extend(["--records-per-patient", str(args.records_per_patient)])
        if args.smoke:
            command.append("--smoke")
        task_result = {**task, "log": str(log_path), "command": command}
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        task_result.update(
            {
                "returncode": completed.returncode,
                "status": "complete" if completed.returncode == 0 else "failed",
                "elapsed_s": time.monotonic() - started,
            }
        )
        payload["results"].append(task_result)
        payload["last_completed_utc"] = datetime.now(UTC).isoformat()
        _write_state(state_path, payload)
        if completed.returncode != 0 and not args.continue_on_error:
            payload["status"] = "failed"
            _write_state(state_path, payload)
            raise SystemExit(completed.returncode)

    payload["status"] = "complete"
    payload["completed_utc"] = datetime.now(UTC).isoformat()
    _write_state(state_path, payload)


if __name__ == "__main__":
    main()

