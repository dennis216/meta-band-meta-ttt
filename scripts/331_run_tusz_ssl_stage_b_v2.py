#!/usr/bin/env python3
"""Resumable Stage-B queue: nine SSL warmups, selection, and inner calibration."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"
SOURCE = ROOT / "outputs/reports/tusz_meta_ttt_v1/runs/supervised/development/s1_seed3407_check0.25/best.pt"
CANDIDATES = {
    "band": (0.25, 0.5, 0.75),
    "temporal": (2, 5, 10),
    "mask": (3, 5, 7),
}


def call(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        result = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    state_path = OUT / "queues/stage_b_v2.json"
    rows = [
        {"objective": objective, "difficulty": difficulty, "status": "pending"}
        for objective, difficulties in CANDIDATES.items()
        for difficulty in difficulties
    ]
    state = {
        "created_utc": datetime.now(UTC).isoformat(),
        "estimated_from_4096_window_benchmark_hours": {
            "band_three_candidates_two_epochs": 4.7,
            "temporal_three_candidates_two_epochs": 1.4,
            "mask_three_candidates_two_epochs": 14.5,
            "total_plus_20_percent": 24.7,
        },
        "runs": rows,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    if not args.execute:
        print(json.dumps({"queue": str(state_path), "runs": len(rows)}))
        return
    ssl_root = OUT / "runs/ssl/development"
    for row in rows:
        name = f'{row["objective"]}_{row["difficulty"]:g}_seed3407'
        run = ssl_root / name
        if not (run / "epoch_02.pt").is_file():
            command = [
                sys.executable,
                str(ROOT / "scripts/321_train_tusz_ssl_v2.py"),
                "--source", str(SOURCE),
                "--objective", row["objective"],
                "--difficulty", str(row["difficulty"]),
                "--epochs", "2",
                "--batch-size", str(args.batch_size),
                "--workers", str(args.workers),
            ]
            if (run / "last.pt").is_file():
                command.extend(("--resume", str(run / "last.pt")))
            call(command, OUT / "logs" / f"ssl_warmup_{name}.log")
        row["status"] = "complete"
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    selection_path = OUT / "selection/ssl_difficulty_seed3407.json"
    call(
        [
            sys.executable, str(ROOT / "scripts/325_select_tusz_ssl_v2.py"),
            "--runs", str(ssl_root), "--output", str(selection_path),
        ],
        OUT / "logs/ssl_selection.log",
    )
    selected = json.loads(selection_path.read_text())["selected"]
    state["selection"] = selected
    for objective, choice in selected.items():
        if choice is None:
            continue
        difficulty = float(choice["difficulty"])
        calibration = OUT / "calibration" / f"{objective}_{difficulty:g}_seed3407/gradient_calibration.json"
        checkpoint = max(Path(choice["run"]).glob("epoch_*.pt"))
        calibration_is_current = False
        if calibration.is_file():
            try:
                existing = json.loads(calibration.read_text())
                calibration_is_current = (
                    {"health_check_passed", "selected_relative_step"} <= existing.keys()
                    and Path(existing.get("objective_checkpoint", "")).resolve()
                    == checkpoint.resolve()
                )
            except (json.JSONDecodeError, OSError, TypeError):
                calibration_is_current = False
        if not calibration_is_current:
            call(
                [
                    sys.executable, str(ROOT / "scripts/322_calibrate_tusz_inner_v2.py"),
                    "--source", str(SOURCE),
                    "--objective-checkpoint", str(checkpoint),
                    "--objective", objective,
                    "--difficulty", str(difficulty),
                ],
                OUT / "logs" / f"inner_calibration_{objective}_{difficulty:g}.log",
            )
        calibration_state = json.loads(calibration.read_text())
        state.setdefault("calibration", {})[objective] = {
            "path": str(calibration),
            "health_check_passed": calibration_state["health_check_passed"],
            "selected_relative_step": calibration_state["selected_relative_step"],
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    state["completed_utc"] = datetime.now(UTC).isoformat()
    state_path.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
