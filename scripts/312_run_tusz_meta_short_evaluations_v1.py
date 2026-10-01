#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v1"
SOURCE = OUT / "runs/supervised/development/s1_seed3407_check0.25/best.pt"
OBJECTIVES = ("band", "temporal", "mask", "learned")
INNER_LRS = (1e-6, 1e-5, 1e-4)
FROZEN = (
    OUT
    / "evaluation/band_lr1e-06/online/seed3407/development_validation"
    / "probabilities.parquet"
)


def run_dir(objective: str, inner_lr: float) -> Path:
    return (
        OUT
        / "evaluation"
        / f"{objective}_lr{inner_lr:g}"
        / "online/seed3407/development_validation"
    )


def complete(path: Path) -> bool:
    summary = path / "summary.json"
    if not summary.is_file():
        return False
    state = json.loads(summary.read_text())
    adapted = state.get("conditions", {}).get("adapted", {})
    return state.get("records") == 1218 and "maximum_sensitivity_metrics" in adapted


def main() -> None:
    state_path = OUT / "evaluation/short_grid_state.json"
    state = {"started_utc": datetime.now(UTC).isoformat(), "runs": []}
    for objective in OBJECTIVES:
        for inner_lr in INNER_LRS:
            destination = run_dir(objective, inner_lr)
            row = {
                "objective": objective,
                "inner_lr": inner_lr,
                "directory": str(destination),
            }
            if complete(destination):
                row["status"] = "existing"
            else:
                checkpoint = (
                    OUT
                    / "runs/meta/development/online"
                    / f"{objective}_lr{inner_lr:g}_seed3407/epoch_01.pt"
                )
                command = [
                    sys.executable,
                    str(ROOT / "scripts/304_evaluate_tusz_meta_ttt_v1.py"),
                    "--source",
                    str(SOURCE),
                    "--objective-checkpoint",
                    str(checkpoint),
                    "--partition",
                    "train",
                    "--cohort",
                    "development_validation",
                    "--mode",
                    "online",
                    "--seed",
                    "3407",
                    "--calibrate",
                    "--frozen-probabilities",
                    str(FROZEN),
                ]
                log = OUT / "logs" / f"eval_short_{objective}_lr{inner_lr:g}.log"
                with log.open("w") as stream:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                row["status"] = "complete" if result.returncode == 0 else "failed"
                row["returncode"] = result.returncode
                if result.returncode:
                    state["runs"].append(row)
                    state_path.write_text(json.dumps(state, indent=2) + "\n")
                    raise SystemExit(result.returncode)
            state["runs"].append(row)
            state_path.write_text(json.dumps(state, indent=2) + "\n")
    state["completed_utc"] = datetime.now(UTC).isoformat()
    state_path.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
