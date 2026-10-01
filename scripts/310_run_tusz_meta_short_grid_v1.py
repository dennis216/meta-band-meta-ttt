#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v1"
SOURCE = (
    OUT
    / "runs/supervised/development/s1_seed3407_check0.25/best.pt"
)
OBJECTIVES = ("band", "temporal", "mask", "learned")
INNER_LRS = (1e-6, 1e-5, 1e-4)


def checkpoint(objective: str, inner_lr: float) -> Path:
    return (
        OUT
        / "runs/meta/development/online"
        / f"{objective}_lr{inner_lr:g}_seed3407/epoch_01.pt"
    )


def main() -> None:
    state_path = OUT / "runs/meta/development/online/short_grid_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"started_utc": datetime.now(UTC).isoformat(), "runs": []}
    for objective in OBJECTIVES:
        for inner_lr in INNER_LRS:
            target = checkpoint(objective, inner_lr)
            row = {
                "objective": objective,
                "inner_lr": inner_lr,
                "checkpoint": str(target),
            }
            if target.is_file():
                row["status"] = "existing"
            else:
                command = [
                    sys.executable,
                    str(ROOT / "scripts/303_train_tusz_meta_ttt_v1.py"),
                    "--objective",
                    objective,
                    "--mode",
                    "online",
                    "--stage",
                    "development",
                    "--source",
                    str(SOURCE),
                    "--inner-lr",
                    str(inner_lr),
                    "--seed",
                    "3407",
                    "--epochs",
                    "1",
                    "--records-per-patient",
                    "1",
                ]
                log = OUT / "logs" / f"meta_short_{objective}_lr{inner_lr:g}.log"
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
