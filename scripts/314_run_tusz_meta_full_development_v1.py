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
SELECTED = (("band", 1e-4), ("temporal", 1e-4), ("mask", 1e-6), ("learned", 1e-5))


def run_dir(objective: str, inner_lr: float) -> Path:
    return OUT / "runs/meta/development/online" / f"{objective}_lr{inner_lr:g}_seed3407_full"


def main() -> None:
    state_path = OUT / "runs/meta/development/online/full_development_state.json"
    state = {"started_utc": datetime.now(UTC).isoformat(), "runs": []}
    for objective, inner_lr in SELECTED:
        destination = run_dir(objective, inner_lr)
        row = {"objective": objective, "inner_lr": inner_lr, "directory": str(destination)}
        epoch_two = destination / "epoch_02.pt"
        if epoch_two.is_file():
            row["status"] = "existing"
        else:
            command = [
                sys.executable,
                str(ROOT / "scripts/303_train_tusz_meta_ttt_v1.py"),
                "--objective", objective,
                "--mode", "online",
                "--stage", "development",
                "--source", str(SOURCE),
                "--inner-lr", str(inner_lr),
                "--seed", "3407",
                "--epochs", "2",
                "--run-tag", "full",
            ]
            if (destination / "last.pt").is_file():
                command.append("--resume")
            log = OUT / "logs" / f"meta_full_{objective}_lr{inner_lr:g}.log"
            with log.open("a") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
            row.update(status="complete" if result.returncode == 0 else "failed", returncode=result.returncode)
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
