#!/usr/bin/env python3
"""Apply the preregistered SSL health checks without consulting seizure BCE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def latest_metrics(run: Path) -> dict[str, float]:
    history = json.loads((run / "history.json").read_text())
    if not history:
        raise ValueError(f"empty history: {run}")
    return history[-1]["validation"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates: dict[str, list[dict]] = {name: [] for name in ("band", "temporal", "mask")}
    for run in sorted(args.runs.glob(f"*_seed{args.seed}")):
        parts = run.name.removesuffix(f"_seed{args.seed}").split("_", 1)
        if len(parts) != 2 or parts[0] not in candidates:
            continue
        objective, difficulty_text = parts
        metrics = latest_metrics(run)
        row = {
            "objective": objective,
            "difficulty": float(difficulty_text),
            "run": str(run.resolve()),
            "metrics": metrics,
        }
        if objective in {"band", "temporal"}:
            random_accuracy = 0.2 if objective == "band" else 0.5
            accuracy = (
                sum(metrics[f"class_{index}_accuracy"] for index in range(5)) / 5
                if objective == "band"
                else metrics["balanced_accuracy"]
            )
            gain = metrics["normalized_ce_gain"]
            row["healthy"] = random_accuracy + 0.05 < accuracy < 0.95 and 0.10 <= gain <= 0.95
            row["selection_distance"] = abs(gain - 0.50)
        else:
            improvement = metrics.get("relative_improvement_over_training_mean", float("-inf"))
            row["healthy"] = improvement >= 0.05
            row["selection_priority"] = {5.0: 0, 3.0: 1, 7.0: 2}[float(row["difficulty"])]
        candidates[objective].append(row)

    selected = {}
    for objective, rows in candidates.items():
        healthy = [row for row in rows if row["healthy"]]
        if objective in {"band", "temporal"}:
            healthy.sort(key=lambda row: (row["selection_distance"], row["difficulty"]))
        else:
            healthy.sort(key=lambda row: row["selection_priority"])
        selected[objective] = healthy[0] if healthy else None
    report = {"seed": args.seed, "candidates": candidates, "selected": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value and value["difficulty"] for key, value in selected.items()}))


if __name__ == "__main__":
    main()
