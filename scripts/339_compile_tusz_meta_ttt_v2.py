#!/usr/bin/env python3
"""Compile training coverage, event results, and mechanism summaries into deliverables."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def main() -> None:
    destination = OUT / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    result_rows = []
    for path in sorted((OUT / "evaluation").rglob("summary.json")):
        if any(token in str(path) for token in ("_smoke", "throughput", "_invalid")):
            continue
        state = json.loads(path.read_text())
        conditions = state.get("conditions")
        if not conditions:
            continue
        for condition, choice in conditions.items():
            metrics = choice.get("metrics") or choice.get("maximum_sensitivity_metrics")
            result_rows.append({
                "summary": str(path.resolve()),
                "mode": state.get("mode", "detector"),
                "run": path.parent.parent.name,
                "cohort": path.parent.name,
                "condition": condition,
                "reachable": choice.get("reachable", choice.get("reachable_on_dev")),
                "threshold": choice.get("threshold"),
                **(metrics or {}),
            })
    result_frame = pd.DataFrame(result_rows)
    result_frame.to_csv(destination / "event_results.csv", index=False)

    training_rows = []
    for path in sorted((OUT / "runs").rglob("history.json")):
        if any(token in str(path) for token in ("_smoke", "throughput", "_invalid")):
            continue
        history = json.loads(path.read_text())
        for row in history:
            training_rows.append({"run": str(path.parent.resolve()), **row})
    (destination / "training_manifest.json").write_text(
        json.dumps(training_rows, indent=2) + "\n"
    )

    mechanisms = []
    for path in sorted((OUT / "mechanisms").rglob("summary.json")):
        mechanisms.append({"path": str(path.resolve()), "summary": json.loads(path.read_text())})
    (destination / "mechanisms.json").write_text(json.dumps(mechanisms, indent=2) + "\n")

    lines = [
        "# TUSZ Meta-TTT v2 results",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        f"Event result rows: {len(result_frame)}",
        f"Training history rows: {len(training_rows)}",
        f"Mechanism summaries: {len(mechanisms)}",
        "",
    ]
    if not result_frame.empty:
        display_columns = [
            column for column in (
                "mode", "run", "cohort", "condition", "reachable", "sensitivity",
                "false_alarms_per_hour", "false_alarm_minutes_per_hour", "median_delay_s",
            ) if column in result_frame
        ]
        lines.extend((markdown_table(result_frame[display_columns]), ""))
    (destination / "main_results.md").write_text("\n".join(lines))
    print(json.dumps({
        "destination": str(destination),
        "event_rows": len(result_frame),
        "training_rows": len(training_rows),
        "mechanism_reports": len(mechanisms),
    }))


if __name__ == "__main__":
    main()
