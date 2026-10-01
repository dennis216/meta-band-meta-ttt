#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "outputs/reports/tusz_meta_ttt_v1/evaluation"
SELECTED = {"band": 1e-4, "temporal": 1e-4, "mask": 1e-6, "learned": 1e-5}


def main() -> None:
    rows = json.loads((EVALUATION / "short_grid_metrics.json").read_text())
    selected = []
    for objective, inner_lr in SELECTED.items():
        row = next(item for item in rows if item["objective"] == objective and item["inner_lr"] == inner_lr)
        selected.append(row)
    payload = {
        "source_health_target_reachable": False,
        "target_sensitivity": 0.80,
        "selection_metric_when_unreachable": [
            "maximum sensitivity descending",
            "false alarms/hour ascending",
            "false-alarm time ascending",
            "median delay ascending",
        ],
        "selected": selected,
    }
    (EVALUATION / "short_grid_selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# TUSZ Meta-TTT short-grid selection",
        "",
        "The 80% event-sensitivity target was unreachable for every condition; selections are diagnostic and do not change the formal success threshold.",
        "",
        "| Objective | Inner LR | Max sensitivity | FA/hour | FA time min/hour | delta FA/hour | delta FA time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['objective']} | {row['inner_lr']:.0e} | {row['a_sens']:.4f} | {row['a_fah']:.4f} | "
            f"{row['a_fat']:.4f} | {row['delta_fah']:+.4f} | {row['delta_fat']:+.4f} |"
        )
    (EVALUATION / "short_grid_selection.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
