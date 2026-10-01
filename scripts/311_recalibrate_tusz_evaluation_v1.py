#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bfa.tusz_meta_ttt.scoring import choose_threshold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.evaluation_dir.resolve()
    frame = pd.read_parquet(run / "probabilities.parquet")
    inventory = json.loads((OUT / "manifests/records.json").read_text())
    lookup = {
        (item["patient_id"], item["session_id"], item["montage"], item["record_id"]): item
        for item in inventory
    }
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text())
    keys = ["patient_id", "session_id", "montage", "record_id"]
    for condition in ("frozen", "adapted"):
        records = []
        for key, group in frame.groupby(keys, sort=False):
            item = lookup[tuple(key)]
            records.append(
                {
                    "patient_id": item["patient_id"],
                    "times": group["decision_end_s"].to_numpy(float),
                    "probabilities": group[f"{condition}_probability"].to_numpy(float),
                    "truths": item["seizures"],
                    "duration_s": item["duration_s"],
                }
            )
        choice = choose_threshold(records)
        summary["conditions"][condition] = {
            "threshold": choice.threshold,
            "reachable": choice.reachable,
            "metrics": vars(choice.metrics) if choice.metrics else None,
            "maximum_sensitivity_threshold": choice.maximum_sensitivity_threshold,
            "maximum_sensitivity_metrics": (
                vars(choice.maximum_sensitivity_metrics)
                if choice.maximum_sensitivity_metrics
                else None
            ),
        }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
