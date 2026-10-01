#!/usr/bin/env python3
"""Create the fixed-budget v3 development table and exact weighted BCE audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.tusz_meta_ttt.scoring import score_at_threshold

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"
V3 = ROOT / "outputs/reports/tusz_meta_ttt_v3"


def weighted_bce(frame: pd.DataFrame, probability: str) -> tuple[float, dict[str, float]]:
    work = frame.copy()
    work["class"] = np.where(work.label.to_numpy() > 0, "seizure", "background")
    record = ["patient_id", "session_id", "montage", "record_id"]
    patients = work.groupby("class").patient_id.nunique().to_dict()
    records = work.groupby(["class", "patient_id"])["record_id"].nunique().to_dict()
    counts = work.groupby([*record, "class"]).size().to_dict()
    mass = np.asarray([
        1 / (2 * patients[row["class"]]
             * records[(row["class"], row.patient_id)]
             * counts[(row.patient_id, row.session_id, row.montage, row.record_id, row["class"])])
        for _, row in work.iterrows()
    ])
    target = work.label.to_numpy(float)
    score = np.clip(work[probability].to_numpy(float), 1e-7, 1 - 1e-7)
    loss = -(target * np.log(score) + (1 - target) * np.log1p(-score))
    parts = {name: float((loss * mass * (work["class"].to_numpy() == name)).sum())
             for name in ("seizure", "background")}
    return float((loss * mass).sum()), parts


def records(frame: pd.DataFrame, condition: str, lookup: dict) -> list[dict]:
    result = []
    keys = ["patient_id", "session_id", "montage", "record_id"]
    for key, group in frame.groupby(keys, sort=False):
        item = lookup[tuple(key)]
        result.append(dict(times=group.decision_end_s.to_numpy(),
                           probabilities=group[condition].to_numpy(),
                           truths=item["seizures"], duration_s=item["duration_s"]))
    return result


def main() -> None:
    inventory = json.loads((V1 / "manifests/records.json").read_text())
    lookup = {(x["patient_id"], x["session_id"], x["montage"], x["record_id"]): x
              for x in inventory}
    rows = []
    for objective in ("mask", "band"):
        for condition in ("b0", "b1", "b2", "b3"):
            run = V3 / "evaluation/future" / f"{condition}_{objective}" / "development_validation"
            summary = json.loads((run / "summary.json").read_text())
            frame = pd.read_parquet(run / "probabilities.parquet")
            frozen = summary["conditions"]["meta_frozen"]
            adapted = summary["conditions"]["adapted"]
            fm, am = frozen["maximum_sensitivity_metrics"], adapted["maximum_sensitivity_metrics"]
            fixed = score_at_threshold(records(frame, "adapted", lookup),
                                       float(frozen["maximum_sensitivity_threshold"]))
            frozen_bce, frozen_parts = weighted_bce(frame, "meta_frozen")
            adapted_bce, adapted_parts = weighted_bce(frame, "adapted")
            rows.append(dict(
                objective=objective, condition=condition,
                reachable_80_frozen=frozen["reachable"], reachable_80_adapted=adapted["reachable"],
                frozen_sensitivity=fm["sensitivity"], adapted_sensitivity=am["sensitivity"],
                sensitivity_delta=am["sensitivity"] - fm["sensitivity"],
                frozen_fa_h=fm["false_alarms_per_hour"], adapted_fa_h=am["false_alarms_per_hour"],
                fa_h_relative_change=am["false_alarms_per_hour"] / fm["false_alarms_per_hour"] - 1,
                frozen_fa_time=fm["false_alarm_minutes_per_hour"], adapted_fa_time=am["false_alarm_minutes_per_hour"],
                fa_time_relative_change=am["false_alarm_minutes_per_hour"] / fm["false_alarm_minutes_per_hour"] - 1,
                frozen_delay=fm["median_delay_s"], adapted_delay=am["median_delay_s"],
                fixed_frozen_threshold=float(frozen["maximum_sensitivity_threshold"]),
                fixed_threshold_adapted_sensitivity=fixed.sensitivity,
                fixed_threshold_adapted_fa_h=fixed.false_alarms_per_hour,
                weighted_bce_frozen=frozen_bce, weighted_bce_adapted=adapted_bce,
                weighted_bce_delta=adapted_bce - frozen_bce,
                seizure_bce_delta=adapted_parts["seizure"] - frozen_parts["seizure"],
                background_bce_delta=adapted_parts["background"] - frozen_parts["background"],
            ))
    table = pd.DataFrame(rows)
    report = V3 / "reports"
    report.mkdir(parents=True, exist_ok=True)
    table.to_csv(report / "development_results.csv", index=False)
    (report / "development_results.json").write_text(table.to_json(orient="records", indent=2))
    display = table[["objective", "condition", "frozen_sensitivity", "adapted_sensitivity",
                     "frozen_fa_h", "adapted_fa_h", "fa_h_relative_change",
                     "fa_time_relative_change", "weighted_bce_delta"]].copy()
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |"
            for row in display.itertuples(index=False, name=None)]
    (report / "development_results.md").write_text("\n".join([header, separator, *body]) + "\n")


if __name__ == "__main__":
    main()
