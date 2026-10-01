from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatientComponents:
    patient_id: str
    detected: int
    total: int
    false_alarms: int
    background_hours: float
    false_alarm_seconds: float


def aggregate_components(rows: list[PatientComponents]) -> dict[str, float]:
    detected = sum(row.detected for row in rows)
    total = sum(row.total for row in rows)
    false_alarms = sum(row.false_alarms for row in rows)
    background_hours = sum(row.background_hours for row in rows)
    false_alarm_seconds = sum(row.false_alarm_seconds for row in rows)
    return {
        "sensitivity": detected / total if total else float("nan"),
        "false_alarms_per_hour": false_alarms / background_hours if background_hours else float("nan"),
        "false_alarm_minutes_per_hour": false_alarm_seconds / 60 / background_hours if background_hours else float("nan"),
    }


def paired_patient_bootstrap(
    frozen: list[PatientComponents],
    adapted: list[PatientComponents],
    *,
    replicates: int = 10_000,
    seed: int = 3407,
) -> dict[str, dict[str, float]]:
    frozen_map = {row.patient_id: row for row in frozen}
    adapted_map = {row.patient_id: row for row in adapted}
    if set(frozen_map) != set(adapted_map):
        raise ValueError("paired conditions must contain identical patients")
    patients = sorted(frozen_map)
    if not patients:
        raise ValueError("no patients to bootstrap")
    rng = np.random.default_rng(seed)
    sensitivity_differences = []
    fa_ratios = []
    fa_differences = []
    for _ in range(replicates):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        left = aggregate_components([frozen_map[patient] for patient in sampled])
        right = aggregate_components([adapted_map[patient] for patient in sampled])
        sensitivity_differences.append(right["sensitivity"] - left["sensitivity"])
        fa_differences.append(right["false_alarms_per_hour"] - left["false_alarms_per_hour"])
        if left["false_alarms_per_hour"] > 0:
            fa_ratios.append(right["false_alarms_per_hour"] / left["false_alarms_per_hour"])

    def summary(values):
        array = np.asarray(values, dtype=float)
        return {
            "estimate": float(np.mean(array)),
            "ci_lower": float(np.quantile(array, 0.025)),
            "ci_upper": float(np.quantile(array, 0.975)),
        }

    result = {
        "sensitivity_difference": summary(sensitivity_differences),
        "false_alarms_per_hour_difference": summary(fa_differences),
    }
    result["false_alarms_per_hour_ratio"] = (
        summary(fa_ratios) if fa_ratios else {"estimate": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
    )
    return result
