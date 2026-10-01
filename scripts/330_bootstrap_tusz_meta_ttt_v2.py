#!/usr/bin/env python3
"""Patient-paired bootstrap across one or more confirmation seeds."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.match import match_events
from bfa.tusz_meta_ttt.scoring import (
    EventMetrics,
    consecutive_eventize,
    score_record,
)

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"


def records(frame: pd.DataFrame, condition: str, inventory: dict) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    keys = ["patient_id", "session_id", "montage", "record_id"]
    for key, group in frame.groupby(keys, sort=False):
        patient = str(key[0])
        item = inventory[tuple(key)]
        output.setdefault(patient, []).append({
            "times": group["decision_end_s"].to_numpy(float),
            "probabilities": group[condition].to_numpy(float),
            "truths": item["seizures"],
            "duration_s": item["duration_s"],
        })
    return output


def effects(adapted, frozen) -> dict[str, float]:
    return {
        "sensitivity_difference": adapted.sensitivity - frozen.sensitivity,
        "fa_per_hour_difference": adapted.false_alarms_per_hour - frozen.false_alarms_per_hour,
        "fa_per_hour_ratio": (
            adapted.false_alarms_per_hour / frozen.false_alarms_per_hour
            if frozen.false_alarms_per_hour > 0 else float("nan")
        ),
        "fa_time_difference": (
            adapted.false_alarm_minutes_per_hour - frozen.false_alarm_minutes_per_hour
        ),
        "median_delay_difference_s": adapted.median_delay_s - frozen.median_delay_s,
    }


def union_duration(intervals) -> float:
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def patient_component(patient_records: list[dict], threshold: float) -> dict:
    component = {
        "detected": 0,
        "total": 0,
        "false_alarms": 0,
        "background_hours": 0.0,
        "false_alarm_seconds": 0.0,
        "delays": [],
    }
    for record in patient_records:
        events = consecutive_eventize(
            record["times"], record["probabilities"], threshold=threshold
        )
        metrics = score_record(events, record["truths"], duration_s=record["duration_s"])
        background_hours = (
            record["duration_s"] - union_duration(record["truths"])
        ) / 3600
        component["detected"] += metrics.detected
        component["total"] += metrics.total
        component["false_alarms"] += metrics.false_alarms
        component["background_hours"] += background_hours
        component["false_alarm_seconds"] += (
            metrics.false_alarm_minutes_per_hour * 60 * background_hours
        )
        matched = match_events(events, record["truths"])
        component["delays"].extend(
            max(
                0.0,
                events[pair.prediction_index].start_s
                - record["truths"][pair.truth_index][0],
            )
            for pair in matched.pairs
        )
    return component


def aggregate_components(components: list[dict]) -> EventMetrics:
    detected = sum(row["detected"] for row in components)
    total = sum(row["total"] for row in components)
    false_alarms = sum(row["false_alarms"] for row in components)
    background_hours = sum(row["background_hours"] for row in components)
    false_alarm_seconds = sum(row["false_alarm_seconds"] for row in components)
    delays = list(chain.from_iterable(row["delays"] for row in components))
    denominator = max(np.finfo(float).eps, background_hours)
    return EventMetrics(
        detected / total if total else float("nan"),
        false_alarms / denominator,
        false_alarm_seconds / 60 / denominator,
        float(np.median(delays)) if delays else float("nan"),
        detected,
        total,
        false_alarms,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--dev-summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--adapted-condition", default="adapted")
    parser.add_argument("--frozen-condition", default="meta_frozen")
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.evaluations) != len(args.dev_summaries):
        raise ValueError("each seed evaluation requires its matching Dev summary")
    inventory_rows = json.loads((V1 / "manifests/records.json").read_text())
    inventory = {
        (row["patient_id"], row["session_id"], row["montage"], row["record_id"]): row
        for row in inventory_rows
    }
    seeds = []
    for evaluation_path, summary_path in zip(
        args.evaluations, args.dev_summaries, strict=True
    ):
        frame = pd.read_parquet(evaluation_path)
        summary = json.loads(summary_path.read_text())
        adapted_threshold = summary["conditions"][args.adapted_condition]["threshold"]
        frozen_threshold = summary["conditions"][args.frozen_condition]["threshold"]
        if adapted_threshold is None or frozen_threshold is None:
            raise ValueError(f"Dev operating point unreachable in {summary_path}")
        seeds.append({
            "adapted": records(frame, args.adapted_condition, inventory),
            "frozen": records(frame, args.frozen_condition, inventory),
            "adapted_threshold": float(adapted_threshold),
            "frozen_threshold": float(frozen_threshold),
        })
    patient_sets = [set(seed["adapted"]) & set(seed["frozen"]) for seed in seeds]
    patients = sorted(set.intersection(*patient_sets))
    if not patients:
        raise ValueError("no patients shared by every seed")

    for seed in seeds:
        seed["adapted_components"] = {
            patient: patient_component(seed["adapted"][patient], seed["adapted_threshold"])
            for patient in patients
        }
        seed["frozen_components"] = {
            patient: patient_component(seed["frozen"][patient], seed["frozen_threshold"])
            for patient in patients
        }

    point_by_seed = []
    for seed in seeds:
        point_by_seed.append(effects(
            aggregate_components([seed["adapted_components"][patient] for patient in patients]),
            aggregate_components([seed["frozen_components"][patient] for patient in patients]),
        ))
    rng = np.random.default_rng(args.seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(args.replicates):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        seed_effects = []
        for seed in seeds:
            seed_effects.append(effects(
                aggregate_components([
                    seed["adapted_components"][patient] for patient in sampled
                ]),
                aggregate_components([
                    seed["frozen_components"][patient] for patient in sampled
                ]),
            ))
        for metric in seed_effects[0]:
            draws[metric].append(float(np.nanmean([row[metric] for row in seed_effects])))
    intervals = {}
    for metric, values in draws.items():
        array = np.asarray(values, dtype=float)
        null = 1.0 if metric == "fa_per_hour_ratio" else 0.0
        finite = array[np.isfinite(array)]
        intervals[metric] = {
            "mean": float(np.nanmean(array)),
            "ci95": np.nanquantile(array, [0.025, 0.975]).tolist(),
            "two_sided_bootstrap_p": (
                float(min(1.0, 2 * min(np.mean(finite <= null), np.mean(finite >= null))))
                if len(finite) else None
            ),
        }
    result = {
        "patients": len(patients),
        "seeds": len(seeds),
        "replicates": args.replicates,
        "adapted_condition": args.adapted_condition,
        "frozen_condition": args.frozen_condition,
        "point_effects_by_seed": point_by_seed,
        "bootstrap": intervals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
