#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import causal_ema
from bfa.evaluation.match import match_events
from bfa.tusz_meta_ttt.scoring import (
    EventMetrics,
    _union_duration,
    consecutive_eventize,
    score_record,
)

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"
V2 = ROOT / "outputs/reports/tusz_meta_ttt_v2"
V3 = ROOT / "outputs/reports/tusz_meta_ttt_v3"
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v4/rescoring"
KEY = ("patient_id", "session_id", "montage", "record_id")
TARGETS = (0.60, 0.65, 0.70, 0.75, 0.80)

_WORKER_RECORDS: list[dict] | None = None


def _aggregate_at_threshold(records: list[dict], threshold: float) -> dict:
    detected = total = false_alarms = 0
    background_hours = false_alarm_seconds = 0.0
    delays: list[float] = []
    for record in records:
        events = consecutive_eventize(
            record["times"], record["probabilities"], threshold=threshold,
            record_end_s=record["duration_s"],
        )
        metrics = score_record(events, record["truths"], duration_s=record["duration_s"])
        detected += metrics.detected
        total += metrics.total
        false_alarms += metrics.false_alarms
        background = max(
            0.0, (record["duration_s"] - _union_duration(record["truths"])) / 3600
        )
        background_hours += background
        false_alarm_seconds += metrics.false_alarm_minutes_per_hour * 60 * background
        matches = match_events(events, record["truths"])
        delays.extend(
            max(0.0, events[p.prediction_index].start_s - record["truths"][p.truth_index][0])
            for p in matches.pairs
        )
    denominator = max(np.finfo(float).eps, background_hours)
    return asdict(EventMetrics(
        detected / total if total else float("nan"),
        false_alarms / denominator,
        false_alarm_seconds / 60 / denominator,
        float(np.median(delays)) if delays else float("nan"),
        detected, total, false_alarms,
    )) | {"threshold": float(threshold), "background_hours": background_hours}


def _init_worker(records: list[dict]) -> None:
    global _WORKER_RECORDS
    _WORKER_RECORDS = records


def _worker(threshold: float) -> dict:
    assert _WORKER_RECORDS is not None
    return _aggregate_at_threshold(_WORKER_RECORDS, threshold)


def _thresholds(records: list[dict]) -> np.ndarray:
    values = [
        causal_ema(record["probabilities"], 1 / 3)
        for record in records if len(record["probabilities"])
    ]
    if not values:
        raise ValueError("condition has no prediction windows")
    combined = np.concatenate(values)
    return np.unique(np.concatenate((
        [-np.inf], np.quantile(combined, np.linspace(0, 1, 1001)), [np.inf],
    )))


def _select(curve: pd.DataFrame, target: float) -> dict | None:
    feasible = curve[curve.sensitivity >= target]
    if feasible.empty:
        return None
    row = feasible.sort_values(
        ["false_alarms_per_hour", "false_alarm_minutes_per_hour", "threshold"],
        ascending=[True, True, False], kind="mergesort",
    ).iloc[0]
    return {name: (None if pd.isna(value) else float(value)) for name, value in row.items()}


def _record_rows(records: list[dict], threshold: float) -> pd.DataFrame:
    rows = []
    for record in records:
        events = consecutive_eventize(
            record["times"], record["probabilities"], threshold=threshold,
            record_end_s=record["duration_s"],
        )
        metrics = score_record(events, record["truths"], duration_s=record["duration_s"])
        background = max(
            0.0, (record["duration_s"] - _union_duration(record["truths"])) / 3600
        )
        rows.append(dict(zip(KEY, record["key"], strict=True)) | asdict(metrics) | {
            "duration_s": record["duration_s"],
            "background_hours": background,
            "false_alarm_seconds": metrics.false_alarm_minutes_per_hour * 60 * background,
            "threshold": threshold,
            "window_count": len(record["times"]),
        })
    return pd.DataFrame(rows)


def _common_delay(left: list[dict], right: list[dict], left_t: float, right_t: float) -> dict:
    differences = []
    for left_record, right_record in zip(left, right, strict=True):
        if left_record["key"] != right_record["key"]:
            raise ValueError("paired records are not aligned")
        left_events = consecutive_eventize(
            left_record["times"], left_record["probabilities"], threshold=left_t,
            record_end_s=left_record["duration_s"],
        )
        right_events = consecutive_eventize(
            right_record["times"], right_record["probabilities"], threshold=right_t,
            record_end_s=right_record["duration_s"],
        )
        left_matches = {
            p.truth_index: max(0.0, left_events[p.prediction_index].start_s - left_record["truths"][p.truth_index][0])
            for p in match_events(left_events, left_record["truths"]).pairs
        }
        right_matches = {
            p.truth_index: max(0.0, right_events[p.prediction_index].start_s - right_record["truths"][p.truth_index][0])
            for p in match_events(right_events, right_record["truths"]).pairs
        }
        differences.extend(left_matches[i] - right_matches[i] for i in left_matches.keys() & right_matches.keys())
    return {
        "common_detected_events": len(differences),
        "median_adapted_minus_frozen_delay_s": float(np.median(differences)) if differences else None,
    }


def _inventory() -> tuple[list[dict], set[str]]:
    inventory = json.loads((V1 / "manifests/records.json").read_text())
    patients = set(json.loads((V1 / "manifests/development_split.json").read_text())["development_validation"])
    selected = [
        item for item in inventory
        if item["partition"] == "train"
        and item["patient_id"] in patients
        and item.get("exclusion") is None
    ]
    selected.sort(key=lambda item: tuple(item[name] for name in KEY))
    return selected, patients


def _records(frame: pd.DataFrame, score_column: str, inventory: list[dict]) -> list[dict]:
    grouped = {tuple(key): group for key, group in frame.groupby(list(KEY), sort=False)}
    records = []
    for item in inventory:
        key = tuple(item[name] for name in KEY)
        group = grouped.get(key)
        if group is None:
            times = probabilities = np.array([], dtype=float)
        else:
            group = group.sort_values("decision_end_s")
            times = group.decision_end_s.to_numpy(dtype=float)
            probabilities = group[score_column].to_numpy(dtype=float)
        records.append({
            "key": key, "times": times, "probabilities": probabilities,
            "truths": [tuple(pair) for pair in item["seizures"]],
            "duration_s": float(item["duration_s"]),
        })
    return records


def _evaluate(name: str, records: list[dict], workers: int) -> tuple[dict, list[dict]]:
    output = OUT / name
    output.mkdir(parents=True, exist_ok=True)
    thresholds = _thresholds(records)
    context = mp.get_context("fork")
    with context.Pool(workers, initializer=_init_worker, initargs=(records,)) as pool:
        curve_rows = list(pool.imap(_worker, thresholds, chunksize=4))
    curve = pd.DataFrame(curve_rows).sort_values("threshold")
    curve.to_parquet(output / "operating_curve.parquet", index=False)
    operating_points = {str(int(target * 100)): _select(curve, target) for target in TARGETS}
    maximum = curve.sort_values(
        ["sensitivity", "false_alarms_per_hour", "threshold"],
        ascending=[False, True, False], kind="mergesort",
    ).iloc[0].to_dict()
    summary = {
        "condition": name,
        "records": len(records),
        "records_without_windows": sum(not len(record["times"]) for record in records),
        "events": sum(len(record["truths"]) for record in records),
        "threshold_candidates": len(thresholds),
        "operating_points": operating_points,
        "maximum_sensitivity": {key: (None if pd.isna(value) else float(value)) for key, value in maximum.items()},
    }
    point80 = operating_points["80"]
    if point80 is not None:
        detail = _record_rows(records, point80["threshold"])
        detail.to_parquet(output / "record_metrics_80.parquet", index=False)
        patient = detail.groupby("patient_id", as_index=False).agg(
            detected=("detected", "sum"), total=("total", "sum"),
            false_alarms=("false_alarms", "sum"), background_hours=("background_hours", "sum"),
            false_alarm_seconds=("false_alarm_seconds", "sum"), records=("record_id", "size"),
        )
        patient.to_parquet(output / "patient_metrics_80.parquet", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(16, max(1, (mp.cpu_count() or 2) - 2)))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inventory, _ = _inventory()
    specifications: list[tuple[str, Path, str, str | None]] = []
    for ssl in ("band", "mask"):
        for condition in ("b0", "b1", "b2", "b3"):
            path = V3 / f"evaluation/future/{condition}_{ssl}/development_validation/probabilities.parquet"
            specifications.extend([
                (f"v3_{ssl}_{condition}_frozen", path, "meta_frozen", None),
                (f"v3_{ssl}_{condition}_adapted", path, "adapted", f"v3_{ssl}_{condition}_frozen"),
            ])
    for ssl in ("band", "mask"):
        for condition in ("a", "b", "c"):
            path = ROOT / f"outputs/reports/tusz_meta_ttt_v4/evaluation/future/{condition}_{condition}_{ssl}/development_validation/probabilities.parquet"
            specifications.extend([
                (f"v4_{ssl}_{condition}_frozen", path, "meta_frozen", None),
                (f"v4_{ssl}_{condition}_adapted", path, "adapted", f"v4_{ssl}_{condition}_frozen"),
            ])
    source_path = V3 / "evaluation/future/b0_mask/development_validation/probabilities.parquet"
    specifications.insert(0, ("development_s1", source_path, "source_frozen", None))
    for control in ("e", "ed"):
        path = V2 / f"evaluation/detectors/fast_control_{control}_seed3407/development_validation/probabilities.parquet"
        specifications.append((f"continued_supervision_{control}", path, "probability", None))

    all_summaries: dict[str, dict] = {}
    all_records: dict[str, list[dict]] = {}
    pairs: dict[str, str] = {}
    loaded: dict[Path, pd.DataFrame] = {}
    for index, (name, path, column, frozen_name) in enumerate(specifications, 1):
        print(json.dumps({"condition": name, "index": index, "total": len(specifications)}), flush=True)
        if path not in loaded:
            loaded[path] = pd.read_parquet(path)
        frame = loaded[path]
        records = _records(frame, column, inventory)
        summary, records = _evaluate(name, records, args.workers)
        all_summaries[name], all_records[name] = summary, records
        if frozen_name:
            pairs[name] = frozen_name

    for adapted_name, frozen_name in pairs.items():
        comparison = {"adapted": adapted_name, "frozen": frozen_name}
        comparison["operating_points"] = {}
        for target in TARGETS:
            key = str(int(target * 100))
            adapted_point = all_summaries[adapted_name]["operating_points"][key]
            frozen_point = all_summaries[frozen_name]["operating_points"][key]
            paired = {"adapted": adapted_point, "frozen": frozen_point}
            if frozen_point is not None:
                paired["adapted_at_frozen_threshold"] = _aggregate_at_threshold(
                    all_records[adapted_name], frozen_point["threshold"]
                )
            if adapted_point is not None and frozen_point is not None:
                paired["common_delay"] = _common_delay(
                    all_records[adapted_name], all_records[frozen_name],
                    adapted_point["threshold"], frozen_point["threshold"],
                )
            comparison["operating_points"][key] = paired
        (OUT / adapted_name / "paired_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (OUT / "index.json").write_text(json.dumps(all_summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
