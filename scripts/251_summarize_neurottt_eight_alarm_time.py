"""Summarize alarm-time occupancy for the eight non-baseline NeuroTTT conditions.

The common ``supervised_frozen`` condition is retained as a reference, while
the requested eight conditions are the four joint variants and four meta-TTT
variants.  Only completed outer-test folds are scored.  The scoring contract
matches the frozen NeuroTTT evaluator: event matching uses the -30/+60 s
collar, alarm-time subtraction uses the raw seizure annotation, and the time
denominator is total monitoring time after the first evaluable window.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/b_false_alarm_atlas")
NEURO_CODE = Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/NeuroTTT/CBraMod")
JOINT_ROOT = ROOT / "outputs/reports/neurottt-chbmit-5fold-v1-bandfix-v5"
META_ROOT = ROOT / "outputs/reports/meta-ttt-chbmit-5fold-v1"
OUT = ROOT / "outputs/reports/neurottt-eight-alarm-time-v1"
RECORDINGS = ROOT / "manifests/recordings.parquet"
SEIZURES = ROOT / "manifests/seizures.parquet"
JOINT_CONDITIONS = ("band_joint_frozen", "band_joint_band_ttt", "mask_joint_frozen", "mask_joint_mask_ttt")
META_CONDITIONS = ("meta_band_frozen", "meta_band_ttt", "meta_temporal_frozen", "meta_temporal_ttt")
REQUESTED = JOINT_CONDITIONS + META_CONDITIONS
ONSET_PRE_S = 30.0
OFFSET_POST_S = 60.0

sys.path.insert(0, str(NEURO_CODE))
from bfa.evaluation.eventize import eventize  # noqa: E402
from bfa.evaluation.match import match_events  # noqa: E402


def interval_union(intervals: list[tuple[float, float]]) -> float:
    ordered = sorted((float(start), float(end)) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
    total = 0.0
    start, end = ordered[0]
    for left, right in ordered[1:]:
        if left <= end:
            end = max(end, right)
        else:
            total += end - start
            start, end = left, right
    return float(total + end - start)


def truths(seizures: pd.DataFrame, recording: str, evaluation_start: float, duration: float) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    scoring: list[tuple[float, float]] = []
    raw: list[tuple[float, float]] = []
    selected = seizures[seizures.recording_id.astype(str).eq(str(recording))]
    for row in selected.itertuples(index=False):
        raw_start, raw_end = float(row.start_s), float(row.end_s)
        if raw_end <= evaluation_start:
            continue
        clipped_start = max(evaluation_start, raw_start)
        clipped_end = min(duration, raw_end)
        collar_start = max(evaluation_start, raw_start - ONSET_PRE_S)
        collar_end = min(duration, raw_end + OFFSET_POST_S)
        if clipped_end > clipped_start and collar_end > collar_start:
            raw.append((clipped_start, clipped_end))
            scoring.append((collar_start, collar_end))
    return scoring, raw


def score_table(table: pd.DataFrame, seizures: pd.DataFrame, recordings: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    true_positive = false_alarm = truth_count = 0
    alarm_seconds = false_alarm_seconds = monitoring_seconds = nonseizure_seconds = 0.0
    delays: list[float] = []
    for recording, group in table.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        if len(ordered) < 2:
            continue
        evaluation_start = float(ordered.end.iloc[0])
        metadata = recordings[recordings.recording_id.astype(str).eq(str(recording))]
        if metadata.empty:
            raise ValueError(f"missing recording metadata: {recording}")
        duration = float(metadata.duration_s.iloc[0])
        scoring_truths, raw_truths = truths(seizures, str(recording), evaluation_start, duration)
        predictions = eventize(
            ordered.end.to_numpy(dtype=float),
            ordered.probability.to_numpy(dtype=float),
            threshold=float(threshold),
        )
        matched = match_events(predictions, scoring_truths)
        true_positive += len(matched.pairs)
        false_alarm += len(matched.unmatched_predictions)
        truth_count += len(scoring_truths)
        alarm_seconds += interval_union([(event.start_s, event.end_s) for event in predictions if event.end_s > evaluation_start and event.start_s < duration])
        for event in predictions:
            event_length = max(0.0, min(float(event.end_s), duration) - max(float(event.start_s), evaluation_start))
            overlap = interval_union([
                (max(float(event.start_s), start), min(float(event.end_s), end))
                for start, end in raw_truths
                if min(float(event.end_s), end) > max(float(event.start_s), start)
            ])
            false_alarm_seconds += max(0.0, event_length - overlap)
        for pair in matched.pairs:
            delays.append(max(0.0, float(predictions[pair.prediction_index].start_s) - raw_truths[pair.truth_index][0]))
        raw_seizure_seconds = interval_union(raw_truths)
        monitoring_seconds += max(0.0, duration - evaluation_start)
        nonseizure_seconds += max(0.0, duration - evaluation_start - raw_seizure_seconds)
    return {
        "true_positive_events": int(true_positive),
        "false_alarm_events": int(false_alarm),
        "truth_events": int(truth_count),
        "alarm_time_seconds": float(alarm_seconds),
        "false_alarm_time_seconds": float(false_alarm_seconds),
        "total_monitoring_seconds": float(monitoring_seconds),
        "nonseizure_seconds": float(nonseizure_seconds),
        "event_sensitivity": float(true_positive / truth_count) if truth_count else float("nan"),
        "fa_per_24h": float(false_alarm * 86400.0 / nonseizure_seconds) if nonseizure_seconds else float("nan"),
        "alarm_time_pct": float(100.0 * alarm_seconds / monitoring_seconds) if monitoring_seconds else float("nan"),
        "false_alarm_time_pct": float(100.0 * false_alarm_seconds / monitoring_seconds) if monitoring_seconds else float("nan"),
        "false_alarm_time_min_per_24h": float(false_alarm_seconds / 60.0 * 86400.0 / monitoring_seconds) if monitoring_seconds else float("nan"),
        "detection_delay_mean_s": float(np.mean(delays)) if delays else float("nan"),
    }


def condition_root(condition: str) -> Path:
    return JOINT_ROOT if condition in JOINT_CONDITIONS or condition == "supervised_frozen" else META_ROOT


def status_for(condition: str) -> dict[str, object]:
    root = condition_root(condition)
    folds: list[dict[str, object]] = []
    for fold in range(5):
        run = root / "evaluation" / condition / f"fold{fold}_seed3407"
        validation = run / "validation_metrics.json"
        test = run / "test_completed.json"
        probabilities = run / "test_probabilities.parquet"
        folds.append({"fold": fold, "validation": validation.is_file(), "test": test.is_file() and probabilities.is_file()})
    return {"condition": condition, "root": str(root), "validation_folds": int(sum(bool(row["validation"]) for row in folds)), "test_folds": int(sum(bool(row["test"]) for row in folds)), "complete_test": all(bool(row["test"]) for row in folds), "folds": folds}


def score_condition(condition: str, recordings: pd.DataFrame, seizures: pd.DataFrame) -> dict[str, object] | None:
    root = condition_root(condition)
    parts: list[pd.DataFrame] = []
    used_patients: set[str] = set()
    thresholds: list[float] = []
    for fold in range(5):
        run = root / "evaluation" / condition / f"fold{fold}_seed3407"
        validation_path = run / "validation_metrics.json"
        test_path = run / "test_probabilities.parquet"
        completion_path = run / "test_completed.json"
        if not (validation_path.is_file() and test_path.is_file() and completion_path.is_file()):
            return None
        validation = json.loads(validation_path.read_text())
        completion = json.loads(completion_path.read_text())
        if completion.get("test_evaluation_count") != 1 or completion.get("threshold_source") != "validation_only":
            raise ValueError(f"invalid test lock: {completion_path}")
        threshold = float(validation["selected_event_operating_point"]["threshold"])
        thresholds.append(threshold)
        table = pd.read_parquet(test_path)
        patients = set(table.patient.astype(str).unique())
        if used_patients & patients:
            raise ValueError(f"outer-test patient overlap for {condition} fold {fold}")
        used_patients.update(patients)
        parts.append(table)
    table = pd.concat(parts, ignore_index=True)
    if len(set(thresholds)) != 5:
        threshold_summary = ",".join(f"{value:.4f}" for value in thresholds)
    else:
        threshold_summary = f"{thresholds[0]:.4f}"
    metrics = score_table(table, seizures, recordings, thresholds[0]) if len(set(thresholds)) == 1 else None
    if metrics is None:
        # Score each fold at its own validation-locked threshold, preserving
        # the condition's exact threshold discipline.
        metrics = {key: 0.0 for key in ("true_positive_events", "false_alarm_events", "truth_events", "alarm_time_seconds", "false_alarm_time_seconds", "total_monitoring_seconds", "nonseizure_seconds")}
        metrics["detection_delay_mean_s"] = float("nan")
        delay_sum = 0.0
        delay_count = 0
        for fold, threshold in enumerate(thresholds):
            run = root / "evaluation" / condition / f"fold{fold}_seed3407"
            fold_metrics = score_table(pd.read_parquet(run / "test_probabilities.parquet"), seizures, recordings, threshold)
            for key in ("true_positive_events", "false_alarm_events", "truth_events", "alarm_time_seconds", "false_alarm_time_seconds", "total_monitoring_seconds", "nonseizure_seconds"):
                metrics[key] += fold_metrics[key]
            if np.isfinite(float(fold_metrics["detection_delay_mean_s"])):
                count = int(fold_metrics["true_positive_events"])
                delay_sum += float(fold_metrics["detection_delay_mean_s"]) * count
                delay_count += count
        metrics["event_sensitivity"] = float(metrics["true_positive_events"] / metrics["truth_events"]) if metrics["truth_events"] else float("nan")
        metrics["fa_per_24h"] = float(metrics["false_alarm_events"] * 86400.0 / metrics["nonseizure_seconds"]) if metrics["nonseizure_seconds"] else float("nan")
        metrics["alarm_time_pct"] = float(100.0 * metrics["alarm_time_seconds"] / metrics["total_monitoring_seconds"]) if metrics["total_monitoring_seconds"] else float("nan")
        metrics["false_alarm_time_pct"] = float(100.0 * metrics["false_alarm_time_seconds"] / metrics["total_monitoring_seconds"]) if metrics["total_monitoring_seconds"] else float("nan")
        metrics["false_alarm_time_min_per_24h"] = float(metrics["false_alarm_time_seconds"] / 60.0 * 86400.0 / metrics["total_monitoring_seconds"]) if metrics["total_monitoring_seconds"] else float("nan")
        metrics["detection_delay_mean_s"] = delay_sum / delay_count if delay_count else float("nan")
    return {"condition": condition, "patients": len(used_patients), "records": int(table.recording.nunique()), "thresholds": threshold_summary, **metrics}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recordings = pd.read_parquet(RECORDINGS)
    seizures = pd.read_parquet(SEIZURES)
    statuses = [status_for(condition) for condition in REQUESTED]
    # Include the common baseline in the report when it is complete, so the
    # eight requested conditions have a directly interpretable reference.
    baseline_status = status_for("supervised_frozen")
    result_rows: list[dict[str, object]] = []
    if baseline_status["complete_test"]:
        baseline = score_condition("supervised_frozen", recordings, seizures)
        if baseline is not None:
            baseline["condition"] = "supervised_frozen_reference"
            result_rows.append(baseline)
    for condition in REQUESTED:
        scored = score_condition(condition, recordings, seizures)
        if scored is not None:
            result_rows.append(scored)
    status_frame = pd.DataFrame(statuses)
    status_frame.to_json(OUT / "status.json", orient="records", indent=2)
    if result_rows:
        pd.DataFrame(result_rows).to_csv(OUT / "condition_metrics.csv", index=False)
    manifest = {
        "release_id": "neurottt-eight-alarm-time-v1",
        "status": "complete" if all(row["complete_test"] for row in statuses) else "incomplete_test_outputs",
        "requested_conditions": list(REQUESTED),
        "reference_condition": "supervised_frozen",
        "event_matching_collar": {"onset_pre_s": ONSET_PRE_S, "offset_post_s": OFFSET_POST_S},
        "alarm_time_definition": "union of predicted alarm intervals clipped to evaluable monitoring interval",
        "false_alarm_time_definition": "predicted alarm union minus raw seizure union",
        "denominator": "total monitoring time after first evaluable window",
        "status_rows": statuses,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": manifest["status"], "status_rows": statuses, "scored": result_rows}, indent=2, sort_keys=True, default=float), flush=True)


if __name__ == "__main__":
    main()
