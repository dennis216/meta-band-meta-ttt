#!/usr/bin/env python3
"""Causal continuous validation/test evaluator for the repaired release."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F

SCRIPT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_ROOT / "280_retrain_band_ttt_v2.py"
spec = importlib.util.spec_from_file_location("retrain_band_ttt_v2", TRAIN_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(TRAIN_SCRIPT)
retrain = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = retrain
spec.loader.exec_module(retrain)

EXTERNAL_ROOT = retrain.EXTERNAL_ROOT
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))
from chbmit_groupkfold.data import DEFAULT_CACHE, DEFAULT_FOLDS, DEFAULT_WINDOWS, load_rows, make_eval_loader  # noqa: E402
from bfa.evaluation.eventize import Event, eventize  # noqa: E402
from bfa.evaluation.match import match_events  # noqa: E402

SEIZURE_ONSET_PRE_S = 30.0
SEIZURE_OFFSET_POST_S = 60.0


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def all_records(rows: pd.DataFrame) -> list[retrain.Record]:
    output: list[retrain.Record] = []
    for (patient, recording), group in rows.groupby(["patient", "recording"], sort=False):
        group = group.sort_values("start", kind="stable").reset_index(drop=True)
        output.append(retrain.Record(
            patient=str(patient), recording=str(recording), relative_path=str(group.relative_path.iloc[0]), rows=group,
            starts=group.start.to_numpy(dtype=np.float64), labels=group.label.to_numpy(dtype=np.float32),
            sample_ids=group.sample_id.astype(str).to_numpy(), candidate_starts=np.empty(0, dtype=np.int64),
            seizure_record=bool((group.label.astype(int) == 1).any()),
        ))
    return output


def load_checkpoint(args: argparse.Namespace, checkpoint: Path, device: torch.device) -> tuple[retrain.CHBJointModel, retrain.LearnedAuxiliaryLoss | None, str]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("release_id") != retrain.RELEASE_ID:
        raise RuntimeError(f"wrong release checkpoint: {checkpoint}")
    if int(payload.get("fold", -1)) != args.fold or payload.get("branch") != args.branch:
        raise RuntimeError(f"checkpoint identity mismatch: {checkpoint}")
    model, _, _ = retrain.load_common_model(args, args.fold, prepared_band=True, device=device)
    model.load_state_dict(payload["model"], strict=True)
    objective = None if args.branch == "band" else retrain.LearnedAuxiliaryLoss().to(device)
    if objective is not None:
        objective.load_state_dict(payload["objective"], strict=True)
        objective.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    classifier_hash = retrain.stable_classifier_hash(model)
    if classifier_hash != payload.get("classifier_hash"):
        raise RuntimeError("checkpoint classifier hash does not match current model")
    return model, objective, classifier_hash


def recording_metadata(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    recordings = pd.read_parquet(args.recordings)
    recordings = recordings.rename(columns={"recording_id": "recording_id"})
    seizures = pd.read_parquet(args.seizures)
    return recordings, seizures


def union_length(intervals: list[tuple[float, float]]) -> float:
    ordered = sorted((float(left), float(right)) for left, right in intervals if right > left)
    total = 0.0
    if not ordered:
        return total
    left, right = ordered[0]
    for next_left, next_right in ordered[1:]:
        if next_left <= right:
            right = max(right, next_right)
        else:
            total += right - left
            left, right = next_left, next_right
    return total + right - left


def truths_for(recording: str, evaluation_start: float, recordings: pd.DataFrame, seizures: pd.DataFrame) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float]:
    match = recordings[recordings.recording_id.astype(str) == str(recording)]
    if match.empty:
        raise ValueError(f"missing recording metadata: {recording}")
    duration = float(match.duration_s.iloc[0])
    expanded: list[tuple[float, float]] = []
    raw: list[tuple[float, float]] = []
    for row in seizures[seizures.recording_id.astype(str) == str(recording)].itertuples(index=False):
        start = float(row.start_s)
        end = float(row.end_s)
        if end <= evaluation_start:
            continue
        raw_start = max(evaluation_start, start)
        raw_end = min(duration, end)
        expanded_start = max(evaluation_start, start - SEIZURE_ONSET_PRE_S)
        expanded_end = min(duration, end + SEIZURE_OFFSET_POST_S)
        if raw_end > raw_start and expanded_end > expanded_start:
            raw.append((raw_start, raw_end))
            expanded.append((expanded_start, expanded_end))
    return expanded, raw, duration


def score_table(table: pd.DataFrame, recordings: pd.DataFrame, seizures: pd.DataFrame, threshold: float) -> dict[str, Any]:
    true_positives = false_alarms = truth_events = 0
    false_alarm_seconds = 0.0
    total_seconds = nonseizure_seconds = 0.0
    delays: list[float] = []
    for recording, group in table.groupby("recording", sort=False):
        ordered = group.sort_values("end", kind="stable")
        eval_start = float(ordered.end.iloc[0])
        expanded, raw_truths, duration = truths_for(str(recording), eval_start, recordings, seizures)
        predictions_raw = eventize(ordered.end.to_numpy(dtype=float), ordered.probability.to_numpy(dtype=float), threshold=float(threshold))
        predictions = [Event(max(0.0, event.start_s), min(duration, event.end_s), event.peak_probability) for event in predictions_raw if min(duration, event.end_s) > max(0.0, event.start_s)]
        matched = match_events(predictions, expanded)
        true_positives += len(matched.pairs)
        false_alarms += len(matched.unmatched_predictions)
        truth_events += len(expanded)
        truth_union = union_length(raw_truths)
        for prediction in predictions:
            overlap = union_length([(max(prediction.start_s, left), min(prediction.end_s, right)) for left, right in raw_truths if min(prediction.end_s, right) > max(prediction.start_s, left)])
            false_alarm_seconds += max(0.0, prediction.end_s - prediction.start_s - overlap)
        for pair in matched.pairs:
            delays.append(max(0.0, predictions[pair.prediction_index].start_s - raw_truths[pair.truth_index][0]))
        total_seconds += max(0.0, duration - eval_start)
        nonseizure_seconds += max(0.0, duration - eval_start - truth_union)
    total_hours = total_seconds / 3600.0
    nonseizure_hours = nonseizure_seconds / 3600.0
    return {
        "threshold": float(threshold), "true_positive_events": int(true_positives), "false_alarm_events": int(false_alarms), "truth_events": int(truth_events),
        "event_sensitivity": float(true_positives / truth_events) if truth_events else 1.0,
        "false_alarm_time_seconds": float(false_alarm_seconds),
        "false_alarm_time_min_per_24h": float(false_alarm_seconds / 60.0 * 24.0 / total_hours) if total_hours else 0.0,
        "false_alarm_time_s_per_24h": float(false_alarm_seconds * 24.0 / total_hours) if total_hours else 0.0,
        "fa_per_24h": float(false_alarms * 24.0 / nonseizure_hours) if nonseizure_hours else 0.0,
        "total_monitoring_hours": float(total_hours), "nonseizure_hours": float(nonseizure_hours),
        "detection_delay_mean_s": float(np.mean(delays)) if delays else float("nan"),
        "detection_delay_median_s": float(np.median(delays)) if delays else float("nan"),
        "detection_delay_count": int(len(delays)),
    }


def window_metrics(table: pd.DataFrame) -> dict[str, float]:
    y = table.label.to_numpy(dtype=np.float32)
    p = table.probability.to_numpy(dtype=np.float32)
    return {"window_auprc": float(average_precision_score(y, p)), "window_auroc": float(roc_auc_score(y, p))}


def atomic_parquet(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_parquet(temporary, index=False)
    os.replace(temporary, path)


@torch.inference_mode()
def frozen_table(args: argparse.Namespace, rows: pd.DataFrame, model: retrain.CHBJointModel, split_dir: Path, device: torch.device) -> pd.DataFrame:
    output_path = split_dir / "frozen_probabilities.parquet"
    if output_path.is_file() and not args.force:
        return pd.read_parquet(output_path)
    loader = make_eval_loader(rows, batch_size=args.eval_batch, workers=args.workers, cache_root=args.cache_root)
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    started = time.monotonic()
    processed = 0
    for signal, target, _ in loader:
        signal = signal.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model.detect(signal)
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(target.numpy())
        processed += len(target)
        if processed % 100_000 < len(target):
            atomic_json(split_dir / "progress.json", {"status": "frozen", "split": args.split, "processed_windows": processed, "windows_per_s": processed / max(time.monotonic() - started, 1e-9), "elapsed_s": time.monotonic() - started})
    table = rows[["patient", "recording", "start", "end", "sample_id"]].copy()
    table["label"] = rows.label.to_numpy(dtype=np.float32)
    table["probability"] = np.concatenate(probabilities).astype(np.float32)
    table = table.reset_index(drop=True)
    if not np.array_equal(table.label.to_numpy(dtype=np.float32), np.concatenate(labels).astype(np.float32)):
        raise RuntimeError("frozen loader order does not match sorted manifest")
    atomic_parquet(output_path, table)
    atomic_json(split_dir / "frozen_completed.json", {"status": "completed", "split": args.split, "rows": len(table), "elapsed_s": time.monotonic() - started})
    return table


def ttt_table(args: argparse.Namespace, rows: pd.DataFrame, checkpoint: Path, candidate_dir: Path, device: torch.device) -> pd.DataFrame:
    output_path = candidate_dir / f"{args.split}_ttt_probabilities.parquet"
    if output_path.is_file() and not args.force:
        return pd.read_parquet(output_path)
    model, objective_head, classifier_hash = load_checkpoint(args, checkpoint, device)
    records = all_records(rows)
    store = retrain.SignalStore(args.cache_root)
    cache = retrain.PrefixCache(args.prefix_cache, classifier_hash, args.prefix_cache_bytes, not args.no_prefix_cache)
    pieces: list[pd.DataFrame] = []
    started = time.monotonic()
    processed = updates = 0
    last_progress = started
    for record_index, record in enumerate(records):
        params = retrain.detached_tail_values(model, requires_grad=True)
        count = len(record.rows)
        for left in range(0, count, retrain.CHUNK_SIZE):
            right = min(count, left + retrain.CHUNK_SIZE)
            positions = np.arange(left, right, dtype=np.int64)
            raw_prefix, transformed_prefix, labels, _, sample_ids = retrain._episode_chunk(store, record, positions, device, cache, model)
            with torch.enable_grad():
                logits = retrain.detector_from_prefix(model, raw_prefix, params)
            probabilities = torch.sigmoid(logits.detach().float()).cpu().numpy()
            group = record.rows.iloc[positions][["patient", "recording", "start", "end", "label", "sample_id"]].copy()
            group["probability"] = probabilities.astype(np.float32)
            group["adapted"] = bool(updates > 0 or left > 0)
            pieces.append(group)
            processed += len(positions)
            if right < count:
                # This is the only update for the chunk.  The classification
                # label is never passed to the inner loss.
                with torch.enable_grad():
                    params, _, _, _ = retrain.clipped_inner_update(model, objective_head, raw_prefix, transformed_prefix, labels, params, args.branch, create_graph=False)
                params = {name: value.detach().requires_grad_(True) for name, value in params.items()}
                updates += 1
            now = time.monotonic()
            if now - last_progress >= 30:
                atomic_json(candidate_dir / "progress.json", {
                    "status": "ttt", "split": args.split, "record_index": record_index,
                    "record": record.recording, "patient": record.patient, "record_windows_done": right,
                    "processed_windows": processed, "updates": updates,
                    "windows_per_s": processed / max(now - started, 1e-9), "updates_per_s": updates / max(now - started, 1e-9),
                    "elapsed_s": now - started,
                })
                last_progress = now
        del params
    table = pd.concat(pieces, ignore_index=True)
    table = table.sort_values(["patient", "recording", "start"], kind="stable").reset_index(drop=True)
    if len(table) != len(rows) or not np.array_equal(table.sample_id.astype(str).to_numpy(), rows.sample_id.astype(str).to_numpy()):
        raise RuntimeError("TTT output order does not match input rows")
    atomic_parquet(output_path, table)
    atomic_json(candidate_dir / "completed.json", {"status": "completed", "split": args.split, "rows": len(table), "updates": updates, "elapsed_s": time.monotonic() - started, "checkpoint": str(checkpoint)})
    return table


def select_threshold(table: pd.DataFrame, recordings: pd.DataFrame, seizures: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    sweep = [score_table(table, recordings, seizures, threshold) for threshold in np.round(np.arange(0.01, 1.0, 0.01), 2)]
    feasible = [row for row in sweep if row["event_sensitivity"] >= 0.80]
    if feasible:
        selected = min(feasible, key=lambda row: (row["false_alarm_time_min_per_24h"], row["fa_per_24h"], row["detection_delay_mean_s"] if np.isfinite(row["detection_delay_mean_s"]) else float("inf"), -row["threshold"]))
    else:
        selected = min(sweep, key=lambda row: (-row["event_sensitivity"], row["false_alarm_time_min_per_24h"], row["fa_per_24h"], -row["threshold"]))
    return pd.DataFrame(sweep), selected


def metric_bundle(table: pd.DataFrame, recordings: pd.DataFrame, seizures: pd.DataFrame, threshold: float) -> dict[str, Any]:
    return {**window_metrics(table), **score_table(table, recordings, seizures, threshold)}


def gate_against_frozen(ttt: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    fa_base = float(frozen["false_alarm_time_min_per_24h"])
    fa_ttt = float(ttt["false_alarm_time_min_per_24h"])
    fa_pass = fa_ttt <= fa_base * 0.95 if fa_base > 0 else fa_ttt <= 0.0
    sensitivity_pass = float(ttt["event_sensitivity"]) >= float(frozen["event_sensitivity"])
    events_pass = float(ttt["fa_per_24h"]) <= float(frozen["fa_per_24h"]) * 1.05
    base_delay = float(frozen["detection_delay_mean_s"])
    ttt_delay = float(ttt["detection_delay_mean_s"])
    delay_pass = (not np.isfinite(base_delay) and not np.isfinite(ttt_delay)) or (np.isfinite(ttt_delay) and ttt_delay <= base_delay + 2.0)
    return {
        "passed": bool(fa_pass and sensitivity_pass and events_pass and delay_pass),
        "false_alarm_minutes_pass": bool(fa_pass), "sensitivity_pass": bool(sensitivity_pass),
        "false_alarm_events_pass": bool(events_pass), "delay_pass": bool(delay_pass),
        "requirements": {"false_alarm_ratio_max": 0.95, "sensitivity_min": float(frozen["event_sensitivity"]), "fa_per_24h_ratio_max": 1.05, "delay_increase_max_s": 2.0},
        "observed": {"false_alarm_ratio": fa_ttt / fa_base if fa_base else (0.0 if fa_ttt == 0 else float("inf")), "sensitivity": float(ttt["event_sensitivity"]), "fa_per_24h_ratio": float(ttt["fa_per_24h"]) / float(frozen["fa_per_24h"]) if frozen["fa_per_24h"] else (0.0 if ttt["fa_per_24h"] == 0 else float("inf")), "delay_delta_s": ttt_delay - base_delay if np.isfinite(ttt_delay) and np.isfinite(base_delay) else float("nan")},
    }


def validation_run(args: argparse.Namespace) -> dict[str, Any]:
    eval_dir = args.output_root / "evaluations" / f"fold{args.fold}" / args.branch / "validation"
    selection_path = eval_dir / "validation_selection.json"
    if selection_path.is_file() and not args.force:
        return json.loads(selection_path.read_text())
    meta_dir = retrain.meta_run_dir(args.output_root, args.fold, args.branch, args.seed)
    candidates = [path for path in (meta_dir / "best_future.pt", meta_dir / "best_nonseizure.pt") if path.is_file()]
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        digest = sha256(candidate)
        if digest not in seen:
            unique.append(candidate)
            seen.add(digest)
    if not unique:
        raise FileNotFoundError(f"no meta candidates in {meta_dir}")
    device = torch.device(args.device)
    rows = load_rows(args.fold, "validation", args.windows, args.fold_root)
    recordings, seizures = recording_metadata(args)
    eval_dir.mkdir(parents=True, exist_ok=True)
    # The common Frozen validation result is computed once per fold and then
    # reused by both branches/candidates.
    base_model, _, _ = load_checkpoint(args, unique[0], device)
    frozen = frozen_table(args, rows, base_model, eval_dir.parent.parent / "frozen_validation", device)
    frozen_sweep, frozen_selected = select_threshold(frozen, recordings, seizures)
    atomic_parquet(eval_dir / "frozen_threshold_sweep.parquet", frozen_sweep)
    atomic_json(eval_dir / "frozen_metrics.json", {"selected": {**frozen_selected, **window_metrics(frozen)}, "common_threshold": frozen_selected["threshold"]})
    candidate_results: list[dict[str, Any]] = []
    for candidate in unique:
        candidate_name = candidate.stem
        candidate_dir = eval_dir / candidate_name
        model, objective, classifier_hash = load_checkpoint(args, candidate, device)
        table = ttt_table(args, rows, candidate, candidate_dir, device)
        sweep, selected = select_threshold(table, recordings, seizures)
        atomic_parquet(candidate_dir / "threshold_sweep.parquet", sweep)
        selected_full = {**selected, **window_metrics(table)}
        common_frozen_metrics = metric_bundle(frozen, recordings, seizures, float(selected["threshold"]))
        gate = gate_against_frozen(selected_full, frozen_selected)
        candidate_results.append({
            "candidate": candidate_name, "checkpoint": str(candidate), "checkpoint_sha256": sha256(candidate),
            "classifier_hash": classifier_hash, "selected_threshold": selected_full, "common_frozen_at_ttt_threshold": common_frozen_metrics,
            "frozen_selected": {**frozen_selected, **window_metrics(frozen)}, "gate": gate,
        })
        atomic_json(candidate_dir / "validation_metrics.json", candidate_results[-1])
        del model, objective
        if device.type == "cuda":
            torch.cuda.empty_cache()
    passing = [result for result in candidate_results if result["gate"]["passed"]]
    selected_result = min(passing, key=lambda result: (result["selected_threshold"]["false_alarm_time_min_per_24h"], result["selected_threshold"]["fa_per_24h"])) if passing else None
    selection = {
        "release_id": retrain.RELEASE_ID, "stage": "validation_selection", "fold": args.fold, "branch": args.branch,
        "seed": args.seed, "status": "test_allowed" if selected_result is not None else "gate_failed",
        "frozen_validation": {"table": str(eval_dir.parent.parent / "frozen_validation" / "frozen_probabilities.parquet"), "selected": {**frozen_selected, **window_metrics(frozen)}, "threshold_sweep": str(eval_dir / "frozen_threshold_sweep.parquet")},
        "candidates": candidate_results,
        "selected_candidate": selected_result,
        "test_allowed": selected_result is not None,
        "completed_at": retrain.utc_now(),
    }
    atomic_json(selection_path, selection)
    atomic_json(eval_dir / "completed.json", selection)
    return selection


def test_run(args: argparse.Namespace) -> dict[str, Any]:
    validation_dir = args.output_root / "evaluations" / f"fold{args.fold}" / args.branch / "validation"
    selection_path = validation_dir / "validation_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("test cannot start before validation lock")
    selection = json.loads(selection_path.read_text())
    if not selection.get("test_allowed", False):
        result = {"release_id": retrain.RELEASE_ID, "stage": "test", "fold": args.fold, "branch": args.branch, "status": "not_run_gate_failed", "validation_selection": str(selection_path)}
        atomic_json(args.output_root / "evaluations" / f"fold{args.fold}" / args.branch / "test_not_run.json", result)
        return result
    selected = selection["selected_candidate"]
    checkpoint = Path(selected["checkpoint"])
    if sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError("locked validation checkpoint changed before test")
    eval_dir = args.output_root / "evaluations" / f"fold{args.fold}" / args.branch / "test"
    completed_path = eval_dir / "test_completed.json"
    if completed_path.is_file() and not args.force:
        return json.loads(completed_path.read_text())
    device = torch.device(args.device)
    rows = load_rows(args.fold, "test", args.windows, args.fold_root)
    recordings, seizures = recording_metadata(args)
    model, objective, classifier_hash = load_checkpoint(args, checkpoint, device)
    frozen_dir = args.output_root / "evaluations" / f"fold{args.fold}" / "frozen_test"
    frozen = frozen_table(args, rows, model, frozen_dir, device)
    candidate_dir = eval_dir / selected["candidate"]
    table = ttt_table(args, rows, checkpoint, candidate_dir, device)
    threshold = float(selected["selected_threshold"]["threshold"])
    ttt_metrics = metric_bundle(table, recordings, seizures, threshold)
    frozen_at_locked = metric_bundle(frozen, recordings, seizures, threshold)
    frozen_sweep, frozen_selected = select_threshold(frozen, recordings, seizures)
    atomic_parquet(eval_dir / "frozen_threshold_sweep.parquet", frozen_sweep)
    result = {
        "release_id": retrain.RELEASE_ID, "stage": "test", "fold": args.fold, "branch": args.branch,
        "status": "completed", "validation_selection": str(selection_path), "candidate": selected,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "classifier_hash": classifier_hash,
        "locked_validation_threshold": threshold, "ttt_metrics": ttt_metrics,
        "frozen_at_locked_threshold": frozen_at_locked,
        "frozen_own_validation_threshold": {**frozen_selected, **window_metrics(frozen)},
        "table": str(candidate_dir / "test_ttt_probabilities.parquet"), "completed_at": retrain.utc_now(),
    }
    atomic_json(completed_path, result)
    atomic_json(eval_dir / "progress.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--branch", choices=("band", "learned"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--seed", type=int, default=retrain.SEED)
    parser.add_argument("--output-root", type=Path, default=SCRIPT_ROOT.parent / "results" / retrain.RELEASE_ID)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--prefix-cache", type=Path, default=Path("/mnt/d/EEGData/meta_ttt_prefix_v2"))
    parser.add_argument("--prefix-cache-bytes", type=int, default=128 * 2**30)
    parser.add_argument("--pretrained", type=Path, default=EXTERNAL_ROOT / "pretrained_weights/pretrained_weights.pth")
    parser.add_argument("--recordings", type=Path, default=Path("/root/b_false_alarm_atlas/manifests/recordings.parquet"))
    parser.add_argument("--seizures", type=Path, default=Path("/root/b_false_alarm_atlas/manifests/seizures.parquet"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--candidate-only", choices=("best_future", "best_nonseizure"),
                        help="materialize one validation candidate table without selecting or gating")
    return parser.parse_args()


def candidate_only_run(args: argparse.Namespace) -> dict[str, Any]:
    if args.split != "validation":
        raise ValueError("--candidate-only is restricted to validation")
    checkpoint = retrain.meta_run_dir(args.output_root, args.fold, args.branch, args.seed) / f"{args.candidate_only}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    rows = load_rows(args.fold, "validation", args.windows, args.fold_root)
    candidate_dir = args.output_root / "evaluations" / f"fold{args.fold}" / args.branch / "validation" / args.candidate_only
    table = ttt_table(args, rows, checkpoint, candidate_dir, torch.device(args.device))
    return {"status": "candidate_table_completed", "branch": args.branch,
            "candidate": args.candidate_only, "rows": len(table), "table": str(candidate_dir)}


def main() -> None:
    args = parse_args()
    if args.seed != retrain.SEED:
        raise ValueError("formal release locks seed=3407")
    if args.split == "test" and not args.allow_test:
        raise PermissionError("test requires --allow-test after validation lock")
    result = candidate_only_run(args) if args.candidate_only else (validation_run(args) if args.split == "validation" else test_run(args))
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
