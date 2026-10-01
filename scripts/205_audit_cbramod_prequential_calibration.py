"""Prequential, prior-record-only calibration audit for same-patient adaptation.

This is a new, non-overwriting analysis of the completed formal CBraMod
same-patient runs.  It does not rerun a model.  The adaptation trajectory and
probabilities are read from the frozen formal outputs; only the operating
threshold is changed according to a rule that is causal at recording level:

* before a recording is scored, use only *completed earlier recordings*;
* require at least two previously observed seizure events and one hour of
  non-seizure time before fitting a threshold;
* choose the smallest-FP threshold with event sensitivity >= 0.80 on that
  prior buffer; otherwise use the frozen source threshold (0.01);
* score the complete current recording with that already-frozen threshold;
* only after scoring the recording may its labels enter the calibration buffer.

This is intentionally labelled an online target-patient calibration audit.
It is not an independent generalization result.  For TTT, model adaptation
remains unlabeled, but the operating threshold is label-calibrated from prior
recordings; the audit therefore separates model adaptation from calibration.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bfa.evaluation.eventize import eventize
from bfa.evaluation.match import match_events

ROOT = Path("/root/b_false_alarm_atlas")
SOURCE_NS = "cbramod-chb-same-patient-online-adaptation-formal-v1"
NS = os.environ.get("CBRAMOD_CALIBRATION_NAMESPACE", "cbramod-chb-same-patient-prequential-calibration-v2")
SOURCE = ROOT / "outputs/reports" / SOURCE_NS
OUT = ROOT / "outputs/reports" / NS
RECORDINGS = ROOT / "manifests/recordings.parquet"
SEIZURES = ROOT / "manifests/seizures.parquet"
WINDOWS = ROOT / "manifests/windows.parquet"
SOURCE_THRESHOLD = 0.01
TARGET_SENSITIVITY = 0.80
MIN_PRIOR_SEIZURES = 2
MIN_PRIOR_NONSEIZURE_HOURS = 1.0
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(tmp, path)


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recordings = pd.read_parquet(RECORDINGS)
    seizures = pd.read_parquet(SEIZURES)
    windows = pd.read_parquet(WINDOWS)
    return recordings, seizures, windows


def truths_for(patient: str, recording: str, seizures: pd.DataFrame) -> list[tuple[float, float]]:
    rows = seizures[(seizures.patient_id == patient) & (seizures.recording_id == recording)]
    return [(float(r.start_s), float(r.end_s)) for r in rows.itertuples(index=False)]


def record_score(
    rows: pd.DataFrame,
    patient: str,
    recording: str,
    threshold: float,
    recordings: pd.DataFrame,
    seizures: pd.DataFrame,
) -> dict[str, float | int]:
    if rows.empty:
        return {"tp": 0, "fp": 0, "truth": 0, "nonseizure_hours": 0.0}
    rec = recordings[recordings.recording_id == recording]
    if rec.empty:
        raise RuntimeError(f"recording missing from manifest: {recording}")
    duration = float(rec.duration_s.iloc[0])
    truths = truths_for(patient, recording, seizures)
    predictions = eventize(
        rows.time_s.to_numpy(dtype=float),
        rows.probability.to_numpy(dtype=float),
        threshold=float(threshold),
    )
    matched = match_events(predictions, truths)
    seizure_seconds = sum(max(0.0, end - start) for start, end in truths)
    nonseizure_hours = max(0.0, duration - 60.0 - seizure_seconds) / 3600.0
    return {
        "tp": int(len(matched.pairs)),
        "fp": int(len(matched.unmatched_predictions)),
        "truth": int(len(truths)),
        "nonseizure_hours": float(nonseizure_hours),
    }


def aggregate(
    rows: pd.DataFrame,
    patient: str,
    threshold: float,
    recordings: pd.DataFrame,
    seizures: pd.DataFrame,
) -> dict[str, float | int]:
    total = {"tp": 0, "fp": 0, "truth": 0, "nonseizure_hours": 0.0}
    for recording, group in rows.groupby("recording", sort=False):
        s = record_score(group.sort_values("time_s"), patient, str(recording), threshold, recordings, seizures)
        for key in total:
            total[key] += s[key]
    sens = total["tp"] / total["truth"] if total["truth"] else float("nan")
    fa = total["fp"] * 24.0 / total["nonseizure_hours"] if total["nonseizure_hours"] > 0 else float("nan")
    return {**total, "event_sensitivity": float(sens), "fa_per_24h": float(fa)}


def choose_threshold(
    prior: pd.DataFrame,
    patient: str,
    recordings: pd.DataFrame,
    seizures: pd.DataFrame,
) -> tuple[float, str, dict[str, Any]]:
    prior_truth = int(sum(len(truths_for(str(r), str(rec), seizures)) for r, rec in []))
    del prior_truth
    truth_count = 0
    for recording in prior.recording.astype(str).unique():
        truth_count += len(truths_for(patient, recording, seizures))
    prior_summary = aggregate(prior, patient, SOURCE_THRESHOLD, recordings, seizures)
    prior_hours = float(prior_summary["nonseizure_hours"])
    if truth_count < MIN_PRIOR_SEIZURES or prior_hours < MIN_PRIOR_NONSEIZURE_HOURS:
        return SOURCE_THRESHOLD, "frozen_source_fallback_insufficient_prior", {
            "prior_truth_events": truth_count,
            "prior_nonseizure_hours": prior_hours,
            "selected": SOURCE_THRESHOLD,
        }
    candidates: list[tuple[float, dict[str, Any]]] = []
    for threshold in THRESHOLDS:
        score = aggregate(prior, patient, float(threshold), recordings, seizures)
        if np.isfinite(score["event_sensitivity"]) and score["event_sensitivity"] >= TARGET_SENSITIVITY:
            candidates.append((float(threshold), score))
    if not candidates:
        best = min(
            ((float(t), aggregate(prior, patient, float(t), recordings, seizures)) for t in THRESHOLDS),
            key=lambda item: (-item[1]["event_sensitivity"], item[1]["fa_per_24h"], -item[0]),
        )
        return best[0], "prior_only_max_sensitivity_fallback", {
            "prior_truth_events": truth_count,
            "prior_nonseizure_hours": prior_hours,
            "selected": best[0],
            "selected_prior_score": best[1],
        }
    selected = min(candidates, key=lambda item: (item[1]["fa_per_24h"], -item[0]))
    return selected[0], "prior_only_target_sensitivity", {
        "prior_truth_events": truth_count,
        "prior_nonseizure_hours": prior_hours,
        "selected": selected[0],
        "selected_prior_score": selected[1],
    }


def run_method(
    method: str,
    patient: str,
    recordings: pd.DataFrame,
    seizures: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    path = SOURCE / "runs" / f"{method}__{patient}__seed17" / "probabilities.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_parquet(path)
    required = {"patient", "recording", "time_s", "probability", "scored_before_update"}
    if not required.issubset(table.columns):
        raise RuntimeError(f"missing columns in {path}: {sorted(required - set(table.columns))}")
    if not bool(table.scored_before_update.all()):
        raise RuntimeError(f"score-before-update audit failed: {path}")
    rec_order = recordings[recordings.patient_id == patient].sort_values("recording_id").recording_id.astype(str).tolist()
    prior_parts: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    total = {"tp": 0, "fp": 0, "truth": 0, "nonseizure_hours": 0.0}
    for rec_idx, recording in enumerate(rec_order):
        current = table[table.recording == recording].copy().sort_values("time_s")
        if current.empty:
            continue
        prior = pd.concat(prior_parts, ignore_index=True) if prior_parts else table.iloc[0:0].copy()
        threshold, rule, details = choose_threshold(prior, patient, recordings, seizures)
        scored = record_score(current, patient, recording, threshold, recordings, seizures)
        for key in total:
            total[key] += scored[key]
        rows.append({
            "patient_id": patient,
            "method": method,
            "recording_index": rec_idx,
            "recording": recording,
            "threshold_used": threshold,
            "threshold_rule": rule,
            "prior_truth_events": details["prior_truth_events"],
            "prior_nonseizure_hours": details["prior_nonseizure_hours"],
            "tp": scored["tp"],
            "fp": scored["fp"],
            "truth": scored["truth"],
            "nonseizure_hours": scored["nonseizure_hours"],
            "scored_before_any_current_recording_update": True,
        })
        # Labels from this completed recording enter the buffer only now.
        prior_parts.append(current)
    summary = {
        "patient_id": patient,
        "method": method,
        "tp": total["tp"],
        "fp": total["fp"],
        "truth": total["truth"],
        "nonseizure_hours": total["nonseizure_hours"],
        "event_sensitivity": total["tp"] / total["truth"] if total["truth"] else float("nan"),
        "fa_per_24h": total["fp"] * 24.0 / total["nonseizure_hours"] if total["nonseizure_hours"] > 0 else float("nan"),
        "source_probability_sha256": sha256(path),
        "source_run": str(path.relative_to(ROOT)),
    }
    return summary, pd.DataFrame(rows)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing output namespace: {OUT}")
    recordings, seizures, windows = load_sources()
    del windows
    patients = ["chb12", "chb02", "chb17"]
    methods = ["frozen", "ttt", "supervised_oracle"]
    OUT.mkdir(parents=True)
    summaries: list[dict[str, Any]] = []
    record_rows: list[pd.DataFrame] = []
    source_files: dict[str, str] = {}
    for patient in patients:
        for method in methods:
            summary, rows = run_method(method, patient, recordings, seizures)
            summaries.append(summary)
            record_rows.append(rows)
            source_files[f"{method}__{patient}"] = summary["source_run"]
    pd.DataFrame(summaries).to_csv(OUT / "prequential_calibration_summary.csv", index=False)
    pd.concat(record_rows, ignore_index=True).to_csv(OUT / "prequential_recording_trace.csv", index=False)
    payload = {
        "release_id": NS,
        "status": "complete_posthoc_causal_recording_calibration_audit",
        "source_namespace": SOURCE_NS,
        "source_manifest_sha256": sha256(SOURCE / "experiment_manifest.json"),
        "patients": patients,
        "methods": methods,
        "threshold_policy": {
            "source_fallback": SOURCE_THRESHOLD,
            "candidate_grid": THRESHOLDS.tolist(),
            "target_sensitivity": TARGET_SENSITIVITY,
            "minimum_prior_truth_events": MIN_PRIOR_SEIZURES,
            "minimum_prior_nonseizure_hours": MIN_PRIOR_NONSEIZURE_HOURS,
            "selection": "prior_completed_recordings only; minimum FA/24h among sensitivity-feasible candidates",
        },
        "protocol": {
            "current_recording_scored_before_labels_enter_calibration": True,
            "current_recording_threshold_frozen_before_scoring": True,
            "future_labels_used_for_current_recording": False,
            "model_training_rerun": False,
            "independent_generalization_claim": False,
            "ttt_label_use": "model remains unlabeled; labels are used only by this threshold-calibration audit",
            "supervised_oracle": "same source online adaptation probabilities; target labels were used after score in source run",
        },
        "source_probability_files": source_files,
        "output_sha256_after_write": {},
        "created_utc": now(),
    }
    atomic_json(OUT / "manifest.json", payload)
    payload["output_sha256_after_write"] = {
        "prequential_calibration_summary.csv": sha256(OUT / "prequential_calibration_summary.csv"),
        "prequential_recording_trace.csv": sha256(OUT / "prequential_recording_trace.csv"),
    }
    atomic_json(OUT / "manifest.json", payload)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(json.dumps({"status": payload["status"], "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
