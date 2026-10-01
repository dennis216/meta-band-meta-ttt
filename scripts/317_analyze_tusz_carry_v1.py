#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v1"
POSITION_BINS = ((0, 120, "0-2min"), (120, 300, "2-5min"), (300, 600, "5-10min"), (600, np.inf, "10min+"))


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def safe_metric(function, labels, scores):
    if len(np.unique(labels)) < 2:
        return None
    return float(function(labels, scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.probabilities)
    inventory = json.loads((OUT / "manifests/records.json").read_text())
    lookup = {
        (item["patient_id"], item["session_id"], item["montage"], item["record_id"]): item
        for item in inventory
    }
    keys = ["patient_id", "session_id", "montage", "record_id"]
    enriched = []
    reset_max = 0.0
    for key, group in frame.groupby(keys, sort=False):
        item = lookup[tuple(key)]
        group = group.sort_values("decision_end_s").copy()
        times = group["decision_end_s"].to_numpy(float)
        frozen = group["frozen_probability"].to_numpy(float)
        adapted = group["adapted_probability"].to_numpy(float)
        if len(group):
            reset_max = max(reset_max, float(np.max(np.abs(adapted[:15] - frozen[:15]))))
        labels = group["label"].to_numpy(float)
        context = np.full(len(group), "other", dtype=object)
        for onset, offset in item["seizures"]:
            context[(labels == 0) & (times > offset) & (times <= offset + 120)] = "post_seizure_background"
            context[(labels == 0) & (times >= onset - 120) & (times < onset)] = "pre_seizure_background"
        group["context"] = context
        group["delta_probability"] = adapted - frozen
        group["delta_logit"] = logit(adapted) - logit(frozen)
        group["update_count"] = np.arange(len(group)) // 15
        enriched.append(group)
    frame = pd.concat(enriched, ignore_index=True)
    rows = []
    for start, end, label in POSITION_BINS:
        subset = frame[(frame["decision_end_s"] >= start) & (frame["decision_end_s"] < end)]
        for category, selected in (
            ("all", subset),
            ("seizure", subset[subset["label"] > 0]),
            ("background", subset[subset["label"] == 0]),
        ):
            labels = (selected["label"].to_numpy(float) > 0).astype(int)
            adapted = selected["adapted_probability"].to_numpy(float)
            rows.append({
                "position": label,
                "category": category,
                "windows": len(selected),
                "mean_delta_probability": float(selected["delta_probability"].mean()) if len(selected) else None,
                "mean_delta_logit": float(selected["delta_logit"].mean()) if len(selected) else None,
                "adapted_auprc": safe_metric(average_precision_score, labels, adapted),
                "adapted_auroc": safe_metric(roc_auc_score, labels, adapted),
            })
    context_rows = []
    for context in ("post_seizure_background", "pre_seizure_background", "other"):
        subset = frame[(frame["label"] == 0) & (frame["context"] == context)]
        context_rows.append({
            "context": context,
            "windows": len(subset),
            "mean_delta_probability": float(subset["delta_probability"].mean()) if len(subset) else None,
            "mean_delta_logit": float(subset["delta_logit"].mean()) if len(subset) else None,
        })
    output = {
        "name": args.name,
        "probabilities": str(args.probabilities.resolve()),
        "record_reset_first_chunk_max_abs_delta": reset_max,
        "position": rows,
        "context": context_rows,
    }
    destination = OUT / "mechanisms" / args.name
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "carry_stability.json").write_text(json.dumps(output, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(destination / "carry_position.csv", index=False)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
