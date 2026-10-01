#!/usr/bin/env python3
"""Evaluate S1 or continued-supervision checkpoints without constructing a TTT objective."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bfa.tusz_meta_ttt.dataset import filter_inventory_records, load_cached_arrays
from bfa.tusz_meta_ttt.scoring import choose_threshold, score_at_threshold
from bfa.tusz_meta_ttt_v2.protocol import chunk_rows
from bfa.tusz_meta_ttt_v2.runtime import load_source, make_windows

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition", choices=["train", "dev", "eval"], required=True)
    parser.add_argument(
        "--cohort", choices=["all", "development_fit", "development_validation"], default="all"
    )
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--threshold-summary", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.calibrate == bool(args.threshold_summary):
        raise ValueError("choose exactly one of --calibrate or --threshold-summary")
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    source = Path(checkpoint.get("source", args.checkpoint))
    model = load_source(source)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False).eval()
    inventory_rows = json.loads((V1 / "manifests/records.json").read_text())
    lookup = {
        (row["patient_id"], row["session_id"], row["montage"], row["record_id"]): row
        for row in inventory_rows
    }
    paths = filter_inventory_records(
        sorted((V1 / "cache" / args.partition).rglob("*.npz")), inventory_rows,
        partition=args.partition,
    )
    if args.cohort != "all":
        if args.partition != "train":
            raise ValueError("development cohorts exist only in Train")
        patients = set(json.loads((V1 / "manifests/development_split.json").read_text())[args.cohort])
        paths = [path for path in paths if path.parts[-4] in patients]
    if args.limit is not None:
        paths = paths[: args.limit]
    probability_rows = []
    scoring_records = []
    for record_index, path in enumerate(paths, 1):
        archive = load_cached_arrays(path)
        scores = np.empty(len(archive["labels"]), dtype=np.float32)
        for chunk in chunk_rows(archive["decision_end_s"]):
            with torch.no_grad():
                scores[list(chunk.rows)] = torch.sigmoid(
                    model(make_windows(archive["signal"], chunk.rows))
                ).cpu().numpy()
        key = (path.parts[-4], path.parts[-3], path.parts[-2], path.stem)
        item = lookup[key]
        scoring_records.append({
            "times": archive["decision_end_s"],
            "probabilities": scores,
            "truths": item["seizures"],
            "duration_s": item["duration_s"],
        })
        for index, time_s in enumerate(archive["decision_end_s"]):
            probability_rows.append({
                "partition": args.partition,
                "patient_id": key[0],
                "session_id": key[1],
                "montage": key[2],
                "record_id": key[3],
                "decision_end_s": float(time_s),
                "label": float(archive["labels"][index]),
                "probability": float(scores[index]),
            })
        print(json.dumps({"record": record_index, "total": len(paths)}), flush=True)
    destination = OUT / "evaluation/detectors" / args.tag / (
        args.partition if args.cohort == "all" else args.cohort
    )
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(probability_rows).to_parquet(destination / "probabilities.parquet", index=False)
    if args.calibrate:
        choice = choose_threshold(scoring_records)
        result = {
            "reachable": choice.reachable,
            "threshold": choice.threshold,
            "metrics": vars(choice.metrics) if choice.metrics else None,
            "maximum_sensitivity_threshold": choice.maximum_sensitivity_threshold,
            "maximum_sensitivity_metrics": (
                vars(choice.maximum_sensitivity_metrics)
                if choice.maximum_sensitivity_metrics else None
            ),
        }
    else:
        calibration = json.loads(args.threshold_summary.read_text())
        threshold = calibration["threshold"]
        result = {
            "reachable_on_dev": threshold is not None,
            "threshold": threshold,
            "metrics": (
                vars(score_at_threshold(scoring_records, threshold))
                if threshold is not None else None
            ),
        }
    (destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
