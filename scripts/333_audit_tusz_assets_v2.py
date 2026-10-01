#!/usr/bin/env python3
"""Verify reused S1 checkpoints, partition isolation, and v1 cache invariants."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from bfa.tusz_meta_ttt.dataset import load_cached_arrays
from bfa.tusz_meta_ttt_v2.runtime import load_source, make_windows

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--maximum-cache-records", type=int)
    args = parser.parse_args()
    inventory = json.loads((V1 / "manifests/records.json").read_text())
    patients = {
        partition: {row["patient_id"] for row in inventory if row["partition"] == partition}
        for partition in ("train", "dev", "eval")
    }
    overlaps = {
        "train_dev": sorted(patients["train"] & patients["dev"]),
        "train_eval": sorted(patients["train"] & patients["eval"]),
        "dev_eval": sorted(patients["dev"] & patients["eval"]),
    }
    source_rows = []
    for source_path in args.sources:
        source_path = source_path.resolve()
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        model = load_source(source_path)
        sample_path = next((V1 / "cache/train").rglob("*.npz"))
        archive = load_cached_arrays(sample_path)
        batch = make_windows(archive["signal"], (0,))
        with torch.no_grad():
            first = model(batch)
            second = model(batch)
        source_rows.append({
            "path": str(source_path),
            "sha256": sha256(source_path),
            "epoch": checkpoint.get("epoch"),
            "pretrained_source_hash": checkpoint.get("source_hash"),
            "deterministic_forward_max_abs": float((first - second).abs().max()),
            "finite_forward": bool(torch.isfinite(first).all()),
            "input_shape": list(batch.shape),
        })
        del model, batch
        torch.cuda.empty_cache()

    expected = {
        (row["partition"], row["patient_id"], row["session_id"], row["montage"], row["record_id"])
        for row in inventory if row["exclusion"] is None
    }
    excluded = {
        (row["partition"], row["patient_id"], row["session_id"], row["montage"], row["record_id"]): row["exclusion"]
        for row in inventory if row["exclusion"] is not None
    }
    cache_paths = sorted((V1 / "cache").rglob("*.npz"))
    if args.maximum_cache_records is not None:
        cache_paths = cache_paths[: args.maximum_cache_records]
    observed = set()
    invalid = []
    total_windows = 0
    for path in cache_paths:
        key = (path.parts[-5], path.parts[-4], path.parts[-3], path.parts[-2], path.stem)
        observed.add(key)
        archive = load_cached_arrays(path)
        signal = archive["signal"]
        times = archive["decision_end_s"]
        labels = archive["labels"]
        reasons = []
        if signal.dtype != np.float32 or signal.ndim != 2 or signal.shape[0] != 16:
            reasons.append("signal_shape_or_dtype")
        if labels.dtype != np.float32 or len(labels) != len(times):
            reasons.append("label_shape_or_dtype")
        if len(times) and (not np.all(np.diff(times) > 0) or times[0] < 10):
            reasons.append("timestamps")
        if len(labels) and (labels.min() < 0 or labels.max() > 1):
            reasons.append("label_range")
        expected_samples = 0 if not len(times) else round(times[-1] * 200)
        if signal.shape[1] < expected_samples:
            reasons.append("signal_too_short")
        total_windows += len(labels)
        if reasons:
            invalid.append({"path": str(path), "reasons": reasons})
    complete_scan = args.maximum_cache_records is None
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "inventory_records": len(inventory),
        "included_inventory_records": len(expected),
        "cache_records_scanned": len(cache_paths),
        "cache_windows_scanned": total_windows,
        "complete_cache_scan": complete_scan,
        "missing_cache_records": sorted(map(str, expected - observed)) if complete_scan else None,
        "unexpected_cache_records": sorted(map(str, observed - expected - excluded.keys())) if complete_scan else None,
        "quarantined_excluded_cache_records": (
            [
                {"key": str(key), "exclusion": excluded[key]}
                for key in sorted(observed & excluded.keys())
            ]
            if complete_scan else None
        ),
        "invalid_cache_records": invalid,
        "patient_overlap": overlaps,
        "sources": source_rows,
        "passed": (
            not any(overlaps.values())
            and not invalid
            and all(row["finite_forward"] and row["deterministic_forward_max_abs"] == 0 for row in source_rows)
            and (
                not complete_scan
                or (
                    expected <= observed
                    and not (observed - expected - excluded.keys())
                )
            )
        ),
    }
    destination = OUT / "audits/assets_v2.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "passed": report["passed"]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
