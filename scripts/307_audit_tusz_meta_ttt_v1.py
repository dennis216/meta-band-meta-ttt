#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v1"


def artifact(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}


def main() -> None:
    records_path = OUT / "manifests/records.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    inventory_counts = Counter(item["partition"] for item in records)
    cache_counts = {partition: len(list((OUT / "cache" / partition).rglob("*.npz"))) for partition in ("train", "dev", "eval")}
    expected = {"train": 5057, "dev": 2203, "eval": 880}
    requirements = {
        "protocol": artifact(ROOT / "protocols/tusz-meta-ttt-v1.md"),
        "config": artifact(ROOT / "configs/tusz_meta_ttt_v1.yaml"),
        "inventory": artifact(records_path),
        "development_split": artifact(OUT / "manifests/development_split.json"),
        "s0_development": artifact(OUT / "runs/supervised/development/s0_seed3407/complete.json"),
        "s1_development": artifact(OUT / "runs/supervised/development/s1_seed3407/complete.json"),
    }
    for objective in ("band", "temporal", "mask", "learned"):
        requirements[f"meta_{objective}_development"] = {
            "exists": any((OUT / "runs/meta/development/online").glob(f"{objective}_*/history.json")),
            "path": f"outputs/reports/tusz_meta_ttt_v1/runs/meta/development/online/{objective}_*/history.json",
        }
    requirements.update(
        {
            "candidate_selection": artifact(OUT / "selection/development_candidates.json"),
            "same_window_candidates": artifact(OUT / "selection/same_window_complete.json"),
            "mechanism_analysis": artifact(OUT / "mechanisms/summary.json"),
            "joint_comparison": artifact(OUT / "runs/joint/development/complete.json"),
            "formal_training": artifact(OUT / "runs/formal_complete.json"),
            "dev_calibration": artifact(OUT / "evaluation/dev_complete.json"),
            "eval_confirmation": artifact(OUT / "evaluation/eval_complete.json"),
            "patient_bootstrap": artifact(OUT / "statistics/paired_bootstrap.json"),
            "final_report": artifact(OUT / "final_report.md"),
            "reproducibility_manifest": artifact(OUT / "release_manifest.json"),
        }
    )
    payload = {
        "created_utc": datetime.now(UTC).isoformat(),
        "inventory_counts": dict(inventory_counts),
        "cache_counts": cache_counts,
        "cache_complete": inventory_counts == Counter(expected) and cache_counts == expected,
        "requirements": requirements,
        "complete": all(item["exists"] for item in requirements.values()),
    }
    destination = OUT / "pipeline_status.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
