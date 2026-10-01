#!/usr/bin/env python3
"""Fail-closed preflight for the TUSZ Meta-TTT v1 experiment namespace."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/tusz_meta_ttt_v1.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect() -> dict:
    config = yaml.safe_load(CONFIG.read_text())
    data_root = Path(config["data"]["root"])
    checkpoint = ROOT / config["model"]["checkpoint"]
    partitions = {}
    patient_sets = {}
    for partition in config["data"]["partitions"]:
        path = data_root / "edf" / partition
        if not path.is_dir():
            raise FileNotFoundError(path)
        patients = sorted(item.name for item in path.iterdir() if item.is_dir())
        patient_sets[partition] = set(patients)
        partitions[partition] = {
            "patients": len(patients),
            "edf_files": sum(1 for _ in path.rglob("*.edf")),
            "binary_annotations": sum(1 for _ in path.rglob("*.csv_bi")),
        }
    overlap = {
        f"{left}_{right}": sorted(patient_sets[left] & patient_sets[right])
        for index, left in enumerate(patient_sets)
        for right in list(patient_sets)[index + 1 :]
    }
    if any(overlap.values()):
        raise RuntimeError(f"patient overlap across official partitions: {overlap}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return {
        "protocol_id": config["protocol"]["id"],
        "created_utc": datetime.now(UTC).isoformat(),
        "config": str(CONFIG),
        "config_sha256": sha256(CONFIG),
        "data_root": str(data_root),
        "partitions": partitions,
        "patient_overlap": overlap,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "status": "ready",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write the preflight manifest")
    args = parser.parse_args()
    payload = inspect()
    if args.write:
        destination = ROOT / "outputs/reports/tusz_meta_ttt_v1/preflight.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        payload["written"] = str(destination)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
