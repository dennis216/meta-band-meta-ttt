#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from bfa.tusz_meta_ttt.data import (
    TUSZRecord,
    build_inventory,
    cache_is_valid,
    cache_record,
    stratified_development_split,
    write_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/mnt/d/TUH_EEG/TUSZ_v2.0.6")
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--content-hash", action="store_true")
    cache = sub.add_parser("cache")
    cache.add_argument("--partition", choices=["train", "dev", "eval"], required=True)
    cache.add_argument("--limit", type=int)
    args = parser.parse_args()
    inventory_path = OUT / "manifests/records.json"
    if args.command == "audit":
        records = build_inventory(DATA, hash_edf=args.content_hash)
        write_inventory(records, inventory_path)
        split = stratified_development_split(records)
        (OUT / "manifests/development_split.json").write_text(json.dumps(split, indent=2) + "\n")
        print(json.dumps({"records": len(records), "split": {k: len(v) for k, v in split.items()}}, indent=2))
        return
    payload = json.loads(inventory_path.read_text())
    records = [TUSZRecord(**{**item, "channel_names": tuple(item["channel_names"]), "seizures": tuple(tuple(x) for x in item["seizures"])}) for item in payload]
    selected = [record for record in records if record.partition == args.partition and record.exclusion is None]
    if args.limit is not None:
        selected = selected[: args.limit]
    completed = []
    for record in selected:
        destination = (
            OUT
            / "cache"
            / record.partition
            / record.patient_id
            / record.session_id
            / record.montage
            / f"{record.record_id}.npz"
        )
        if cache_is_valid(destination):
            continue
        completed.append({**asdict(record), **cache_record(record, DATA, destination)})
        print(json.dumps({"cached": str(destination), "completed": len(completed)}), flush=True)


if __name__ == "__main__":
    main()
