#!/usr/bin/env python3
"""Continue supervised training with the v2 data weights and no inner update."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from bfa.tusz_meta_ttt.dataset import filter_inventory_records, load_cached_arrays
from bfa.tusz_meta_ttt_v2.protocol import assert_train_cache_paths, class_patient_record_weights, weight_audit
from bfa.tusz_meta_ttt_v2.runtime import load_source, make_windows, optimizer_groups

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scope", choices=["e", "ed"], required=True)
    parser.add_argument("--stage", choices=["development", "formal"], default="development")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--maximum-records", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = load_source(args.source.resolve(), detector_trainable="d" in args.scope)
    inventory = json.loads((V1 / "manifests/records.json").read_text())
    split = json.loads((V1 / "manifests/development_split.json").read_text())
    patients = (
        set(split["development_fit"])
        if args.stage == "development"
        else {item["patient_id"] for item in inventory if item["partition"] == "train"}
    )
    paths = [path for path in sorted((V1 / "cache/train").rglob("*.npz")) if path.parts[-4] in patients]
    paths = filter_inventory_records(paths, inventory, partition="train")
    assert_train_cache_paths(paths, V1 / "cache/train")
    if args.maximum_records is not None:
        paths = paths[: args.maximum_records]
    labels = {path: load_cached_arrays(path)["labels"] for path in paths}
    weights = class_patient_record_weights(labels)
    by_patient: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_patient[path.parts[-4]].append(path)

    groups = optimizer_groups(model, "backbone.encoder.layers.10.", 1e-5, 0.01)
    groups += optimizer_groups(model, "backbone.encoder.layers.11.", 1e-5, 0.01)
    if "d" in args.scope:
        groups += optimizer_groups(model, "detector.", 3e-6, 0.01)
    optimizer = torch.optim.AdamW(groups, fused=True)
    run = OUT / "runs/supervised_controls" / args.stage / f"{args.scope}_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    history = []
    first_epoch = 1
    if args.resume:
        resumed = torch.load(args.resume.resolve(), map_location="cpu", weights_only=False)
        expected = {"scope": args.scope, "stage": args.stage, "seed": args.seed}
        mismatches = {
            key: (resumed.get(key), value)
            for key, value in expected.items()
            if resumed.get(key) != value
        }
        if mismatches:
            raise ValueError(f"resume configuration mismatch: {mismatches}")
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        history = list(resumed.get("history", []))
        first_epoch = int(resumed["epoch"]) + 1
        random.setstate(resumed["rng_state"]["python"])
        np.random.set_state(resumed["rng_state"]["numpy"])
        torch.random.set_rng_state(resumed["rng_state"]["torch"])
        torch.cuda.set_rng_state_all(resumed["rng_state"]["cuda"])
    total_patients = len(by_patient)
    for epoch in range(first_epoch, args.epochs + 1):
        order = sorted(by_patient)
        random.Random(args.seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        group_size = 0
        for patient_index, patient in enumerate(order, 1):
            group_size += 1
            for path in by_patient[patient]:
                archive = load_cached_arrays(path)
                row_count = len(archive["labels"])
                for start in range(0, row_count, 64):
                    rows = tuple(range(start, min(start + 64, row_count)))
                    signal = make_windows(archive["signal"], rows)
                    target = torch.from_numpy(archive["labels"][list(rows)]).cuda()
                    row_weights = torch.from_numpy(weights[path][list(rows)]).float().cuda()
                    logits = model(signal)
                    losses = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, target, reduction="none"
                    )
                    loss = (losses * row_weights).sum()
                    loss.backward()
                    loss_sum += float(loss.detach())
            if patient_index % 4 == 0 or patient_index == total_patients:
                scale = total_patients / group_size
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
                encoder = [
                    parameter for name, parameter in model.named_parameters()
                    if name.startswith(("backbone.encoder.layers.10.", "backbone.encoder.layers.11."))
                ]
                torch.nn.utils.clip_grad_norm_(encoder, 1.0)
                if "d" in args.scope:
                    torch.nn.utils.clip_grad_norm_(model.detector.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                group_size = 0
        row = {
            "epoch": epoch,
            "weighted_loss": loss_sum,
            "patients": total_patients,
            "records": len(paths),
            "class_weight_audit": weight_audit(weights, labels),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
                "model": model.state_dict(),
                "source": str(args.source.resolve()),
                "scope": args.scope,
                "stage": args.stage,
                "seed": args.seed,
                "epoch": epoch,
                "created_utc": datetime.now(UTC).isoformat(),
            }
        torch.save(checkpoint, run / f"epoch_{epoch:02d}.pt")
        torch.save(
            {
                **checkpoint,
                "optimizer": optimizer.state_dict(),
                "history": history,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.random.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                },
            },
            run / "last.pt",
        )
        (run / "history.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
