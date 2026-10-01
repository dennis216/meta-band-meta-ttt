#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.tusz_meta_ttt.dataset import (
    CachedTUSZWindows,
    PatientUniformSampler,
    cache_paths_for_patients,
    filter_inventory_records,
)
from bfa.tusz_meta_ttt.model import TUSZDetector
from bfa.tusz_meta_ttt.objectives import build_objective

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v1"
PRETRAINED = ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"


def load_source(path: Path) -> TUSZDetector:
    adapter = CBraModAdapter(PRETRAINED, train_backbone=True)
    adapter.backbone.proj_out = torch.nn.Identity()
    model = TUSZDetector(adapter.backbone)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return model.requires_grad_(False).cuda().eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=["band", "temporal", "mask"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inner-lr", type=float, required=True)
    parser.add_argument("--stage", choices=["development", "formal"], default="development")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    split = json.loads((OUT / "manifests/development_split.json").read_text())
    inventory = json.loads((OUT / "manifests/records.json").read_text())
    patients = (
        set(split["development_fit"])
        if args.stage == "development"
        else {item["patient_id"] for item in inventory if item["partition"] == "train"}
    )
    paths = cache_paths_for_patients(OUT / "cache", "train", patients)
    paths = filter_inventory_records(paths, inventory, partition="train")
    if args.smoke:
        paths, args.epochs = sorted((OUT / "cache/train").rglob("*.npz"))[:1], 1
    data = CachedTUSZWindows(paths)
    requested_samples = min(32, len(data)) if args.smoke else len(data)
    samples_per_epoch = max(
        args.batch_size, requested_samples // args.batch_size * args.batch_size
    )
    sampler = PatientUniformSampler(
        data, samples=samples_per_epoch * args.epochs, seed=args.seed
    )
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    source = load_source(args.source.resolve())
    objective = build_objective(args.objective).cuda().train()
    optimizer = torch.optim.AdamW(objective.parameters(), lr=1e-3, weight_decay=0.0)
    run = OUT / "runs/ssl" / args.stage / f"{args.objective}_seed{args.seed}"
    if args.smoke:
        run = run.with_name(run.name + "_smoke")
    run.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    history = []
    started = time.monotonic()
    batches = int(np.ceil(samples_per_epoch / args.batch_size))
    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch_index in range(batches):
            signal = next(iterator)["signal"].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = objective.loss(
                    signal,
                    feature_fn=source.features,
                    transform_seed=args.seed + epoch * batches + batch_index,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(objective.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "samples": samples_per_epoch, "elapsed_s": time.monotonic() - started}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save(
            {"objective": objective.state_dict(), "objective_name": args.objective, "training": "ssl_only", "stage": args.stage, "inner_lr": args.inner_lr, "epoch": epoch, "source": str(args.source.resolve()), "created_utc": datetime.now(UTC).isoformat()},
            run / f"epoch_{epoch:02d}.pt",
        )
        (run / "history.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
