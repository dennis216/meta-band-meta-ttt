#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.tusz_meta_ttt.dataset import filter_inventory_records, load_cached_arrays
from bfa.tusz_meta_ttt.functional import FunctionalTUSZModel
from bfa.tusz_meta_ttt.model import TUSZDetector, balanced_soft_bce
from bfa.tusz_meta_ttt.objectives import build_objective

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v1"
PRETRAINED = ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"


def make_model() -> TUSZDetector:
    adapter = CBraModAdapter(PRETRAINED, train_backbone=True)
    adapter.backbone.proj_out = torch.nn.Identity()
    return TUSZDetector(adapter.backbone).cuda().eval()


def windows(signal, rows):
    values = np.stack([signal[:, row * 400 : row * 400 + 2000] for row in rows])
    return torch.from_numpy(values.reshape(-1, 16, 10, 200)).cuda()


def record_losses(model, functional, objective, path, inner_lr, seed, smoke):
    archive = load_cached_arrays(path)
    signal, labels = archive["signal"], archive["labels"]
    fast = functional.initial_fast_parameters(detach=False)
    update_count = max(0, (len(labels) - 15) // 15)
    if smoke:
        update_count = min(2, update_count)
    accumulated = []
    for update in range(update_count):
        start = update * 15
        support = windows(signal, [start, start + 5, start + 10])
        query_rows = list(range(start + 15, min(start + 30, len(labels))))
        query = windows(signal, query_rows)
        ssl_loss = objective.loss(
            support,
            feature_fn=lambda x, parameters=fast: functional.features(x, parameters),
            fast_parameters=fast,
            transform_seed=seed + update,
        )
        gradients = torch.autograd.grad(ssl_loss, tuple(fast.values()), create_graph=True)
        fast = {
            name: value - inner_lr * gradient
            for (name, value), gradient in zip(fast.items(), gradients, strict=True)
        }
        target = torch.from_numpy(labels[query_rows]).cuda()
        frozen_loss = balanced_soft_bce(model(query), target)
        adapted_loss = balanced_soft_bce(functional.logits(query, fast), target)
        accumulated.append(0.5 * frozen_loss + 0.5 * adapted_loss)
        if (update + 1) % 4 == 0:
            yield torch.stack(accumulated).mean()
            accumulated = []
            fast = {name: value.detach().requires_grad_(True) for name, value in fast.items()}
    if accumulated:
        yield torch.stack(accumulated).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=["band", "temporal", "mask", "learned"], required=True)
    parser.add_argument("--inner-lr", type=float, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--maximum-records", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = make_model()
    functional = FunctionalTUSZModel(model)
    objective = build_objective(args.objective).cuda()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.detector.parameters(), "lr": 1e-4},
            {"params": objective.parameters(), "lr": 1e-4},
        ],
        weight_decay=0.01,
    )
    split = json.loads((OUT / "manifests/development_split.json").read_text())
    inventory = json.loads((OUT / "manifests/records.json").read_text())
    fit = set(split["development_fit"])
    paths = sorted(path for path in (OUT / "cache/train").rglob("*.npz") if path.parts[-4] in fit)
    paths = filter_inventory_records(paths, inventory, partition="train")
    if args.smoke:
        paths, args.epochs = paths[:1], 1
    if args.maximum_records is not None:
        paths = paths[: args.maximum_records]
    run = OUT / "runs/joint/development" / f"{args.objective}_lr{args.inner_lr:g}_seed{args.seed}"
    if args.smoke:
        run = run.with_name(run.name + "_smoke")
    run.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_paths = paths.copy()
        random.Random(args.seed + epoch).shuffle(epoch_paths)
        losses = []
        pending_patients = set()
        optimizer.zero_grad(set_to_none=True)
        for index, path in enumerate(epoch_paths):
            pending_patients.add(path.parts[-4])
            for loss in record_losses(
                model, functional, objective, path, args.inner_lr, args.seed + index, args.smoke
            ):
                loss.backward()
                losses.append(float(loss.detach()))
            if len(pending_patients) >= 4 or index + 1 == len(epoch_paths):
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(objective.parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_patients.clear()
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "records": len(paths)}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save(
            {"model": model.state_dict(), "objective": objective.state_dict(), "objective_name": args.objective, "inner_lr": args.inner_lr, "epoch": epoch, "created_utc": datetime.now(UTC).isoformat()},
            run / f"epoch_{epoch:02d}.pt",
        )
        (run / "history.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
