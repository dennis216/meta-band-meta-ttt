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
    cache_paths_for_patients,
    filter_inventory_records,
)
from bfa.tusz_meta_ttt.model import TUSZDetector
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.protocol import RecordLocalBatchSampler, stable_transform_seed

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"
PRETRAINED = ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"


def load_source(path: Path) -> TUSZDetector:
    adapter = CBraModAdapter(PRETRAINED, train_backbone=True)
    adapter.backbone.proj_out = torch.nn.Identity()
    model = TUSZDetector(adapter.backbone)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return model.requires_grad_(False).cuda().eval()


@torch.inference_mode()
def evaluate(source, objective, loader, dataset, seed: int) -> dict[str, float | None]:
    totals: dict[str, float] = {}
    examples = 0
    objective.eval()
    for index, batch in enumerate(loader):
        signal = batch["signal"].cuda(non_blocking=True)
        file_indexes = batch["file_index"].numpy()
        if not np.all(file_indexes == file_indexes[0]):
            raise RuntimeError("SSL evaluation batches must remain record-local")
        transform_seed = [
            stable_transform_seed(
                seed, dataset.cache_paths[int(file_indexes[0])], objective.name, int(row)
            )
            for row in batch["row"].tolist()
        ]
        metrics = objective.metrics(
            signal, feature_fn=source.features, transform_seed=transform_seed
        )
        examples += signal.shape[0]
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value) * signal.shape[0]
    return {name: value / examples for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--objective", choices=["band", "temporal", "mask"], required=True)
    parser.add_argument("--difficulty", type=float, required=True)
    parser.add_argument("--stage", choices=["development", "formal"], default="development")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-samples", type=int, default=4096)
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    split = json.loads((V1 / "manifests/development_split.json").read_text())
    inventory = json.loads((V1 / "manifests/records.json").read_text())
    if args.stage == "development":
        fit = set(split["development_fit"])
        validation = set(split["development_validation"])
        validation_partition = "train"
    else:
        fit = {item["patient_id"] for item in inventory if item["partition"] == "train"}
        validation = {item["patient_id"] for item in inventory if item["partition"] == "dev"}
        validation_partition = "dev"
    fit_paths = filter_inventory_records(
        cache_paths_for_patients(V1 / "cache", "train", fit), inventory, partition="train"
    )
    validation_paths = filter_inventory_records(
        cache_paths_for_patients(V1 / "cache", validation_partition, validation),
        inventory,
        partition=validation_partition,
    )
    if args.smoke:
        fit_paths = validation_paths = (fit_paths + validation_paths)[:1]
        args.epochs = 1
        args.maximum_samples = 32
        args.validation_samples = 32
    fit_data = CachedTUSZWindows(fit_paths)
    validation_data = CachedTUSZWindows(validation_paths)
    samples = len(fit_data) if args.maximum_samples is None else min(len(fit_data), args.maximum_samples)
    samples = max(1, samples)
    source = load_source(args.source.resolve())
    objective = build_objective(args.objective, args.difficulty).cuda().train()
    optimizer = torch.optim.AdamW(objective.parameters(), lr=1e-3, weight_decay=0.0, fused=True)
    run = OUT / "runs/ssl" / args.stage / f"{args.objective}_{args.difficulty:g}_seed{args.seed}"
    if args.smoke:
        run = run.with_name(run.name + "_smoke")
    elif args.run_tag:
        run = run.with_name(run.name + f"_{args.run_tag}")
    run.mkdir(parents=True, exist_ok=True)
    history = []
    first_epoch = 1
    if args.resume:
        resumed = torch.load(args.resume.resolve(), map_location="cpu", weights_only=False)
        expected = {
            "objective_name": args.objective,
            "difficulty": args.difficulty,
            "stage": args.stage,
            "seed": args.seed,
        }
        mismatches = {
            key: (resumed.get(key), value)
            for key, value in expected.items()
            if resumed.get(key) != value
        }
        if mismatches:
            raise ValueError(f"resume configuration mismatch: {mismatches}")
        objective.load_state_dict(resumed["objective"])
        optimizer.load_state_dict(resumed["optimizer"])
        history = list(resumed.get("history", []))
        first_epoch = int(resumed["epoch"]) + 1
        random.setstate(resumed["rng_state"]["python"])
        np.random.set_state(resumed["rng_state"]["numpy"])
        torch.random.set_rng_state(resumed["rng_state"]["torch"])
        torch.cuda.set_rng_state_all(resumed["rng_state"]["cuda"])
    started = time.monotonic()
    for epoch in range(first_epoch, args.epochs + 1):
        sampler = RecordLocalBatchSampler(
            fit_data, batch_size=args.batch_size, samples=samples, seed=args.seed + epoch
        )
        loader = DataLoader(
            fit_data, batch_sampler=sampler, num_workers=args.workers,
            pin_memory=True, persistent_workers=args.workers > 0,
        )
        losses = []
        training_seen = 0
        objective.train()
        for batch in loader:
            signal = batch["signal"].cuda(non_blocking=True)
            training_seen += signal.shape[0]
            file_indexes = batch["file_index"].numpy()
            if not np.all(file_indexes == file_indexes[0]):
                raise RuntimeError("SSL training batches must remain record-local")
            transform_seed = [
                stable_transform_seed(
                    args.seed,
                    fit_data.cache_paths[int(file_indexes[0])],
                    args.objective,
                    int(row),
                    epoch,
                )
                for row in batch["row"].tolist()
            ]
            optimizer.zero_grad(set_to_none=True)
            loss = objective.loss(
                signal, feature_fn=source.features,
                transform_seed=transform_seed,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(objective.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if training_seen != samples:
            raise RuntimeError(f"sampler emitted {training_seen} windows, expected {samples}")
        validation_count = min(args.validation_samples, len(validation_data))
        validation_sampler = RecordLocalBatchSampler(
            validation_data,
            batch_size=args.batch_size,
            samples=validation_count,
            seed=args.seed,
        )
        validation_loader = DataLoader(
            validation_data, batch_sampler=validation_sampler,
            num_workers=args.workers, pin_memory=True,
        )
        metrics = evaluate(source, objective, validation_loader, validation_data, args.seed)
        row = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "training_samples": training_seen,
            "complete_training_window_coverage": training_seen == len(fit_data),
            "validation_samples": validation_count,
            "validation": metrics,
            "elapsed_s": time.monotonic() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "objective": objective.state_dict(), "objective_name": args.objective,
            "difficulty": args.difficulty, "training": "ssl_warmup_v2",
            "source": str(args.source.resolve()), "stage": args.stage, "seed": args.seed,
            "epoch": epoch, "created_utc": datetime.now(UTC).isoformat(),
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
