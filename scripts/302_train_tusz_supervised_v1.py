#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.tusz_meta_ttt.dataset import (
    CachedTUSZWindows,
    PatientClassSampler,
    cache_paths_for_patients,
    filter_inventory_records,
)
from bfa.tusz_meta_ttt.model import TUSZDetector
from bfa.tusz_meta_ttt.scoring import choose_threshold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v1"
PRETRAINED = ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"


def hash_state(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_model(freeze_backbone: bool) -> TUSZDetector:
    adapter = CBraModAdapter(PRETRAINED, train_backbone=not freeze_backbone)
    adapter.backbone.proj_out = torch.nn.Identity()
    return TUSZDetector(adapter.backbone, freeze_backbone=freeze_backbone)


def optimizer_groups(module: torch.nn.Module, lr: float) -> list[dict]:
    decay, no_decay = [], []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        target = no_decay if name.endswith("bias") or parameter.ndim == 1 else decay
        target.append(parameter)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": 0.01})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


@torch.inference_mode()
def evaluate(model, dataset, batch_size, inventory, workers):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )
    model.eval()
    probabilities = []
    losses = []
    for batch in loader:
        signal = batch["signal"].cuda(non_blocking=True)
        labels = batch["label"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(signal)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        losses.append(float(loss))
    probabilities = np.concatenate(probabilities)
    cursor = 0
    records = []
    lookup = {
        (item["patient_id"], item["session_id"], item["montage"], item["record_id"]): item
        for item in inventory
    }
    for path in dataset.cache_paths:
        with np.load(path, allow_pickle=False) as archive:
            count = len(archive["labels"])
            times = archive["decision_end_s"]
        key = (path.parts[-4], path.parts[-3], path.parts[-2], path.stem)
        item = lookup[key]
        records.append({
            "times": times,
            "probabilities": probabilities[cursor : cursor + count],
            "truths": item["seizures"],
            "duration_s": item["duration_s"],
        })
        cursor += count
    choice = choose_threshold(records)
    binary_labels = np.fromiter(
        (positive for _, _, _, positive in dataset.rows), dtype=np.uint8, count=len(dataset.rows)
    )
    auprc = float(average_precision_score(binary_labels, probabilities))
    auroc = float(roc_auc_score(binary_labels, probabilities))
    return float(np.mean(losses)), choice, auprc, auroc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["s0", "s1"], required=True)
    parser.add_argument("--stage", choices=["development", "formal"], default="development")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, choices=[1e-5, 3e-5], default=1e-5)
    parser.add_argument(
        "--epoch-fraction",
        type=float,
        choices=[0.25, 1.0],
        default=1.0,
        help="Fraction of a full sampled data traversal between validations.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark-updates", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fixed-data-epochs", type=float)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    split = json.loads((OUT / "manifests/development_split.json").read_text())
    inventory = json.loads((OUT / "manifests/records.json").read_text())
    if args.stage == "development":
        fit_patients = set(split["development_fit"])
        validation_patients = set(split["development_validation"])
    else:
        fit_patients = {item["patient_id"] for item in inventory if item["partition"] == "train"}
        validation_patients = {item["patient_id"] for item in inventory if item["partition"] == "dev"}
    cache_root = OUT / "cache"
    fit_paths = cache_paths_for_patients(cache_root, "train", fit_patients)
    validation_partition = "train" if args.stage == "development" else "dev"
    validation_paths = cache_paths_for_patients(cache_root, validation_partition, validation_patients)
    fit_paths = filter_inventory_records(fit_paths, inventory, partition="train")
    validation_paths = filter_inventory_records(
        validation_paths, inventory, partition=validation_partition
    )
    if args.smoke:
        all_paths = fit_paths + validation_paths
        if not all_paths:
            raise RuntimeError("no cached records available for smoke training")
        fit_paths = validation_paths = all_paths[:1]
    elif args.benchmark_updates is not None:
        validation_paths = validation_paths[:8]
    if not fit_paths or not validation_paths:
        raise RuntimeError("required caches are incomplete or absent")
    train_data = CachedTUSZWindows(fit_paths)
    validation_data = CachedTUSZWindows(validation_paths)
    effective_batch = 64
    accumulation = max(1, effective_batch // args.micro_batch)
    full_epoch_updates = max(1, len(train_data) // effective_batch)
    updates_per_epoch = (
        2
        if args.smoke
        else max(1, round(full_epoch_updates * args.epoch_fraction))
    )
    if args.smoke:
        max_epochs, min_epochs, patience = 1, 1, 1
    else:
        max_epochs = round(30 / args.epoch_fraction)
        min_epochs = round(5 / args.epoch_fraction)
        patience = round(5 / args.epoch_fraction)
    if args.fixed_data_epochs is not None:
        if args.stage != "formal":
            raise ValueError("fixed data epochs are reserved for formal Train-only fitting")
        chunks = args.fixed_data_epochs / args.epoch_fraction
        if chunks <= 0 or not np.isclose(chunks, round(chunks)):
            raise ValueError("fixed data epochs must be a positive multiple of epoch fraction")
        max_epochs = round(chunks)
        min_epochs = max_epochs + 1
        patience = max_epochs + 1
    if args.benchmark_updates is not None:
        updates_per_epoch = args.benchmark_updates
        max_epochs = min_epochs = patience = 1
    model = build_model(args.baseline == "s0").cuda()
    groups = optimizer_groups(model.detector, 1e-4)
    if args.baseline == "s1":
        groups = optimizer_groups(model.backbone, args.backbone_lr) + groups
    optimizer = torch.optim.AdamW(groups, fused=True)
    total_updates = updates_per_epoch * max_epochs
    warmup = max(1, round(total_updates * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup if step < warmup else 0.5 * (1 + np.cos(np.pi * (step - warmup) / max(1, total_updates - warmup))),
    )
    run = OUT / "runs" / "supervised" / args.stage / f"{args.baseline}_seed{args.seed}"
    if args.baseline == "s1" and args.backbone_lr != 1e-5:
        run = run.with_name(f"{args.baseline}_backbone_lr{args.backbone_lr:g}_seed{args.seed}")
    if args.epoch_fraction != 1.0:
        run = run.with_name(f"{run.name}_check{args.epoch_fraction:g}")
    if args.smoke:
        run = run.with_name(run.name + "_smoke")
    elif args.benchmark_updates is not None:
        run = run.with_name(run.name + f"_benchmark{args.benchmark_updates}")
    run.mkdir(parents=True, exist_ok=True)
    source_hash = hash_state(model)
    best_key = None
    stale = 0
    history = []
    start_epoch = 1
    last_checkpoint = run / "last.pt"
    if args.resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(last_checkpoint)
        state = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best_key = tuple(state["best_key"]) if state["best_key"] is not None else None
        stale = int(state["stale"])
        if best_key is not None and len(best_key) == 3:
            feasible = [row for row in history if row["reachable"]]
            if feasible:
                selected = min(
                    feasible,
                    key=lambda row: (
                        row["metrics"]["false_alarms_per_hour"],
                        row["metrics"]["false_alarm_minutes_per_hour"],
                        row["epoch"],
                    ),
                )
                best_key = (
                    0.0,
                    selected["metrics"]["false_alarms_per_hour"],
                    selected["metrics"]["false_alarm_minutes_per_hour"],
                    selected["epoch"],
                )
            else:
                selected = min(
                    history,
                    key=lambda row: (
                        -row["auprc"] if "auprc" in row else row["validation_loss"],
                        row["epoch"],
                    ),
                )
                fallback = -selected["auprc"] if "auprc" in selected else selected["validation_loss"]
                best_key = (1.0, fallback, 0.0, selected["epoch"])
            stale = int(state["epoch"]) - int(selected["epoch"])
        if best_key is not None and len(best_key) != 5:
            best_key = None
            stale = 0
        start_epoch = int(state["epoch"]) + 1
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    started = time.monotonic()
    for epoch in range(start_epoch, max_epochs + 1):
        checks_per_data_epoch = round(1 / args.epoch_fraction)
        data_epoch_index = (epoch - 1) // checks_per_data_epoch + 1
        check_index = (epoch - 1) % checks_per_data_epoch
        sampler = PatientClassSampler(
            train_data,
            samples=updates_per_epoch * effective_batch,
            seed=args.seed + data_epoch_index,
            offset=check_index * updates_per_epoch * effective_batch,
        )
        loader = DataLoader(
            train_data,
            batch_size=args.micro_batch,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        iterator = iter(loader)
        model.train()
        training_losses: list[float] = []
        for _ in range(updates_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accumulation):
                batch = next(iterator)
                signal = batch["signal"].cuda(non_blocking=True)
                labels = batch["label"].cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(signal), labels)
                training_losses.append(float(loss.detach()))
                (loss / accumulation).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        should_validate = args.fixed_data_epochs is None or epoch == max_epochs
        if should_validate:
            validation_loss, choice, auprc, auroc = evaluate(
                model, validation_data, args.eval_batch, inventory, args.workers
            )
            maximum = choice.maximum_sensitivity_metrics
            key = (
                0.0 if choice.reachable else 1.0,
                choice.metrics.false_alarms_per_hour if choice.reachable else -maximum.sensitivity,
                choice.metrics.false_alarm_minutes_per_hour if choice.reachable else -auprc,
                0.0 if choice.reachable else maximum.false_alarms_per_hour,
                epoch,
            )
            threshold = choice.threshold
            metrics = vars(choice.metrics) if choice.metrics else None
            maximum_metrics = vars(maximum) if maximum else None
        else:
            validation_loss = auprc = auroc = None
            key = None
            threshold = metrics = maximum_metrics = None
        row = {
            "epoch": epoch,
            "data_epoch": epoch * args.epoch_fraction,
            "training_loss": float(np.mean(training_losses)),
            "validation_loss": validation_loss,
            "auprc": auprc,
            "auroc": auroc,
            "reachable": choice.reachable if should_validate else None,
            "maximum_sensitivity_metrics": maximum_metrics,
            "threshold": threshold,
            "metrics": metrics,
            "elapsed_s": time.monotonic() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "threshold": threshold,
            "metrics": metrics,
            "maximum_sensitivity_metrics": maximum_metrics,
            "source_hash": source_hash,
            "fixed_data_epochs": args.fixed_data_epochs,
        }
        torch.save(checkpoint, run / f"epoch_{epoch:02d}.pt")
        if args.fixed_data_epochs is not None:
            if epoch == max_epochs:
                best_key = key
                torch.save(checkpoint, run / "best.pt")
        elif best_key is None or key < best_key:
            best_key, stale = key, 0
            torch.save(checkpoint, run / "best.pt")
        else:
            stale += 1
        (run / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "history": history,
                "best_key": best_key,
                "stale": stale,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            },
            last_checkpoint,
        )
        if epoch >= min_epochs and stale >= patience:
            break
    (run / "complete.json").write_text(json.dumps({"status": "complete", "created_utc": datetime.now(UTC).isoformat(), "best": str(run / "best.pt"), "epochs": len(history)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
