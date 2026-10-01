#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.tusz_meta_ttt.dataset import (
    CachedTUSZWindows,
    PatientUniformSampler,
    cache_paths_for_patients,
    filter_inventory_records,
)
from bfa.tusz_meta_ttt.model import TUSZDetector
from bfa.tusz_meta_ttt.objectives import (
    BANDS_HZ,
    TEMPORAL_PERMUTATIONS,
    build_objective,
    remove_frequency_band,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v1"
PRETRAINED = ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"


def load_source(path: Path) -> TUSZDetector:
    adapter = CBraModAdapter(PRETRAINED, train_backbone=True)
    adapter.backbone.proj_out = nn.Identity()
    model = TUSZDetector(adapter.backbone)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return model.requires_grad_(False).cuda().eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--objective-checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    state = torch.load(args.objective_checkpoint, map_location="cpu", weights_only=False)
    name = state["objective_name"]
    if name not in {"band", "temporal", "mask"}:
        raise ValueError("auxiliary evaluation supports band, temporal, and mask")
    split = json.loads((OUT / "manifests/development_split.json").read_text())
    inventory = json.loads((OUT / "manifests/records.json").read_text())
    paths = cache_paths_for_patients(OUT / "cache", "train", set(split["development_validation"]))
    paths = filter_inventory_records(paths, inventory, partition="train")
    data = CachedTUSZWindows(paths)
    sampler = PatientUniformSampler(data, samples=min(args.samples, len(data)), seed=args.seed)
    loader = DataLoader(data, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True)
    source = load_source(args.source.resolve())
    objective = build_objective(name).cuda().eval()
    objective.load_state_dict(state["objective"])
    total_loss = 0.0
    examples = 0
    correct = 0
    per_class_correct: list[int] = []
    per_class_total: list[int] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            signal = batch["signal"].cuda(non_blocking=True)
            if name == "band":
                views = torch.cat([remove_frequency_band(signal, band) for band in BANDS_HZ], dim=0)
                features = source.features(views).mean(dim=1).flatten(1)
                labels = torch.arange(len(BANDS_HZ), device=signal.device).repeat_interleave(signal.shape[0])
                logits = objective.head(features)
                loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
            elif name == "temporal":
                views = torch.cat([objective.permute(signal, order) for order in TEMPORAL_PERMUTATIONS], dim=0)
                features = source.features(views).mean(dim=1).flatten(1)
                labels = torch.arange(len(TEMPORAL_PERMUTATIONS), device=signal.device).repeat_interleave(signal.shape[0])
                logits = objective.head(features)
                loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
            else:
                loss_value = objective.loss(signal, feature_fn=source.features, transform_seed=args.seed + batch_index)
                total_loss += float(loss_value) * signal.shape[0]
                examples += signal.shape[0]
                continue
            predictions = logits.argmax(dim=1)
            total_loss += float(loss)
            examples += labels.numel()
            correct += int((predictions == labels).sum())
            classes = logits.shape[1]
            if not per_class_total:
                per_class_total = [0] * classes
                per_class_correct = [0] * classes
            for class_index in range(classes):
                mask = labels == class_index
                per_class_total[class_index] += int(mask.sum())
                per_class_correct[class_index] += int(((predictions == labels) & mask).sum())
    result = {
        "objective": name,
        "checkpoint": str(args.objective_checkpoint.resolve()),
        "patients": len(set(split["development_validation"])),
        "sampled_windows": min(args.samples, len(data)),
        "loss": total_loss / examples,
        "accuracy": correct / examples if name != "mask" else None,
        "per_class_accuracy": [
            value / count for value, count in zip(per_class_correct, per_class_total, strict=True)
        ] if name != "mask" else None,
        "seed": args.seed,
    }
    destination = OUT / "runs/ssl/development" / f"{name}_seed{args.seed}" / "validation_metrics.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
