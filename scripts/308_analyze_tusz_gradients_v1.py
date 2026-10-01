#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bfa.evaluation.match import match_events
from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.tusz_meta_ttt.dataset import filter_inventory_records, load_cached_arrays
from bfa.tusz_meta_ttt.functional import FunctionalTUSZModel
from bfa.tusz_meta_ttt.model import TUSZDetector
from bfa.tusz_meta_ttt.objectives import build_objective
from bfa.tusz_meta_ttt.scoring import consecutive_eventize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v1"
PRETRAINED = ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"


def load_model(checkpoint):
    adapter = CBraModAdapter(PRETRAINED, train_backbone=True)
    adapter.backbone.proj_out = torch.nn.Identity()
    model = TUSZDetector(adapter.backbone)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    return model.requires_grad_(False).cuda().eval()


def flatten(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--objective-checkpoint", type=Path, required=True)
    parser.add_argument("--partition", choices=["train", "dev", "eval"], default="train")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument(
        "--cohort", choices=["all", "development_fit", "development_validation"], default="all"
    )
    parser.add_argument("--samples-per-group", type=int, default=256)
    parser.add_argument("--max-samples-per-patient-group", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    model = load_model(args.source.resolve())
    functional = FunctionalTUSZModel(model)
    state = torch.load(args.objective_checkpoint, map_location="cpu", weights_only=False)
    objective = build_objective(state["objective_name"]).cuda().eval()
    objective.load_state_dict(state["objective"])
    inner_lr = float(state["inner_lr"])
    candidates = []
    inventory = json.loads((OUT / "manifests/records.json").read_text())
    item_lookup = {
        (item["patient_id"], item["session_id"], item["montage"], item["record_id"]): item
        for item in inventory
    }
    allowed_patients = None
    if args.cohort != "all":
        if args.partition != "train":
            raise ValueError(
                "development cohorts are only defined inside the official Train partition"
            )
        split = json.loads((OUT / "manifests/development_split.json").read_text())
        allowed_patients = set(split[args.cohort])
    paths = filter_inventory_records(
        sorted((OUT / "cache" / args.partition).rglob("*.npz")), inventory, partition=args.partition
    )
    if allowed_patients is not None:
        paths = [path for path in paths if path.parts[-4] in allowed_patients]
    probability_frame = pd.read_parquet(args.probabilities)
    probability_groups = {
        tuple(key): group.sort_values("decision_end_s")
        for key, group in probability_frame.groupby(
            ["patient_id", "session_id", "montage", "record_id"], sort=False
        )
    }
    for path in paths:
        key = (path.parts[-4], path.parts[-3], path.parts[-2], path.stem)
        item = item_lookup[key]
        archive = load_cached_arrays(path)
        times, labels = archive["decision_end_s"], archive["labels"]
        if key not in probability_groups:
            if len(labels) == 0:
                continue
            raise KeyError(f"nonempty record missing saved probabilities: {key}")
        group = probability_groups[key]
        probabilities = group["frozen_probability"].to_numpy(float)
        np.testing.assert_allclose(times, group["decision_end_s"].to_numpy(float))
        alarms = consecutive_eventize(times, probabilities, threshold=args.threshold)
        matched = match_events(alarms, item["seizures"])
        false_alarms = [alarms[index] for index in matched.unmatched_predictions]
        for row, (label, end_s, probability) in enumerate(
            zip(labels, times, probabilities, strict=True)
        ):
            if label > 0:
                if any(abs(end_s - onset) <= 4 for onset, _ in item["seizures"]):
                    category = "seizure_onset"
                elif any(abs(end_s - offset) <= 4 for _, offset in item["seizures"]):
                    category = "seizure_offset"
                else:
                    category = "seizure"
            elif any(alarm.start_s <= end_s < alarm.end_s for alarm in false_alarms):
                category = "frozen_false_alarm"
            elif probability >= args.threshold:
                category = "high_score_background"
            else:
                category = "background"
            candidates.append((path, row, float(label), float(end_s), category, float(probability)))
    rng = np.random.default_rng(args.seed)
    selected = []
    for category in (
        "seizure",
        "seizure_onset",
        "seizure_offset",
        "background",
        "high_score_background",
        "frozen_false_alarm",
    ):
        pool = [row for row in candidates if row[4] == category]
        rng.shuffle(pool)
        patient_counts: dict[str, int] = {}
        balanced_pool = []
        for candidate in pool:
            patient_id = candidate[0].parts[-4]
            count = patient_counts.get(patient_id, 0)
            if count >= args.max_samples_per_patient_group:
                continue
            patient_counts[patient_id] = count + 1
            balanced_pool.append(candidate)
        pool = balanced_pool
        count = min(len(pool), args.samples_per_group)
        if count:
            selected.extend(
                pool[index] for index in rng.choice(len(pool), size=count, replace=False)
            )
    outputs = []
    last_path = None
    signal = None
    for path, row, label, end_s, category, cached_probability in selected:
        if path != last_path:
            signal = load_cached_arrays(path)["signal"]
            last_path = path
        assert signal is not None
        value = signal[:, row * 400 : row * 400 + 2000].reshape(1, 16, 10, 200)
        window = torch.from_numpy(value).cuda()
        fast = functional.initial_fast_parameters()
        ssl_loss = objective.loss(
            window,
            feature_fn=lambda x, parameters=fast: functional.features(x, parameters),
            fast_parameters=fast,
            transform_seed=args.seed + row,
        )
        ssl_gradients = torch.autograd.grad(ssl_loss, tuple(fast.values()), retain_graph=True)
        logits = functional.logits(window, fast)
        target = torch.tensor([label], device=logits.device)
        classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        classification_gradients = torch.autograd.grad(classification_loss, tuple(fast.values()))
        ssl_vector, classification_vector = (
            flatten(ssl_gradients),
            flatten(classification_gradients),
        )
        cosine = torch.nn.functional.cosine_similarity(ssl_vector, classification_vector, dim=0)
        updated = {
            name: parameter - inner_lr * gradient
            for (name, parameter), gradient in zip(fast.items(), ssl_gradients, strict=True)
        }
        post_logits = functional.logits(window, updated)
        post_loss = torch.nn.functional.binary_cross_entropy_with_logits(post_logits, target)
        post_ssl_loss = objective.loss(
            window,
            feature_fn=lambda x, parameters=updated: functional.features(x, parameters),
            fast_parameters=updated,
            transform_seed=args.seed + row,
        )
        probability = float(torch.sigmoid(logits).detach())
        if not np.isclose(probability, cached_probability, atol=2e-5):
            raise RuntimeError("recomputed Frozen probability differs from saved evaluation")
        outputs.append(
            {
                "partition": args.partition,
                "patient_id": path.parts[-4],
                "record_id": path.stem,
                "decision_end_s": end_s,
                "label": label,
                "group": category,
                "frozen_probability": probability,
                "adapted_probability": float(torch.sigmoid(post_logits).detach()),
                "ssl_loss_before": float(ssl_loss.detach()),
                "ssl_loss_after": float(post_ssl_loss.detach()),
                "delta_ssl_loss": float((post_ssl_loss - ssl_loss).detach()),
                "cosine": float(cosine.detach()),
                "ssl_gradient_norm": float(ssl_vector.norm().detach()),
                "classification_gradient_norm": float(classification_vector.norm().detach()),
                "parameter_update_norm": float((inner_lr * ssl_vector).norm().detach()),
                "delta_logit": float((post_logits - logits).detach()),
                "delta_bce": float((post_loss - classification_loss).detach()),
            }
        )
    frame = pd.DataFrame(outputs)
    run = (
        OUT
        / "mechanisms"
        / "{}_{}".format(state["objective_name"], args.objective_checkpoint.parent.name)
        / args.partition
    )
    run.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(run / "gradient_samples.parquet", index=False)
    summary = (
        frame.groupby("group")
        .agg(
            {
                "cosine": ["count", "mean", "std", "median"],
                "delta_bce": ["mean", "std", "median"],
                "delta_ssl_loss": ["mean", "std", "median"],
                "delta_logit": ["mean", "std", "median"],
                "ssl_gradient_norm": ["mean", "median"],
                "classification_gradient_norm": ["mean", "median"],
                "parameter_update_norm": ["mean", "median"],
            }
        )
        .to_dict()
    )
    serializable = {
        "partition": args.partition,
        "cohort": args.cohort,
        "samples_per_group_limit": args.samples_per_group,
        "max_samples_per_patient_group": args.max_samples_per_patient_group,
        "candidate_counts": {
            category: sum(row[4] == category for row in candidates)
            for category in sorted({row[4] for row in candidates})
        },
        "statistics": {
            str(key): {str(group): float(value) for group, value in values.items()}
            for key, values in summary.items()
        },
    }
    (run / "summary.json").write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
