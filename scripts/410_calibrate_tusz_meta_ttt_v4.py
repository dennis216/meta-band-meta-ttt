#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from bfa.tusz_meta_ttt.dataset import filter_inventory_records, load_cached_arrays
from bfa.tusz_meta_ttt_v2.batched import (
    ConditionEnsemble, EnsembleBandObjective, EnsembleMaskObjective,
    enable_second_order_batched_attention,
)
from bfa.tusz_meta_ttt_v2.grouping import patient_order
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.protocol import class_patient_record_weights
from bfa.tusz_meta_ttt_v2.runtime import load_source, optimizer_groups
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig
from bfa.tusz_meta_ttt_v4.joint_schedule import process_future_group
from bfa.tusz_meta_ttt_v4.losses import LossScale, V4Condition, source_probability_lookup

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"
V2 = ROOT / "outputs/reports/tusz_meta_ttt_v2"
CALIBRATION_CONDITIONS = tuple(
    V4Condition(name, name != "cal_post", True)
    for name in ("cal_base", "cal_post", "cal_gain", "cal_damage", "cal_kl")
)


def encoder_vector(model):
    pieces = []
    for name, parameter in model.named_parameters():
        if name.startswith(("backbone.encoder.layers.10.", "backbone.encoder.layers.11.")):
            pieces.append((torch.zeros_like(parameter) if parameter.grad is None else parameter.grad).flatten())
    return torch.cat(pieces).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=["band", "mask"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-probabilities", type=Path, required=True)
    parser.add_argument("--patients-per-batch", type=int, choices=[1, 2, 4], required=True)
    args = parser.parse_args()
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.set_float32_matmul_precision("highest")
    enable_second_order_batched_attention()
    source = V1 / "runs/supervised/development/s1_seed3407_check0.25/best.pt"
    difficulty = 0.5 if args.objective == "band" else 5
    task = f"{args.objective}_{difficulty:g}_seed3407"
    head = V2 / f"runs/ssl/development/{task}/epoch_02.pt"
    calibration_path = V2 / f"calibration/{task}/gradient_calibration.json"
    payload = json.loads(calibration_path.read_text())
    inner = InnerStepConfig(
        payload["selected_relative_step"],
        {int(k): v for k, v in payload["block_reference_norms"].items()},
        {int(k): v for k, v in payload["block_gradient_medians"].items()},
    )
    split = json.loads((V1 / "manifests/development_split.json").read_text())
    fit = set(split["development_fit"])
    inventory = json.loads((V1 / "manifests/records.json").read_text())
    paths = filter_inventory_records(
        [p for p in sorted((V1 / "cache/train").rglob("*.npz")) if p.parts[-4] in fit],
        inventory, partition="train",
    )
    labels = {path: load_cached_arrays(path)["labels"] for path in paths}
    weights = class_patient_record_weights(labels)
    patients = defaultdict(list)
    for path in paths:
        patients[path.parts[-4]].append(path)
    workloads = {patient: sum(len(labels[path]) for path in records) for patient, records in patients.items()}
    order = patient_order(workloads, seed=3408, bucket_size=8)
    reference = source_probability_lookup(args.reference_probabilities)

    # One fixed short supervised probe creates a non-zero KL gradient.  The
    # probed state is local to calibration and is never saved as a training init.
    probe_model = load_source(source, detector_trainable=True)
    probe_ssl = build_objective(args.objective, difficulty).cuda().eval()
    probe_ssl.load_state_dict(torch.load(head, map_location="cpu", weights_only=False)["objective"])
    probe_ssl.requires_grad_(True)
    probe_functional = ConditionEnsemble([probe_model])
    probe_objective = (EnsembleBandObjective if args.objective == "band" else EnsembleMaskObjective)([probe_ssl], deduplicate_views=True)
    optimizer = torch.optim.AdamW(
        optimizer_groups(probe_model, "backbone.encoder.layers.10.", 1e-5, 0.01)
        + optimizer_groups(probe_model, "backbone.encoder.layers.11.", 1e-5, 0.01)
        + optimizer_groups(probe_model, "detector.", 3e-6, 0.01), fused=True,
    )
    probe_condition = (V4Condition("cal_base", False, True),)
    for loss, _, _ in process_future_group(
        probe_functional, probe_objective, [patients[p] for p in order[:4]], weights,
        inner, probe_condition, reference, LossScale(), maximum_updates_per_patient=4,
    ):
        loss.backward()
    optimizer.step()
    probed_state = {name: value.detach().cpu() for name, value in probe_model.state_dict().items()}

    models, objectives = [], []
    for _ in CALIBRATION_CONDITIONS:
        model = load_source(source, detector_trainable=True)
        model.load_state_dict(probed_state)
        ssl = build_objective(args.objective, difficulty).cuda().eval()
        ssl.load_state_dict(torch.load(head, map_location="cpu", weights_only=False)["objective"])
        ssl.requires_grad_(True)
        models.append(model)
        objectives.append(ssl)
    functional = ConditionEnsemble(models)
    ensemble = (EnsembleBandObjective if args.objective == "band" else EnsembleMaskObjective)(objectives, deduplicate_views=True)
    rows = []
    for start in range(0, len(order), 4):
        selected = order[start:start + 4]
        for model, ssl in zip(models, objectives, strict=True):
            model.zero_grad(set_to_none=True)
            ssl.zero_grad(set_to_none=True)
        for lane_start in range(0, len(selected), args.patients_per_batch):
            lane = selected[lane_start:lane_start + args.patients_per_batch]
            for loss, _, _ in process_future_group(
                functional, ensemble, [patients[p] for p in lane], weights, inner,
                CALIBRATION_CONDITIONS, reference, LossScale(), maximum_updates_per_patient=4,
            ):
                loss.backward()
        vectors = [encoder_vector(model) for model in models]
        base = vectors[0]
        rows.append({
            "patients": selected,
            "base": float(base.norm()),
            "post": float((vectors[1] - base).norm()),
            "gain": float((vectors[2] - base).norm()),
            "paired_damage": float((vectors[3] - base).norm()),
            "kl": float((vectors[4] - base).norm()),
        })
        print(json.dumps({"groups": len(rows), "patients": min(start + 4, len(order))}), flush=True)
    medians = {name: float(np.median([row[name] for row in rows])) for name in ("base", "post", "gain", "paired_damage", "kl")}
    if any(not np.isfinite(value) or value <= 0 for value in medians.values()):
        raise RuntimeError(f"unstable v4 loss calibration: {medians}")
    scale = {
        "post": medians["base"] / medians["post"],
        "gain": medians["base"] / medians["gain"],
        "paired_damage": medians["base"] / medians["paired_damage"],
        "kl": 0.1 * medians["base"] / medians["kl"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "loss_scale": scale, "gradient_norm_medians": medians,
        "groups": len(rows), "patients": len(order), "probe_patients": order[:4],
        "maximum_updates_per_patient": 4, "rows": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
