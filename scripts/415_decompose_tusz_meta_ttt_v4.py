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
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.protocol import class_patient_record_weights
from bfa.tusz_meta_ttt_v2.runtime import load_source
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig
from bfa.tusz_meta_ttt_v4.joint_schedule import process_future_group
from bfa.tusz_meta_ttt_v4.losses import CONDITIONS, LossScale, source_probability_lookup

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"


def vector(named, gradients, prefix):
    pieces = []
    for (name, parameter), gradient in zip(named, gradients, strict=True):
        if name.startswith(prefix):
            pieces.append((torch.zeros_like(parameter) if gradient is None else gradient).flatten())
    return torch.cat(pieces).float() if pieces else torch.zeros(1, device="cuda")


def statistics(full, direct, encoder_stop):
    through = full - direct
    encoder_second_order = full - encoder_stop
    reconstruction = (direct + through - full).norm() / full.norm().clamp_min(1e-12)
    return {
        "full_norm": float(full.norm()), "direct_norm": float(direct.norm()),
        "through_norm": float(through.norm()),
        "through_to_full": float(through.norm() / full.norm().clamp_min(1e-12)),
        "direct_through_cosine": float(torch.nn.functional.cosine_similarity(direct, through, dim=0)),
        "encoder_second_order_ablation_difference_norm": float(encoder_second_order.norm()),
        "reconstruction_relative_error": float(reconstruction),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-probabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    enable_second_order_batched_attention()
    torch.set_float32_matmul_precision("highest")
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
    reference = source_probability_lookup(args.reference_probabilities)
    condition = tuple(item for item in CONDITIONS if item.name == state["v4_condition"])
    config = InnerStepConfig(**state["inner_config"])
    scale = LossScale(**state["loss_scale"])

    def gradients(selected, *, direct=False, encoder_stop=False):
        model = load_source(Path(state["source"]), detector_trainable=condition[0].train_detector)
        model.load_state_dict(state["model"])
        objective = build_objective(state["objective_name"], float(state["difficulty"])).cuda().eval()
        objective.load_state_dict(state["objective"])
        objective.requires_grad_(True)
        functional = ConditionEnsemble([model])
        ensemble = (EnsembleBandObjective if state["objective_name"] == "band" else EnsembleMaskObjective)([objective], deduplicate_views=True)
        losses = []
        for loss, _, _ in process_future_group(
            functional, ensemble, [patients[p] for p in selected], weights, config,
            condition, reference, scale, maximum_updates_per_patient=4,
            detach_all_update=direct, stop_encoder_through_inner=encoder_stop,
        ):
            losses.append(loss)
        total = torch.stack(losses).sum()
        named = [(name, parameter) for name, parameter in [
            *model.named_parameters(),
            *((f"objective.{name}", parameter) for name, parameter in objective.named_parameters()),
        ] if parameter.requires_grad]
        values = torch.autograd.grad(total, [parameter for _, parameter in named], allow_unused=True)
        return named, values

    rows = []
    order = sorted(patients)
    for start in range(0, len(order), 4):
        selected = order[start:start + 4]
        full_names, full_gradients = gradients(selected)
        direct_names, direct_gradients = gradients(selected, direct=True)
        stop_names, stop_gradients = gradients(selected, encoder_stop=True)
        if [name for name, _ in full_names] != [name for name, _ in direct_names] or [name for name, _ in full_names] != [name for name, _ in stop_names]:
            raise RuntimeError("gradient parameter coordinates differ")
        row = {"patients": selected}
        for group, prefix in (
            ("encoder10", "backbone.encoder.layers.10."),
            ("encoder11", "backbone.encoder.layers.11."),
            ("detector", "detector."), ("ssl", "objective."),
        ):
            row[group] = statistics(
                vector(full_names, full_gradients, prefix),
                vector(direct_names, direct_gradients, prefix),
                vector(stop_names, stop_gradients, prefix),
            )
        rows.append(row)
        print(json.dumps({"groups": len(rows), "patients": min(start + 4, len(order))}), flush=True)
    groups = ("encoder10", "encoder11", "detector", "ssl")
    summary = {
        group: {name: float(np.median([row[group][name] for row in rows])) for name in rows[0][group]}
        for group in groups
    }
    ssl_direct_max = max(row["ssl"]["direct_norm"] for row in rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "patients": len(order), "groups": len(rows),
        "post_bce_ssl_direct_max_norm": ssl_direct_max,
        "summary_medians": summary, "rows": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
