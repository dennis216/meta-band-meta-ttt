#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.tusz_meta_ttt.dataset import filter_inventory_records, load_cached_arrays
from bfa.tusz_meta_ttt.model import TUSZDetector
from bfa.tusz_meta_ttt_v2.functional import SplitFunctionalTUSZModel
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.protocol import chunk_rows, stable_transform_seed
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig, normalized_inner_step, parameter_block

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


def windows(signal, rows):
    array = np.stack([signal[:, row * 400 : row * 400 + 2000] for row in rows])
    return torch.from_numpy(array.reshape(-1, 16, 10, 200)).cuda()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--objective-checkpoint", type=Path, required=True)
    parser.add_argument("--objective", choices=["band", "temporal", "mask"], required=True)
    parser.add_argument("--difficulty", type=float, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--stage", choices=["development", "formal"], default="development")
    parser.add_argument("--chunks-per-patient", type=int, default=4)
    parser.add_argument("--maximum-patients", type=int)
    parser.add_argument("--relative-steps", type=float, nargs="+", default=[1e-5, 3e-5, 1e-4])
    args = parser.parse_args()
    model = load_source(args.source.resolve())
    functional = SplitFunctionalTUSZModel(model)
    state = torch.load(args.objective_checkpoint, map_location="cpu", weights_only=False)
    objective = build_objective(args.objective, args.difficulty).cuda().eval()
    objective.load_state_dict(state["objective"])
    inventory = json.loads((V1 / "manifests/records.json").read_text())
    split = json.loads((V1 / "manifests/development_split.json").read_text())
    fit = (
        set(split["development_fit"])
        if args.stage == "development"
        else {item["patient_id"] for item in inventory if item["partition"] == "train"}
    )
    paths = [path for path in sorted((V1 / "cache/train").rglob("*.npz")) if path.parts[-4] in fit]
    paths = filter_inventory_records(paths, inventory, partition="train")
    by_patient: dict[str, list[tuple[Path, int, tuple[int, ...]]]] = defaultdict(list)
    for path in paths:
        times = load_cached_arrays(path)["decision_end_s"]
        by_patient[path.parts[-4]].extend((path, chunk.index, chunk.rows) for chunk in chunk_rows(times))
    patients = sorted(by_patient)
    if args.maximum_patients is not None:
        patients = patients[: args.maximum_patients]
    rng = np.random.default_rng(args.seed)
    gradient_norms: dict[int, list[float]] = defaultdict(list)
    samples = []
    selected_chunks = []
    reference_norms = {}
    base = functional.initial_fast_parameters()
    for block in (10, 11):
        reference_norms[block] = float(torch.sqrt(sum(
            value.float().square().sum() for name, value in base.items() if parameter_block(name) == block
        )).detach())
    for patient in patients:
        candidates = by_patient[patient]
        count = min(args.chunks_per_patient, len(candidates))
        selected = rng.choice(len(candidates), count, replace=False)
        for selected_index in selected:
            path, chunk_index, rows = candidates[int(selected_index)]
            signal = load_cached_arrays(path)["signal"]
            support = windows(signal, rows)
            fast = functional.initial_fast_parameters()
            prepared = objective.prepare_loss(
                support, prefix_fn=functional.prefix,
                transform_seed=[
                    stable_transform_seed(
                        args.seed, path.as_posix(), args.objective, row
                    )
                    for row in rows
                ],
            )
            loss = prepared(
                lambda prefix_features, fast=fast: functional.features_from_prefix(
                    prefix_features, fast
                )
            )
            gradients = torch.autograd.grad(loss, tuple(fast.values()))
            row = {"patient": patient, "record": path.stem, "chunk": chunk_index, "loss": float(loss.detach())}
            for block in (10, 11):
                norm = float(torch.sqrt(sum(
                    gradient.float().square().sum()
                    for (name, _), gradient in zip(fast.items(), gradients, strict=True)
                    if parameter_block(name) == block
                )).detach())
                gradient_norms[block].append(norm)
                row[f"block_{block}_gradient_norm"] = norm
            samples.append(row)
            selected_chunks.append((path, chunk_index, rows))
            print(json.dumps({"sample": len(samples), "patient": patient}), flush=True)
    medians = {block: float(np.median(values)) for block, values in gradient_norms.items()}
    candidate_diagnostics = {}
    for relative_step in args.relative_steps:
        config = InnerStepConfig(relative_step, reference_norms, medians)
        rows_out = []
        for path, chunk_index, rows in selected_chunks:
            signal = load_cached_arrays(path)["signal"]
            support = windows(signal, rows)
            fast = functional.initial_fast_parameters()
            prepared = objective.prepare_loss(
                support, prefix_fn=functional.prefix,
                transform_seed=[
                    stable_transform_seed(
                        args.seed, path.as_posix(), args.objective, row
                    )
                    for row in rows
                ],
            )

            def loss_fn(parameters, prepared=prepared):
                return prepared(
                    lambda prefix_features: functional.features_from_prefix(
                        prefix_features, dict(parameters)
                    )
                )

            with torch.no_grad():
                feature_before = functional.features(support, fast)
                logits_before = functional.logits(support, fast)
                logits_repeat = functional.logits(support, fast)
                repeated_forward_error = (logits_repeat - logits_before).abs().max()
            step = normalized_inner_step(
                fast, loss_fn=loss_fn, record_initial=fast, config=config, create_graph=False
            )
            with torch.no_grad():
                feature_after = functional.features(support, step.parameters)
                logits_after = functional.logits(support, step.parameters)
                feature_change = (feature_after - feature_before).float().norm() / feature_before.float().norm().clamp_min(1e-12)
                logit_change = (logits_after - logits_before).abs().median()
            planned = {
                block: relative_step * reference_norms[block] * (step.trial_scale or 0.0)
                for block in (10, 11)
            }
            actual_ratio = [
                step.block_update_norms.get(block, 0.0) / max(1e-30, planned[block])
                for block in (10, 11) if planned[block] > 0
            ]
            rows_out.append({
                "accepted": step.accepted,
                "reason": step.reason,
                "trial_scale": step.trial_scale,
                "feature_relative_change": float(feature_change),
                "median_absolute_logit_change": float(logit_change),
                "maximum_repeated_forward_error": float(repeated_forward_error),
                "actual_to_planned_update_ratio": min(actual_ratio) if actual_ratio else 0.0,
                "ssl_loss_before": float(step.loss_before.detach()),
                "ssl_loss_after": float(step.loss_after.detach()),
            })
        candidate_diagnostics[f"{relative_step:g}"] = {
            "chunks": len(rows_out),
            "accepted_fraction": float(np.mean([row["accepted"] for row in rows_out])),
            "median_feature_relative_change": float(np.median([row["feature_relative_change"] for row in rows_out])),
            "median_absolute_logit_change": float(np.median([row["median_absolute_logit_change"] for row in rows_out])),
            "maximum_repeated_forward_error": float(max(
                row["maximum_repeated_forward_error"] for row in rows_out
            )),
            "median_actual_to_planned_update_ratio": float(np.median([row["actual_to_planned_update_ratio"] for row in rows_out])),
            "minimum_actual_to_planned_update_ratio": float(min(
                (
                    row["actual_to_planned_update_ratio"]
                    for row in rows_out if row["accepted"]
                ),
                default=0.0,
            )),
            "rows": rows_out,
        }
    qualified = []
    for relative_step in sorted(args.relative_steps):
        candidate = candidate_diagnostics[f"{relative_step:g}"]
        candidate["passes_update_acceptance"] = (
            candidate["accepted_fraction"] >= 0.95
            and candidate["median_feature_relative_change"] >= 1e-4
            and candidate["median_absolute_logit_change"]
            >= max(1e-3, 10 * candidate["maximum_repeated_forward_error"])
            and candidate["minimum_actual_to_planned_update_ratio"] >= 0.90
        )
        if candidate["passes_update_acceptance"]:
            qualified.append(relative_step)
    result = {
        "objective": args.objective,
        "difficulty": args.difficulty,
        "source": str(args.source.resolve()),
        "objective_checkpoint": str(args.objective_checkpoint.resolve()),
        "patients": len(patients),
        "chunks": len(samples),
        "block_reference_norms": {str(key): value for key, value in reference_norms.items()},
        "block_gradient_medians": {str(key): value for key, value in medians.items()},
        "block_gradient_quantiles": {
            str(key): np.quantile(value, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]).tolist()
            for key, value in gradient_norms.items()
        },
        "samples": samples,
        "relative_step_candidates": candidate_diagnostics,
        "selected_relative_step": min(qualified) if qualified else None,
        "health_check_passed": bool(qualified),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    run = OUT / "calibration"
    if args.stage == "formal":
        run = run / "formal"
    run = run / f"{args.objective}_{args.difficulty:g}_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "gradient_calibration.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
