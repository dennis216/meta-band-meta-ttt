#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.tusz_meta_ttt.adaptation import module_hash
from bfa.tusz_meta_ttt.dataset import filter_inventory_records, load_cached_arrays
from bfa.tusz_meta_ttt.functional import FunctionalTUSZModel
from bfa.tusz_meta_ttt.model import TUSZDetector
from bfa.tusz_meta_ttt.objectives import build_objective
from bfa.tusz_meta_ttt.scoring import choose_threshold, consecutive_eventize, score_record
from bfa.tusz_meta_ttt.statistics import (
    PatientComponents,
    aggregate_components,
    paired_patient_bootstrap,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v1"
PRETRAINED = ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"


def load_source(path: Path):
    adapter = CBraModAdapter(PRETRAINED, train_backbone=True)
    adapter.backbone.proj_out = torch.nn.Identity()
    model = TUSZDetector(adapter.backbone)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return model.requires_grad_(False).cuda().eval()


def make_windows(signal, rows):
    array = np.stack([signal[:, row * 400 : row * 400 + 2000] for row in rows])
    return torch.from_numpy(array.reshape(-1, 16, 10, 200)).cuda()


def update_fast(functional, objective, support, fast, inner_lr, seed):
    loss = objective.loss(
        support,
        feature_fn=lambda x: functional.features(x, fast),
        fast_parameters=fast,
        transform_seed=seed,
    )
    gradients = torch.autograd.grad(loss, tuple(fast.values()))
    candidate = {
        name: (value - inner_lr * gradient).detach().requires_grad_(True)
        for (name, value), gradient in zip(fast.items(), gradients, strict=True)
    }
    return candidate if all(torch.isfinite(value).all() for value in candidate.values()) else fast


def infer_record(
    functional, objective, path, inner_lr, mode, seed, frozen_override=None
):
    archive = load_cached_arrays(path)
    signal = archive["signal"]
    times = archive["decision_end_s"]
    labels = archive["labels"]
    frozen = [] if frozen_override is None else list(frozen_override)
    adapted = []
    fast = functional.initial_fast_parameters()
    for chunk_start in range(0, len(labels), 15):
        rows = list(range(chunk_start, min(chunk_start + 15, len(labels))))
        batch = make_windows(signal, rows)
        with torch.no_grad():
            if frozen_override is None:
                frozen.extend(torch.sigmoid(functional.logits(batch, functional.initial_fast_parameters())).cpu().numpy())
            if mode in {"online", "online_no_carry"}:
                adapted.extend(torch.sigmoid(functional.logits(batch, fast)).cpu().numpy())
        if mode in {"online", "online_no_carry"} and len(rows) >= 11:
            support = make_windows(signal, [chunk_start, chunk_start + 5, chunk_start + 10])
            update_base = functional.initial_fast_parameters() if mode == "online_no_carry" else fast
            fast = update_fast(functional, objective, support, update_base, inner_lr, seed + chunk_start)
        elif mode == "same_window":
            for offset, row in enumerate(rows):
                local = functional.initial_fast_parameters()
                support = batch[offset : offset + 1]
                local = update_fast(functional, objective, support, local, inner_lr, seed + row)
                with torch.no_grad():
                    adapted.append(float(torch.sigmoid(functional.logits(support, local))[0]))
    if len(frozen) != len(labels):
        raise RuntimeError("cached Frozen probabilities do not align with record")
    return times, labels, np.asarray(frozen), np.asarray(adapted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--objective-checkpoint", type=Path, required=True)
    parser.add_argument("--partition", choices=["train", "dev", "eval"], required=True)
    parser.add_argument(
        "--cohort",
        choices=["all", "development_fit", "development_validation"],
        default="all",
        help="Patient cohort from the frozen development split (train only).",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Select independent Frozen/Adapted thresholds on this cohort.",
    )
    parser.add_argument(
        "--mode", choices=["online", "online_no_carry", "same_window"], required=True
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--variant-suffix", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--frozen-threshold", type=float)
    parser.add_argument("--adapted-threshold", type=float)
    parser.add_argument(
        "--frozen-probabilities",
        type=Path,
        help="Reuse exact Frozen scores from a prior evaluation of this source.",
    )
    args = parser.parse_args()
    objective_state = torch.load(args.objective_checkpoint, map_location="cpu", weights_only=False)
    objective = build_objective(objective_state["objective_name"]).cuda().eval()
    objective.load_state_dict(objective_state["objective"])
    inner_lr = float(objective_state["inner_lr"])
    model = load_source(args.source.resolve())
    source_hash = module_hash(model)
    functional = FunctionalTUSZModel(model)
    inventory = json.loads((OUT / "manifests/records.json").read_text())
    if args.cohort != "all" and args.partition != "train":
        parser.error("development cohorts are only defined for --partition train")
    if args.cohort != "all":
        split = json.loads((OUT / "manifests/development_split.json").read_text())
        patients = set(split[args.cohort])
        inventory = [item for item in inventory if item["patient_id"] in patients]
    lookup = {(item["patient_id"], item["session_id"], item["montage"], item["record_id"]): item for item in inventory}
    paths = sorted((OUT / "cache" / args.partition).rglob("*.npz"))
    paths = filter_inventory_records(paths, inventory, partition=args.partition)
    if args.limit is not None:
        paths = paths[: args.limit]
    rows = []
    calibration_records = {"frozen": [], "adapted": []}
    frozen_lookup = {}
    if args.frozen_probabilities is not None:
        frozen_frame = pd.read_parquet(args.frozen_probabilities)
        group_keys = ["patient_id", "session_id", "montage", "record_id"]
        frozen_lookup = {
            tuple(key): group["frozen_probability"].to_numpy(float)
            for key, group in frozen_frame.groupby(group_keys, sort=False)
        }
    for record_index, path in enumerate(paths):
        key = (path.parts[-4], path.parts[-3], path.parts[-2], path.stem)
        override = frozen_lookup.get(key) if frozen_lookup else None
        if frozen_lookup and override is None:
            cached = load_cached_arrays(path)
            if len(cached["labels"]) == 0:
                override = np.empty(0, dtype=float)
            else:
                raise RuntimeError(f"Frozen cache is missing {key}")
        times, labels, frozen, adapted = infer_record(
            functional,
            objective,
            path,
            inner_lr,
            args.mode,
            args.seed + record_index * 100000,
            override,
        )
        item = lookup[key]
        for condition, probabilities in (("frozen", frozen), ("adapted", adapted)):
            calibration_records[condition].append({"patient_id": item["patient_id"], "times": times, "probabilities": probabilities, "truths": item["seizures"], "duration_s": item["duration_s"]})
        rows.extend({"partition": args.partition, "patient_id": key[0], "session_id": key[1], "montage": key[2], "record_id": key[3], "decision_end_s": float(time_s), "label": float(label), "frozen_probability": float(pf), "adapted_probability": float(pa)} for time_s, label, pf, pa in zip(times, labels, frozen, adapted, strict=True))
        print(json.dumps({"record": record_index + 1, "total": len(paths), "path": str(path)}), flush=True)
    cohort_name = args.partition if args.cohort == "all" else args.cohort
    variant = f"{objective_state['objective_name']}_lr{inner_lr:g}"
    training = objective_state.get("training")
    if training:
        variant = f"{variant}_{training}"
    if args.variant_suffix:
        if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in args.variant_suffix):
            raise ValueError("variant suffix contains unsupported characters")
        variant = f"{variant}_{args.variant_suffix}"
    run = OUT / "evaluation" / variant / args.mode / f"seed{args.seed}" / cohort_name
    run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(run / "probabilities.parquet", index=False)
    summary = {"variant": variant, "objective_checkpoint": str(args.objective_checkpoint.resolve()), "source_hash_before": source_hash, "source_hash_after": module_hash(model), "records": len(paths), "conditions": {}}
    patient_outputs = {}
    for condition, records in calibration_records.items():
        supplied_threshold = args.frozen_threshold if condition == "frozen" else args.adapted_threshold
        if (args.partition == "dev" or args.calibrate) and supplied_threshold is None:
            choice = choose_threshold(records)
            threshold = choice.threshold
            summary["conditions"][condition] = {
                "threshold": threshold,
                "reachable": choice.reachable,
                "metrics": vars(choice.metrics) if choice.metrics else None,
                "maximum_sensitivity_threshold": choice.maximum_sensitivity_threshold,
                "maximum_sensitivity_metrics": (
                    vars(choice.maximum_sensitivity_metrics)
                    if choice.maximum_sensitivity_metrics
                    else None
                ),
            }
        else:
            threshold = supplied_threshold
            summary["conditions"][condition] = {
                "threshold": threshold,
                "reachable": threshold is not None,
            }
        if threshold is not None:
            patient_rows = {}
            for record in records:
                events = consecutive_eventize(record["times"], record["probabilities"], threshold=threshold)
                metrics = score_record(events, record["truths"], duration_s=record["duration_s"])
                patient = record["patient_id"]
                row = patient_rows.setdefault(patient, {"detected": 0, "total": 0, "false_alarms": 0, "background_hours": 0.0, "false_alarm_seconds": 0.0})
                seizure_seconds = sum(end - start for start, end in record["truths"])
                background_hours = max(0.0, (record["duration_s"] - seizure_seconds) / 3600)
                row["detected"] += metrics.detected
                row["total"] += metrics.total
                row["false_alarms"] += metrics.false_alarms
                row["background_hours"] += background_hours
                row["false_alarm_seconds"] += metrics.false_alarm_minutes_per_hour * 60 * background_hours
            components = [PatientComponents(patient, **values) for patient, values in patient_rows.items()]
            patient_outputs[condition] = components
            summary["conditions"][condition].update({"aggregate": aggregate_components(components), "patients": [vars(value) for value in components]})
    if set(patient_outputs) == {"frozen", "adapted"}:
        summary["paired_bootstrap"] = paired_patient_bootstrap(
            patient_outputs["frozen"], patient_outputs["adapted"], replicates=10_000, seed=args.seed
        )
    if summary["source_hash_before"] != summary["source_hash_after"]:
        raise RuntimeError("evaluation mutated the source model")
    (run / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
