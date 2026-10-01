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
from bfa.tusz_meta_ttt.model import TUSZDetector
from bfa.tusz_meta_ttt.scoring import (
    choose_threshold,
    consecutive_eventize,
    score_at_threshold,
)
from bfa.tusz_meta_ttt_v2.functional import SplitFunctionalTUSZModel
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.protocol import (
    availability_times,
    chunk_rows,
    stable_transform_seed,
    validate_evaluation_request,
)
from bfa.tusz_meta_ttt_v2.update import (
    InnerStepConfig,
    fixed_sgd_inner_step,
    normalized_inner_step,
    packed_normalized_inner_step,
)

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"
PRETRAINED = ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"


def build_model(state_dict) -> TUSZDetector:
    adapter = CBraModAdapter(PRETRAINED, train_backbone=True)
    adapter.backbone.proj_out = torch.nn.Identity()
    model = TUSZDetector(adapter.backbone)
    model.load_state_dict(state_dict)
    return model.requires_grad_(False).cuda().eval()


def windows(signal, rows):
    values = np.stack([signal[:, row * 400 : row * 400 + 2000] for row in rows])
    return torch.from_numpy(values.reshape(-1, 16, 10, 200)).cuda()


def infer(functional, objective, archive, mode, config, seed, inner_rule, fixed_inner_lr, inner_kernel="reference"):
    signal, times = archive["signal"], archive["decision_end_s"]
    chunks = chunk_rows(times)
    initial = functional.initial_fast_parameters()
    fast = dict(initial)
    meta_frozen = np.empty(len(times), dtype=np.float32)
    adapted = np.empty(len(times), dtype=np.float32)
    available_at = availability_times(times, mode)
    diagnostics = []
    for chunk in chunks:
        batch = windows(signal, chunk.rows)
        batch_prefix = functional.prefix(batch)
        with torch.no_grad():
            meta_frozen[list(chunk.rows)] = torch.sigmoid(
                functional.logits_from_prefix(batch_prefix, initial)
            ).cpu().numpy()
            if mode == "future":
                adapted[list(chunk.rows)] = torch.sigmoid(
                    functional.logits_from_prefix(batch_prefix, fast)
                ).cpu().numpy()
        perform_update = mode == "current" or chunk != chunks[-1]
        if perform_update:
            transform_seed = [
                stable_transform_seed(seed, objective.name, row) for row in chunk.rows
            ]
            prepared_ssl = objective.prepare_loss(
                batch, prefix_fn=functional.prefix, transform_seed=transform_seed
            )

            def loss_fn(parameters, prepared_ssl=prepared_ssl):
                return prepared_ssl(
                    lambda prefix_features: functional.features_from_prefix(
                        prefix_features, dict(parameters)
                    )
                )

            step_function = normalized_inner_step if inner_rule == "normalized" else fixed_sgd_inner_step
            if inner_rule == "normalized" and inner_kernel == "packed":
                step_function = packed_normalized_inner_step
            kwargs = {"learning_rate": fixed_inner_lr} if inner_rule == "sgd" else {}
            result = step_function(
                fast, loss_fn=loss_fn, record_initial=initial, config=config,
                create_graph=False, **kwargs,
            )
            fast = {name: value.detach().requires_grad_(True) for name, value in result.parameters.items()}
            diagnostics.append({
                "chunk_index": chunk.index,
                "observed_end_s": float(times[chunk.rows[-1]]),
                "accepted": result.accepted,
                "reason": result.reason,
                "trial_scale": result.trial_scale,
                "ssl_loss_before": float(result.loss_before.detach()),
                "ssl_loss_after": float(result.loss_after.detach()),
                "block_gradient_norms": json.dumps(result.block_gradient_norms),
                "block_update_norms": json.dumps(result.block_update_norms),
            })
        if mode == "current":
            with torch.no_grad():
                adapted[list(chunk.rows)] = torch.sigmoid(
                    functional.logits_from_prefix(batch_prefix, fast)
                ).cpu().numpy()
    return meta_frozen, adapted, available_at, diagnostics


def common_detection_delay_difference(
    left_records: list[dict], left_threshold: float,
    right_records: list[dict], right_threshold: float,
) -> dict[str, float | int | None]:
    differences = []
    for left, right in zip(left_records, right_records, strict=True):
        if left["truths"] != right["truths"]:
            raise ValueError("paired conditions have different truth events")
        left_events = consecutive_eventize(
            left["times"], left["probabilities"], threshold=left_threshold
        )
        right_events = consecutive_eventize(
            right["times"], right["probabilities"], threshold=right_threshold
        )
        left_pairs = {
            pair.truth_index: max(
                0.0,
                left_events[pair.prediction_index].start_s
                - left["truths"][pair.truth_index][0],
            )
            for pair in match_events(left_events, left["truths"]).pairs
        }
        right_pairs = {
            pair.truth_index: max(
                0.0,
                right_events[pair.prediction_index].start_s
                - right["truths"][pair.truth_index][0],
            )
            for pair in match_events(right_events, right["truths"]).pairs
        }
        differences.extend(
            left_pairs[index] - right_pairs[index]
            for index in left_pairs.keys() & right_pairs.keys()
        )
    return {
        "common_detected_events": len(differences),
        "median_delay_difference_s": (
            float(np.median(differences)) if differences else None
        ),
    }


def retrospective_availability_metrics(
    records: list[dict], threshold: float
) -> dict[str, float | int | None]:
    availability_delays = []
    extra_waits = []
    for record in records:
        events = consecutive_eventize(
            record["times"], record["probabilities"], threshold=threshold
        )
        matched = match_events(events, record["truths"])
        for pair in matched.pairs:
            event = events[pair.prediction_index]
            truth_onset = record["truths"][pair.truth_index][0]
            index = int(np.searchsorted(record["times"], event.start_s))
            index = min(index, len(record["times"]) - 1)
            available_at = float(record["available_at"][index])
            availability_delays.append(max(0.0, available_at - truth_onset))
            extra_waits.append(max(0.0, available_at - event.start_s))
    return {
        "detected_events": len(availability_delays),
        "median_result_available_delay_s": (
            float(np.median(availability_delays)) if availability_delays else None
        ),
        "median_extra_wait_s": float(np.median(extra_waits)) if extra_waits else None,
        "maximum_extra_wait_s": float(np.max(extra_waits)) if extra_waits else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--partition", choices=["train", "dev", "eval"], required=True)
    parser.add_argument("--cohort", choices=["all", "development_fit", "development_validation"], default="all")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument(
        "--thresholds",
        type=Path,
        help="Dev summary.json whose three fixed thresholds are applied without recalibration",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["source_frozen", "meta_frozen", "adapted"],
        default=["source_frozen", "meta_frozen", "adapted"],
    )
    args = parser.parse_args()
    validate_evaluation_request(args.partition, args.calibrate, args.thresholds)
    state = torch.load(args.meta_checkpoint, map_location="cpu", weights_only=False)
    if state.get("training_contract", {}).get("version") in {
        "ensemble_v2_1", "meta_ttt_v3_1", "meta_ttt_v4_1"
    }:
        from bfa.tusz_meta_ttt_v2.batched import enable_second_order_batched_attention
        enable_second_order_batched_attention()
        torch.set_float32_matmul_precision(state["training_contract"]["matmul_precision"])
    model = build_model(state["model"])
    functional = SplitFunctionalTUSZModel(model,
        prefix_precision=state.get("prefix_precision", "fp32"),
        prefix_microbatch=state.get("prefix_microbatch", 0) or None,
        prefix_cuda_graph=state.get("prefix_cuda_graph", False))
    source = None
    if "source_frozen" in args.conditions:
        source_state = torch.load(state["source"], map_location="cpu", weights_only=False)["model"]
        source = build_model(source_state)
    objective = build_objective(state["objective_name"], float(state["difficulty"])).cuda().eval()
    objective.load_state_dict(state["objective"])
    objective.requires_grad_(False)
    config = InnerStepConfig(**state["inner_config"])
    inventory = json.loads((V1 / "manifests/records.json").read_text())
    paths = filter_inventory_records(
        sorted((V1 / "cache" / args.partition).rglob("*.npz")), inventory, partition=args.partition
    )
    if args.cohort != "all":
        if args.partition != "train":
            raise ValueError("development cohorts are defined only within Train")
        patients = set(json.loads((V1 / "manifests/development_split.json").read_text())[args.cohort])
        paths = [path for path in paths if path.parts[-4] in patients]
    if args.limit is not None:
        paths = paths[: args.limit]
    rows, update_rows = [], []
    records = {name: [] for name in args.conditions}
    lookup = {
        (item["patient_id"], item["session_id"], item["montage"], item["record_id"]): item
        for item in inventory
    }
    for record_index, path in enumerate(paths):
        archive = load_cached_arrays(path)
        signal, times, labels = archive["signal"], archive["decision_end_s"], archive["labels"]
        source_scores = None
        if source is not None:
            source_scores = []
            # Frozen source inference has no temporal dependency.  Larger batches
            # remove the 15-window chunk launch overhead without changing scores.
            for start in range(0, len(times), 64):
                batch_rows = tuple(range(start, min(len(times), start + 64)))
                with torch.no_grad():
                    source_scores.extend(
                        torch.sigmoid(source(windows(signal, batch_rows))).cpu().numpy()
                    )
        meta_frozen, adapted, available_at, diagnostics = infer(
            functional, objective, archive, state["mode"], config,
            stable_transform_seed(state["seed"], path.as_posix()),
            state.get("inner_rule", "normalized"), state.get("fixed_inner_lr"),
            state.get("inner_kernel", "reference"),
        )
        adapted_update_count = np.zeros(len(times), dtype=np.int32)
        accepted_by_chunk = {
            diagnostic["chunk_index"]: bool(diagnostic["accepted"])
            for diagnostic in diagnostics
        }
        accepted_updates = 0
        for chunk in chunk_rows(times):
            if state["mode"] == "current":
                accepted_updates += int(accepted_by_chunk.get(chunk.index, False))
            adapted_update_count[list(chunk.rows)] = accepted_updates
            if state["mode"] == "future":
                accepted_updates += int(accepted_by_chunk.get(chunk.index, False))
        key = (path.parts[-4], path.parts[-3], path.parts[-2], path.stem)
        item = lookup[key]
        update_rows.extend({
            "partition": args.partition,
            "patient_id": key[0],
            "session_id": key[1],
            "montage": key[2],
            "record_id": key[3],
            **diagnostic,
        } for diagnostic in diagnostics)
        all_scores = {"meta_frozen": meta_frozen, "adapted": adapted}
        if source_scores is not None:
            all_scores["source_frozen"] = np.asarray(source_scores)
        scores = {name: all_scores[name] for name in args.conditions}
        for name, values in scores.items():
            records[name].append({
                "times": times, "probabilities": values, "truths": item["seizures"],
                "duration_s": item["duration_s"], "available_at": available_at,
                "record_key": key,
            })
        for row, (time_s, label) in enumerate(zip(times, labels, strict=True)):
            rows.append({
                "partition": args.partition, "patient_id": key[0], "session_id": key[1],
                "montage": key[2], "record_id": key[3], "decision_end_s": float(time_s),
                "available_at_s": float(available_at[row]),
                "label": float(label), **{name: float(values[row]) for name, values in scores.items()},
                "adapted_update_count": int(adapted_update_count[row]),
                "adapted_state_version": f"record:{key[3]}:update:{adapted_update_count[row]}",
            })
        print(json.dumps({"record": record_index + 1, "total": len(paths)}), flush=True)
    cohort = args.partition if args.cohort == "all" else args.cohort
    run_name = args.meta_checkpoint.parent.name + (f"_{args.tag}" if args.tag else "")
    run = args.output_root / "evaluation" / state["mode"] / run_name / cohort
    run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(run / "probabilities.parquet", index=False)
    pd.DataFrame(update_rows).to_parquet(run / "adaptation_updates.parquet", index=False)
    summary = {
        "checkpoint": str(args.meta_checkpoint.resolve()), "mode": state["mode"],
        "records": len(paths), "conditions": {},
    }
    if args.calibrate:
        for name, values in records.items():
            choice = choose_threshold(values)
            summary["conditions"][name] = {
                "reachable": choice.reachable, "threshold": choice.threshold,
                "metrics": vars(choice.metrics) if choice.metrics else None,
                "maximum_sensitivity_threshold": choice.maximum_sensitivity_threshold,
                "maximum_sensitivity_metrics": vars(choice.maximum_sensitivity_metrics) if choice.maximum_sensitivity_metrics else None,
            }
    elif args.thresholds:
        calibration = json.loads(args.thresholds.read_text())
        for name, values in records.items():
            source_choice = calibration["conditions"][name]
            threshold = source_choice.get("threshold")
            if threshold is None:
                summary["conditions"][name] = {
                    "reachable_on_dev": False,
                    "threshold": None,
                    "metrics": None,
                }
                continue
            metrics = score_at_threshold(values, float(threshold))
            summary["conditions"][name] = {
                "reachable_on_dev": True,
                "threshold": float(threshold),
                "metrics": vars(metrics),
            }
    thresholds = {
        name: condition.get("threshold")
        for name, condition in summary["conditions"].items()
    }
    summary["paired_delay"] = {}
    for frozen_name in ("meta_frozen", "source_frozen"):
        if "adapted" not in records or frozen_name not in records:
            continue
        if thresholds.get("adapted") is None or thresholds.get(frozen_name) is None:
            continue
        summary["paired_delay"][f"adapted_vs_{frozen_name}"] = (
            common_detection_delay_difference(
                records["adapted"], float(thresholds["adapted"]),
                records[frozen_name], float(thresholds[frozen_name]),
            )
        )
    if state["mode"] == "current":
        summary["retrospective_availability"] = {
            name: retrospective_availability_metrics(records[name], float(threshold))
            for name, threshold in thresholds.items() if threshold is not None
        }
    alarm_rows = []
    for name, threshold in thresholds.items():
        if threshold is None:
            continue
        for record in records[name]:
            events = consecutive_eventize(
                record["times"], record["probabilities"], threshold=float(threshold)
            )
            matched = match_events(events, record["truths"])
            matched_predictions = {pair.prediction_index for pair in matched.pairs}
            key = record["record_key"]
            for event_index, event in enumerate(events):
                score_index = min(
                    int(np.searchsorted(record["times"], event.start_s)),
                    len(record["times"]) - 1,
                )
                alarm_rows.append({
                    "condition": name,
                    "patient_id": key[0],
                    "session_id": key[1],
                    "montage": key[2],
                    "record_id": key[3],
                    "alarm_start_s": event.start_s,
                    "alarm_end_s": event.end_s,
                    "peak_probability": event.peak_probability,
                    "result_available_s": float(record["available_at"][score_index]),
                    "matched": event_index in matched_predictions,
                })
    pd.DataFrame(alarm_rows).to_parquet(run / "alarm_log.parquet", index=False)
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
