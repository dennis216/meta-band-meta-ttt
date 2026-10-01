#!/usr/bin/env python3
"""Large-sample, state-replayed gradient diagnostics for v2 F and C semantics."""
from __future__ import annotations

import argparse
import fcntl
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bfa.evaluation.match import match_events
from bfa.tusz_meta_ttt.dataset import filter_inventory_records, load_cached_arrays
from bfa.tusz_meta_ttt.scoring import consecutive_eventize
from bfa.tusz_meta_ttt_v2.functional import SplitFunctionalTUSZModel
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.protocol import chunk_rows, stable_transform_seed
from bfa.tusz_meta_ttt_v2.runtime import load_source, make_windows
from bfa.tusz_meta_ttt_v2.update import (
    InnerStepConfig,
    fixed_sgd_inner_step,
    normalized_inner_step,
    packed_normalized_inner_step,
    parameter_block,
)

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"


def flatten(values) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in values])


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(left, right, dim=0).detach())


def position_bin(minutes: float) -> str:
    if minutes < 2:
        return "0-2"
    if minutes < 5:
        return "2-5"
    if minutes < 10:
        return "5-10"
    return "10+"


def candidate_group(labels, times, scores, truths, false_alarms, high_score_threshold):
    positive = labels > 0
    for event_index, (onset, _) in enumerate(truths):
        if any(abs(times[row] - onset) <= 4 for row in np.where(positive)[0]):
            return "onset", event_index
    for event_index, (_, offset) in enumerate(truths):
        if any(abs(times[row] - offset) <= 4 for row in np.where(positive)[0]):
            return "offset", event_index
    if positive.any():
        for event_index, (onset, offset) in enumerate(truths):
            if any(onset <= times[row] <= offset for row in np.where(positive)[0]):
                return "seizure", event_index
        return "seizure", None
    if any(alarm.start_s <= time < alarm.end_s for alarm in false_alarms for time in times):
        return "s1_false_alarm", None
    if np.max(scores, initial=-np.inf) >= high_score_threshold:
        return "high_score_background", None
    return "background", None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--source-threshold", type=float, required=True)
    parser.add_argument("--high-score-threshold", type=float)
    parser.add_argument("--partition", choices=["train", "dev", "eval"], required=True)
    parser.add_argument(
        "--cohort", choices=["all", "development_fit", "development_validation"], default="all"
    )
    parser.add_argument("--samples-per-group", type=int, default=1024)
    parser.add_argument("--maximum-per-patient-group", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument('--tag',default='')
    parser.add_argument('--method-probabilities',type=Path)
    parser.add_argument('--extended-controls',action='store_true')
    args = parser.parse_args()
    if args.high_score_threshold is None:
        args.high_score_threshold = args.source_threshold

    state = torch.load(args.meta_checkpoint, map_location="cpu", weights_only=False)
    run_name=args.meta_checkpoint.parent.name + ('_'+args.tag if args.tag else '')
    run=args.output_root/'mechanisms'/state['mode']/run_name/(args.partition if args.cohort=='all' else args.cohort)
    run.mkdir(parents=True,exist_ok=True)
    # Independent probes may run concurrently. Serialize the same output and
    # skip completed work before allocating a model or GPU memory.
    output_lock=(run/'.analysis.lock').open('a')
    fcntl.flock(output_lock.fileno(),fcntl.LOCK_EX)
    if (run/'summary.json').is_file():return
    if args.method_probabilities is None:
        scope=state.get('outer_scope','eds')
        stage=state.get('stage','development')
        name=(f'{scope}_formal_fast_{state["objective_name"]}_{scope}_seed{state["seed"]}' if stage=='formal' else f'{scope}_fast_{state["objective_name"]}_{scope}_epoch02')
        candidate=args.output_root/'evaluation'/state['mode']/name/(args.partition if args.cohort=='all' else args.cohort)/'probabilities.parquet'
        if candidate.is_file():args.method_probabilities=candidate
    torch.set_float32_matmul_precision('highest')
    if state.get('inner_kernel')=='packed':
        from bfa.tusz_meta_ttt_v2.batched import enable_second_order_batched_attention
        enable_second_order_batched_attention()
    model = load_source(Path(state["source"]), detector_trainable=False)
    model.load_state_dict(state["model"])
    functional = SplitFunctionalTUSZModel(model,prefix_cuda_graph=state.get('prefix_cuda_graph',False))
    objective = build_objective(state["objective_name"], float(state["difficulty"])).cuda().eval()
    objective.load_state_dict(state["objective"])
    objective.requires_grad_(False)
    config = InnerStepConfig(**state["inner_config"])
    inventory = json.loads((V1 / "manifests/records.json").read_text())
    lookup = {
        (item["patient_id"], item["session_id"], item["montage"], item["record_id"]): item
        for item in inventory
    }
    paths = filter_inventory_records(
        sorted((V1 / "cache" / args.partition).rglob("*.npz")), inventory,
        partition=args.partition,
    )
    if args.cohort != "all":
        if args.partition != "train":
            raise ValueError("development cohorts exist only inside official Train")
        allowed = set(json.loads((V1 / "manifests/development_split.json").read_text())[args.cohort])
        paths = [path for path in paths if path.parts[-4] in allowed]

    probability_frame = pd.read_parquet(args.probabilities)
    probability_groups = {
        tuple(key): group.sort_values("decision_end_s")
        for key, group in probability_frame.groupby(
            ["patient_id", "session_id", "montage", "record_id"], sort=False
        )
    }
    method_groups=None
    if args.method_probabilities:
        method_frame=pd.read_parquet(args.method_probabilities)
        method_groups={tuple(key):group.sort_values('decision_end_s')['adapted'].to_numpy() for key,group in method_frame.groupby(['patient_id','session_id','montage','record_id'],sort=False)}
    candidates = []
    for path in paths:
        key = (path.parts[-4], path.parts[-3], path.parts[-2], path.stem)
        if key not in probability_groups:
            continue
        archive = load_cached_arrays(path)
        group_frame = probability_groups[key]
        times = archive["decision_end_s"]
        np.testing.assert_allclose(times, group_frame["decision_end_s"].to_numpy())
        scores = group_frame["source_frozen"].to_numpy()
        truths = lookup[key]["seizures"]
        alarms = consecutive_eventize(times, scores, threshold=args.source_threshold)
        matched = match_events(alarms, truths)
        false_alarms = [alarms[index] for index in matched.unmatched_predictions]
        chunks = chunk_rows(times)
        if state["mode"] == "future":
            chunks = chunks[:-1]
        for chunk in chunks:
            indexes = np.asarray(chunk.rows)
            category, event_index = candidate_group(
                archive["labels"][indexes], times[indexes], scores[indexes], truths,
                false_alarms, args.high_score_threshold,
            )
            candidates.append((path, chunk.index, category, event_index))

    rng = np.random.default_rng(args.seed)
    category_lookup = {(row[0], row[1]): row[2] for row in candidates}
    selected: set[tuple[Path, int]] = set()
    candidate_counts = defaultdict(int)
    for category in ("seizure", "background", "high_score_background", "s1_false_alarm", "onset", "offset"):
        pool = [row for row in candidates if row[2] == category]
        candidate_counts[category] = len(pool)
        rng.shuffle(pool)
        patient_counts = defaultdict(int)
        event_counts = defaultdict(int)
        selected_count = 0
        for path, chunk_index, _, event_index in pool:
            patient = path.parts[-4]
            if patient_counts[patient] >= args.maximum_per_patient_group:
                continue
            event_key = (path, event_index)
            if event_index is not None and event_counts[event_key] >= 2:
                continue
            selected.add((path, chunk_index))
            patient_counts[patient] += 1
            if event_index is not None:
                event_counts[event_key] += 1
            selected_count += 1
            if selected_count >= args.samples_per_group:
                break

    outputs = []
    selected_by_path = defaultdict(set)
    for path, chunk_index in selected:
        selected_by_path[path].add(chunk_index)
    for record_number, (path, wanted) in enumerate(sorted(selected_by_path.items()), 1):
        archive = load_cached_arrays(path)
        signal, labels, times = archive["signal"], archive["labels"], archive["decision_end_s"]
        chunks = chunk_rows(times)
        initial = functional.initial_fast_parameters()
        fast = dict(initial)
        record_seed=stable_transform_seed(state['seed'],path.as_posix())
        for chunk_position, chunk in enumerate(chunks):
            if state["mode"] == "future" and chunk_position == len(chunks) - 1:
                break
            support = make_windows(signal, chunk.rows)
            location_seed = stable_transform_seed(record_seed, objective.name, chunk.index)
            transform_seed = [
                stable_transform_seed(record_seed, objective.name, row)
                for row in chunk.rows
            ]
            prefix_cache={}
            def prefix_for(tensor,key):
                if key not in prefix_cache:prefix_cache[key]=functional.prefix(tensor)
                return prefix_cache[key]
            def features_for(tensor,parameters,key):
                return functional.features_from_prefix(prefix_for(tensor,key),parameters)
            def logits_for(tensor,parameters,key='task'):
                return functional.logits_from_prefix(prefix_for(tensor,key),parameters)
            prepared_ssl = objective.prepare_loss(
                support, prefix_fn=lambda tensor:prefix_for(tensor,'same'), transform_seed=transform_seed
            )

            def ssl_loss(parameters, prepared_ssl=prepared_ssl):
                return prepared_ssl(
                    lambda prefix_features: functional.features_from_prefix(
                        prefix_features, dict(parameters)
                    )
                )

            should_measure = chunk.index in wanted
            task_chunk = chunks[chunk_position + 1] if state["mode"] == "future" else chunk
            task = make_windows(signal, task_chunk.rows)
            target = torch.from_numpy(labels[list(task_chunk.rows)]).cuda()
            if should_measure:
                same_before_metrics = objective.metrics(
                    support,
                    feature_fn=lambda tensor, parameters=fast: features_for(tensor,parameters,'same'),
                    transform_seed=transform_seed,
                )
                ssl_before = same_before_metrics["loss"]
                ssl_grad = torch.autograd.grad(ssl_before, tuple(fast.values()), retain_graph=True)
                task_logits = logits_for(task, fast)
                task_before = torch.nn.functional.binary_cross_entropy_with_logits(task_logits, target)
                task_grad = torch.autograd.grad(task_before, tuple(fast.values()))
                support_target = torch.from_numpy(labels[list(chunk.rows)]).cuda()
                support_task_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits_for(support, fast, 'support_task'), support_target
                )
                support_task_grad = torch.autograd.grad(support_task_loss, tuple(fast.values()))
                ssl_vector, task_vector = flatten(ssl_grad), flatten(task_grad)
            step_function = (
                normalized_inner_step
                if state.get("inner_rule", "normalized") == "normalized"
                else fixed_sgd_inner_step
            )
            if state.get('inner_rule','normalized')=='normalized' and state.get('inner_kernel')=='packed':
                step_function=packed_normalized_inner_step
            kwargs = (
                {"learning_rate": state["fixed_inner_lr"]}
                if state.get("inner_rule", "normalized") == "sgd" else {}
            )
            step = step_function(
                fast, loss_fn=ssl_loss, record_initial=initial, config=config,
                create_graph=False, **kwargs,
            )
            if should_measure:
                delta = flatten([step.parameters[name] - fast[name] for name in fast])
                with torch.no_grad():
                    post_logits = logits_for(task, step.parameters)
                    replay_error=float('nan')
                    if method_groups is not None:
                        key=(path.parts[-4],path.parts[-3],path.parts[-2],path.stem)
                        expected=method_groups[key][list(task_chunk.rows)]
                        replay_error=float(np.max(np.abs(torch.sigmoid(post_logits).cpu().numpy()-expected)))
                        if replay_error>5e-5:raise RuntimeError(f'Adaptation replay mismatch {path} chunk {chunk.index}: {replay_error}')
                    task_after = torch.nn.functional.binary_cross_entropy_with_logits(
                        post_logits, target
                    )
                    scaled_task_deltas = {}
                    for scale in (0.5, 1.0, 2.0):
                        scaled_parameters = {
                            name: fast[name] + scale * (step.parameters[name] - fast[name])
                            for name in fast
                        }
                        scaled_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits_for(task, scaled_parameters), target
                        )
                        scaled_task_deltas[f"step_{scale:g}x_delta_bce"] = float(
                            (scaled_loss - task_before).detach()
                        )
                    generator = torch.Generator(device="cuda").manual_seed(
                        location_seed + 30_000_000
                    )
                    random_delta = {}
                    for block in (10, 11):
                        names = [name for name in fast if parameter_block(name) == block]
                        values = {
                            name: torch.randn(
                                fast[name].shape,
                                generator=generator,
                                device=fast[name].device,
                                dtype=fast[name].dtype,
                            )
                            for name in names
                        }
                        norm = torch.sqrt(sum(value.float().square().sum() for value in values.values()))
                        target_norm = torch.sqrt(sum(
                            (step.parameters[name] - fast[name]).float().square().sum()
                            for name in names
                        ))
                        random_delta.update({
                            name: value * target_norm / norm.clamp_min(1e-12)
                            for name, value in values.items()
                        })
                    random_parameters = {
                        name: fast[name] + random_delta[name] for name in fast
                    }
                    random_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits_for(task, random_parameters), target
                    )
                    random_delta_bce = float((random_loss - task_before).detach())
                    def matched_supervised(grads):
                        candidate = dict(fast)
                        for block in (10, 11):
                            names = [name for name in fast if parameter_block(name) == block]
                            indexes = [list(fast).index(name) for name in names]
                            gradient_norm = flatten([grads[index] for index in indexes]).norm().clamp_min(1e-12)
                            update_norm = flatten([step.parameters[name] - fast[name] for name in names]).norm()
                            for name, index in zip(names, indexes, strict=True):
                                candidate[name] = fast[name] - grads[index] * update_norm / gradient_norm
                        return candidate
                    support_supervised = matched_supervised(support_task_grad)
                    local_supervised = matched_supervised(task_grad)
                    support_supervised_delta_bce = float(
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            logits_for(task, support_supervised), target) - task_before
                    )
                    local_supervised_delta_bce = float(
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            logits_for(task, local_supervised), target) - task_before
                    )
                same_after_metrics = objective.metrics(
                    support,
                    feature_fn=lambda tensor, parameters=step.parameters: features_for(tensor,parameters,'same'),
                    transform_seed=transform_seed,
                )
                same_after = same_after_metrics["loss"]
                independent_before_metrics = objective.metrics(
                    support,
                    feature_fn=lambda tensor, parameters=fast: features_for(tensor,parameters,'independent'),
                    transform_seed=[
                        stable_transform_seed(record_seed, objective.name, source_row, 1)
                        for source_row in chunk.rows
                    ],
                )
                independent_after_metrics = objective.metrics(
                    support,
                    feature_fn=lambda tensor, parameters=step.parameters: features_for(tensor,parameters,'independent'),
                    transform_seed=[
                        stable_transform_seed(record_seed, objective.name, source_row, 1)
                        for source_row in chunk.rows
                    ],
                )
                independent_before = independent_before_metrics["loss"]
                independent_after = independent_after_metrics["loss"]
                future_delta = float("nan")
                future_cosine = float("nan")
                future_ssl_delta = float("nan")
                if chunk_position + 1 < len(chunks):
                    future_chunk = chunks[chunk_position + 1]
                    future = make_windows(signal, future_chunk.rows)
                    future_target = torch.from_numpy(labels[list(future_chunk.rows)]).cuda()
                    future_key='task' if state['mode']=='future' else 'future_task'
                    future_logits = logits_for(future, fast,future_key)
                    future_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        future_logits, future_target
                    )
                    future_grad = torch.autograd.grad(future_loss, tuple(fast.values()))
                    future_cosine = cosine(ssl_vector, flatten(future_grad))
                    with torch.no_grad():
                        future_after = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits_for(future, step.parameters,future_key), future_target
                        )
                    future_delta = float((future_after - future_loss).detach())
                    future_ssl_before = objective.loss(
                        future,
                        feature_fn=lambda tensor, parameters=fast: features_for(tensor,parameters,'future_ssl'),
                        transform_seed=[
                            stable_transform_seed(record_seed, objective.name, source_row, 2)
                            for source_row in future_chunk.rows
                        ],
                    )
                    future_ssl_after = objective.loss(
                        future,
                        feature_fn=lambda tensor, parameters=step.parameters: features_for(tensor,parameters,'future_ssl'),
                        transform_seed=[
                            stable_transform_seed(record_seed, objective.name, source_row, 2)
                            for source_row in future_chunk.rows
                        ],
                    )
                    future_ssl_delta = float((future_ssl_after - future_ssl_before).detach())
                category = category_lookup[(path, chunk.index)]
                record_minutes = float(times[chunk.rows[-1]]) / 60
                row = {
                    "partition": args.partition,
                    "cohort": args.cohort,
                    "patient_id": path.parts[-4],
                    "record_id": path.stem,
                    "chunk_index": chunk.index,
                    "record_minutes": record_minutes,
                    "record_position_bin": position_bin(record_minutes),
                    "group": category,
                    "accepted": step.accepted,
                    "replay_probability_max_error": replay_error,
                    "raw_gradient_cosine": cosine(ssl_vector, task_vector),
                    "actual_descent_cosine": cosine(delta, -task_vector) if delta.norm() else float("nan"),
                    "gradient_dot_product": float(torch.dot(ssl_vector, task_vector).detach()),
                    "first_order_delta_bce": float(torch.dot(task_vector, delta).detach()),
                    "actual_delta_bce": float((task_after - task_before).detach()),
                    "median_delta_logit": float((post_logits - task_logits).abs().median().detach()),
                    "median_signed_delta_logit": float(
                        (post_logits - task_logits).median().detach()
                    ),
                    "ssl_same_delta": float((same_after - ssl_before).detach()),
                    "ssl_independent_delta": float((independent_after - independent_before).detach()),
                    "support_to_future_cosine": future_cosine,
                    "future_delta_bce": future_delta,
                    "ssl_future_independent_delta": future_ssl_delta,
                    "random_norm_matched_delta_bce": random_delta_bce,
                    "support_supervised_norm_matched_delta_bce": support_supervised_delta_bce,
                    "future_local_supervised_norm_matched_delta_bce": local_supervised_delta_bce,
                    **scaled_task_deltas,
                }
                other_support_delta = float("nan")
                if chunk_position > 0:
                    other_chunk = chunks[chunk_position - 1]
                    other_support = make_windows(signal, other_chunk.rows)
                    other_prepared = objective.prepare_loss(
                        other_support, prefix_fn=functional.prefix,
                        transform_seed=[
                            stable_transform_seed(
                                state["seed"], path.as_posix(), objective.name, source_row, 3
                            )
                            for source_row in other_chunk.rows
                        ],
                    )

                    def other_loss(
                        parameters, prepared=other_prepared
                    ):
                        return prepared(
                            lambda prefix_features: functional.features_from_prefix(
                                prefix_features, dict(parameters)
                            )
                        )

                    other_step = step_function(
                        fast, loss_fn=other_loss, record_initial=initial, config=config,
                        create_graph=False, **kwargs,
                    )
                    with torch.no_grad():
                        other_task_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            functional.logits(task, other_step.parameters), target
                        )
                    other_support_delta = float((other_task_loss - task_before).detach())
                row["other_observed_support_delta_bce"] = other_support_delta
                for metric_name in same_before_metrics:
                    row[f"ssl_same_{metric_name}_before"] = float(
                        same_before_metrics[metric_name].detach()
                    )
                    row[f"ssl_same_{metric_name}_after"] = float(
                        same_after_metrics[metric_name].detach()
                    )
                    row[f"ssl_same_{metric_name}_delta"] = float(
                        (same_after_metrics[metric_name] - same_before_metrics[metric_name]).detach()
                    )
                for metric_name in independent_before_metrics:
                    row[f"ssl_independent_{metric_name}_before"] = float(
                        independent_before_metrics[metric_name].detach()
                    )
                    row[f"ssl_independent_{metric_name}_after"] = float(
                        independent_after_metrics[metric_name].detach()
                    )
                    row[f"ssl_independent_{metric_name}_delta"] = float(
                        (
                            independent_after_metrics[metric_name]
                            - independent_before_metrics[metric_name]
                        ).detach()
                    )
                for block in (10, 11):
                    indexes = [
                        index for index, name in enumerate(fast) if parameter_block(name) == block
                    ]
                    row[f"ssl_gradient_norm_block_{block}"] = float(
                        flatten([ssl_grad[index] for index in indexes]).norm().detach()
                    )
                    row[f"task_gradient_norm_block_{block}"] = float(
                        flatten([task_grad[index] for index in indexes]).norm().detach()
                    )
                if args.extended_controls:
                    if chunk_position+1<len(chunks):
                        future_metric_seeds=[stable_transform_seed(record_seed,objective.name,source_row,2) for source_row in future_chunk.rows]
                        for moment,parameters in [('before',fast),('after',step.parameters)]:
                            future_metrics=objective.metrics(future,feature_fn=lambda tensor,parameters=parameters:features_for(tensor,parameters,'future_ssl'),transform_seed=future_metric_seeds)
                            for metric,value in future_metrics.items():row[f'ssl_future_{metric}_{moment}']=float(value.detach())
                    with torch.no_grad():
                        before_features=functional.features_from_prefix(prefix_for(task,'task'),fast)
                        after_features=functional.features_from_prefix(prefix_for(task,'task'),step.parameters)
                        row['task_bce_before']=float(task_before.detach())
                        row['task_bce_after']=float(task_after.detach())
                        row['task_feature_relative_change']=float((after_features-before_features).norm()/before_features.norm().clamp_min(1e-12))
                        for block in (10,11):
                            names=[name for name in fast if parameter_block(name)==block]
                            update_norm=flatten([step.parameters[n]-fast[n] for n in names]).norm()
                            before_drift=flatten([fast[n]-initial[n] for n in names]).norm()
                            after_drift=flatten([step.parameters[n]-initial[n] for n in names]).norm()
                            row[f'actual_update_norm_block_{block}']=float(update_norm)
                            row[f'relative_update_norm_block_{block}']=float(update_norm/config.block_reference_norms[block])
                            row[f'record_drift_before_block_{block}']=float(before_drift/config.block_reference_norms[block])
                            row[f'record_drift_after_block_{block}']=float(after_drift/config.block_reference_norms[block])
                        if objective.name=='band':
                            for phase in ['same','independent','future_ssl']:
                                if phase not in prefix_cache:continue
                                prefix=prefix_cache[phase]
                                count=prefix.shape[0]//5
                                truth=torch.arange(5,device='cuda').repeat_interleave(count)
                                for moment,parameters in [('before',fast),('after',step.parameters)]:
                                    features=functional.features_from_prefix(prefix,parameters)
                                    predicted=objective.head(features.mean(1).flatten(1)).argmax(1)
                                    matrix=torch.bincount(truth*5+predicted,minlength=25).reshape(5,5)
                                    row[f'band_{phase}_confusion_{moment}']=json.dumps(matrix.cpu().tolist())
                                    row[f'band_{phase}_accuracy_{moment}']=float(matrix.diag().sum()/matrix.sum())
                    row['other_observed_support_delta_bce']=float('nan')
                    if chunk_position>0:
                        previous_chunk=chunks[chunk_position-1]
                        other=make_windows(signal,previous_chunk.rows)
                        other_seeds=[stable_transform_seed(record_seed,objective.name,source_row) for source_row in previous_chunk.rows]
                        other_prepared=objective.prepare_loss(other,prefix_fn=functional.prefix,transform_seed=other_seeds)
                        other_step=step_function(fast,loss_fn=lambda parameters:other_prepared(lambda features:functional.features_from_prefix(features,dict(parameters))),record_initial=initial,config=config,create_graph=False,**kwargs)
                        with torch.no_grad():
                            other_loss=torch.nn.functional.binary_cross_entropy_with_logits(logits_for(task,other_step.parameters),target)
                            other_delta=flatten([other_step.parameters[name]-fast[name] for name in fast])
                            row['other_observed_support_delta_bce']=float(other_loss-task_before)
                            row['other_observed_support_accepted']=other_step.accepted
                            row['other_observed_support_descent_cosine']=cosine(other_delta,-task_vector) if other_delta.norm() else float('nan')
                outputs.append(row)
            fast = {
                name: value.detach().requires_grad_(True)
                for name, value in step.parameters.items()
            }
        print(json.dumps({"record": record_number, "total": len(selected_by_path)}), flush=True)

    frame = pd.DataFrame(outputs)
    run_name=args.meta_checkpoint.parent.name + ('_'+args.tag if args.tag else '')
    run = args.output_root / "mechanisms" / state["mode"] / run_name / (
        args.partition if args.cohort == "all" else args.cohort
    )
    run.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(run / "gradient_samples.parquet", index=False)
    numeric = [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {"chunk_index"}
    ]
    statistics = {}
    for group_name, group in frame.groupby("group"):
        statistics[group_name] = {
            column: {
                "mean": float(group[column].mean()),
                "median": float(group[column].median()),
                "std": float(group[column].std()),
            }
            for column in numeric
        }
    summary = {
        "mode": state["mode"],
        "partition": args.partition,
        "cohort": args.cohort,
        "candidate_counts": dict(candidate_counts),
        "sample_counts": frame.groupby("group").size().to_dict(),
        "position_counts": frame.groupby("record_position_bin").size().to_dict(),
        "position_statistics": {
            position: {
                "accepted_fraction": float(group["accepted"].mean()),
                "median_ssl_same_delta": float(group["ssl_same_delta"].median()),
                "median_actual_delta_bce": float(group["actual_delta_bce"].median()),
            }
            for position, group in frame.groupby("record_position_bin")
        },
        "statistics": statistics,
        "extended_controls":args.extended_controls,
    }
    matrix_columns=[column for column in frame.columns if '_confusion_' in column]
    if matrix_columns:
        summary['band_confusion_counts']={group_name:{column:np.stack([json.loads(value) for value in group[column].dropna()]).sum(0).tolist() for column in matrix_columns if group[column].notna().any()} for group_name,group in frame.groupby('group')}
    (run / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
