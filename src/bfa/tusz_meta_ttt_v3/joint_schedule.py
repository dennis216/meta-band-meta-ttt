"""Future-only v3 schedule with paired query, alignment, and damage losses."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from bfa.tusz_meta_ttt.dataset import load_cached_arrays
from bfa.tusz_meta_ttt_v2.batched import lane_features, normalized_lane_step, prepare_ssl_lanes
from bfa.tusz_meta_ttt_v2.protocol import chunk_rows, stable_transform_seed
from .losses import (
    PenaltyScale,
    V3Condition,
    actual_update_alignment,
    condition_objective,
    damage_components,
    path_key,
    weighted_group_sums,
)


def process_future_group(
    functional,
    objective,
    paths_by_patient,
    weights,
    config,
    conditions: tuple[V3Condition, ...],
    high_score_lookup,
    high_score_threshold: float,
    penalty_scale: PenaltyScale,
    *,
    seed: int = 3407,
    compact: bool = True,
    audit=None,
    maximum_updates_per_patient: int | None = None,
    stop_encoder_through_inner: bool = False,
):
    condition_count = len(conditions)
    patients = len(paths_by_patient)
    initial = functional.initial_lane_parameters(condition_count * patients)
    fast = initial
    device = next(iter(initial.values())).device
    states = [dict(paths=paths, index=0, sampled_updates=0) for paths in paths_by_patient]
    raw_cache = OrderedDict()
    pending = []
    ticks = 0
    update_count = 0
    accepted_count = torch.zeros((), device=device)

    def advance(state):
        while state["index"] < len(state["paths"]):
            path = state["paths"][state["index"]]
            archive = load_cached_arrays(path)
            chunks = chunk_rows(archive["decision_end_s"])
            if chunks:
                reference = high_score_lookup[path_key(path)]
                if len(reference) != len(archive["labels"]):
                    raise ValueError(f"high-score reference length mismatch: {path}")
                state.update(path=path, archive=archive, chunks=chunks, reference=reference,
                             chunk=0, new=True, active=True)
                return
            state["index"] += 1
        state.update(new=False, active=False)

    for state in states:
        advance(state)

    def raw_prefix(patient, rows):
        if not rows:
            return torch.zeros(15, 16, 10, 200, device=device)
        state = states[patient]
        key = (state["path"], rows)
        if key in raw_cache:
            raw_cache.move_to_end(key)
            return raw_cache[key]
        values = np.zeros((15, 16, 10, 200), dtype=np.float32)
        for j, row in enumerate(rows):
            values[j] = state["archive"]["signal"][:, row * 400:row * 400 + 2000].reshape(16, 10, 200)
        prefix = functional.functionals[0].prefix(torch.from_numpy(values).to(device))
        raw_cache[key] = prefix
        while len(raw_cache) > 3 * patients:
            raw_cache.popitem(last=False)
        return prefix

    def query_inputs(rows_by_lane):
        prefixes = []
        labels = np.zeros((condition_count * patients, 15), dtype=np.float32)
        mass = np.zeros_like(labels)
        high = np.zeros_like(labels, dtype=bool)
        for lane, rows in enumerate(rows_by_lane):
            patient = lane % patients
            prefixes.append(raw_prefix(patient, rows))
            if rows:
                state = states[patient]
                indexes = list(rows)
                labels[lane, :len(rows)] = state["archive"]["labels"][indexes]
                mass[lane, :len(rows)] = weights[state["path"]][indexes]
                high[lane, :len(rows)] = (
                    (state["archive"]["labels"][indexes] <= 0)
                    & (state["reference"][indexes] >= high_score_threshold)
                )
        return (
            torch.stack(prefixes),
            torch.from_numpy(labels).to(device),
            torch.from_numpy(mass).to(device),
            torch.from_numpy(high).to(device),
        )

    def logits(prefix, parameters):
        features = lane_features(functional, prefix, parameters)
        return functional.detect_lanes(features.mean(2).flatten(2)).squeeze(-1)

    def base_query(rows_by_lane, parameters):
        prefix, labels, mass, _ = query_inputs(rows_by_lane)
        per_window = torch.nn.functional.binary_cross_entropy_with_logits(
            logits(prefix, parameters), labels, reduction="none"
        )
        per_condition = (per_window * mass).reshape(condition_count, -1).sum(1)
        if audit is not None:
            # The first chunk has no preceding update. Keep its direct term
            # separate so paired adaptation gain excludes static training.
            audit["base_bce"] = audit.get("base_bce", 0) + per_condition.detach()
            audit["windows"] = audit.get("windows", 0) + torch.tensor(
                [sum(len(rows) for rows in rows_by_lane[c * patients:(c + 1) * patients])
                 for c in range(condition_count)], device=device
            )
        return per_condition.sum()

    def paired_query(rows_by_lane, before, after, accepted):
        prefix, labels, mass, high = query_inputs(rows_by_lane)
        pre_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits(prefix, before), labels, reduction="none"
        )
        post_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits(prefix, after), labels, reduction="none"
        )
        post_by_condition = (post_bce * mass).reshape(condition_count, -1).sum(1)
        pre_by_condition = (pre_bce * mass).reshape(condition_count, -1).sum(1)
        damage = damage_components(pre_bce, post_bce, labels, mass, high)
        group_mass = weighted_group_sums(torch.ones_like(pre_bce), labels, mass, high)
        gradients = {}
        for group in ("seizure", "background", "high_background"):
            values = torch.autograd.grad(
                group_mass[group].new_tensor(0.0) +
                weighted_group_sums(pre_bce, labels, mass, high)[group].sum(),
                tuple(before.values()), retain_graph=True, create_graph=False, allow_unused=False,
            )
            gradients[group] = {name: value.detach() for name, value in zip(before, values, strict=True)}
        alignment_by_condition = []
        damage_by_condition = []
        for c in range(condition_count):
            lane_slice = slice(c * patients, (c + 1) * patients)
            before_c = {name: value[lane_slice] for name, value in before.items()}
            after_c = {name: value[lane_slice] for name, value in after.items()}
            group_alignment = {}
            for group in ("seizure", "background", "high_background"):
                target_c = {name: value[lane_slice] for name, value in gradients[group].items()}
                lane_mass = group_mass[group][lane_slice]
                group_alignment[group] = actual_update_alignment(
                    before_c, after_c, target_c,
                    valid=accepted[lane_slice] & (lane_mass > 0), mass=lane_mass,
                )
            alignment_by_condition.append(
                0.5 * group_alignment["seizure"]
                + 0.5 * group_alignment["background"]
                + group_alignment["high_background"]
            )
            damage_by_condition.append(
                0.5 * damage["seizure"][lane_slice].sum()
                + 0.5 * damage["background"][lane_slice].sum()
                + damage["high_background"][lane_slice].sum()
            )
        alignment_tensor = torch.stack(alignment_by_condition)
        damage_tensor = torch.stack(damage_by_condition)
        losses = torch.stack([
            condition_objective(post_by_condition[c], alignment_tensor[c], damage_tensor[c],
                                conditions[c], penalty_scale)
            for c in range(condition_count)
        ])
        if audit is not None:
            for name, value in {
                "post_bce": post_by_condition,
                "pre_bce": pre_by_condition,
                "alignment": alignment_tensor,
                "damage": damage_tensor,
                "seizure_damage": torch.stack([damage["seizure"][c * patients:(c + 1) * patients].sum() for c in range(condition_count)]),
                "background_damage": torch.stack([damage["background"][c * patients:(c + 1) * patients].sum() for c in range(condition_count)]),
                "high_background_damage": torch.stack([damage["high_background"][c * patients:(c + 1) * patients].sum() for c in range(condition_count)]),
            }.items():
                audit[name] = audit.get(name, 0) + value.detach()
            audit["paired_windows"] = audit.get("paired_windows", 0) + torch.tensor(
                [sum(len(rows) for rows in rows_by_lane[c * patients:(c + 1) * patients])
                 for c in range(condition_count)], device=device
            )
            audit["windows"] = audit.get("windows", 0) + torch.tensor(
                [sum(len(rows) for rows in rows_by_lane[c * patients:(c + 1) * patients])
                 for c in range(condition_count)], device=device
            )
        return losses.sum()

    while any(state["active"] for state in states):
        live = [i for i, state in enumerate(states) if state["active"]]
        if compact and len(live) < patients:
            indexes = torch.tensor(
                [c * patients + i for c in range(condition_count) for i in live], device=device
            )
            fast = {name: value.index_select(0, indexes) for name, value in fast.items()}
            states = [states[i] for i in live]
            patients = len(states)
            initial = functional.initial_lane_parameters(condition_count * patients)
        reset = [state["active"] and state["new"] for state in states]
        if any(reset):
            fast = {
                name: torch.stack([
                    initial[name][c * patients + i] if reset[i] else fast[name][c * patients + i]
                    for c in range(condition_count) for i in range(patients)
                ]) for name in initial
            }
            prime = [state["chunks"][0].rows if reset[i] else ()
                     for _condition in conditions for i, state in enumerate(states)]
            if any(prime):
                pending.append(base_query(prime, fast))
        signal = np.zeros((patients, 15, 16, 10, 200), dtype=np.float32)
        valid = np.zeros((condition_count, patients, 15), dtype=np.float32)
        seeds = []
        queries = []
        for patient, state in enumerate(states):
            rows = state["chunks"][state["chunk"]].rows if state["active"] else ()
            if rows:
                for j, row in enumerate(rows):
                    signal[patient, j] = state["archive"]["signal"][:, row * 400:row * 400 + 2000].reshape(16, 10, 200)
                record_seed = stable_transform_seed(seed, state["path"].as_posix())
                seeds.extend(stable_transform_seed(record_seed, objective.name, row) for row in rows)
            seeds.extend([seed] * (15 - len(rows)))
            if state["active"] and state["chunk"] + 1 < len(state["chunks"]):
                valid[:, patient, :len(rows)] = 1
        for _condition in conditions:
            for state in states:
                index = state["chunk"] + 1 if state["active"] else 0
                queries.append(state["chunks"][index].rows if state["active"] and index < len(state["chunks"]) else ())
        if valid.any():
            signals = torch.from_numpy(signal).to(device).repeat(condition_count, 1, 1, 1, 1)
            valid_tensor = torch.from_numpy(valid.reshape(condition_count * patients, 15)).to(device)
            prepared = prepare_ssl_lanes(
                objective, signals, valid_tensor, prefix_fn=functional.prefix,
                seeds=seeds * condition_count,
                prefix_unique_fn=getattr(functional, "prefix_unique", None),
            )
            before = fast
            result = normalized_lane_step(
                before, loss_fn=lambda parameters: prepared(
                    lambda features: lane_features(functional, features, parameters)
                ), record_initial=initial, config=config,
                stop_encoder_through_inner=stop_encoder_through_inner,
            )
            fast = result.parameters
            pending.append(paired_query(queries, before, fast, result.accepted))
            update_count += int((valid.sum(2) > 0).sum())
            accepted_count = accepted_count + result.accepted.sum()
            if audit is not None:
                audit["updates"] = audit.get("updates", 0) + torch.from_numpy((valid.sum(2) > 0).sum(1)).to(device)
                audit["accepted"] = audit.get("accepted", 0) + result.accepted.reshape(condition_count, patients).sum(1)
                audit["gradient_norm_sum"] = audit.get("gradient_norm_sum", 0) + result.gradient_norms.reshape(condition_count, patients, -1).sum(1)
            ticks += 1
        for patient_index, state in enumerate(states):
            if not state["active"]:
                continue
            if valid[:, patient_index].any():
                state["sampled_updates"] += 1
            state["new"] = False
            state["chunk"] += 1
            if (maximum_updates_per_patient is not None
                    and state["sampled_updates"] >= maximum_updates_per_patient):
                state["active"] = False
                continue
            if state["chunk"] >= len(state["chunks"]):
                state["index"] += 1
                advance(state)
        if ticks == 4:
            yield torch.stack(pending).sum(), update_count, accepted_count
            pending = []
            update_count = 0
            accepted_count = torch.zeros((), device=device)
            fast = {name: initial[name] + (value - initial[name]).detach() for name, value in fast.items()}
            ticks = 0
    if pending:
        yield torch.stack(pending).sum(), update_count, accepted_count
