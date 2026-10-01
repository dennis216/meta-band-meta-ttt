#!/usr/bin/env python3
"""Preflight checks required before the formal repaired release queue."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

SCRIPT_ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("retrain280_preflight", SCRIPT_ROOT / "280_retrain_band_ttt_v2.py")
if spec is None or spec.loader is None:
    raise ImportError("cannot import retraining release")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(temporary, path)


def grad_norms(values: list[torch.Tensor | None]) -> list[float | None]:
    return [None if value is None else float(value.detach().float().norm().cpu()) for value in values]


def check_branch(model, branch: str, x: torch.Tensor, ids: list[str]) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    objective = None if branch == "band" else m.LearnedAuxiliaryLoss().to(x.device)
    parameters = list(model.band_head.parameters()) if branch == "band" else list(objective.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    transformed, labels = m.deterministic_band_view(x, ids)
    raw_prefix = m.encode_prefix(model, x)
    transformed_prefix = m.encode_prefix(model, transformed)
    params = m.detached_tail_values(model, requires_grad=True)
    loss, _ = m.auxiliary_loss(model, objective, raw_prefix, transformed_prefix, params, labels, branch)
    values = list(params.values())
    gradients = torch.autograd.grad(loss, values, retain_graph=True, allow_unused=True)
    finite_inner = all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    result = m.clipped_inner_update(model, objective, raw_prefix, transformed_prefix, labels, params, branch, create_graph=True)
    updated, _, grad_norm, update_norm = result
    outer = m.detector_from_prefix(model, raw_prefix, updated).mean()
    outer_grad = torch.autograd.grad(outer, parameters, retain_graph=True, allow_unused=True)
    finite_outer = all(gradient is not None and torch.isfinite(gradient).all() for gradient in outer_grad)
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=0.0)
    objective_loss = outer
    optimizer.zero_grad(set_to_none=True)
    objective_loss.backward()
    optimizer.step()
    changed = any(not torch.equal(before_value, parameter.detach()) for before_value, parameter in zip(before, parameters, strict=True))
    return {"inner_loss": float(loss.detach().cpu()), "inner_grad_norms": grad_norms(list(gradients)), "inner_finite": bool(finite_inner), "outer_grad_norm": grad_norms(list(outer_grad)), "outer_finite": bool(finite_outer), "head_changed_after_optimizer_step": bool(changed), "update_norm": float(update_norm.detach().cpu()), "grad_norm": float(grad_norm.detach().cpu())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("/root/b_false_alarm_atlas/outputs/reports/meta-ttt-chbmit-v2-repaired"))
    parser.add_argument("--windows", type=Path, default=m.DEFAULT_WINDOWS)
    parser.add_argument("--fold-root", type=Path, default=m.DEFAULT_FOLDS)
    parser.add_argument("--cache-root", type=Path, default=m.DEFAULT_CACHE)
    parser.add_argument("--pretrained", type=Path, default=m.EXTERNAL_ROOT / "pretrained_weights/pretrained_weights.pth")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    m.set_seed(m.SEED + 9876)
    device = torch.device(args.device)
    model = m.model_from_official(args.pretrained, device)
    model.eval()
    source_hash = m.stable_classifier_hash(model)
    x = torch.randn(2, 16, 10, 200, device=device)
    ids = ["preflight-a", "preflight-b"]
    branch_results = {branch: check_branch(model, branch, x, ids) for branch in ("band", "learned")}

    # Mean batch gradient must equal the mean of independent-window gradients.
    transformed, labels = m.deterministic_band_view(x, ids)
    raw_prefix = m.encode_prefix(model, x)
    transformed_prefix = m.encode_prefix(model, transformed)
    params = m.detached_tail_values(model, requires_grad=True)
    batch_loss, _ = m.auxiliary_loss(model, None, raw_prefix, transformed_prefix, params, labels, "band")
    batch_grad = torch.autograd.grad(batch_loss, list(params.values()), retain_graph=True)
    independent: list[list[torch.Tensor]] = []
    for index in range(len(x)):
        p = m.detached_tail_values(model, requires_grad=True)
        one_loss, _ = m.auxiliary_loss(model, None, raw_prefix[index:index + 1], transformed_prefix[index:index + 1], p, labels[index:index + 1], "band")
        independent.append(list(torch.autograd.grad(one_loss, list(p.values()))))
    independent_mean = [sum(row[i] for row in independent) / len(independent) for i in range(len(batch_grad))]
    gradient_error = max(float((left - right).abs().max().detach().cpu()) for left, right in zip(batch_grad, independent_mean, strict=True))
    gradient_scale = max(float(left.abs().max().detach().cpu()) for left in batch_grad)
    gradient_relative_error = gradient_error / max(gradient_scale, 1e-12)

    rows = m.load_rows(0, "validation", args.windows, args.fold_root)
    records = m.records_from_rows(rows)
    record = records[0]
    store = m.SignalStore(args.cache_root)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.band_head.parameters():
        parameter.requires_grad_(True)
    hash_before = m.stable_classifier_hash(model)
    original_inner_lr = m.INNER_LR
    m.INNER_LR = 0.0
    zero_result = m.run_episode(model, None, store, record, int(record.candidate_starts[0]), device, "band", None, differentiable=False)
    m.INNER_LR = original_inner_lr
    zero_update_error = abs(float(zero_result["future_loss"].detach().cpu()) - float(zero_result["future_frozen_loss"].detach().cpu()))
    hash_after = m.stable_classifier_hash(model)

    class AlteringStore(m.SignalStore):
        def __init__(self, base: m.SignalStore, cutoff: int) -> None:
            super().__init__(base.cache_root)
            self.base = base
            self.cutoff = cutoff
        def read(self, record, positions):
            output = self.base.read(record, positions)
            if int(positions[0]) >= self.cutoff:
                output = output.copy()
                output += 0.123
            return output

    start = int(record.candidate_starts[0])
    normal_trace = m.run_episode(model, None, store, record, start, device, "band", None, differentiable=False)
    altered_trace = m.run_episode(model, None, AlteringStore(store, start + (m.BURN_IN_CHUNKS + 2) * m.CHUNK_STRIDE), record, start, device, "band", None, differentiable=False)
    causal_error = float((normal_trace["post0_logits"].detach() - altered_trace["post0_logits"].detach()).abs().max().cpu())

    cache_root = Path("/tmp/meta_ttt_preflight_cache")
    cache = m.PrefixCache(cache_root, hash_before, 2**30, enabled=True)
    cached_trace = m.run_episode(model, None, store, record, start, device, "band", cache, differentiable=False)
    cache_error = abs(float(cached_trace["future_loss"].detach().cpu()) - float(normal_trace["future_loss"].detach().cpu()))
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            bf16 = torch.sigmoid(model.detect(x).float())
        fp32 = torch.sigmoid(model.detect(x).float())
    bf16_error = float((bf16 - fp32).abs().max().cpu())
    report = {
        "release_id": m.RELEASE_ID, "status": "passed" if branch_results["band"]["inner_finite"] and branch_results["band"]["outer_finite"] and branch_results["learned"]["inner_finite"] and branch_results["learned"]["outer_finite"] and (gradient_error < 1e-4 or gradient_relative_error < 1e-3) and zero_update_error < 1e-6 and causal_error < 1e-6 and cache_error < 1e-6 and hash_before == hash_after else "failed",
        "source_classifier_hash_before": source_hash, "source_classifier_hash_after": hash_after, "source_hash_unchanged": source_hash == hash_after,
        "branch_results": branch_results, "batch_vs_independent_gradient_max_abs_error": gradient_error, "batch_vs_independent_gradient_relative_error": gradient_relative_error,
        "zero_update_future_vs_frozen_abs_error": zero_update_error, "future_rewrite_history_prediction_abs_error": causal_error,
        "cache_on_off_future_loss_abs_error": cache_error, "bf16_probability_max_abs_error": bf16_error,
        "record_used": {"patient": record.patient, "recording": record.recording, "candidate_start": start}, "completed_at": m.utc_now(),
    }
    atomic_json(args.output_root / "preflight" / "preflight.json", report)
    print(json.dumps(report, indent=2, allow_nan=True))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
