#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
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
from bfa.tusz_meta_ttt_v2.protocol import (
    assert_train_cache_paths,
    chunk_rows,
    class_patient_record_weights,
    stable_transform_seed,
    weight_audit,
)
from bfa.tusz_meta_ttt_v2.update import (
    InnerStepConfig,
    fixed_sgd_inner_step,
    normalized_inner_step,
    packed_normalized_inner_step,
    reattach_initialization,
)

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"
PRETRAINED = ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"


def load_source(path: Path) -> TUSZDetector:
    adapter = CBraModAdapter(PRETRAINED, train_backbone=True)
    adapter.backbone.proj_out = torch.nn.Identity()
    model = TUSZDetector(adapter.backbone, dropout=0.1)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith(("backbone.encoder.layers.10.", "backbone.encoder.layers.11.")):
            parameter.requires_grad_(True)
    return model.cuda().eval()


def make_windows(signal: np.ndarray, rows: tuple[int, ...]) -> torch.Tensor:
    values = np.stack([signal[:, row * 400 : row * 400 + 2000] for row in rows])
    return torch.from_numpy(values.reshape(-1, 16, 10, 200)).cuda(non_blocking=True)


def optimizer_groups(module: torch.nn.Module, prefix: str, lr: float, decay: float) -> list[dict]:
    regular, exempt = [], []
    for name, parameter in module.named_parameters():
        if parameter.requires_grad and name.startswith(prefix):
            (exempt if name.endswith("bias") or parameter.ndim == 1 else regular).append(parameter)
    groups = []
    if regular:
        groups.append({"params": regular, "lr": lr, "weight_decay": decay})
    if exempt:
        groups.append({"params": exempt, "lr": lr, "weight_decay": 0.0})
    return groups


def weighted_bce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (losses * weights).sum()


def l2_change(parameters: list[torch.Tensor], before: list[torch.Tensor]) -> float:
    return float(torch.sqrt(sum(
        (parameter.detach().float().cpu() - old.float()).square().sum()
        for parameter, old in zip(parameters, before, strict=True)
    )).cpu())


def process_record(
    functional: SplitFunctionalTUSZModel,
    objective: torch.nn.Module,
    path: Path,
    row_weights: np.ndarray,
    mode: str,
    inner_config: InnerStepConfig,
    transform_seed: int,
    truncate_updates: int,
    inner_rule: str,
    fixed_inner_lr: float | None,
    stop_encoder_through_inner: bool,
    inner_kernel: str = "reference",
):
    archive = load_cached_arrays(path)
    signal, labels, times = archive["signal"], archive["labels"], archive["decision_end_s"]
    chunks = chunk_rows(times)
    initial = functional.initial_fast_parameters(detach=False)
    fast = dict(initial)
    pending_losses: list[torch.Tensor] = []
    pending_diagnostics: list[dict] = []
    updates_since_truncation = 0

    def query_loss(rows: tuple[int, ...], parameters) -> torch.Tensor:
        query = make_windows(signal, rows)
        query_prefix = functional.prefix(query)
        target = torch.from_numpy(labels[list(rows)]).cuda()
        weights = torch.from_numpy(row_weights[list(rows)]).float().cuda()
        return weighted_bce(
            functional.logits_from_prefix(query_prefix, parameters), target, weights
        )

    if mode == "future" and chunks:
        pending_losses.append(query_loss(chunks[0].rows, fast))

    update_chunks = chunks[:-1] if mode == "future" else chunks
    for chunk_index, chunk in enumerate(update_chunks):
        support = make_windows(signal, chunk.rows)
        seed = [
            stable_transform_seed(transform_seed, objective.name, row) for row in chunk.rows
        ]
        prepared_ssl = objective.prepare_loss(
            support, prefix_fn=functional.prefix, transform_seed=seed
        )

        def ssl_loss(parameters, prepared_ssl=prepared_ssl):
            return prepared_ssl(
                lambda prefix_features: functional.features_from_prefix(
                    prefix_features, dict(parameters)
                )
            )

        inner_parameters = fast
        inner_initial = initial
        if stop_encoder_through_inner:
            inner_parameters = {
                name: value.detach().requires_grad_(True) for name, value in fast.items()
            }
            inner_initial = {
                name: initial[name].detach() + (value - initial[name]).detach()
                for name, value in fast.items()
            }
        step_function = normalized_inner_step if inner_rule == "normalized" else fixed_sgd_inner_step
        if inner_rule == "normalized" and inner_kernel == "packed":
            step_function = packed_normalized_inner_step
        step_kwargs = {}
        if inner_rule == "sgd":
            if fixed_inner_lr is None:
                raise ValueError("fixed_inner_lr is required for the SGD mechanism control")
            step_kwargs["learning_rate"] = fixed_inner_lr
        result = step_function(
            inner_parameters,
            loss_fn=ssl_loss,
            record_initial=inner_initial,
            config=inner_config,
            create_graph=True,
            **step_kwargs,
        )
        if stop_encoder_through_inner:
            fast = {
                name: fast[name] + (result.parameters[name] - inner_parameters[name])
                for name in fast
            }
        else:
            fast = result.parameters
        query = chunks[chunk_index + 1] if mode == "future" else chunk
        pending_losses.append(query_loss(query.rows, fast))
        pending_diagnostics.append(
            {
                "accepted": result.accepted,
                "reason": result.reason,
                "trial_scale": result.trial_scale,
                "ssl_before": float(result.loss_before.detach()),
                "ssl_after": float(result.loss_after.detach()),
                "gradient_norms": result.block_gradient_norms,
                "update_norms": result.block_update_norms,
            }
        )
        updates_since_truncation += 1
        if updates_since_truncation == truncate_updates:
            yield torch.stack(pending_losses).sum(), pending_diagnostics
            pending_losses, pending_diagnostics = [], []
            fast = reattach_initialization(fast, initial)
            updates_since_truncation = 0
    if pending_losses:
        yield torch.stack(pending_losses).sum(), pending_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--objective-checkpoint", type=Path, required=True)
    parser.add_argument("--objective", choices=["band", "temporal", "mask"], required=True)
    parser.add_argument("--difficulty", type=float, required=True)
    parser.add_argument("--mode", choices=["future", "current"], required=True)
    parser.add_argument("--outer-scope", choices=["e", "ed", "es", "eds"], required=True)
    parser.add_argument("--relative-step", type=float, choices=[1e-5, 3e-5, 1e-4], required=True)
    parser.add_argument("--gradient-calibration", type=Path, required=True)
    parser.add_argument("--stage", choices=["development", "formal"], default="development")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--truncate-updates", type=int, default=4)
    parser.add_argument("--inner-rule", choices=["normalized", "sgd"], default="normalized")
    parser.add_argument("--inner-kernel", choices=["reference", "packed"], default="reference")
    parser.add_argument("--prefix-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--prefix-microbatch", type=int, default=0)
    parser.add_argument("--prefix-cuda-graph", action="store_true")
    parser.add_argument("--fixed-inner-lr", type=float)
    parser.add_argument("--stop-encoder-through-inner", action="store_true")
    parser.add_argument("--detector-freeze-fraction", type=float, default=0.25)
    parser.add_argument("--maximum-records", type=int)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.detector_freeze_fraction <= 1:
        raise ValueError("detector freeze fraction must be in [0, 1]")
    if args.inner_rule == "sgd" and args.fixed_inner_lr is None:
        raise ValueError("--fixed-inner-lr is required with --inner-rule sgd")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model = load_source(args.source.resolve())
    functional = SplitFunctionalTUSZModel(
        model,
        prefix_precision=args.prefix_precision,
        prefix_microbatch=args.prefix_microbatch or None,
        prefix_cuda_graph=args.prefix_cuda_graph,
    )
    head_state = torch.load(args.objective_checkpoint, map_location="cpu", weights_only=False)
    objective = build_objective(args.objective, args.difficulty).cuda().eval()
    objective.load_state_dict(head_state["objective"], strict=True)
    objective.requires_grad_("s" in args.outer_scope)
    model.detector.requires_grad_("d" in args.outer_scope)
    calibration = json.loads(args.gradient_calibration.read_text())
    reference_norms = {
        block: float(calibration["block_reference_norms"][str(block)]) for block in (10, 11)
    }
    gradient_medians = {
        block: float(calibration["block_gradient_medians"][str(block)]) for block in (10, 11)
    }
    inner_config = InnerStepConfig(args.relative_step, reference_norms, gradient_medians)

    inventory = json.loads((V1 / "manifests/records.json").read_text())
    split = json.loads((V1 / "manifests/development_split.json").read_text())
    patients = (
        set(split["development_fit"])
        if args.stage == "development"
        else {item["patient_id"] for item in inventory if item["partition"] == "train"}
    )
    paths = [path for path in sorted((V1 / "cache/train").rglob("*.npz")) if path.parts[-4] in patients]
    paths = filter_inventory_records(paths, inventory, partition="train")
    assert_train_cache_paths(paths, V1 / "cache/train")
    if args.maximum_records is not None:
        paths = paths[: args.maximum_records]
    if args.smoke:
        paths, args.epochs = paths[:1], 1
    labels_by_path = {path: load_cached_arrays(path)["labels"] for path in paths}
    weights = class_patient_record_weights(labels_by_path)
    audit = weight_audit(weights, labels_by_path)

    groups = optimizer_groups(model, "backbone.encoder.layers.10.", 1e-5, 0.01)
    groups += optimizer_groups(model, "backbone.encoder.layers.11.", 1e-5, 0.01)
    if "d" in args.outer_scope:
        groups += optimizer_groups(model, "detector.", 3e-6, 0.01)
    if "s" in args.outer_scope:
        groups.append({"params": list(objective.parameters()), "lr": 1e-4, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups, fused=True)
    run = OUT / "runs/meta" / args.stage / args.mode / f"{args.objective}_{args.difficulty:g}_{args.outer_scope}_rho{args.relative_step:g}_seed{args.seed}"
    if args.run_directory is not None:
        run = args.run_directory.resolve()
    elif args.prefix_precision == "bf16":
        run = run.with_name(run.name + "_prefixbf16")
    if args.detector_freeze_fraction == 0 and "d" in args.outer_scope:
        run = run.with_name(run.name + "_detector_open")
    if args.inner_rule == "sgd":
        run = run.with_name(run.name + f"_sgd{args.fixed_inner_lr:g}")
    if args.stop_encoder_through_inner:
        run = run.with_name(run.name + "_stop_encoder_through_inner")
    if args.smoke:
        run = run.with_name(run.name + "_smoke")
    run.mkdir(parents=True, exist_ok=True)
    history = []
    first_epoch = 1
    if args.resume:
        resumed = torch.load(args.resume.resolve(), map_location="cpu", weights_only=False)
        expected = {
            "objective_name": args.objective,
            "difficulty": args.difficulty,
            "mode": args.mode,
            "outer_scope": args.outer_scope,
            "seed": args.seed,
            "inner_rule": args.inner_rule,
            "inner_kernel": args.inner_kernel,
            "fixed_inner_lr": args.fixed_inner_lr,
            "stop_encoder_through_inner": args.stop_encoder_through_inner,
            "prefix_precision": args.prefix_precision,
            "prefix_microbatch": args.prefix_microbatch,
            "prefix_cuda_graph": args.prefix_cuda_graph,
        }
        resume_defaults = {
            # Checkpoints written before these controls were made explicit used
            # exactly these behaviours.  Treat only those missing legacy fields
            # as their historical defaults; every non-default mismatch remains
            # a hard error.
            "inner_rule": "normalized",
            "inner_kernel": "reference",
            "stop_encoder_through_inner": False,
            "prefix_precision": "fp32",
            "prefix_microbatch": 0,
            "prefix_cuda_graph": False,
        }
        mismatches = {
            key: (resumed.get(key, resume_defaults.get(key)), value)
            for key, value in expected.items()
            if resumed.get(key, resume_defaults.get(key)) != value
        }
        if mismatches:
            raise ValueError(f"resume configuration mismatch: {mismatches}")
        model.load_state_dict(resumed["model"])
        objective.load_state_dict(resumed["objective"])
        optimizer.load_state_dict(resumed["optimizer"])
        history = list(resumed.get("history", []))
        first_epoch = int(resumed["epoch"]) + 1
        random.setstate(resumed["rng_state"]["python"])
        np.random.set_state(resumed["rng_state"]["numpy"])
        torch.random.set_rng_state(resumed["rng_state"]["torch"])
        torch.cuda.set_rng_state_all(resumed["rng_state"]["cuda"])
    started = time.monotonic()
    total_patients = len({path.parts[-4] for path in paths})
    for epoch in range(first_epoch, args.epochs + 1):
        by_patient: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            by_patient[path.parts[-4]].append(path)
        patient_order = sorted(by_patient)
        random.Random(args.seed + epoch).shuffle(patient_order)
        optimizer.zero_grad(set_to_none=True)
        diagnostics = defaultdict(float)
        encoder_parameters = [
            parameter for name, parameter in model.named_parameters()
            if name in functional.adaptable_names
        ]
        detector_parameters = list(model.detector.parameters())
        ssl_parameters = list(objective.parameters())
        before = {
            "encoder": [parameter.detach().cpu().clone() for parameter in encoder_parameters],
            "detector": [parameter.detach().cpu().clone() for parameter in detector_parameters],
            "ssl_head": [parameter.detach().cpu().clone() for parameter in ssl_parameters],
        }
        outer_value = 0.0
        group_size = 0
        detector_warm = "d" in args.outer_scope and epoch == 1
        for patient_index, patient in enumerate(patient_order, 1):
            group_size += 1
            for path in by_patient[patient]:
                for loss, rows in process_record(
                    functional, objective, path, weights[path], args.mode, inner_config,
                    stable_transform_seed(args.seed, path.as_posix()),
                    args.truncate_updates,
                    args.inner_rule,
                    args.fixed_inner_lr,
                    args.stop_encoder_through_inner,
                    args.inner_kernel,
                ):
                    loss.backward()
                    outer_value += float(loss.detach())
                    for row in rows:
                        diagnostics["updates"] += 1
                        diagnostics["accepted"] += int(row["accepted"])
                        diagnostics[f"reason_{row['reason']}"] += 1
            boundary = patient_index % 4 == 0 or patient_index == len(patient_order)
            if boundary:
                scale = total_patients / group_size
                for parameter in list(model.parameters()) + list(objective.parameters()):
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
                encoder_gradient_norm = torch.nn.utils.clip_grad_norm_(encoder_parameters, 1.0)
                diagnostics["encoder_gradient_norm_sum"] += float(encoder_gradient_norm)
                diagnostics["encoder_gradient_clipped"] += int(encoder_gradient_norm > 1)
                if "s" in args.outer_scope:
                    ssl_gradient_norm = torch.nn.utils.clip_grad_norm_(ssl_parameters, 1.0)
                    diagnostics["ssl_gradient_norm_sum"] += float(ssl_gradient_norm)
                    diagnostics["ssl_gradient_clipped"] += int(ssl_gradient_norm > 1)
                open_detector = "d" in args.outer_scope and not (
                    detector_warm
                    and patient_index
                    <= max(1, int(total_patients * args.detector_freeze_fraction))
                )
                if open_detector:
                    detector_gradient_norm = torch.nn.utils.clip_grad_norm_(
                        detector_parameters, 1.0
                    )
                    diagnostics["detector_gradient_norm_sum"] += float(detector_gradient_norm)
                    diagnostics["detector_gradient_clipped"] += int(detector_gradient_norm > 1)
                else:
                    for parameter in model.detector.parameters():
                        parameter.grad = None
                optimizer.step()
                diagnostics["outer_steps"] += 1
                optimizer.zero_grad(set_to_none=True)
                group_size = 0
        row = {
            "epoch": epoch,
            "outer_weighted_loss": outer_value,
            "patients": total_patients,
            "records": len(paths),
            "class_weight_audit": audit,
            "updates": int(diagnostics["updates"]),
            "accepted_fraction": diagnostics["accepted"] / max(1, diagnostics["updates"]),
            "update_reasons": {key.removeprefix("reason_"): int(value) for key, value in diagnostics.items() if key.startswith("reason_")},
            "outer_steps": int(diagnostics["outer_steps"]),
            "gradient_clipped_fraction": {
                name: diagnostics[f"{name}_gradient_clipped"] / max(1, diagnostics["outer_steps"])
                for name in ("encoder", "detector", "ssl")
            },
            "mean_preclip_gradient_norm": {
                name: diagnostics[f"{name}_gradient_norm_sum"] / max(1, diagnostics["outer_steps"])
                for name in ("encoder", "detector", "ssl")
            },
            "parameter_change_l2": {
                "encoder": l2_change(encoder_parameters, before["encoder"]),
                "detector": l2_change(detector_parameters, before["detector"]),
                "ssl_head": l2_change(ssl_parameters, before["ssl_head"]),
            },
            "elapsed_s": time.monotonic() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "objective": objective.state_dict(),
            "objective_name": args.objective,
            "difficulty": args.difficulty,
            "mode": args.mode,
            "outer_scope": args.outer_scope,
            "inner_rule": args.inner_rule,
            "inner_kernel": args.inner_kernel,
            "fixed_inner_lr": args.fixed_inner_lr,
            "stop_encoder_through_inner": args.stop_encoder_through_inner,
            "prefix_precision": args.prefix_precision,
            "prefix_microbatch": args.prefix_microbatch,
            "prefix_cuda_graph": args.prefix_cuda_graph,
            "detector_freeze_fraction": args.detector_freeze_fraction,
            "inner_config": vars(inner_config),
            "source": str(args.source.resolve()),
            "objective_checkpoint": str(args.objective_checkpoint.resolve()),
            "stage": args.stage,
            "seed": args.seed,
            "epoch": epoch,
            "created_utc": datetime.now(UTC).isoformat(),
        }
        torch.save(checkpoint, run / f"epoch_{epoch:02d}.pt")
        resumable = {
            **checkpoint,
            "optimizer": optimizer.state_dict(),
            "history": history,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
        }
        torch.save(resumable, run / "last.pt")
        (run / "history.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
