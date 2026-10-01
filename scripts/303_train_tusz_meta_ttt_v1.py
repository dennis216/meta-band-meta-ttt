#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.tusz_meta_ttt.alignment import GradientAlignment, compute_gradient_alignment
from bfa.tusz_meta_ttt.dataset import filter_inventory_records, load_cached_arrays
from bfa.tusz_meta_ttt.functional import FunctionalTUSZModel
from bfa.tusz_meta_ttt.model import TUSZDetector, balanced_soft_bce
from bfa.tusz_meta_ttt.objectives import build_objective

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v1"
PRETRAINED = ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"


def _alignment_stats(alignment: GradientAlignment, *, inner_lr: float) -> dict[str, float | bool]:
    """Convert one differentiable alignment result into logging scalars."""
    return {
        "valid": alignment.valid,
        "cosine": float(alignment.cosine.detach()) if alignment.valid else float("nan"),
        "ssl_gradient_norm": float(alignment.ssl_norm.detach()),
        "classification_gradient_norm": float(alignment.classification_norm.detach()),
        "dot": float(alignment.dot.detach()),
        "predicted_delta_bce": float((-inner_lr * alignment.dot).detach()),
        "parameter_update_norm": float((inner_lr * alignment.ssl_norm).detach()),
    }


def _merge_alignment_stats(
    target: dict[str, float], sample: dict[str, float | bool]
) -> None:
    """Accumulate valid/invalid episode diagnostics without graph references."""
    target["episodes"] += 1.0
    if not bool(sample["valid"]):
        target["invalid_episodes"] += 1.0
        return
    target["valid_episodes"] += 1.0
    target["negative_episodes"] += float(float(sample["cosine"]) < 0.0)
    for key in (
        "cosine",
        "ssl_gradient_norm",
        "classification_gradient_norm",
        "dot",
        "predicted_delta_bce",
        "parameter_update_norm",
    ):
        target[f"sum_{key}"] += float(sample[key])
    target["sum_actual_delta_bce"] += float(sample["actual_delta_bce"])
    target["sum_alignment_penalty"] += 1.0 - float(sample["cosine"])


def _empty_alignment_totals() -> dict[str, float]:
    return {
        "episodes": 0.0,
        "valid_episodes": 0.0,
        "invalid_episodes": 0.0,
        "negative_episodes": 0.0,
        "sum_cosine": 0.0,
        "sum_ssl_gradient_norm": 0.0,
        "sum_classification_gradient_norm": 0.0,
        "sum_dot": 0.0,
        "sum_predicted_delta_bce": 0.0,
        "sum_parameter_update_norm": 0.0,
        "sum_actual_delta_bce": 0.0,
        "sum_alignment_penalty": 0.0,
    }


def _finalize_alignment_totals(totals: dict[str, float]) -> dict[str, float | int | None]:
    valid = totals["valid_episodes"]
    return {
        "alignment_episodes": int(totals["episodes"]),
        "alignment_valid_episodes": int(valid),
        "alignment_invalid_episodes": int(totals["invalid_episodes"]),
        "alignment_negative_fraction": (
            totals["negative_episodes"] / valid if valid else None
        ),
        "alignment_mean_cosine": (
            totals["sum_cosine"] / valid if valid else None
        ),
        "alignment_mean_ssl_gradient_norm": (
            totals["sum_ssl_gradient_norm"] / valid if valid else None
        ),
        "alignment_mean_classification_gradient_norm": (
            totals["sum_classification_gradient_norm"] / valid if valid else None
        ),
        "alignment_mean_dot": totals["sum_dot"] / valid if valid else None,
        "alignment_mean_predicted_delta_bce": (
            totals["sum_predicted_delta_bce"] / valid if valid else None
        ),
        "alignment_mean_parameter_update_norm": (
            totals["sum_parameter_update_norm"] / valid if valid else None
        ),
        "alignment_mean_actual_delta_bce": (
            totals["sum_actual_delta_bce"] / valid if valid else None
        ),
        "alignment_mean_penalty": (
            totals["sum_alignment_penalty"] / valid if valid else None
        ),
    }


def build_source(checkpoint: Path) -> TUSZDetector:
    adapter = CBraModAdapter(PRETRAINED, train_backbone=True)
    adapter.backbone.proj_out = torch.nn.Identity()
    model = TUSZDetector(adapter.backbone)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False)
    return model.cuda().eval()


def windows(signal: np.ndarray, rows: list[int], device: torch.device) -> torch.Tensor:
    array = np.stack([signal[:, row * 400 : row * 400 + 2000] for row in rows])
    return torch.from_numpy(array.reshape(-1, 16, 10, 200)).to(device)


def _meta_episode(
    functional,
    objective,
    support: torch.Tensor,
    query: torch.Tensor,
    target: torch.Tensor,
    fast: dict[str, torch.Tensor],
    inner_lr: float,
    transform_seed: int,
    alignment_lambda: float,
    alignment_norm_floor: float,
    alignment_eps: float,
):
    """Run one label-free inner update and one labelled outer query.

    The classification gradient is computed only when an alignment term is
    requested.  It is a detached target direction for the outer loss; the
    support SSL gradient keeps its graph so the auxiliary module receives the
    Meta gradient.
    """
    parameters = tuple(fast.values())
    ssl_loss = objective.loss(
        support,
        feature_fn=lambda x, parameters=fast: functional.features(x, parameters),
        fast_parameters=fast,
        transform_seed=transform_seed,
    )
    ssl_gradients = torch.autograd.grad(
        ssl_loss,
        parameters,
        create_graph=True,
        allow_unused=True,
    )

    alignment = None
    pre_loss = None
    if alignment_lambda != 0.0:
        pre_logits = functional.logits(query, fast)
        pre_loss = balanced_soft_bce(pre_logits, target)
        classification_gradients = torch.autograd.grad(
            pre_loss,
            parameters,
            allow_unused=True,
        )
        alignment = compute_gradient_alignment(
            ssl_gradients,
            classification_gradients,
            parameters,
            inner_lr=inner_lr,
            norm_floor=alignment_norm_floor,
            eps=alignment_eps,
        )

    candidate = {}
    for (name, value), gradient in zip(fast.items(), ssl_gradients, strict=True):
        candidate[name] = value if gradient is None else value - inner_lr * gradient
    if not all(torch.isfinite(value).all() for value in candidate.values()):
        candidate = fast

    post_logits = functional.logits(query, candidate)
    post_loss = balanced_soft_bce(post_logits, target)
    outer_loss = post_loss
    diagnostics = None
    if alignment is not None:
        outer_loss = outer_loss + alignment_lambda * alignment.penalty
        diagnostics = _alignment_stats(alignment, inner_lr=inner_lr)
        diagnostics["actual_delta_bce"] = float((post_loss - pre_loss).detach())
    return outer_loss, candidate, diagnostics


def process_record(
    functional,
    objective,
    path,
    inner_lr,
    seed,
    smoke,
    alignment_lambda=0.0,
    alignment_norm_floor=1.0e-12,
    alignment_eps=1.0e-12,
):
    archive = load_cached_arrays(path)
    signal = archive["signal"]
    labels = archive["labels"]
    fast = functional.initial_fast_parameters()
    losses = []
    diagnostics = []
    update_count = max(0, (len(labels) - 15) // 15)
    if smoke:
        update_count = min(update_count, 2)
    for update in range(update_count):
        chunk_start = update * 15
        support_rows = [chunk_start, chunk_start + 5, chunk_start + 10]
        query_rows = list(range(chunk_start + 15, min(chunk_start + 30, len(labels))))
        if not query_rows:
            break
        support = windows(signal, support_rows, torch.device("cuda"))
        query = windows(signal, query_rows, torch.device("cuda"))
        target = torch.from_numpy(labels[query_rows]).to(query.device)
        loss, fast, episode_diagnostics = _meta_episode(
            functional,
            objective,
            support,
            query,
            target,
            fast,
            inner_lr,
            seed + update,
            alignment_lambda,
            alignment_norm_floor,
            alignment_eps,
        )
        losses.append(loss)
        if episode_diagnostics is not None:
            diagnostics.append(episode_diagnostics)
        if (update + 1) % 4 == 0:
            yield torch.stack(losses).mean(), update + 1, diagnostics
            losses = []
            diagnostics = []
            fast = {name: value.detach().requires_grad_(True) for name, value in fast.items()}
    if losses:
        yield torch.stack(losses).mean(), update_count, diagnostics


def same_window_rows(row_count: int, samples_per_record: int, seed: int) -> list[int]:
    if row_count <= 0 or samples_per_record <= 0:
        return []
    count = min(row_count, samples_per_record)
    rng = np.random.default_rng(seed)
    return sorted(int(row) for row in rng.choice(row_count, size=count, replace=False))


def process_same_window(
    functional,
    objective,
    path,
    inner_lr,
    seed,
    smoke,
    samples_per_record,
    alignment_lambda=0.0,
    alignment_norm_floor=1.0e-12,
    alignment_eps=1.0e-12,
):
    archive = load_cached_arrays(path)
    signal = archive["signal"]
    labels = archive["labels"]
    sample_count = 8 if smoke else samples_per_record
    selected_rows = same_window_rows(len(labels), sample_count, seed)
    losses = []
    diagnostics = []
    for item_index, row in enumerate(selected_rows, 1):
        fast = functional.initial_fast_parameters()
        support = windows(signal, [row], torch.device("cuda"))
        target = torch.tensor([labels[row]], device=support.device)
        loss, _, episode_diagnostics = _meta_episode(
            functional,
            objective,
            support,
            support,
            target,
            fast,
            inner_lr,
            seed + row,
            alignment_lambda,
            alignment_norm_floor,
            alignment_eps,
        )
        losses.append(loss)
        if episode_diagnostics is not None:
            diagnostics.append(episode_diagnostics)
        if len(losses) == 4:
            yield torch.stack(losses).mean(), item_index, diagnostics
            losses = []
            diagnostics = []
    if losses:
        yield torch.stack(losses).mean(), len(selected_rows), diagnostics


def segment_count(path: Path, mode: str, smoke: bool, same_window_samples_per_record: int) -> int:
    labels = load_cached_arrays(path)["labels"]
    if mode == "online":
        updates = max(0, (len(labels) - 15) // 15)
        if smoke:
            updates = min(updates, 2)
        return math.ceil(updates / 4)
    rows = min(len(labels), 8 if smoke else same_window_samples_per_record)
    return math.ceil(rows / 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=["band", "temporal", "mask", "learned"], required=True)
    parser.add_argument("--mode", choices=["online", "same_window"], default="online")
    parser.add_argument("--stage", choices=["development", "formal"], default="development")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inner-lr", type=float, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--maximum-records", type=int)
    parser.add_argument("--records-per-patient", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--same-window-samples-per-record", type=int, default=4)
    parser.add_argument(
        "--alignment-lambda",
        type=float,
        default=0.0,
        help="Outer cosine-alignment weight; zero preserves the original Meta objective.",
    )
    parser.add_argument("--alignment-norm-floor", type=float, default=1.0e-12)
    parser.add_argument("--alignment-eps", type=float, default=1.0e-12)
    args = parser.parse_args()
    if args.alignment_lambda < 0.0:
        parser.error("--alignment-lambda must be non-negative")
    if args.alignment_norm_floor <= 0.0 or args.alignment_eps <= 0.0:
        parser.error("alignment numerical thresholds must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_source(args.source.resolve())
    functional = FunctionalTUSZModel(model)
    objective = build_objective(args.objective).cuda().train()
    optimizer = torch.optim.AdamW(objective.parameters(), lr=1e-4, weight_decay=0.0)
    split = json.loads((OUT / "manifests/development_split.json").read_text())
    inventory = json.loads((OUT / "manifests/records.json").read_text())
    fit = (
        set(split["development_fit"])
        if args.stage == "development"
        else {
            item["patient_id"]
            for item in inventory
            if item["partition"] == "train"
        }
    )
    paths = sorted(
        path
        for path in (OUT / "cache/train").rglob("*.npz")
        if path.parts[-4] in fit
    )
    paths = filter_inventory_records(paths, inventory, partition="train")
    if args.smoke:
        paths = sorted((OUT / "cache/train").rglob("*.npz"))[:1]
        args.epochs = 1
    if args.maximum_records is not None:
        paths = paths[: args.maximum_records]
    if args.records_per_patient is not None:
        counts = {}
        selected = []
        for path in paths:
            patient = path.parts[-4]
            if counts.get(patient, 0) < args.records_per_patient:
                selected.append(path)
                counts[patient] = counts.get(patient, 0) + 1
        paths = selected
    if not paths:
        raise RuntimeError("no cached training records")
    run = OUT / "runs/meta" / args.stage / args.mode / f"{args.objective}_lr{args.inner_lr:g}_seed{args.seed}"
    if args.alignment_lambda != 0.0:
        run = run.with_name(run.name + f"_cosine_lambda{args.alignment_lambda:g}")
    if args.smoke:
        run = run.with_name(run.name + "_smoke")
    if args.run_tag:
        if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in args.run_tag):
            raise ValueError("run tag may contain only letters, digits, underscores, and hyphens")
        run = run.with_name(run.name + "_" + args.run_tag)
    run.mkdir(parents=True, exist_ok=True)
    history = []
    start_epoch = 1
    last_checkpoint = run / "last.pt"
    if args.resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(last_checkpoint)
        state = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        objective.load_state_dict(state["objective"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        start_epoch = int(state["epoch"]) + 1
    segment_counts = {
        path: segment_count(path, args.mode, args.smoke, args.same_window_samples_per_record)
        for path in paths
    }
    eligible_paths = [path for path in paths if segment_counts[path] > 0]
    if not eligible_paths:
        raise RuntimeError("no records contain a valid support-to-query Meta segment")
    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs + 1):
        alignment_totals = _empty_alignment_totals()
        by_patient: dict[str, list[Path]] = {}
        for path in eligible_paths:
            by_patient.setdefault(path.parts[-4], []).append(path)
        patients = sorted(by_patient)
        epoch_rng = random.Random(args.seed + epoch)
        epoch_rng.shuffle(patients)
        for patient_paths in by_patient.values():
            epoch_rng.shuffle(patient_paths)
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []
        record_index = 0
        for patient_index, patient in enumerate(patients, 1):
            patient_paths = by_patient[patient]
            processor = process_record if args.mode == "online" else process_same_window
            for path in patient_paths:
                record_index += 1
                segments = segment_counts[path]
                scale = 1.0 / (len(patient_paths) * segments)
                processor_args = (
                    (
                        functional,
                        objective,
                        path,
                        args.inner_lr,
                        args.seed + record_index,
                        args.smoke,
                        args.alignment_lambda,
                        args.alignment_norm_floor,
                        args.alignment_eps,
                    )
                    if args.mode == "online"
                    else (
                        functional,
                        objective,
                        path,
                        args.inner_lr,
                        args.seed + record_index,
                        args.smoke,
                        args.same_window_samples_per_record,
                        args.alignment_lambda,
                        args.alignment_norm_floor,
                        args.alignment_eps,
                    )
                )
                for loss, _, batch_diagnostics in processor(*processor_args):
                    (loss * scale).backward()
                    epoch_losses.append(float(loss.detach()))
                    for diagnostic in batch_diagnostics:
                        _merge_alignment_stats(alignment_totals, diagnostic)
            if patient_index % 4 == 0 or patient_index == len(patients):
                torch.nn.utils.clip_grad_norm_(objective.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        row = {
            "epoch": epoch,
            "mean_outer_loss": float(np.mean(epoch_losses)),
            "records": len(paths),
            "eligible_records": len(eligible_paths),
            "patients_seen": len({path.parts[-4] for path in eligible_paths}),
            "elapsed_s": time.monotonic() - started,
        }
        row.update(_finalize_alignment_totals(alignment_totals))
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save(
            {
                "objective": objective.state_dict(),
                "objective_name": args.objective,
                "mode": args.mode,
                "stage": args.stage,
                "inner_lr": args.inner_lr,
                "alignment_lambda": args.alignment_lambda,
                "alignment_norm_floor": args.alignment_norm_floor,
                "alignment_eps": args.alignment_eps,
                "epoch": epoch,
                "source": str(args.source.resolve()),
                "same_window_samples_per_record": args.same_window_samples_per_record,
                "created_utc": datetime.now(UTC).isoformat(),
            },
            run / f"epoch_{epoch:02d}.pt",
        )
        (run / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        torch.save(
            {
                "objective": objective.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
                "alignment_lambda": args.alignment_lambda,
                "alignment_norm_floor": args.alignment_norm_floor,
                "alignment_eps": args.alignment_eps,
            },
            last_checkpoint,
        )


if __name__ == "__main__":
    main()
