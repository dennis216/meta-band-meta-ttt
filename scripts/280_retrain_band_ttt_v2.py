#!/usr/bin/env python3
"""Registered two-fold Band-TTT retraining release.

This file is intentionally independent of the paused Band-TTT evaluation
queue.  It owns the common supervised classifier, the prepared Band head, and
the two meta objectives used by the repaired experiment.  The evaluator in
281 imports the small functional adaptation helpers from this file.

The meta path keeps the first ten CBraMod layers as a prefix and differentiates
only through encoder layers 10 and 11.  Prefix tensors are detached by design:
the deployed TTT update is restricted to those two layers and outer training
must not move the common classifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call
from sklearn.metrics import average_precision_score, roc_auc_score

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/NeuroTTT/CBraMod")
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))

from chbmit_groupkfold.data import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_FOLDS,
    DEFAULT_WINDOWS,
    WindowDataset,
    load_rows,
    make_eval_loader,
    make_train_loader,
)
from chbmit_groupkfold.model import CHBJointModel  # noqa: E402
from chbmit_groupkfold.transforms import deterministic_band_view  # noqa: E402


RELEASE_ID = "meta-ttt-chbmit-v2-repaired"
SEED = 3407
CHUNK_SIZE = 16
CHUNK_STRIDE = 20  # 16 windows x 2 s + 10 s window has no query overlap.
BURN_IN_CHUNKS = 4
QUERY_CHUNKS = 3
EPISODE_SPAN = (BURN_IN_CHUNKS + QUERY_CHUNKS - 1) * CHUNK_STRIDE + CHUNK_SIZE
TAIL_LAYER_START = 10
TAIL_LAYER_END = 12
INNER_LR = 1e-4
OUTER_LR = 1e-3
OUTER_MAX_STEPS = 3000
EVAL_INTERVAL = 250
MIN_META_STEPS = 1000
META_PATIENCE_EVALS = 4
GRAD_CLIP = 1.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def formal_config() -> dict[str, Any]:
    return {
        "release_id": RELEASE_ID,
        "seed": SEED,
        "folds": [0, 1],
        "classifier": {
            "init": "official CBraMod pretrained backbone; fresh detector and Band head",
            "optimizer": "AdamW",
            "lr": 1e-4,
            "effective_batch": 128,
            "weight_decay": 0.05,
            "weight_decay_exclusions": "bias and normalization parameters",
            "max_epochs": 50,
            "min_epochs": 5,
            "early_stop_patience": 7,
        },
        "band_head_prepare": {"steps": 500, "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 0.0},
        "meta": {
            "branches": ["band", "learned"],
            "adapted_layers": [10, 11],
            "inner_optimizer": "SGD",
            "inner_lr": INNER_LR,
            "inner_steps": 1,
            "chunk_size": CHUNK_SIZE,
            "chunk_stride": CHUNK_STRIDE,
            "burn_in_chunks": BURN_IN_CHUNKS,
            "query_chunks": QUERY_CHUNKS,
            "outer_optimizer": "AdamW",
            "outer_lr": OUTER_LR,
            "outer_weight_decay": 0.0,
            "effective_episodes": 8,
            "microbatch_episodes": 2,
            "gradient_accumulation": 4,
            "max_steps": OUTER_MAX_STEPS,
            "validation_interval": EVAL_INTERVAL,
            "minimum_steps": MIN_META_STEPS,
            "patience_evaluations": META_PATIENCE_EVALS,
            "outer_loss": "0.8 post0 + 0.2 post1 + 2*class_weighted_deterioration; w0=0.8,w1=0.2",
            "band_regularizer_weight": 0.05,
            "dropout": "disabled via eval mode throughout meta path",
        },
        "data_protocol": {
            "training_sampler": "balanced patient sampler only for common classifier",
            "episode_sampling": "75% duration-weighted records, 25% seizure-containing continuous records",
            "adaptation": "one mean SSL gradient per chunk; no averaging across independent episodes",
            "future_query": "query starts at least at previous support chunk end",
            "test_read_during_training": False,
        },
    }


def classifier_run_dir(output_root: Path, fold: int, seed: int) -> Path:
    return output_root / "runs" / f"fold{fold}" / f"common_classifier_seed{seed}"


def band_prepare_path(output_root: Path, fold: int, seed: int) -> Path:
    return output_root / "runs" / f"fold{fold}" / f"band_head_prepared_seed{seed}.pt"


def meta_run_dir(output_root: Path, fold: int, branch: str, seed: int) -> Path:
    return output_root / "runs" / f"fold{fold}" / f"meta_{branch}_seed{seed}"


def model_from_official(pretrained: Path, device: torch.device) -> CHBJointModel:
    model = CHBJointModel(pretrained).to(device)
    return model


def load_common_model(args: argparse.Namespace, fold: int, *, prepared_band: bool, device: torch.device) -> tuple[CHBJointModel, Path, dict[str, Any]]:
    common_dir = classifier_run_dir(args.output_root, fold, args.seed)
    checkpoint = common_dir / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = model_from_official(args.pretrained, device)
    model.load_state_dict(payload["model"], strict=True)
    if prepared_band:
        prep_path = band_prepare_path(args.output_root, fold, args.seed)
        prep = torch.load(prep_path, map_location="cpu", weights_only=False)
        model.band_head.load_state_dict(prep["band_head"], strict=True)
    return model, checkpoint, payload


def _decay_groups(model: nn.Module, include_prefix: str | None = None) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or (include_prefix is not None and not name.startswith(include_prefix)):
            continue
        lower = name.lower()
        if parameter.ndim <= 1 or lower.endswith(".bias") or "norm" in lower:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": 0.05})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


@torch.inference_mode()
def classifier_validation(model: CHBJointModel, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    losses: list[tuple[float, int]] = []
    for signal, target, _ in loader:
        signal = signal.to(device=device, dtype=torch.float32, non_blocking=True)
        target = target.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model.detect(signal)
            loss = F.binary_cross_entropy_with_logits(logits, target)
        labels.append(target.cpu().numpy())
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
        losses.append((float(loss.cpu()), len(target)))
    y = np.concatenate(labels)
    p = np.concatenate(probabilities)
    return {
        "rows": int(len(y)),
        "loss": float(sum(value * count for value, count in losses) / max(1, sum(count for _, count in losses))),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "positive_prevalence": float(y.mean()),
    }


def run_supervised(args: argparse.Namespace) -> dict[str, Any]:
    if args.fold not in (0, 1):
        raise ValueError("formal release is limited to folds 0 and 1")
    run_dir = classifier_run_dir(args.output_root, args.fold, args.seed)
    completed_path = run_dir / "completed.json"
    if completed_path.is_file() and not args.force:
        return json.loads(completed_path.read_text())
    set_seed(args.seed + args.fold)
    device = torch.device(args.device)
    train_rows = load_rows(args.fold, "train", args.windows, args.fold_root)
    val_rows = load_rows(args.fold, "validation", args.windows, args.fold_root)
    physical_batch = args.classifier_batch
    effective_batch = 128
    if effective_batch % physical_batch:
        raise ValueError("classifier batch must divide effective batch 128")
    positives = int((train_rows.label.astype(int) == 1).sum())
    updates_per_epoch = max(1, math.ceil(2 * positives / effective_batch))
    accumulation = effective_batch // physical_batch
    loader, sampler = make_train_loader(
        train_rows,
        batch_size=physical_batch,
        steps=updates_per_epoch * accumulation,
        seed=args.seed + args.fold,
        workers=args.workers,
        cache_root=args.cache_root,
    )
    val_loader = make_eval_loader(val_rows, batch_size=args.eval_batch, workers=args.workers, cache_root=args.cache_root)
    model = model_from_official(args.pretrained, device)
    model.band_head.requires_grad_(False)
    optimizer = torch.optim.AdamW(_decay_groups(model), lr=1e-4)
    total_updates = args.classifier_epochs * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates), eta_min=1e-6)
    start_epoch = 0
    global_update = 0
    history: list[dict[str, Any]] = []
    best_auprc = -float("inf")
    patience = 0
    if (run_dir / "last.pt").is_file() and not args.force:
        saved = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved.get("epoch", -1)) + 1
        global_update = int(saved.get("update", 0))
        history = list(saved.get("history", []))
        best_auprc = float(saved.get("best_auprc", -float("inf")))
        patience = int(saved.get("patience", 0))
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **formal_config(),
        "status": "running",
        "stage": "common_classifier",
        "fold": args.fold,
        "seed": args.seed,
        "started_at": utc_now(),
        "train_rows": len(train_rows),
        "train_positive_rows": positives,
        "validation_rows": len(val_rows),
        "updates_per_epoch": updates_per_epoch,
        "physical_batch": physical_batch,
        "device": str(device),
        "source_hashes": {
            "windows": sha256(args.windows),
            "fold": sha256(args.fold_root / f"fold_{args.fold}.json"),
            "pretrained": sha256(args.pretrained),
        },
    }
    atomic_json(run_dir / "manifest.json", manifest)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    processed = 0
    last_progress = started
    for epoch in range(start_epoch, args.classifier_epochs):
        sampler.set_epoch(epoch)
        model.train()
        model.band_head.eval()
        epoch_losses: list[float] = []
        epoch_started = time.monotonic()
        for micro_index, (signal, target, _) in enumerate(loader):
            signal = signal.to(device=device, dtype=torch.float32, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model.detect(signal)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite classifier loss at epoch={epoch} micro={micro_index}")
            (loss / accumulation).backward()
            epoch_losses.append(float(loss.detach().cpu()))
            processed += len(signal)
            if (micro_index + 1) % accumulation:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_update += 1
            now = time.monotonic()
            if now - last_progress >= 30.0:
                atomic_json(run_dir / "progress.json", {
                    **manifest,
                    "status": "training",
                    "epoch": epoch,
                    "update": global_update,
                    "processed_windows": processed,
                    "windows_per_s": processed / max(now - started, 1e-9),
                    "stage_elapsed_s": now - started,
                    "gpu_peak_mib": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
                })
                last_progress = now
        validation = classifier_validation(model, val_loader, device)
        row = {
            "epoch": epoch,
            "update": global_update,
            "train_loss": float(np.mean(epoch_losses)),
            "validation": validation,
            "epoch_seconds": time.monotonic() - epoch_started,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        improved = validation["auprc"] > best_auprc + 1e-5
        if improved:
            best_auprc = validation["auprc"]
            patience = 0
            atomic_torch_save(run_dir / "best.pt", {
                "release_id": RELEASE_ID,
                "stage": "common_classifier",
                "fold": args.fold,
                "seed": args.seed,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "update": global_update,
                "best_auprc": best_auprc,
                "history": history,
            })
        else:
            patience += 1
        payload = {
            "release_id": RELEASE_ID,
            "stage": "common_classifier",
            "fold": args.fold,
            "seed": args.seed,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "update": global_update,
            "best_auprc": best_auprc,
            "patience": patience,
            "history": history,
        }
        atomic_torch_save(run_dir / "last.pt", payload)
        atomic_json(run_dir / "history.json", {"epochs": history})
        print(json.dumps(row, allow_nan=True), flush=True)
        if epoch + 1 >= args.classifier_min_epochs and patience >= args.classifier_patience:
            break
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    final_model = model_from_official(args.pretrained, torch.device("cpu"))
    final_model.load_state_dict(best["model"], strict=True)
    completed = {
        **manifest,
        "status": "completed",
        "completed_at": utc_now(),
        "epochs_completed": len(history),
        "updates_completed": global_update,
        "best_validation_auprc": best_auprc,
        "best_checkpoint": str(run_dir / "best.pt"),
        "classifier_hash": state_hash(final_model),
        "elapsed_s": time.monotonic() - started,
    }
    atomic_json(completed_path, completed)
    atomic_json(run_dir / "progress.json", completed)
    return completed


def run_prepare_band_head(args: argparse.Namespace) -> dict[str, Any]:
    output_path = band_prepare_path(args.output_root, args.fold, args.seed)
    completed_path = output_path.with_suffix(".completed.json")
    if completed_path.is_file() and not args.force:
        return json.loads(completed_path.read_text())
    set_seed(args.seed + 20_000 + args.fold)
    device = torch.device(args.device)
    model, common_path, common_payload = load_common_model(args, args.fold, prepared_band=False, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.band_head.parameters():
        parameter.requires_grad_(True)
    rows = load_rows(args.fold, "train", args.windows, args.fold_root)
    loader = make_eval_loader(rows, batch_size=args.band_batch, workers=args.workers, cache_root=args.cache_root)
    optimizer = torch.optim.AdamW(model.band_head.parameters(), lr=1e-3, weight_decay=0.0)
    iterator = iter(loader)
    losses: list[float] = []
    started = time.monotonic()
    for step in range(args.band_steps):
        try:
            signal, _, sample_ids = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            signal, _, sample_ids = next(iterator)
        signal = signal.to(device=device, dtype=torch.float32, non_blocking=True)
        filtered, labels = deterministic_band_view(signal, list(sample_ids))
        with torch.no_grad():
            features = model.encode(filtered)
        logits = model.band_logits(features.detach())
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 50 == 0:
            atomic_json(output_path.parent / "band_prepare_progress.json", {
                "release_id": RELEASE_ID,
                "stage": "band_head_prepare",
                "fold": args.fold,
                "step": step + 1,
                "steps": args.band_steps,
                "loss": float(np.mean(losses[-50:])),
                "elapsed_s": time.monotonic() - started,
            })
    model_cpu = model.to("cpu")
    classifier_model = model_from_official(args.pretrained, torch.device("cpu"))
    classifier_model.load_state_dict(common_payload["model"], strict=True)
    classifier_hash = state_hash(classifier_model)
    prepared_hash = state_hash(model_cpu.band_head)
    atomic_torch_save(output_path, {
        "release_id": RELEASE_ID,
        "stage": "band_head_prepare",
        "fold": args.fold,
        "seed": args.seed,
        "common_classifier": str(common_path),
        "classifier_hash": classifier_hash,
        "band_head": model_cpu.band_head.state_dict(),
        "band_head_hash": prepared_hash,
        "steps": args.band_steps,
        "mean_loss_last50": float(np.mean(losses[-50:])),
    })
    completed = {
        "release_id": RELEASE_ID,
        "stage": "band_head_prepare",
        "fold": args.fold,
        "seed": args.seed,
        "status": "completed",
        "checkpoint": str(output_path),
        "classifier_hash": classifier_hash,
        "band_head_hash": prepared_hash,
        "steps": args.band_steps,
        "mean_loss_last50": float(np.mean(losses[-50:])),
        "elapsed_s": time.monotonic() - started,
    }
    atomic_json(completed_path, completed)
    return completed


@dataclass
class Record:
    patient: str
    recording: str
    relative_path: str
    rows: pd.DataFrame
    starts: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    candidate_starts: np.ndarray
    seizure_record: bool


class SignalStore:
    """Small LRU around the existing read-only mmap cache."""

    def __init__(self, cache_root: Path, max_open: int = 64) -> None:
        self.cache_root = Path(cache_root)
        self.max_open = int(max_open)
        self._views: dict[str, np.ndarray] = {}
        self._order: list[str] = []

    def _view(self, relative_path: str) -> np.ndarray:
        key = str(relative_path)
        if key in self._views:
            self._order.remove(key)
            self._order.append(key)
            return self._views[key]
        path = self.cache_root / Path(key).with_suffix(".npy")
        view = np.load(path, mmap_mode="r", allow_pickle=False)
        if view.ndim != 2 or view.shape[0] != 16:
            raise ValueError(f"bad cached signal shape {view.shape}: {path}")
        self._views[key] = view
        self._order.append(key)
        while len(self._order) > self.max_open:
            old = self._order.pop(0)
            self._views.pop(old, None)
        return view

    def read(self, record: Record, positions: np.ndarray) -> np.ndarray:
        view = self._view(record.relative_path)
        starts = record.starts[positions]
        batch: list[np.ndarray] = []
        for start in starts:
            left = int(round(float(start) * 200.0))
            right = left + 2000
            signal = np.asarray(view[:, left:right], dtype=np.float32)
            if signal.shape != (16, 2000) or not np.isfinite(signal).all():
                raise ValueError(f"invalid cached window {record.recording}:{start}")
            batch.append(np.ascontiguousarray(signal.reshape(16, 10, 200)))
        return np.stack(batch, axis=0)


def records_from_rows(rows: pd.DataFrame, *, min_episode: int = EPISODE_SPAN) -> list[Record]:
    output: list[Record] = []
    for (patient, recording), group in rows.groupby(["patient", "recording"], sort=False):
        group = group.sort_values("start", kind="stable").reset_index(drop=True)
        starts = group.start.to_numpy(dtype=np.float64)
        if len(starts) < min_episode:
            continue
        increments = np.isclose(np.diff(starts), 2.0, atol=0.02)
        if len(increments) >= min_episode - 1:
            runs = np.convolve(increments.astype(np.int32), np.ones(min_episode - 1, dtype=np.int32), mode="valid")
            candidates = np.flatnonzero(runs == min_episode - 1)
        else:
            candidates = np.empty(0, dtype=np.int64)
        if len(candidates) == 0:
            continue
        output.append(Record(
            patient=str(patient),
            recording=str(recording),
            relative_path=str(group.relative_path.iloc[0]),
            rows=group,
            starts=starts,
            labels=group.label.to_numpy(dtype=np.float32),
            sample_ids=group.sample_id.astype(str).to_numpy(),
            candidate_starts=candidates.astype(np.int64),
            seizure_record=bool((group.label.astype(int) == 1).any()),
        ))
    if not output:
        raise RuntimeError(f"no records contain a continuous {min_episode}-window meta episode")
    return output


def choose_episode(records: list[Record], rng: np.random.Generator, *, seizure_bias: bool | None = None) -> tuple[Record, int]:
    if seizure_bias is None:
        seizure_bias = bool(rng.random() < 0.25)
    pool = [record for record in records if record.seizure_record] if seizure_bias else records
    if not pool:
        pool = records
    weights = np.asarray([len(record.rows) * max(1, len(record.candidate_starts)) for record in pool], dtype=np.float64)
    weights /= weights.sum()
    record = pool[int(rng.choice(len(pool), p=weights))]
    start = int(rng.choice(record.candidate_starts))
    return record, start


def episode_positions(start: int) -> list[np.ndarray]:
    return [
        start + chunk * CHUNK_STRIDE + np.arange(CHUNK_SIZE, dtype=np.int64)
        for chunk in range(BURN_IN_CHUNKS + QUERY_CHUNKS)
    ]


def fixed_episode_specs(records: list[Record], count: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    specs: list[tuple[int, int]] = []
    for _ in range(count):
        record, start = choose_episode(records, rng)
        specs.append((records.index(record), start))
    return specs


class PrefixCache:
    """Disk LRU for prefix tensors, keyed by classifier and sample/view.

    The index is advisory: a missing or corrupt entry is simply recomputed.
    Only one meta worker writes a formal fold at a time, so atomic replacement
    is sufficient and avoids introducing a cross-process lock into training.
    """

    def __init__(self, root: Path, source_hash: str, max_bytes: int = 128 * 2**30, enabled: bool = True) -> None:
        self.root = Path(root) / source_hash
        self.index_path = self.root / "index.json"
        self.max_bytes = int(max_bytes)
        self.enabled = bool(enabled)
        self.index: dict[str, dict[str, Any]] = {}
        if self.enabled and self.index_path.is_file():
            try:
                self.index = json.loads(self.index_path.read_text())
            except (OSError, ValueError):
                self.index = {}

    def _key(self, sample_id: str, view: str) -> str:
        return hashlib.sha256(f"{sample_id}|{view}".encode("utf-8")).hexdigest()

    def get(self, sample_ids: list[str], view: str, device: torch.device) -> tuple[torch.Tensor | None, list[int]]:
        if not self.enabled:
            return None, list(range(len(sample_ids)))
        tensors: list[torch.Tensor | None] = [None] * len(sample_ids)
        missing: list[int] = []
        changed = False
        for i, sample_id in enumerate(sample_ids):
            key = self._key(sample_id, view)
            entry = self.index.get(key)
            if not entry:
                missing.append(i)
                continue
            path = self.root / entry["file"]
            try:
                tensors[i] = torch.from_numpy(np.load(path, allow_pickle=False)).to(device=device)
                entry["atime"] = time.time()
                changed = True
            except (OSError, ValueError, KeyError):
                self.index.pop(key, None)
                missing.append(i)
                changed = True
        if changed:
            self._flush_index()
        if missing:
            return None, missing
        return torch.stack([tensor for tensor in tensors if tensor is not None]), []

    def put(self, sample_ids: list[str], view: str, values: torch.Tensor) -> None:
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        values_cpu = values.detach().float().cpu().numpy()
        now = time.time()
        for sample_id, value in zip(sample_ids, values_cpu, strict=True):
            key = self._key(sample_id, view)
            filename = f"{key}.npy"
            path = self.root / filename
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, value, allow_pickle=False)
            os.replace(temporary, path)
            self.index[key] = {"file": filename, "bytes": int(path.stat().st_size), "atime": now}
        self._evict()
        self._flush_index()

    def _evict(self) -> None:
        total = sum(int(item.get("bytes", 0)) for item in self.index.values())
        if total <= self.max_bytes:
            return
        for key, entry in sorted(self.index.items(), key=lambda item: float(item[1].get("atime", 0.0))):
            if total <= self.max_bytes:
                break
            try:
                (self.root / entry["file"]).unlink(missing_ok=True)
            finally:
                total -= int(entry.get("bytes", 0))
                self.index.pop(key, None)

    def _flush_index(self) -> None:
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.index, sort_keys=True))
        os.replace(temporary, self.index_path)


def make_prefix_batch(
    model: CHBJointModel,
    signals: torch.Tensor,
    sample_ids: list[str],
    view: str,
    cache: PrefixCache | None,
) -> torch.Tensor:
    if cache is None:
        with torch.no_grad():
            return encode_prefix(model, signals)
    cached, missing = cache.get(sample_ids, view, signals.device)
    if not missing and cached is not None:
        return cached
    with torch.no_grad():
        computed = encode_prefix(model, signals[missing])
    cache.put([sample_ids[i] for i in missing], view, computed)
    if cached is None:
        cached_values: list[torch.Tensor | None] = [None] * len(sample_ids)
    else:
        cached_values = [value for value in cached]
    for index, value in zip(missing, computed, strict=True):
        cached_values[index] = value
    return torch.stack([value for value in cached_values if value is not None])


def encode_prefix(model: CHBJointModel, signal: torch.Tensor) -> torch.Tensor:
    """Run patch embedding and the frozen first ten encoder blocks."""
    value = model.backbone.patch_embedding(signal)
    for layer in model.backbone.encoder.layers[:TAIL_LAYER_START]:
        value = layer(value)
    return value.detach()


def _layer_param_values(params: dict[str, torch.Tensor], layer_index: int) -> dict[str, torch.Tensor]:
    prefix = f"backbone.encoder.layers.{layer_index}."
    values = {name[len(prefix):]: value for name, value in params.items() if name.startswith(prefix)}
    if not values:
        raise KeyError(f"missing functional parameters for encoder layer {layer_index}")
    return values


def encode_tail(model: CHBJointModel, prefix: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run the two functional encoder blocks using the supplied parameters."""
    value = prefix
    for layer_index in range(TAIL_LAYER_START, TAIL_LAYER_END):
        layer = model.backbone.encoder.layers[layer_index]
        value = functional_call(layer, _layer_param_values(params, layer_index), (value,), strict=False)
    return value


def tail_named_parameters(model: CHBJointModel) -> dict[str, nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.encoder.layers.10.") or name.startswith("backbone.encoder.layers.11.")
    }


def detached_tail_values(model: CHBJointModel, *, requires_grad: bool = True) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone().requires_grad_(requires_grad)
        for name, parameter in tail_named_parameters(model).items()
    }


class LearnedAuxiliaryLoss(nn.Module):
    """Learned unlabeled loss with the fixed 4005-dimensional contract."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4005, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        raw_features: torch.Tensor,
        transformed_features: torch.Tensor,
        labels: torch.Tensor,
        fixed_band_head: nn.Module,
    ) -> torch.Tensor:
        raw_vector = raw_features.mean(dim=1).flatten(1)
        transformed_vector = transformed_features.mean(dim=1).flatten(1)
        one_hot = F.one_hot(labels, num_classes=5).to(dtype=raw_vector.dtype)
        features = torch.cat([raw_vector, transformed_vector, one_hot], dim=1)
        if features.shape[1] != 4005:
            raise RuntimeError(f"learned auxiliary input is {features.shape[1]}, expected 4005")
        fixed_ce = F.cross_entropy(fixed_band_head(transformed_features.mean(dim=1).flatten(1)), labels, reduction="none")
        prediction = self.net(features).squeeze(-1)
        return F.softplus(fixed_ce + prediction)


def auxiliary_loss(
    model: CHBJointModel,
    objective_head: LearnedAuxiliaryLoss | None,
    raw_prefix: torch.Tensor,
    transformed_prefix: torch.Tensor,
    params: dict[str, torch.Tensor],
    labels: torch.Tensor,
    branch: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    transformed_features = encode_tail(model, transformed_prefix, params)
    if branch == "band":
        per_sample = F.cross_entropy(model.band_logits(transformed_features), labels, reduction="none")
    elif branch == "learned":
        if objective_head is None:
            raise ValueError("learned branch needs an objective head")
        raw_features = encode_tail(model, raw_prefix, params)
        per_sample = objective_head(raw_features, transformed_features, labels, model.band_head)
    else:
        raise ValueError(branch)
    return per_sample.mean(), transformed_features


def clipped_inner_update(
    model: CHBJointModel,
    objective_head: LearnedAuxiliaryLoss | None,
    raw_prefix: torch.Tensor,
    transformed_prefix: torch.Tensor,
    labels: torch.Tensor,
    params: dict[str, torch.Tensor],
    branch: str,
    *,
    create_graph: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    loss, _ = auxiliary_loss(model, objective_head, raw_prefix, transformed_prefix, params, labels, branch)
    values = list(params.values())
    gradients = torch.autograd.grad(
        loss,
        values,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=True,
    )
    finite_gradients = [gradient for gradient in gradients if gradient is not None]
    if not finite_gradients:
        raise RuntimeError(f"{branch} auxiliary loss produced no inner gradient")
    squared = torch.stack([gradient.float().square().sum() for gradient in finite_gradients]).sum()
    grad_norm = squared.sqrt()
    scale = torch.clamp(torch.as_tensor(GRAD_CLIP, device=grad_norm.device) / (grad_norm + 1e-12), max=1.0)
    updated = {
        name: value - INNER_LR * (gradient if gradient is not None else torch.zeros_like(value)) * scale
        for (name, value), gradient in zip(params.items(), gradients, strict=True)
    }
    update_norm = torch.stack([
        (updated[name] - params[name]).float().square().sum()
        for name in params
    ]).sum().sqrt()
    return updated, loss, grad_norm, update_norm


def binary_losses(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none")


def detector_from_prefix(model: CHBJointModel, prefix: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    features = encode_tail(model, prefix, params)
    return model.detect_from_features(features)


def _episode_chunk(
    store: SignalStore,
    record: Record,
    positions: np.ndarray,
    device: torch.device,
    cache: PrefixCache | None,
    model: CHBJointModel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    signal_np = store.read(record, positions)
    signal = torch.from_numpy(signal_np).to(device=device, dtype=torch.float32, non_blocking=True)
    sample_ids = [str(item) for item in record.sample_ids[positions]]
    transformed, labels = deterministic_band_view(signal, sample_ids)
    raw_prefix = make_prefix_batch(model, signal, sample_ids, "raw", cache)
    transformed_prefix = make_prefix_batch(model, transformed, sample_ids, "band", cache)
    target = torch.from_numpy(record.labels[positions]).to(device=device, dtype=torch.float32)
    return raw_prefix, transformed_prefix, labels, target, sample_ids


def run_episode(
    model: CHBJointModel,
    objective_head: LearnedAuxiliaryLoss | None,
    store: SignalStore,
    record: Record,
    start: int,
    device: torch.device,
    branch: str,
    cache: PrefixCache | None,
    *,
    differentiable: bool,
) -> dict[str, torch.Tensor | float | int]:
    """Run one 4-burn-in + 3-query episode with causal chunk updates."""
    positions = episode_positions(start)
    base_params = detached_tail_values(model, requires_grad=False)
    params = detached_tail_values(model, requires_grad=True)
    chunks = [
        _episode_chunk(store, record, chunk_positions, device, cache, model)
        for chunk_positions in positions
    ]
    burnin_losses: list[torch.Tensor] = []
    grad_norms: list[torch.Tensor] = []
    update_norms: list[torch.Tensor] = []
    for raw_prefix, transformed_prefix, labels, _, _ in chunks[:BURN_IN_CHUNKS]:
        with torch.enable_grad():
            params, loss, grad_norm, update_norm = clipped_inner_update(
                model, objective_head, raw_prefix, transformed_prefix, labels, params, branch,
                create_graph=False,
            )
        params = {name: value.detach().requires_grad_(True) for name, value in params.items()}
        burnin_losses.append(loss.detach())
        grad_norms.append(grad_norm.detach())
        update_norms.append(update_norm.detach())

    # q0 is predicted using the post-burn-in state, then becomes the first
    # support chunk. q1 and q2 are the two genuinely future post-update
    # queries used by the outer objective.
    q0 = chunks[BURN_IN_CHUNKS]
    q1 = chunks[BURN_IN_CHUNKS + 1]
    q2 = chunks[BURN_IN_CHUNKS + 2]
    with torch.enable_grad():
        _ = detector_from_prefix(model, q0[0], params)
        params_after_q0, ssl0, grad0, update0 = clipped_inner_update(
            model, objective_head, q0[0], q0[1], q0[2], params, branch,
            create_graph=differentiable,
        )
        post0_logits = detector_from_prefix(model, q1[0], params_after_q0)
        frozen0_logits = detector_from_prefix(model, q1[0], base_params)
        params_after_q1, ssl1, grad1, update1 = clipped_inner_update(
            model, objective_head, q1[0], q1[1], q1[2], params_after_q0, branch,
            create_graph=differentiable,
        )
        post1_logits = detector_from_prefix(model, q2[0], params_after_q1)
        frozen1_logits = detector_from_prefix(model, q2[0], base_params)
    post0_loss = binary_losses(post0_logits, q1[3])
    post1_loss = binary_losses(post1_logits, q2[3])
    frozen0_loss = binary_losses(frozen0_logits, q1[3]).detach()
    frozen1_loss = binary_losses(frozen1_logits, q2[3]).detach()
    target = torch.cat([q1[3], q2[3]], dim=0)
    post = torch.cat([post0_loss, post1_loss], dim=0)
    frozen = torch.cat([frozen0_loss, frozen1_loss], dim=0)
    deterioration_terms: list[torch.Tensor] = []
    missing = 0
    for label, weight in ((0.0, 0.8), (1.0, 0.2)):
        mask = target == label
        if bool(mask.any()):
            deterioration_terms.append(float(weight) * F.relu(post[mask] - frozen[mask]).mean())
        else:
            missing += 1
    deterioration = torch.stack(deterioration_terms).sum() if deterioration_terms else post.mean() * 0.0
    outer_loss = 0.8 * post0_loss.mean() + 0.2 * post1_loss.mean() + 2.0 * deterioration
    if branch == "band":
        outer_loss = outer_loss + 0.05 * (ssl0 + ssl1)
    return {
        "outer_loss": outer_loss,
        "post0_loss": post0_loss.mean(),
        "post1_loss": post1_loss.mean(),
        "frozen0_loss": frozen0_loss.mean(),
        "frozen1_loss": frozen1_loss.mean(),
        "future_loss": 0.5 * (post0_loss.mean() + post1_loss.mean()),
        "future_frozen_loss": 0.5 * (frozen0_loss.mean() + frozen1_loss.mean()),
        "nonseizure_post_loss": torch.cat([post0_loss[q1[3] == 0], post1_loss[q2[3] == 0]]).mean() if bool(((q1[3] == 0).any() or (q2[3] == 0).any())) else post0_loss.mean() * 0.0,
        "nonseizure_frozen_loss": torch.cat([frozen0_loss[q1[3] == 0], frozen1_loss[q2[3] == 0]]).mean() if bool(((q1[3] == 0).any() or (q2[3] == 0).any())) else frozen0_loss.mean() * 0.0,
        "deterioration": deterioration,
        "missing_class_terms": missing,
        "grad_norm": torch.stack(grad_norms + [grad0.detach(), grad1.detach()]).mean(),
        "update_norm": torch.stack(update_norms + [update0.detach(), update1.detach()]).mean(),
        "band_ssl_loss": 0.5 * (ssl0 + ssl1),
        "post0_logits": post0_logits,
        "post1_logits": post1_logits,
    }


def stable_classifier_hash(model: CHBJointModel) -> str:
    """Hash backbone and detector only; the prepared/learned Band head is auxiliary."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith("band_head."):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def objective_parameters(model: CHBJointModel, objective_head: LearnedAuxiliaryLoss | None, branch: str) -> list[nn.Parameter]:
    if branch == "band":
        return list(model.band_head.parameters())
    if objective_head is None:
        raise ValueError("learned branch missing objective head")
    return list(objective_head.parameters())


def fresh_objective(model: CHBJointModel, branch: str, device: torch.device) -> LearnedAuxiliaryLoss | None:
    if branch == "band":
        return None
    return LearnedAuxiliaryLoss().to(device)


def episode_metrics(result: dict[str, torch.Tensor | float | int]) -> dict[str, float | int]:
    def scalar(name: str) -> float:
        value = result[name]
        return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    return {
        "outer_loss": scalar("outer_loss"),
        "future_loss": scalar("future_loss"),
        "future_frozen_loss": scalar("future_frozen_loss"),
        "nonseizure_post_loss": scalar("nonseizure_post_loss"),
        "nonseizure_frozen_loss": scalar("nonseizure_frozen_loss"),
        "deterioration": scalar("deterioration"),
        "band_ssl_loss": scalar("band_ssl_loss"),
        "grad_norm": scalar("grad_norm"),
        "update_norm": scalar("update_norm"),
        "missing_class_terms": int(result["missing_class_terms"]),
    }


def evaluate_episode_specs(
    model: CHBJointModel,
    objective_head: LearnedAuxiliaryLoss | None,
    store: SignalStore,
    records: list[Record],
    specs: list[tuple[int, int]],
    device: torch.device,
    branch: str,
    cache: PrefixCache | None,
) -> dict[str, float | int]:
    rows: list[dict[str, float | int]] = []
    for record_index, start in specs:
        result = run_episode(
            model, objective_head, store, records[record_index], start, device, branch, cache,
            differentiable=False,
        )
        rows.append(episode_metrics(result))
    if not rows:
        raise RuntimeError("empty episode validation set")
    numeric = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in rows[0]
        if key != "missing_class_terms"
    }
    numeric["missing_class_terms"] = int(sum(int(row["missing_class_terms"]) for row in rows))
    numeric["episodes"] = len(rows)
    numeric["nonseizure_delta"] = numeric["nonseizure_post_loss"] - numeric["nonseizure_frozen_loss"]
    return numeric


def meta_checkpoint(
    path: Path,
    *,
    model: CHBJointModel,
    objective_head: LearnedAuxiliaryLoss | None,
    optimizer: torch.optim.Optimizer,
    fold: int,
    branch: str,
    seed: int,
    step: int,
    validation: dict[str, Any] | None,
    classifier_hash: str,
    history: list[dict[str, Any]],
) -> None:
    atomic_torch_save(path, {
        "release_id": RELEASE_ID,
        "stage": f"meta_{branch}",
        "fold": fold,
        "branch": branch,
        "seed": seed,
        "step": step,
        "model": model.state_dict(),
        "objective": objective_head.state_dict() if objective_head is not None else None,
        "optimizer": optimizer.state_dict(),
        "validation": validation,
        "classifier_hash": classifier_hash,
        "history": history,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "saved_at": utc_now(),
    })


def run_meta(args: argparse.Namespace) -> dict[str, Any]:
    if args.fold not in (0, 1):
        raise ValueError("formal release is limited to folds 0 and 1")
    if args.branch not in {"band", "learned"}:
        raise ValueError(args.branch)
    run_dir = meta_run_dir(args.output_root, args.fold, args.branch, args.seed)
    completed_path = run_dir / "completed.json"
    if completed_path.is_file() and not args.force:
        return json.loads(completed_path.read_text())
    set_seed(args.seed + 40_000 + args.fold * 10 + (0 if args.branch == "band" else 1))
    device = torch.device(args.device)
    model, common_path, common_payload = load_common_model(args, args.fold, prepared_band=True, device=device)
    classifier_hash = stable_classifier_hash(model)
    train_rows = load_rows(args.fold, "train", args.windows, args.fold_root)
    val_rows = load_rows(args.fold, "validation", args.windows, args.fold_root)
    train_records = records_from_rows(train_rows)
    val_records = records_from_rows(val_rows)
    store = SignalStore(args.cache_root)
    prefix_cache = PrefixCache(args.prefix_cache, classifier_hash, max_bytes=args.prefix_cache_bytes, enabled=not args.no_prefix_cache)
    fixed_specs = fixed_episode_specs(val_records, args.validation_episodes, args.seed + 50_000 + args.fold)
    objective_head = fresh_objective(model, args.branch, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # The Band head is outer-trainable only for the Band branch.  It is fixed
    # for the learned objective, whose MLP is the sole outer parameter set.
    if args.branch == "band":
        for parameter in model.band_head.parameters():
            parameter.requires_grad_(True)
    if objective_head is not None:
        objective_head.eval()
        for parameter in objective_head.parameters():
            parameter.requires_grad_(True)
    outer_parameters = objective_parameters(model, objective_head, args.branch)
    optimizer = torch.optim.AdamW(outer_parameters, lr=OUTER_LR, weight_decay=0.0)
    step = 0
    history: list[dict[str, Any]] = []
    best_future = float("inf")
    best_nonseizure_delta = float("inf")
    best_future_step: int | None = None
    best_nonseizure_step: int | None = None
    stale_evals = 0
    if (run_dir / "last.pt").is_file() and not args.force:
        saved = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        if saved.get("branch") != args.branch or int(saved.get("fold", -1)) != args.fold:
            raise RuntimeError("meta checkpoint identity mismatch")
        if saved.get("classifier_hash") != classifier_hash:
            raise RuntimeError("common classifier hash changed while resuming meta training")
        model.load_state_dict(saved["model"], strict=True)
        if objective_head is not None:
            objective_head.load_state_dict(saved["objective"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        step = int(saved.get("step", 0))
        history = list(saved.get("history", []))
        for row in history:
            if row.get("kind") != "validation":
                continue
            metrics = row["metrics"]
            best_future = min(best_future, float(metrics["future_loss"]))
            best_nonseizure_delta = min(best_nonseizure_delta, float(metrics["nonseizure_delta"]))
        if history:
            best_future_step = next((int(row["step"]) for row in history if row.get("kind") == "validation" and float(row["metrics"]["future_loss"]) == best_future), None)
            best_nonseizure_step = next((int(row["step"]) for row in history if row.get("kind") == "validation" and float(row["metrics"]["nonseizure_delta"]) == best_nonseizure_delta), None)
    model.eval()
    if objective_head is not None:
        objective_head.eval()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **formal_config(),
        "status": "running",
        "stage": f"meta_{args.branch}",
        "fold": args.fold,
        "branch": args.branch,
        "seed": args.seed,
        "started_at": utc_now(),
        "common_classifier": str(common_path),
        "classifier_hash": classifier_hash,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_records_with_episodes": len(train_records),
        "validation_records_with_episodes": len(val_records),
        "validation_episode_count": len(fixed_specs),
        "prefix_cache": str(prefix_cache.root),
        "prefix_cache_max_bytes": args.prefix_cache_bytes,
        "device": str(device),
        "test_partition_read": False,
    }
    atomic_json(run_dir / "manifest.json", manifest)
    rng = np.random.default_rng(args.seed + 60_000 + args.fold * 10 + (0 if args.branch == "band" else 1))
    started = time.monotonic()
    last_progress = started
    processed_chunks = 0
    last_validation: dict[str, Any] | None = None
    while step < args.meta_steps:
        optimizer.zero_grad(set_to_none=True)
        batch_metrics: list[dict[str, float | int]] = []
        for episode_index in range(args.effective_episodes):
            record, start = choose_episode(train_records, rng)
            result = run_episode(
                model, objective_head, store, record, start, device, args.branch, prefix_cache,
                differentiable=True,
            )
            (result["outer_loss"] / args.effective_episodes).backward()
            batch_metrics.append(episode_metrics(result))
            processed_chunks += BURN_IN_CHUNKS + QUERY_CHUNKS
            atomic_json(run_dir / "chunk_progress.json", {
                "release_id": RELEASE_ID,
                "status": "running",
                "fold": args.fold,
                "branch": args.branch,
                "outer_step": step + 1,
                "episode": episode_index + 1,
                "episodes_per_step": args.effective_episodes,
                "record": record.recording,
                "patient": record.patient,
                "episode_start_row": start,
                "completed_chunks": processed_chunks,
                "elapsed_s": time.monotonic() - started,
            })
        torch.nn.utils.clip_grad_norm_(outer_parameters, 1.0)
        optimizer.step()
        step += 1
        now = time.monotonic()
        if now - last_progress >= 30.0:
            atomic_json(run_dir / "progress.json", {
                **manifest,
                "status": "training",
                "step": step,
                "max_steps": args.meta_steps,
                "processed_chunks": processed_chunks,
                "chunks_per_s": processed_chunks / max(now - started, 1e-9),
                "last_batch": {key: float(np.mean([row[key] for row in batch_metrics])) for key in batch_metrics[0] if key != "missing_class_terms"},
                "missing_class_terms": int(sum(int(row["missing_class_terms"]) for row in batch_metrics)),
                "elapsed_s": now - started,
                "gpu_peak_mib": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
            })
            last_progress = now
        should_validate = step % args.eval_interval == 0 or step == args.meta_steps
        if should_validate:
            validation = evaluate_episode_specs(model, objective_head, store, val_records, fixed_specs, device, args.branch, prefix_cache)
            last_validation = validation
            row = {"kind": "validation", "step": step, "metrics": validation, "elapsed_s": now - started}
            history.append(row)
            future_improved = float(validation["future_loss"]) < best_future - 1e-7
            nonseizure_improved = float(validation["nonseizure_delta"]) < best_nonseizure_delta - 1e-7
            if future_improved:
                best_future = float(validation["future_loss"])
                best_future_step = step
                meta_checkpoint(run_dir / "best_future.pt", model=model, objective_head=objective_head, optimizer=optimizer, fold=args.fold, branch=args.branch, seed=args.seed, step=step, validation=validation, classifier_hash=classifier_hash, history=history)
            if nonseizure_improved:
                best_nonseizure_delta = float(validation["nonseizure_delta"])
                best_nonseizure_step = step
                meta_checkpoint(run_dir / "best_nonseizure.pt", model=model, objective_head=objective_head, optimizer=optimizer, fold=args.fold, branch=args.branch, seed=args.seed, step=step, validation=validation, classifier_hash=classifier_hash, history=history)
            if future_improved or nonseizure_improved:
                stale_evals = 0
            else:
                stale_evals += 1
            atomic_json(run_dir / "validation_history.json", {"evaluations": [item for item in history if item.get("kind") == "validation"]})
            atomic_torch_save(run_dir / "last.pt", {
                "release_id": RELEASE_ID,
                "stage": f"meta_{args.branch}",
                "fold": args.fold,
                "branch": args.branch,
                "seed": args.seed,
                "step": step,
                "model": model.state_dict(),
                "objective": objective_head.state_dict() if objective_head is not None else None,
                "optimizer": optimizer.state_dict(),
                "validation": validation,
                "classifier_hash": classifier_hash,
                "history": history,
            })
            print(json.dumps(row, allow_nan=True), flush=True)
            if step >= args.min_meta_steps and stale_evals >= args.meta_patience:
                break
    # If a process stops before its first scheduled evaluation, retain a
    # recoverable final state but do not call the stage complete.
    if not (run_dir / "best_future.pt").is_file() and last_validation is not None:
        meta_checkpoint(run_dir / "best_future.pt", model=model, objective_head=objective_head, optimizer=optimizer, fold=args.fold, branch=args.branch, seed=args.seed, step=step, validation=last_validation, classifier_hash=classifier_hash, history=history)
    final_classifier_hash = stable_classifier_hash(model)
    completed = {
        **manifest,
        "status": "completed",
        "completed_at": utc_now(),
        "steps_completed": step,
        "best_future_step": best_future_step,
        "best_future_loss": best_future if np.isfinite(best_future) else None,
        "best_nonseizure_step": best_nonseizure_step,
        "best_nonseizure_delta": best_nonseizure_delta if np.isfinite(best_nonseizure_delta) else None,
        "stale_evaluations": stale_evals,
        "classifier_hash_after": final_classifier_hash,
        "classifier_unchanged": final_classifier_hash == classifier_hash,
        "best_future_checkpoint": str(run_dir / "best_future.pt") if (run_dir / "best_future.pt").is_file() else None,
        "best_nonseizure_checkpoint": str(run_dir / "best_nonseizure.pt") if (run_dir / "best_nonseizure.pt").is_file() else None,
        "elapsed_s": time.monotonic() - started,
        "test_partition_read": False,
    }
    if final_classifier_hash != classifier_hash:
        completed["status"] = "failed_classifier_mutation_check"
    atomic_json(completed_path, completed)
    atomic_json(run_dir / "progress.json", completed)
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("supervised", "band_prepare", "meta"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--branch", choices=("band", "learned"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/reports" / RELEASE_ID)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--prefix-cache", type=Path, default=Path("/mnt/d/EEGData/meta_ttt_prefix_v2"))
    parser.add_argument("--prefix-cache-bytes", type=int, default=128 * 2**30)
    parser.add_argument("--pretrained", type=Path, default=EXTERNAL_ROOT / "pretrained_weights/pretrained_weights.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--classifier-batch", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--classifier-epochs", type=int, default=50)
    parser.add_argument("--classifier-min-epochs", type=int, default=5)
    parser.add_argument("--classifier-patience", type=int, default=7)
    parser.add_argument("--band-steps", type=int, default=500)
    parser.add_argument("--band-batch", type=int, default=256)
    parser.add_argument("--meta-steps", type=int, default=OUTER_MAX_STEPS)
    parser.add_argument("--min-meta-steps", type=int, default=MIN_META_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--meta-patience", type=int, default=META_PATIENCE_EVALS)
    parser.add_argument("--effective-episodes", type=int, default=8)
    parser.add_argument("--validation-episodes", type=int, default=256)
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != SEED:
        raise ValueError("formal release locks seed=3407")
    if args.mode == "supervised":
        result = run_supervised(args)
    elif args.mode == "band_prepare":
        result = run_prepare_band_head(args)
    else:
        if args.branch is None:
            raise ValueError("--branch is required for --mode meta")
        result = run_meta(args)
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
