#!/usr/bin/env python3
"""Two-fold repaired Band-TTT retraining release.

The common supervised classifier and both meta objectives are independent of
the paused legacy Band-TTT queue.  The first ten CBraMod blocks are a detached
prefix; only encoder blocks 10 and 11 are functionally updated in meta TTT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

RELEASE_ID = "meta-ttt-chbmit-v2-repaired"
SEED = 3407
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/NeuroTTT/CBraMod")
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))

from chbmit_groupkfold.data import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_FOLDS,
    DEFAULT_WINDOWS,
    load_rows,
    make_eval_loader,
    make_train_loader,
)
from chbmit_groupkfold.model import CHBJointModel  # noqa: E402
from chbmit_groupkfold.transforms import deterministic_band_view  # noqa: E402

CHUNK_SIZE = 16
CHUNK_STRIDE = 20
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
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def stable_classifier_hash(model: CHBJointModel) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith("band_head."):
            continue
        digest.update(name.encode())
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
    return CHBJointModel(pretrained).to(device)


def load_common_model(args: argparse.Namespace, fold: int, *, prepared_band: bool, device: torch.device) -> tuple[CHBJointModel, Path, dict[str, Any]]:
    checkpoint = classifier_run_dir(args.output_root, fold, args.seed) / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = model_from_official(args.pretrained, device)
    model.load_state_dict(payload["model"], strict=True)
    if prepared_band:
        preparation = torch.load(band_prepare_path(args.output_root, fold, args.seed), map_location="cpu", weights_only=False)
        model.band_head.load_state_dict(preparation["band_head"], strict=True)
    return model, checkpoint, payload


def _decay_groups(model: nn.Module) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
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
    run_dir = classifier_run_dir(args.output_root, args.fold, args.seed)
    completed_path = run_dir / "completed.json"
    if completed_path.is_file() and not args.force:
        return json.loads(completed_path.read_text())
    set_seed(args.seed + args.fold)
    device = torch.device(args.device)
    train_rows = load_rows(args.fold, "train", args.windows, args.fold_root)
    val_rows = load_rows(args.fold, "validation", args.windows, args.fold_root)
    physical_batch = args.classifier_batch
    if 128 % physical_batch:
        raise ValueError("classifier batch must divide effective batch 128")
    positives = int((train_rows.label.astype(int) == 1).sum())
    updates_per_epoch = max(1, math.ceil(2 * positives / 128))
    accumulation = 128 // physical_batch
    train_loader, sampler = make_train_loader(
        train_rows, batch_size=physical_batch, steps=updates_per_epoch * accumulation,
        seed=args.seed + args.fold, workers=args.workers, cache_root=args.cache_root,
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
        **formal_config(), "status": "running", "stage": "common_classifier",
        "fold": args.fold, "seed": args.seed, "started_at": utc_now(),
        "train_rows": len(train_rows), "train_positive_rows": positives,
        "validation_rows": len(val_rows), "updates_per_epoch": updates_per_epoch,
        "physical_batch": physical_batch, "device": str(device),
        "source_hashes": {"windows": sha256(args.windows), "fold": sha256(args.fold_root / f"fold_{args.fold}.json"), "pretrained": sha256(args.pretrained)},
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
        for micro_index, (signal, target, _) in enumerate(train_loader):
            signal = signal.to(device=device, dtype=torch.float32, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = F.binary_cross_entropy_with_logits(model.detect(signal), target)
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
                    **manifest, "status": "training", "epoch": epoch, "update": global_update,
                    "processed_windows": processed, "windows_per_s": processed / max(now - started, 1e-9),
                    "elapsed_s": now - started,
                    "gpu_peak_mib": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
                })
                last_progress = now
        validation = classifier_validation(model, val_loader, device)
        row = {"epoch": epoch, "update": global_update, "train_loss": float(np.mean(epoch_losses)), "validation": validation, "epoch_seconds": time.monotonic() - epoch_started, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        improved = validation["auprc"] > best_auprc + 1e-5
        if improved:
            best_auprc = validation["auprc"]
            patience = 0
            atomic_torch_save(run_dir / "best.pt", {"release_id": RELEASE_ID, "stage": "common_classifier", "fold": args.fold, "seed": args.seed, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "update": global_update, "best_auprc": best_auprc, "history": history})
        else:
            patience += 1
        atomic_torch_save(run_dir / "last.pt", {"release_id": RELEASE_ID, "stage": "common_classifier", "fold": args.fold, "seed": args.seed, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "update": global_update, "best_auprc": best_auprc, "patience": patience, "history": history})
        atomic_json(run_dir / "history.json", {"epochs": history})
        print(json.dumps(row, allow_nan=True), flush=True)
        if epoch + 1 >= args.classifier_min_epochs and patience >= args.classifier_patience:
            break
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    final_model = model_from_official(args.pretrained, torch.device("cpu"))
    final_model.load_state_dict(best["model"], strict=True)
    completed = {
        **manifest, "status": "completed", "completed_at": utc_now(),
        "epochs_completed": len(history), "updates_completed": global_update,
        "best_validation_auprc": best_auprc, "best_checkpoint": str(run_dir / "best.pt"),
        "classifier_hash": stable_classifier_hash(final_model), "elapsed_s": time.monotonic() - started,
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
        loss = F.cross_entropy(model.band_logits(features), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 50 == 0:
            atomic_json(output_path.parent / "band_prepare_progress.json", {"release_id": RELEASE_ID, "stage": "band_head_prepare", "fold": args.fold, "step": step + 1, "steps": args.band_steps, "loss": float(np.mean(losses[-50:])), "elapsed_s": time.monotonic() - started})
    classifier_model = model_from_official(args.pretrained, torch.device("cpu"))
    classifier_model.load_state_dict(common_payload["model"], strict=True)
    classifier_hash = stable_classifier_hash(classifier_model)
    prepared_hash = state_hash(model.band_head)
    atomic_torch_save(output_path, {"release_id": RELEASE_ID, "stage": "band_head_prepare", "fold": args.fold, "seed": args.seed, "common_classifier": str(common_path), "classifier_hash": classifier_hash, "band_head": {k: v.detach().cpu() for k, v in model.band_head.state_dict().items()}, "band_head_hash": prepared_hash, "steps": args.band_steps, "mean_loss_last50": float(np.mean(losses[-50:]))})
    completed = {"release_id": RELEASE_ID, "stage": "band_head_prepare", "fold": args.fold, "seed": args.seed, "status": "completed", "checkpoint": str(output_path), "classifier_hash": classifier_hash, "band_head_hash": prepared_hash, "steps": args.band_steps, "mean_loss_last50": float(np.mean(losses[-50:])), "elapsed_s": time.monotonic() - started}
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
    seizure_candidate_starts: np.ndarray | None = None


class SignalStore:
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
            self._views.pop(self._order.pop(0), None)
        return view

    def read(self, record: Record, positions: np.ndarray) -> np.ndarray:
        view = self._view(record.relative_path)
        output: list[np.ndarray] = []
        for start in record.starts[positions]:
            left = int(round(float(start) * 200.0))
            signal = np.asarray(view[:, left:left + 2000], dtype=np.float32)
            if signal.shape != (16, 2000) or not np.isfinite(signal).all():
                raise ValueError(f"invalid cached window {record.recording}:{start}")
            output.append(np.ascontiguousarray(signal.reshape(16, 10, 200)))
        return np.stack(output, axis=0)


def records_from_rows(rows: pd.DataFrame, *, min_episode: int = EPISODE_SPAN) -> list[Record]:
    records: list[Record] = []
    for (patient, recording), group in rows.groupby(["patient", "recording"], sort=False):
        group = group.sort_values("start", kind="stable").reset_index(drop=True)
        starts = group.start.to_numpy(dtype=np.float64)
        if len(starts) < min_episode:
            continue
        adjacent = np.isclose(np.diff(starts), 2.0, atol=0.02)
        if len(adjacent) < CHUNK_SIZE - 1:
            continue
        # Require regular 2-s windows within each 16-window chunk.  The
        # 4-window gaps between chunks are deliberately not required to be
        # present: some EDF manifests contain a genuine 30-s row gap around
        # an annotated seizure.  Treating that as an invalid whole episode
        # removed every positive future query while the per-chunk EEG was
        # still valid and causal.
        chunk_ok = np.convolve(adjacent.astype(np.int32), np.ones(CHUNK_SIZE - 1, dtype=np.int32), mode="valid") == (CHUNK_SIZE - 1)
        candidate_count = len(starts) - min_episode + 1
        if candidate_count <= 0:
            continue
        valid = np.ones(candidate_count, dtype=bool)
        for chunk in range(BURN_IN_CHUNKS + QUERY_CHUNKS):
            offset = chunk * CHUNK_STRIDE
            valid &= chunk_ok[offset:offset + candidate_count]
        candidates = np.flatnonzero(valid).astype(np.int64)
        if len(candidates) == 0:
            continue
        records.append(Record(
            patient=str(patient), recording=str(recording), relative_path=str(group.relative_path.iloc[0]), rows=group,
            starts=starts, labels=group.label.to_numpy(dtype=np.float32), sample_ids=group.sample_id.astype(str).to_numpy(),
            candidate_starts=candidates, seizure_record=bool((group.label.astype(int) == 1).any()),
        ))
    if not records:
        raise RuntimeError(f"no continuous {min_episode}-window records found")
    return records


def choose_episode(records: list[Record], rng: np.random.Generator, *, seizure_bias: bool | None = None) -> tuple[Record, int]:
    if seizure_bias is None:
        seizure_bias = bool(rng.random() < 0.25)
    if seizure_bias:
        pool = [record for record in records if record.seizure_record and len(positive_query_candidates(record))]
    else:
        pool = records
    if not pool:
        pool = records
    weights = np.asarray([len(record.rows) * max(1, len(record.candidate_starts)) for record in pool], dtype=np.float64)
    weights /= weights.sum()
    record = pool[int(rng.choice(len(pool), p=weights))]
    candidates = record.candidate_starts
    if seizure_bias:
        # A seizure-containing record is not enough: an arbitrary 136-window
        # excerpt can still miss its seizure in both future query chunks.
        # Cache candidate starts whose q1/q2 query windows include a positive
        # label so the registered 25% seizure-fragment stream really carries
        # the intended class term without stitching labels or records.
        positive = positive_query_candidates(record)
        if len(positive):
            candidates = positive
    return record, int(rng.choice(candidates))


def fixed_episode_specs(records: list[Record], count: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    specs: list[tuple[int, int]] = []
    seizure_count = int(round(count * 0.25))
    for index in range(count):
        # Make the registered 75/25 split exact in the fixed validation set;
        # the order remains interleaved so it is not a temporal block.
        seizure_bias = index % 4 == 0 and index // 4 < seizure_count
        record, start = choose_episode(records, rng, seizure_bias=seizure_bias)
        specs.append((records.index(record), start))
    return specs


def positive_query_candidates(record: Record) -> np.ndarray:
    if record.seizure_candidate_starts is None:
        positive: list[int] = []
        for candidate in record.candidate_starts:
            q1 = record.labels[int(candidate + (BURN_IN_CHUNKS + 1) * CHUNK_STRIDE):int(candidate + (BURN_IN_CHUNKS + 1) * CHUNK_STRIDE + CHUNK_SIZE)]
            q2 = record.labels[int(candidate + (BURN_IN_CHUNKS + 2) * CHUNK_STRIDE):int(candidate + (BURN_IN_CHUNKS + 2) * CHUNK_STRIDE + CHUNK_SIZE)]
            if bool((q1 == 1).any() or (q2 == 1).any()):
                positive.append(int(candidate))
        record.seizure_candidate_starts = np.asarray(positive, dtype=np.int64)
    return record.seizure_candidate_starts


class PrefixCache:
    """Advisory disk LRU plus a bounded pinned GPU validation cache."""

    def __init__(self, root: Path, source_hash: str, max_bytes: int, enabled: bool = True, gpu_pin_bytes: int = 0) -> None:
        self.root = Path(root) / source_hash
        self.index_path = self.root / "index.json"
        self.max_bytes = int(max_bytes)
        self.enabled = bool(enabled)
        self.index: dict[str, dict[str, Any]] = {}
        self.dirty = 0
        self.gpu_pin_bytes = int(gpu_pin_bytes)
        self.gpu_pinned_bytes = 0
        self.gpu_pinned: dict[str, torch.Tensor] = {}
        self.pin_gpu_writes = False
        self.gpu_hits = 0
        self.disk_hits = 0
        if enabled and self.index_path.is_file():
            try:
                self.index = json.loads(self.index_path.read_text())
            except (OSError, ValueError):
                self.index = {}

    def _key(self, sample_id: str, view: str) -> str:
        return hashlib.sha256(f"{sample_id}|{view}".encode()).hexdigest()

    def _maybe_pin_gpu(self, key: str, tensor: torch.Tensor) -> None:
        if not self.pin_gpu_writes or tensor.device.type != "cuda" or key in self.gpu_pinned:
            return
        size = tensor.numel() * tensor.element_size()
        if self.gpu_pinned_bytes + size > self.gpu_pin_bytes:
            return
        value = tensor.detach()
        self.gpu_pinned[key] = value
        self.gpu_pinned_bytes += size

    @contextmanager
    def pin_validation_on_gpu(self):
        previous = self.pin_gpu_writes
        self.pin_gpu_writes = self.gpu_pin_bytes > 0
        try:
            yield
        finally:
            self.pin_gpu_writes = previous

    def get(self, sample_ids: list[str], view: str, device: torch.device) -> tuple[list[torch.Tensor | None], list[int]]:
        values: list[torch.Tensor | None] = [None] * len(sample_ids)
        missing: list[int] = []
        if not self.enabled:
            return values, list(range(len(sample_ids)))
        changed = False
        for index, sample_id in enumerate(sample_ids):
            key = self._key(sample_id, view)
            pinned = self.gpu_pinned.get(key)
            if pinned is not None:
                values[index] = pinned
                self.gpu_hits += 1
                continue
            entry = self.index.get(key)
            if not entry:
                missing.append(index)
                continue
            path = self.root / entry["file"]
            try:
                values[index] = torch.from_numpy(np.load(path, allow_pickle=False)).to(device=device)
                self.disk_hits += 1
                self._maybe_pin_gpu(key, values[index])
                entry["atime"] = time.time()
                changed = True
            except (OSError, ValueError, KeyError, EOFError):
                self.index.pop(key, None)
                # An interrupted np.save can leave a truncated entry behind.
                # Drop only that exact cache file so the caller recomputes it.
                try:
                    path.unlink()
                except OSError:
                    pass
                missing.append(index)
                changed = True
        return values, missing

    def put(self, sample_ids: list[str], view: str, tensors: torch.Tensor) -> None:
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        now = time.time()
        detached = tensors.detach()
        arrays = detached.float().cpu().numpy()
        for sample_id, gpu_tensor, tensor in zip(sample_ids, detached, arrays, strict=True):
            key = self._key(sample_id, view)
            path = self.root / f"{key}.npy"
            # Multiple independent record/candidate evaluators may share the
            # immutable prefix cache.  A PID-qualified staging name preserves
            # atomic publication without processes deleting each other's tmp.
            temporary = path.with_suffix(f".npy.tmp.{os.getpid()}")
            with temporary.open("wb") as stream:
                np.save(stream, tensor, allow_pickle=False)
            os.replace(temporary, path)
            self.index[key] = {"file": path.name, "bytes": path.stat().st_size, "atime": now}
            self._maybe_pin_gpu(key, gpu_tensor)
            self.dirty += 1
        # Rewriting a large JSON index for every 16-window chunk dominates
        # meta throughput on mounted Windows filesystems.  The tensor files
        # are already atomic; an index may lag after an interruption and those
        # entries are simply recomputed.  Batch index/eviction work instead.
        if self.dirty >= 2048:
            self._evict()
            self._flush()

    def _evict(self) -> None:
        total = sum(int(item.get("bytes", 0)) for item in self.index.values())
        for key, item in sorted(self.index.items(), key=lambda pair: float(pair[1].get("atime", 0))):
            if total <= self.max_bytes:
                break
            try:
                (self.root / item["file"]).unlink(missing_ok=True)
            finally:
                total -= int(item.get("bytes", 0))
                self.index.pop(key, None)

    def _flush(self) -> None:
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_suffix(f".json.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(self.index, sort_keys=True))
        os.replace(temporary, self.index_path)
        self.dirty = 0

    def flush(self) -> None:
        if self.enabled and self.dirty:
            self._evict()
            self._flush()


def encode_prefix(model: CHBJointModel, signal: torch.Tensor) -> torch.Tensor:
    value = model.backbone.patch_embedding(signal)
    for layer in model.backbone.encoder.layers[:TAIL_LAYER_START]:
        value = layer(value)
    return value.detach()


def make_prefix_batch(model: CHBJointModel, signals: torch.Tensor, sample_ids: list[str], view: str, cache: PrefixCache | None) -> torch.Tensor:
    if cache is None:
        with torch.no_grad():
            return encode_prefix(model, signals)
    values, missing = cache.get(sample_ids, view, signals.device)
    if missing:
        with torch.no_grad():
            computed = encode_prefix(model, signals[missing])
        cache.put([sample_ids[index] for index in missing], view, computed)
        for index, value in zip(missing, computed, strict=True):
            values[index] = value
    if any(value is None for value in values):
        raise RuntimeError("prefix cache assembly failed")
    return torch.stack([value for value in values if value is not None])


def episode_positions(start: int) -> list[np.ndarray]:
    return [start + chunk * CHUNK_STRIDE + np.arange(CHUNK_SIZE, dtype=np.int64) for chunk in range(BURN_IN_CHUNKS + QUERY_CHUNKS)]


def _layer_param_values(params: dict[str, torch.Tensor], layer_index: int) -> dict[str, torch.Tensor]:
    prefix = f"backbone.encoder.layers.{layer_index}."
    values = {name[len(prefix):]: value for name, value in params.items() if name.startswith(prefix)}
    if not values:
        raise KeyError(f"missing functional parameters for layer {layer_index}")
    return values


def encode_tail(model: CHBJointModel, prefix: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    value = prefix
    for layer_index in range(TAIL_LAYER_START, TAIL_LAYER_END):
        layer = model.backbone.encoder.layers[layer_index]
        value = functional_call(layer, _layer_param_values(params, layer_index), (value,), strict=False)
    return value


def tail_named_parameters(model: CHBJointModel) -> dict[str, nn.Parameter]:
    return {
        name: parameter for name, parameter in model.named_parameters()
        if name.startswith("backbone.encoder.layers.10.") or name.startswith("backbone.encoder.layers.11.")
    }


def detached_tail_values(model: CHBJointModel, *, requires_grad: bool) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone().requires_grad_(requires_grad) for name, parameter in tail_named_parameters(model).items()}


class LearnedAuxiliaryLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4005, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, raw_features: torch.Tensor, transformed_features: torch.Tensor, labels: torch.Tensor, fixed_band_head: nn.Module) -> torch.Tensor:
        raw_vector = raw_features.mean(dim=1).flatten(1)
        transformed_vector = transformed_features.mean(dim=1).flatten(1)
        one_hot = F.one_hot(labels, num_classes=5).to(dtype=raw_vector.dtype)
        inputs = torch.cat([raw_vector, transformed_vector, one_hot], dim=1)
        if inputs.shape[1] != 4005:
            raise RuntimeError(f"learned loss input {inputs.shape[1]} != 4005")
        fixed_ce = F.cross_entropy(fixed_band_head(transformed_features.mean(dim=1).flatten(1)), labels, reduction="none")
        return F.softplus(fixed_ce + self.net(inputs).squeeze(-1))


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
            raise ValueError("learned branch requires MLP")
        per_sample = objective_head(encode_tail(model, raw_prefix, params), transformed_features, labels, model.band_head)
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
    gradients = torch.autograd.grad(loss, values, create_graph=create_graph, retain_graph=create_graph, allow_unused=True)
    usable = [gradient for gradient in gradients if gradient is not None]
    if not usable:
        raise RuntimeError(f"{branch} produced no inner gradient")
    grad_norm = torch.stack([gradient.float().square().sum() for gradient in usable]).sum().sqrt()
    scale = torch.clamp(torch.as_tensor(GRAD_CLIP, device=grad_norm.device) / (grad_norm + 1e-12), max=1.0)
    updated = {
        name: value - INNER_LR * (gradient if gradient is not None else torch.zeros_like(value)) * scale
        for (name, value), gradient in zip(params.items(), gradients, strict=True)
    }
    update_norm = torch.stack([(updated[name] - params[name]).float().square().sum() for name in params]).sum().sqrt()
    return updated, loss, grad_norm, update_norm


def binary_losses(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none")


def detector_from_prefix(model: CHBJointModel, prefix: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    return model.detect_from_features(encode_tail(model, prefix, params))


def _episode_chunk(store: SignalStore, record: Record, positions: np.ndarray, device: torch.device, cache: PrefixCache | None, model: CHBJointModel) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
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
    positions = episode_positions(start)
    base_params = detached_tail_values(model, requires_grad=False)
    params = detached_tail_values(model, requires_grad=True)
    chunks = [_episode_chunk(store, record, item, device, cache, model) for item in positions]
    diagnostics: list[torch.Tensor] = []
    update_diagnostics: list[torch.Tensor] = []
    for raw_prefix, transformed_prefix, labels, _, _ in chunks[:BURN_IN_CHUNKS]:
        with torch.enable_grad():
            params, loss, grad_norm, update_norm = clipped_inner_update(model, objective_head, raw_prefix, transformed_prefix, labels, params, branch, create_graph=False)
        params = {name: value.detach().requires_grad_(True) for name, value in params.items()}
        diagnostics.append(grad_norm.detach())
        update_diagnostics.append(update_norm.detach())

    q0 = chunks[BURN_IN_CHUNKS]
    q1 = chunks[BURN_IN_CHUNKS + 1]
    q2 = chunks[BURN_IN_CHUNKS + 2]
    with torch.enable_grad():
        # q0 is predicted before it is used as support; q1/q2 are the two
        # future queries whose classification losses drive the outer update.
        _ = detector_from_prefix(model, q0[0], params)
        params_after_q0, ssl0, grad0, update0 = clipped_inner_update(model, objective_head, q0[0], q0[1], q0[2], params, branch, create_graph=differentiable)
        post0_logits = detector_from_prefix(model, q1[0], params_after_q0)
        frozen0_logits = detector_from_prefix(model, q1[0], base_params)
        params_after_q1, ssl1, grad1, update1 = clipped_inner_update(model, objective_head, q1[0], q1[1], q1[2], params_after_q0, branch, create_graph=differentiable)
        post1_logits = detector_from_prefix(model, q2[0], params_after_q1)
        frozen1_logits = detector_from_prefix(model, q2[0], base_params)
    post0 = binary_losses(post0_logits, q1[3])
    post1 = binary_losses(post1_logits, q2[3])
    frozen0 = binary_losses(frozen0_logits, q1[3]).detach()
    frozen1 = binary_losses(frozen1_logits, q2[3]).detach()
    target = torch.cat([q1[3], q2[3]])
    post = torch.cat([post0, post1])
    frozen = torch.cat([frozen0, frozen1])
    penalties: list[torch.Tensor] = []
    missing = 0
    for label, weight in ((0.0, 0.8), (1.0, 0.2)):
        mask = target == label
        if bool(mask.any()):
            penalties.append(float(weight) * F.relu(post[mask] - frozen[mask]).mean())
        else:
            missing += 1
    deterioration = torch.stack(penalties).sum() if penalties else post.mean() * 0.0
    outer = 0.8 * post0.mean() + 0.2 * post1.mean() + 2.0 * deterioration
    if branch == "band":
        outer = outer + 0.05 * (ssl0 + ssl1)
    ns0 = post0[q1[3] == 0]
    ns1 = post1[q2[3] == 0]
    fns0 = frozen0[q1[3] == 0]
    fns1 = frozen1[q2[3] == 0]
    ns_post = torch.cat([item for item in (ns0, ns1) if item.numel()]).mean() if (ns0.numel() + ns1.numel()) else post0.mean() * 0.0
    ns_frozen = torch.cat([item for item in (fns0, fns1) if item.numel()]).mean() if (fns0.numel() + fns1.numel()) else frozen0.mean() * 0.0
    return {
        "outer_loss": outer, "post0_loss": post0.mean(), "post1_loss": post1.mean(),
        "frozen0_loss": frozen0.mean(), "frozen1_loss": frozen1.mean(),
        "future_loss": 0.5 * (post0.mean() + post1.mean()),
        "future_frozen_loss": 0.5 * (frozen0.mean() + frozen1.mean()),
        "nonseizure_post_loss": ns_post, "nonseizure_frozen_loss": ns_frozen,
        "deterioration": deterioration, "missing_class_terms": missing,
        "grad_norm": torch.stack(diagnostics + [grad0.detach(), grad1.detach()]).mean(),
        "update_norm": torch.stack(update_diagnostics + [update0.detach(), update1.detach()]).mean(),
        "band_ssl_loss": 0.5 * (ssl0 + ssl1), "post0_logits": post0_logits, "post1_logits": post1_logits,
    }


def episode_metrics(result: dict[str, torch.Tensor | float | int]) -> dict[str, float | int]:
    def value(name: str) -> float:
        item = result[name]
        return float(item.detach().cpu()) if isinstance(item, torch.Tensor) else float(item)
    return {
        "outer_loss": value("outer_loss"), "future_loss": value("future_loss"),
        "future_frozen_loss": value("future_frozen_loss"),
        "nonseizure_post_loss": value("nonseizure_post_loss"),
        "nonseizure_frozen_loss": value("nonseizure_frozen_loss"),
        "deterioration": value("deterioration"), "band_ssl_loss": value("band_ssl_loss"),
        "grad_norm": value("grad_norm"), "update_norm": value("update_norm"),
        "missing_class_terms": int(result["missing_class_terms"]),
    }


def evaluate_episode_specs(model: CHBJointModel, objective_head: LearnedAuxiliaryLoss | None, store: SignalStore, records: list[Record], specs: list[tuple[int, int]], device: torch.device, branch: str, cache: PrefixCache | None) -> dict[str, float | int]:
    metrics = [episode_metrics(run_episode(model, objective_head, store, records[index], start, device, branch, cache, differentiable=False)) for index, start in specs]
    if not metrics:
        raise RuntimeError("empty validation episode list")
    summary = {key: float(np.mean([float(row[key]) for row in metrics])) for key in metrics[0] if key != "missing_class_terms"}
    summary["missing_class_terms"] = int(sum(int(row["missing_class_terms"]) for row in metrics))
    summary["episodes"] = len(metrics)
    summary["nonseizure_delta"] = summary["nonseizure_post_loss"] - summary["nonseizure_frozen_loss"]
    return summary


def meta_checkpoint(path: Path, model: CHBJointModel, objective_head: LearnedAuxiliaryLoss | None, optimizer: torch.optim.Optimizer, fold: int, branch: str, seed: int, step: int, validation: dict[str, Any], classifier_hash: str, history: list[dict[str, Any]]) -> None:
    atomic_torch_save(path, {
        "release_id": RELEASE_ID, "stage": f"meta_{branch}", "fold": fold, "branch": branch,
        "seed": seed, "step": step, "model": model.state_dict(),
        "objective": objective_head.state_dict() if objective_head is not None else None,
        "optimizer": optimizer.state_dict(), "validation": validation,
        "classifier_hash": classifier_hash, "history": history,
        "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "saved_at": utc_now(),
    })


def run_meta(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = meta_run_dir(args.output_root, args.fold, args.branch, args.seed)
    completed_path = run_dir / "completed.json"
    if completed_path.is_file() and not args.force:
        return json.loads(completed_path.read_text())
    set_seed(args.seed + 40_000 + args.fold * 10 + (0 if args.branch == "band" else 1))
    device = torch.device(args.device)
    model, common_path, _ = load_common_model(args, args.fold, prepared_band=True, device=device)
    classifier_hash = stable_classifier_hash(model)
    train_rows = load_rows(args.fold, "train", args.windows, args.fold_root)
    val_rows = load_rows(args.fold, "validation", args.windows, args.fold_root)
    train_records = records_from_rows(train_rows)
    val_records = records_from_rows(val_rows)
    store = SignalStore(args.cache_root)
    cache = PrefixCache(
        args.prefix_cache,
        classifier_hash,
        args.prefix_cache_bytes,
        not args.no_prefix_cache,
        gpu_pin_bytes=args.validation_gpu_cache_bytes,
    )
    specs = fixed_episode_specs(val_records, args.validation_episodes, args.seed + 50_000 + args.fold)
    objective_head = None if args.branch == "band" else LearnedAuxiliaryLoss().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.branch == "band":
        for parameter in model.band_head.parameters():
            parameter.requires_grad_(True)
    if objective_head is not None:
        for parameter in objective_head.parameters():
            parameter.requires_grad_(True)
        objective_head.eval()
    outer_parameters = list(model.band_head.parameters()) if args.branch == "band" else list(objective_head.parameters())
    optimizer = torch.optim.AdamW(outer_parameters, lr=OUTER_LR, weight_decay=0.0)
    step = 0
    history: list[dict[str, Any]] = []
    best_future = float("inf")
    best_ns_delta = float("inf")
    best_future_step: int | None = None
    best_ns_step: int | None = None
    stale = 0
    resume_rng_state: dict[str, Any] | None = None
    resume_path = run_dir / "outer_state.pt" if (run_dir / "outer_state.pt").is_file() else run_dir / "last.pt"
    if resume_path.is_file() and not args.force:
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved.get("classifier_hash") != classifier_hash or saved.get("branch") != args.branch:
            raise RuntimeError("meta resume identity mismatch")
        if "outer_model" in saved:
            if args.branch == "band":
                model.band_head.load_state_dict(saved["outer_model"], strict=True)
            elif objective_head is not None:
                objective_head.load_state_dict(saved["outer_model"], strict=True)
        else:
            model.load_state_dict(saved["model"], strict=True)
            if objective_head is not None:
                objective_head.load_state_dict(saved["objective"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        step = int(saved.get("step", 0))
        history = list(saved.get("history", []))
        resume_rng_state = saved.get("rng_state")
        stale = int(saved.get("stale", 0))
        for row in history:
            if row.get("kind") == "validation":
                best_future = min(best_future, float(row["metrics"]["future_loss"]))
                best_ns_delta = min(best_ns_delta, float(row["metrics"]["nonseizure_delta"]))
        best_future_step = next((int(row["step"]) for row in history if row.get("kind") == "validation" and float(row["metrics"]["future_loss"]) == best_future), None)
        best_ns_step = next((int(row["step"]) for row in history if row.get("kind") == "validation" and float(row["metrics"]["nonseizure_delta"]) == best_ns_delta), None)
    model.eval()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **formal_config(), "status": "running", "stage": f"meta_{args.branch}", "fold": args.fold,
        "branch": args.branch, "seed": args.seed, "started_at": utc_now(),
        "common_classifier": str(common_path), "classifier_hash": classifier_hash,
        "train_rows": len(train_rows), "validation_rows": len(val_rows),
        "train_records_with_episodes": len(train_records), "validation_records_with_episodes": len(val_records),
        "validation_episode_count": len(specs), "prefix_cache": str(cache.root),
        "prefix_cache_max_bytes": args.prefix_cache_bytes, "device": str(device), "test_partition_read": False,
        "validation_gpu_cache_max_bytes": args.validation_gpu_cache_bytes,
    }
    atomic_json(run_dir / "manifest.json", manifest)
    rng = np.random.default_rng(args.seed + 60_000 + args.fold * 10 + (0 if args.branch == "band" else 1))
    if resume_rng_state is not None:
        rng.bit_generator.state = resume_rng_state
    started = time.monotonic()
    last_progress = started
    processed_chunks = 0
    last_validation: dict[str, Any] | None = None
    while step < args.meta_steps:
        optimizer.zero_grad(set_to_none=True)
        batch_metrics: list[dict[str, float | int]] = []
        for episode_index in range(args.effective_episodes):
            record, start = choose_episode(train_records, rng, seizure_bias=(episode_index % 4 == 0))
            result = run_episode(model, objective_head, store, record, start, device, args.branch, cache, differentiable=True)
            (result["outer_loss"] / args.effective_episodes).backward()
            batch_metrics.append(episode_metrics(result))
            processed_chunks += BURN_IN_CHUNKS + QUERY_CHUNKS
            atomic_json(run_dir / "chunk_progress.json", {
                "release_id": RELEASE_ID, "status": "running", "fold": args.fold, "branch": args.branch,
                "outer_step": step + 1, "episode": episode_index + 1, "episodes_per_step": args.effective_episodes,
                "record": record.recording, "patient": record.patient, "episode_start_row": start,
                "completed_chunks": processed_chunks, "elapsed_s": time.monotonic() - started,
            })
        torch.nn.utils.clip_grad_norm_(outer_parameters, 1.0)
        optimizer.step()
        step += 1
        atomic_torch_save(run_dir / "outer_state.pt", {
            "release_id": RELEASE_ID, "stage": f"meta_{args.branch}", "fold": args.fold, "branch": args.branch,
            "seed": args.seed, "step": step,
            "outer_model": model.band_head.state_dict() if args.branch == "band" else objective_head.state_dict(),
            "optimizer": optimizer.state_dict(), "classifier_hash": classifier_hash, "history": history,
            "rng_state": rng.bit_generator.state, "saved_at": utc_now(),
        })
        now = time.monotonic()
        if now - last_progress >= 30:
            atomic_json(run_dir / "progress.json", {
                **manifest, "status": "training", "step": step, "max_steps": args.meta_steps,
                "processed_chunks": processed_chunks, "chunks_per_s": processed_chunks / max(now - started, 1e-9),
                "last_batch": {key: float(np.mean([row[key] for row in batch_metrics])) for key in batch_metrics[0] if key != "missing_class_terms"},
                "missing_class_terms": int(sum(int(row["missing_class_terms"]) for row in batch_metrics)),
                "elapsed_s": now - started,
                "gpu_peak_mib": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0,
            })
            last_progress = now
        if step % args.eval_interval == 0 or step == args.meta_steps:
            validation_context = cache.pin_validation_on_gpu() if cache is not None else nullcontext()
            with validation_context:
                validation = evaluate_episode_specs(model, objective_head, store, val_records, specs, device, args.branch, cache)
            validation["gpu_cache_entries"] = len(cache.gpu_pinned)
            validation["gpu_cache_bytes"] = cache.gpu_pinned_bytes
            validation["gpu_cache_hits"] = cache.gpu_hits
            validation["disk_cache_hits"] = cache.disk_hits
            last_validation = validation
            row = {"kind": "validation", "step": step, "metrics": validation, "elapsed_s": now - started}
            history.append(row)
            future_improved = float(validation["future_loss"]) < best_future - 1e-7
            ns_improved = float(validation["nonseizure_delta"]) < best_ns_delta - 1e-7
            if future_improved:
                best_future = float(validation["future_loss"])
                best_future_step = step
                meta_checkpoint(run_dir / "best_future.pt", model, objective_head, optimizer, args.fold, args.branch, args.seed, step, validation, classifier_hash, history)
            if ns_improved:
                best_ns_delta = float(validation["nonseizure_delta"])
                best_ns_step = step
                meta_checkpoint(run_dir / "best_nonseizure.pt", model, objective_head, optimizer, args.fold, args.branch, args.seed, step, validation, classifier_hash, history)
            stale = 0 if (future_improved or ns_improved) else stale + 1
            atomic_json(run_dir / "validation_history.json", {"evaluations": [item for item in history if item.get("kind") == "validation"]})
            atomic_torch_save(run_dir / "last.pt", {
                "release_id": RELEASE_ID, "stage": f"meta_{args.branch}", "fold": args.fold, "branch": args.branch,
                "seed": args.seed, "step": step, "model": model.state_dict(),
                "objective": objective_head.state_dict() if objective_head is not None else None,
                "optimizer": optimizer.state_dict(), "validation": validation, "classifier_hash": classifier_hash, "history": history,
            })
            atomic_torch_save(run_dir / "outer_state.pt", {
                "release_id": RELEASE_ID, "stage": f"meta_{args.branch}", "fold": args.fold, "branch": args.branch,
                "seed": args.seed, "step": step,
                "outer_model": model.band_head.state_dict() if args.branch == "band" else objective_head.state_dict(),
                "optimizer": optimizer.state_dict(), "classifier_hash": classifier_hash, "history": history,
                "stale": stale, "rng_state": rng.bit_generator.state, "saved_at": utc_now(),
            })
            print(json.dumps(row, allow_nan=True), flush=True)
            if step >= args.min_meta_steps and stale >= args.meta_patience:
                break
    if not (run_dir / "best_future.pt").is_file() and last_validation is not None:
        meta_checkpoint(run_dir / "best_future.pt", model, objective_head, optimizer, args.fold, args.branch, args.seed, step, last_validation, classifier_hash, history)
    cache.flush()
    after_hash = stable_classifier_hash(model)
    completed = {
        **manifest, "status": "completed" if after_hash == classifier_hash else "failed_classifier_mutation_check",
        "completed_at": utc_now(), "steps_completed": step, "best_future_step": best_future_step,
        "best_future_loss": best_future if np.isfinite(best_future) else None,
        "best_nonseizure_step": best_ns_step, "best_nonseizure_delta": best_ns_delta if np.isfinite(best_ns_delta) else None,
        "stale_evaluations": stale, "classifier_hash_after": after_hash, "classifier_unchanged": after_hash == classifier_hash,
        "best_future_checkpoint": str(run_dir / "best_future.pt") if (run_dir / "best_future.pt").is_file() else None,
        "best_nonseizure_checkpoint": str(run_dir / "best_nonseizure.pt") if (run_dir / "best_nonseizure.pt").is_file() else None,
        "elapsed_s": time.monotonic() - started, "test_partition_read": False,
    }
    atomic_json(completed_path, completed)
    atomic_json(run_dir / "progress.json", completed)
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("supervised", "band_prepare", "meta"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--branch", choices=("band", "learned"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-root", type=Path, default=SCRIPT_ROOT / "results" / RELEASE_ID)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--prefix-cache", type=Path, default=Path("/mnt/d/EEGData/meta_ttt_prefix_v2"))
    parser.add_argument("--prefix-cache-bytes", type=int, default=128 * 2**30)
    parser.add_argument("--validation-gpu-cache-bytes", type=int, default=8 * 2**30)
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
