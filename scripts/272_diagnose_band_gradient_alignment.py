"""Diagnose Band-SSL/classification gradient alignment before retraining.

The diagnostic is deliberately independent of the Band-TTT matrix.  It uses
the same deterministic five-class band view as evaluation, compares the
original supervised and Meta-Band checkpoints, decomposes the Band loss by
band class, reports the final two Transformer blocks separately, and checks a
single tiny raw-gradient update against the first-order prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.func import grad, vmap
from torch.nn import functional as F


ROOT = Path("/root/b_false_alarm_atlas")
CODE_ROOT = Path("/mnt/c/Users/User/Documents/Codex/2026-08-03/du-q/work/NeuroTTT/CBraMod")
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(ROOT / "src"))

from chbmit_groupkfold.data import DEFAULT_CACHE, DEFAULT_FOLDS, DEFAULT_WINDOWS, WindowDataset, load_rows  # noqa: E402
from chbmit_groupkfold.meta_model import CHBMetaTTTModel  # noqa: E402
from chbmit_groupkfold.transforms import BANDS, deterministic_band_view  # noqa: E402


META_ROOT = ROOT / "outputs/reports/meta-ttt-chbmit-5fold-v1"
SUP_ROOT = META_ROOT / "runs"
PRETRAINED = CODE_ROOT / "pretrained_weights/pretrained_weights.pth"
OUT_DEFAULT = ROOT / "outputs/reports/band-gradient-alignment-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    temporary.replace(path)


def module_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_tail_class():
    path = ROOT / "scripts/265_evaluate_band_ttt_v2.py"
    spec = importlib.util.spec_from_file_location("band_ttt_v2_for_alignment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BandTail, module.frozen_prefix_pair


def load_pair(fold: int, device: torch.device):
    meta_path = META_ROOT / "runs" / f"meta_band_fold{fold}_seed3407" / "best.pt"
    supervised_path = SUP_ROOT / f"detection_only_fold{fold}_seed3407" / "best.pt"
    if not meta_path.is_file() or not supervised_path.is_file():
        raise FileNotFoundError(fold, meta_path, supervised_path)
    meta_state = torch.load(meta_path, map_location="cpu", weights_only=False)
    supervised_state = torch.load(supervised_path, map_location="cpu", weights_only=False)
    meta = CHBMetaTTTModel(PRETRAINED)
    meta.load_state_dict(meta_state["model"], strict=True)
    supervised = CHBMetaTTTModel(PRETRAINED)
    missing, unexpected = supervised.load_state_dict(supervised_state["model"], strict=False)
    allowed_missing = {name for name in meta.state_dict() if name.startswith("temporal_head.")}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(f"unexpected supervised checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    # The detection-only checkpoint has the shared Band head but no temporal
    # head.  Use the same Band head in both models so this comparison isolates
    # the backbone/checkpoint origin rather than an auxiliary-head mismatch.
    supervised.band_head.load_state_dict(meta.band_head.state_dict(), strict=True)
    meta.to(device).eval()
    supervised.to(device).eval()
    return supervised, meta, supervised_path, meta_path


def sample_validation_rows(fold: int, per_class: int, seed: int) -> pd.DataFrame:
    rows = load_rows(fold, "validation", DEFAULT_WINDOWS, DEFAULT_FOLDS)
    rng = np.random.default_rng(seed + fold * 1009)
    pieces = []
    for label in (0.0, 1.0):
        group = rows[rows.label == label]
        if len(group) < per_class:
            raise RuntimeError(f"fold={fold} label={label} has only {len(group)} rows")
        selected = np.sort(rng.choice(len(group), size=per_class, replace=False))
        pieces.append(group.iloc[selected])
    result = pd.concat(pieces, ignore_index=True)
    result = result.sort_values(["patient", "recording", "start"], kind="stable").reset_index(drop=True)
    return result


def group_names(parameters: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    names = list(parameters)
    return {
        "adaptive_all": names,
        "shared_last2": [name for name in names if name.startswith("layer10.") or name.startswith("layer11.")],
        "layer10": [name for name in names if name.startswith("layer10.")],
        "layer11": [name for name in names if name.startswith("layer11.")],
        "band_head": [name for name in names if name.startswith("band_head.")],
    }


def flatten(mapping: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    if not names:
        return torch.empty((len(next(iter(mapping.values()))), 0), device=next(iter(mapping.values())).device)
    return torch.cat([mapping[name].reshape(mapping[name].shape[0], -1) for name in names], dim=1)


def cosine_and_dot(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dot = (first * second).sum(dim=1)
    first_norm = first.norm(dim=1)
    second_norm = second.norm(dim=1)
    denominator = first_norm * second_norm
    cosine = torch.where(denominator > 0, dot / denominator, torch.full_like(dot, float("nan")))
    return cosine, dot, first_norm, second_norm


def mean_vector_cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(first, second) / denominator)


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def run_model(
    model: CHBMetaTTTModel,
    source: str,
    fold: int,
    rows: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    cache_root: Path,
    tail_class,
    frozen_prefix_pair,
    tiny_alpha: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    model.requires_grad_(False)
    for parameter in model.backbone.encoder.layers[-2].parameters():
        parameter.requires_grad_(True)
    for parameter in model.backbone.encoder.layers[-1].parameters():
        parameter.requires_grad_(True)
    model.band_head.requires_grad_(True)
    tail = tail_class(model).to(device).eval()
    adaptive = tail.adaptive_state()
    groups = group_names(adaptive)
    # torch.func vmap has no batching rule for the native MHA fastpath in the
    # pinned build.  The immutable prefix helper temporarily enables it, while
    # all differentiable tail calls below use ordinary batched kernels.
    torch.backends.mha.set_fastpath_enabled(False)
    dataset = WindowDataset(rows, cache_root=cache_root, max_open=64)
    sample_records: list[dict[str, Any]] = []
    aggregate: dict[tuple[str, str, str], dict[str, Any]] = {}
    global_cls_total: dict[str, np.ndarray] = {}
    source_before = module_hash(model)
    for begin in range(0, len(rows), batch_size):
        indices = list(range(begin, min(begin + batch_size, len(rows))))
        samples = [dataset[index] for index in indices]
        signal = torch.stack([item[0] for item in samples]).to(device=device, dtype=torch.float32)
        target = torch.tensor([float(item[1]) for item in samples], device=device, dtype=torch.float32)
        sample_ids = [str(item[2]) for item in samples]
        transformed, band_target = deterministic_band_view(signal, sample_ids)
        transformed_prefix, signal_prefix = frozen_prefix_pair(model, transformed, signal)

        def ssl_loss(current: dict[str, torch.Tensor], sample: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
            logits = torch.func.functional_call(tail, current, (sample.unsqueeze(0),), {"mode": "band"}, strict=False)
            return F.cross_entropy(logits, label.unsqueeze(0))

        def cls_loss(current: dict[str, torch.Tensor], sample: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
            logits = torch.func.functional_call(tail, current, (sample.unsqueeze(0),), {"mode": "detect"}, strict=False)
            return F.binary_cross_entropy_with_logits(logits, label.unsqueeze(0))

        ssl_grad = vmap(grad(ssl_loss), in_dims=(None, 0, 0), randomness="same")(adaptive, transformed_prefix, band_target)
        cls_grad = vmap(grad(cls_loss), in_dims=(None, 0, 0), randomness="same")(adaptive, signal_prefix, target)
        ssl_flat = {name: value for name, value in ssl_grad.items()}
        cls_flat = {name: value for name, value in cls_grad.items()}
        metrics: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for group, names in groups.items():
            ssl_values = flatten(ssl_flat, names)
            cls_values = flatten(cls_flat, names)
            metrics[group] = cosine_and_dot(ssl_values, cls_values)

        updated = {name: parameter.unsqueeze(0) - tiny_alpha * ssl_grad[name] for name, parameter in adaptive.items()}
        actual = vmap(cls_loss, in_dims=(0, 0, 0), randomness="same")(updated, signal_prefix, target)
        base = vmap(cls_loss, in_dims=(None, 0, 0), randomness="same")(adaptive, signal_prefix, target)
        primary_cos, primary_dot, primary_ssl_norm, primary_cls_norm = metrics["shared_last2"]
        predicted = -tiny_alpha * primary_dot

        metric_arrays = {group: tuple(value.detach().cpu().numpy() for value in values) for group, values in metrics.items()}
        labels_np = target.detach().cpu().numpy().astype(int)
        band_np = band_target.detach().cpu().numpy().astype(int)
        actual_np = actual.detach().cpu().numpy()
        base_np = base.detach().cpu().numpy()
        predicted_np = predicted.detach().cpu().numpy()
        for local, row_index in enumerate(indices):
            record = {
                "source": source,
                "fold": fold,
                "row_index": row_index,
                "patient": str(rows.iloc[row_index].patient),
                "recording": str(rows.iloc[row_index].recording),
                "start": float(rows.iloc[row_index].start),
                "sample_id": sample_ids[local],
                "seizure_label": labels_np[local],
                "band_component": band_np[local],
                "actual_cls_loss_before": float(base_np[local]),
                "actual_cls_loss_after": float(actual_np[local]),
                "actual_delta_cls": float(actual_np[local] - base_np[local]),
                "predicted_delta_cls": float(predicted_np[local]),
                "primary_dot": float(primary_dot[local].detach().cpu()),
                "primary_ssl_norm": float(primary_ssl_norm[local].detach().cpu()),
                "primary_cls_norm": float(primary_cls_norm[local].detach().cpu()),
            }
            for group, values in metric_arrays.items():
                record[f"{group}_cosine"] = float(values[0][local])
                record[f"{group}_dot"] = float(values[1][local])
                record[f"{group}_ssl_norm"] = float(values[2][local])
                record[f"{group}_cls_norm"] = float(values[3][local])
            sample_records.append(record)
        # Keep enough vector information to calculate cosine of mean gradients
        # for seizure class and each Band component without retaining per-row
        # parameter tensors.
        group_matrices = {
            group: (flatten(ssl_flat, names), flatten(cls_flat, names))
            for group, names in groups.items()
        }
        for group, (_, cls_matrix) in group_matrices.items():
            global_cls_vec = cls_matrix.sum(dim=0).detach().cpu().numpy()
            global_cls_total[group] = global_cls_vec if group not in global_cls_total else global_cls_total[group] + global_cls_vec
        for subset_name, subset_values in (
            ("all", np.ones(len(rows), dtype=bool)[begin:begin + len(indices)]),
            ("seizure", labels_np == 1),
            ("nonseizure", labels_np == 0),
        ):
            if not subset_values.any():
                continue
            for group, (ssl_matrix, cls_matrix) in group_matrices.items():
                key = ("seizure_class", subset_name, group)
                item = aggregate.setdefault(key, {"n": 0, "ssl": None, "cls": None})
                ssl_vec = ssl_matrix[subset_values].sum(dim=0).detach().cpu().numpy()
                cls_vec = cls_matrix[subset_values].sum(dim=0).detach().cpu().numpy()
                item["n"] += int(subset_values.sum())
                item["ssl"] = ssl_vec if item["ssl"] is None else item["ssl"] + ssl_vec
                item["cls"] = cls_vec if item["cls"] is None else item["cls"] + cls_vec
        for component in range(len(BANDS)):
            subset_values = band_np == component
            if not subset_values.any():
                continue
            for group, (ssl_matrix, cls_matrix) in group_matrices.items():
                ssl_vec = ssl_matrix[subset_values].sum(dim=0).detach().cpu().numpy()
                local_cls_vec = cls_matrix[subset_values].sum(dim=0).detach().cpu().numpy()
                for kind, cls_vec in (("band_component_global", global_cls_total[group]), ("band_component_local", local_cls_vec)):
                    key = (kind, str(component), group)
                    item = aggregate.setdefault(key, {"n": 0, "ssl": None, "cls": None})
                    item["n"] += int(subset_values.sum())
                    item["ssl"] = ssl_vec if item["ssl"] is None else item["ssl"] + ssl_vec
                    if kind == "band_component_local":
                        item["cls"] = cls_vec if item["cls"] is None else item["cls"] + cls_vec
        for class_name, class_values in (("seizure", labels_np == 1), ("nonseizure", labels_np == 0)):
            if not class_values.any():
                continue
            for component in range(len(BANDS)):
                subset_values = class_values & (band_np == component)
                if not subset_values.any():
                    continue
                for group, (ssl_matrix, cls_matrix) in group_matrices.items():
                    key = ("band_component_class", f"{class_name}:{component}", group)
                    item = aggregate.setdefault(key, {"n": 0, "ssl": None, "cls": None})
                    ssl_vec = ssl_matrix[subset_values].sum(dim=0).detach().cpu().numpy()
                    cls_vec = cls_matrix[class_values].sum(dim=0).detach().cpu().numpy()
                    item["n"] += int(subset_values.sum())
                    item["ssl"] = ssl_vec if item["ssl"] is None else item["ssl"] + ssl_vec
                    item["cls"] = cls_vec if item["cls"] is None else item["cls"] + cls_vec
    if not global_cls_total:
        raise RuntimeError(f"no classification gradients accumulated: {source}")
    for (kind, _, group), item in aggregate.items():
        if kind == "band_component_global":
            item["cls"] = global_cls_total[group]
    if module_hash(model) != source_before:
        raise RuntimeError(f"diagnostic mutated model parameters: {source}")

    for (kind, subset, group), item in aggregate.items():
        item["source"] = source
        item["fold"] = fold
        item["subset_type"] = kind
        item["subset"] = subset
        item["gradient_group"] = group
        item["mean_gradient_cosine"] = mean_vector_cosine(item["ssl"], item["cls"])
        del item["ssl"], item["cls"]
    return sample_records, [{k: v for k, v in item.items()} for item in aggregate.values()], {"source": source, "fold": fold, "rows": len(rows), "parameter_groups": {key: len(value) for key, value in groups.items()}}


def summarize_sample(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in sorted(records.source.unique()):
        subset = records[records.source == source]
        for group in ("adaptive_all", "shared_last2", "layer10", "layer11"):
            cosine = subset[f"{group}_cosine"].to_numpy(dtype=float)
            f = np.isfinite(cosine)
            rows.append({
                "source": source,
                "subset_type": "all",
                "subset": "all",
                "gradient_group": group,
                "n": int(np.isfinite(cosine).sum()),
                "mean_sample_cosine": float(np.nanmean(cosine)),
                "negative_fraction": float(np.mean(cosine[f] < 0)) if f.any() else float("nan"),
                "median_sample_cosine": float(np.nanmedian(cosine)),
            })
        for class_value, class_name in ((1, "seizure"), (0, "nonseizure")):
            class_rows = subset[subset.seizure_label == class_value]
            for group in ("adaptive_all", "shared_last2", "layer10", "layer11"):
                cosine = class_rows[f"{group}_cosine"].to_numpy(dtype=float)
                f = np.isfinite(cosine)
                rows.append({
                    "source": source,
                    "subset_type": "seizure_class",
                    "subset": class_name,
                    "gradient_group": group,
                    "n": int(np.isfinite(cosine).sum()),
                    "mean_sample_cosine": float(np.nanmean(cosine)),
                    "negative_fraction": float(np.mean(cosine[f] < 0)) if f.any() else float("nan"),
                    "median_sample_cosine": float(np.nanmedian(cosine)),
                })
        for component in range(len(BANDS)):
            component_rows = subset[subset.band_component == component]
            for group in ("adaptive_all", "shared_last2", "layer10", "layer11"):
                cosine = component_rows[f"{group}_cosine"].to_numpy(dtype=float)
                f = np.isfinite(cosine)
                rows.append({
                    "source": source,
                    "subset_type": "band_component",
                    "subset": f"{BANDS[component][0]}-{BANDS[component][1]}Hz",
                    "gradient_group": group,
                    "n": int(np.isfinite(cosine).sum()),
                    "mean_sample_cosine": float(np.nanmean(cosine)),
                    "negative_fraction": float(np.mean(cosine[f] < 0)) if f.any() else float("nan"),
                    "median_sample_cosine": float(np.nanmedian(cosine)),
                })
    return pd.DataFrame(rows)


def summarize_causal(causal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in sorted(causal.source.unique()):
        subset = causal[causal.source == source]
        for name, group in [("all", subset), ("seizure", subset[subset.seizure_label == 1]), ("nonseizure", subset[subset.seizure_label == 0])]:
            predicted = group.predicted_delta_cls.to_numpy(dtype=float)
            actual = group.actual_delta_cls.to_numpy(dtype=float)
            finite = np.isfinite(predicted) & np.isfinite(actual)
            predicted, actual = predicted[finite], actual[finite]
            rows.append({
                "source": source,
                "subset": name,
                "n": int(len(actual)),
                "tiny_alpha": float(group.tiny_alpha.iloc[0]) if len(group) else float("nan"),
                "mean_predicted_delta": float(np.mean(predicted)) if len(actual) else float("nan"),
                "mean_actual_delta": float(np.mean(actual)) if len(actual) else float("nan"),
                "median_predicted_delta": float(np.median(predicted)) if len(actual) else float("nan"),
                "median_actual_delta": float(np.median(actual)) if len(actual) else float("nan"),
                "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(actual))) if len(actual) else float("nan"),
                "actual_negative_fraction": float(np.mean(actual < 0)) if len(actual) else float("nan"),
                "predicted_negative_fraction": float(np.mean(predicted < 0)) if len(actual) else float("nan"),
                "pearson_predicted_actual": correlation(predicted, actual),
                "mean_abs_error": float(np.mean(np.abs(predicted - actual))) if len(actual) else float("nan"),
            })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--per-class", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tiny-alpha", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=OUT_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tiny_alpha <= 0 or args.tiny_alpha > 1e-4:
        raise ValueError("tiny-alpha must be in (0, 1e-4]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_float32_matmul_precision("high")
    tail_class, frozen_prefix_pair = load_tail_class()
    all_samples: list[dict[str, Any]] = []
    all_mean_vectors: list[dict[str, Any]] = []
    run_meta = {
        "release_id": "band-gradient-alignment-v1",
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "folds": args.folds,
        "per_class": args.per_class,
        "total_rows": len(args.folds) * args.per_class * 2,
        "batch_size": args.batch_size,
        "tiny_alpha": args.tiny_alpha,
        "objective": "deterministic Band view; five component losses are per-band CE subsets",
        "component_alignment": "band_component_global compares each component gradient with the full-set classification gradient; band_component_local is the same-component sensitivity analysis; band_component_class compares each component gradient within seizure/nonseizure samples with the classification gradient for that class",
        "adaptation_parameter_scope": "last two Transformer blocks plus Band head; primary alignment excludes Band head because seizure loss has no Band-head path",
        "labels_used_for": "diagnostic g_cls and subgroup reporting only; never adaptation",
        "pretrained_sha256": sha256(PRETRAINED),
    }
    started = time.monotonic()
    for fold in args.folds:
        rows = sample_validation_rows(fold, args.per_class, args.seed)
        supervised, meta, supervised_path, meta_path = load_pair(fold, device)
        for source, model in (("supervised", supervised), ("meta_band", meta)):
            samples, mean_vectors, details = run_model(
                model, source, fold, rows, device, args.batch_size, args.cache_root,
                tail_class, frozen_prefix_pair, args.tiny_alpha,
            )
            all_samples.extend(samples)
            all_mean_vectors.extend(mean_vectors)
            run_meta.setdefault("runs", []).append({**details, "supervised_checkpoint": str(supervised_path), "meta_checkpoint": str(meta_path)})
        print(f"completed fold {fold}; elapsed_s={time.monotonic() - started:.1f}", flush=True)

    # Reconstruct causal rows directly from sample-level columns.  This keeps
    # the output topology simple and identical to the per-sample gradient table.
    samples_frame = pd.DataFrame(all_samples)
    causal_frame = samples_frame[["source", "fold", "row_index", "seizure_label", "band_component", "actual_delta_cls", "predicted_delta_cls"]].copy()
    causal_frame["tiny_alpha"] = args.tiny_alpha
    sample_summary = summarize_sample(samples_frame)
    causal_summary = summarize_causal(causal_frame)
    mean_frame = pd.DataFrame(all_mean_vectors)
    run_meta["elapsed_s"] = time.monotonic() - started
    run_meta["sample_rows"] = int(len(samples_frame))
    run_meta["source_counts"] = {str(key): int(value) for key, value in samples_frame.source.value_counts().to_dict().items()}
    run_meta["output_files"] = ["sample_metrics.parquet", "sample_alignment_summary.csv", "mean_gradient_alignment.csv", "causal_one_step_summary.csv", "run_manifest.json"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    samples_frame.to_parquet(args.output_root / "sample_metrics.parquet", index=False)
    sample_summary.to_csv(args.output_root / "sample_alignment_summary.csv", index=False)
    mean_frame.to_csv(args.output_root / "mean_gradient_alignment.csv", index=False)
    causal_summary.to_csv(args.output_root / "causal_one_step_summary.csv", index=False)
    atomic_json(args.output_root / "run_manifest.json", run_meta)
    print(json.dumps({"status": "complete", "output_root": str(args.output_root), "elapsed_s": run_meta["elapsed_s"], "rows": len(samples_frame)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
