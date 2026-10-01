#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.tusz_meta_ttt.dataset import load_cached_arrays
from bfa.tusz_meta_ttt.model import TUSZDetector
from bfa.tusz_meta_ttt_v2.functional import SplitFunctionalTUSZModel
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.protocol import chunk_rows, stable_transform_seed
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig, normalized_inner_step

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"
PRETRAINED = ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"


def load_source(path):
    adapter = CBraModAdapter(PRETRAINED, train_backbone=True)
    adapter.backbone.proj_out = torch.nn.Identity()
    model = TUSZDetector(adapter.backbone)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return model.requires_grad_(False).cuda().eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--objective-checkpoint", type=Path, required=True)
    parser.add_argument("--gradient-calibration", type=Path, required=True)
    parser.add_argument("--objective", choices=["band", "temporal", "mask"], required=True)
    parser.add_argument("--difficulty", type=float, required=True)
    parser.add_argument("--relative-step", type=float, required=True)
    parser.add_argument("--chunks", type=int, default=12)
    args = parser.parse_args()
    model = load_source(args.source.resolve())
    functional = SplitFunctionalTUSZModel(model)
    objective = build_objective(args.objective, args.difficulty).cuda().eval()
    state = torch.load(args.objective_checkpoint, map_location="cpu", weights_only=False)
    objective.load_state_dict(state["objective"])
    calibration = json.loads(args.gradient_calibration.read_text())
    config = InnerStepConfig(
        args.relative_step,
        {int(key): value for key, value in calibration["block_reference_norms"].items()},
        {int(key): value for key, value in calibration["block_gradient_medians"].items()},
    )
    paths = sorted((V1 / "cache/train").rglob("*.npz"))
    selected = []
    for path in paths:
        archive = load_cached_arrays(path)
        chunks = chunk_rows(archive["decision_end_s"])
        if chunks:
            selected.append((path, chunks[len(chunks) // 2]))
        if len(selected) >= args.chunks:
            break
    timings = []
    torch.cuda.reset_peak_memory_stats()
    for index, (path, chunk) in enumerate(selected):
        archive = load_cached_arrays(path)
        started_io = time.perf_counter()
        values = np.stack([archive["signal"][:, row * 400 : row * 400 + 2000] for row in chunk.rows])
        support = torch.from_numpy(values.reshape(-1, 16, 10, 200)).cuda()
        torch.cuda.synchronize()
        after_io = time.perf_counter()
        fast = functional.initial_fast_parameters()
        prepared = objective.prepare_loss(
            support, prefix_fn=functional.prefix,
            transform_seed=[
                stable_transform_seed(3407, path.as_posix(), args.objective, row)
                for row in chunk.rows
            ],
        )

        def loss_fn(parameters, prepared=prepared):
            return prepared(
                lambda prefix_features: functional.features_from_prefix(
                    prefix_features, dict(parameters)
                )
            )

        started_update = time.perf_counter()
        result = normalized_inner_step(
            fast, loss_fn=loss_fn, record_initial=fast, config=config, create_graph=True
        )
        functional.logits(support, result.parameters).mean().backward()
        torch.cuda.synchronize()
        finished = time.perf_counter()
        timings.append({
            "windows": len(chunk.rows), "io_transfer_s": after_io - started_io,
            "inner_and_meta_backward_s": finished - started_update,
            "accepted": result.accepted,
        })
    result = {
        "objective": args.objective, "difficulty": args.difficulty,
        "chunks": len(timings), "windows": sum(row["windows"] for row in timings),
        "elapsed_io_transfer_s": sum(row["io_transfer_s"] for row in timings),
        "elapsed_inner_and_meta_backward_s": sum(row["inner_and_meta_backward_s"] for row in timings),
        "updates_per_second": len(timings) / sum(row["inner_and_meta_backward_s"] for row in timings),
        "windows_per_second": sum(row["windows"] for row in timings) / sum(row["inner_and_meta_backward_s"] for row in timings),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "accepted_fraction": float(np.mean([row["accepted"] for row in timings])),
        "rows": timings,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
