#!/usr/bin/env python3
"""Package S1 plus a warm-started SSL head for the shared F/C evaluator."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--objective-checkpoint", type=Path, required=True)
    parser.add_argument("--gradient-calibration", type=Path, required=True)
    parser.add_argument("--mode", choices=["future", "current"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = torch.load(args.source.resolve(), map_location="cpu", weights_only=False)
    objective = torch.load(
        args.objective_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    calibration = json.loads(args.gradient_calibration.read_text())
    relative_step = calibration.get("selected_relative_step")
    if relative_step is None:
        raise ValueError("inner update health check did not select a relative step")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": source["model"],
            "objective": objective["objective"],
            "objective_name": objective["objective_name"],
            "difficulty": objective["difficulty"],
            "mode": args.mode,
            "outer_scope": "none_nonmeta_control",
            "inner_rule": "normalized",
            "inner_kernel": "packed",
            "prefix_precision": "fp32",
            "prefix_cuda_graph": True,
            "inner_config": {
                "relative_step": relative_step,
                "block_reference_norms": {
                    int(key): value for key, value in calibration["block_reference_norms"].items()
                },
                "block_gradient_medians": {
                    int(key): value for key, value in calibration["block_gradient_medians"].items()
                },
            },
            "source": str(args.source.resolve()),
            "objective_checkpoint": str(args.objective_checkpoint.resolve()),
            "seed": int(objective["seed"]),
            "created_utc": datetime.now(UTC).isoformat(),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
