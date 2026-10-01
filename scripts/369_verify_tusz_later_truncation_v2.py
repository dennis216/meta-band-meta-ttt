"""Check a later truncated Meta segment on the real CBraMod tail."""
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from bfa.tusz_meta_ttt.dataset import load_cached_arrays
from bfa.tusz_meta_ttt_v2.batched import ConditionEnsemble, EnsembleMaskObjective, enable_second_order_batched_attention
from bfa.tusz_meta_ttt_v2.joint_schedule import process_joint_group
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.runtime import load_source
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"


def tail_norm(model):
    values = [p.grad.detach().float().square().sum() for name, p in model.named_parameters()
              if name.startswith(("backbone.encoder.layers.10.", "backbone.encoder.layers.11.")) and p.grad is not None]
    return float(torch.sqrt(sum(values)).detach()) if values else 0.0


def main():
    enable_second_order_batched_attention()
    torch.set_float32_matmul_precision("highest")
    source = V1 / "runs/supervised/formal/s1_seed3407_check0.25/best.pt"
    head = OUT / "runs/ssl/formal/mask_5_seed3407/epoch_02.pt"
    calibration = json.loads((OUT / "calibration/formal/mask_5_seed3407/gradient_calibration.json").read_text())
    config = InnerStepConfig(calibration["selected_relative_step"],
                             {int(k): v for k, v in calibration["block_reference_norms"].items()},
                             {int(k): v for k, v in calibration["block_gradient_medians"].items()})
    model = load_source(source)
    ssl = build_objective("mask", 5).cuda().eval()
    ssl.load_state_dict(torch.load(head, map_location="cpu", weights_only=False)["objective"])
    ssl.requires_grad_(True)
    functional = ConditionEnsemble([model])
    objective = EnsembleMaskObjective([ssl], deduplicate_views=True)
    original = next(p for p in sorted((V1 / "cache/train").rglob("*.npz"))
                    if len(load_cached_arrays(p)["decision_end_s"]) >= 131)
    archive = load_cached_arrays(original)
    rows = 131
    times = archive["decision_end_s"][:rows].copy()
    signal = archive["signal"][:, :(rows - 1) * 400 + 2000].copy()
    labels = archive["labels"][:rows].copy()
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "patient/session/montage/record.npz"
        path.parent.mkdir(parents=True)
        np.savez(path, signal=signal, decision_end_s=times, labels=labels)
        weights = {path: np.full(rows, 1.0 / rows, dtype=np.float32)}
        iterator = process_joint_group(functional, objective, [[path]], weights,
                                       config, ["current"], seed=3407, compact=True)
        segments = []
        for loss, updates, _ in iterator:
            loss.backward()
            gradient = tail_norm(model)
            segments.append({"updates": updates, "encoder_initialization_gradient_norm": gradient})
            assert np.isfinite(gradient) and gradient > 0.0
            model.zero_grad(set_to_none=True)
            objective.zero_grad(set_to_none=True)
        assert len(segments) >= 3 and segments[0]["updates"] == 4
        assert all(item["updates"] >= 1 for item in segments[1:])
    result = {"status": "passed", "source_record": str(original), "windows": rows,
              "segments": segments,
              "elapsed_s": time.monotonic() - started}
    destination = OUT / "audits/later_truncation_seed3407_v2.json"
    destination.write_text(json.dumps(result, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
