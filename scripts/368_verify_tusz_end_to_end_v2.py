"""Small real-checkpoint acceptance probe for the formal seed-3407 evaluator."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from bfa.tusz_meta_ttt.dataset import load_cached_arrays
from bfa.tusz_meta_ttt_v2.functional import SplitFunctionalTUSZModel
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(module):
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def compare_states(before, module):
    return max(float((before[name] - tensor.detach().cpu()).abs().max()) for name, tensor in module.state_dict().items())


def main():
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("tusz_eval_v2", ROOT / "scripts/323_evaluate_tusz_meta_ttt_v2.py")
    evaluator = module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    paths = sorted((V1 / "cache/train").rglob("*.npz"))
    path = next(path for path in paths if len(load_cached_arrays(path)["decision_end_s"]) >= 27)
    archive = load_cached_arrays(path)
    times = archive["decision_end_s"][:27].copy()
    signal = archive["signal"][:, : 26 * 400 + 2000].copy()
    sample = {"signal": signal, "decision_end_s": times}
    results = {}
    for mode in ["future", "current"]:
        checkpoint = OUT / "runs/meta/formal_fast_seed3407/mask" / mode / "eds/epoch_02.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source_hash_before = digest(state["source"])
        model = evaluator.build_model(state["model"])
        functional = SplitFunctionalTUSZModel(model)
        objective = build_objective(state["objective_name"], float(state["difficulty"])).cuda().eval()
        objective.load_state_dict(state["objective"])
        objective.requires_grad_(False)
        head_before = state_digest(model.detector)
        ssl_before = state_digest(objective)
        config = replace(InnerStepConfig(**state["inner_config"]), relative_step=0.0)
        frozen, zero, _, _ = evaluator.infer(functional, objective, sample, mode, config, 3407, "normalized", None)
        zero_error = float(np.max(np.abs(frozen - zero)))
        assert zero_error == 0.0, (mode, zero_error)
        assert compare_states(head_before, model.detector) == 0.0
        assert compare_states(ssl_before, objective) == 0.0
        assert digest(state["source"]) == source_hash_before
        item = dict(zero_update_max_abs=zero_error, detector_change_max_abs=0.0, ssl_head_change_max_abs=0.0, source_file_unchanged=True)
        if mode == "future":
            changed = {"signal": signal.copy(), "decision_end_s": times}
            changed["signal"][:, 6000:] += np.float32(7.0)
            _, original_scores, _, _ = evaluator.infer(functional, objective, sample, mode, InnerStepConfig(**state["inner_config"]), 3407, "normalized", None)
            _, changed_scores, _, _ = evaluator.infer(functional, objective, changed, mode, InnerStepConfig(**state["inner_config"]), 3407, "normalized", None)
            prefix_error = float(np.max(np.abs(original_scores[:11] - changed_scores[:11])))
            assert prefix_error == 0.0, prefix_error
            item["future_chunk_change_on_first_chunk_max_abs"] = prefix_error
        results[mode] = item
    destination = OUT / "audits/end_to_end_seed3407_v2.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"status": "passed", "record": str(path), "windows": len(times), "results": results}, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
