"""Verify every formal seed-3407 Eval threshold equals its saved Dev choice."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"


def main():
    rows = []
    for mode in ["future", "current"]:
        for path in sorted((OUT / "evaluation" / mode).glob("*/eval/summary.json")):
            if "formal_fast_" not in str(path):
                continue
            dev_path = path.parent.parent / "dev/summary.json"
            if not dev_path.is_file():
                continue
            evaluation = json.loads(path.read_text())
            dev = json.loads(dev_path.read_text())
            assert evaluation["checkpoint"] == dev["checkpoint"], path
            assert evaluation["mode"] == dev["mode"], path
            for name, condition in evaluation["conditions"].items():
                assert condition.get("threshold") == dev["conditions"][name].get("threshold"), (path, name)
            rows.append({"run": path.parent.parent.name, "mode": mode,
                         "conditions": list(evaluation["conditions"]), "thresholds_equal": True})
    if len(rows) != 14:
        raise RuntimeError(f"expected 14 Meta/non-Meta Eval/Dev pairs, got {len(rows)}")
    detectors = []
    for path in sorted((OUT / "evaluation/detectors").glob("formal_*_seed3407/eval/summary.json")):
        dev_path = path.parent.parent / "dev/summary.json"
        evaluation = json.loads(path.read_text())
        dev = json.loads(dev_path.read_text())
        assert evaluation["threshold"] == dev["threshold"], path
        detectors.append({"run": path.parent.parent.name, "thresholds_equal": True})
    if len(detectors) != 2:
        raise RuntimeError(f"expected 2 supervised-control pairs, got {len(detectors)}")
    destination = OUT / "audits/eval_threshold_lock_seed3407_v2.json"
    destination.write_text(json.dumps({"status": "passed", "total_pairs": len(rows) + len(detectors),
                                       "meta_or_nonmeta_pairs": len(rows), "detector_pairs": len(detectors),
                                       "rows": rows, "detector_rows": detectors}, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
