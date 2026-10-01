#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESCORE = ROOT / "outputs/reports/tusz_meta_ttt_v4/rescoring"
spec = importlib.util.spec_from_file_location("v4_rescore", ROOT / "scripts/401_rescore_tusz_meta_ttt_v4.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def patient_totals(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("patient_id", as_index=False).agg(
        detected=("detected", "sum"), total=("total", "sum"),
        false_alarms=("false_alarms", "sum"), background_hours=("background_hours", "sum"),
        false_alarm_seconds=("false_alarm_seconds", "sum"),
    ).sort_values("patient_id").reset_index(drop=True)


def bootstrap(frozen, adapted, *, seed=3407, replicates=10_000):
    if frozen.patient_id.tolist() != adapted.patient_id.tolist():
        raise ValueError("patient bootstrap inputs are not paired")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(frozen), size=(replicates, len(frozen)))
    def sums(frame, column):
        return frame[column].to_numpy()[indexes].sum(1)
    frozen_sens = sums(frozen, "detected") / sums(frozen, "total")
    adapted_sens = sums(adapted, "detected") / sums(adapted, "total")
    frozen_fa = sums(frozen, "false_alarms") / sums(frozen, "background_hours")
    adapted_fa = sums(adapted, "false_alarms") / sums(adapted, "background_hours")
    frozen_time = sums(frozen, "false_alarm_seconds") / 60 / sums(frozen, "background_hours")
    adapted_time = sums(adapted, "false_alarm_seconds") / 60 / sums(adapted, "background_hours")
    def interval(values):
        return [float(value) for value in np.quantile(values, [0.025, 0.5, 0.975])]
    return {
        "replicates": replicates,
        "patients": len(frozen),
        "sensitivity_difference": interval(adapted_sens - frozen_sens),
        "fa_per_hour_ratio": interval(adapted_fa / frozen_fa),
        "fa_per_hour_difference": interval(adapted_fa - frozen_fa),
        "fa_time_difference": interval(adapted_time - frozen_time),
    }


def main():
    inventory, _ = module._inventory()
    results = {}
    for objective in ("band", "mask"):
        for condition in ("a", "b", "c"):
            base = ROOT / f"outputs/reports/tusz_meta_ttt_v4/evaluation/future/{condition}_{condition}_{objective}/development_validation/probabilities.parquet"
            frame = pd.read_parquet(base)
            frozen_records = module._records(frame, "meta_frozen", inventory)
            adapted_records = module._records(frame, "adapted", inventory)
            pair_name = f"v4_{objective}_{condition}_adapted"
            pair = json.loads((RESCORE / pair_name / "paired_comparison.json").read_text())
            targets = {}
            for target in ("60", "65", "70", "75"):
                point = pair["operating_points"][target]
                if point["frozen"] is None or point["adapted"] is None:
                    targets[target] = None
                    continue
                frozen = patient_totals(module._record_rows(frozen_records, point["frozen"]["threshold"]))
                adapted = patient_totals(module._record_rows(adapted_records, point["adapted"]["threshold"]))
                targets[target] = bootstrap(frozen, adapted, seed=3407 + int(target))
            results[pair_name] = targets
            output = RESCORE / pair_name / "patient_bootstrap.json"
            output.write_text(json.dumps(targets, indent=2) + "\n")
            print(json.dumps({"condition": pair_name}), flush=True)
    (RESCORE / "patient_bootstrap_all.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
