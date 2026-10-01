#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "outputs/reports/tusz_meta_ttt_v4"
RESCORE = V4 / "rescoring"
HIGH = 0.732138168811798


def bce(probability, label):
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    label = np.asarray(label, dtype=np.float64)
    logits = np.log(probability) - np.log1p(-probability)
    return np.maximum(logits, 0) - logits * label + np.log1p(np.exp(-np.abs(logits)))


def cohort_weights(frame):
    keys = ["patient_id", "session_id", "montage", "record_id"]
    working = frame[keys + ["label"]].copy()
    working["positive"] = working.label > 0
    working["record_token"] = (
        working.session_id.astype(str) + "|" + working.montage.astype(str)
        + "|" + working.record_id.astype(str)
    )
    patient_counts = working.groupby("positive")["patient_id"].nunique()
    patient_factor = working.positive.map(patient_counts).to_numpy(dtype=np.float64)
    record_factor = working.groupby(["positive", "patient_id"])["record_token"].transform("nunique").to_numpy(dtype=np.float64)
    window_factor = working.groupby(["positive", "patient_id", "record_token"])["label"].transform("size").to_numpy(dtype=np.float64)
    return 0.5 / (patient_factor * record_factor * window_factor)


def carry_summary(path, source_scores):
    frame = pd.read_parquet(path).sort_values([
        "patient_id", "session_id", "montage", "record_id", "decision_end_s"
    ]).reset_index(drop=True)
    source = source_scores[["patient_id", "session_id", "montage", "record_id", "decision_end_s", "source_frozen"]]
    if "source_frozen" not in frame:
        frame = frame.merge(source, on=["patient_id", "session_id", "montage", "record_id", "decision_end_s"], validate="one_to_one")
    delta = bce(frame.adapted, frame.label) - bce(frame.meta_frozen, frame.label)
    weights = cohort_weights(frame)
    positive = frame.label.to_numpy() > 0
    high = (~positive) & (frame.source_frozen.to_numpy() >= HIGH)
    groups = {"seizure": positive, "high_background": high, "ordinary_background": (~positive) & (~high)}
    output = {
        "windows": len(frame), "patients": int(frame.patient_id.nunique()),
        "class_weight_mass": {
            "seizure": float(weights[positive].sum()),
            "background": float(weights[~positive].sum()),
        },
        "total_weighted_delta_bce": float((delta * weights).sum()),
        "unweighted_mean_delta_bce": float(delta.mean()),
        "beneficial_window_fraction": float((delta < 0).mean()),
        "worst_10_percent_threshold": float(np.quantile(delta, 0.9)),
        "groups": {}, "record_position": {},
    }
    for name, mask in groups.items():
        output["groups"][name] = {
            "windows": int(mask.sum()), "weight_mass": float(weights[mask].sum()),
            "weighted_contribution": float((delta[mask] * weights[mask]).sum()),
            "weighted_mean": float(np.average(delta[mask], weights=weights[mask])),
            "unweighted_mean": float(delta[mask].mean()),
            "beneficial_fraction": float((delta[mask] < 0).mean()),
        }
    minutes = frame.decision_end_s.to_numpy() / 60
    bins = {
        "0-2": minutes < 2, "2-5": (minutes >= 2) & (minutes < 5),
        "5-10": (minutes >= 5) & (minutes < 10), "10+": minutes >= 10,
    }
    for name, mask in bins.items():
        output["record_position"][name] = {
            "windows": int(mask.sum()), "mean_delta_bce": float(delta[mask].mean()),
            "beneficial_fraction": float((delta[mask] < 0).mean()),
        }
    return output


def metric_rows():
    rows = []
    for objective in ("band", "mask"):
        for condition in ("a", "b", "c"):
            name = f"v4_{objective}_{condition}_adapted"
            pair = json.loads((RESCORE / name / "paired_comparison.json").read_text())
            bootstrap = json.loads((RESCORE / name / "patient_bootstrap.json").read_text())
            for target in ("60", "65", "70", "75", "80"):
                point = pair["operating_points"][target]
                frozen, adapted = point["frozen"], point["adapted"]
                row = {"objective": objective, "condition": condition, "target": int(target), "reachable": frozen is not None and adapted is not None}
                if row["reachable"]:
                    row |= {
                        "frozen_sensitivity": frozen["sensitivity"], "adapted_sensitivity": adapted["sensitivity"],
                        "frozen_fa_per_hour": frozen["false_alarms_per_hour"], "adapted_fa_per_hour": adapted["false_alarms_per_hour"],
                        "fa_relative_change": adapted["false_alarms_per_hour"] / frozen["false_alarms_per_hour"] - 1,
                        "frozen_fa_time": frozen["false_alarm_minutes_per_hour"], "adapted_fa_time": adapted["false_alarm_minutes_per_hour"],
                        "common_delay_difference_s": point["common_delay"]["median_adapted_minus_frozen_delay_s"],
                    }
                    if target != "80":
                        row |= {
                            "bootstrap_sensitivity_difference_low": bootstrap[target]["sensitivity_difference"][0],
                            "bootstrap_fa_ratio_high": bootstrap[target]["fa_per_hour_ratio"][2],
                        }
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    source_path = V4 / "evaluation/future/a_a_mask/development_validation/probabilities.parquet"
    source = pd.read_parquet(source_path)
    carry = {}
    for objective in ("band", "mask"):
        for condition in ("a", "b", "c"):
            path = V4 / f"evaluation/future/{condition}_{condition}_{objective}/development_validation/probabilities.parquet"
            carry[f"{objective}_{condition}"] = carry_summary(path, source)
    (V4 / "mechanisms/carry_summary.json").write_text(json.dumps(carry, indent=2) + "\n")
    results = metric_rows()
    reports = V4 / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    results.to_csv(reports / "development_results.csv", index=False)

    lines = [
        "# TUSZ Meta-TTT v4 final development report", "",
        "## Decision", "",
        "No v4 condition passed the preregistered F-mode promotion rule. The corrected alarm pipeline could not reach 80% event sensitivity for any v4 checkpoint, and the largest FA/hour reduction at reachable matched-sensitivity points was 4.46%, below the required 10%. C-mode and Eval were therefore not run.", "",
        "## Corrected scoring", "",
        "Event matching now maximizes the number of valid >=1 s matches before overlap duration. Threshold candidates come from per-EDF causal EMA scores. Alarm offset is the second below-threshold decision time and open alarms are clamped to the record endpoint. The development cohort contains 1,218 valid records, including 31 records with no prediction windows, and 472 original seizure events.", "",
        "## Development results", "",
        "| SSL | Condition | 60% FA/h change | 65% | 70% | 75% | 80% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for objective in ("band", "mask"):
        for condition in ("a", "b", "c"):
            values = []
            for target in (60, 65, 70, 75):
                row = results[(results.objective == objective) & (results.condition == condition) & (results.target == target)].iloc[0]
                values.append(f"{100 * row.fa_relative_change:+.2f}%")
            lines.append(f"| {objective.title()} | {condition.upper()} | {' | '.join(values)} | unreachable |")
    lines += [
        "", "Condition A is protected base plus Post-BCE; B uses paired gain with a trainable detector; C uses paired gain with the detector fixed. Band C is the strongest stable result. At 60/65/70/75% it reduces FA/hour by 4.46/3.66/2.32/2.37%, reduces false-alarm time at every point, and has zero median delay change on commonly detected events. Its patient bootstrap FA-rate-ratio upper bound is below 1 at all four points, while the sensitivity-difference lower bound remains above -2 percentage points. The effect is statistically consistent but smaller than the practical 10% target.", "",
        "## Corrected v3 comparison", "",
        "Under the corrected scorer, no v3 condition reaches 80%. At the 60% point, the best v3 reductions are Band B1 at 4.85% and Mask B0 at 1.95%; most v3 checkpoints cannot reach 70–75%. v4 does not exceed the single best v3 60% number, but Band C preserves a consistent reduction through 75% sensitivity and reduces false-alarm time at every reported point.", "",
        "## Mechanism", "",
    ]
    for condition in ("a", "b", "c"):
        frame = pd.read_parquet(V4 / f"mechanisms/future/{condition}_v4_band_{condition}/development_validation/gradient_samples.parquet")
        lines.append(
            f"- Band {condition.upper()}: n={len(frame):,}, mean actual descent cosine={frame.actual_descent_cosine.mean():.3f}, mean immediate ΔBCE={frame.actual_delta_bce.mean():+.6f}, beneficial updates={100*(frame.actual_delta_bce < 0).mean():.1f}%."
        )
    lines += [
        "", "Paired-gain training changed the update rule substantially: B/C achieve approximately 0.30 mean descent cosine and improve sampled future BCE in about 88% of locations; A remains near zero cosine and has positive mean ΔBCE. The gain is concentrated in background and S1 false-alarm locations. For Band C, mean ΔBCE is -0.001685 on ordinary background, -0.004051 on high-score background and -0.007466 on S1 false alarms, but +0.001979 near onset and +0.002663 in seizure chunks. This opposing state dependence explains why a strong window-level average gain becomes only a 2–4% event-level FA reduction after threshold recalibration.", "",
        "The auxiliary Band task improves in aggregate on the same support, an independent transform and the future chunk. B/C therefore do not fail because the SSL task is unlearnable. Norm-matched random directions have approximately zero effect; future-label local updates are much more beneficial, confirming that useful update space exists.", "",
        "Full-cohort carry analysis sharpens the result. Band B improves 92.2% of validation windows and has a negative class-balanced total ΔBCE (-9.90e-5), but Band C has positive class-balanced total ΔBCE (+1.06e-3) despite improving 92.1% of windows. For C, background improves strongly while seizure weighted mean ΔBCE is +0.03797. Mask B/C likewise improve about 91.7% of windows but have positive class-balanced totals (+0.0100/+0.0104). Window counts therefore conceal a small, heavily weighted seizure subset that dominates the balanced objective.", "",
        "## Gradient paths", "",
    ]
    for condition in ("b", "c"):
        decomposition = json.loads((V4 / f"mechanisms/decomposition_band_{condition}.json").read_text())
        e10 = decomposition["summary_medians"]["encoder10"]
        e11 = decomposition["summary_medians"]["encoder11"]
        lines.append(
            f"- Band {condition.upper()}: through-inner/full norm ratio is {e10['through_to_full']:.3f} in block 10 and {e11['through_to_full']:.3f} in block 11. SSL-head direct gradient is exactly zero; its full gradient is entirely through the update. Full = direct + through reconstruction error is below 2e-8."
        )
    lines += [
        "", "Meta-learning is therefore using the second-order update path; the remaining limitation is not a missing Meta gradient. The learned Band direction systematically lowers background scores and helps false alarms, while harming seizure/onset chunks. The paired objective improves the average update but cannot express different directions for different latent states with one unconditional SSL updater.", "",
        "## Throughput and coverage", "",
    ]
    for objective in ("mask", "band"):
        history = json.loads((V4 / f"runs/development/{objective}_f/a/history.json").read_text())
        elapsed = history[-1]["elapsed_s"]
        lines.append(f"- {objective.title()}: two complete traversals, 463 patients, 3,838 records and 1,261,284 windows per traversal; wall time {elapsed/3600:.2f} h for all three conditions jointly.")
    lines += [
        "- Mask used four patient lanes (about 25 GiB peak PyTorch allocation); Band used two patient lanes (about 25 GiB). The 1-vs-4 lane parameter difference was <=5.96e-8 and accepted-update states were identical.",
        "- All six conditions completed exactly two traversals. Update acceptance was >=99.9%. The shared cohort weights sum to 0.5 seizure and 0.5 background.", "",
        "## Validation", "",
        "121 TUSZ/evaluation tests pass. Added tests cover lexicographic event matching, actual alarm offset, EMA-derived thresholds, non-detached paired gain, zero-update gain/gradient, KL stationarity and paired damage. Deployment evaluation replays every adapted probability exactly in the mechanism audit (maximum error 0).", "",
        "## Failure attribution", "",
        "v4 succeeds at its narrow mechanism goal: it learns a nontrivial, second-order-dependent update that improves future BCE and consistently reduces background false-alarm burden. It fails the practical promotion criterion because the same update harms seizure/onset states and because the paired objective degrades the Frozen initialization relative to protected Post-BCE training. The next justified experiment is a state-conditioned updater or an update gate trained to suppress adaptation near seizure-like states; further unconditional gain scaling is not supported by these results.",
    ]
    (reports / "V4_FINAL_REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
