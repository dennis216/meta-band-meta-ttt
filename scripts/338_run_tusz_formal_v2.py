#!/usr/bin/env python3
"""Resumable three-seed formal training, Dev calibration, and fixed Eval scoring."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bfa.tusz_meta_ttt_v2.launch import run_monitored

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"
SEEDS = (17, 42, 3407)


def call(command: list[str], log: Path) -> None:
    run_monitored(command, cwd=ROOT, log=log)


def source(seed: int) -> Path:
    return V1 / f"runs/supervised/formal/s1_seed{seed}_check0.25/best.pt"


def train_ssl(seed: int, objective: str, difficulty: float) -> Path:
    run = OUT / "runs/ssl/formal" / f"{objective}_{difficulty:g}_seed{seed}"
    checkpoint = run / "epoch_02.pt"
    if not checkpoint.is_file():
        command = [
            sys.executable, str(ROOT / "scripts/321_train_tusz_ssl_v2.py"),
            "--source", str(source(seed)), "--stage", "formal", "--seed", str(seed),
            "--objective", objective, "--difficulty", str(difficulty), "--epochs", "2",
            "--batch-size", "64", "--workers", "4",
        ]
        if (run / "last.pt").is_file():
            command.extend(("--resume", str(run / "last.pt")))
        call(command, OUT / "logs/formal" / f"ssl_{objective}_{difficulty:g}_seed{seed}.log")
    return checkpoint


def calibrate(seed: int, objective: str, difficulty: float, ssl_checkpoint: Path) -> Path:
    destination = (
        OUT / "calibration/formal" / f"{objective}_{difficulty:g}_seed{seed}"
        / "gradient_calibration.json"
    )
    if not destination.is_file():
        call(
            [
                sys.executable, str(ROOT / "scripts/322_calibrate_tusz_inner_v2.py"),
                "--source", str(source(seed)), "--stage", "formal", "--seed", str(seed),
                "--objective-checkpoint", str(ssl_checkpoint), "--objective", objective,
                "--difficulty", str(difficulty),
            ],
            OUT / "logs/formal" / f"calibrate_{objective}_{difficulty:g}_seed{seed}.log",
        )
    return destination


def evaluate_meta(checkpoint: Path, tag: str) -> tuple[Path, Path]:
    state = torch_load(checkpoint)
    name = checkpoint.parent.name
    dev = OUT / "evaluation" / state["mode"] / name / "dev/summary.json"
    if not dev.is_file():
        call(
            [
                sys.executable, str(ROOT / "scripts/323_evaluate_tusz_meta_ttt_v2.py"),
                "--meta-checkpoint", str(checkpoint), "--partition", "dev", "--calibrate",
            ],
            OUT / "logs/formal" / f"evaluate_dev_{tag}.log",
        )
    evaluation = OUT / "evaluation" / state["mode"] / name / "eval/summary.json"
    if not evaluation.is_file():
        call(
            [
                sys.executable, str(ROOT / "scripts/323_evaluate_tusz_meta_ttt_v2.py"),
                "--meta-checkpoint", str(checkpoint), "--partition", "eval",
                "--thresholds", str(dev),
            ],
            OUT / "logs/formal" / f"evaluate_eval_{tag}.log",
        )
    return dev, evaluation


def torch_load(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--development-queue", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=[3407])
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()
    if set(args.seeds) != set(SEEDS):
        args.skip_aggregate = True
    selected = json.loads(args.selection.read_text())["modes"]
    development = json.loads(args.development_queue.read_text())["runs"]
    conditions = []
    for mode in ("future", "current"):
        first = selected[mode]["first"]
        for rank in ("first", "second"):
            row = selected[mode][rank]
            conditions.append({**row, "mode": mode, "outer_scope": "eds"})
        for scope in ("e", "ed", "es"):
            match = next(
                row for row in development
                if row.get("mode") == mode
                and row.get("objective") == first["objective"]
                and row.get("outer_scope") == scope
                and row.get("status") == "complete"
            )
            conditions.append(match)
    unique_conditions = {
        (row["mode"], row["objective"], row["outer_scope"]): row for row in conditions
    }
    plan = {
        "seeds": list(args.seeds),
        "conditions": list(unique_conditions.values()),
        "status": "planned",
    }
    state_path = OUT / "queues/formal_v2.json"
    if args.skip_aggregate:
        suffix = "_".join(map(str, args.seeds))
        state_path = OUT / "queues" / f"formal_seed_{suffix}_v2.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(plan, indent=2) + "\n")
    if not args.execute:
        print(json.dumps({"queue": str(state_path), "conditions_per_seed": len(unique_conditions)}))
        return

    for seed in args.seeds:
        assets = {}
        formal_checkpoints = {}
        for row in unique_conditions.values():
            key = (row["objective"], float(row["difficulty"]))
            if key in assets:
                continue
            ssl_checkpoint = train_ssl(seed, *key)
            calibration_path = calibrate(seed, *key, ssl_checkpoint)
            calibration_state = json.loads(calibration_path.read_text())
            if not calibration_state["health_check_passed"]:
                raise RuntimeError(f"formal inner health check failed: seed={seed}, task={key}")
            assets[key] = (ssl_checkpoint, calibration_path, calibration_state["selected_relative_step"])
        for row in unique_conditions.values():
            key = (row["objective"], float(row["difficulty"]))
            ssl_checkpoint, calibration_path, relative_step = assets[key]
            epochs = int(row["best_epoch"])
            name = (
                f'{row["objective"]}_{row["difficulty"]:g}_{row["outer_scope"]}_'
                f'rho{relative_step:g}_seed{seed}'
            )
            run = OUT / "runs/meta/formal" / row["mode"] / name
            checkpoint = run / f"epoch_{epochs:02d}.pt"
            if not checkpoint.is_file():
                command = [
                    sys.executable, str(ROOT / "scripts/320_train_tusz_meta_ttt_v2.py"),
                    "--source", str(source(seed)), "--stage", "formal", "--seed", str(seed),
                    "--objective-checkpoint", str(ssl_checkpoint),
                    "--objective", row["objective"], "--difficulty", str(row["difficulty"]),
                    "--mode", row["mode"], "--outer-scope", row["outer_scope"],
                    "--relative-step", str(relative_step),
                    "--gradient-calibration", str(calibration_path), "--epochs", str(epochs),
                ]
                if (run / "last.pt").is_file():
                    command.extend(("--resume", str(run / "last.pt")))
                call(command, OUT / "logs/formal" / f"meta_{row['mode']}_{name}.log")
            evaluate_meta(checkpoint, f"{row['mode']}_{name}")
            formal_checkpoints[(row["mode"], row["objective"], row["outer_scope"])] = checkpoint

        for mode in ("future", "current"):
            for rank in ("first", "second"):
                row = selected[mode][rank]
                key = (row["objective"], float(row["difficulty"]))
                ssl_checkpoint, calibration_path, _relative_step = assets[key]
                package = (
                    OUT / "runs/nonmeta/formal" / mode
                    / f'{row["objective"]}_{row["difficulty"]:g}_seed{seed}' / "checkpoint.pt"
                )
                if not package.is_file():
                    call(
                        [
                            sys.executable,
                            str(ROOT / "scripts/329_package_tusz_nonmeta_ttt_v2.py"),
                            "--source", str(source(seed)),
                            "--objective-checkpoint", str(ssl_checkpoint),
                            "--gradient-calibration", str(calibration_path),
                            "--mode", mode, "--output", str(package),
                        ],
                        OUT / "logs/formal" / f"nonmeta_package_{mode}_{row['objective']}_seed{seed}.log",
                    )
                evaluate_meta(package, f"nonmeta_{mode}_{row['objective']}_seed{seed}")

        for mode in ("future", "current"):
            row = selected[mode]["first"]
            checkpoint = formal_checkpoints[(mode, row["objective"], "eds")]
            name = checkpoint.parent.name
            dev_directory = OUT / "evaluation" / mode / name / "dev"
            dev_summary = json.loads((dev_directory / "summary.json").read_text())
            source_choice = dev_summary["conditions"]["source_frozen"]
            source_threshold = source_choice["threshold"]
            if source_threshold is None:
                source_threshold = source_choice["maximum_sensitivity_threshold"]
            for partition in ("train", "dev", "eval"):
                evaluation = OUT / "evaluation" / mode / name / partition
                if not (evaluation / "probabilities.parquet").is_file():
                    call(
                        [
                            sys.executable,
                            str(ROOT / "scripts/323_evaluate_tusz_meta_ttt_v2.py"),
                            "--meta-checkpoint", str(checkpoint),
                            "--partition", partition,
                            "--thresholds", str(dev_directory / "summary.json"),
                        ],
                        OUT / "logs/formal" / f"gradient_scores_{mode}_{partition}_{name}.log",
                    )
                mechanism_summary = (
                    OUT / "mechanisms" / mode / name / partition / "summary.json"
                )
                if not mechanism_summary.is_file():
                    call(
                        [
                            sys.executable,
                            str(ROOT / "scripts/328_analyze_tusz_gradients_v2.py"),
                            "--meta-checkpoint", str(checkpoint),
                            "--probabilities", str(evaluation / "probabilities.parquet"),
                            "--source-threshold", str(source_threshold),
                            "--partition", partition,
                        ],
                        OUT / "logs/formal" / f"gradient_{mode}_{partition}_{name}.log",
                    )

        for scope in ("e", "ed"):
            control_run = OUT / "runs/supervised_controls/formal" / f"{scope}_seed{seed}"
            control_checkpoint = control_run / "epoch_02.pt"
            if not control_checkpoint.is_file():
                command = [
                    sys.executable,
                    str(ROOT / "scripts/326_train_tusz_supervised_control_v2.py"),
                    "--source", str(source(seed)), "--stage", "formal", "--seed", str(seed),
                    "--scope", scope, "--epochs", "2",
                ]
                if (control_run / "last.pt").is_file():
                    command.extend(("--resume", str(control_run / "last.pt")))
                call(
                    command,
                    OUT / "logs/formal" / f"supervised_{scope}_seed{seed}.log",
                )
            dev_tag = f"formal_{scope}_seed{seed}"
            dev_summary = OUT / "evaluation/detectors" / dev_tag / "dev/summary.json"
            if not dev_summary.is_file():
                call(
                    [
                        sys.executable, str(ROOT / "scripts/335_evaluate_tusz_detector_v2.py"),
                        "--checkpoint", str(control_checkpoint), "--partition", "dev",
                        "--calibrate", "--tag", dev_tag,
                    ],
                    OUT / "logs/formal" / f"supervised_eval_dev_{scope}_seed{seed}.log",
                )
            eval_summary = OUT / "evaluation/detectors" / dev_tag / "eval/summary.json"
            if not eval_summary.is_file():
                call(
                    [
                        sys.executable, str(ROOT / "scripts/335_evaluate_tusz_detector_v2.py"),
                        "--checkpoint", str(control_checkpoint), "--partition", "eval",
                        "--threshold-summary", str(dev_summary), "--tag", dev_tag,
                    ],
                    OUT / "logs/formal" / f"supervised_eval_eval_{scope}_seed{seed}.log",
                )
        plan.setdefault("completed_seeds", []).append(seed)
        state_path.write_text(json.dumps(plan, indent=2) + "\n")
    if args.skip_aggregate:
        plan["status"] = "seed_runs_complete"
        state_path.write_text(json.dumps(plan, indent=2) + "\n")
        return
    bootstrap_outputs = []
    for row in unique_conditions.values():
        evaluation_paths = []
        dev_summaries = []
        for seed in SEEDS:
            pattern = (
                f'{row["objective"]}_{row["difficulty"]:g}_{row["outer_scope"]}_'
                f'rho*_seed{seed}'
            )
            matches = [
                path for path in (OUT / "runs/meta/formal" / row["mode"]).glob(pattern)
                if path.is_dir() and "detector_open" not in path.name
                and "stop_encoder" not in path.name and "_sgd" not in path.name
            ]
            if len(matches) != 1:
                raise RuntimeError(f"cannot uniquely resolve formal run: {pattern}")
            evaluation_root = OUT / "evaluation" / row["mode"] / matches[0].name
            evaluation_paths.append(evaluation_root / "eval/probabilities.parquet")
            dev_summaries.append(evaluation_root / "dev/summary.json")
        bootstrap = (
            OUT / "statistics/bootstrap"
            / f'{row["mode"]}_{row["objective"]}_{row["outer_scope"]}.json'
        )
        call(
            [
                sys.executable, str(ROOT / "scripts/330_bootstrap_tusz_meta_ttt_v2.py"),
                "--evaluations", *map(str, evaluation_paths),
                "--dev-summaries", *map(str, dev_summaries),
                "--output", str(bootstrap),
            ],
            OUT / "logs/formal" / f"bootstrap_{row['mode']}_{row['objective']}_{row['outer_scope']}.log",
        )
        bootstrap_outputs.append(bootstrap)
    call(
        [
            sys.executable, str(ROOT / "scripts/332_holm_tusz_secondary_v2.py"),
            "--comparisons", *map(str, bootstrap_outputs),
            "--output", str(OUT / "statistics/holm_secondary.json"),
        ],
        OUT / "logs/formal/holm_secondary.log",
    )
    plan["status"] = "complete"
    state_path.write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    main()
