"""Train the three future-mode v4 objectives in one shared-prefix ensemble."""
import argparse
import fcntl
import hashlib
import json
import math
import random
import signal
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from bfa.tusz_meta_ttt.dataset import filter_inventory_records, load_cached_arrays
from bfa.tusz_meta_ttt_v2.batched import (
    ConditionEnsemble, EnsembleBandObjective, EnsembleMaskObjective,
    enable_second_order_batched_attention,
)
from bfa.tusz_meta_ttt_v2.grouping import patient_order
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.protocol import assert_train_cache_paths, class_patient_record_weights, weight_audit
from bfa.tusz_meta_ttt_v2.runtime import load_source, optimizer_groups
from bfa.tusz_meta_ttt_v2.training_state import atomic_save, capture_rng, detector_open, restore_rng, validate_resume
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig
from bfa.tusz_meta_ttt_v4.joint_schedule import process_future_group
from bfa.tusz_meta_ttt_v4.losses import CONDITIONS, LossScale, source_probability_lookup

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/reports/tusz_meta_ttt_v1"
V2 = ROOT / "outputs/reports/tusz_meta_ttt_v2"
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v4"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_state(module, names=None):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()
            if names is None or name in names}


def as_json(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=["band", "mask"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loss-scale", type=Path, required=True)
    parser.add_argument("--reference-probabilities", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=[item.name for item in CONDITIONS],
                        default=[item.name for item in CONDITIONS])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patients-per-batch", type=int, choices=[1, 2, 4], required=True)
    parser.add_argument("--patient-bucket-size", type=int, choices=[0, 8, 16, 32], default=8)
    parser.add_argument("--seed", type=int, choices=[3407], default=3407)
    parser.add_argument("--stage", choices=["development", "formal"], default="development")
    parser.add_argument("--detector-freeze-fraction", type=float, default=0.25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument("--maximum-records", type=int)
    parser.add_argument("--maximum-updates-per-patient", type=int)
    parser.add_argument("--stop-after-groups", type=int)
    args = parser.parse_args()
    selected_conditions = tuple(item for item in CONDITIONS if item.name in args.conditions)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    output_lock = (args.output / ".training.lock").open("a")
    fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("highest")
    enable_second_order_batched_attention()

    source = V1 / f"runs/supervised/{args.stage}/s1_seed3407_check0.25/best.pt"
    difficulty = 0.5 if args.objective == "band" else 5
    task = f"{args.objective}_{difficulty:g}_seed3407"
    head = V2 / f"runs/ssl/{args.stage}/{task}/epoch_02.pt"
    calibration_path = V2 / ("calibration/formal" if args.stage == "formal" else "calibration") / task / "gradient_calibration.json"
    calibration = json.loads(calibration_path.read_text())
    inner_config = InnerStepConfig(
        calibration["selected_relative_step"],
        {int(key): value for key, value in calibration["block_reference_norms"].items()},
        {int(key): value for key, value in calibration["block_gradient_medians"].items()},
    )
    loss_scale = LossScale(**json.loads(args.loss_scale.read_text())["loss_scale"])
    probabilities = args.reference_probabilities.resolve()
    source_lookup = source_probability_lookup(probabilities)

    split = json.loads((V1 / "manifests/development_split.json").read_text())
    fit = set(split["development_fit"]) if args.stage == "development" else None
    inventory = json.loads((V1 / "manifests/records.json").read_text())
    paths = filter_inventory_records(
        [path for path in sorted((V1 / "cache/train").rglob("*.npz"))
         if fit is None or path.parts[-4] in fit], inventory, partition="train"
    )
    assert_train_cache_paths(paths, V1 / "cache/train")
    if args.maximum_records:
        paths = paths[:args.maximum_records]
    labels = {path: load_cached_arrays(path)["labels"] for path in paths}
    missing_reference = [str(path) for path in paths if len(labels[path]) and
                         (path.parts[-4], path.parts[-3], path.parts[-2], path.stem) not in source_lookup]
    if missing_reference:
        raise ValueError(f"reference probabilities missing {len(missing_reference)} training records")
    weights = class_patient_record_weights(labels)
    patients = defaultdict(list)
    for path in paths:
        patients[path.parts[-4]].append(path)
    if not args.maximum_records and len(patients) != (len(fit) if fit is not None else 579):
        raise ValueError("incomplete patient coverage")
    workloads = {patient: sum(len(labels[path]) for path in records) for patient, records in patients.items()}
    class_audit = weight_audit(weights, labels)
    if not all(abs(class_audit[name] - 0.5) < 1e-6 for name in ["background", "seizure"]):
        raise ValueError(f"invalid class weights: {class_audit}")

    contract = dict(
        version="meta_ttt_v4_1", objective=args.objective, difficulty=difficulty, seed=args.seed,
        mode="future", conditions=[condition.name for condition in selected_conditions], outer_scope="eds",
        source_sha256=sha256(source), head_sha256=sha256(head), calibration_sha256=sha256(calibration_path),
        loss_scale_sha256=sha256(args.loss_scale), loss_scale=vars(loss_scale),
        reference_probabilities=str(probabilities),
        patients_per_batch=args.patients_per_batch, patient_bucket_size=args.patient_bucket_size,
        maximum_updates_per_patient=args.maximum_updates_per_patient,
        detector_freeze_fraction=args.detector_freeze_fraction, inner_config=vars(inner_config),
        patients=len(patients), records=len(paths), windows=sum(workloads.values()),
        class_weight_audit=class_audit, stage="smoke" if args.maximum_records else args.stage,
        truncation="four synchronized updates; straight-through initialization reattachment",
        prefix_precision="fp32", tail_precision="fp32", matmul_precision="highest",
    )
    if (args.output / "complete.json").is_file():
        completed = json.loads((args.output / "complete.json").read_text())
        if completed["epochs"] >= args.epochs and completed["contract"] == contract:
            print(json.dumps({"phase": "already_complete", "output": str(args.output)}), flush=True)
            return
        raise ValueError("completed output has a different contract")
    if (args.output / "last.pt").exists() and not args.resume:
        raise FileExistsError("use --resume for an existing v4 run")

    models, objectives, optimizers = [], [], []
    for condition in selected_conditions:
        model = load_source(source, detector_trainable=condition.train_detector)
        ssl = build_objective(args.objective, difficulty).cuda().eval()
        ssl.load_state_dict(torch.load(head, map_location="cpu", weights_only=False)["objective"])
        ssl.requires_grad_(True)
        groups = (
            optimizer_groups(model, "backbone.encoder.layers.10.", 1e-5, 0.01)
            + optimizer_groups(model, "backbone.encoder.layers.11.", 1e-5, 0.01)
            + (optimizer_groups(model, "detector.", 3e-6, 0.01) if condition.train_detector else [])
            + [dict(params=list(ssl.parameters()), lr=1e-4, weight_decay=0.0)]
        )
        models.append(model)
        objectives.append(ssl)
        optimizers.append(torch.optim.AdamW(groups, fused=True))
    functional = ConditionEnsemble(models)
    ensemble_objective = (EnsembleBandObjective if args.objective == "band" else EnsembleMaskObjective)(objectives, deduplicate_views=True)
    mutable = set(functional.adaptable_names) | {name for name, _ in models[0].named_parameters() if name.startswith("detector.")}
    histories = [[] for _ in selected_conditions]
    epoch, cursor, elapsed_prior, global_steps, epoch_steps = 1, 0, 0.0, 0, 0
    audit = {}
    if args.resume:
        saved = torch.load(args.output / "last.pt", map_location="cpu", weights_only=False)
        validate_resume(saved["contract"], contract)
        for model, ssl, optimizer, condition_state in zip(models, objectives, optimizers, saved["conditions"], strict=True):
            model.load_state_dict(condition_state["model_mutable"], strict=False)
            ssl.load_state_dict(condition_state["objective"])
            optimizer.load_state_dict(condition_state["optimizer"])
        histories = saved["histories"]
        epoch, cursor = saved["epoch"], saved["next_patient"]
        audit = {key: torch.tensor(value, device="cuda") for key, value in saved["audit"].items()}
        elapsed_prior, global_steps, epoch_steps = saved["elapsed_s"], saved["global_steps"], saved["epoch_steps"]
        restore_rng(saved["rng"])
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2))
    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.perf_counter()

    def save(next_epoch, next_patient):
        states = [dict(model_mutable=cpu_state(model, mutable), objective=cpu_state(ssl), optimizer=optimizer.state_dict())
                  for model, ssl, optimizer in zip(models, objectives, optimizers, strict=True)]
        atomic_save(dict(contract=contract, conditions=states, histories=histories, epoch=next_epoch,
                         next_patient=next_patient, audit={key: as_json(value) for key, value in audit.items()},
                         rng=capture_rng(), elapsed_s=elapsed_prior + time.perf_counter() - started,
                         global_steps=global_steps, epoch_steps=epoch_steps), args.output / "last.pt")

    print(json.dumps({"phase": "training_started", "contract": contract}), flush=True)
    executed_groups = 0
    try:
        while epoch <= args.epochs:
            order = patient_order(workloads, seed=args.seed + epoch, bucket_size=args.patient_bucket_size)
            for start in range(cursor, len(order), 4):
                selected = order[start:start + 4]
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                group_audit = {}
                step_started = time.perf_counter()
                for lane_start in range(0, len(selected), args.patients_per_batch):
                    lane_patients = selected[lane_start:lane_start + args.patients_per_batch]
                    for loss, _count, _accepted in process_future_group(
                        functional, ensemble_objective, [patients[patient] for patient in lane_patients],
                        weights, inner_config, selected_conditions, source_lookup, loss_scale,
                        seed=args.seed, compact=True, audit=group_audit,
                        maximum_updates_per_patient=args.maximum_updates_per_patient,
                    ):
                        loss.backward()
                open_head = detector_open(epoch, start // 4, math.ceil(len(order) / 4), args.detector_freeze_fraction)
                norms, changes, before = [], [], []
                for condition, model, ssl in zip(selected_conditions, models, objectives, strict=True):
                    for parameter in [*model.parameters(), *ssl.parameters()]:
                        if parameter.grad is not None:
                            parameter.grad.mul_(len(order) / len(selected))
                    groups = [[parameter for name, parameter in model.named_parameters() if name in functional.adaptable_names],
                              list(model.detector.parameters()), list(ssl.parameters())]
                    if not open_head or not condition.train_detector:
                        for parameter in model.detector.parameters():
                            parameter.grad = None
                    norms.append(torch.stack([torch.nn.utils.clip_grad_norm_(group, 1.0, error_if_nonfinite=True).cuda() for group in groups]))
                    before.append([[parameter.detach().clone() for parameter in group] for group in groups])
                for optimizer in optimizers:
                    optimizer.step()
                for model, ssl, old in zip(models, objectives, before, strict=True):
                    groups = [[parameter for name, parameter in model.named_parameters() if name in functional.adaptable_names],
                              list(model.detector.parameters()), list(ssl.parameters())]
                    changes.append(torch.stack([
                        sum((parameter.detach() - prior).float().square().sum()
                            for parameter, prior in zip(group, old_group, strict=True)).sqrt()
                        for group, old_group in zip(groups, old, strict=True)
                    ]))
                group_audit["outer_gradient_norm_sum"] = torch.stack(norms)
                group_audit["outer_clipped_count"] = (torch.stack(norms) > 1).long()
                group_audit["parameter_step_norm_sum"] = torch.stack(changes)
                for key, value in group_audit.items():
                    audit[key] = audit.get(key, 0) + value
                global_steps += 1
                epoch_steps += 1
                executed_groups += 1
                cursor = start + len(selected)
                progress = dict(phase="outer_step", epoch=epoch, patients_complete=cursor,
                                patients_total=len(order), records_complete=sum(len(patients[p]) for p in order[:cursor]),
                                step_s=time.perf_counter() - step_started, global_steps=global_steps,
                                detector_open=open_head, conditions={key: as_json(value) for key, value in group_audit.items()},
                                peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3)
                print(json.dumps(progress), flush=True)
                (args.output / "progress.json").write_text(json.dumps(progress, indent=2))
                requested_stop = stopping or (args.stop_after_groups and executed_groups >= args.stop_after_groups)
                if global_steps % args.checkpoint_every == 0 or cursor == len(order) or requested_stop:
                    save(epoch, cursor)
                if requested_stop:
                    return
            values = {key: as_json(value) for key, value in audit.items()}
            if (args.maximum_updates_per_patient is None
                    and values.get("windows") != [sum(workloads.values())] * len(selected_conditions)):
                raise RuntimeError(f"window coverage mismatch: {values.get('windows')}")
            for index, (condition, model, ssl) in enumerate(zip(selected_conditions, models, objectives, strict=True)):
                row = dict(epoch=epoch, condition=condition.name, patients=len(order), records=len(paths),
                           windows=values["windows"][index], class_weight_audit=class_audit,
                           base_bce=values["base_bce"][index],
                           post_bce=values["post_bce"][index], pre_bce=values["pre_bce"][index],
                           paired_windows=values["paired_windows"][index],
                           immediate_delta_bce=values["post_bce"][index] - values["pre_bce"][index],
                           gain=values["gain"][index], paired_damage=values["paired_damage"][index],
                           kl=values["kl"][index],
                           updates=values["updates"][index],
                           accepted_fraction=values["accepted"][index] / max(1, values["updates"][index]),
                           outer_steps=epoch_steps,
                           mean_preclip_gradient_norm=[value / epoch_steps for value in values["outer_gradient_norm_sum"][index]],
                           gradient_clipped_fraction=[value / epoch_steps for value in values["outer_clipped_count"][index]],
                           parameter_step_norm_sum=values["parameter_step_norm_sum"][index],
                           elapsed_s=elapsed_prior + time.perf_counter() - started)
                histories[index].append(row)
                run = args.output / condition.name
                checkpoint = dict(model=cpu_state(model), objective=cpu_state(ssl), objective_name=args.objective,
                                  difficulty=difficulty, mode="future",
                                  outer_scope="eds" if condition.train_detector else "es",
                                  v4_condition=condition.name,
                                  inner_rule="normalized", inner_kernel="packed", prefix_precision="fp32",
                                  prefix_microbatch=0, prefix_cuda_graph=True, inner_config=vars(inner_config),
                                  loss_scale=vars(loss_scale), source=str(source), objective_checkpoint=str(head),
                                  stage=contract["stage"], seed=args.seed, epoch=epoch,
                                  created_utc=datetime.now(UTC).isoformat(), training_contract=contract,
                                  history=histories[index])
                atomic_save(checkpoint, run / f"epoch_{epoch:02d}.pt")
                (run / "history.json").write_text(json.dumps(histories[index], indent=2))
            epoch += 1
            cursor, epoch_steps, audit = 0, 0, {}
            save(epoch, 0)
        (args.output / "complete.json").write_text(json.dumps(
            dict(status="training_complete", epochs=args.epochs,
                 conditions=[condition.name for condition in selected_conditions], contract=contract), indent=2))
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps(
            dict(error=repr(error), epoch=epoch, last_completed_patient=cursor,
                 checkpoint_exists=(args.output / "last.pt").exists()), indent=2))
        raise


if __name__ == "__main__":
    main()
