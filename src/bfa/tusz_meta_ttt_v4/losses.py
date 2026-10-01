from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class V4Condition:
    name: str
    paired_gain: bool
    train_detector: bool


CONDITIONS = (
    V4Condition("a", False, True),
    V4Condition("b", True, True),
    V4Condition("c", True, False),
)


@dataclass(frozen=True)
class LossScale:
    post: float = 1.0
    gain: float = 1.0
    paired_damage: float = 1.0
    kl: float = 1.0


def path_key(path: Path) -> tuple[str, str, str, str]:
    path = Path(path)
    return path.parts[-4], path.parts[-3], path.parts[-2], path.stem


def source_probability_lookup(probabilities: Path):
    columns = ["patient_id", "session_id", "montage", "record_id", "source_frozen"]
    frame = pd.read_parquet(probabilities, columns=columns)
    lookup = {
        tuple(key): group["source_frozen"].to_numpy(dtype=np.float32)
        for key, group in frame.groupby(columns[:4], sort=False)
    }
    return lookup


def bernoulli_kl_from_logits(reference_probability, logits, epsilon: float = 1e-6):
    reference = reference_probability.clamp(epsilon, 1 - epsilon)
    log_q = torch.nn.functional.logsigmoid(logits)
    log_one_minus_q = torch.nn.functional.logsigmoid(-logits)
    return reference * (reference.log() - log_q) + (1 - reference) * (
        torch.log1p(-reference) - log_one_minus_q
    )


def condition_objective(base, post, gain, damage, kl, condition: V4Condition, scale: LossScale):
    if condition.name == "cal_base":
        return base
    if condition.name == "cal_post":
        return base + post
    if condition.name == "cal_gain":
        return base + gain
    if condition.name == "cal_damage":
        return base + damage
    if condition.name == "cal_kl":
        return base + kl
    protected_base = base + scale.kl * kl
    if condition.paired_gain:
        return protected_base + scale.gain * gain + scale.paired_damage * damage
    return protected_base + scale.post * post
