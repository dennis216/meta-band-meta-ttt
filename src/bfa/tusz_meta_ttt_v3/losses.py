from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from bfa.tusz_meta_ttt_v2.update import parameter_block


@dataclass(frozen=True)
class V3Condition:
    name: str
    alignment: bool
    damage: bool


CONDITIONS = (
    V3Condition("b0", False, False),
    V3Condition("b1", True, False),
    V3Condition("b2", False, True),
    V3Condition("b3", True, True),
)


@dataclass(frozen=True)
class PenaltyScale:
    alignment: float = 1.0
    damage: float = 1.0
    alignment_coefficient: float = 0.1
    damage_coefficient: float = 1.0


def high_score_reference(probabilities: Path, *, quantile: float = 0.9):
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    columns = ["patient_id", "session_id", "montage", "record_id", "label", "source_frozen"]
    frame = pd.read_parquet(probabilities, columns=columns)
    background = frame.loc[frame["label"] <= 0, "source_frozen"].to_numpy()
    if not len(background) or not np.isfinite(background).all():
        raise ValueError("reference probabilities lack finite background scores")
    threshold = float(np.quantile(background, quantile))
    lookup = {}
    for key, group in frame.groupby(["patient_id", "session_id", "montage", "record_id"], sort=False):
        lookup[tuple(key)] = group["source_frozen"].to_numpy(dtype=np.float32)
    return threshold, lookup


def path_key(path: Path) -> tuple[str, str, str, str]:
    path = Path(path)
    return path.parts[-4], path.parts[-3], path.parts[-2], path.stem


def weighted_group_sums(values, labels, mass, high_score):
    positive = labels > 0
    background = ~positive
    high = background & high_score
    return {
        "seizure": (values * mass * positive).sum(1),
        "background": (values * mass * background).sum(1),
        "high_background": (values * mass * high).sum(1),
    }


def damage_components(pre_bce, post_bce, labels, mass, high_score):
    damage = torch.relu(post_bce - pre_bce.detach())
    return weighted_group_sums(damage, labels, mass, high_score)


def actual_update_alignment(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    target_gradients: Mapping[str, torch.Tensor],
    *,
    valid: torch.Tensor,
    mass: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Weighted per-lane cost aligning the accepted update with -task gradient."""
    lanes = next(iter(before.values())).shape[0]
    if valid.shape != (lanes,) or mass.shape != (lanes,):
        raise ValueError("valid and mass must have one value per lane")
    costs = before[next(iter(before))].new_zeros(lanes)
    blocks = sorted({parameter_block(name) for name in before})
    for block in blocks:
        names = [name for name in before if parameter_block(name) == block]
        update = torch.cat([(after[name] - before[name]).reshape(lanes, -1) for name in names], 1)
        descent = torch.cat([-target_gradients[name].detach().reshape(lanes, -1) for name in names], 1)
        dot = (update * descent).sum(1)
        update_sq = update.square().sum(1)
        descent_sq = descent.square().sum(1)
        # A missing class gives an exactly zero target gradient. sqrt(0) has
        # an infinite derivative and poisons double backward even when a later
        # torch.where masks the lane. Smooth both norms before forming cosine.
        denominator = (update_sq + epsilon ** 2).sqrt() * (descent_sq + epsilon ** 2).sqrt()
        finite = (torch.isfinite(dot) & torch.isfinite(denominator)
                  & (update_sq > epsilon ** 2) & (descent_sq > epsilon ** 2))
        cosine = torch.where(finite, dot / denominator, torch.zeros_like(dot))
        costs = costs + (1.0 - cosine)
        valid = valid & finite
    costs = costs / max(1, len(blocks))
    return (costs * mass * valid).sum()


def condition_objective(post_bce, alignment, damage, condition: V3Condition, scale: PenaltyScale):
    result = post_bce
    if condition.alignment:
        result = result + scale.alignment_coefficient * scale.alignment * alignment
    if condition.damage:
        result = result + scale.damage_coefficient * scale.damage * damage
    return result
