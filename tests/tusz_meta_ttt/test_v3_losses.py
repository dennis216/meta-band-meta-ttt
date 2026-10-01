from pathlib import Path

import pytest
import torch

from bfa.tusz_meta_ttt_v3.losses import (
    CONDITIONS,
    PenaltyScale,
    actual_update_alignment,
    condition_objective,
    damage_components,
)


def test_zero_penalties_exactly_reproduce_post_bce_and_gradient():
    post = torch.tensor(1.7, requires_grad=True)
    alignment = torch.tensor(3.0, requires_grad=True)
    damage = torch.tensor(4.0, requires_grad=True)
    scale = PenaltyScale(alignment=0.0, damage=0.0)
    for condition in CONDITIONS:
        value = condition_objective(post, alignment, damage, condition, scale)
        gradient = torch.autograd.grad(value, post, retain_graph=True)[0]
        assert float(value) == pytest.approx(1.7)
        assert float(gradient) == pytest.approx(1.0)


def test_alignment_uses_actual_update_and_detached_descent_target():
    name = "backbone.encoder.layers.10.weight"
    before = {name: torch.tensor([[0.0, 0.0]], requires_grad=True)}
    direction = torch.tensor([[1.0, 0.0]], requires_grad=True)
    after = {name: before[name] + direction}
    target = {name: torch.tensor([[-2.0, 0.0]], requires_grad=True)}
    loss = actual_update_alignment(before, after, target, valid=torch.tensor([True]), mass=torch.tensor([1.0]))
    assert float(loss) == pytest.approx(0.0, abs=1e-7)
    loss.backward()
    assert target[name].grad is None


def test_alignment_penalizes_opposite_update_direction():
    name = "backbone.encoder.layers.10.weight"
    before = {name: torch.zeros(1, 2)}
    after = {name: torch.tensor([[-1.0, 0.0]], requires_grad=True)}
    target = {name: torch.tensor([[-2.0, 0.0]])}
    loss = actual_update_alignment(before, after, target, valid=torch.tensor([True]), mass=torch.tensor([1.0]))
    assert float(loss) == pytest.approx(2.0)


def test_damage_is_positive_only_for_post_update_harm_and_high_tail_overlaps_background():
    pre = torch.tensor([[0.4, 0.7, 0.5]])
    post = torch.tensor([[0.6, 0.2, 0.8]], requires_grad=True)
    labels = torch.tensor([[0.0, 1.0, 0.0]])
    mass = torch.tensor([[0.2, 0.3, 0.5]])
    high = torch.tensor([[False, False, True]])
    values = damage_components(pre, post, labels, mass, high)
    assert float(values["seizure"]) == 0.0
    assert float(values["background"]) == pytest.approx(0.19)
    assert float(values["high_background"]) == pytest.approx(0.15)
    sum(values.values()).backward()
    assert post.grad is not None and pre.grad is None
