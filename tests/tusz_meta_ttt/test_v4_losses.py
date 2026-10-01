from __future__ import annotations

import torch

from bfa.tusz_meta_ttt_v4.losses import bernoulli_kl_from_logits


def test_gain_and_shared_parameter_gradient_are_zero_when_update_is_disabled():
    parameter = torch.tensor(0.7, dtype=torch.double, requires_grad=True)
    label = torch.tensor(1.0, dtype=torch.double)
    pre = torch.nn.functional.binary_cross_entropy_with_logits(parameter, label)
    post = torch.nn.functional.binary_cross_entropy_with_logits(parameter, label)
    gain = post - pre
    gradient = torch.autograd.grad(gain, parameter)[0]
    assert gain.item() == 0.0
    assert gradient.item() == 0.0


def test_non_detached_pre_cancels_shared_direct_gradient():
    parameter = torch.tensor(-0.4, dtype=torch.double, requires_grad=True)
    update = torch.tensor(0.1, dtype=torch.double)
    label = torch.tensor(1.0, dtype=torch.double)
    post = torch.nn.functional.binary_cross_entropy_with_logits(parameter + update, label)
    pre = torch.nn.functional.binary_cross_entropy_with_logits(parameter, label)
    full = torch.autograd.grad(post - pre, parameter, retain_graph=True)[0]
    detached = torch.autograd.grad(post - pre.detach(), parameter)[0]
    assert abs(full) < abs(detached)


def test_bernoulli_kl_is_zero_and_stationary_at_reference_logit():
    reference = torch.tensor([0.2, 0.7], dtype=torch.double)
    logits = torch.logit(reference).requires_grad_()
    loss = bernoulli_kl_from_logits(reference, logits).sum()
    gradient = torch.autograd.grad(loss, logits)[0]
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-12, rtol=0)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)


def test_paired_damage_zero_exactly_without_bce_increase():
    pre = torch.tensor([1.0, 1.0, 2.0])
    post = torch.tensor([0.5, 1.0, 2.5])
    damage = torch.relu(post - pre)
    torch.testing.assert_close(damage, torch.tensor([0.0, 0.0, 0.5]))

