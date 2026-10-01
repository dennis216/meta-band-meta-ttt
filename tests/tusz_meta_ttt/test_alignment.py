from __future__ import annotations

import torch

from bfa.tusz_meta_ttt.alignment import compute_gradient_alignment


def test_cosine_penalty_is_differentiable_only_through_ssl_gradient():
    reference = torch.zeros(2, requires_grad=True)
    ssl = torch.tensor([1.0, 0.0], requires_grad=True)
    classification = torch.tensor([1.0, 1.0], requires_grad=True)

    result = compute_gradient_alignment(
        [ssl], [classification], [reference], inner_lr=1.0
    )

    torch.testing.assert_close(result.cosine, torch.tensor(2.0**-0.5))
    assert result.valid and not result.negative
    result.penalty.backward()
    assert ssl.grad is not None and torch.isfinite(ssl.grad).all()
    assert classification.grad is None


def test_zero_norm_returns_zero_penalty_and_invalid_diagnostic():
    reference = torch.zeros(2, requires_grad=True)
    ssl = torch.zeros(2, requires_grad=True)
    classification = torch.tensor([1.0, 0.0], requires_grad=True)

    result = compute_gradient_alignment(
        [ssl], [classification], [reference], inner_lr=1.0
    )

    assert not result.valid
    torch.testing.assert_close(result.penalty, torch.tensor(0.0))
    result.penalty.backward()
    assert ssl.grad is not None and torch.equal(ssl.grad, torch.zeros_like(ssl))

