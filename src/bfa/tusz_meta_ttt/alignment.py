"""Gradient-alignment utilities for the TUSZ Meta-TTT protocol.

The alignment target is used only by the Meta-training outer loss.  In
particular, the classification gradient is computed on a labelled query
episode during training and is never part of the deployment-time inner
update.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GradientAlignment:
    """Differentiable alignment values plus detached diagnostics."""

    cosine: torch.Tensor
    penalty: torch.Tensor
    dot: torch.Tensor
    ssl_norm: torch.Tensor
    classification_norm: torch.Tensor
    valid: bool

    @property
    def negative(self) -> bool:
        return self.valid and bool((self.cosine.detach() < 0).item())


def _flatten(
    gradients: Sequence[torch.Tensor | None],
    references: Sequence[torch.Tensor],
    *,
    preserve_graph: bool,
) -> torch.Tensor:
    if len(gradients) != len(references):
        raise ValueError("gradient and reference sequences must have equal length")
    pieces = []
    for gradient, reference in zip(gradients, references, strict=True):
        if gradient is None:
            # A zero with a graph-connected term keeps the returned vector
            # differentiable when another parameter has a usable gradient.
            value = reference * 0.0 if preserve_graph else torch.zeros_like(reference)
        else:
            value = gradient if preserve_graph else gradient.detach()
        pieces.append(value.reshape(-1))
    if not pieces:
        raise ValueError("at least one gradient is required")
    return torch.cat(pieces)


def compute_gradient_alignment(
    ssl_gradients: Sequence[torch.Tensor | None],
    classification_gradients: Sequence[torch.Tensor | None],
    references: Sequence[torch.Tensor],
    *,
    inner_lr: float,
    norm_floor: float = 1.0e-12,
    eps: float = 1.0e-12,
) -> GradientAlignment:
    """Compute a cosine penalty with a label-free deployment boundary.

    ``ssl_gradients`` retain their graph so ``penalty`` can train the
    auxiliary objective.  The classification gradient is detached and is
    therefore only a target direction for the outer update.  A zero or
    non-finite norm produces a zero-valued penalty connected to the SSL graph;
    callers should record ``valid=False`` and exclude it from statistics.
    """

    ssl_vector = _flatten(ssl_gradients, references, preserve_graph=True)
    classification_vector = _flatten(
        classification_gradients, references, preserve_graph=False
    )
    ssl_norm = ssl_vector.norm()
    classification_norm = classification_vector.norm()
    dot = torch.dot(ssl_vector, classification_vector)
    finite = bool(
        torch.isfinite(ssl_norm).detach().item()
        and torch.isfinite(classification_norm).detach().item()
        and torch.isfinite(dot).detach().item()
    )
    valid = (
        finite
        and float(ssl_norm.detach()) > norm_floor
        and float(classification_norm.detach()) > norm_floor
    )
    if valid:
        cosine = dot / (
            ssl_norm.clamp_min(eps) * classification_norm.clamp_min(eps)
        )
        penalty = 1.0 - cosine
    else:
        # Do not inject a constant loss for an invalid episode, but retain a
        # graph connection so ``backward`` remains well-defined.
        cosine = torch.full(
            (), float("nan"), device=ssl_vector.device, dtype=ssl_vector.dtype
        )
        penalty = ssl_vector.sum() * 0.0
    return GradientAlignment(
        cosine=cosine,
        penalty=penalty,
        dot=dot,
        ssl_norm=ssl_norm,
        classification_norm=classification_norm,
        valid=valid,
    )
