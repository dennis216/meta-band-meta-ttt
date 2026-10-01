from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from bfa.tusz_meta_ttt.objectives import SSLObjective


def module_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class AdaptationState:
    source_hash: str
    fast_parameters: dict[str, torch.Tensor]
    velocity: dict[str, torch.Tensor] | None = None
    updates: int = 0
    observed_seconds: float = 0.0

    def detached(self) -> AdaptationState:
        return AdaptationState(
            self.source_hash,
            {name: value.detach().requires_grad_(value.requires_grad) for name, value in self.fast_parameters.items()},
            None
            if self.velocity is None
            else {name: value.detach() for name, value in self.velocity.items()},
            self.updates,
            self.observed_seconds,
        )


class OnlineAdapter:
    """Predict first, then update fast weights from the observed support windows."""

    def __init__(
        self,
        source: nn.Module,
        objective: SSLObjective,
        *,
        feature_fn: Callable[[torch.Tensor, dict[str, torch.Tensor]], torch.Tensor],
        predict_fn: Callable[[torch.Tensor, dict[str, torch.Tensor]], torch.Tensor],
        adaptable_parameters: dict[str, nn.Parameter],
        inner_lr: float,
        inner_momentum: float = 0.0,
    ) -> None:
        self.source = source
        self.objective = objective
        self.feature_fn = feature_fn
        self.predict_fn = predict_fn
        self.adaptable_parameters = adaptable_parameters
        self.inner_lr = float(inner_lr)
        if not 0 <= inner_momentum < 1:
            raise ValueError("inner_momentum must be in [0, 1)")
        self.inner_momentum = float(inner_momentum)
        self.source_hash = module_hash(source)

    def reset(self) -> AdaptationState:
        return AdaptationState(
            self.source_hash,
            {name: value.detach().clone().requires_grad_(True) for name, value in self.adaptable_parameters.items()},
            {
                name: torch.zeros_like(value)
                for name, value in self.adaptable_parameters.items()
            },
        )

    def predict_then_update(
        self,
        prediction_windows: torch.Tensor,
        support_windows: torch.Tensor,
        state: AdaptationState,
        *,
        observed_seconds: float,
        transform_seed: int,
        create_graph: bool = False,
    ) -> tuple[torch.Tensor, AdaptationState]:
        if state.source_hash != self.source_hash or module_hash(self.source) != self.source_hash:
            raise RuntimeError("source model changed after adaptation state was created")
        logits = self.predict_fn(prediction_windows, state.fast_parameters)
        loss = self.objective.loss(
            support_windows,
            feature_fn=lambda x: self.feature_fn(x, state.fast_parameters),
            fast_parameters=state.fast_parameters,
            transform_seed=transform_seed,
        )
        gradients = torch.autograd.grad(
            loss, tuple(state.fast_parameters.values()), create_graph=create_graph, allow_unused=True
        )
        updated = {}
        velocity = {}
        for (name, value), gradient in zip(state.fast_parameters.items(), gradients, strict=True):
            if gradient is None:
                updated[name] = value
                velocity[name] = state.velocity[name] if state.velocity is not None else torch.zeros_like(value)
            elif torch.isfinite(gradient).all():
                previous = state.velocity[name] if state.velocity is not None else torch.zeros_like(gradient)
                velocity[name] = self.inner_momentum * previous + gradient
                updated[name] = value - self.inner_lr * velocity[name]
            else:
                updated[name] = value
                velocity[name] = state.velocity[name] if state.velocity is not None else torch.zeros_like(value)
        return logits, AdaptationState(
            state.source_hash,
            updated,
            velocity,
            state.updates + 1,
            state.observed_seconds + float(observed_seconds),
        )
