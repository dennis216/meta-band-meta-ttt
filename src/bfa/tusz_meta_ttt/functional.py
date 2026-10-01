from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.func import functional_call


class FunctionalTUSZModel:
    """Functional access to CBraMod blocks 10/11 without mutating the source."""

    def __init__(self, model: nn.Module, *, adapted_blocks: tuple[int, ...] = (10, 11)) -> None:
        self.model = model
        prefixes = tuple(f"backbone.encoder.layers.{index}." for index in adapted_blocks)
        self.base_parameters = OrderedDict(model.named_parameters())
        self.base_buffers = OrderedDict(model.named_buffers())
        self.adaptable_names = tuple(
            name for name in self.base_parameters if name.startswith(prefixes)
        )
        if not self.adaptable_names:
            raise ValueError("no parameters matched the requested CBraMod blocks")

    def initial_fast_parameters(self, *, detach: bool = True) -> dict[str, torch.Tensor]:
        if not detach:
            return {name: self.base_parameters[name] for name in self.adaptable_names}
        return {
            name: self.base_parameters[name].detach().clone().requires_grad_(True)
            for name in self.adaptable_names
        }

    def _state(self, fast_parameters: dict[str, torch.Tensor]) -> tuple[dict, dict]:
        unknown = set(fast_parameters) - set(self.adaptable_names)
        if unknown:
            raise KeyError(f"unknown fast parameters: {sorted(unknown)}")
        parameters = {**self.base_parameters, **fast_parameters}
        return parameters, self.base_buffers

    def logits(self, signal: torch.Tensor, fast_parameters: dict[str, torch.Tensor]) -> torch.Tensor:
        return functional_call(self.model, self._state(fast_parameters), (signal,))

    def features(self, signal: torch.Tensor, fast_parameters: dict[str, torch.Tensor]) -> torch.Tensor:
        parameters, buffers = self._state(fast_parameters)
        return functional_call(
            self.model.backbone,
            (
                {name.removeprefix("backbone."): value for name, value in parameters.items() if name.startswith("backbone.")},
                {name.removeprefix("backbone."): value for name, value in buffers.items() if name.startswith("backbone.")},
            ),
            (signal,),
            strict=False,
        )
