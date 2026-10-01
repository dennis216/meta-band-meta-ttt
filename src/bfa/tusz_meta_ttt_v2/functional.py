from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.func import functional_call


class SplitFunctionalTUSZModel:
    """Run the frozen CBraMod prefix once and functionally adapt blocks 10/11."""

    def __init__(
        self,
        model: nn.Module,
        split_block: int = 10,
        *,
        prefix_precision: str = "fp32",
        prefix_microbatch: int | None = None,
        prefix_cuda_graph: bool = False,
    ) -> None:
        self.model = model
        self.split_block = split_block
        if prefix_precision not in {"fp32", "bf16"}:
            raise ValueError("prefix_precision must be fp32 or bf16")
        self.prefix_precision = prefix_precision
        if prefix_microbatch is not None and prefix_microbatch <= 0:
            raise ValueError("prefix_microbatch must be positive or None")
        self.prefix_microbatch = prefix_microbatch
        self.prefix_cuda_graph = prefix_cuda_graph
        self._prefix_graphs = {}
        self.base_parameters = OrderedDict(model.named_parameters())
        prefixes = tuple(
            f"backbone.encoder.layers.{index}." for index in range(split_block, 12)
        )
        self.adaptable_names = tuple(
            name for name in self.base_parameters if name.startswith(prefixes)
        )
        if not self.adaptable_names:
            raise ValueError("no parameters found in the adaptable tail")

    def initial_fast_parameters(self, *, detach: bool = True) -> dict[str, torch.Tensor]:
        if detach:
            return {
                name: self.base_parameters[name].detach().clone().requires_grad_(True)
                for name in self.adaptable_names
            }
        return {name: self.base_parameters[name] for name in self.adaptable_names}

    def prefix(self, signal: torch.Tensor) -> torch.Tensor:
        if self.prefix_microbatch is not None and signal.shape[0] > self.prefix_microbatch:
            return torch.cat(
                [
                    self._prefix_dispatch(part)
                    for part in signal.split(self.prefix_microbatch, dim=0)
                ],
                dim=0,
            )
        return self._prefix_dispatch(signal)

    def _prefix_dispatch(self, signal: torch.Tensor) -> torch.Tensor:
        if not self.prefix_cuda_graph or not signal.is_cuda:
            return self._prefix_once(signal)
        key = (tuple(signal.shape), signal.dtype, signal.device)
        if key not in self._prefix_graphs:
            # Bound static memory use; uncommon partial-chunk shapes use eager.
            if len(self._prefix_graphs) >= 4:
                return self._prefix_once(signal)
            static_input = signal.detach().clone()
            stream = torch.cuda.Stream(device=signal.device)
            stream.wait_stream(torch.cuda.current_stream(signal.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._prefix_once(static_input)
            torch.cuda.current_stream(signal.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                static_output = self._prefix_once(static_input)
            self._prefix_graphs[key] = (graph, static_input, static_output)
        graph, static_input, static_output = self._prefix_graphs[key]
        static_input.copy_(signal)
        graph.replay()
        # Tail autograd saves prefix values across four updates. Never expose
        # replay-owned storage that the next chunk will overwrite.
        return static_output.clone()

    def _prefix_once(self, signal: torch.Tensor) -> torch.Tensor:
        use_bf16 = self.prefix_precision == "bf16" and signal.is_cuda
        with torch.no_grad(), torch.autocast(
            device_type=signal.device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            value = self.model.backbone.patch_embedding(signal)
            for layer in self.model.backbone.encoder.layers[: self.split_block]:
                value = layer(value)
        return value.float().detach()

    def features_from_prefix(
        self, prefix_features: torch.Tensor, fast_parameters: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        value = prefix_features
        for index in range(self.split_block, 12):
            layer = self.model.backbone.encoder.layers[index]
            stem = f"backbone.encoder.layers.{index}."
            parameters = {
                name.removeprefix(stem): tensor
                for name, tensor in fast_parameters.items()
                if name.startswith(stem)
            }
            value = functional_call(layer, parameters, (value,), strict=True)
        return self.model.backbone.proj_out(value)

    def features(
        self, signal: torch.Tensor, fast_parameters: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.features_from_prefix(self.prefix(signal), fast_parameters)

    def logits_from_prefix(
        self, prefix_features: torch.Tensor, fast_parameters: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        features = self.features_from_prefix(prefix_features, fast_parameters)
        pooled = features.mean(dim=1).flatten(1)
        return self.model.detector(pooled).squeeze(-1)

    def logits(self, signal: torch.Tensor, fast_parameters: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.logits_from_prefix(self.prefix(signal), fast_parameters)
