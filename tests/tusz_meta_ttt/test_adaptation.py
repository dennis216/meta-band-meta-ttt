from __future__ import annotations

import torch
from torch import nn

from bfa.tusz_meta_ttt.adaptation import OnlineAdapter, module_hash
from bfa.tusz_meta_ttt.functional import FunctionalTUSZModel
from bfa.tusz_meta_ttt.objectives import SSLObjective


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(12)])

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

    def forward(self, x):
        return self.encoder(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.detector = nn.Linear(2, 1)

    def forward(self, x):
        return self.detector(self.backbone(x)).squeeze(-1)


class Energy(SSLObjective):
    def loss(self, signal, *, feature_fn, fast_parameters=None, transform_seed=0):
        del fast_parameters, transform_seed
        return feature_fn(signal).square().mean()


def test_zero_update_matches_frozen_and_source_is_unchanged():
    model = Model()
    functional = FunctionalTUSZModel(model)
    before = module_hash(model)
    adapter = OnlineAdapter(
        model,
        Energy(),
        feature_fn=functional.features,
        predict_fn=functional.logits,
        adaptable_parameters={name: parameter for name, parameter in model.named_parameters() if name in functional.adaptable_names},
        inner_lr=0.0,
    )
    signal = torch.randn(4, 2)
    frozen = model(signal)
    adapted, state = adapter.predict_then_update(signal, signal, adapter.reset(), observed_seconds=30, transform_seed=17)
    torch.testing.assert_close(adapted, frozen)
    assert state.updates == 1 and state.observed_seconds == 30
    assert module_hash(model) == before


def test_meta_gradient_reaches_objective_parameter():
    model = Model()
    functional = FunctionalTUSZModel(model)

    class ScaledEnergy(SSLObjective):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def loss(self, signal, *, feature_fn, fast_parameters=None, transform_seed=0):
            return self.scale * feature_fn(signal).square().mean()

    objective = ScaledEnergy()
    adapter = OnlineAdapter(
        model,
        objective,
        feature_fn=functional.features,
        predict_fn=functional.logits,
        adaptable_parameters={name: parameter for name, parameter in model.named_parameters() if name in functional.adaptable_names},
        inner_lr=1e-2,
    )
    signal = torch.randn(4, 2)
    _, state = adapter.predict_then_update(signal, signal, adapter.reset(), observed_seconds=30, transform_seed=17, create_graph=True)
    query = functional.logits(signal, state.fast_parameters).mean()
    gradient = torch.autograd.grad(query, objective.scale)[0]
    assert torch.isfinite(gradient) and gradient.abs() > 0
