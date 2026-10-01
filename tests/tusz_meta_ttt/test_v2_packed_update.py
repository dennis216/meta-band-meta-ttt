from dataclasses import replace

import pytest
import torch

from bfa.tusz_meta_ttt_v2.update import InnerStepConfig, normalized_inner_step, packed_normalized_inner_step


@pytest.mark.parametrize('case', ['accepted', 'drift', 'degenerate', 'nonfinite', 'unused', 'zero'])
def test_packed_update_matches_reference_decision_and_meta_gradient(case):
    torch.manual_seed(3407)
    parameters = {f'backbone.encoder.layers.{b}.{name}': torch.randn(shape, requires_grad=True)
                  for b in (10, 11) for name, shape in [('weight', (4, 3)), ('bias', (4,))]}
    phi = torch.tensor(0.7, requires_grad=True)
    config = InnerStepConfig(1e-4, {10: 3., 11: 3.}, {10: 1., 11: 1.})
    if case == 'drift': config = replace(config, maximum_record_drift=1e-9)
    if case == 'zero': config = replace(config, relative_step=0.)

    def ssl(state):
        values = list(state.values())[:-1] if case == 'unused' else state.values()
        loss = sum(((v * phi).sin() - 0.2).square().sum() for v in values)
        if case == 'degenerate': loss = loss * 0
        if case == 'nonfinite': loss = loss * float('nan')
        return loss

    before = {n: p.detach().clone() for n, p in parameters.items()}
    outputs = []
    for step in [normalized_inner_step, packed_normalized_inner_step]:
        result = step(parameters, loss_fn=ssl, record_initial=parameters, config=config, create_graph=True)
        outer = sum(v.square().sum() for v in result.parameters.values())
        gradients = torch.autograd.grad(outer, (*parameters.values(), phi), allow_unused=True)
        outputs.append((result, gradients))
    old, new = outputs
    assert (old[0].accepted, old[0].reason, old[0].trial_scale) == (new[0].accepted, new[0].reason, new[0].trial_scale)
    for name in parameters:
        torch.testing.assert_close(parameters[name], before[name], rtol=0, atol=0)
        torch.testing.assert_close(old[0].parameters[name], new[0].parameters[name], rtol=1e-6, atol=1e-7)
    for a, b in zip(old[1], new[1], strict=True):
        if a is None or b is None:
            assert a is None and b is None
        else:
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_packed_meta_gradient_finite_difference_with_multicoordinate_blocks():
    config = InnerStepConfig(0.001, {10: 3., 11: 3.}, {10: 1., 11: 1.})
    def outer(value):
        phi = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        p = {f'backbone.encoder.layers.{b}.weight': torch.tensor([.4, .8, 1.2], dtype=torch.float64, requires_grad=True) for b in (10, 11)}
        result = packed_normalized_inner_step(p, loss_fn=lambda s: sum(((v*phi).sin()-.1).square().sum() for v in s.values()), record_initial=p, config=config, create_graph=True)
        assert result.accepted
        loss = sum(v.square().sum() for v in result.parameters.values())
        return loss, phi
    loss, phi = outer(.7)
    analytical = torch.autograd.grad(loss, phi)[0].item()
    numerical = (outer(.7001)[0].item() - outer(.6999)[0].item()) / .0002
    assert analytical == pytest.approx(numerical, rel=0.01, abs=1e-5)


@pytest.mark.parametrize('positions', [3, 5, 7])
def test_mask_fixed_gather_matches_boolean_reference_through_second_derivative(positions):
    from bfa.tusz_meta_ttt_v2.objectives import MaskObjectiveV2
    torch.manual_seed(3407)
    objective = MaskObjectiveV2(positions).eval()
    signal = torch.randn(2, 16, 10, 200)
    theta = torch.tensor(.7, requires_grad=True)
    direct = objective.loss(signal, feature_fn=lambda x: x.sin() * theta, transform_seed=[3407, 3408])
    prepared = objective.prepare_loss(signal, prefix_fn=lambda x: x, transform_seed=[3407, 3408])
    gathered = prepared(lambda x: x.sin() * theta)
    torch.testing.assert_close(direct, gathered, atol=0, rtol=0)
    targets = (theta, objective.head.weight, objective.head.bias)
    old = torch.autograd.grad(direct, targets, create_graph=True)
    new = torch.autograd.grad(gathered, targets, create_graph=True)
    for a, b in zip(old, new, strict=True):
        torch.testing.assert_close(a, b)
    old_second = torch.autograd.grad(sum(g.square().sum() for g in old), targets)
    new_second = torch.autograd.grad(sum(g.square().sum() for g in new), targets)
    for a, b in zip(old_second, new_second, strict=True):
        torch.testing.assert_close(a, b)
