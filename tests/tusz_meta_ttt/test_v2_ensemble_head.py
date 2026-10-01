import torch

from bfa.tusz_meta_ttt_v2.batched import EnsembleHead


def test_condition_heads_match_independent_gradients_and_lane_order():
    torch.manual_seed(3407)
    heads=[torch.nn.Linear(5,3) for _ in range(4)]
    heads[0].requires_grad_(False)
    inputs=torch.randn(12,7,5,requires_grad=True)
    ensemble=EnsembleHead(heads)
    actual=ensemble(inputs)
    expected=torch.stack([head(inputs[i*3+j]) for i,head in enumerate(heads) for j in range(3)])
    torch.testing.assert_close(actual,expected)
    params=[inputs,*[p for head in heads for p in head.parameters() if p.requires_grad]]
    ga=torch.autograd.grad(actual.square().sum(),params)
    ge=torch.autograd.grad(expected.square().sum(),params)
    for a,b in zip(ga,ge,strict=True): torch.testing.assert_close(a,b)


def test_condition_head_change_does_not_affect_other_conditions():
    heads=[torch.nn.Linear(5,3) for _ in range(4)]
    ensemble=EnsembleHead(heads)
    inputs=torch.randn(8,7,5)
    before=ensemble(inputs)
    with torch.no_grad(): heads[1].weight.add_(1.)
    after=ensemble(inputs)
    torch.testing.assert_close(before[:2],after[:2],atol=0,rtol=0)
    torch.testing.assert_close(before[4:],after[4:],atol=0,rtol=0)
    assert not torch.equal(before[2:4],after[2:4])
