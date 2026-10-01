import pytest
import torch

from bfa.tusz_meta_ttt_v2.batched import prepare_ssl_lanes
from bfa.tusz_meta_ttt_v2.objectives import build_objective


@pytest.mark.parametrize('name,difficulty',[('band',.5),('mask',5)])
def test_condition_transform_deduplication_preserves_meta_gradients(name,difficulty):
    from bfa.tusz_meta_ttt_v2.batched import EnsembleBandObjective,EnsembleMaskObjective
    torch.manual_seed(3407)
    objectives=[build_objective(name,difficulty).eval() for _ in range(2)]
    ensemble=(EnsembleBandObjective if name=='band' else EnsembleMaskObjective)(objectives)
    signal=torch.randn(2,2,16,10,200).repeat(2,1,1,1,1)*.1
    valid=torch.tensor([[0.,0.],[1.,0.],[1.,1.],[1.,0.]])
    seeds=[1,2,3,4]*2
    theta=torch.tensor(.7,requires_grad=True)
    full=prepare_ssl_lanes(ensemble,signal,valid,prefix_fn=lambda x:x,seeds=seeds)(lambda x:x*theta)
    ensemble.deduplicate_views=True
    compact=prepare_ssl_lanes(ensemble,signal,valid,prefix_fn=lambda x:x,prefix_unique_fn=lambda x:x,seeds=seeds)(lambda x:x*theta)
    torch.testing.assert_close(full,compact)
    parameters=(theta,*ensemble.head.parameters())
    first=[torch.autograd.grad(v.sum(),parameters,create_graph=True) for v in [full,compact]]
    for a,b in zip(*first,strict=True): torch.testing.assert_close(a,b)
    second=[torch.autograd.grad(sum(g.square().sum() for g in grads),parameters) for grads in first]
    for a,b in zip(*second,strict=True): torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)


@pytest.mark.parametrize('name,difficulty',[('band',.5),('mask',5)])
def test_padded_support_lanes_equal_independent_valid_windows(name,difficulty):
    torch.manual_seed(3407)
    objective=build_objective(name,difficulty).eval()
    signal=torch.randn(2,3,16,10,200)*.1
    valid=torch.tensor([[1.,1.,1.],[1.,0.,0.]])
    seeds=[3407,3408,3409,3410,3411,3412]
    theta=torch.tensor(.7,requires_grad=True)
    prepared=prepare_ssl_lanes(objective,signal,valid,prefix_fn=lambda x:x,seeds=seeds)
    actual=prepared(lambda z:z*theta)
    expected=[]
    for lane,count in [(0,3),(1,1)]:
        separate=objective.prepare_loss(signal[lane,:count],prefix_fn=lambda x:x,transform_seed=seeds[lane*3:lane*3+count])
        expected.append(separate(lambda z:z*theta))
    expected=torch.stack(expected)
    torch.testing.assert_close(actual,expected)
    parameters=(theta,*objective.head.parameters())
    ga=torch.autograd.grad(actual.sum(),parameters,create_graph=True)
    ge=torch.autograd.grad(expected.sum(),parameters,create_graph=True)
    for a,b in zip(ga,ge,strict=True): torch.testing.assert_close(a,b)
    ha=torch.autograd.grad(sum(g.square().sum() for g in ga),parameters)
    he=torch.autograd.grad(sum(g.square().sum() for g in ge),parameters)
    for a,b in zip(ha,he,strict=True): torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
