import torch

from bfa.tusz_meta_ttt_v2.batched import normalized_lane_step
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig, packed_normalized_inner_step


def test_inactive_lane_has_no_second_order_nan_and_does_not_change_active_lane():
    config=InnerStepConfig(1e-4,{10:1.,11:1.},{10:1.,11:1.})
    source={f'backbone.encoder.layers.{b}.weight':torch.tensor([.4,.8],requires_grad=True) for b in (10,11)}
    phi=torch.tensor(.7,requires_grad=True)
    parameters={n:p.unsqueeze(0).expand(2,-1) for n,p in source.items()}
    mask=torch.tensor([1.,0.])
    loss=lambda state:sum(((p*phi).sin()-.1).square().sum(1) for p in state.values())*mask
    result=normalized_lane_step(parameters,loss_fn=loss,record_initial=parameters,config=config)
    assert result.accepted.tolist()==[True,False]
    actual_loss=sum(p[0].square().sum() for p in result.parameters.values())
    actual_grad=torch.autograd.grad(actual_loss,(*source.values(),phi))
    reference=packed_normalized_inner_step(source,loss_fn=lambda state:sum(((p*phi).sin()-.1).square().sum() for p in state.values()),record_initial=source,config=config,create_graph=True)
    expected_loss=sum(p.square().sum() for p in reference.parameters.values())
    expected_grad=torch.autograd.grad(expected_loss,(*source.values(),phi))
    for a,b in zip(actual_grad,expected_grad,strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    for n in source:
        torch.testing.assert_close(result.parameters[n][1],source[n],rtol=0,atol=0)
def test_nonfinite_inner_graph_is_discarded_without_poisoning_outer():
    import torch
    from bfa.tusz_meta_ttt_v2.batched import normalized_lane_step
    from bfa.tusz_meta_ttt_v2.update import InnerStepConfig
    parameters={f'backbone.encoder.layers.{b}.weight':torch.tensor([[0.],[1.]],requires_grad=True) for b in (10,11)}
    result=normalized_lane_step(parameters,loss_fn=lambda p:sum(v.sqrt().sum(1) for v in p.values()),record_initial=parameters,
        config=InnerStepConfig(1e-5,{10:1.,11:1.},{10:1.,11:1.}))
    assert result.nonfinite_batch_rejection
    assert not result.accepted.any()
    assert all(result.parameters[n] is p for n,p in parameters.items())
    loss=sum((p+1).square().sum() for p in result.parameters.values())
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in parameters.values())


def test_stopped_encoder_hessian_preserves_direct_and_ssl_head_gradients():
    parameters={f'backbone.encoder.layers.{b}.weight':torch.tensor([[.4,.8]],requires_grad=True) for b in (10,11)}
    phi=torch.tensor(.7,requires_grad=True)
    config=InnerStepConfig(1e-4,{10:1.,11:1.},{10:1.,11:1.},maximum_record_drift=.5)
    loss=lambda state:sum((p*phi).square().sum(1) for p in state.values())
    result=normalized_lane_step(parameters,loss_fn=loss,record_initial=parameters,config=config,learning_rate=.1,stop_encoder_through_inner=True)
    assert result.accepted.all()
    outer=sum(p.square().sum() for p in result.parameters.values())
    gradients=torch.autograd.grad(outer,(*parameters.values(),phi))
    expected_phi=phi.new_zeros(())
    for (name,source),actual in zip(parameters.items(),gradients[:-1],strict=True):
        post=source.detach()*(1-.2*phi.detach().square())
        torch.testing.assert_close(result.parameters[name],post)
        torch.testing.assert_close(actual,2*post)
        expected_phi+=(-.4*phi.detach()*source.detach()*2*post).sum()
    torch.testing.assert_close(gradients[-1],expected_phi)
    assert gradients[-1].abs()>0
