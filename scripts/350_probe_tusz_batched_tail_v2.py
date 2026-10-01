"""Isolated throughput/independence probe; not a production training entry point."""
import argparse
import json
import time
from pathlib import Path

import torch

from bfa.tusz_meta_ttt_v2.functional import SplitFunctionalTUSZModel
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.runtime import load_source
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig, packed_normalized_inner_step, parameter_block

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/reports/tusz_meta_ttt_v2'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lanes',type=int,default=4)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sustain-seconds',type=float,default=0)
    args=parser.parse_args()
    torch.manual_seed(3407)
    model=load_source(ROOT/'outputs/reports/tusz_meta_ttt_v1/runs/supervised/development/s1_seed3407_check0.25/best.pt',detector_trainable=True)
    functional=SplitFunctionalTUSZModel(model,prefix_cuda_graph=True)
    objective=build_objective('mask',5).cuda().eval()
    objective.load_state_dict(torch.load(OUT/'runs/ssl/development/mask_5_seed3407/epoch_02.pt',map_location='cpu',weights_only=False)['objective'])
    c=json.loads((OUT/'calibration/mask_5_seed3407/gradient_calibration.json').read_text())
    config=InnerStepConfig(c['selected_relative_step'],{int(k):v for k,v in c['block_reference_norms'].items()},{int(k):v for k,v in c['block_gradient_medians'].items()})
    lanes=args.lanes
    x=torch.randn(lanes,15,16,10,200,device='cuda')*.1
    seeds=[3407+i*100 for i in range(lanes)]
    # Only the frozen prefixes are shared. Each lane has independent fast weights.
    prefix=torch.stack([functional.prefix(v) for v in x])
    masks=torch.stack([objective._mask(x[i],seeds[i]) for i in range(lanes)])
    masked=x.masked_fill(masks[:,:,None,:,None],0)
    masked_prefix=torch.stack([functional.prefix(v) for v in masked])
    indexes=masks.long().argsort(dim=-1,descending=True,stable=True)
    spectrum=torch.fft.rfft(x,dim=-1,norm='forward').abs().square()[...,1:76]

    def select(value,idx):
        return value.gather(3,idx[:,:,None,:,None].expand(-1,-1,16,-1,75)).flatten(2)

    scale=select(spectrum,indexes[:,:,5:]).median(-1).values.clamp_min(objective.scale_floor)
    target=select(torch.log1p(spectrum/scale[:,:,None,None,None]),indexes[:,:,:5])
    initial=functional.initial_fast_parameters(detach=False)
    names=list(initial)
    blocks={b:[n for n in names if parameter_block(n)==b] for b in (10,11)}
    vf=torch.vmap(functional.features_from_prefix,in_dims=(0,0),randomness='error')
    # BatchedTensor may hide requires_grad from MHA's inference fastpath check.
    # Use the differentiable math path explicitly for the second-order probe.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)

    def ssl(weights):
        features=vf(masked_prefix,weights)
        pred=select(objective.head(features),indexes[:,:,:5])
        return torch.nn.functional.smooth_l1_loss(pred,target,reduction='none').mean((1,2))

    def batched_step(weights,record_initial):
        before=ssl(weights)
        gradients=torch.autograd.grad(before.sum(),tuple(weights.values()),create_graph=True)
        gradient=dict(zip(names,gradients,strict=True))
        directions={}
        slope=before.new_zeros(lanes)
        for b,ns in blocks.items():
            g=torch.cat([gradient[n].reshape(lanes,-1) for n in ns],1)
            norm=g.square().sum(1).sqrt()
            if not bool(torch.isfinite(norm).all()) or bool((norm<config.degenerate_fraction*config.block_gradient_medians[b]).any()):
                raise RuntimeError('Probe encountered invalid/degenerate lane; production fallback required')
            d=-config.relative_step*config.block_reference_norms[b]*g/(norm.square()+(config.epsilon_fraction*config.block_gradient_medians[b])**2).sqrt()[:,None]
            slope=slope+(g*d).sum(1)
            for n,piece in zip(ns,d.split([weights[n][0].numel() for n in ns],dim=1),strict=True):
                directions[n]=piece.reshape_as(weights[n])
        accepted=torch.zeros(lanes,device='cuda',dtype=torch.bool)
        result=weights
        for scale_value in config.trial_scales:
            trial={n:weights[n]+scale_value*directions[n] for n in names}
            drift_ok=torch.ones_like(accepted)
            for b,ns in blocks.items():
                drift=torch.cat([(trial[n]-record_initial[n]).reshape(lanes,-1) for n in ns],1).square().sum(1).sqrt()
                drift_ok=drift_ok & (drift<=config.maximum_record_drift*config.block_reference_norms[b]*(1+1e-4))
            after=ssl(trial).detach()
            take=(~accepted)&drift_ok&torch.isfinite(after)&(after<=before.detach()+config.armijo*scale_value*slope.detach())
            result={n:torch.where(take.reshape(lanes,*([1]*(weights[n].ndim-1))),trial[n],result[n]) for n in names}
            accepted=accepted|take
            if bool(accepted.all()): break
        return result,accepted

    def batched():
        weights={n:p.unsqueeze(0).expand(lanes,*p.shape) for n,p in initial.items()}
        origin=weights
        for _ in range(4): weights,accepted=batched_step(weights,origin)
        logits=model.detector(vf(prefix,weights).mean(2).flatten(2)).squeeze(-1)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits),reduction='sum')/15
        grads=torch.autograd.grad(loss,(*initial.values(),*model.detector.parameters(),*objective.parameters()),allow_unused=True)
        return logits.detach(),[None if g is None else g.detach() for g in grads],accepted

    def serial():
        losses=[]
        logits=[]
        for i in range(lanes):
            prepared=objective.prepare_loss(x[i],prefix_fn=functional.prefix,transform_seed=seeds[i])
            weights=initial
            for _ in range(4):
                result=packed_normalized_inner_step(weights,loss_fn=lambda p:prepared(lambda z:functional.features_from_prefix(z,p)),record_initial=initial,config=config,create_graph=True)
                weights=result.parameters
                assert result.accepted
            z=functional.logits_from_prefix(prefix[i],weights)
            logits.append(z.detach())
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(z,torch.zeros_like(z)))
        grads=torch.autograd.grad(sum(losses),(*initial.values(),*model.detector.parameters(),*objective.parameters()),allow_unused=True)
        return torch.stack(logits),[None if g is None else g.detach() for g in grads]

    a,b=serial(),batched()
    torch.testing.assert_close(a[0],b[0],rtol=1e-4,atol=1e-5)
    max_error=0.
    for old,new in zip(a[1],b[1],strict=True):
        if old is None or new is None: assert old is None and new is None
        else:
            torch.testing.assert_close(old,new,rtol=3e-4,atol=3e-5)
            max_error=max(max_error,float((old-new).abs().max()))
    elapsed={'serial':[],'batched':[]}
    for repeat in range(3):
        for name,run in ([('serial',serial),('batched',batched)] if repeat%2==0 else [('batched',batched),('serial',serial)]):
            torch.cuda.synchronize()
            start=time.perf_counter()
            run()
            torch.cuda.synchronize()
            elapsed[name].append(time.perf_counter()-start)
    sustained_start=time.perf_counter()
    sustained_runs=0
    while time.perf_counter()-sustained_start<args.sustain_seconds:
        batched()
        sustained_runs+=1
    torch.cuda.synchronize()
    sustained_elapsed=time.perf_counter()-sustained_start
    report=dict(lanes=lanes,passed=True,max_meta_gradient_error=max_error,elapsed_s=elapsed,
                sustained_runs=sustained_runs,sustained_elapsed_s=sustained_elapsed,
                sustained_inner_updates_per_s=sustained_runs*lanes*4/sustained_elapsed,
                scope='synthetic Mask four-update probe with cached prefixes; not complete-record throughput')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
