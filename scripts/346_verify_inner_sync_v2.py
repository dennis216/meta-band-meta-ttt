"""Paired real-model verification of fewer inner-update host synchronizations."""
from __future__ import annotations

import importlib.util
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bfa.tusz_meta_ttt.dataset import load_cached_arrays
from bfa.tusz_meta_ttt_v2.functional import SplitFunctionalTUSZModel
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.runtime import load_source
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig, normalized_inner_step, packed_normalized_inner_step

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/reports/tusz_meta_ttt_v2'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--packed', action='store_true')
    args = parser.parse_args()
    candidate_step = packed_normalized_inner_step if args.packed else normalized_inner_step
    reference_path = OUT / 'benchmarks/sync_optimization/update_reference.py'
    spec = importlib.util.spec_from_file_location('inner_reference', reference_path)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    model = load_source(ROOT / 'outputs/reports/tusz_meta_ttt_v1/runs/supervised/development/s1_seed3407_check0.25/best.pt')
    model.eval()
    functional = SplitFunctionalTUSZModel(model)
    path = next((ROOT / 'outputs/reports/tusz_meta_ttt_v1/cache/train').rglob('*.npz'))
    archive = load_cached_arrays(path)
    count = min(15, len(archive['labels']))
    x = torch.from_numpy(np.stack([
        archive['signal'][:, i*400:i*400+2000].reshape(16,10,200)
        for i in range(count)
    ])).cuda()
    prefix = functional.prefix(x)
    reports = []
    for name, difficulty in [('mask',5),('band',0.5)]:
        objective = build_objective(name,difficulty).cuda().eval()
        ckpt = torch.load(OUT / f'runs/ssl/development/{name}_{difficulty:g}_seed3407/epoch_02.pt',map_location='cpu',weights_only=False)
        objective.load_state_dict(ckpt['objective'])
        prepared = objective.prepare_loss(x,prefix_fn=functional.prefix,transform_seed=3407)
        data=json.loads((OUT / f'calibration/{name}_{difficulty:g}_seed3407/gradient_calibration.json').read_text())
        config=InnerStepConfig(data['selected_relative_step'],
            {int(k):v for k,v in data['block_reference_norms'].items()},
            {int(k):v for k,v in data['block_gradient_medians'].items()})

        def run(step):
            initial=functional.initial_fast_parameters()
            fast=initial
            decisions=[]
            for _ in range(4):
                result=step(fast,loss_fn=lambda p:prepared(lambda z:functional.features_from_prefix(z,p)),
                    record_initial=initial,config=config,create_graph=True)
                fast=result.parameters
                decisions.append((result.accepted,result.reason,result.trial_scale))
            logits=functional.logits_from_prefix(prefix,fast)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits))
            gradients=torch.autograd.grad(loss,tuple(initial.values())+tuple(objective.parameters()),allow_unused=True)
            return (decisions, logits.detach().cpu(),
                    [v.detach().cpu() for v in fast.values()],
                    [None if v is None else v.detach().cpu() for v in gradients])

        old=run(reference.normalized_inner_step)
        new=run(candidate_step)
        assert old[0]==new[0], (old[0],new[0])
        errors=[]
        for a,b in zip([old[1],*old[2],*old[3]],[new[1],*new[2],*new[3]],strict=True):
            if a is None or b is None:
                assert a is None and b is None
                continue
            errors.append(float((a-b).abs().max()))
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        elapsed={'reference':[],'optimized':[]}
        # Alternate order to reduce warm-cache and temperature bias.
        for repeat in range(4):
            order=[('reference',reference.normalized_inner_step),('optimized',candidate_step)]
            if repeat%2: order.reverse()
            for label,step in order:
                torch.cuda.synchronize()
                start=time.perf_counter()
                run(step)
                torch.cuda.synchronize()
                elapsed[label].append(time.perf_counter()-start)
        counts={}
        for label,step in [('reference',reference.normalized_inner_step),('optimized',candidate_step)]:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                run(step)
            counts[label]=sum(e.count for e in prof.key_averages() if e.key=='aten::_local_scalar_dense')
        reports.append({'objective':name,'windows':count,'updates':4,
            'max_parameter_logit_meta_gradient_error':max(errors),'decisions':new[0],
            'paired_elapsed_s':elapsed,'host_scalar_reads':counts,
            'timing_context':'paired microbenchmark; consult process/GPU launch evidence for contention; not end-to-end speedup'})
    target=OUT / ('benchmarks/sync_optimization/verification_packed.json' if args.packed else 'benchmarks/sync_optimization/verification.json')
    target.write_text(json.dumps({'passed':True,'reports':reports},indent=2)+'\n')
    print(target.read_text())


if __name__=='__main__':
    main()
