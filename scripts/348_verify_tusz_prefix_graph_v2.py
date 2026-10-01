"""Real-model CUDA Graph prefix check, including retained four-step Meta state."""
import json
from pathlib import Path

import torch

from bfa.tusz_meta_ttt_v2.functional import SplitFunctionalTUSZModel
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.runtime import load_source
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig, packed_normalized_inner_step

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/reports/tusz_meta_ttt_v2'


def main():
    torch.manual_seed(3407)
    model = load_source(ROOT / 'outputs/reports/tusz_meta_ttt_v1/runs/supervised/development/s1_seed3407_check0.25/best.pt', detector_trainable=True)
    eager = SplitFunctionalTUSZModel(model)
    graphed = SplitFunctionalTUSZModel(model, prefix_cuda_graph=True)
    reports = []
    for objective_name, difficulty in [('mask',5), ('band',.5)]:
        objective = build_objective(objective_name, difficulty).cuda().eval()
        name = f'{objective_name}_{difficulty:g}_seed3407'
        objective.load_state_dict(torch.load(OUT / f'runs/ssl/development/{name}/epoch_02.pt', map_location='cpu', weights_only=False)['objective'])
        data = json.loads((OUT / f'calibration/{name}/gradient_calibration.json').read_text())
        config = InnerStepConfig(data['selected_relative_step'], {int(k):v for k,v in data['block_reference_norms'].items()}, {int(k):v for k,v in data['block_gradient_medians'].items()})
        # Exercise changing values, full and partial chunk shapes, and cache fallback.
        signals = [torch.randn(n,16,10,200,device='cuda') * .1 for n in (11,15,15,3)]
        prefix_error = 0.
        retained = []
        for x in signals:
            a, b = eager.prefix(x), graphed.prefix(x)
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
            prefix_error = max(prefix_error,float((a-b).abs().max()))
            retained.append((b,b.clone()))
        for actual, saved in retained:
            torch.testing.assert_close(actual,saved,atol=0,rtol=0)

        def run(functional):
            initial = functional.initial_fast_parameters(detach=False)
            fast = initial
            losses, decisions = [], []
            for i, x in enumerate(signals):
                prepared = objective.prepare_loss(x, prefix_fn=functional.prefix, transform_seed=3407+i)
                result = packed_normalized_inner_step(fast, loss_fn=lambda p: prepared(lambda z:functional.features_from_prefix(z,p)), record_initial=initial, config=config, create_graph=True)
                fast = result.parameters
                decisions.append((result.accepted,result.reason,result.trial_scale))
                logits = functional.logits_from_prefix(functional.prefix(x),fast)
                losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits)))
            gradients = torch.autograd.grad(sum(losses), (*initial.values(), *model.detector.parameters(), *objective.parameters()), allow_unused=True)
            return decisions, [p.detach().clone() for p in fast.values()], [None if g is None else g.detach() for g in gradients]

        a,b = run(eager),run(graphed)
        assert a[0]==b[0]
        errors=[]
        for old,new in zip(a[1]+a[2],b[1]+b[2],strict=True):
            if old is None or new is None:
                assert old is None and new is None
            else:
                torch.testing.assert_close(old,new,atol=1e-5,rtol=1e-4)
                errors.append(float((old-new).abs().max()))
        reports.append(dict(objective=objective_name,prefix_max_error=prefix_error,parameter_and_meta_gradient_max_error=max(errors),decisions=a[0]))
    target = OUT / 'benchmarks/prefix_cuda_graph_verification.json'
    target.write_text(json.dumps({'passed':True,'reports':reports},indent=2))
    print(target.read_text(),flush=True)


if __name__=='__main__':
    main()
