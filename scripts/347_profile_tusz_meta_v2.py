"""Profile real four-update Meta segments without modifying training checkpoints."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from bfa.tusz_meta_ttt.dataset import load_cached_arrays
from bfa.tusz_meta_ttt_v2.functional import SplitFunctionalTUSZModel
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.runtime import load_source
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/reports/tusz_meta_ttt_v2'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', choices=['band', 'mask'], default='mask')
    parser.add_argument('--inner-kernel', choices=['reference', 'packed'], default='reference')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('trainer_profile', ROOT / 'scripts/320_train_tusz_meta_ttt_v2.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    model = load_source(ROOT / 'outputs/reports/tusz_meta_ttt_v1/runs/supervised/development/s1_seed3407_check0.25/best.pt', detector_trainable=True)
    functional = SplitFunctionalTUSZModel(model)
    difficulty = 5 if args.objective == 'mask' else 0.5
    name = f'{args.objective}_{difficulty:g}_seed3407'
    objective = build_objective(args.objective, difficulty).cuda().eval()
    objective.load_state_dict(torch.load(OUT / f'runs/ssl/development/{name}/epoch_02.pt', map_location='cpu', weights_only=False)['objective'])
    data = json.loads((OUT / f'calibration/{name}/gradient_calibration.json').read_text())
    config = InnerStepConfig(data['selected_relative_step'], {int(k): v for k,v in data['block_reference_norms'].items()}, {int(k): v for k,v in data['block_gradient_medians'].items()})
    fit = set(json.loads((ROOT / 'outputs/reports/tusz_meta_ttt_v1/manifests/development_split.json').read_text())['development_fit'])
    path = next(p for p in sorted((ROOT / 'outputs/reports/tusz_meta_ttt_v1/cache/train').rglob('*.npz')) if p.parts[-4] in fit and len(load_cached_arrays(p)['labels']) >= 75)
    weights = np.ones(len(load_cached_arrays(path)['labels']), dtype=np.float32)
    weights /= len(weights)

    def wrap(owner, method, label):
        original = getattr(owner, method)
        def measured(*a, **kw):
            with torch.profiler.record_function(label):
                return original(*a, **kw)
        setattr(owner, method, measured)

    wrap(functional, 'prefix', 'stage/frozen_prefix')
    wrap(functional, 'features_from_prefix', 'stage/tail_forward')
    wrap(objective, 'prepare_loss', 'stage/ssl_prepare_including_prefix')
    wrap(trainer, 'normalized_inner_step', 'stage/inner_including_tail')
    wrap(trainer, 'packed_normalized_inner_step', 'stage/inner_including_tail')

    def segment():
        model.zero_grad(set_to_none=True)
        objective.zero_grad(set_to_none=True)
        generator = trainer.process_record(functional, objective, path, weights, 'future', config, 3407, 4, 'normalized', None, False, args.inner_kernel)
        loss, diagnostics = next(generator)
        with torch.profiler.record_function('stage/outer_backward'):
            loss.backward()
        generator.close()
        return diagnostics

    segment()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(3):
            diagnostics = segment()
    args.output.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.output / 'trace.json'))
    averages = prof.key_averages()
    (args.output / 'cuda_table.txt').write_text(averages.table(sort_by='self_cuda_time_total', row_limit=40))
    (args.output / 'cpu_table.txt').write_text(averages.table(sort_by='self_cpu_time_total', row_limit=40))
    report = {'record': str(path), 'segments': 3, 'updates_per_segment': len(diagnostics), 'stages': [dict(name=e.key, calls=e.count, cpu_total_us=e.cpu_time_total, device_total_us=e.device_time_total) for e in averages if e.key.startswith('stage/')]}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
