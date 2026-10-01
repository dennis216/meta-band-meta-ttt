"""Complete-record benchmark: four independent conditions share one frozen prefix."""
import argparse
import importlib.util
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from bfa.tusz_meta_ttt.dataset import filter_inventory_records,load_cached_arrays
from bfa.tusz_meta_ttt_v2.batched import enable_second_order_batched_attention,ConditionEnsemble,EnsembleMaskObjective,EnsembleBandObjective
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.protocol import class_patient_record_weights,weight_audit
from bfa.tusz_meta_ttt_v2.runtime import load_source,optimizer_groups
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig
from bfa.tusz_meta_ttt_v2.joint_schedule import process_joint_group
from bfa.tusz_meta_ttt_v2.grouping import patient_order,nominal_lane_efficiency

ROOT=Path(__file__).resolve().parents[1]
V1=ROOT/'outputs/reports/tusz_meta_ttt_v1'
OUT=ROOT/'outputs/reports/tusz_meta_ttt_v2'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--records',type=int,default=48)
    parser.add_argument('--mode',choices=['future','current'],default='future')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--matmul-precision',choices=['highest','high'],default='highest')
    parser.add_argument('--scopes',nargs='+',choices=['e','ed','es','eds'],default=['e','ed','es','eds'])
    parser.add_argument('--objective',choices=['band','mask'],default='mask')
    parser.add_argument('--patients-per-batch',type=int,choices=[1,2,4],default=4)
    parser.add_argument('--compact-lanes',action='store_true')
    parser.add_argument('--joint-modes',action='store_true')
    parser.add_argument('--reference-future',type=Path)
    parser.add_argument('--reference-current',type=Path)
    parser.add_argument('--patient-bucket-size',type=int,choices=[0,8,16,32],default=0)
    parser.add_argument('--deduplicate-views',action='store_true')
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('lane_benchmark',ROOT/'scripts/352_benchmark_tusz_patient_lanes_v2.py')
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    torch.set_float32_matmul_precision(args.matmul_precision)
    enable_second_order_batched_attention()
    entries=[(mode,scope) for mode in (['future','current'] if args.joint_modes else [args.mode]) for scope in args.scopes]
    scopes=[scope for mode,scope in entries]
    modes=[mode for mode,scope in entries]
    difficulty=5 if args.objective=='mask' else .5
    objective_name=f'{args.objective}_{difficulty:g}_seed3407'
    models=[]
    objectives=[]
    optimizers=[]
    for scope in scopes:
        model=load_source(V1/'runs/supervised/development/s1_seed3407_check0.25/best.pt',detector_trainable='d' in scope)
        objective=build_objective(args.objective,difficulty).cuda().eval()
        objective.load_state_dict(torch.load(OUT/f'runs/ssl/development/{objective_name}/epoch_02.pt',map_location='cpu',weights_only=False)['objective'])
        objective.requires_grad_('s' in scope)
        groups=optimizer_groups(model,'backbone.encoder.layers.10.',1e-5,.01)+optimizer_groups(model,'backbone.encoder.layers.11.',1e-5,.01)
        if 'd' in scope: groups+=optimizer_groups(model,'detector.',3e-6,.01)
        if 's' in scope: groups.append(dict(params=list(objective.parameters()),lr=1e-4,weight_decay=0))
        models.append(model)
        objectives.append(objective)
        optimizers.append(torch.optim.AdamW(groups,fused=True))
    functional=ConditionEnsemble(models)
    objective=(EnsembleMaskObjective if args.objective=='mask' else EnsembleBandObjective)(objectives,deduplicate_views=args.deduplicate_views)
    c=json.loads((OUT/f'calibration/{objective_name}/gradient_calibration.json').read_text())
    config=InnerStepConfig(c['selected_relative_step'],{int(k):v for k,v in c['block_reference_norms'].items()},{int(k):v for k,v in c['block_gradient_medians'].items()})
    fit=set(json.loads((V1/'manifests/development_split.json').read_text())['development_fit'])
    inventory=json.loads((V1/'manifests/records.json').read_text())
    paths=filter_inventory_records([p for p in sorted((V1/'cache/train').rglob('*.npz')) if p.parts[-4] in fit],inventory,partition='train')[:args.records]
    labels={p:load_cached_arrays(p)['labels'] for p in paths}
    weights=class_patient_record_weights(labels)
    patients=defaultdict(list)
    for p in paths: patients[p.parts[-4]].append(p)
    workloads={p:sum(len(labels[path]) for path in records) for p,records in patients.items()}
    order=patient_order(workloads,seed=3408,bucket_size=args.patient_bucket_size)
    total_loss=0.
    updates=0
    accepted=0.
    started=time.perf_counter()
    for start in range(0,len(order),4):
        selected=order[start:start+4]
        for optimizer in optimizers: optimizer.zero_grad(set_to_none=True)
        # Identical paths and deterministic transform seeds, condition-major.
        for lane_start in range(0,len(selected),args.patients_per_batch):
            lane_patients=selected[lane_start:lane_start+args.patients_per_batch]
            if args.joint_modes:
                iterator=process_joint_group(functional,objective,[patients[p] for p in lane_patients],weights,config,modes,compact=args.compact_lanes)
            else:
                group=[patients[p] for scope in scopes for p in lane_patients]
                iterator=runner.process_group(functional,objective,group,weights,config,args.mode,compact_lanes=args.compact_lanes)
            for loss,count,success in iterator:
                loss.backward()
                total_loss+=float(loss.detach())
                updates+=count
                accepted+=float(success)
        for scope,model,ssl,optimizer in zip(scopes,models,objectives,optimizers,strict=True):
            for p in [*model.parameters(),*ssl.parameters()]:
                if p.grad is not None: p.grad.mul_(len(order)/len(selected))
            encoder=[p for n,p in model.named_parameters() if n in functional.adaptable_names]
            torch.nn.utils.clip_grad_norm_(encoder,1.,error_if_nonfinite=True)
            if 'd' in scope: torch.nn.utils.clip_grad_norm_(model.detector.parameters(),1.,error_if_nonfinite=True)
            if 's' in scope: torch.nn.utils.clip_grad_norm_(ssl.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    args.output.mkdir(parents=True,exist_ok=True)
    comparisons=[]
    for i,(scope,model,ssl) in enumerate(zip(scopes,models,objectives,strict=True)):
        state=dict(model=model.state_dict(),objective=ssl.state_dict())
        condition_name=f'{modes[i]}_{scope}' if args.joint_modes else scope
        torch.save(state,args.output/f'{condition_name}.pt')
        reference_path=None
        mode_reference=args.reference_future if modes[i]=='future' else args.reference_current
        if args.joint_modes and mode_reference:
            reference_path=mode_reference/f'{scope}.pt'
        elif args.reference:
            reference_index=['e','ed','es','eds'].index(scope)
            reference_path=args.reference/f'{scope}_{reference_index}.pt'
        if reference_path:
            reference=torch.load(reference_path,map_location='cpu',weights_only=False)
            error={}
            for kind,values in state.items():
                error[kind]=max(float((value.detach().cpu()-reference[kind][n]).abs().max()) for n,value in values.items())
            comparisons.append(dict(scope=scope,mode=modes[i],max_parameter_absolute_error=error))
    report=dict(records_per_condition=len(paths),patients_per_condition=len(order),scopes=scopes,objective=args.objective,patients_per_batch=args.patients_per_batch,compact_lanes=args.compact_lanes,updates=updates,accepted_fraction=accepted/max(1,updates),outer_weighted_loss_sum=total_loss,elapsed_s=elapsed,aggregate_updates_per_s=updates/elapsed,class_weight_audit_per_condition=weight_audit(weights,labels),reference_comparisons=comparisons,mode=args.mode,matmul_precision=args.matmul_precision,status='benchmark_only',protocol_change='4 synchronized lane ticks per truncation; independent condition optimizers; shared read-only prefix')
    report['joint_modes']=args.joint_modes
    report['condition_modes']=modes
    report['patient_bucket_size']=args.patient_bucket_size
    report['deduplicate_views']=args.deduplicate_views
    report['nominal_lane_efficiency']=nominal_lane_efficiency(order,workloads,args.patients_per_batch)
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
