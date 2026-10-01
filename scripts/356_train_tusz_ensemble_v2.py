"""Full development/formal training with independent F/C models and a shared prefix.

Resume occurs at completed four-patient outer steps. The frozen S1 prefix is
reloaded; every mutable parameter, optimizer, RNG and coverage cursor is saved.
"""
import argparse
import fcntl
import hashlib
import json
import math
import random
import signal
import time
from collections import defaultdict
from datetime import UTC,datetime
from pathlib import Path

import numpy as np
import torch

from bfa.tusz_meta_ttt.dataset import filter_inventory_records,load_cached_arrays
from bfa.tusz_meta_ttt_v2.batched import ConditionEnsemble,EnsembleBandObjective,EnsembleMaskObjective,enable_second_order_batched_attention
from bfa.tusz_meta_ttt_v2.grouping import patient_order
from bfa.tusz_meta_ttt_v2.joint_schedule import process_joint_group
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.protocol import assert_train_cache_paths,class_patient_record_weights,weight_audit
from bfa.tusz_meta_ttt_v2.runtime import load_source,optimizer_groups
from bfa.tusz_meta_ttt_v2.training_state import atomic_save,capture_rng,restore_rng,detector_open,validate_resume
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig

ROOT=Path(__file__).resolve().parents[1]
V1=ROOT/'outputs/reports/tusz_meta_ttt_v1'
OUT=ROOT/'outputs/reports/tusz_meta_ttt_v2'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return h.hexdigest()


def cpu_state(module,names=None):
    return {n:v.detach().cpu().clone() for n,v in module.state_dict().items() if names is None or n in names}


def as_json(audit):
    return {k:v.detach().cpu().tolist() if isinstance(v,torch.Tensor) else v for k,v in audit.items()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--objective',choices=['band','mask'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--patients-per-batch',type=int,choices=[1,2,4],required=True)
    parser.add_argument('--patient-bucket-size',type=int,choices=[0,8,16,32],default=8)
    parser.add_argument('--seed',type=int,choices=[17,42,3407],default=3407)
    parser.add_argument('--stage',choices=['development','formal'],default='development')
    parser.add_argument('--outer-scopes',nargs='+',choices=['e','ed','es','eds'],default=['e','ed','es','eds'])
    parser.add_argument('--detector-freeze-fraction',type=float,default=.25)
    parser.add_argument('--fixed-inner-lr',type=float)
    parser.add_argument('--stop-encoder-through-inner',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--allow-smaller-patient-microbatch',action='store_true')
    parser.add_argument('--checkpoint-every',type=int,default=4)
    parser.add_argument('--maximum-records',type=int)
    parser.add_argument('--stop-after-groups',type=int)
    args=parser.parse_args()
    if args.epochs<1 or args.checkpoint_every<1: raise ValueError('Positive epochs/checkpoint interval required')
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    output_lock=(args.output/'.training.lock').open('a')
    fcntl.flock(output_lock.fileno(),fcntl.LOCK_EX)
    if (args.output/'complete.json').is_file():
        completed=json.loads((args.output/'complete.json').read_text())
        expected=dict(objective=args.objective,seed=args.seed,entries=[[mode,scope] for mode in ['future','current'] for scope in args.outer_scopes],detector_freeze_fraction=args.detector_freeze_fraction,stage='smoke' if args.maximum_records else args.stage)
        if args.fixed_inner_lr is not None or args.stop_encoder_through_inner:expected.update(fixed_inner_lr=args.fixed_inner_lr,stop_encoder_through_inner=args.stop_encoder_through_inner)
        if completed['epochs']>=args.epochs and all(completed['contract'].get(k)==v for k,v in expected.items()):
            print(json.dumps(dict(phase='already_complete',output=str(args.output))),flush=True)
            return
        raise ValueError('Completed output has a different training contract')
    if (args.output/'last.pt').exists() and not args.resume: raise FileExistsError('Use --resume for an existing training run')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision('highest')
    enable_second_order_batched_attention()
    source=V1/f'runs/supervised/{args.stage}/s1_seed{args.seed}_check0.25/best.pt'
    difficulty=.5 if args.objective=='band' else 5
    task=f'{args.objective}_{difficulty:g}_seed{args.seed}'
    head=OUT/f'runs/ssl/{args.stage}/{task}/epoch_02.pt'
    calibration_path=OUT/('calibration/formal' if args.stage=='formal' else 'calibration')/task/'gradient_calibration.json'
    calibration=json.loads(calibration_path.read_text())
    config=InnerStepConfig(calibration['selected_relative_step'],{int(k):v for k,v in calibration['block_reference_norms'].items()},{int(k):v for k,v in calibration['block_gradient_medians'].items()})
    fit=(set(json.loads((V1/'manifests/development_split.json').read_text())['development_fit']) if args.stage=='development' else None)
    inventory=json.loads((V1/'manifests/records.json').read_text())
    paths=filter_inventory_records([p for p in sorted((V1/'cache/train').rglob('*.npz')) if fit is None or p.parts[-4] in fit],inventory,partition='train')
    assert_train_cache_paths(paths,V1/'cache/train')
    if args.maximum_records: paths=paths[:args.maximum_records]
    labels={p:load_cached_arrays(p)['labels'] for p in paths}
    weights=class_patient_record_weights(labels)
    patients=defaultdict(list)
    for p in paths: patients[p.parts[-4]].append(p)
    if not args.maximum_records and len(patients)!=(len(fit) if fit is not None else 579): raise ValueError('Incomplete patient coverage')
    workloads={p:sum(len(labels[r]) for r in records) for p,records in patients.items()}
    class_audit=weight_audit(weights,labels)
    if not all(abs(class_audit[c]-.5)<1e-6 for c in ['background','seizure']):
        raise ValueError(f'Invalid class weights: {class_audit}')
    entries=[(mode,scope) for mode in ['future','current'] for scope in args.outer_scopes]
    modes=[m for m,s in entries]
    contract=dict(version='ensemble_v2_1',objective=args.objective,difficulty=difficulty,seed=args.seed,
        source_sha256=sha256(source),head_sha256=sha256(head),calibration_sha256=sha256(calibration_path),
        inventory_sha256=sha256(V1/'manifests/records.json'),split_sha256=sha256(V1/'manifests/development_split.json') if fit is not None else None,
        paths_sha256=hashlib.sha256('\n'.join(map(str,paths)).encode()).hexdigest(),
        patients_per_batch=args.patients_per_batch,patient_bucket_size=args.patient_bucket_size,
        entries=entries,detector_freeze_fraction=args.detector_freeze_fraction,truncation='four synchronized chunk ticks; straight-through initialization reattachment',
        prefix_precision='fp32',tail_precision='fp32',matmul_precision='highest',deduplicate_views=True,
        inner_config=vars(config),patients=len(patients),records=len(paths),windows=sum(workloads.values()),
        class_weight_audit=class_audit,stage='smoke' if args.maximum_records else args.stage)
    if args.fixed_inner_lr is not None or args.stop_encoder_through_inner:
        contract.update(fixed_inner_lr=args.fixed_inner_lr,stop_encoder_through_inner=args.stop_encoder_through_inner)
    models=[]
    objectives=[]
    optimizers=[]
    for mode,scope in entries:
        model=load_source(source,detector_trainable='d' in scope)
        objective=build_objective(args.objective,difficulty).cuda().eval()
        objective.load_state_dict(torch.load(head,map_location='cpu',weights_only=False)['objective'])
        objective.requires_grad_('s' in scope)
        groups=optimizer_groups(model,'backbone.encoder.layers.10.',1e-5,.01)+optimizer_groups(model,'backbone.encoder.layers.11.',1e-5,.01)
        if 'd' in scope: groups+=optimizer_groups(model,'detector.',3e-6,.01)
        if 's' in scope: groups.append(dict(params=list(objective.parameters()),lr=1e-4,weight_decay=0.))
        models.append(model)
        objectives.append(objective)
        optimizers.append(torch.optim.AdamW(groups,fused=True))
    functional=ConditionEnsemble(models)
    objective=(EnsembleBandObjective if args.objective=='band' else EnsembleMaskObjective)(objectives,deduplicate_views=True)
    mutable=set(functional.adaptable_names)|{n for n,p in models[0].named_parameters() if n.startswith('detector.')}
    histories=[[] for _ in entries]
    epoch=1
    cursor=0
    audit={}
    elapsed_prior=0.
    global_steps=0
    epoch_steps=0
    if args.resume:
        saved=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
        previous=dict(saved['contract'])
        if args.allow_smaller_patient_microbatch:
            old_batch=previous['patients_per_batch']
            if args.patients_per_batch>old_batch: raise ValueError('Only microbatch contraction is permitted')
            previous['patients_per_batch']=args.patients_per_batch
            with (args.output/'microbatch_revisions.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(epoch=saved['epoch'],next_patient=saved['next_patient'],old=old_batch,new=args.patients_per_batch,reason='physical VRAM exceeded; preserve four-patient outer accumulation'))+'\n')
        validate_resume(previous,contract)
        for i,(model,ssl,optimizer) in enumerate(zip(models,objectives,optimizers,strict=True)):
            model.load_state_dict(saved['conditions'][i]['model_mutable'],strict=False)
            ssl.load_state_dict(saved['conditions'][i]['objective'])
            optimizer.load_state_dict(saved['conditions'][i]['optimizer'])
        histories=saved['histories']
        epoch,cursor=saved['epoch'],saved['next_patient']
        audit={k:torch.tensor(v,device='cuda') for k,v in saved['audit'].items()}
        elapsed_prior=saved['elapsed_s']
        global_steps=saved['global_steps']
        epoch_steps=saved['epoch_steps']
        restore_rng(saved['rng'])
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    stopping=False
    def request_stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    started=time.perf_counter()
    def save(next_epoch,next_patient):
        conditions=[dict(model_mutable=cpu_state(m,mutable),objective=cpu_state(o),optimizer=opt.state_dict()) for m,o,opt in zip(models,objectives,optimizers,strict=True)]
        atomic_save(dict(contract=contract,conditions=conditions,histories=histories,epoch=next_epoch,next_patient=next_patient,
            audit=as_json(audit),rng=capture_rng(),elapsed_s=elapsed_prior+time.perf_counter()-started,global_steps=global_steps,epoch_steps=epoch_steps),args.output/'last.pt')
    print(json.dumps(dict(phase='training_started',epoch=epoch,next_patient=cursor,contract=contract)),flush=True)
    executed_groups=0
    try:
        while epoch<=args.epochs:
            order=patient_order(workloads,seed=args.seed+epoch,bucket_size=args.patient_bucket_size)
            for start in range(cursor,len(order),4):
                selected=order[start:start+4]
                for optimizer in optimizers: optimizer.zero_grad(set_to_none=True)
                group_audit={}
                step_started=time.perf_counter()
                for lane_start in range(0,len(selected),args.patients_per_batch):
                    lane_patients=selected[lane_start:lane_start+args.patients_per_batch]
                    for loss,count,accepted in process_joint_group(functional,objective,[patients[p] for p in lane_patients],weights,config,modes,seed=args.seed,compact=True,audit=group_audit,fixed_inner_lr=args.fixed_inner_lr,stop_encoder_through_inner=args.stop_encoder_through_inner):
                        loss.backward()
                norms=[]
                changes=[]
                open_head=detector_open(epoch,start//4,math.ceil(len(order)/4),args.detector_freeze_fraction)
                # Validate every condition before advancing any optimizer.
                before=[]
                for (mode,scope),model,ssl in zip(entries,models,objectives,strict=True):
                    for p in [*model.parameters(),*ssl.parameters()]:
                        if p.grad is not None: p.grad.mul_(len(order)/len(selected))
                    groups=[[p for n,p in model.named_parameters() if n in functional.adaptable_names],list(model.detector.parameters()),list(ssl.parameters())]
                    if not open_head:
                        for p in model.detector.parameters(): p.grad=None
                    group_norms=[torch.nn.utils.clip_grad_norm_(ps,1.,error_if_nonfinite=True).to('cuda') for ps in groups]
                    norms.append(torch.stack(group_norms))
                    before.append([[p.detach().clone() for p in ps] for ps in groups])
                for optimizer in optimizers: optimizer.step()
                for model,ssl,old in zip(models,objectives,before,strict=True):
                    groups=[[p for n,p in model.named_parameters() if n in functional.adaptable_names],list(model.detector.parameters()),list(ssl.parameters())]
                    changes.append(torch.stack([sum((p.detach()-v).float().square().sum() for p,v in zip(ps,vs,strict=True)).sqrt() for ps,vs in zip(groups,old,strict=True)]))
                group_audit['outer_gradient_norm_sum']=torch.stack(norms)
                group_audit['outer_clipped_count']=(torch.stack(norms)>1).long()
                group_audit['parameter_step_norm_sum']=torch.stack(changes)
                for key,value in group_audit.items(): audit[key]=audit.get(key,0)+value
                global_steps+=1
                epoch_steps+=1
                executed_groups+=1
                cursor=start+len(selected)
                progress=dict(phase='outer_step',epoch=epoch,patients_complete=cursor,patients_total=len(order),records_complete=sum(len(patients[p]) for p in order[:cursor]),
                    step_s=time.perf_counter()-step_started,global_steps=global_steps,detector_open=open_head,conditions=as_json(group_audit),peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
                print(json.dumps(progress),flush=True)
                (args.output/'progress.json').write_text(json.dumps(progress,indent=2))
                requested_stop=stopping or (args.stop_after_groups and executed_groups>=args.stop_after_groups)
                if global_steps%args.checkpoint_every==0 or cursor==len(order) or requested_stop: save(epoch,cursor)
                if requested_stop:
                    print(json.dumps(dict(phase='paused_at_outer_boundary',checkpoint=str(args.output/'last.pt'))),flush=True)
                    return
            values=as_json(audit)
            if values.get('windows')!=[sum(workloads.values())]*len(entries): raise RuntimeError(f'Window coverage mismatch: {values.get("windows")}')
            for i,((mode,scope),model,ssl) in enumerate(zip(entries,models,objectives,strict=True)):
                row=dict(epoch=epoch,patients=len(order),records=len(paths),windows=values['windows'][i],class_weight_audit=class_audit,
                    outer_weighted_loss=values['loss'][i],updates=values['updates'][i],accepted_fraction=values['accepted'][i]/max(1,values['updates'][i]),outer_steps=epoch_steps,
                    mean_preclip_gradient_norm=[v/epoch_steps for v in values['outer_gradient_norm_sum'][i]],
                    gradient_clipped_fraction=[v/epoch_steps for v in values['outer_clipped_count'][i]],
                    parameter_step_norm_sum=values['parameter_step_norm_sum'][i],elapsed_s=elapsed_prior+time.perf_counter()-started)
                histories[i].append(row)
                run=args.output/mode/scope
                checkpoint=dict(model=cpu_state(model),objective=cpu_state(ssl),objective_name=args.objective,difficulty=difficulty,mode=mode,outer_scope=scope,
                    inner_rule='sgd' if args.fixed_inner_lr is not None else 'normalized',inner_kernel='packed',prefix_precision='fp32',prefix_microbatch=0,prefix_cuda_graph=True,
                    stop_encoder_through_inner=args.stop_encoder_through_inner,fixed_inner_lr=args.fixed_inner_lr,detector_freeze_fraction=args.detector_freeze_fraction,inner_config=vars(config),source=str(source),objective_checkpoint=str(head),
                    stage=contract['stage'],seed=args.seed,epoch=epoch,created_utc=datetime.now(UTC).isoformat(),training_contract=contract,history=histories[i])
                atomic_save(checkpoint,run/f'epoch_{epoch:02d}.pt')
                (run/'history.json').write_text(json.dumps(histories[i],indent=2))
            print(json.dumps(dict(phase='epoch_complete',epoch=epoch,conditions=[h[-1] for h in histories])),flush=True)
            epoch+=1
            cursor=0
            epoch_steps=0
            audit={}
            save(epoch,0)
        (args.output/'complete.json').write_text(json.dumps(dict(status='training_complete',epochs=args.epochs,conditions=entries,contract=contract),indent=2))
    except Exception as error:
        # last.pt always precedes the incomplete outer group; resume replays it.
        (args.output/'failure.json').write_text(json.dumps(dict(error=repr(error),epoch=epoch,last_completed_patient=cursor,checkpoint_exists=(args.output/'last.pt').exists()),indent=2))
        raise


if __name__=='__main__': main()
