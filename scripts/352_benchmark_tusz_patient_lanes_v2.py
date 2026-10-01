"""Real-record Mask training benchmark with independent synchronized patient lanes.

Experimental truncation: backward every four lane ticks; EDF resets can shorten
the first/last segment. Source parameters still update only after four patients.
"""
import argparse
import importlib.util
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from bfa.tusz_meta_ttt.dataset import filter_inventory_records, load_cached_arrays
from bfa.tusz_meta_ttt_v2.batched import enable_second_order_batched_attention, lane_features, prepare_ssl_lanes, normalized_lane_step
from bfa.tusz_meta_ttt_v2.functional import SplitFunctionalTUSZModel
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.protocol import chunk_rows, class_patient_record_weights, stable_transform_seed, weight_audit
from bfa.tusz_meta_ttt_v2.runtime import load_source, optimizer_groups
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reports/tusz_meta_ttt_v2'
V1=ROOT/'outputs/reports/tusz_meta_ttt_v1'


def process_group(functional, objective, paths_by_lane, weights, config, mode='future', seed=3407, compact_lanes=False):
    lanes=len(paths_by_lane)
    if hasattr(functional,'initial_lane_parameters'):
        initial=functional.initial_lane_parameters(lanes)
    else:
        base=functional.initial_fast_parameters(detach=False)
        initial={n:p.unsqueeze(0).expand(lanes,*p.shape) for n,p in base.items()}
    fast=initial
    device=next(iter(initial.values())).device
    states=[dict(paths=paths,index=0,new=True) for paths in paths_by_lane]
    pending=[]
    ticks=0
    updates=0
    accepted=torch.zeros((),device=device)

    def advance(state):
        while state['index']<len(state['paths']):
            path=state['paths'][state['index']]
            archive=load_cached_arrays(path)
            chunks=chunk_rows(archive['decision_end_s'])
            if chunks:
                state.update(path=path,archive=archive,chunks=chunks,chunk=0,active=True,new=True)
                return
            state['index']+=1
        state.update(active=False,new=False)

    for state in states: advance(state)

    def pack(rows_by_lane):
        signals=np.zeros((lanes,15,16,10,200),dtype=np.float32)
        labels=np.zeros((lanes,15),dtype=np.float32)
        mass=np.zeros_like(labels)
        valid=np.zeros_like(labels)
        seeds=[]
        for lane,rows in enumerate(rows_by_lane):
            state=states[lane]
            count=len(rows)
            if count:
                archive=state['archive']
                for j,row in enumerate(rows):
                    signals[lane,j]=archive['signal'][:,row*400:row*400+2000].reshape(16,10,200)
                labels[lane,:count]=archive['labels'][list(rows)]
                mass[lane,:count]=weights[state['path']][list(rows)]
                valid[lane,:count]=1
                record_seed=stable_transform_seed(seed,state['path'].as_posix())
                seeds.extend(stable_transform_seed(record_seed,objective.name,row) for row in rows)
            seeds.extend([seed]*(15-count))
        return tuple(torch.from_numpy(a).to(device=device) for a in (signals,labels,mass,valid)),seeds

    def query_loss(rows_by_lane,parameters):
        (signal,target,mass,_),_=pack(rows_by_lane)
        prefix=functional.prefix(signal.flatten(0,1)).reshape(lanes,15,16,10,200)
        features=lane_features(functional,prefix,parameters)
        pooled=features.mean(2).flatten(2)
        logits=(functional.detect_lanes(pooled) if hasattr(functional,'detect_lanes') else functional.model.detector(pooled)).squeeze(-1)
        return (torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*mass).sum()

    while any(s['active'] for s in states):
        active_indexes=[i for i,s in enumerate(states) if s['active']]
        if compact_lanes and len(active_indexes)<lanes:
            index=torch.tensor(active_indexes,device=device)
            fast={n:p.index_select(0,index) for n,p in fast.items()}
            states=[states[i] for i in active_indexes]
            lanes=len(states)
            # Rebuild from source leaves. Reusing an index_select graph for the
            # initialization across multiple backwards would retain freed state.
            if hasattr(functional,'initial_lane_parameters'):
                initial=functional.initial_lane_parameters(lanes)
            else:
                initial={n:p.unsqueeze(0).expand(lanes,*p.shape) for n,p in base.items()}
        reset=[s['active'] and s['new'] for s in states]
        if any(reset):
            fast={n:torch.stack([initial[n][i] if reset[i] else fast[n][i] for i in range(lanes)]) for n in initial}
        if mode=='future' and any(reset):
            pending.append(query_loss([s['chunks'][0].rows if reset[i] else () for i,s in enumerate(states)],fast))
        support_rows=[]
        query_rows=[]
        advancing=[]
        for state in states:
            if not state['active']:
                support_rows.append(())
                query_rows.append(())
                advancing.append(False)
                continue
            index=state['chunk']
            chunks=state['chunks']
            usable=mode=='current' or index+1<len(chunks)
            support_rows.append(chunks[index].rows if usable else ())
            query_rows.append(chunks[index+(mode=='future')].rows if usable else ())
            advancing.append(usable)
            state['new']=False
        if any(advancing):
            (signal,_,_,valid),seeds=pack(support_rows)
            prepared=prepare_ssl_lanes(objective,signal,valid,prefix_fn=functional.prefix,seeds=seeds,prefix_unique_fn=getattr(functional,'prefix_unique',None))
            result=normalized_lane_step(fast,loss_fn=lambda p:prepared(lambda z:lane_features(functional,z,p)),record_initial=initial,config=config)
            fast=result.parameters
            pending.append(query_loss(query_rows,fast))
            updates+=sum(advancing)
            accepted=accepted+result.accepted.sum()
            ticks+=1
        for state,used in zip(states,advancing,strict=True):
            if not state['active']: continue
            state['chunk']+=int(used)
            done=state['chunk']>=len(state['chunks'])-(mode=='future')
            if done:
                state['index']+=1
                advance(state)
        if ticks==4:
            yield torch.stack(pending).sum(),updates,accepted
            pending=[]
            updates=0
            accepted=torch.zeros((),device=device)
            fast={n:initial[n]+(p-initial[n]).detach() for n,p in fast.items()}
            ticks=0
    if pending:
        yield torch.stack(pending).sum(),updates,accepted


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--records',type=int,default=24)
    parser.add_argument('--mode',choices=['future','current'],default='future')
    parser.add_argument('--outer-scope',choices=['e','ed','es','eds'],default='ed')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lanes',type=int,choices=[1,4],default=4)
    args=parser.parse_args()
    random.seed(3407)
    np.random.seed(3407)
    torch.manual_seed(3407)
    enable_second_order_batched_attention()
    model=load_source(V1/'runs/supervised/development/s1_seed3407_check0.25/best.pt',detector_trainable='d' in args.outer_scope)
    functional=SplitFunctionalTUSZModel(model,prefix_cuda_graph=True)
    objective=build_objective('mask',5).cuda().eval()
    objective.load_state_dict(torch.load(OUT/'runs/ssl/development/mask_5_seed3407/epoch_02.pt',map_location='cpu',weights_only=False)['objective'])
    objective.requires_grad_('s' in args.outer_scope)
    c=json.loads((OUT/'calibration/mask_5_seed3407/gradient_calibration.json').read_text())
    config=InnerStepConfig(c['selected_relative_step'],{int(k):v for k,v in c['block_reference_norms'].items()},{int(k):v for k,v in c['block_gradient_medians'].items()})
    fit=set(json.loads((V1/'manifests/development_split.json').read_text())['development_fit'])
    inventory=json.loads((V1/'manifests/records.json').read_text())
    paths=filter_inventory_records([p for p in sorted((V1/'cache/train').rglob('*.npz')) if p.parts[-4] in fit],inventory,partition='train')[:args.records]
    labels={p:load_cached_arrays(p)['labels'] for p in paths}
    weights=class_patient_record_weights(labels)
    patients=defaultdict(list)
    for p in paths: patients[p.parts[-4]].append(p)
    order=sorted(patients)
    random.Random(3408).shuffle(order)
    groups=optimizer_groups(model,'backbone.encoder.layers.10.',1e-5,.01)+optimizer_groups(model,'backbone.encoder.layers.11.',1e-5,.01)
    if 'd' in args.outer_scope: groups+=optimizer_groups(model,'detector.',3e-6,.01)
    if 's' in args.outer_scope: groups.append(dict(params=list(objective.parameters()),lr=1e-4,weight_decay=0))
    optimizer=torch.optim.AdamW(groups,fused=True)
    encoder=[p for n,p in model.named_parameters() if n in functional.adaptable_names]
    updates=0
    accepted=0.
    total_loss=0.
    started=time.perf_counter()
    for start in range(0,len(order),4):
        selected=order[start:start+4]
        optimizer.zero_grad(set_to_none=True)
        for lane_start in range(0,len(selected),args.lanes):
            group=[patients[p] for p in selected[lane_start:lane_start+args.lanes]]
            for loss,count,success in process_group(functional,objective,group,weights,config,args.mode):
                loss.backward()
                total_loss+=float(loss.detach())
                updates+=count
                accepted+=float(success)
        for p in [*model.parameters(),*objective.parameters()]:
            if p.grad is not None: p.grad.mul_(len(order)/len(selected))
        torch.nn.utils.clip_grad_norm_(encoder,1.)
        if 'd' in args.outer_scope:
            torch.nn.utils.clip_grad_norm_(model.detector.parameters(),1.)
        if 's' in args.outer_scope:
            torch.nn.utils.clip_grad_norm_(objective.parameters(),1.)
        optimizer.step()
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    report=dict(lanes=args.lanes,records=len(paths),patients=len(order),updates=updates,accepted_fraction=accepted/max(1,updates),outer_weighted_loss=total_loss,elapsed_s=elapsed,updates_per_s=updates/elapsed,class_weights=weight_audit(weights,labels),mode=args.mode,outer_scope=args.outer_scope,seed=3407,protocol_change='4 synchronized lane ticks per truncation; record reset may shorten segment',status='benchmark_only')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    torch.save(dict(model=model.state_dict(),objective=objective.state_dict(),report=report),args.output.with_suffix('.pt'))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
