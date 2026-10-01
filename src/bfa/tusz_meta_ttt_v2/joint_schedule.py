"""Shared frozen-prefix scheduling for independently trained F/C conditions.

All conditions wait at the same record/chunk boundary. F omits the terminal
update, while C performs it. Labels enter only the query loss below.
"""
from collections import OrderedDict

import numpy as np
import torch

from bfa.tusz_meta_ttt.dataset import load_cached_arrays
from .batched import lane_features,normalized_lane_step,prepare_ssl_lanes
from .protocol import chunk_rows,stable_transform_seed


def process_joint_group(functional,objective,paths_by_patient,weights,config,modes,*,seed=3407,compact=True,audit=None,fixed_inner_lr=None,stop_encoder_through_inner=False):
    conditions=len(modes)
    patients=len(paths_by_patient)
    initial=functional.initial_lane_parameters(conditions*patients)
    fast=initial
    device=next(iter(initial.values())).device
    states=[dict(paths=paths,index=0) for paths in paths_by_patient]
    raw_cache=OrderedDict()
    pending=[]
    ticks=0
    update_count=0
    accepted=torch.zeros((),device=device)

    def advance(state):
        while state['index']<len(state['paths']):
            path=state['paths'][state['index']]
            archive=load_cached_arrays(path)
            chunks=chunk_rows(archive['decision_end_s'])
            if chunks:
                state.update(path=path,archive=archive,chunks=chunks,chunk=0,new=True,active=True)
                return
            state['index']+=1
        state.update(new=False,active=False)
    for state in states: advance(state)

    def raw_prefix(patient,rows):
        if not rows:
            return torch.zeros(15,16,10,200,device=device)
        state=states[patient]
        key=(state['path'],rows)
        if key in raw_cache:
            raw_cache.move_to_end(key)
            return raw_cache[key]
        values=np.zeros((15,16,10,200),dtype=np.float32)
        for j,row in enumerate(rows):
            values[j]=state['archive']['signal'][:,row*400:row*400+2000].reshape(16,10,200)
        # One patient/raw chunk, shared by both modes and every outer scope.
        prefix=functional.functionals[0].prefix(torch.from_numpy(values).to(device))
        raw_cache[key]=prefix
        # Keep previous/current/next slots so inserting F's next query cannot
        # evict the same prefix C is about to consume in this tick.
        while len(raw_cache)>3*patients: raw_cache.popitem(last=False)
        return prefix

    def query(rows_by_lane,parameters):
        prefixes=[]
        labels=np.zeros((conditions*patients,15),dtype=np.float32)
        mass=np.zeros_like(labels)
        for lane,rows in enumerate(rows_by_lane):
            patient=lane%patients
            prefixes.append(raw_prefix(patient,rows))
            if rows:
                state=states[patient]
                labels[lane,:len(rows)]=state['archive']['labels'][list(rows)]
                mass[lane,:len(rows)]=weights[state['path']][list(rows)]
        prefix=torch.stack(prefixes)
        features=lane_features(functional,prefix,parameters)
        logits=functional.detect_lanes(features.mean(2).flatten(2)).squeeze(-1)
        per_condition=(torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.from_numpy(labels).to(device),reduction='none')*torch.from_numpy(mass).to(device)).reshape(conditions,-1).sum(1)
        if audit is not None:
            audit['loss']=audit.get('loss',0)+per_condition.detach()
            audit['windows']=audit.get('windows',0)+torch.tensor([sum(len(rows) for rows in rows_by_lane[c*patients:(c+1)*patients]) for c in range(conditions)],device=device)
        return per_condition.sum()

    while any(s['active'] for s in states):
        live=[i for i,s in enumerate(states) if s['active']]
        if compact and len(live)<patients:
            indexes=torch.tensor([c*patients+i for c in range(conditions) for i in live],device=device)
            fast={n:p.index_select(0,indexes) for n,p in fast.items()}
            states=[states[i] for i in live]
            patients=len(states)
            initial=functional.initial_lane_parameters(conditions*patients)
        reset=[s['active'] and s['new'] for s in states]
        if any(reset):
            fast={n:torch.stack([initial[n][c*patients+i] if reset[i] else fast[n][c*patients+i] for c in range(conditions) for i in range(patients)]) for n in initial}
            prime=[s['chunks'][0].rows if mode=='future' and reset[i] else () for mode in modes for i,s in enumerate(states)]
            if any(prime): pending.append(query(prime,fast))
        signal=np.zeros((patients,15,16,10,200),dtype=np.float32)
        valid=np.zeros((conditions,patients,15),dtype=np.float32)
        seeds=[]
        queries=[]
        for patient,state in enumerate(states):
            rows=state['chunks'][state['chunk']].rows if state['active'] else ()
            if rows:
                for j,row in enumerate(rows):
                    signal[patient,j]=state['archive']['signal'][:,row*400:row*400+2000].reshape(16,10,200)
                record_seed=stable_transform_seed(seed,state['path'].as_posix())
                seeds.extend(stable_transform_seed(record_seed,objective.name,row) for row in rows)
            seeds.extend([seed]*(15-len(rows)))
            for c,mode in enumerate(modes):
                if state['active'] and (mode=='current' or state['chunk']+1<len(state['chunks'])):
                    valid[c,patient,:len(rows)]=1
        for mode in modes:
            for state in states:
                index=state['chunk']+(mode=='future') if state['active'] else 0
                queries.append(state['chunks'][index].rows if state['active'] and index<len(state['chunks']) else ())
        # Signals stay identical across F/C even for the F terminal no-op. Only
        # valid loss mass is zeroed; zeroing the signal would corrupt shared C views.
        if valid.any():
            signals=torch.from_numpy(signal).to(device).repeat(conditions,1,1,1,1)
            valid_tensor=torch.from_numpy(valid.reshape(conditions*patients,15)).to(device)
            prepared=prepare_ssl_lanes(objective,signals,valid_tensor,prefix_fn=functional.prefix,seeds=seeds*conditions,prefix_unique_fn=getattr(functional,'prefix_unique',None))
            result=normalized_lane_step(fast,loss_fn=lambda p:prepared(lambda z:lane_features(functional,z,p)),record_initial=initial,config=config,learning_rate=fixed_inner_lr,stop_encoder_through_inner=stop_encoder_through_inner)
            fast=result.parameters
            pending.append(query(queries,fast))
            update_count+=int((valid.sum(2)>0).sum())
            accepted=accepted+result.accepted.sum()
            if audit is not None:
                audit['updates']=audit.get('updates',0)+torch.from_numpy((valid.sum(2)>0).sum(1)).to(device)
                audit['accepted']=audit.get('accepted',0)+result.accepted.reshape(conditions,patients).sum(1)
                audit['gradient_norm_sum']=audit.get('gradient_norm_sum',0)+result.gradient_norms.reshape(conditions,patients,-1).sum(1)
                audit['nonfinite_batch_rejections']=audit.get('nonfinite_batch_rejections',0)+torch.full((conditions,),int(result.nonfinite_batch_rejection),device=device)
            ticks+=1
        for state in states:
            if not state['active']: continue
            state['new']=False
            state['chunk']+=1
            if state['chunk']>=len(state['chunks']):
                state['index']+=1
                advance(state)
        if ticks==4:
            yield torch.stack(pending).sum(),update_count,accepted
            pending=[]
            update_count=0
            accepted=torch.zeros((),device=device)
            fast={n:initial[n]+(p-initial[n]).detach() for n,p in fast.items()}
            ticks=0
    if pending: yield torch.stack(pending).sum(),update_count,accepted
