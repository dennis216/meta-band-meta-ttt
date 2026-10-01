#!/usr/bin/env python3
"""Exact full/direct/through-inner outer-gradient decomposition on Train diagnostics."""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from bfa.tusz_meta_ttt.dataset import filter_inventory_records, load_cached_arrays
from bfa.tusz_meta_ttt_v2.batched import ConditionEnsemble, EnsembleBandObjective, EnsembleMaskObjective, enable_second_order_batched_attention
from bfa.tusz_meta_ttt_v2.objectives import build_objective
from bfa.tusz_meta_ttt_v2.protocol import class_patient_record_weights
from bfa.tusz_meta_ttt_v2.runtime import load_source
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig
from bfa.tusz_meta_ttt_v3.joint_schedule import process_future_group
from bfa.tusz_meta_ttt_v3.losses import CONDITIONS, PenaltyScale, high_score_reference

ROOT=Path(__file__).resolve().parents[1]
V1=ROOT/'outputs/reports/tusz_meta_ttt_v1'

def vector(parameters, gradients, prefix):
    pieces=[]
    for (name,p),g in zip(parameters,gradients,strict=True):
        if name.startswith(prefix): pieces.append((torch.zeros_like(p) if g is None else g).flatten())
    return torch.cat(pieces).float()

def metrics(full,direct):
    through=full-direct
    return dict(full_norm=float(full.norm()),direct_norm=float(direct.norm()),through_norm=float(through.norm()),
                through_to_full=float(through.norm()/full.norm().clamp_min(1e-12)),
                direct_through_cosine=float(torch.nn.functional.cosine_similarity(direct,through,dim=0)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--probabilities',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    enable_second_order_batched_attention(); torch.set_float32_matmul_precision('highest'); torch.manual_seed(3407)
    split=json.loads((V1/'manifests/development_split.json').read_text()); fit=set(split['development_fit'])
    inventory=json.loads((V1/'manifests/records.json').read_text())
    paths=filter_inventory_records([p for p in sorted((V1/'cache/train').rglob('*.npz')) if p.parts[-4] in fit],inventory,partition='train')
    labels={p:load_cached_arrays(p)['labels'] for p in paths}; weights=class_patient_record_weights(labels)
    patients=defaultdict(list)
    for p in paths: patients[p.parts[-4]].append(p)
    threshold,lookup=high_score_reference(args.probabilities)
    condition=tuple(item for item in CONDITIONS if item.name == state['v3_condition'])
    if len(condition) != 1: raise ValueError('unknown v3 condition')
    penalty=PenaltyScale(**state['penalty_scale'])
    config=InnerStepConfig(**state['inner_config']); rows=[]
    def components(stop, selected):
        model=load_source(Path(state['source']),detector_trainable=True); model.load_state_dict(state['model'])
        objective=build_objective(state['objective_name'],float(state['difficulty'])).cuda().eval(); objective.load_state_dict(state['objective']); objective.requires_grad_(True)
        functional=ConditionEnsemble([model]); ensemble=(EnsembleBandObjective if state['objective_name']=='band' else EnsembleMaskObjective)([objective],deduplicate_views=True)
        losses=[]
        for loss,_,_ in process_future_group(functional,ensemble,[patients[p] for p in selected],weights,config,condition,lookup,threshold,penalty,maximum_updates_per_patient=4,stop_encoder_through_inner=stop): losses.append(loss)
        total=torch.stack(losses).sum(); named=[(n,p) for n,p in [*model.named_parameters(),*((f'objective.{n}',p) for n,p in objective.named_parameters())] if p.requires_grad]
        grads=torch.autograd.grad(total,[p for _,p in named],allow_unused=True)
        return named,grads
    order=sorted(patients)
    for start in range(0,len(order),4):
        selected=order[start:start+4]
        fn,fg=components(False,selected); dn,dg=components(True,selected)
        if [n for n,_ in fn] != [n for n,_ in dn]: raise RuntimeError('parameter order mismatch')
        row={'patients':selected}
        for group,prefix in [('encoder10','backbone.encoder.layers.10.'),('encoder11','backbone.encoder.layers.11.'),('detector','detector.'),('ssl','objective.')]:
            row[group]=metrics(vector(fn,fg,prefix),vector(dn,dg,prefix))
        rows.append(row); print(json.dumps({'groups':len(rows),'patients':min(start+4,len(order))}),flush=True)
    summary={g:{k:float(np.median([r[g][k] for r in rows])) for k in rows[0][g]} for g in ['encoder10','encoder11','detector','ssl']}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps({'checkpoint':str(args.checkpoint),'groups':len(rows),'patients':len(order),'summary_medians':summary,'rows':rows},indent=2))

if __name__=='__main__': main()
