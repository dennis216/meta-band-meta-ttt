import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bfa.tusz_meta_ttt_v2 import joint_schedule
from bfa.tusz_meta_ttt_v2.update import InnerStepConfig


@pytest.mark.parametrize('long_windows',[32,150])
def test_joint_modes_keep_labels_once_and_f_terminal_step_is_a_noop(monkeypatch,long_windows):
    paths=[Path('/cache/train/p1/s/m/a.npz'),Path('/cache/train/p1/s/m/b.npz'),Path('/cache/train/p2/s/m/a.npz')]
    archives={}
    weights={}
    for i,(path,n) in enumerate(zip(paths,[long_windows,1,20],strict=True)):
        archives[path]=dict(signal=np.full((16,2000+400*(n-1)),.1*(i+1),dtype=np.float32),labels=np.linspace(0,1,n,dtype=np.float32),decision_end_s=10.+2*np.arange(n))
        weights[path]=np.ones(n,dtype=np.float32)
    monkeypatch.setattr(joint_schedule,'load_cached_arrays',lambda path:archives[path])
    valid_counts=[0,0]
    terminal_seen=[]
    def prepare(objective,signal,valid,*,prefix_fn,seeds,prefix_unique_fn=None):
        patients=signal.shape[0]//2
        torch.testing.assert_close(signal[:patients],signal[patients:],rtol=0,atol=0)
        for c in range(2): valid_counts[c]+=int((valid[c*patients:(c+1)*patients].sum(1)>0).sum())
        terminal_seen.append(bool(((valid[:patients].sum(1)==0)&(valid[patients:].sum(1)>0)).any()))
        return lambda feature_fn:(feature_fn(signal).square().mean((2,3,4))*valid).sum(1)/valid.sum(1).clamp_min(1)
    monkeypatch.setattr(joint_schedule,'prepare_ssl_lanes',prepare)
    class Functional:
        def __init__(self):
            self.source=[{f'backbone.encoder.layers.{b}.weight':torch.tensor([.7+.1*c],requires_grad=True) for b in (10,11)} for c in range(2)]
            self.functionals=[SimpleNamespace(prefix=lambda x:x)]
        def initial_lane_parameters(self,lanes):
            return {n:torch.stack([s[n] for s in self.source]).repeat_interleave(lanes//2,0) for n in self.source[0]}
        def prefix(self,x): return x
        def features_from_prefix(self,x,p): return x*p['backbone.encoder.layers.10.weight']+p['backbone.encoder.layers.11.weight']
        def detect_lanes(self,x): return x.sum(-1,keepdim=True)*0
    functional=Functional()
    value=0.
    updates=0
    for loss,count,accepted in joint_schedule.process_joint_group(functional,SimpleNamespace(name='mask'),[[paths[0],paths[1]],[paths[2]]],weights,InnerStepConfig(1e-5,{10:1.,11:1.},{10:1.,11:1.}),['future','current']):
        loss.backward()
        value+=float(loss.detach())
        updates+=count
        assert int(accepted)==count
        assert all(torch.isfinite(p.grad).all() for source in functional.source for p in source.values())
    n_chunks=math.ceil((10+2*(long_windows-1))/30)
    assert valid_counts==[n_chunks,n_chunks+3]
    assert updates==sum(valid_counts)
    assert value==pytest.approx(2*(long_windows+21)*math.log(2),rel=1e-6)
    assert any(terminal_seen)
