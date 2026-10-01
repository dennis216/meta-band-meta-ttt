import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bfa.tusz_meta_ttt_v2.update import InnerStepConfig


@pytest.mark.parametrize('mode',['future','current'])
@pytest.mark.parametrize('compact',[False,True])
@pytest.mark.parametrize('long_windows',[32,150])
def test_dense_lane_schedule_keeps_each_label_once_and_handles_record_resets(monkeypatch,mode,compact,long_windows):
    path=Path(__file__).resolve().parents[2]/'scripts/352_benchmark_tusz_patient_lanes_v2.py'
    spec=importlib.util.spec_from_file_location('lane_schedule',path)
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    records={}
    weights={}
    paths=[Path('/cache/train/p1/s/m/long.npz'),Path('/cache/train/p1/s/m/short.npz'),Path('/cache/train/p2/s/m/long.npz')]
    for index,(p,n) in enumerate(zip(paths,[long_windows,1,20],strict=True)):
        records[p]=dict(signal=np.full((16,2000+(n-1)*400),.1*(index+1),dtype=np.float32),labels=np.linspace(0,1,n,dtype=np.float32),decision_end_s=np.arange(n)*2.+10.)
        weights[p]=np.ones(n,dtype=np.float32)
    monkeypatch.setattr(runner,'load_cached_arrays',lambda p:records[p])
    supports=[]
    def prepare(objective,signal,valid,*,prefix_fn,seeds,prefix_unique_fn=None):
        supports.append(valid.sum(1).tolist())
        def loss(feature_fn):
            features=feature_fn(signal)
            per_window=features.square().mean((2,3,4))
            return (per_window*valid).sum(1)/valid.sum(1).clamp_min(1)
        return loss
    monkeypatch.setattr(runner,'prepare_ssl_lanes',prepare)
    class Functional:
        def __init__(self):
            self.params={f'backbone.encoder.layers.{b}.weight':torch.tensor([.7],requires_grad=True) for b in (10,11)}
            detector=torch.nn.Linear(2000,1)
            torch.nn.init.zeros_(detector.weight)
            torch.nn.init.zeros_(detector.bias)
            self.model=SimpleNamespace(detector=detector)
        def initial_fast_parameters(self,detach=False): return self.params
        def prefix(self,x): return x
        def features_from_prefix(self,x,p): return x*p['backbone.encoder.layers.10.weight']+p['backbone.encoder.layers.11.weight']
    functional=Functional()
    total=0.
    updates=0
    accepted=0.
    for loss,count,success in runner.process_group(functional,SimpleNamespace(name='mask'),[[paths[0],paths[1]],[paths[2]]],weights,InnerStepConfig(1e-5,{10:1.,11:1.},{10:1.,11:1.}),mode,compact_lanes=compact):
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in functional.params.values())
        total+=float(loss.detach())
        updates+=count
        accepted+=float(success)
    expected_updates=math.ceil((10+2*(long_windows-1))/30)+(0 if mode=='future' else 3)
    assert total==pytest.approx((long_windows+21)*math.log(2),rel=1e-6)
    assert updates==expected_updates
    assert accepted==expected_updates
    assert supports[0]==[11.,11.]
