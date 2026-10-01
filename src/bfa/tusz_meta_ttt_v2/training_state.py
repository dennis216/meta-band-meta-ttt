"""Atomic checkpoints at outer-step boundaries; no partial patient state is saved."""
import os
import random
from pathlib import Path

import numpy as np
import torch


def atomic_save(value,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary)
    os.replace(temporary,path)


def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def detector_open(epoch,group_index,total_groups,freeze_fraction=.25):
    return epoch>1 or group_index>=max(1,int(total_groups*freeze_fraction)) if freeze_fraction else True


def validate_resume(saved,expected):
    mismatch={k:(saved.get(k),v) for k,v in expected.items() if saved.get(k)!=v}
    if mismatch: raise ValueError(f'Resume contract mismatch: {mismatch}')
