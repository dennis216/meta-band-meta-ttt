import random

import numpy as np
import pytest
import torch

from bfa.tusz_meta_ttt_v2.training_state import atomic_save,capture_rng,restore_rng,detector_open,validate_resume


def test_rng_checkpoint_roundtrip(tmp_path):
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state=capture_rng()
    expected=(random.random(),np.random.random(),torch.rand(3))
    path=tmp_path/'last.pt'
    atomic_save(state,path)
    restore_rng(torch.load(path,weights_only=False))
    assert random.random()==expected[0]
    assert np.random.random()==expected[1]
    torch.testing.assert_close(torch.rand(3),expected[2],rtol=0,atol=0)
    assert not path.with_suffix('.pt.tmp').exists()


def test_detector_warmup_and_resume_contract():
    assert not detector_open(1,0,116)
    assert not detector_open(1,28,116)
    assert detector_open(1,29,116)
    assert detector_open(2,0,116)
    assert detector_open(1,0,116,0)
    validate_resume({'source':'a','patients':463},{'source':'a','patients':463})
    with pytest.raises(ValueError): validate_resume({'source':'a'},{'source':'b'})
