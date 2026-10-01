import importlib.util
from pathlib import Path


def test_mechanism_names_do_not_parse_unrelated_flags_as_learning_rates():
    path = Path(__file__).resolve().parents[2] / 'scripts/336_run_tusz_mechanisms_v2.py'
    spec = importlib.util.spec_from_file_location('mechanisms_queue', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.mechanism_suffix({'kind': 'stop_encoder_through_inner', 'extra': ['--stop-encoder-through-inner']}) == '_stop_encoder_through_inner'
    assert module.mechanism_suffix({'kind': 'detector_open', 'extra': ['--detector-freeze-fraction', '0']}) == '_detector_open'
    assert module.mechanism_suffix({'kind': 'fixed_sgd', 'extra': ['--fixed-inner-lr', '0.0001']}) == '_sgd0.0001'
