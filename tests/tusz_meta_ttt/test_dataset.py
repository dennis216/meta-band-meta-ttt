from __future__ import annotations

from pathlib import Path

import numpy as np

from bfa.tusz_meta_ttt.dataset import (
    CachedTUSZWindows,
    PatientClassSampler,
    PatientUniformSampler,
)


def make_cache(path: Path, labels: list[float]) -> None:
    path.parent.mkdir(parents=True)
    samples = 2000 + 400 * (len(labels) - 1)
    np.savez(path, signal=np.zeros((16, samples), np.float32), labels=np.array(labels, np.float32))


def test_cached_dataset_and_patient_sampler(tmp_path):
    left = tmp_path / "train" / "p1" / "s1" / "m" / "r.npz"
    right = tmp_path / "train" / "p2" / "s1" / "m" / "r.npz"
    make_cache(left, [0, 1])
    make_cache(right, [0, 1])
    dataset = CachedTUSZWindows([left, right])
    assert dataset[1]["signal"].shape == (16, 10, 200)
    sampler = PatientClassSampler(dataset, samples=100, seed=17, positive_fraction=0.25)
    positives = sum(dataset.rows[index][3] for index in sampler)
    assert 10 <= positives <= 40
    uniform = PatientUniformSampler(dataset, samples=20, seed=17)
    assert len(list(uniform)) == 20


def test_positive_sampling_balances_events_before_windows(tmp_path):
    path = tmp_path / "train" / "p1" / "s1" / "m" / "r.npz"
    make_cache(path, [1, 1, 0, 1])
    dataset = CachedTUSZWindows([path])
    sampler = PatientClassSampler(
        dataset, samples=1000, seed=17, positive_fraction=1.0
    )
    event_counts = {}
    for index in sampler:
        event = dataset.event_keys[index]
        event_counts[event] = event_counts.get(event, 0) + 1
    fraction = next(iter(event_counts.values())) / 1000
    assert len(event_counts) == 2
    assert 0.4 < fraction < 0.6


def test_sampler_offset_reproduces_one_continuous_draw_sequence(tmp_path):
    path = tmp_path / "train" / "p1" / "s1" / "m" / "r.npz"
    make_cache(path, [0, 1, 1, 0, 1])
    dataset = CachedTUSZWindows([path])
    full = list(PatientClassSampler(dataset, samples=100, seed=42))
    left = list(PatientClassSampler(dataset, samples=40, seed=42))
    right = list(PatientClassSampler(dataset, samples=60, seed=42, offset=40))
    assert left + right == full
