from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from bfa.evaluation.eventize import Event
from bfa.tusz_meta_ttt.data import intervals_match
from bfa.tusz_meta_ttt.labels import decision_interval_labels
from bfa.tusz_meta_ttt.model import TUSZDetector, balanced_soft_bce
from bfa.tusz_meta_ttt.objectives import (
    BandObjective,
    LearnedObjective,
    MaskObjective,
    TemporalObjective,
    remove_frequency_band,
)
from bfa.tusz_meta_ttt.preprocessing import CausalTUSZPreprocessor, frame_windows
from bfa.tusz_meta_ttt.scoring import (
    choose_threshold,
    consecutive_eventize,
    score_at_threshold,
    score_record,
)


class IdentityBackbone(nn.Module):
    def forward(self, signal):
        return signal


def feature_fn(signal):
    return signal


def test_soft_labels_measure_recent_interval_occupancy():
    labels = decision_interval_labels(np.array([10.0, 12.0, 14.0]), [(9.0, 11.0)])
    np.testing.assert_allclose(labels, [0.5, 0.5, 0.0])


def test_detailed_and_binary_interval_comparison_has_explicit_tolerance():
    assert intervals_match(((1.0, 2.0),), ((1.01, 2.01),))
    assert not intervals_match(((1.0, 2.0),), ((1.0, 3.0),))


def test_detector_shape_and_balanced_loss():
    detector = TUSZDetector(IdentityBackbone(), dropout=0)
    logits = detector(torch.zeros(3, 16, 10, 200))
    assert logits.shape == (3,)
    assert torch.isfinite(balanced_soft_bce(logits, torch.tensor([0.0, 0.5, 1.0])))


def test_frozen_detector_keeps_backbone_in_eval_mode():
    detector = TUSZDetector(IdentityBackbone(), freeze_backbone=True)
    detector.train()
    assert detector.training and not detector.backbone.training
    assert not any(parameter.requires_grad for parameter in detector.backbone.parameters())


@pytest.mark.parametrize("objective", [BandObjective(), TemporalObjective(), MaskObjective(), LearnedObjective()])
def test_ssl_objectives_are_finite_and_differentiable(objective):
    signal = torch.randn(2, 16, 10, 200, requires_grad=True)
    loss = objective.loss(signal, feature_fn=feature_fn, transform_seed=17)
    gradient = torch.autograd.grad(loss, signal)[0]
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert gradient.shape == signal.shape and torch.isfinite(gradient).all()


def test_learned_objective_batched_views_match_serial_definition():
    torch.manual_seed(23)
    objective = LearnedObjective()
    signal = torch.randn(2, 16, 10, 200)
    batched = objective.loss(signal, feature_fn=feature_fn)
    original = feature_fn(signal).mean(dim=(1, 2))
    serial = []
    bands = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
    for index, band in enumerate(bands):
        transformed = feature_fn(remove_frequency_band(signal, band)).mean(dim=(1, 2))
        identity = nn.functional.one_hot(torch.full((2,), index), num_classes=5).float()
        serial.append(nn.functional.softplus(objective.network(torch.cat([original, transformed, identity], dim=1))))
    torch.testing.assert_close(batched, torch.stack(serial).mean())


def test_band_removal_suppresses_target_frequency():
    time = torch.arange(200) / 200
    signal = torch.sin(2 * torch.pi * 10 * time).reshape(1, 1, 1, 200)
    removed = remove_frequency_band(signal, (8, 13))
    assert removed.square().mean() < signal.square().mean() * 1e-4


def test_causal_preprocessor_emitted_prefix_is_future_invariant():
    rng = np.random.default_rng(17)
    prefix = rng.normal(size=(16, 1000))
    future_a = rng.normal(size=(16, 1000))
    future_b = rng.normal(size=(16, 1000)) * 100
    left = CausalTUSZPreprocessor(250)
    right = CausalTUSZPreprocessor(250)
    prefix_left = left.update(prefix)
    prefix_right = right.update(prefix)
    np.testing.assert_array_equal(prefix_left, prefix_right)
    left.update(future_a)
    right.update(future_b)
    np.testing.assert_array_equal(prefix_left, prefix_right)


def test_window_framing_uses_dense_two_second_stride():
    signal = np.zeros((16, 2800), dtype=np.float32)
    assert frame_windows(signal).shape == (3, 16, 2000)


def test_alarm_starts_at_second_crossing_without_backdating():
    events = consecutive_eventize(
        np.array([10.0, 12.0, 14.0, 16.0]),
        np.array([0.0, 1.0, 1.0, 0.0]),
        threshold=0.2,
        ema_alpha=1.0,
    )
    assert events[0].start_s == 14.0


def test_alarm_ends_at_second_below_decision_and_clamps_record_tail():
    events = consecutive_eventize(
        np.array([10.0, 12.0, 14.0, 16.0, 18.0]),
        np.array([1.0, 1.0, 0.0, 0.0, 1.0]),
        threshold=0.5,
        ema_alpha=1.0,
        record_end_s=18.5,
    )
    assert events == [Event(12.0, 16.0, 1.0)]


def test_threshold_candidates_are_drawn_from_per_record_ema(monkeypatch):
    seen = []
    original = np.quantile

    def capture(values, quantiles):
        seen.append(np.asarray(values).copy())
        return original(values, quantiles)

    monkeypatch.setattr(np, "quantile", capture)
    records = [{
        "times": np.array([10.0, 12.0, 14.0]),
        "probabilities": np.array([0.0, 0.9, 0.0]),
        "truths": [],
        "duration_s": 20.0,
    }]
    choose_threshold(records, target_sensitivity=0.8, quantiles=3)
    np.testing.assert_allclose(seen[0], [0.0, 0.3, 0.2])


def test_short_record_has_no_alarm():
    assert consecutive_eventize(np.array([]), np.array([]), threshold=0.5) == []
    assert consecutive_eventize(np.array([10.0]), np.array([0.9]), threshold=0.5) == []


def test_threshold_selector_enforces_sensitivity():
    records = [{
        "times": np.arange(10.0, 30.0, 2.0),
        "probabilities": np.array([0, 0, 0.9, 0.9, 0, 0, 0, 0, 0, 0], dtype=float),
        "truths": [(14.0, 20.0)],
        "duration_s": 30.0,
    }]
    choice = choose_threshold(records, target_sensitivity=0.8, quantiles=21)
    assert choice.reachable and choice.metrics is not None
    assert choice.metrics.sensitivity >= 0.8


def test_fixed_threshold_scoring_matches_calibrated_metrics():
    records = [{
        "times": np.arange(10.0, 30.0, 2.0),
        "probabilities": np.array([0, 0, 0.9, 0.9, 0, 0, 0, 0, 0, 0], dtype=float),
        "truths": [(14.0, 20.0)],
        "duration_s": 30.0,
    }]
    choice = choose_threshold(records, target_sensitivity=0.8, quantiles=21)
    assert choice.threshold is not None and choice.metrics is not None
    assert score_at_threshold(records, choice.threshold) == choice.metrics


def test_false_alarm_time_includes_tail_of_matched_alarm():
    metrics = score_record([Event(5.0, 20.0, 1.0)], [(10.0, 15.0)], duration_s=60.0)
    assert metrics.false_alarms == 0
    assert metrics.false_alarm_minutes_per_hour > 0
