from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from bfa.tusz_meta_ttt_v2.objectives import (
    BandObjectiveV2,
    MaskObjectiveV2,
    TemporalObjectiveV2,
)
from bfa.tusz_meta_ttt_v2.protocol import (
    RecordLocalBatchSampler,
    assert_train_cache_paths,
    availability_times,
    chunk_rows,
    class_patient_record_weights,
    stable_transform_seed,
    weight_audit,
    validate_evaluation_request,
)


def test_training_entry_rejects_dev_eval_and_escaped_paths(tmp_path):
    train = tmp_path / "cache/train"
    dev = tmp_path / "cache/dev"
    evaluation = tmp_path / "cache/eval"
    for directory in [train, dev, evaluation]:
        directory.mkdir(parents=True)
    accepted = train / "record.npz"
    accepted.touch()
    assert_train_cache_paths([accepted], train)
    for rejected in [dev / "record.npz", evaluation / "record.npz", train / "../dev/record.npz"]:
        with pytest.raises(ValueError, match="outside Train"):
            assert_train_cache_paths([rejected], train)


def test_eval_cannot_recalibrate_or_use_nondev_thresholds(tmp_path):
    dev = tmp_path / "dev/summary.json"
    dev.parent.mkdir()
    dev.touch()
    validate_evaluation_request("eval", False, dev)
    with pytest.raises(ValueError, match="fixed Dev thresholds"):
        validate_evaluation_request("eval", True, None)
    with pytest.raises(ValueError, match="fixed Dev thresholds"):
        validate_evaluation_request("eval", False, None)
    with pytest.raises(ValueError, match="Dev summary"):
        validate_evaluation_request("eval", False, tmp_path / "eval/summary.json")
from bfa.tusz_meta_ttt_v2.update import (
    InnerStepConfig,
    fixed_sgd_inner_step,
    normalized_inner_step,
    reattach_initialization,
)


def test_real_time_chunks_keep_boundary_in_preceding_chunk():
    times = np.arange(10.0, 72.0, 2.0)
    chunks = chunk_rows(times)
    assert chunks[0].rows == tuple(range(11))  # 10 through 30 seconds
    assert chunks[1].rows == tuple(range(11, 26))  # 32 through 60 seconds
    assert chunks[2].rows == tuple(range(26, 31))


def test_current_availability_uses_actual_partial_chunk_end():
    times = np.arange(10.0, 73.0, 2.0)
    current = availability_times(times, "current")
    assert current[0] == 30.0
    assert current[-1] == 72.0
    np.testing.assert_array_equal(availability_times(times, "future"), times)


def test_global_weights_balance_class_patient_record_and_window():
    labels = {
        Path("cache/train/p1/s/m/r1.npz"): np.array([0.0, 0.0, 1.0]),
        Path("cache/train/p1/s/m/r2.npz"): np.array([0.0, 1.0, 1.0]),
        Path("cache/train/p2/s/m/r3.npz"): np.array([0.0, 0.0, 0.0]),
    }
    weights = class_patient_record_weights(labels)
    audit = weight_audit(weights, labels)
    assert audit == pytest.approx({"background": 0.5, "seizure": 0.5, "total": 1.0})
    # Patient p1 receives half of background class mass, split equally over its two records.
    assert weights[next(iter(labels))][:2].sum() == pytest.approx(0.125)


def test_record_local_sampler_is_exhaustive_without_mixing_files():
    class Dataset:
        def __init__(self):
            self.rows = [
                (0, 0, "p1", False),
                (0, 1, "p1", False),
                (0, 2, "p1", False),
                (1, 0, "p2", False),
                (1, 1, "p2", False),
            ]

        def __len__(self):
            return len(self.rows)

    dataset = Dataset()
    batches = list(RecordLocalBatchSampler(dataset, batch_size=2, samples=5, seed=17))
    assert sorted(index for batch in batches for index in batch) == list(range(5))
    assert all(len({dataset.rows[index][0] for index in batch}) == 1 for batch in batches)


def test_transform_seed_depends_on_identity_not_processing_order():
    first = stable_transform_seed(3407, "patient/record", "band", 3)
    second = stable_transform_seed(3407, "patient/record", "band", 3)
    assert first == second
    assert first != stable_transform_seed(3407, "patient/other", "band", 3)
    assert first != stable_transform_seed(3407, "patient/record", "band", 4)


def test_normalized_update_has_requested_block_norm_and_reduces_loss():
    parameters = {
        "backbone.encoder.layers.10.weight": torch.tensor([1.0, -1.0], requires_grad=True),
        "backbone.encoder.layers.11.weight": torch.tensor([0.5], requires_grad=True),
    }
    config = InnerStepConfig(
        relative_step=1e-2,
        block_reference_norms={10: 2.0, 11: 1.0},
        block_gradient_medians={10: 1.0, 11: 1.0},
    )
    result = normalized_inner_step(
        parameters,
        loss_fn=lambda state: sum(value.square().sum() for value in state.values()),
        record_initial=parameters,
        config=config,
        create_graph=True,
    )
    assert result.accepted
    assert result.block_update_norms[10] == pytest.approx(0.02, rel=1e-5)
    assert result.block_update_norms[11] == pytest.approx(0.01, rel=1e-5)
    assert result.loss_after < result.loss_before


def test_fixed_sgd_control_keeps_shared_acceptance_checks():
    parameters = {
        "backbone.encoder.layers.10.weight": torch.tensor([1.0], requires_grad=True),
        "backbone.encoder.layers.11.weight": torch.tensor([0.5], requires_grad=True),
    }
    config = InnerStepConfig(
        relative_step=1e-5,
        block_reference_norms={10: 1.0, 11: 1.0},
        block_gradient_medians={10: 1.0, 11: 1.0},
    )
    result = fixed_sgd_inner_step(
        parameters,
        loss_fn=lambda state: sum(value.square().sum() for value in state.values()),
        record_initial=parameters,
        learning_rate=1e-3,
        config=config,
        create_graph=True,
    )
    assert result.accepted
    assert result.loss_after < result.loss_before


def test_normalized_inner_step_meta_gradient_matches_finite_difference():
    config = InnerStepConfig(
        relative_step=1e-3,
        block_reference_norms={10: 1.0, 11: 1.0},
        block_gradient_medians={10: 1.0, 11: 1.0},
        maximum_record_drift=0.1,
    )

    def outer(value: float, requires_grad: bool = False):
        first = torch.tensor([value], dtype=torch.float64, requires_grad=True)
        second = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
        parameters = {
            "backbone.encoder.layers.10.weight": first,
            "backbone.encoder.layers.11.weight": second,
        }
        result = normalized_inner_step(
            parameters,
            loss_fn=lambda state: (state[next(iter(state))] - 0.2).square().sum()
            + (state[tuple(state)[1]] + 0.1).square().sum(),
            record_initial=parameters,
            config=config,
            create_graph=requires_grad,
        )
        return sum(parameter.square().sum() for parameter in result.parameters.values()), first

    value, leaf = outer(1.1, True)
    analytical = torch.autograd.grad(value, leaf)[0].item()
    epsilon = 1e-5
    numerical = (outer(1.1 + epsilon)[0].item() - outer(1.1 - epsilon)[0].item()) / (2 * epsilon)
    assert analytical == pytest.approx(numerical, rel=1e-4, abs=1e-5)


def test_reattach_preserves_value_and_restores_initialization_gradient():
    initial = {
        "backbone.encoder.layers.10.weight": torch.tensor(1.0, requires_grad=True)
    }
    carried = {next(iter(initial)): initial[next(iter(initial))] * 3}
    reattached = reattach_initialization(carried, initial)
    torch.testing.assert_close(reattached[next(iter(initial))], carried[next(iter(initial))])
    gradient = torch.autograd.grad(reattached[next(iter(initial))].square(), tuple(initial.values()))[0]
    torch.testing.assert_close(gradient, torch.tensor(6.0))


def test_second_truncated_segment_still_updates_encoder_initialization():
    name = "backbone.encoder.layers.10.weight"
    initial = {name: torch.tensor(0.8, dtype=torch.float64, requires_grad=True)}
    # The first segment is already committed when the second begins.
    first_carried = {name: initial[name] - 0.1 * (initial[name] - 0.2)}
    second_start = reattach_initialization(first_carried, initial)
    second_fast = {name: second_start[name] - 0.1 * (second_start[name] + 0.4)}
    future_bce_surrogate = (second_fast[name] - 1.0).square()
    gradient = torch.autograd.grad(future_bce_surrogate, initial[name])[0]
    assert torch.isfinite(gradient)
    assert abs(float(gradient)) > 1e-6
    # The first segment's history is detached: the retained gradient is the
    # straight-through initialization path used by later truncated segments.
    expected = 2 * (float(second_fast[name]) - 1.0) * 0.9
    assert float(gradient) == pytest.approx(expected)


@pytest.mark.parametrize(
    "objective",
    [BandObjectiveV2(0.5), TemporalObjectiveV2(5), MaskObjectiveV2(5)],
)
def test_v2_objectives_are_finite_and_reach_signal(objective):
    signal = torch.randn(2, 16, 10, 200, requires_grad=True)
    loss = objective.loss(signal, feature_fn=lambda value: value, transform_seed=17)
    gradient = torch.autograd.grad(loss, signal)[0]
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()


@pytest.mark.parametrize(
    "objective",
    [BandObjectiveV2(0.5), TemporalObjectiveV2(5), MaskObjectiveV2(5)],
)
def test_prepared_ssl_loss_matches_direct_path(objective):
    objective.eval()
    signal = torch.randn(2, 16, 10, 200)
    direct = objective.loss(signal, feature_fn=lambda value: value, transform_seed=42)
    prepared = objective.prepare_loss(
        signal, prefix_fn=lambda value: value, transform_seed=42
    )
    cached = prepared(lambda value: value)
    torch.testing.assert_close(cached, direct)


@pytest.mark.parametrize(
    "objective",
    [BandObjectiveV2(0.5), TemporalObjectiveV2(5), MaskObjectiveV2(5)],
)
def test_ssl_loss_is_microbatch_invariant_with_per_window_seeds(objective):
    objective.eval()
    signal = torch.randn(4, 16, 10, 200)
    seeds = [101, 202, 303, 404]
    full = objective.loss(signal, feature_fn=lambda value: value, transform_seed=seeds)
    halves = torch.stack([
        objective.loss(
            signal[:2], feature_fn=lambda value: value, transform_seed=seeds[:2]
        ),
        objective.loss(
            signal[2:], feature_fn=lambda value: value, transform_seed=seeds[2:]
        ),
    ]).mean()
    torch.testing.assert_close(full, halves)


def test_mask_hides_all_channels_at_selected_time_positions():
    objective = MaskObjectiveV2(5)
    signal = torch.ones(2, 16, 10, 200)
    observed = []

    def capture(value):
        observed.append(value.detach().clone())
        return value

    objective.loss(signal, feature_fn=capture, transform_seed=9)
    masked = observed[0]
    zero_positions = (masked == 0).all(dim=(1, 3))
    assert torch.equal(zero_positions.sum(1), torch.tensor([5, 5]))


def test_masked_target_cannot_reenter_either_input_branch():
    objective = MaskObjectiveV2(5)
    signal = torch.randn(1, 16, 10, 200)
    seen = []

    def capture(value):
        seen.append(value.detach().clone())
        return value

    objective.loss(signal, feature_fn=capture, transform_seed=3407)
    masked_positions = (seen[-1] == 0).all(dim=(1, 3))[0]
    changed = signal.clone()
    changed[:, :, masked_positions, :] += 100.0
    objective.loss(changed, feature_fn=capture, transform_seed=3407)
    torch.testing.assert_close(seen[-1], seen[-2], rtol=0, atol=0)
