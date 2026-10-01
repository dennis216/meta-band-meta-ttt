from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class InnerStepConfig:
    relative_step: float
    block_reference_norms: dict[int, float]
    block_gradient_medians: dict[int, float]
    armijo: float = 1e-4
    trial_scales: tuple[float, ...] = (1.0, 0.5, 0.25)
    degenerate_fraction: float = 1e-2
    epsilon_fraction: float = 1e-3
    maximum_record_drift: float = 1e-2


@dataclass
class InnerStepResult:
    parameters: dict[str, torch.Tensor]
    accepted: bool
    trial_scale: float | None
    loss_before: torch.Tensor
    loss_after: torch.Tensor
    block_gradient_norms: dict[int, float]
    block_update_norms: dict[int, float]
    reason: str


def parameter_block(name: str) -> int:
    marker = "backbone.encoder.layers."
    if marker not in name:
        raise ValueError(f"not an adapted encoder parameter: {name}")
    return int(name.split(marker, 1)[1].split(".", 1)[0])


def _all_finite(values) -> bool:
    """One host synchronization for the whole parameter set, not one per tensor."""
    return bool(torch.stack([torch.isfinite(value).all() for value in values]).all())


def normalized_inner_step(
    parameters: Mapping[str, torch.Tensor],
    *,
    loss_fn: Callable[[Mapping[str, torch.Tensor]], torch.Tensor],
    record_initial: Mapping[str, torch.Tensor],
    config: InnerStepConfig,
    create_graph: bool,
) -> InnerStepResult:
    names = tuple(parameters)
    values = tuple(parameters.values())
    loss_before = loss_fn(parameters)
    gradients = torch.autograd.grad(
        loss_before, values, create_graph=create_graph, allow_unused=True
    )
    if any(gradient is None for gradient in gradients) or not _all_finite(gradients):
        return InnerStepResult(dict(parameters), False, None, loss_before, loss_before, {}, {}, "invalid_gradient")
    by_block: dict[int, list[tuple[str, torch.Tensor, torch.Tensor]]] = {}
    for name, value, gradient in zip(names, values, gradients, strict=True):
        by_block.setdefault(parameter_block(name), []).append((name, value, gradient))
    gradient_norms = {
        block: torch.sqrt(sum(gradient.float().square().sum() for _, _, gradient in entries))
        for block, entries in by_block.items()
    }
    for block, norm in gradient_norms.items():
        if float(norm.detach()) < config.degenerate_fraction * config.block_gradient_medians[block]:
            return InnerStepResult(
                dict(parameters), False, None, loss_before, loss_before,
                {key: float(value.detach()) for key, value in gradient_norms.items()}, {}, "degenerate_gradient"
            )
    directions: dict[str, torch.Tensor] = {}
    for block, entries in by_block.items():
        epsilon = config.epsilon_fraction * config.block_gradient_medians[block]
        denominator = torch.sqrt(gradient_norms[block].square() + epsilon**2)
        amplitude = config.relative_step * config.block_reference_norms[block]
        for name, _, gradient in entries:
            directions[name] = -amplitude * gradient / denominator
    slope = sum((gradient * directions[name]).float().sum() for name, gradient in zip(names, gradients, strict=True))
    detached_before = float(loss_before.detach())
    for scale in config.trial_scales:
        candidate = {name: value + scale * directions[name] for name, value in parameters.items()}
        finite = _all_finite(candidate.values())
        if not finite:
            continue
        drift_ok = True
        for block, entries in by_block.items():
            drift = torch.sqrt(sum((candidate[name] - record_initial[name]).float().square().sum() for name, _, _ in entries))
            limit = config.maximum_record_drift * config.block_reference_norms[block]
            if float(drift.detach()) > limit * (1 + 1e-4):
                drift_ok = False
                break
        if not drift_ok:
            continue
        # Armijo only makes a discrete acceptance decision. The candidate itself
        # retains its differentiable update graph for the future detection BCE.
        with torch.no_grad():
            loss_after = loss_fn(candidate)
        bound = detached_before + config.armijo * scale * float(slope.detach())
        if torch.isfinite(loss_after) and float(loss_after.detach()) <= bound:
            update_norms = {
                block: float(torch.sqrt(sum((scale * directions[name]).float().square().sum() for name, _, _ in entries)).detach())
                for block, entries in by_block.items()
            }
            return InnerStepResult(
                candidate, True, scale, loss_before, loss_after,
                {key: float(value.detach()) for key, value in gradient_norms.items()}, update_norms, "accepted"
            )
    return InnerStepResult(
        dict(parameters), False, None, loss_before, loss_before,
        {key: float(value.detach()) for key, value in gradient_norms.items()}, {}, "line_search_failed"
    )


def fixed_sgd_inner_step(
    parameters: Mapping[str, torch.Tensor],
    *,
    loss_fn: Callable[[Mapping[str, torch.Tensor]], torch.Tensor],
    record_initial: Mapping[str, torch.Tensor],
    learning_rate: float,
    config: InnerStepConfig,
    create_graph: bool,
) -> InnerStepResult:
    """Plain SGD control with the same validity, Armijo, and drift checks."""
    names = tuple(parameters)
    values = tuple(parameters.values())
    loss_before = loss_fn(parameters)
    gradients = torch.autograd.grad(
        loss_before, values, create_graph=create_graph, allow_unused=True
    )
    if any(gradient is None for gradient in gradients) or not _all_finite(gradients):
        return InnerStepResult(dict(parameters), False, None, loss_before, loss_before, {}, {}, "invalid_gradient")
    by_block: dict[int, list[tuple[str, torch.Tensor, torch.Tensor]]] = {}
    for name, value, gradient in zip(names, values, gradients, strict=True):
        by_block.setdefault(parameter_block(name), []).append((name, value, gradient))
    gradient_norms = {
        block: torch.sqrt(sum(gradient.float().square().sum() for _, _, gradient in entries))
        for block, entries in by_block.items()
    }
    for block, norm in gradient_norms.items():
        if float(norm.detach()) < config.degenerate_fraction * config.block_gradient_medians[block]:
            return InnerStepResult(
                dict(parameters), False, None, loss_before, loss_before,
                {key: float(value.detach()) for key, value in gradient_norms.items()}, {},
                "degenerate_gradient",
            )
    directions = {
        name: -learning_rate * gradient
        for name, gradient in zip(names, gradients, strict=True)
    }
    slope = sum(
        (gradient * directions[name]).float().sum()
        for name, gradient in zip(names, gradients, strict=True)
    )
    detached_before = float(loss_before.detach())
    for scale in config.trial_scales:
        candidate = {name: value + scale * directions[name] for name, value in parameters.items()}
        if not _all_finite(candidate.values()):
            continue
        drift_ok = True
        for block, entries in by_block.items():
            drift = torch.sqrt(sum(
                (candidate[name] - record_initial[name]).float().square().sum()
                for name, _, _ in entries
            ))
            if float(drift.detach()) > (
                config.maximum_record_drift * config.block_reference_norms[block] * (1 + 1e-4)
            ):
                drift_ok = False
                break
        if not drift_ok:
            continue
        with torch.no_grad():
            loss_after = loss_fn(candidate)
        bound = detached_before + config.armijo * scale * float(slope.detach())
        if torch.isfinite(loss_after) and float(loss_after.detach()) <= bound:
            return InnerStepResult(
                candidate,
                True,
                scale,
                loss_before,
                loss_after,
                {key: float(value.detach()) for key, value in gradient_norms.items()},
                {
                    block: float(torch.sqrt(sum(
                        (scale * directions[name]).float().square().sum()
                        for name, _, _ in entries
                    )).detach())
                    for block, entries in by_block.items()
                },
                "accepted",
            )
    return InnerStepResult(
        dict(parameters), False, None, loss_before, loss_before,
        {key: float(value.detach()) for key, value in gradient_norms.items()}, {},
        "line_search_failed",
    )


def reattach_initialization(
    carried: Mapping[str, torch.Tensor], initial: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Retain carried values while reconnecting every truncated segment to the initialization."""
    return {name: initial[name] + (value - initial[name]).detach() for name, value in carried.items()}


def packed_normalized_inner_step(
    parameters: Mapping[str, torch.Tensor], *,
    loss_fn: Callable[[Mapping[str, torch.Tensor]], torch.Tensor],
    record_initial: Mapping[str, torch.Tensor],
    config: InnerStepConfig, create_graph: bool,
) -> InnerStepResult:
    """Block-packed equivalent update, reducing launches for small parameter tensors.

    This is opt-in until numerical and whole-workload comparisons pass. No
    denominator or update direction is detached from the Meta computation graph.
    """
    names = tuple(parameters)
    loss_before = loss_fn(parameters)
    gradients = torch.autograd.grad(loss_before, tuple(parameters.values()), create_graph=create_graph, allow_unused=True)
    if any(g is None for g in gradients):
        return InnerStepResult(dict(parameters), False, None, loss_before, loss_before, {}, {}, 'invalid_gradient')
    groups = {}
    for name, gradient in zip(names, gradients, strict=True):
        groups.setdefault(parameter_block(name), []).append((name, gradient))
    packed = {}
    for block, entries in groups.items():
        g = torch.cat([gradient.reshape(-1) for _, gradient in entries])
        value = torch.cat([parameters[name].reshape(-1) for name, _ in entries])
        initial = torch.cat([record_initial[name].reshape(-1) for name, _ in entries])
        packed[block] = (g, value, initial)
    blocks = list(packed)
    norms = torch.stack([g.float().square().sum().sqrt() for g, _, _ in packed.values()])
    norm_values = norms.detach().tolist()
    norm_report = dict(zip(blocks, norm_values, strict=True))
    # Finite norm also detects nonfinite coordinates and squared-norm overflow.
    if not all(math.isfinite(n) for n in norm_values):
        return InnerStepResult(dict(parameters), False, None, loss_before, loss_before, {}, {}, 'invalid_gradient')
    if any(norm_report[b] < config.degenerate_fraction * config.block_gradient_medians[b] for b in blocks):
        return InnerStepResult(dict(parameters), False, None, loss_before, loss_before, norm_report, {}, 'degenerate_gradient')
    directions = {}
    for index, block in enumerate(blocks):
        g, _, _ = packed[block]
        epsilon = config.epsilon_fraction * config.block_gradient_medians[block]
        directions[block] = -config.relative_step * config.block_reference_norms[block] * g / (norms[index].square() + epsilon**2).sqrt()
    slope = sum((packed[b][0] * directions[b]).float().sum() for b in blocks)
    before_value, slope_value = torch.stack([loss_before.detach(), slope.detach()]).tolist()
    for scale in config.trial_scales:
        trial = {b: packed[b][1] + scale * directions[b] for b in blocks}
        drift_values = torch.stack([(trial[b] - packed[b][2]).float().square().sum().sqrt() for b in blocks]).detach().tolist()
        if any(not math.isfinite(d) or d > config.maximum_record_drift * config.block_reference_norms[b] * (1 + 1e-4) for b, d in zip(blocks, drift_values, strict=True)):
            continue
        candidate = {}
        for b in blocks:
            entries = groups[b]
            pieces = trial[b].split([parameters[name].numel() for name, _ in entries])
            candidate.update((name, piece.view_as(parameters[name])) for (name, _), piece in zip(entries, pieces, strict=True))
        candidate = {name: candidate[name] for name in names}
        with torch.no_grad():
            loss_after = loss_fn(candidate)
        after_value = float(loss_after)
        if math.isfinite(after_value) and after_value <= before_value + config.armijo * scale * slope_value:
            update_values = torch.stack([(scale * directions[b]).float().square().sum().sqrt() for b in blocks]).detach().tolist()
            return InnerStepResult(candidate, True, scale, loss_before, loss_after, norm_report, dict(zip(blocks, update_values, strict=True)), 'accepted')
    return InnerStepResult(dict(parameters), False, None, loss_before, loss_before, norm_report, {}, 'line_search_failed')
