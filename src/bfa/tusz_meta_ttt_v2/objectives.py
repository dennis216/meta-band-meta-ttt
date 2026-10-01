from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import torch
from torch import nn

BANDS_HZ = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))


class SSLObjectiveV2(nn.Module, ABC):
    name: str

    @abstractmethod
    def loss(
        self,
        signal: torch.Tensor,
        *,
        feature_fn: Callable[[torch.Tensor], torch.Tensor],
        transform_seed: int,
    ) -> torch.Tensor: ...

    @abstractmethod
    def metrics(
        self,
        signal: torch.Tensor,
        *,
        feature_fn: Callable[[torch.Tensor], torch.Tensor],
        transform_seed: int,
    ) -> dict[str, torch.Tensor]: ...

    @abstractmethod
    def prepare_loss(
        self,
        signal: torch.Tensor,
        *,
        prefix_fn: Callable[[torch.Tensor], torch.Tensor],
        transform_seed: int,
    ) -> Callable[[Callable[[torch.Tensor], torch.Tensor]], torch.Tensor]:
        """Prepare deterministic views and their frozen-prefix features exactly once."""
        ...


TransformSeed = int | Sequence[int] | torch.Tensor


def _seed_list(seed: TransformSeed, count: int) -> list[int]:
    if isinstance(seed, torch.Tensor):
        values = seed.detach().cpu().reshape(-1).tolist()
    elif isinstance(seed, int):
        values = [seed + index for index in range(count)]
    else:
        values = list(seed)
    if len(values) != count:
        raise ValueError(f"expected {count} per-window transform seeds, received {len(values)}")
    return [int(value) % (2**63 - 1) for value in values]


def _uniform_from_seeds(
    seed: TransformSeed, count: int, streams: int, device: torch.device
) -> torch.Tensor:
    values = torch.tensor(_seed_list(seed, count), dtype=torch.int64)
    outputs = []
    # A vectorized integer mixer provides deterministic, order-independent uniforms.
    for stream in range(streams):
        mixed = values ^ (0x4F1BBCDCBFA54001 * (stream + 1) & 0x7FFFFFFFFFFFFFFF)
        mixed = mixed * 6364136223846793005 + 1442695040888963407
        mixed = mixed ^ (mixed >> 21)
        outputs.append((mixed & 0x1FFFFFFFFFFFFF).double() / float(2**53))
    return torch.stack(outputs, dim=1).to(device=device, dtype=torch.float32)


def attenuate_band(
    signal: torch.Tensor,
    band: tuple[float, float],
    depth: torch.Tensor,
    *,
    sampling_hz: int = 200,
    transition_hz: float = 0.5,
) -> torch.Tensor:
    """Smoothly attenuate one band over a reflected ten-second signal and restore RMS."""
    padding = sampling_hz
    shape = signal.shape
    continuous = signal.float().reshape(shape[0], shape[1], -1)
    padded = nn.functional.pad(continuous, (padding, padding), mode="reflect")
    spectrum = torch.fft.rfft(padded, dim=-1)
    frequencies = torch.fft.rfftfreq(padded.shape[-1], d=1 / sampling_hz).to(signal.device)
    left = torch.clamp((frequencies - (band[0] - transition_hz)) / transition_hz, 0, 1)
    right = torch.clamp(((band[1] + transition_hz) - frequencies) / transition_hz, 0, 1)
    window = torch.minimum(left, right)
    scale = 1 - depth[:, None] * window[None, :]
    transformed = torch.fft.irfft(spectrum * scale[:, None, :], n=padded.shape[-1], dim=-1)
    transformed = transformed[..., padding:-padding].reshape(shape)
    source_rms = signal.float().square().mean(dim=(-2, -1), keepdim=True).sqrt()
    target_rms = transformed.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
    return (transformed * source_rms / target_rms).to(signal.dtype)


class BandObjectiveV2(SSLObjectiveV2):
    name = "band"

    def __init__(self, depth: float) -> None:
        super().__init__()
        self.depth = float(depth)
        self.head = nn.Linear(2000, len(BANDS_HZ))

    def _batch(self, signal: torch.Tensor, seed: TransformSeed) -> tuple[torch.Tensor, torch.Tensor]:
        views = []
        random_values = _uniform_from_seeds(
            seed, signal.shape[0], len(BANDS_HZ), signal.device
        )
        for band_index, band in enumerate(BANDS_HZ):
            jitter = random_values[:, band_index] * 0.2 - 0.1
            depth = torch.clamp(self.depth + jitter, 0.01, 0.99)
            views.append(attenuate_band(signal, band, depth))
        labels = torch.arange(len(BANDS_HZ), device=signal.device).repeat_interleave(signal.shape[0])
        return torch.cat(views), labels

    def metrics(self, signal, *, feature_fn, transform_seed):
        views, labels = self._batch(signal, transform_seed)
        logits = self.head(feature_fn(views).mean(dim=1).flatten(1))
        loss = nn.functional.cross_entropy(logits, labels)
        prediction = logits.argmax(1)
        continuous = views.float().reshape(views.shape[0], views.shape[1], -1)
        spectrum = torch.fft.rfft(continuous, dim=-1).abs().square().mean(1)
        frequencies = torch.fft.rfftfreq(continuous.shape[-1], d=1 / 200).to(signal.device)
        energies = torch.stack([
            spectrum[:, (frequencies >= low) & (frequencies < high)].mean(1)
            for low, high in BANDS_HZ
        ], dim=1)
        energy_prediction = energies.argmin(1)
        output = {
            "loss": loss,
            "accuracy": (prediction == labels).float().mean(),
            "normalized_ce_gain": 1 - loss / torch.log(loss.new_tensor(len(BANDS_HZ))),
            "simple_band_energy_accuracy": (
                energy_prediction == labels
            ).float().mean(),
        }
        for index in range(len(BANDS_HZ)):
            output[f"class_{index}_accuracy"] = (
                prediction[labels == index] == index
            ).float().mean()
        return output

    def loss(self, signal, *, feature_fn, transform_seed):
        return self.metrics(signal, feature_fn=feature_fn, transform_seed=transform_seed)["loss"]

    def prepare_loss(self, signal, *, prefix_fn, transform_seed):
        views, labels = self._batch(signal, transform_seed)
        prefix_features = prefix_fn(views)

        def prepared(tail_fn):
            logits = self.head(tail_fn(prefix_features).mean(dim=1).flatten(1))
            return nn.functional.cross_entropy(logits, labels)

        return prepared


def _edge_taper(blocks: torch.Tensor, samples: int = 5) -> torch.Tensor:
    if samples <= 0:
        return blocks
    result = blocks.clone()
    ramp = torch.linspace(0, 1, samples + 2, device=blocks.device, dtype=blocks.dtype)[1:-1]
    result[..., :samples] *= ramp
    result[..., -samples:] *= ramp.flip(0)
    return result


def _broken_adjacencies(order: torch.Tensor) -> int:
    return int((order[1:] != order[:-1] + 1).sum())


class TemporalObjectiveV2(SSLObjectiveV2):
    name = "temporal"

    def __init__(self, blocks: int) -> None:
        super().__init__()
        if blocks not in {2, 5, 10} or 10 % blocks:
            raise ValueError("temporal blocks must be 2, 5, or 10")
        self.blocks = blocks
        self.head = nn.Linear(2000, 2)

    def _negative_orders(self, device: torch.device, seed: TransformSeed, count: int) -> torch.Tensor:
        if self.blocks == 2:
            return torch.tensor([1, 0], device=device).repeat(count, 1)
        seeds = _seed_list(seed, count)
        orders = []
        for sample_index, sample_seed in enumerate(seeds):
            for attempt in range(128):
                scores = _uniform_from_seeds(
                    [sample_seed], 1, self.blocks,
                    torch.device("cpu"),
                )[0]
                if attempt:
                    scores = _uniform_from_seeds(
                        [sample_seed + attempt * 1_000_003], 1, self.blocks,
                        torch.device("cpu"),
                    )[0]
                order = scores.argsort()
                if _broken_adjacencies(order) >= 3:
                    orders.append(order)
                    break
            else:
                raise RuntimeError(
                    f"could not sample a disrupted order for sample {sample_index}"
                )
        return torch.stack(orders).to(device)

    def _batch(self, signal: torch.Tensor, seed: TransformSeed) -> tuple[torch.Tensor, torch.Tensor]:
        patch_count = 10 // self.blocks
        blocks = signal.reshape(signal.shape[0], 16, self.blocks, patch_count * 200)
        blocks = _edge_taper(blocks, samples=5)
        positive = blocks.reshape_as(signal)
        orders = self._negative_orders(signal.device, seed, signal.shape[0])
        gather_index = orders[:, None, :, None].expand_as(blocks)
        negative = torch.gather(blocks, 2, gather_index).reshape_as(signal)
        views = torch.cat((positive, negative))
        labels = torch.cat((torch.ones(signal.shape[0]), torch.zeros(signal.shape[0]))).long().to(signal.device)
        return views, labels

    def metrics(self, signal, *, feature_fn, transform_seed):
        views, labels = self._batch(signal, transform_seed)
        logits = self.head(feature_fn(views).mean(dim=1).flatten(1))
        loss = nn.functional.cross_entropy(logits, labels)
        prediction = logits.argmax(1)
        continuous = views.float().reshape(views.shape[0], views.shape[1], -1)
        boundary_indexes = torch.arange(
            2000 // self.blocks, 2000, 2000 // self.blocks, device=signal.device
        )
        edge_score = torch.stack([
            (continuous[..., index] - continuous[..., index - 1]).abs().mean(1)
            for index in boundary_indexes
        ], dim=1).mean(1)
        return {
            "loss": loss,
            "accuracy": (prediction == labels).float().mean(),
            "balanced_accuracy": torch.stack([
                (prediction[labels == index] == index).float().mean() for index in range(2)
            ]).mean(),
            "normalized_ce_gain": 1 - loss / torch.log(loss.new_tensor(2)),
            "ordered_accuracy": (prediction[labels == 1] == 1).float().mean(),
            "shuffled_accuracy": (prediction[labels == 0] == 0).float().mean(),
            "edge_score_ordered": edge_score[labels == 1].mean(),
            "edge_score_shuffled": edge_score[labels == 0].mean(),
        }

    def loss(self, signal, *, feature_fn, transform_seed):
        return self.metrics(signal, feature_fn=feature_fn, transform_seed=transform_seed)["loss"]

    def prepare_loss(self, signal, *, prefix_fn, transform_seed):
        views, labels = self._batch(signal, transform_seed)
        prefix_features = prefix_fn(views)

        def prepared(tail_fn):
            logits = self.head(tail_fn(prefix_features).mean(dim=1).flatten(1))
            return nn.functional.cross_entropy(logits, labels)

        return prepared


class MaskObjectiveV2(SSLObjectiveV2):
    name = "mask"

    def __init__(self, masked_positions: int, scale_floor: float = 1e-4) -> None:
        super().__init__()
        if masked_positions not in {3, 5, 7}:
            raise ValueError("masked positions must be 3, 5, or 7")
        self.masked_positions = masked_positions
        self.scale_floor = float(scale_floor)
        self.head = nn.Linear(200, 75)
        self.register_buffer("training_target_sum", torch.zeros(75))
        self.register_buffer("training_target_count", torch.zeros(()))

    def _mask(self, signal: torch.Tensor, seed: TransformSeed) -> torch.Tensor:
        scores = _uniform_from_seeds(seed, signal.shape[0], 10, signal.device)
        indexes = scores.topk(self.masked_positions, dim=1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        return mask.scatter(1, indexes, True)

    def _components(self, signal: torch.Tensor, feature_fn, seed: int):
        mask = self._mask(signal, seed)
        visible = ~mask
        spectrum = torch.fft.rfft(signal.float(), dim=-1, norm="forward").abs().square()[..., 1:76]
        visible_values = spectrum.masked_select(visible[:, None, :, None]).reshape(
            signal.shape[0], -1
        )
        scale = visible_values.median(dim=1).values.clamp_min(self.scale_floor)
        scale = scale[:, None, None, None]
        target = torch.log1p(spectrum / scale)
        masked = signal.masked_fill(mask[:, None, :, None], 0)
        prediction = self.head(feature_fn(masked))
        selected = mask[:, None, :, None].expand_as(prediction)
        selected_prediction = prediction[selected].reshape(signal.shape[0], -1)
        selected_target = target[selected].reshape(signal.shape[0], -1)
        if self.training:
            with torch.no_grad():
                self.training_target_sum.add_(selected_target.reshape(-1, 75).sum(0))
                self.training_target_count.add_(selected_target.numel() // 75)

        # Nearest visible neighbors for all positions, without per-sample
        # Python branches or CUDA-to-host scalar synchronizations.
        positions = torch.arange(10, device=target.device)
        query = positions[None, :, None]
        candidate = positions[None, None, :]
        valid = visible[:, None, :]
        left = torch.where(valid & (candidate <= query), candidate, -1).amax(-1)
        right = torch.where(valid & (candidate >= query), candidate, 10).amin(-1)
        left = torch.where(left < 0, right, left)
        right = torch.where(right >= 10, left, right)
        left_values = target.gather(2, left[:, None, :, None].expand(-1, 16, -1, 75))
        right_values = target.gather(2, right[:, None, :, None].expand(-1, 16, -1, 75))
        fraction = (positions[None, :] - left).float() / (right - left).clamp_min(1)
        interpolation = left_values * (1-fraction[:, None, :, None]) + right_values * fraction[:, None, :, None]
        selected_interpolation = interpolation[selected].reshape(signal.shape[0], -1)
        return selected_prediction, selected_target, selected_interpolation

    def metrics(self, signal, *, feature_fn, transform_seed):
        prediction, target, interpolation = self._components(signal, feature_fn, transform_seed)
        per_window = nn.functional.smooth_l1_loss(prediction, target, reduction="none").mean(1)
        zero = nn.functional.smooth_l1_loss(torch.zeros_like(target), target, reduction="none").mean(1)
        context = nn.functional.smooth_l1_loss(interpolation, target, reduction="none").mean(1)
        output = {
            "loss": per_window.mean(),
            "zero_baseline_loss": zero.mean(),
            "context_interpolation_loss": context.mean(),
        }
        energy = signal.float().square().mean(dim=(1, 2, 3))
        order = energy.argsort()
        for label, indexes in zip(("low", "mid", "high"), order.tensor_split(3), strict=True):
            if len(indexes):
                output[f"loss_energy_{label}"] = per_window[indexes].mean()
        if self.training_target_count.item() > 0:
            mean = self.training_target_sum / self.training_target_count
            mean_prediction = mean.repeat(target.shape[0], target.shape[1] // 75)
            mean_loss = nn.functional.smooth_l1_loss(
                mean_prediction, target, reduction="none"
            ).mean(1)
            output["training_mean_loss"] = mean_loss.mean()
            output["relative_improvement_over_training_mean"] = (
                1 - per_window.mean() / mean_loss.mean().clamp_min(1e-12)
            )
        return output

    def loss(self, signal, *, feature_fn, transform_seed):
        # Interpolation and energy baselines belong to validation diagnostics;
        # computing them during training adds Python/CUDA synchronization only.
        mask = self._mask(signal, transform_seed)
        ordered = mask.to(torch.int64).argsort(dim=1, descending=True, stable=True)
        masked_indexes = ordered[:, :self.masked_positions]
        visible_indexes = ordered[:, self.masked_positions:]

        def select_positions(value, indexes):
            index = indexes[:, None, :, None].expand(-1, value.shape[1], -1, value.shape[3])
            return value.gather(2, index).reshape(signal.shape[0], -1)

        spectrum = torch.fft.rfft(signal.float(), dim=-1, norm="forward").abs().square()[..., 1:76]
        visible_values = select_positions(spectrum, visible_indexes)
        scale = visible_values.median(dim=1).values.clamp_min(self.scale_floor)
        target = torch.log1p(spectrum / scale[:, None, None, None])
        masked = signal.masked_fill(mask[:, None, :, None], 0)
        prediction = self.head(feature_fn(masked))
        selected_prediction = select_positions(prediction, masked_indexes)
        selected_target = select_positions(target, masked_indexes)
        if self.training:
            with torch.no_grad():
                self.training_target_sum.add_(selected_target.reshape(-1, 75).sum(0))
                self.training_target_count.add_(selected_target.numel() // 75)
        return nn.functional.smooth_l1_loss(
            selected_prediction, selected_target, reduction="none"
        ).mean(1).mean()

    def prepare_loss(self, signal, *, prefix_fn, transform_seed):
        mask = self._mask(signal, transform_seed)
        # Fixed-size gather avoids CUDA nonzero/boolean-index shape discovery
        # and its host synchronization on every candidate loss evaluation.
        ordered = mask.to(torch.int64).argsort(dim=1, descending=True, stable=True)
        masked_indexes = ordered[:, :self.masked_positions]
        visible_indexes = ordered[:, self.masked_positions:]

        def select_positions(value, indexes):
            index = indexes[:, None, :, None].expand(-1, value.shape[1], -1, value.shape[3])
            return value.gather(2, index).reshape(signal.shape[0], -1)

        spectrum = torch.fft.rfft(signal.float(), dim=-1, norm="forward").abs().square()[..., 1:76]
        visible_values = select_positions(spectrum, visible_indexes)
        scale = visible_values.median(dim=1).values.clamp_min(self.scale_floor)
        target = torch.log1p(spectrum / scale[:, None, None, None])
        selected_target = select_positions(target, masked_indexes)
        masked = signal.masked_fill(mask[:, None, :, None], 0)
        prefix_features = prefix_fn(masked)

        def prepared(tail_fn):
            prediction = self.head(tail_fn(prefix_features))
            selected_prediction = select_positions(prediction, masked_indexes)
            return nn.functional.smooth_l1_loss(
                selected_prediction, selected_target, reduction="none"
            ).mean(1).mean()

        return prepared


def build_objective(name: str, difficulty: float) -> SSLObjectiveV2:
    if name == "band":
        return BandObjectiveV2(float(difficulty))
    if name == "temporal":
        return TemporalObjectiveV2(int(difficulty))
    if name == "mask":
        return MaskObjectiveV2(int(difficulty))
    raise ValueError(f"unknown v2 SSL objective: {name}")
