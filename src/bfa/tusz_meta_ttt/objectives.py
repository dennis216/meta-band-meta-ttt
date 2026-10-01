from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import nn

BANDS_HZ = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
TEMPORAL_PERMUTATIONS = ((0, 1, 2, 3, 4), (4, 3, 2, 1, 0), (1, 2, 3, 4, 0), (0, 2, 4, 1, 3))


def remove_frequency_band(signal: torch.Tensor, band: tuple[float, float], sampling_hz: int = 200) -> torch.Tensor:
    spectrum = torch.fft.rfft(signal.float(), dim=-1)
    frequencies = torch.fft.rfftfreq(signal.shape[-1], d=1.0 / sampling_hz).to(signal.device)
    keep = ~((frequencies >= band[0]) & (frequencies < band[1]))
    return torch.fft.irfft(spectrum * keep, n=signal.shape[-1], dim=-1).to(signal.dtype)


class SSLObjective(nn.Module, ABC):
    """An unlabeled loss. Labels are intentionally absent from this interface."""

    @abstractmethod
    def loss(
        self,
        signal: torch.Tensor,
        *,
        feature_fn,
        fast_parameters: Mapping[str, torch.Tensor] | None = None,
        transform_seed: int = 0,
    ) -> torch.Tensor:
        raise NotImplementedError


class BandObjective(SSLObjective):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(2000, len(BANDS_HZ))

    def loss(self, signal, *, feature_fn, fast_parameters=None, transform_seed=0):
        del fast_parameters, transform_seed
        views = torch.cat([remove_frequency_band(signal, band) for band in BANDS_HZ], dim=0)
        features = feature_fn(views).mean(dim=1).flatten(1)
        labels = torch.arange(len(BANDS_HZ), device=signal.device).repeat_interleave(signal.shape[0])
        return nn.functional.cross_entropy(self.head(features), labels)


class TemporalObjective(SSLObjective):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(2000, len(TEMPORAL_PERMUTATIONS))

    @staticmethod
    def permute(signal: torch.Tensor, order: tuple[int, ...]) -> torch.Tensor:
        blocks = signal.reshape(*signal.shape[:-2], 5, 2, signal.shape[-1])
        return blocks[..., list(order), :, :].reshape_as(signal)

    def loss(self, signal, *, feature_fn, fast_parameters=None, transform_seed=0):
        del fast_parameters, transform_seed
        views = torch.cat([self.permute(signal, order) for order in TEMPORAL_PERMUTATIONS], dim=0)
        features = feature_fn(views).mean(dim=1).flatten(1)
        labels = torch.arange(len(TEMPORAL_PERMUTATIONS), device=signal.device).repeat_interleave(signal.shape[0])
        return nn.functional.cross_entropy(self.head(features), labels)


class MaskObjective(SSLObjective):
    def __init__(self, mask_fraction: float = 0.5) -> None:
        super().__init__()
        self.mask_fraction = float(mask_fraction)
        self.head = nn.Linear(200, 200)

    def loss(self, signal, *, feature_fn, fast_parameters=None, transform_seed=0):
        del fast_parameters
        generator = torch.Generator(device=signal.device).manual_seed(transform_seed)
        mask = torch.rand(signal.shape[:3], generator=generator, device=signal.device) < self.mask_fraction
        masked = signal.clone()
        masked[mask] = 0
        predictions = self.head(feature_fn(masked))
        if not mask.any():
            raise RuntimeError("mask sampling produced no masked patches")
        return nn.functional.mse_loss(predictions[mask], signal.float()[mask])


class LearnedObjective(SSLObjective):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(405, 128), nn.GELU(), nn.Linear(128, 1))

    def loss(self, signal, *, feature_fn, fast_parameters=None, transform_seed=0):
        del fast_parameters, transform_seed
        original = feature_fn(signal).mean(dim=(1, 2))
        # Evaluate all transformed views in one backbone call. The view-major
        # ordering matches BandObjective and preserves the previous mean loss.
        views = torch.cat([remove_frequency_band(signal, band) for band in BANDS_HZ], dim=0)
        transformed = feature_fn(views).mean(dim=(1, 2))
        repeated_original = original.repeat(len(BANDS_HZ), 1)
        identities = nn.functional.one_hot(
            torch.arange(len(BANDS_HZ), device=signal.device).repeat_interleave(signal.shape[0]),
            num_classes=len(BANDS_HZ),
        ).float()
        inputs = torch.cat([repeated_original, transformed, identities], dim=1)
        return nn.functional.softplus(self.network(inputs)).mean()


def build_objective(name: str) -> SSLObjective:
    constructors = {"band": BandObjective, "temporal": TemporalObjective, "mask": MaskObjective, "learned": LearnedObjective}
    try:
        return constructors[name]()
    except KeyError as error:
        raise ValueError(f"unknown SSL objective: {name}") from error
