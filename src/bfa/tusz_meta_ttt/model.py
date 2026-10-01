from __future__ import annotations

import torch
from torch import nn


class TUSZDetector(nn.Module):
    """CBraMod-backed detector with an explicit 2000 -> 256 -> 1 head."""

    def __init__(
        self, backbone: nn.Module, *, dropout: float = 0.1, freeze_backbone: bool = False
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = bool(freeze_backbone)
        self.backbone.requires_grad_(not self.freeze_backbone)
        self.detector = nn.Sequential(
            nn.Linear(10 * 200, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def train(self, mode: bool = True) -> TUSZDetector:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def features(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 4 or signal.shape[1:] != (16, 10, 200):
            raise ValueError("signal must have shape [batch, 16, 10, 200]")
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(signal)
        else:
            features = self.backbone(signal)
        expected = (signal.shape[0], 16, 10, 200)
        if tuple(features.shape) != expected:
            raise ValueError(f"backbone returned {tuple(features.shape)}, expected {expected}")
        return features

    def pooled_features(self, signal: torch.Tensor) -> torch.Tensor:
        return self.features(signal).mean(dim=1).flatten(1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.detector(self.pooled_features(signal)).squeeze(-1)


def balanced_soft_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Average seizure and background losses without infinite class weights."""
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have identical shapes")
    losses = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    positive = labels > 0
    terms = []
    if positive.any():
        terms.append(losses[positive].mean())
    if (~positive).any():
        terms.append(losses[~positive].mean())
    if not terms:
        raise ValueError("cannot compute a loss for an empty batch")
    return torch.stack(terms).mean()
