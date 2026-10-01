from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from bfa.models.cbramod_adapter import CBraModAdapter
from bfa.tusz_meta_ttt.model import TUSZDetector

ROOT = Path(__file__).resolve().parents[3]
PRETRAINED = ROOT / "third_party/CBraMod/pretrained_weights/pretrained_weights.pth"


def load_source(path: Path, *, detector_trainable: bool = False) -> TUSZDetector:
    adapter = CBraModAdapter(PRETRAINED, train_backbone=True)
    adapter.backbone.proj_out = torch.nn.Identity()
    model = TUSZDetector(adapter.backbone, dropout=0.1)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith(("backbone.encoder.layers.10.", "backbone.encoder.layers.11.")):
            parameter.requires_grad_(True)
    model.detector.requires_grad_(detector_trainable)
    return model.cuda().eval()


def make_windows(signal: np.ndarray, rows: tuple[int, ...]) -> torch.Tensor:
    values = np.stack([signal[:, row * 400 : row * 400 + 2000] for row in rows])
    return torch.from_numpy(values.reshape(-1, 16, 10, 200)).cuda(non_blocking=True)


def optimizer_groups(module: torch.nn.Module, prefix: str, lr: float, decay: float) -> list[dict]:
    regular, exempt = [], []
    for name, parameter in module.named_parameters():
        if parameter.requires_grad and name.startswith(prefix):
            (exempt if name.endswith("bias") or parameter.ndim == 1 else regular).append(parameter)
    groups = []
    if regular:
        groups.append({"params": regular, "lr": lr, "weight_decay": decay})
    if exempt:
        groups.append({"params": exempt, "lr": lr, "weight_decay": 0.0})
    return groups
