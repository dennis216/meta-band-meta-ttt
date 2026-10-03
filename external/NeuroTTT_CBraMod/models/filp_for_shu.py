
# Swap symmetric left/right hemisphere channels, keeping midline channels fixed; classify whether swapped.
from __future__ import annotations
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Zero-based indices for 32 channels.
# ['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5','FC6','Cz','C3','C4','T3','T4','A1','A2','CP1','CP2','CP5','CP6','Pz','P3','P4','T5','T6','PO3','PO4','Oz','O1','O2']
DEFAULT_PAIRS_32: List[Tuple[int, int]] = [
    (0, 1),    # Fp1 ↔ Fp2
    (3, 4),    # F3  ↔ F4
    (5, 6),    # F7  ↔ F8
    (7, 8),    # FC1 ↔ FC2
    (9,10),    # FC5 ↔ FC6
    (12,13),   # C3  ↔ C4
    (14,15),   # T3  ↔ T4
    (16,17),   # A1  ↔ A2
    (18,19),   # CP1 ↔ CP2
    (20,21),   # CP5 ↔ CP6
    (23,24),   # P3  ↔ P4
    (25,26),   # T5  ↔ T6
    (27,28),   # PO3 ↔ PO4
    (30,31),   # O1  ↔ O2
]
DEFAULT_MIDLINE_32 = [2, 11, 22, 29]  # Fz, Cz, Pz, Oz

class JigsawPretext(nn.Module):
    """
    Usage:
      1) Flatten inputs to (B, C, T) and call make_symmetry(x_flat); C must match the 32 channels above.
      2) Obtain (x_out, labels): 0 = unchanged, 1 = swapped.
      3) Pass x_out through the backbone to obtain feats_flat of shape (B, D).
         Then call loss_from_features(feats_flat, labels) to compute cross-entropy.
    """
    def __init__(self,
                 pairs: Optional[List[Tuple[int,int]]] = None,
                 midline: Optional[List[int]] = None,
                 hidden_dim: int = 512):
        super().__init__()
        self.pairs = pairs if pairs is not None else DEFAULT_PAIRS_32
        self.midline = set(midline if midline is not None else DEFAULT_MIDLINE_32)
        self.hidden_dim = hidden_dim
        self.classifier: Optional[nn.Sequential] = None
        self._in_dim: Optional[int] = None

    @torch.no_grad()
    def make_symmetry(self, x_flat: torch.Tensor):
        """
        Input x_flat: (B, C, T), with C in the zero-based 32-channel order defined here.
        Output x_out: (B, C, T); labels: (B,), with 0 = unchanged and 1 = swapped.
        Swap paired left/right hemisphere channels with 50% probability; keep midline channels fixed.
        """
        if x_flat.dim() != 3:
            raise ValueError(f"x_flat must be (B,C,T), got {tuple(x_flat.shape)}")
        B, C, T = x_flat.shape
        out = x_flat.clone()
        labels = torch.zeros(B, dtype=torch.long, device=x_flat.device)

        # Use only pairs with channel indices inside the valid range.
        pairs = [(i, j) for (i, j) in self.pairs if (i < C and j < C)]
        for b in range(B):
            do_swap = torch.randint(0, 2, (1,), device=x_flat.device).item()  # 0/1
            if do_swap == 1:
                for i, j in pairs:
                    if (i in self.midline) or (j in self.midline):
                        continue
                    tmp = out[b, i].clone()
                    out[b, i] = out[b, j]
                    out[b, j] = tmp
                labels[b] = 1
        return out, labels

    def _ensure_head(self, in_dim: int):
        if (self.classifier is None) or (self._in_dim != in_dim):
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, 256),
                nn.GELU(),
                nn.Linear(256, 2),  # 0/1: whether channels were swapped.
            )
            self._in_dim = in_dim

    def loss_from_features(self, feats_flat: torch.Tensor, swap_labels: torch.Tensor):
        """
        feats_flat: (B, D), flattened backbone outputs.
        swap_labels: (B,) in {0,1}
        Return (loss, preds).
        """
        self._ensure_head(feats_flat.size(1))
        logits = self.classifier(feats_flat)
        loss = F.cross_entropy(logits, swap_labels)
        preds = logits.argmax(dim=-1)
        return loss, preds
