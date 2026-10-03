import torch
import torchaudio
import random
from typing import List, Tuple

class BandPretext(torch.nn.Module):
    """
    Band-reject pretext head:
    - Exclude invalid bands based on sfreq and optional max_freq.
    - Randomly select and reject one band for each sample.
    - Return filtered_x and band_labels.
    - Provide a simple linear classifier for band identification.
    """
    def __init__(self, input_dim: int,
                 bands: List[Tuple[float, float]] = None,
                 min_freq: float = 0.3,
                 max_freq: float = 35.0,
                 edge_margin_hz: float = 0.5,
                 min_q: float = 0.5):
        super().__init__()
        # Delta, theta, alpha, beta1, beta2; split beta to keep its upper edge at or below 35 Hz.
        if bands is None:
            bands = [(0.3, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 20.0), (20.0, 35.0)]
        self._orig_bands = bands
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.edge_margin_hz = edge_margin_hz  # Leave a safety margin near 0 Hz and Nyquist.
        self.min_q = min_q

        self.classifier = torch.nn.Linear(
            in_features=input_dim,
            out_features=len(bands),
        )

    def _valid_bands(self, sfreq: float) -> List[Tuple[float, float]]:
        nyq = sfreq / 2.0
        upper_limit = min(self.max_freq, nyq - self.edge_margin_hz)
        lower_limit = max(self.min_freq, 0.0 + self.edge_margin_hz)
        bands = []
        for lo, hi in self._orig_bands:
            lo2 = max(lo, lower_limit)
            hi2 = min(hi, upper_limit)
            if hi2 - lo2 > 0.0:
                bands.append((lo2, hi2))
        # Retain at least one frequency band.
        if not bands:
            # Fall back to treating the entire analyzable frequency range as one band.
            bands = [(lower_limit, upper_limit)]
        return bands

    @torch.no_grad()
    def reject_band(self, x: torch.Tensor, sfreq: torch.Tensor):
        """
        x: (B, C, T) or (B, T); reject bands along T and broadcast over C as an extra batch dimension.
        sfreq: Scalar or tensor of shape [1], in Hz.
        Returns:
            filtered_x: Same shape as x.
            band_labels: (B,), the rejected frequency-band index for each sample.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,T)
            squeeze_back = True
        else:
            squeeze_back = False

        B, C, T = x.shape
        sr = float(sfreq[0].item()) if sfreq.numel() > 0 else float(sfreq)

        bands = self._valid_bands(sr)
        num_bands = len(bands)

        # Output buffer
        y = torch.empty_like(x)
        labels = torch.empty(B, dtype=torch.long, device=x.device)

        for b in range(B):
            idx = random.randrange(num_bands)
            lo, hi = bands[idx]
            cf = (lo + hi) / 2.0
            bw = max(hi - lo, 1e-6)
            Q  = max(cf / bw, self.min_q)

            # torchaudio biquad broadcasts over leading dimensions; apply the same filter to (C, T).
            # Input: (..., time); treat (C, T) as multiple channels within a batch.
            xb = x[b]                      # (C,T)
            yb = torchaudio.functional.bandreject_biquad(
                    xb, sample_rate=sr, central_freq=cf, Q=Q
                 )
            y[b] = yb
            labels[b] = idx

        if squeeze_back:
            y = y.squeeze(1)  # (B,T)
        return y, labels

    def forward(self, feats: torch.Tensor, labels: torch.Tensor):
        """
        feats: (B, D), representations extracted from the backbone.
        labels: (B,), band labels returned by reject_band.
        """
        logits = self.classifier(feats)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, labels
