from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import butter, iirnotch, resample_poly, sosfilt, sosfilt_zi, tf2sos


@dataclass(frozen=True)
class PreprocessingMetadata:
    source_hz: float
    target_hz: float
    bandpass_hz: tuple[float, float]
    bandpass_order: int
    notch_hz: float
    notch_q: float
    resampling_filter_taps: int
    output_delay_s: float


class CausalTUSZPreprocessor:
    """Stateful causal filtering with delayed, prefix-stable polyphase output.

    The resampler is evaluated on the observed prefix. Its boundary region is
    withheld until enough future samples have arrived, making every emitted
    sample immutable. This reference implementation favors auditability; the
    cache builder feeds one full EDF at a time, so it does not incur repeated
    prefix work during normal preprocessing.
    """

    def __init__(
        self,
        source_hz: float,
        *,
        channels: int = 16,
        target_hz: float = 200.0,
        bandpass_hz: tuple[float, float] = (0.3, 75.0),
        bandpass_order: int = 4,
        notch_hz: float = 60.0,
        notch_q: float = 30.0,
    ) -> None:
        if source_hz <= 2 * max(bandpass_hz[1], notch_hz):
            raise ValueError("source sampling rate is too low for the configured filters")
        self.source_hz = float(source_hz)
        self.target_hz = float(target_hz)
        self.channels = int(channels)
        self.bandpass = butter(
            bandpass_order, bandpass_hz, btype="bandpass", fs=source_hz, output="sos"
        )
        notch_b, notch_a = iirnotch(notch_hz, notch_q, fs=source_hz)
        self.notch = tf2sos(notch_b, notch_a)
        self.bandpass_state = np.repeat(sosfilt_zi(self.bandpass)[:, None, :], channels, axis=1)
        self.notch_state = np.repeat(sosfilt_zi(self.notch)[:, None, :], channels, axis=1)
        ratio = Fraction(target_hz / source_hz).limit_denominator(10_000)
        self.up, self.down = ratio.numerator, ratio.denominator
        self.filter_taps = 10 * max(self.up, self.down) + 1
        self.guard_output = int(np.ceil((self.filter_taps // 2) / self.down)) + 1
        self.filtered_history = np.empty((channels, 0), dtype=np.float64)
        self.emitted = 0
        self.metadata = PreprocessingMetadata(
            source_hz=self.source_hz,
            target_hz=self.target_hz,
            bandpass_hz=bandpass_hz,
            bandpass_order=bandpass_order,
            notch_hz=notch_hz,
            notch_q=notch_q,
            resampling_filter_taps=self.filter_taps,
            output_delay_s=self.guard_output / self.target_hz,
        )

    def update(self, raw_uv: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_uv, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[0] != self.channels:
            raise ValueError(f"raw_uv must have shape [{self.channels}, samples]")
        if not np.isfinite(raw).all():
            raise ValueError("raw_uv contains non-finite values")
        if raw.shape[1] == 0:
            return np.empty((self.channels, 0), dtype=np.float32)
        filtered, self.notch_state = sosfilt(self.notch, raw, axis=-1, zi=self.notch_state)
        filtered, self.bandpass_state = sosfilt(
            self.bandpass, filtered, axis=-1, zi=self.bandpass_state
        )
        self.filtered_history = np.concatenate([self.filtered_history, filtered], axis=-1)
        resampled = resample_poly(self.filtered_history, self.up, self.down, axis=-1)
        safe_end = max(self.emitted, resampled.shape[-1] - self.guard_output)
        output = resampled[:, self.emitted:safe_end] * 0.01
        self.emitted = safe_end
        return output.astype(np.float32)


def frame_windows(
    signal: np.ndarray,
    *,
    window_samples: int = 2000,
    stride_samples: int = 400,
) -> np.ndarray:
    array = np.asarray(signal)
    if array.ndim != 2 or array.shape[0] != 16:
        raise ValueError("signal must have shape [16, samples]")
    if array.shape[1] < window_samples:
        return np.empty((0, 16, window_samples), dtype=array.dtype)
    view = np.lib.stride_tricks.sliding_window_view(array, window_samples, axis=-1)
    return np.ascontiguousarray(view[:, ::stride_samples].transpose(1, 0, 2))
