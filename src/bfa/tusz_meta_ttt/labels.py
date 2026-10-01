from __future__ import annotations

import numpy as np


def interval_overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    """Return the duration of the intersection of two half-open intervals."""
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def decision_interval_labels(
    decision_end_s: np.ndarray,
    seizures: list[tuple[float, float]],
    *,
    decision_seconds: float = 2.0,
) -> np.ndarray:
    """Create soft labels from seizure occupancy in each recent decision interval."""
    ends = np.asarray(decision_end_s, dtype=np.float64)
    if ends.ndim != 1 or not np.isfinite(ends).all():
        raise ValueError("decision_end_s must be a finite one-dimensional array")
    if decision_seconds <= 0:
        raise ValueError("decision_seconds must be positive")
    normalized: list[tuple[float, float]] = []
    for start, end in seizures:
        if not np.isfinite([start, end]).all() or start < 0 or end <= start:
            raise ValueError(f"invalid seizure interval: {(start, end)}")
        normalized.append((float(start), float(end)))
    labels = np.zeros(len(ends), dtype=np.float32)
    for index, end in enumerate(ends):
        interval = (float(end - decision_seconds), float(end))
        occupied = sum(interval_overlap(interval, seizure) for seizure in normalized)
        labels[index] = min(1.0, occupied / decision_seconds)
    return labels
