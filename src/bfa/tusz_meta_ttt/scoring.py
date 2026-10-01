from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bfa.evaluation.eventize import Event, causal_ema
from bfa.evaluation.match import match_events


@dataclass(frozen=True)
class EventMetrics:
    sensitivity: float
    false_alarms_per_hour: float
    false_alarm_minutes_per_hour: float
    median_delay_s: float
    detected: int
    total: int
    false_alarms: int


def consecutive_eventize(
    times: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    ema_alpha: float = 1 / 3,
    onset_consecutive: int = 2,
    offset_consecutive: int = 2,
    record_end_s: float | None = None,
) -> list[Event]:
    times = np.asarray(times, dtype=float)
    values = np.asarray(probabilities, dtype=float)
    if times.ndim != 1 or values.shape != times.shape:
        raise ValueError("times and probabilities must be aligned")
    if len(times) < 2:
        return []
    if not np.all(np.diff(times) > 0) or not np.isfinite(values).all():
        raise ValueError("times must increase and probabilities must be finite")
    if onset_consecutive < 1 or offset_consecutive < 1:
        raise ValueError("consecutive counts must be positive")
    smoothed = causal_ema(values, ema_alpha)
    if record_end_s is not None and (not np.isfinite(record_end_s) or record_end_s < times[-1]):
        raise ValueError("record_end_s must be finite and no earlier than the final decision")
    events: list[Event] = []
    active_start: float | None = None
    peak = 0.0
    above_run = below_run = 0
    for index, value in enumerate(smoothed):
        if active_start is None:
            above_run = above_run + 1 if value >= threshold else 0
            if above_run >= onset_consecutive:
                # Alarm is emitted now; do not backdate it to the first crossing.
                active_start = float(times[index])
                peak = float(value)
                below_run = 0
        else:
            peak = max(peak, float(value))
            below_run = below_run + 1 if value < threshold else 0
            if below_run >= offset_consecutive:
                # The alarm ends when the second below-threshold decision is
                # actually emitted.  Do not extend it by another prediction step.
                events.append(Event(active_start, float(times[index]), peak))
                active_start = None
                above_run = below_run = 0
    if active_start is not None:
        events.append(
            Event(
                active_start,
                float(record_end_s if record_end_s is not None else times[-1]),
                peak,
            )
        )
    return events


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def score_record(
    predictions: list[Event],
    truths: list[tuple[float, float]],
    *,
    duration_s: float,
    minimum_overlap_s: float = 1.0,
) -> EventMetrics:
    result = match_events(predictions, truths, minimum_overlap_s)
    seizure_seconds = _union_duration(truths)
    background_hours = max(np.finfo(float).eps, (duration_s - seizure_seconds) / 3600)
    false_alarm_intervals = []
    # Alarm time is measured on every non-seizure second, including tails of
    # otherwise matched alarms. Alarm count remains the unmatched-event count.
    for event in predictions:
        for start, end in [(event.start_s, event.end_s)]:
            pieces = [(start, end)]
            for truth_start, truth_end in truths:
                next_pieces = []
                for left, right in pieces:
                    if truth_end <= left or truth_start >= right:
                        next_pieces.append((left, right))
                    else:
                        if left < truth_start:
                            next_pieces.append((left, truth_start))
                        if truth_end < right:
                            next_pieces.append((truth_end, right))
                pieces = next_pieces
            false_alarm_intervals.extend(pieces)
    delays = [max(0.0, predictions[p.prediction_index].start_s - truths[p.truth_index][0]) for p in result.pairs]
    return EventMetrics(
        sensitivity=len(result.pairs) / len(truths) if truths else float("nan"),
        false_alarms_per_hour=len(result.unmatched_predictions) / background_hours,
        false_alarm_minutes_per_hour=_union_duration(false_alarm_intervals) / 60 / background_hours,
        median_delay_s=float(np.median(delays)) if delays else float("nan"),
        detected=len(result.pairs),
        total=len(truths),
        false_alarms=len(result.unmatched_predictions),
    )


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float | None
    metrics: EventMetrics | None
    reachable: bool
    maximum_sensitivity_metrics: EventMetrics | None = None
    maximum_sensitivity_threshold: float | None = None


def choose_threshold(
    records: list[dict],
    *,
    target_sensitivity: float = 0.80,
    quantiles: int = 1001,
) -> ThresholdChoice:
    if not records:
        raise ValueError("records cannot be empty")
    smoothed_values = [
        causal_ema(np.asarray(record["probabilities"], dtype=float), 1 / 3)
        for record in records
        if len(record["probabilities"])
    ]
    if not smoothed_values:
        raise ValueError("at least one record must contain prediction windows")
    all_smoothed = np.concatenate(smoothed_values)
    candidates = np.unique(
        np.concatenate(
            ([-np.inf], np.quantile(all_smoothed, np.linspace(0, 1, quantiles)), [np.inf])
        )
    )
    feasible: list[tuple[float, EventMetrics]] = []
    maximum: tuple[float, EventMetrics] | None = None
    for threshold in candidates:
        totals = {
            "detected": 0,
            "total": 0,
            "false_alarms": 0,
            "background_hours": 0.0,
            "fa_seconds": 0.0,
        }
        delays = []
        for record in records:
            events = consecutive_eventize(
                record["times"], record["probabilities"], threshold=float(threshold),
                record_end_s=float(record["duration_s"]),
            )
            metrics = score_record(events, record["truths"], duration_s=float(record["duration_s"]))
            totals["detected"] += metrics.detected
            totals["total"] += metrics.total
            totals["false_alarms"] += metrics.false_alarms
            seizure_seconds = _union_duration(record["truths"])
            background_hours = max(0.0, (float(record["duration_s"]) - seizure_seconds) / 3600)
            totals["background_hours"] += background_hours
            totals["fa_seconds"] += metrics.false_alarm_minutes_per_hour * 60 * background_hours
            matched = match_events(events, record["truths"])
            delays.extend(max(0.0, events[p.prediction_index].start_s - record["truths"][p.truth_index][0]) for p in matched.pairs)
        sensitivity = totals["detected"] / totals["total"] if totals["total"] else float("nan")
        background_hours = max(np.finfo(float).eps, totals["background_hours"])
        aggregate = EventMetrics(
            sensitivity,
            totals["false_alarms"] / background_hours,
            totals["fa_seconds"] / 60 / background_hours,
            float(np.median(delays)) if delays else float("nan"),
            totals["detected"], totals["total"], totals["false_alarms"],
        )
        if maximum is None or (aggregate.sensitivity, -aggregate.false_alarms_per_hour) > (
            maximum[1].sensitivity,
            -maximum[1].false_alarms_per_hour,
        ):
            maximum = (float(threshold), aggregate)
        if sensitivity >= target_sensitivity:
            feasible.append((float(threshold), aggregate))
    if not feasible:
        return ThresholdChoice(
            None,
            None,
            False,
            maximum[1] if maximum else None,
            maximum[0] if maximum else None,
        )
    threshold, metrics = min(feasible, key=lambda pair: (pair[1].false_alarms_per_hour, pair[1].false_alarm_minutes_per_hour, -pair[0]))
    return ThresholdChoice(
        threshold,
        metrics,
        True,
        maximum[1] if maximum else None,
        maximum[0] if maximum else None,
    )


def score_at_threshold(records: list[dict], threshold: float) -> EventMetrics:
    """Aggregate record-level event scores at one already fixed threshold."""
    if not records:
        raise ValueError("records cannot be empty")
    detected = total = false_alarms = 0
    background_hours = false_alarm_seconds = 0.0
    delays: list[float] = []
    for record in records:
        events = consecutive_eventize(
            record["times"], record["probabilities"], threshold=float(threshold),
            record_end_s=float(record["duration_s"]),
        )
        metrics = score_record(events, record["truths"], duration_s=float(record["duration_s"]))
        detected += metrics.detected
        total += metrics.total
        false_alarms += metrics.false_alarms
        seizure_seconds = _union_duration(record["truths"])
        record_background_hours = max(
            0.0, (float(record["duration_s"]) - seizure_seconds) / 3600
        )
        background_hours += record_background_hours
        false_alarm_seconds += (
            metrics.false_alarm_minutes_per_hour * 60 * record_background_hours
        )
        matched = match_events(events, record["truths"])
        delays.extend(
            max(
                0.0,
                events[pair.prediction_index].start_s
                - record["truths"][pair.truth_index][0],
            )
            for pair in matched.pairs
        )
    denominator = max(np.finfo(float).eps, background_hours)
    return EventMetrics(
        detected / total if total else float("nan"),
        false_alarms / denominator,
        false_alarm_seconds / 60 / denominator,
        float(np.median(delays)) if delays else float("nan"),
        detected,
        total,
        false_alarms,
    )
