from bfa.evaluation.eventize import Event
from bfa.evaluation.match import match_events
from bfa.evaluation.metrics import ThresholdScore, event_metrics, select_threshold


def test_match_uses_one_to_one_maximum_total_overlap() -> None:
    predictions = [Event(0, 10, 0.7), Event(8, 20, 0.9)]
    result = match_events(predictions, [(5, 15)])
    assert len(result.pairs) == 1
    assert result.pairs[0].prediction_index == 1
    assert result.pairs[0].overlap_s == 7
    assert result.unmatched_predictions == (0,)
    assert result.unmatched_truths == ()


def test_match_prioritizes_cardinality_before_total_overlap() -> None:
    # P0-T0=10, P0-T1=4, P1-T0=4, P1-T1<1.  Pure maximum-overlap
    # assignment selects 10 plus the invalid edge; lexicographic matching must
    # select the two four-second valid edges.
    predictions = [Event(0, 14, 0.9), Event(6, 10, 0.8)]
    truths = [(0, 10), (10, 14)]
    result = match_events(predictions, truths, minimum_overlap_s=1.0)
    assert {(pair.prediction_index, pair.truth_index) for pair in result.pairs} == {
        (0, 1),
        (1, 0),
    }
    assert result.unmatched_predictions == ()
    assert result.unmatched_truths == ()


def test_metrics_and_threshold_selection() -> None:
    predictions = [Event(0, 10, 0.9), Event(30, 40, 0.7)]
    metrics = event_metrics(predictions, [(5, 8)], nonseizure_hours=24)
    assert metrics.true_positive_events == 1
    assert metrics.false_alarm_events == 1
    assert metrics.event_sensitivity == 1.0
    assert metrics.fa_per_24h == 1.0

    selected = select_threshold(
        [
            ThresholdScore(0.4, 0.9, 3.0),
            ThresholdScore(0.5, 0.8, 2.0),
            ThresholdScore(0.6, 0.8, 2.0),
            ThresholdScore(0.7, 0.7, 1.0),
        ]
    )
    assert selected.threshold == 0.6

    fallback = select_threshold([ThresholdScore(0.8, 0.7, 1.0), ThresholdScore(0.9, 0.7, 0.5)])
    assert fallback.threshold == 0.9
