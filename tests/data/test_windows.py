from bfa.data.windows import build_window_index


def test_window_index_labels_context_and_warmup() -> None:
    frame = build_window_index(
        patient_id="chb01_21",
        recording_id="chb01_03.edf",
        duration_s=180.0,
        seizures=[(100.0, 110.0)],
        cache_key="abc123",
    )
    assert list(frame.columns) == [
        "patient",
        "recording",
        "start",
        "end",
        "context_start",
        "label",
        "train_eligible",
        "warmup",
        "cache_key",
    ]
    assert frame.loc[frame.start == 0, "warmup"].item()
    assert frame.loc[frame.start == 58, "warmup"].item()
    assert not frame.loc[frame.start == 60, "warmup"].item()
    assert frame.loc[frame.start == 100, "label"].item() == 1
    assert not frame.loc[frame.start == 88, "train_eligible"].item()
    assert frame.loc[frame.start == 60, "context_start"].item() == 0
    assert frame.recording.nunique() == 1
