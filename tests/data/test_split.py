import pandas as pd

from bfa.data.split import make_group_split


def _patient_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["chb01_21", *[f"chb{i:02d}" for i in range(2, 23)]],
            "seizures": [1 + (i % 5) for i in range(22)],
            "hours": [18.0 + i for i in range(22)],
            "stratum_high_burden": [int(i % 3 == 0) for i in range(22)],
        }
    )


def test_group_split_is_deterministic_and_disjoint() -> None:
    patient_stats = _patient_stats()
    first = make_group_split(patient_stats, 17)
    second = make_group_split(patient_stats, 17)
    assert first == second
    assert len(first["train"]) == 13
    assert len(first["validation"]) == 4
    assert len(first["test"]) == 5
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assigned = first["train"] + first["validation"] + first["test"]
    assert sorted(assigned) == sorted(patient_stats.patient_id)
    assert sum("chb01_21" in first[part] for part in ("train", "validation", "test")) == 1
