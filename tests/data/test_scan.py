from bfa.data.scan import canonical_patient_id, is_primary_case, parse_summary_intervals


def test_patient_alias_and_summary_parser() -> None:
    assert canonical_patient_id("chb21") == "chb01_21"
    assert canonical_patient_id("chb01") == "chb01_21"
    text = (
        "File Name: chb01_03.edf\n"
        "Number of Seizures in File: 1\n"
        "Seizure 1 Start Time: 2996 seconds\n"
        "Seizure 1 End Time: 3036 seconds\n"
    )
    rows = parse_summary_intervals(text)
    assert rows == [("chb01_03.edf", 2996.0, 3036.0)]
    assert is_primary_case("chb23")
    assert not is_primary_case("chb24")
