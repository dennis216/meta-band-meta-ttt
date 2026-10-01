from bfa.tusz_meta_ttt.statistics import PatientComponents, paired_patient_bootstrap


def row(patient, detected, false_alarms):
    return PatientComponents(patient, detected, 10, false_alarms, 10.0, false_alarms * 20.0)


def test_paired_bootstrap_preserves_patient_pairing():
    frozen = [row("a", 8, 10), row("b", 9, 20)]
    adapted = [row("a", 8, 5), row("b", 9, 10)]
    result = paired_patient_bootstrap(frozen, adapted, replicates=500, seed=17)
    assert result["sensitivity_difference"]["estimate"] == 0
    assert result["false_alarms_per_hour_ratio"]["ci_upper"] == 0.5


def test_bootstrap_rejects_unpaired_patients():
    try:
        paired_patient_bootstrap([row("a", 8, 10)], [row("b", 8, 10)], replicates=2)
    except ValueError as error:
        assert "identical patients" in str(error)
    else:
        raise AssertionError("unpaired input should fail")
