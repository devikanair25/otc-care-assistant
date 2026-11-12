from datetime import date

import pytest

from otc_mcp.clients.fhir import LocalBundleSource, PatientNotFound
from otc_mcp.models import PatientProfile
from otc_mcp.profile import age_on, profile_from_bundle
from otc_mcp.safety import check_label

TODAY = date(2026, 11, 5)


@pytest.fixture()
def patients(settings):
    return LocalBundleSource(settings.patients_dir)


def _profile(patients, pid):
    return profile_from_bundle(patients.get_bundle(pid), pid, today=TODAY)


def _label(store, generic):
    return next(lbl for lbl in store.labels.values() if lbl.generic_name == generic)


def test_age_calculation():
    assert age_on("2020-05-20", TODAY) == 6
    assert age_on("1958-11-06", TODAY) == 67  # birthday tomorrow
    assert age_on(None) is None


def test_profile_from_fhir_bundle(patients):
    p = _profile(patients, "demo-older-warfarin")
    assert p.age_years == 68 and p.sex == "male"
    assert "Penicillin" in p.allergies
    assert any("Warfarin" in m for m in p.medications)
    assert "Essential hypertension" in p.conditions
    assert _profile(patients, "demo-pregnant").pregnant is True


def test_invalid_patient_ids_are_rejected(patients):
    with pytest.raises(PatientNotFound):
        patients.get_bundle("../etc/passwd")
    with pytest.raises(PatientNotFound):
        patients.get_bundle("no-such-patient")


def test_healthy_adult_ok_for_loratadine(store, patients):
    report = check_label(_label(store, "loratadine"), _profile(patients, "demo-adult-healthy"))
    assert report.verdict == "ok" and report.findings == []


def test_warfarin_patient_flagged_for_ibuprofen(store, patients):
    report = check_label(_label(store, "ibuprofen"), _profile(patients, "demo-older-warfarin"))
    assert report.verdict == "caution"
    cats = {f.category for f in report.findings}
    assert {"medication", "condition"} <= cats
    med = next(f for f in report.findings if f.category == "medication")
    assert "anticoagulant" in med.evidence.lower() or "blood thinning" in med.evidence.lower()


def test_aspirin_allergy_blocks_nsaid(store, patients):
    report = check_label(_label(store, "ibuprofen"), _profile(patients, "demo-child-asthma"))
    assert report.verdict == "do_not_use"
    assert any(f.category == "allergy" for f in report.findings)


def test_child_age_limits_from_directions(store, patients):
    child = _profile(patients, "demo-child-asthma")  # 6 years old
    report = check_label(_label(store, "dextromethorphan hydrobromide"), child)
    age = next(f for f in report.findings if f.category == "age")
    assert age.severity == "caution" and "4 to under 12 years: ask a doctor" in age.reason
    toddler = child.model_copy(update={"age_years": 3})
    blocked = check_label(_label(store, "dextromethorphan hydrobromide"), toddler)
    assert next(f for f in blocked.findings if f.category == "age").severity == "do_not_use"
    assert check_label(_label(store, "loratadine"), child).verdict == "ok"
    under_six = check_label(_label(store, "diphenhydramine hydrochloride"),
                            child.model_copy(update={"age_years": 5}))
    assert under_six.verdict == "do_not_use"


def test_maoi_in_do_not_use_section_blocks(store):
    p = PatientProfile(patient_id="x", display_name="X", age_years=40,
                       medications=["Phenelzine 15 MG Oral Tablet"])
    report = check_label(_label(store, "dextromethorphan hydrobromide"), p)
    assert report.verdict == "do_not_use"
    assert report.findings[0].label_section == "do_not_use"


def test_duplicate_acetaminophen_is_flagged(store):
    p = PatientProfile(patient_id="x", display_name="X", age_years=30,
                       medications=["Acetaminophen 325 MG / Hydrocodone 5 MG Oral Tablet"])
    report = check_label(_label(store, "acetaminophen"), p)
    assert any(f.category == "duplicate_ingredient" for f in report.findings)


def test_pregnancy_always_cautions(store, patients):
    report = check_label(_label(store, "ibuprofen"), _profile(patients, "demo-pregnant"))
    preg = next(f for f in report.findings if f.category == "pregnancy")
    assert "last 3 months" in preg.evidence
