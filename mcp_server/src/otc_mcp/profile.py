"""Turn a FHIR R4 Bundle into the compact PatientProfile the agent reasons over."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from .models import PatientProfile, PatientSummary

_INACTIVE = {"inactive", "resolved", "remission", "entered-in-error", "refuted"}
_MED_INACTIVE = {"stopped", "completed", "cancelled", "entered-in-error", "not-taken",
                 "revoked", "ended"}


def _codeable_text(cc: dict[str, Any] | None) -> str | None:
    if not cc:
        return None
    if cc.get("text"):
        return cc["text"]
    for coding in cc.get("coding") or []:
        if coding.get("display"):
            return coding["display"]
    return None


def _status(resource: dict[str, Any], key: str) -> str | None:
    value = resource.get(key)
    if isinstance(value, dict):  # CodeableConcept, e.g. clinicalStatus
        codes = [c.get("code") for c in value.get("coding") or []]
        return codes[0] if codes else None
    return value


def age_on(birth_date: str | None, today: date | None = None) -> int | None:
    if not birth_date:
        return None
    try:
        y, m, d = (int(x) for x in birth_date[:10].split("-"))
    except ValueError:
        return None
    today = today or date.today()
    return today.year - y - ((today.month, today.day) < (m, d))


def _display_name(patient: dict[str, Any]) -> str:
    names = patient.get("name") or []
    if not names:
        return patient.get("id", "Unknown patient")
    n = names[0]
    if n.get("text"):
        return n["text"]
    given = " ".join(n.get("given") or [])
    return re.sub(r"\d+", "", f"{given} {n.get('family', '')}").strip()  # Synthea adds digits


def _resources(bundle: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [
        e["resource"]
        for e in bundle.get("entry") or []
        if (e.get("resource") or {}).get("resourceType") == kind
    ]


def _medication_text(resource: dict[str, Any]) -> str | None:
    if "medicationCodeableConcept" in resource:  # R4
        return _codeable_text(resource["medicationCodeableConcept"])
    med = resource.get("medication") or {}  # R5 CodeableReference
    return _codeable_text(med.get("concept")) or (med.get("reference") or {}).get("display")


def profile_from_bundle(bundle: dict[str, Any], patient_id: str,
                        source: str = "local_bundle", today: date | None = None) -> PatientProfile:
    patients = [p for p in _resources(bundle, "Patient") if p.get("id") == patient_id] or \
        _resources(bundle, "Patient")
    if not patients:
        raise LookupError(f"Bundle has no Patient resource for {patient_id}")
    patient = patients[0]

    allergies = sorted({
        t for r in _resources(bundle, "AllergyIntolerance")
        if _status(r, "clinicalStatus") not in _INACTIVE
        and (t := _codeable_text(r.get("code")))
    })
    meds = sorted({
        t
        for kind in ("MedicationStatement", "MedicationRequest")
        for r in _resources(bundle, kind)
        if (r.get("status") or "active") not in _MED_INACTIVE
        and (t := _medication_text(r))
    })
    conditions = sorted({
        t for r in _resources(bundle, "Condition")
        if _status(r, "clinicalStatus") not in _INACTIVE
        and (t := _codeable_text(r.get("code")))
    })
    pregnant = any("pregnan" in c.lower() for c in conditions)

    return PatientProfile(
        patient_id=patient_id,
        display_name=_display_name(patient),
        age_years=age_on(patient.get("birthDate"), today),
        sex=patient.get("gender"),
        pregnant=pregnant,
        allergies=allergies,
        medications=meds,
        conditions=conditions,
        source=source,  # type: ignore[arg-type]
    )


def summary_from_bundle(bundle: dict[str, Any], patient_id: str) -> PatientSummary:
    prof = profile_from_bundle(bundle, patient_id)
    return PatientSummary(patient_id=patient_id, display_name=prof.display_name,
                          age_years=prof.age_years, sex=prof.sex)
