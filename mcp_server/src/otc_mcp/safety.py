"""Rule-based safety checks of an OTC label against a patient profile.

The rules compare the patient's allergies, medications, conditions, age and pregnancy status
with the label's own warning sections. Every finding quotes the label sentence that caused it,
so the agent can cite it and a reviewer can audit it. These rules are deliberately simple and
transparent; they flag for pharmacist review rather than make clinical decisions.
"""

from __future__ import annotations

import re

from .models import ContraindicationReport, DrugLabel, PatientProfile, SafetyFinding

NSAIDS = {"ibuprofen", "naproxen", "naproxen sodium", "aspirin", "ketoprofen"}

# Patient medication (ingredient) -> phrases that OTC labels use for that drug class.
MED_CLASS_TERMS: dict[str, list[str]] = {
    "warfarin": ["anticoagulant", "blood thinning", "blood thinner", "warfarin"],
    "apixaban": ["anticoagulant", "blood thinning", "blood thinner"],
    "rivaroxaban": ["anticoagulant", "blood thinning", "blood thinner"],
    "clopidogrel": ["blood thinning", "blood thinner"],
    "aspirin": ["aspirin"],
    "hydrochlorothiazide": ["diuretic"],
    "furosemide": ["diuretic"],
    "prednisone": ["steroid", "corticosteroid"],
    "phenelzine": ["monoamine oxidase inhibitor", "maoi"],
    "tranylcypromine": ["monoamine oxidase inhibitor", "maoi"],
    "selegiline": ["monoamine oxidase inhibitor", "maoi"],
    "isocarboxazid": ["monoamine oxidase inhibitor", "maoi"],
    "alprazolam": ["sedative", "tranquilizer"],
    "lorazepam": ["sedative", "tranquilizer"],
    "diazepam": ["sedative", "tranquilizer"],
    "zolpidem": ["sedative", "sleep"],
    "acetaminophen": ["acetaminophen"],
    "lithium": ["lithium"],
    "methotrexate": ["methotrexate"],
}

# Condition keyword (as it may appear in a FHIR Condition) -> label phrases.
CONDITION_TERMS: dict[str, list[str]] = {
    "hypertension": ["high blood pressure"],
    "high blood pressure": ["high blood pressure"],
    "kidney": ["kidney disease"],
    "renal": ["kidney disease"],
    "liver": ["liver disease"],
    "cirrhosis": ["liver disease"],
    "hepatitis": ["liver disease"],
    "asthma": ["asthma"],
    "heart failure": ["heart disease"],
    "coronary": ["heart disease"],
    "heart disease": ["heart disease"],
    "ulcer": ["stomach ulcers", "ulcer", "stomach bleeding"],
    "gastrointestinal hemorrhage": ["stomach bleeding"],
    "glaucoma": ["glaucoma"],
    "prostat": ["enlarged prostate", "trouble urinating"],
    "diabetes": ["diabetes"],
    "thyroid": ["thyroid disease"],
    "copd": ["emphysema", "chronic bronchitis", "breathing problem"],
    "emphysema": ["emphysema", "breathing problem"],
    "chronic bronchitis": ["chronic bronchitis", "breathing problem"],
    "phenylketonuria": ["phenylketonurics", "phenylalanine"],
}

BLOCK_SECTIONS = ("do_not_use",)
CAUTION_SECTIONS = ("ask_doctor", "ask_doctor_or_pharmacist", "warnings", "stop_use")

_SENTENCES = re.compile(r"(?<=[.;!?])\s+|\s+(?=•)")
# "children under 12 years: ask a doctor" / "children 4 to under 12 years: ask a doctor" /
# "children under 2 years of age: do not use"
_CHILD_RULE = re.compile(
    r"children (?:(\d{1,2}) to )?under (\d{1,2}) years?(?: of age)?\s*[:\-]?\s*"
    r"(do not use|ask a doctor)",
    re.IGNORECASE,
)


def _ingredient_of(med_text: str) -> str:
    """'Warfarin Sodium 5 MG Oral Tablet' -> 'warfarin sodium'."""
    head = re.split(r"\d", med_text.lower(), maxsplit=1)[0]
    return re.sub(r"[\[\](),]", " ", head).strip()


def _evidence(text: str, phrase: str) -> str:
    for sentence in _SENTENCES.split(text):
        if phrase.lower() in sentence.lower():
            return sentence.strip()[:300]
    return text[:300]


def _find(label: DrugLabel, phrases: list[str], sections: tuple[str, ...]):
    for section in sections:
        text = label.sections.get(section, "")
        low = text.lower()
        for phrase in phrases:
            if re.search(rf"\b{re.escape(phrase.lower())}", low):
                return section, phrase, _evidence(text, phrase)
    return None


def check_label(label: DrugLabel, profile: PatientProfile) -> ContraindicationReport:
    findings: list[SafetyFinding] = []
    label_ings = set(label.active_ingredients)

    # 1. Allergies to an active ingredient (or to NSAIDs as a class)
    for allergy in profile.allergies:
        a = allergy.lower()
        direct = [i for i in label_ings if i in a or a in i]
        nsaid = ("nsaid" in a or any(n in a for n in NSAIDS)) and label_ings & NSAIDS
        if direct or nsaid:
            findings.append(SafetyFinding(
                severity="do_not_use", category="allergy",
                reason=f"Patient has a recorded allergy to '{allergy}', which matches "
                       f"{', '.join(sorted(direct or (label_ings & NSAIDS)))} in this product.",
                label_section="do_not_use" if "do_not_use" in label.sections else None,
                evidence=_evidence(label.sections.get("do_not_use", ""), "allerg") or None,
            ))

    # 2. Current medications: duplicate ingredients and class warnings on the label
    for med in profile.medications:
        ing = _ingredient_of(med)
        base = ing.split()[0] if ing else ""
        if base and any(base in i for i in label_ings):
            findings.append(SafetyFinding(
                severity="caution", category="duplicate_ingredient",
                reason=f"Patient already takes '{med}', which contains the same active "
                       "ingredient. Risk of taking too much.",
            ))
            continue
        terms = MED_CLASS_TERMS.get(base, []) + ([base] if len(base) > 4 else [])
        if not terms:
            continue
        hit = _find(label, terms, BLOCK_SECTIONS) or _find(label, terms, CAUTION_SECTIONS)
        if hit:
            section, phrase, evidence = hit
            findings.append(SafetyFinding(
                severity="do_not_use" if section in BLOCK_SECTIONS else "caution",
                category="medication",
                reason=f"Patient takes '{med}'; the label warns about '{phrase}'.",
                label_section=section, evidence=evidence,
            ))

    # 3. Conditions named in the label's warnings
    for condition in profile.conditions:
        c = condition.lower()
        terms = sorted({t for key, ts in CONDITION_TERMS.items() if key in c for t in ts})
        if not terms:
            continue
        hit = _find(label, terms, BLOCK_SECTIONS) or _find(label, terms, CAUTION_SECTIONS)
        if hit:
            section, phrase, evidence = hit
            findings.append(SafetyFinding(
                severity="do_not_use" if section in BLOCK_SECTIONS else "caution",
                category="condition",
                reason=f"Patient has '{condition}'; the label mentions '{phrase}'.",
                label_section=section, evidence=evidence,
            ))

    # 4. Age limits written in the directions
    if profile.age_years is not None:
        directions = label.sections.get("directions", "")
        for m in _CHILD_RULE.finditer(directions):
            low = int(m.group(1)) if m.group(1) else 0
            high, action = int(m.group(2)), m.group(3).lower()
            if low <= profile.age_years < high:
                findings.append(SafetyFinding(
                    severity="do_not_use" if action == "do not use" else "caution",
                    category="age",
                    reason=f"Patient is {profile.age_years}; label says '{m.group(0)}'.",
                    label_section="directions", evidence=m.group(0),
                ))
                break
        if profile.age_years < 2 and not any(f.category == "age" for f in findings):
            findings.append(SafetyFinding(
                severity="caution", category="age",
                reason="Patient is under 2 years old. OTC use should be confirmed with a "
                       "pediatrician or pharmacist.",
            ))

    # 5. Pregnancy
    if profile.pregnant:
        text = label.sections.get("pregnancy_or_breast_feeding", "")
        findings.append(SafetyFinding(
            severity="caution", category="pregnancy",
            reason="Patient is pregnant. Confirm with a doctor or pharmacist before use.",
            label_section="pregnancy_or_breast_feeding" if text else None,
            evidence=text[:300] or None,
        ))

    if any(f.severity == "do_not_use" for f in findings):
        verdict = "do_not_use"
    elif findings:
        verdict = "caution"
    else:
        verdict = "ok"
    return ContraindicationReport(
        set_id=label.set_id, patient_id=profile.patient_id,
        product=f"{label.brand_name} ({label.generic_name})",
        verdict=verdict, findings=findings, source_url=label.source_url,
    )
