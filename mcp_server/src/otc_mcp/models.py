"""Typed data models shared by the ingestion pipeline and the MCP tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# openFDA label fields we keep. Keys are openFDA field names; values are display titles.
LABEL_SECTIONS: dict[str, str] = {
    "active_ingredient": "Active ingredient",
    "purpose": "Purpose",
    "indications_and_usage": "Uses",
    "warnings": "Warnings",
    "do_not_use": "Do not use",
    "ask_doctor": "Ask a doctor before use if you have",
    "ask_doctor_or_pharmacist": "Ask a doctor or pharmacist before use if you are",
    "stop_use": "Stop use and ask a doctor if",
    "pregnancy_or_breast_feeding": "Pregnancy or breast-feeding",
    "keep_out_of_reach_of_children": "Keep out of reach of children",
    "directions": "Directions",
}

# Sections embedded for symptom search. Warnings are retrieved per label, not by similarity.
SEARCH_SECTIONS: tuple[str, ...] = ("indications_and_usage", "purpose")


def dailymed_url(set_id: str) -> str:
    return f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}"


class DrugLabel(BaseModel):
    set_id: str
    brand_name: str
    generic_name: str
    active_ingredients: list[str] = Field(
        default_factory=list, description="Lower-case ingredient names, e.g. ['ibuprofen']"
    )
    effective_time: str | None = None
    sections: dict[str, str] = Field(default_factory=dict)

    @property
    def source_url(self) -> str:
        return dailymed_url(self.set_id)


class LabelChunk(BaseModel):
    chunk_id: str
    set_id: str
    section: str
    text: str


class SearchHit(BaseModel):
    set_id: str
    brand_name: str
    generic_name: str
    active_ingredients: list[str]
    matched_section: str
    matched_text: str
    score: float = Field(description="Cosine similarity, higher is closer")
    source_url: str


class SearchResult(BaseModel):
    query: str
    hits: list[SearchHit]
    note: str = (
        "Matches are based on the label's stated uses. Always run check_contraindications "
        "for the patient before suggesting a product."
    )


class LabelResult(BaseModel):
    found: bool
    label: DrugLabel | None = None
    source: Literal["local_index", "openfda_live", "none"] = "none"
    source_url: str | None = None
    message: str | None = None


class RxNormConcept(BaseModel):
    input_name: str
    found: bool
    rxcui: str | None = None
    matched_name: str | None = None
    term_type: str | None = None
    ingredients: list[str] = Field(default_factory=list)
    message: str | None = None


class PatientSummary(BaseModel):
    patient_id: str
    display_name: str
    age_years: int | None
    sex: str | None


class PatientProfile(BaseModel):
    patient_id: str
    display_name: str
    age_years: int | None = None
    sex: str | None = None
    pregnant: bool = False
    allergies: list[str] = Field(default_factory=list)
    medications: list[str] = Field(
        default_factory=list, description="Medication names as recorded, e.g. 'Warfarin 5 MG'"
    )
    conditions: list[str] = Field(default_factory=list)
    source: Literal["fhir_server", "local_bundle"] = "local_bundle"


Severity = Literal["do_not_use", "caution"]


class SafetyFinding(BaseModel):
    severity: Severity
    category: Literal["allergy", "medication", "duplicate_ingredient", "condition", "age",
                      "pregnancy"]
    reason: str
    label_section: str | None = None
    evidence: str | None = Field(default=None, description="Label sentence that triggered this")


class ContraindicationReport(BaseModel):
    set_id: str
    patient_id: str
    product: str
    verdict: Literal["ok", "caution", "do_not_use"]
    findings: list[SafetyFinding] = Field(default_factory=list)
    source_url: str
    disclaimer: str = (
        "Automated label check for an educational demo. Not medical advice; "
        "a pharmacist should confirm."
    )
