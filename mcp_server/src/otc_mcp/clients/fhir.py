"""Load FHIR R4 patient data from a FHIR server or from local Bundle files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

_SAFE_ID = re.compile(r"^[A-Za-z0-9\-.]{1,64}$")  # FHIR id rules


class PatientNotFound(LookupError):
    pass


def validate_patient_id(patient_id: str) -> str:
    if not _SAFE_ID.match(patient_id or ""):
        raise PatientNotFound(f"Invalid patient id: {patient_id!r}")
    return patient_id


class LocalBundleSource:
    """Reads `<patients_dir>/<patient_id>.json`, each a FHIR Bundle (Synthea output works)."""

    def __init__(self, patients_dir: Path) -> None:
        self.patients_dir = patients_dir

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.patients_dir.glob("*.json"))

    def get_bundle(self, patient_id: str) -> dict[str, Any]:
        path = self.patients_dir / f"{validate_patient_id(patient_id)}.json"
        if not path.exists():
            raise PatientNotFound(f"No local bundle for patient {patient_id}")
        return json.loads(path.read_text(encoding="utf-8"))


class FHIRServerSource:
    """Reads a patient's record from a FHIR server with Patient/$everything."""

    def __init__(self, base_url: str, timeout: float = 15.0, max_pages: int = 5,
                 client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_pages = max_pages
        self._client = client or httpx.Client(
            timeout=timeout, headers={"Accept": "application/fhir+json"}
        )

    def get_bundle(self, patient_id: str) -> dict[str, Any]:
        url = f"{self.base_url}/Patient/{validate_patient_id(patient_id)}/$everything"
        entries: list[dict[str, Any]] = []
        for _ in range(self.max_pages):
            resp = self._client.get(url)
            if resp.status_code == 404:
                raise PatientNotFound(f"Patient {patient_id} not found on FHIR server")
            resp.raise_for_status()
            page = resp.json()
            entries.extend(page.get("entry") or [])
            url = next((link["url"] for link in page.get("link", [])
                        if link.get("relation") == "next"), None)
            if not url:
                break
        return {"resourceType": "Bundle", "type": "searchset", "entry": entries}
