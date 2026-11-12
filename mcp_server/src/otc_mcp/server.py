"""MCP tool server (Streamable HTTP) for the OTC Care Assistant.

Run locally:
    OTC_EMBEDDER=hashing otc-mcp-server        # serves http://localhost:8080/mcp
On Cloud Run the container listens on $PORT and the service is private (IAM invoker only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse

from .clients.fhir import FHIRServerSource, LocalBundleSource, PatientNotFound
from .clients.openfda import OpenFDAClient
from .clients.rxnav import RxNavClient
from .config import Settings, get_settings
from .embeddings import get_embedder
from .label_store import LabelStore
from .models import (
    ContraindicationReport,
    LabelResult,
    PatientProfile,
    PatientSummary,
    RxNormConcept,
    SearchResult,
)
from .profile import profile_from_bundle, summary_from_bundle
from .safety import check_label

log = logging.getLogger("otc_mcp")
READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
READ_ONLY_REMOTE = ToolAnnotations(readOnlyHint=True, openWorldHint=True)

INSTRUCTIONS = """\
Tools for an OTC (over-the-counter) medication guidance assistant. Data comes from FDA drug
labels (openFDA), RxNorm and synthetic FHIR patient records.
Typical flow: get_patient_profile -> search_otc_labels(symptoms) -> check_contraindications for
each candidate -> get_label for dosing directions. Never suggest a product whose verdict is
'do_not_use'. Quote label text for dosing; do not invent doses. This is an educational demo,
not medical advice.
"""


@dataclass
class Services:
    store: LabelStore
    openfda: OpenFDAClient
    rxnav: RxNavClient
    local_patients: LocalBundleSource
    fhir: FHIRServerSource | None
    settings: Settings

    def load_profile(self, patient_id: str) -> PatientProfile:
        if self.fhir:
            try:
                return profile_from_bundle(self.fhir.get_bundle(patient_id), patient_id,
                                           source="fhir_server")
            except PatientNotFound:
                pass  # fall back to local synthetic bundles
        return profile_from_bundle(self.local_patients.get_bundle(patient_id), patient_id)


def build_services(settings: Settings | None = None, store: LabelStore | None = None) -> Services:
    settings = settings or get_settings()
    store = store or LabelStore.load(settings.index_dir, get_embedder(settings.embedder))
    return Services(
        store=store,
        openfda=OpenFDAClient(settings.openfda_api_key, settings.http_timeout),
        rxnav=RxNavClient(settings.http_timeout),
        local_patients=LocalBundleSource(settings.patients_dir),
        fhir=FHIRServerSource(settings.fhir_base_url, settings.http_timeout)
        if settings.fhir_base_url else None,
        settings=settings,
    )


def create_server(services: Services) -> MCPServer:
    mcp = MCPServer(name="otc-care-tools", version="0.1.0", instructions=INSTRUCTIONS)

    @mcp.tool(annotations=READ_ONLY)
    def search_otc_labels(symptoms: str, top_k: int = 5) -> SearchResult:
        """Find OTC products whose FDA label 'Uses' match the described symptoms.

        Args:
            symptoms: Plain-language symptoms, e.g. "runny nose, sneezing, itchy eyes".
            top_k: Number of distinct products (by active-ingredient combination), 1-10.
        """
        top_k = max(1, min(int(top_k), 10))
        return SearchResult(query=symptoms, hits=services.store.search(symptoms, top_k))

    @mcp.tool(annotations=READ_ONLY_REMOTE)
    def get_label(set_id: str) -> LabelResult:
        """Get the full FDA label sections (uses, warnings, directions) for a product set_id."""
        label = services.store.get(set_id)
        if label:
            return LabelResult(found=True, label=label, source="local_index",
                               source_url=label.source_url)
        if services.settings.live_label_fallback:
            try:
                label = services.openfda.label_by_set_id(set_id)
            except Exception as exc:  # network errors should not crash the tool
                log.warning("openFDA lookup failed for %s: %s", set_id, exc)
                label = None
            if label:
                return LabelResult(found=True, label=label, source="openfda_live",
                                   source_url=label.source_url)
        return LabelResult(found=False, message=f"No label found for set_id {set_id}")

    @mcp.tool(annotations=READ_ONLY_REMOTE)
    def normalize_drug(name: str) -> RxNormConcept:
        """Map a brand or misspelled drug name to RxNorm ingredients (e.g. 'Advil' -> ibuprofen)."""
        return services.rxnav.normalize(name)

    @mcp.tool(annotations=READ_ONLY)
    def list_patients() -> list[PatientSummary]:
        """List the synthetic demo patients available for the patient picker."""
        out = []
        for pid in services.local_patients.list_ids():
            try:
                out.append(summary_from_bundle(services.local_patients.get_bundle(pid), pid))
            except Exception as exc:
                log.warning("Skipping patient %s: %s", pid, exc)
        return out

    @mcp.tool(annotations=READ_ONLY_REMOTE)
    def get_patient_profile(patient_id: str) -> PatientProfile:
        """Get a patient's age, sex, pregnancy status, allergies, medications and conditions
        from their FHIR record."""
        return services.load_profile(patient_id)

    @mcp.tool(annotations=READ_ONLY_REMOTE)
    def check_contraindications(set_id: str, patient_id: str) -> ContraindicationReport:
        """Check one product's FDA label warnings against a patient's FHIR profile.

        Returns verdict 'ok', 'caution' (pharmacist should confirm) or 'do_not_use', with the
        label sentence behind each finding.
        """
        result = get_label(set_id)
        if not result.found or not result.label:
            raise ValueError(f"Unknown product set_id {set_id}")
        return check_label(result.label, services.load_profile(patient_id))

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "labels": len(services.store.labels)})

    return mcp


def build_app():
    """ASGI app factory used by uvicorn (`uvicorn otc_mcp.server:build_app --factory`)."""
    settings = get_settings()
    server = create_server(build_services(settings))
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=settings.allowed_hosts,
        allowed_origins=[],
    )
    # Stateless + JSON responses: each request stands alone, which suits Cloud Run autoscaling.
    return server.streamable_http_app(stateless_http=True, json_response=True,
                                      transport_security=security, host=settings.host)


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    s = get_settings()
    uvicorn.run(build_app(), host=s.host, port=s.port, log_level="info")


if __name__ == "__main__":
    main()
