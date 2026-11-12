"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # .../mcp_server
DEFAULT_DATA_DIR = PACKAGE_ROOT / "data"


def _split(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    index_dir: Path = field(
        default_factory=lambda: Path(os.getenv("OTC_INDEX_DIR", DEFAULT_DATA_DIR / "index"))
    )
    patients_dir: Path = field(
        default_factory=lambda: Path(os.getenv("OTC_PATIENTS_DIR", DEFAULT_DATA_DIR / "patients"))
    )
    # "fastembed" in production, "hashing" for offline tests and quick local runs.
    embedder: str = field(default_factory=lambda: os.getenv("OTC_EMBEDDER", "fastembed"))
    # Optional live FHIR server (e.g. HAPI FHIR). If unset, patients load from local bundles.
    fhir_base_url: str | None = field(default_factory=lambda: os.getenv("FHIR_BASE_URL") or None)
    openfda_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENFDA_API_KEY") or None
    )
    # Allow get_label to fall back to the live openFDA API for labels not in the local index.
    live_label_fallback: bool = field(
        default_factory=lambda: os.getenv("OTC_LIVE_LABEL_FALLBACK", "true").lower() == "true"
    )
    # Host headers accepted by the MCP transport (DNS-rebinding protection).
    # On Cloud Run, set to the service host, e.g. "otc-mcp-xyz-uc.a.run.app".
    allowed_hosts: list[str] = field(
        default_factory=lambda: _split(os.getenv("MCP_ALLOWED_HOSTS"))
        or ["localhost", "localhost:*", "127.0.0.1", "127.0.0.1:*"]
    )
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8080")))
    http_timeout: float = 15.0


def get_settings() -> Settings:
    return Settings()
