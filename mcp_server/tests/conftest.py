from __future__ import annotations

import json
from pathlib import Path

import pytest

from otc_mcp.build_index import load_raw_labels
from otc_mcp.config import Settings
from otc_mcp.embeddings import HashingEmbedder
from otc_mcp.label_store import LabelStore
from otc_mcp.server import build_services, create_server

FIXTURES = Path(__file__).parent / "fixtures"
PATIENTS = Path(__file__).parents[1] / "data" / "patients"


@pytest.fixture(scope="session")
def raw_records() -> list[dict]:
    path = FIXTURES / "sample_openfda_labels.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="session")
def labels():
    return load_raw_labels(FIXTURES / "sample_openfda_labels.jsonl")


@pytest.fixture(scope="session")
def store(labels) -> LabelStore:
    return LabelStore.build(labels, HashingEmbedder())


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(index_dir=tmp_path / "index", patients_dir=PATIENTS, embedder="hashing",
                    fhir_base_url=None, live_label_fallback=False)


@pytest.fixture()
def services(settings, store):
    return build_services(settings, store=store)


@pytest.fixture()
def server(services):
    return create_server(services)
