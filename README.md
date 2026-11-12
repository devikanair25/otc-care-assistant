# OTC Care Assistant

An agentic AI assistant that suggests **over-the-counter (OTC)** medications from FDA drug labels, checks each suggestion against the patient's FHIR record, and escalates red-flag symptoms to a pharmacist or clinician.

> **Educational demo.** It uses synthetic patients only and is not medical advice. Every recommendation cites its FDA label and tells the user to confirm with a pharmacist.

## Architecture

```
Web UI (React, Firebase Hosting)
   │  Firebase Auth token
   ▼
Agent API (FastAPI + Google ADK + Gemini Flash, Cloud Run, public)   ← Phase 2
   │  red-flag rules run first · sessions/audit log in Firestore
   ▼  MCP over Streamable HTTP, IAM-authenticated
MCP tool server (this phase, Cloud Run, private)
   ├─ search_otc_labels        RAG over FDA label "Uses" (FAISS + bge-small embeddings)
   ├─ get_label                full label sections (local index, openFDA fallback)
   ├─ normalize_drug           brand/misspelling → RxNorm ingredients (RxNav)
   ├─ list_patients            synthetic demo patients
   ├─ get_patient_profile      FHIR R4 → age, allergies, meds, conditions, pregnancy
   └─ check_contraindications  label warnings vs. patient profile → ok / caution / do_not_use
```

## Repository layout

```
mcp_server/
  src/otc_mcp/
    server.py         MCP server + tool definitions (MCP Python SDK v2, MCPServer)
    ingest.py         download OTC labels from openFDA
    build_index.py    chunk + embed + build FAISS index
    chunking.py       openFDA record → DrugLabel, section-aware chunks
    label_store.py    load/search the index
    embeddings.py     fastembed (prod) / hashing (offline tests)
    profile.py        FHIR Bundle → PatientProfile
    safety.py         rule-based label-vs-patient checks with quoted evidence
    clients/          openFDA, RxNav, FHIR server clients
  data/
    otc_ingredients.txt   ingredients to pull from openFDA
    patients/             synthetic FHIR R4 bundles (demo patients)
  tests/              pytest suite (offline; HTTP is mocked)
  Dockerfile
.github/workflows/mcp-server-ci.yml
```

## Run locally

```bash
cd mcp_server
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,embed]"

# 1. Download OTC labels from openFDA (needs internet; optional free key raises rate limits)
export OPENFDA_API_KEY=...          # optional, from https://open.fda.gov/apis/authentication/
otc-ingest --per-ingredient 5

# 2. Build the search index
otc-build-index --embedder fastembed

# 3. Start the MCP server on http://localhost:8080/mcp
otc-mcp-server
```

Try it with the MCP Inspector: `npx @modelcontextprotocol/inspector`. Connect to `http://localhost:8080/mcp` using the Streamable HTTP transport.

For a quick start with no model download, use `--embedder hashing` and `OTC_EMBEDDER=hashing`. Search quality is lower with that setting.

## Tests

```bash
cd mcp_server && pytest -q && ruff check .
```

The tests run offline. They use synthetic fixture labels in `tests/fixtures/`, which are simplified and are **not** real label text, plus mocked openFDA, RxNav and FHIR responses. They cover retrieval, FHIR parsing, every safety rule, and a full MCP client → server flow.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OTC_EMBEDDER` | `fastembed` | `fastembed` or `hashing` (must match how the index was built) |
| `OTC_INDEX_DIR` | `data/index` | Prebuilt index location |
| `OTC_PATIENTS_DIR` | `data/patients` | Local FHIR bundles |
| `FHIR_BASE_URL` | unset | Optional FHIR server, e.g. HAPI FHIR R4; falls back to local bundles |
| `OPENFDA_API_KEY` | unset | Higher openFDA rate limits |
| `OTC_LIVE_LABEL_FALLBACK` | `true` | Let `get_label` query openFDA for labels not in the index |
| `MCP_ALLOWED_HOSTS` | localhost | Comma-separated Host headers accepted (set to the Cloud Run host) |
| `PORT` | `8080` | Listen port (Cloud Run sets it) |

## Deploy the MCP server to Cloud Run (free tier)

The index is built **before** `docker build`, either on your machine or in CI, and copied into the image.

```bash
cd mcp_server
gcloud run deploy otc-mcp \
  --source . --region us-central1 \
  --no-allow-unauthenticated \
  --min-instances 0 --max-instances 2 --memory 1Gi \
  --set-env-vars OTC_EMBEDDER=fastembed

# After the first deploy, allow the service's own host header:
HOST=$(gcloud run services describe otc-mcp --region us-central1 --format 'value(status.url)' | sed 's#https://##')
gcloud run services update otc-mcp --region us-central1 --update-env-vars MCP_ALLOWED_HOSTS=$HOST

# Only the agent's service account may call it:
gcloud run services add-iam-policy-binding otc-mcp --region us-central1 \
  --member serviceAccount:otc-agent@PROJECT_ID.iam.gserviceaccount.com --role roles/run.invoker
```

## Design notes

- **Safety checks are rules, not the LLM.** `check_contraindications` compares allergies, medications, conditions, age limits and pregnancy against the label's *Do not use*, *Ask a doctor* and *Directions* sections. It quotes the triggering label sentence so every flag can be audited.
- **Drug interactions.** NLM discontinued the RxNav drug-interaction API in January 2024, so interactions are checked against the label's own warning text.
- **Retrieval** embeds only each label's *Uses* and *Purpose* sections. Warnings are fetched per product, so they never get mixed up between products.
- **Stateless Streamable HTTP** with DNS-rebinding protection. It scales to zero on Cloud Run.

## Data sources

- [openFDA drug label API](https://open.fda.gov/apis/drug/label/) and [DailyMed](https://dailymed.nlm.nih.gov/)
- [RxNav / RxNorm APIs](https://lhncbc.nlm.nih.gov/RxNav/)
- [HL7 FHIR R4](https://hl7.org/fhir/R4/) and [Synthea](https://synthetichealth.github.io/synthea/) synthetic patients

## Roadmap

- [x] Phase 1: MCP tool server, label RAG, FHIR profiles, safety checks, tests
- [ ] Phase 2: ADK agent API with red-flag triage and Gemini Flash
- [ ] Phase 3: React UI, Firebase Auth, Firestore audit log
- [ ] Phase 4: GitHub Actions deploy with Workload Identity Federation, scenario evaluation suite
