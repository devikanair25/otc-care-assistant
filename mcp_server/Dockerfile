# MCP tool server image for Cloud Run.
# Build the index first (otc-ingest + otc-build-index) so data/index/ exists.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/models \
    OTC_INDEX_DIR=/app/data/index \
    OTC_PATIENTS_DIR=/app/data/patients \
    OTC_EMBEDDER=fastembed \
    PORT=8080

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[embed]"

# Bake the embedding model into the image so cold starts never download it.
RUN python -c "from otc_mcp.embeddings import FastEmbedEmbedder; FastEmbedEmbedder()"

COPY data ./data
RUN test -f data/index/meta.json || \
    (echo "ERROR: data/index is missing. Run otc-ingest and otc-build-index first." && exit 1)

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8080
CMD ["otc-mcp-server"]
