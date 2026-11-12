"""Turn raw openFDA label records into clean DrugLabel objects and searchable chunks."""

from __future__ import annotations

import re
from typing import Any

from .models import LABEL_SECTIONS, SEARCH_SECTIONS, DrugLabel, LabelChunk

_WS = re.compile(r"\s+")
_SENTENCE = re.compile(r"(?<=[.;!?])\s+")


def clean_text(text: str) -> str:
    return _WS.sub(" ", text.replace("•", " ").replace("■", " ")).strip()


def _strip_heading(section: str, text: str) -> str:
    """openFDA text often repeats the heading ('Uses temporarily relieves...'). Drop it."""
    title = LABEL_SECTIONS[section]
    for prefix in (title, title.split(" ")[0]):
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix):].lstrip(" :-").strip()
    return text


def _first(values: Any) -> str | None:
    if isinstance(values, list) and values:
        return str(values[0])
    if isinstance(values, str):
        return values
    return None


def parse_openfda_label(record: dict[str, Any]) -> DrugLabel | None:
    """Convert one openFDA /drug/label.json result into a DrugLabel, or None if unusable."""
    openfda = record.get("openfda") or {}
    set_id = record.get("set_id")
    generic = _first(openfda.get("generic_name"))
    if not set_id or not generic:
        return None
    sections: dict[str, str] = {}
    for key in LABEL_SECTIONS:
        raw = record.get(key)
        if raw:
            text = clean_text(" ".join(raw) if isinstance(raw, list) else str(raw))
            text = _strip_heading(key, text)
            if text:
                sections[key] = text
    if not any(s in sections for s in SEARCH_SECTIONS):
        return None
    ingredients = [
        clean_text(i).lower()
        for i in (openfda.get("substance_name") or generic.split(","))
        if clean_text(i)
    ]
    return DrugLabel(
        set_id=set_id,
        brand_name=clean_text(_first(openfda.get("brand_name")) or generic).title(),
        generic_name=clean_text(generic).lower(),
        active_ingredients=ingredients,
        effective_time=record.get("effective_time"),
        sections=sections,
    )


def split_text(text: str, max_chars: int = 700) -> list[str]:
    """Split on sentence boundaries into pieces of at most ~max_chars."""
    if len(text) <= max_chars:
        return [text]
    pieces, current = [], ""
    for sentence in _SENTENCE.split(text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_label(label: DrugLabel) -> list[LabelChunk]:
    """Chunks for symptom search: only the 'Uses' and 'Purpose' sections."""
    chunks: list[LabelChunk] = []
    for section in SEARCH_SECTIONS:
        text = label.sections.get(section)
        if not text:
            continue
        for i, piece in enumerate(split_text(text)):
            chunks.append(
                LabelChunk(
                    chunk_id=f"{label.set_id}:{section}:{i}",
                    set_id=label.set_id,
                    section=section,
                    text=piece,
                )
            )
    return chunks


def embedding_text(label: DrugLabel, chunk: LabelChunk) -> str:
    """Text sent to the embedder: product context plus the chunk."""
    return (
        f"{label.brand_name} ({label.generic_name}). "
        f"{LABEL_SECTIONS[chunk.section]}: {chunk.text}"
    )
