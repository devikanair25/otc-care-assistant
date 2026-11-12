from otc_mcp.chunking import chunk_label, parse_openfda_label, split_text
from otc_mcp.embeddings import HashingEmbedder
from otc_mcp.label_store import LabelStore


def test_parse_strips_headings_and_normalizes(raw_records):
    label = parse_openfda_label(raw_records[0])
    assert label.generic_name == "ibuprofen"
    assert label.active_ingredients == ["ibuprofen"]
    assert label.sections["indications_and_usage"].startswith("temporarily relieves")
    assert label.sections["purpose"] == "Pain reliever/fever reducer"
    assert "dailymed" in label.source_url


def test_parse_skips_records_without_generic_name():
    assert parse_openfda_label({"set_id": "x", "openfda": {}}) is None


def test_split_text_respects_limit():
    text = " ".join(f"Sentence number {i} is here." for i in range(100))
    pieces = split_text(text, max_chars=200)
    assert len(pieces) > 1 and all(len(p) <= 200 for p in pieces)


def test_chunks_only_cover_search_sections(labels):
    sections = {c.section for lbl in labels for c in chunk_label(lbl)}
    assert sections <= {"indications_and_usage", "purpose"}


def test_search_finds_relevant_products(store):
    top = store.search("runny nose sneezing itchy watery eyes hay fever", top_k=3)
    assert top[0].generic_name in {"loratadine", "diphenhydramine hydrochloride"}
    cough = store.search("cough from a cold, throat irritation", top_k=2)
    assert cough[0].generic_name == "dextromethorphan hydrobromide"
    diarrhea = store.search("diarrhea", top_k=1)
    assert diarrhea[0].generic_name == "loperamide hydrochloride"


def test_search_returns_one_hit_per_ingredient_combo(store):
    hits = store.search("headache and fever", top_k=10)
    combos = [tuple(h.active_ingredients) for h in hits]
    assert len(combos) == len(set(combos))


def test_save_and_load_roundtrip(store, tmp_path):
    store.save(tmp_path)
    loaded = LabelStore.load(tmp_path, HashingEmbedder())
    assert len(loaded.labels) == len(store.labels)
    assert loaded.search("heartburn", 1)[0].generic_name == "famotidine"


def test_load_rejects_mismatched_embedder(store, tmp_path):
    store.save(tmp_path)
    try:
        LabelStore.load(tmp_path, HashingEmbedder(dim=256))
    except ValueError as exc:
        assert "Rebuild the index" in str(exc)
    else:
        raise AssertionError("expected ValueError")
