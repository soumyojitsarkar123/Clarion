"""Isolated regression tests for source-offset mapping and migration recovery."""

import sqlite3

import pytest

from services.provenance_migrations import apply_provenance_schema
from services.source_provenance import NORMALIZATION_MAP_VERSION, build_normalization_map, map_normalized_interval


def test_normalization_map_tracks_copy_replacement_and_deleted_text():
    mapping = build_normalization_map("alpha  cafe\u0301\nTAIL", "alpha caf\u00e9")
    mapped, operations = map_normalized_interval(mapping, 0, len("alpha caf\u00e9"))

    assert mapping["version"] == NORMALIZATION_MAP_VERSION
    assert mapped
    assert mapped[0]["source_start"] == 0
    assert "normalization_replacement" in operations
    assert mapping["removed_source_intervals"]


def test_fresh_and_legacy_migrations_are_additive_and_idempotent(tmp_path):
    fresh = tmp_path / "fresh.db"
    apply_provenance_schema(fresh)
    apply_provenance_schema(fresh)
    with sqlite3.connect(fresh) as conn:
        assert conn.execute("SELECT version FROM schema_migrations").fetchall() == [(1,)]
        assert {row[1] for row in conn.execute("PRAGMA table_info(embedding_vector_map)")} >= {
            "vector_id", "segment_id", "retained_chunk_id", "raw_chunk_ids_json"
        }

    legacy = tmp_path / "legacy.db"
    with sqlite3.connect(legacy) as conn:
        conn.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, filename TEXT NOT NULL)")
        conn.execute("CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, document_id TEXT, content TEXT)")
        conn.execute("INSERT INTO documents VALUES ('doc-1', 'legacy.pdf')")
        conn.execute("INSERT INTO chunks VALUES ('chunk-1', 'doc-1', 'old text')")
    apply_provenance_schema(legacy)
    with sqlite3.connect(legacy) as conn:
        row = conn.execute("SELECT chunk_id, content, source_provenance_json FROM chunks").fetchone()
        assert row == ("chunk-1", "old text", None)
        assert conn.execute("SELECT filename FROM documents WHERE id='doc-1'").fetchone() == ("legacy.pdf",)


def test_migration_failure_rolls_back_schema_changes(tmp_path):
    db_path = tmp_path / "rollback.db"
    with pytest.raises(RuntimeError, match="Injected provenance migration failure"):
        apply_provenance_schema(db_path, fail_after_step=2)
    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "schema_migrations" in tables  # bootstrap ledger is outside the schema transaction
        assert "document_source_pages" not in tables
        assert "embedding_vector_map" not in tables
        assert conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone() == (0,)


def test_chunk_to_faiss_vector_keeps_page_split_source_spans(isolated_settings, monkeypatch):
    from services.chunking_service import ChunkingService
    from services.embedding_service import EmbeddingService
    from services.source_provenance import CHUNK_PROVENANCE_VERSION
    import services.embedding_service as embedding_module

    text = "alpha evidence\n\nbeta evidence"
    metadata = {
        "location_kind": "pdf_extracted_page_text",
        "offset_map": {
            "runs": [{
                "normalized_start": 0, "normalized_end": len(text),
                "source_start": 0, "source_end": len(text), "operation": "copy"
            }]
        },
    }
    pages = [
        {"page_number": 1, "source_char_start": 0, "source_char_end": 14},
        {"page_number": 2, "source_char_start": 16, "source_char_end": len(text)},
    ]
    chunk_service = ChunkingService()
    chunks = chunk_service.chunk_document("doc-provenance", text, metadata, pages)
    assert chunks[0].source_provenance["version"] == CHUNK_PROVENANCE_VERSION

    embedding_service = EmbeddingService()
    monkeypatch.setattr(embedding_service, "_load_model", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    segments, vectors = embedding_service.generate_embedding_segments(chunks)
    class IsolatedIndex:
        def __init__(self, document_id, model_id=None, model_revision=None):
            self.model_id = model_id
            self.model_revision = model_revision
            self.ids = []
        def create_index(self, values, ids):
            assert len(values) == len(ids)
            self.ids = list(ids)
        def search(self, _query, _k):
            return [(self.ids[0], 0.0)]
        def delete_index(self):
            self.ids = []
    monkeypatch.setattr(embedding_module, "VectorStore", IsolatedIndex)
    store = embedding_service.create_vector_index("doc-provenance", segments, vectors)
    assert store.search(vectors[0], 1)[0][0] == segments[0].chunk_id

    conn = sqlite3.connect(embedding_service.db_path)
    try:
        rows = conn.execute(
            "SELECT vector_id, segment_id, retained_chunk_id, source_provenance_json "
            "FROM embedding_vector_map WHERE document_id=? ORDER BY vector_index",
            ("doc-provenance",),
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == len(vectors)
    assert rows[0][1] is None  # fallback path has no model-tokenized segment id
    assert rows[0][2] == chunks[0].chunk_id
    import json
    provenance = json.loads(rows[0][3])
    assert provenance["page_references"] == [1, 2]
    assert [span["page_number"] for span in provenance["source_intervals"]] == [1, 2]

    # Reindexing the same model replaces its mapping atomically without
    # duplicating vector rows; deletion clears the isolated document mapping.
    embedding_service.create_vector_index("doc-provenance", segments, vectors)
    conn = sqlite3.connect(embedding_service.db_path)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM embedding_vector_map WHERE document_id=?", ("doc-provenance",)
        ).fetchone()[0] == len(vectors)
    finally:
        conn.close()
    embedding_service.delete_embeddings("doc-provenance")
    conn = sqlite3.connect(embedding_service.db_path)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM embedding_vector_map WHERE document_id=?", ("doc-provenance",)
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_retrieval_exposes_verified_source_citation_fields():
    from models.chunk import Chunk
    from services.retrieval_service import RetrievalService

    provenance = {
        "page_references": [3],
        "source_intervals": [{
            "source_start": 20, "source_end": 42, "page_number": 3,
            "page_char_start": 0, "page_char_end": 22, "mapping": "exact",
        }],
    }

    class Store:
        model_id = "BAAI/bge-large-en-v1.5@fixture-revision"
        model_revision = "fixture-revision"
        def search(self, _query, _k):
            return [("retained-1:seg:0", 0.1)]

    class Embedding:
        def get_vector_store(self, _document_id):
            return Store()
        def embed_query(self, _query):
            return [0.0]
        def get_embedding_segment(self, segment_id, model_id, revision):
            assert segment_id == "retained-1:seg:0"
            assert model_id == Store.model_id and revision == Store.model_revision
            return {
                "source_chunk_id": "retained-1",
                "content": "verified passage",
                "source_provenance": provenance,
            }

    class Chunking:
        def get_chunks(self, _document_id):
            return [Chunk(
                chunk_id="retained-1", document_id="doc-1", content="source passage",
                position_index=0, word_count=2, raw_chunk_ids=["raw-1", "raw-2"],
            )]

    service = RetrievalService.__new__(RetrievalService)
    service.embedding_service = Embedding()
    service.chunking_service = Chunking()
    service.default_top_k = 5
    service.similarity_threshold = 0.0
    result = service.retrieve("doc-1", "question")[0]
    assert result.document_id == "doc-1"
    assert result.segment_id == "retained-1:seg:0"
    assert result.page_references == [3]
    assert result.source_intervals[0]["page_char_start"] == 0
    assert result.raw_chunk_ids == ["raw-1", "raw-2"]
    assert result.embedding_model_id == Store.model_id


def test_query_encoder_refuses_model_mismatch_for_selected_faiss_index(isolated_settings, monkeypatch):
    from services.embedding_service import EmbeddingService

    service = EmbeddingService()
    service.active_model_id = "BAAI/bge-large-en-v1.5@required-revision"
    monkeypatch.setattr(service, "_load_model", lambda: (_ for _ in ()).throw(RuntimeError("model unavailable")))
    with pytest.raises(RuntimeError, match="selected FAISS index requires"):
        service.embed_query("query")
