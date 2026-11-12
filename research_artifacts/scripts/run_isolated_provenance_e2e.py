"""Run real PDF-to-BGE-to-FAISS provenance checks in a dedicated workspace DB."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) not in (3, 4):
        raise SystemExit("usage: run_isolated_provenance_e2e.py <backend-dir> <pdf-path> [--reuse]")
    reuse = len(sys.argv) == 4 and sys.argv[3] == "--reuse"
    backend_dir, pdf_path = map(lambda value: Path(value).resolve(), sys.argv[1:3])
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    from utils.config import settings

    output_dir = Path(__file__).resolve().parents[1] / "isolated-provenance-e2e-2026-11-12"
    settings.data_dir = output_dir / "data"
    settings.vectorstore_dir = settings.data_dir / "vectorstore"
    settings.logs_dir = output_dir / "logs"
    settings.dataset_dir = settings.data_dir / "datasets"
    settings.database_url = f"sqlite:///{settings.data_dir / 'clarion.db'}"
    for directory in (settings.data_dir, settings.vectorstore_dir, settings.logs_dir, settings.dataset_dir):
        directory.mkdir(parents=True, exist_ok=True)

    from services.document_service import DocumentService
    from services.chunking_service import ChunkingService
    from services.embedding_service import EmbeddingService
    from utils.file_handler import extract_text

    pdf_bytes = pdf_path.read_bytes()
    document_service = DocumentService()
    existing_manifest = output_dir / "isolated_provenance_e2e_manifest.json"
    if reuse:
        document_id = json.loads(existing_manifest.read_text(encoding="utf-8"))["document_id"]
        loaded_document = document_service.get_document(document_id)
        if not loaded_document:
            raise RuntimeError("The isolated document record for --reuse was not found")
        refreshed_text, refreshed_meta = extract_text(pdf_bytes, pdf_path.name)
        loaded_document.text_content = refreshed_text
        loaded_document.source_provenance = refreshed_meta.get("source_provenance", {})
        loaded_document.source_pages = refreshed_meta.get("source_pages", [])
        loaded_document.source_blocks = refreshed_meta.get("source_blocks", [])
        loaded_document.metadata.page_count = refreshed_meta.get("page_count")
        loaded_document.metadata.word_count = len(refreshed_text.split())
        document_service._save_to_db(loaded_document)
        document = loaded_document
    else:
        document = document_service.upload_document(pdf_bytes, pdf_path.name)
        loaded_document = document_service.get_document(document.id)
    assert loaded_document and loaded_document.source_pages
    assert len(loaded_document.source_pages) == loaded_document.metadata.page_count

    chunk_service = ChunkingService()
    if reuse:
        chunk_service.delete_chunks(document.id)
    chunks = chunk_service.chunk_document(
        document.id,
        loaded_document.text_content or "",
        source_metadata=loaded_document.source_provenance,
        source_pages=loaded_document.source_pages,
        source_blocks=loaded_document.source_blocks,
    )
    assert chunks, "The selected PDF produced no chunks"

    embedding_service = EmbeddingService()
    if reuse:
        embedding_service._load_model()
        cached_identity = f"{settings.embedding_model_name}@revision-unavailable"
        resolved_identity = embedding_service._configured_model_id()
        if resolved_identity != cached_identity:
            with __import__("utils.sqlite", fromlist=["connect"]).connect(embedding_service.db_path) as conn:
                conn.execute(
                    "UPDATE chunk_embeddings SET model_name=? WHERE model_name=?",
                    (resolved_identity, cached_identity),
                )
                conn.commit()
    segments, vectors = embedding_service.generate_embedding_segments(chunks)
    assert segments and len(segments) == len(vectors)
    assert not embedding_service.using_fallback_embeddings, "Configured BGE did not produce the vectors"
    store = embedding_service.create_vector_index(document.id, segments, vectors)
    assert store.get_stats()["total_vectors"] == len(vectors)

    vector_ids = set(store.chunk_ids)
    with __import__("utils.sqlite", fromlist=["connect"]).connect(embedding_service.db_path) as conn:
        map_rows = conn.execute(
            "SELECT vector_id, segment_id, retained_chunk_id, raw_chunk_ids_json, source_provenance_json "
            "FROM embedding_vector_map WHERE document_id=? AND embedding_model_id=? AND embedding_model_revision=?",
            (document.id, store.model_id, store.model_revision),
        ).fetchall()
    assert len(map_rows) == len(vector_ids) == len(vectors)
    assert {row[0] for row in map_rows} == vector_ids
    provenance_rows = [json.loads(row[4] or "{}") for row in map_rows]
    assert all(
        item.get("document_id") == document.id
        and item.get("retained_chunk_id") == row[2]
        and item.get("raw_chunk_ids") == json.loads(row[3])
        and item.get("vector_id") == row[0]
        and item.get("embedding_model_id") == store.model_id
        and item.get("embedding_model_revision") == store.model_revision
        and item.get("extraction_version") == loaded_document.source_provenance.get("extraction_version")
        and item.get("normalization_version") == loaded_document.source_provenance.get("normalization_version")
        for row, item in zip(map_rows, provenance_rows)
    ), "The persisted provenance chain is missing a required identity/version field"
    cited_rows = [item for item in provenance_rows if item.get("source_intervals")]
    assert cited_rows, "No FAISS vector retains verifiable source spans"
    assert all(
        not span.get("page_number") or 1 <= span["page_number"] <= loaded_document.metadata.page_count
        for item in cited_rows for span in item["source_intervals"]
    )

    query_chunk = max(chunks, key=lambda item: len(re.findall(r"[A-Za-z]{4,}", item.content)))
    query = " ".join(re.findall(r"[A-Za-z]{4,}", query_chunk.content)[:20])
    query_vector = embedding_service.embed_query(query)
    hits = store.search(query_vector, min(5, len(vectors)))
    assert hits, "FAISS query returned no evidence"
    first_hit = embedding_service.get_embedding_segment(hits[0][0], store.model_id, store.model_revision)
    assert first_hit and first_hit["source_provenance"].get("source_intervals")

    manifest = {
        "classification": "VERIFIED PASS for isolated extraction, BGE embedding, FAISS mapping, and source-coordinate persistence",
        "document_id": document.id,
        "filename": pdf_path.name,
        "source_pdf_sha256": sha256(pdf_path),
        "page_count": loaded_document.metadata.page_count,
        "pages_with_extracted_text": sum(bool(page.get("location_available")) for page in loaded_document.source_pages),
        "normalized_text_characters": len(loaded_document.text_content or ""),
        "extraction_version": loaded_document.source_provenance.get("extraction_version"),
        "normalization_version": loaded_document.source_provenance.get("offset_map", {}).get("version"),
        "chunks": len(chunks),
        "segments_and_vectors": len(vectors),
        "segments_with_source_spans": len(cited_rows),
        "page_split_intervals_validated": all(
            1 <= span["page_number"] <= loaded_document.metadata.page_count
            and 0 <= span["page_char_start"] <= span["page_char_end"]
            <= len(loaded_document.source_pages[span["page_number"] - 1]["raw_text"] or "")
            for item in cited_rows for span in item["source_intervals"] if span.get("page_number")
        ),
        "model_id": store.model_id,
        "model_revision": store.model_revision,
        "faiss_index_path": str(store.index_path),
        "query": query,
        "retrieved_vector_id": hits[0][0],
        "retrieved_source_intervals": first_hit["source_provenance"]["source_intervals"],
        "database_path": embedding_service.db_path,
    }
    manifest_path = output_dir / "isolated_provenance_e2e_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
