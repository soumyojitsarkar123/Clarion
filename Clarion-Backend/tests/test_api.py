"""API smoke tests using FastAPI TestClient."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from main import app
from services.document_service import DocumentService
from services.embedding_service import EmbeddingService
from services.knowledge_map_service import LLMInterface
from services.processing_pipeline import ProcessingPipeline


@pytest.fixture
def client(isolated_settings, monkeypatch):
    def fake_embeddings(self, chunks):
        self.using_fallback_embeddings = True
        return [self._fallback_embed_text(chunk.content) for chunk in chunks]

    async def fake_embeddings_async(self, chunks, batch_callback=None):
        self.using_fallback_embeddings = True
        return [self._fallback_embed_text(chunk.content) for chunk in chunks]

    monkeypatch.setattr(EmbeddingService, "generate_embeddings", fake_embeddings)
    monkeypatch.setattr(EmbeddingService, "generate_embeddings_async", fake_embeddings_async)
    monkeypatch.setattr(EmbeddingService, "embed_query", lambda self, q: self._fallback_embed_text(q))
    monkeypatch.setattr(LLMInterface, "_is_service_available", lambda self: False)
    return TestClient(app)


def upload_doc(client, filename, content):
    return client.post(
        "/upload",
        files={"file": (filename, content, "application/octet-stream")},
    )


def test_root_and_health(client):
    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["status"] == "running"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"


def test_upload_rejects_unsupported_type(client):
    response = upload_doc(client, "notes.txt", b"plain text notes here")
    assert response.status_code == 400


def test_upload_rejects_empty_file(client):
    """Verify that uploading 0-byte supported documents returns HTTP 400."""
    res_pdf = upload_doc(client, "empty.pdf", b"")
    assert res_pdf.status_code == 400
    assert "empty" in res_pdf.json()["detail"].lower()

    res_docx = upload_doc(client, "empty.docx", b"")
    assert res_docx.status_code == 400
    assert "empty" in res_docx.json()["detail"].lower()


def test_upload_rejects_oversized_file(client, monkeypatch):
    from utils.config import settings

    monkeypatch.setattr(settings, "max_file_size_mb", 0)
    response = upload_doc(client, "notes.pdf", b"%PDF-1.4 fake content")
    assert response.status_code == 400


def test_oversized_upload_reads_no_more_than_limit_plus_one(monkeypatch):
    from fastapi import HTTPException
    from routers.upload import upload_document
    from utils.config import settings

    class TrackingUpload:
        filename = "oversized.pdf"

        def __init__(self):
            self.read_sizes = []
            self.bytes_read = 0

        async def read(self, size=-1):
            self.read_sizes.append(size)
            assert size >= 0
            remaining = 1024 - self.bytes_read
            count = min(size, remaining)
            self.bytes_read += count
            return b"x" * count

    monkeypatch.setattr(settings, "max_file_size_mb", 0)
    upload = TrackingUpload()
    with pytest.raises(HTTPException) as error:
        asyncio.run(upload_document(upload))

    assert error.value.status_code == 400
    assert upload.bytes_read == 1
    assert upload.read_sizes == [1]


def test_extraction_rejects_content_with_mismatched_file_signature():
    from utils.file_handler import extract_text

    with pytest.raises(ValueError, match="valid PDF signature"):
        extract_text(b"PK\x03\x04not a PDF", "spoofed.pdf")


def test_extraction_rejects_docx_archive_path_traversal():
    import io
    import zipfile

    from utils.file_handler import extract_text

    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", "<document/>")
        archive.writestr("../outside.txt", "unsafe")

    with pytest.raises(ValueError, match="unsafe path"):
        extract_text(archive_bytes.getvalue(), "traversal.docx")


def test_upload_docx_and_analyze_sync(client, stats_docx_bytes):
    response = upload_doc(client, "stats_sample.docx", stats_docx_bytes)
    assert response.status_code == 200, response.text
    document_id = response.json()["document_id"]

    source = client.get(f"/upload/{document_id}/source")
    assert source.status_code == 200
    assert source.content == stats_docx_bytes
    metrics = client.get(f"/upload/{document_id}/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["counts"]["physical_pages"] is None
    assert metrics.json()["counts"]["raw_words"] == metrics.json()["counts"]["normalized_words"]

    analyze = client.post(
        f"/analyze/{document_id}",
        json={"wait_for_completion": True, "timeout_seconds": 300},
    )
    assert analyze.status_code == 200, analyze.text
    payload = analyze.json()
    assert payload["status"] == "completed"

    status = client.get(f"/status/{document_id}")
    assert status.status_code == 200
    assert status.json()["document_status"] == "analyzed"

    knowledge_map = client.get(f"/knowledge-map/{document_id}")
    assert knowledge_map.status_code == 200
    assert knowledge_map.json()["concepts"]
    assert knowledge_map.json()["extraction_method"] in {"heuristic", "llm", "unknown"}
    assert all(concept["source_evidence"] for concept in knowledge_map.json()["concepts"] if concept["chunk_ids"])
    assert all(relation["validation_status"] == "candidate_unverified" for relation in knowledge_map.json()["relations"])

    summary = client.get(f"/summary/{document_id}")
    assert summary.status_code == 200
    assert summary.json()["sections"]

    graph = client.get(f"/graph/{document_id}")
    assert graph.status_code == 200
    graph_payload = graph.json()
    assert graph_payload.get("elements") is not None or graph_payload.get("nodes") is not None
    graph_edges = graph_payload.get("elements", {}).get("edges", [])
    assert all(edge["data"].get("is_candidate") is True for edge in graph_edges if edge["data"].get("relation_type") != "hierarchy")

    report = client.get(f"/system/analysis-report/{document_id}")
    assert report.status_code == 200
    assert report.json()["success"] is True

    dataset = client.get("/dataset/relations")
    assert dataset.status_code == 200
    assert dataset.json()["count"] >= 1

    dataset_stats = client.get("/system/dataset-stats")
    assert dataset_stats.status_code == 200

    pipeline_status = client.get(f"/system/pipeline-status?document_id={document_id}")
    assert pipeline_status.status_code == 200
    assert pipeline_status.json()["status"] in ["completed", "degraded"]


def test_query_after_analysis(client, stats_docx_bytes):
    response = upload_doc(client, "stats_sample.docx", stats_docx_bytes)
    document_id = response.json()["document_id"]
    client.post(
        f"/analyze/{document_id}",
        json={"wait_for_completion": True, "timeout_seconds": 300},
    )
    query = client.post(
        f"/query/{document_id}",
        json={"query": "What is the null hypothesis?"},
    )
    assert query.status_code == 200
    payload = query.json()
    assert payload["query"] == "What is the null hypothesis?"
    assert "fallback_used" in payload


def test_query_fallback_used_flag(client, monkeypatch, stats_docx_bytes):
    from services.retrieval_service import RetrievalResult, RetrievalService
    monkeypatch.setattr(
        RetrievalService,
        "retrieve_with_context",
        lambda self, doc_id, query, top_k: (
            [RetrievalResult(chunk_id="c1", document_id=doc_id, content="sample context", score=0.9, section_title="Sec", position_index=0)],
            "sample context"
        )
    )
    upload_res = upload_doc(client, "sample_stats.docx", stats_docx_bytes)
    doc_id = upload_res.json()["document_id"]
    res = client.post(f"/query/{doc_id}", json={"query": "test query"})
    assert res.status_code == 200
    data = res.json()
    assert data["fallback_used"] is True
    assert "omitted_evidence" in data
    assert data["token_metrics"]["context_pack_exact"] is False


def test_unknown_document_returns_404(client):
    assert client.get("/status/does-not-exist").status_code == 404
    assert client.get("/graph/does-not-exist").status_code == 404
    assert client.get("/knowledge-map/does-not-exist").status_code == 404


def test_logs_endpoint(client):
    response = client.get("/logs/recent?lines=10")
    assert response.status_code == 200
    assert "lines" in response.json()


def test_unclosed_think_tag_sanitization():
    from core.llm.base import sanitize_llm_text

    raw = "Beginning of response <think>Reasoning trace that got cut off mid-stream"
    assert sanitize_llm_text(raw) == "Beginning of response"

    complete = "Intro <think>thoughts</think> Answer"
    assert sanitize_llm_text(complete) == "Intro  Answer"


def test_null_byte_sanitization(monkeypatch):
    from utils import file_handler

    monkeypatch.setattr(
        file_handler,
        "extract_text_from_docx",
        lambda content: ("Clean\x00 text\x00 with nulls", {"metadata": True})
    )
    monkeypatch.setattr(file_handler, "validate_file_signature", lambda *_args: None)
    text, _ = file_handler.extract_text(b"fake docx content", "sample.docx")
    assert "\x00" not in text
    assert text == "Clean text with nulls"


def test_upload_rejects_fake_docx_zip(client):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("foo.txt", "This is not a docx")
    response = upload_doc(client, "fake.docx", buf.getvalue())
    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()


def test_upload_rejects_malformed_xml_docx(client):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<not xml>")
    response = upload_doc(client, "bad_xml.docx", buf.getvalue())
    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()

