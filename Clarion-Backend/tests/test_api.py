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


def test_upload_rejects_oversized_file(client, monkeypatch):
    from utils.config import settings

    monkeypatch.setattr(settings, "max_file_size_mb", 0)
    response = upload_doc(client, "notes.pdf", b"%PDF-1.4 fake content")
    assert response.status_code in {400, 500}


def test_upload_docx_and_analyze_sync(client, stats_docx_bytes):
    response = upload_doc(client, "stats_sample.docx", stats_docx_bytes)
    assert response.status_code == 200, response.text
    document_id = response.json()["document_id"]

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

    summary = client.get(f"/summary/{document_id}")
    assert summary.status_code == 200
    assert summary.json()["sections"]

    graph = client.get(f"/graph/{document_id}")
    assert graph.status_code == 200
    graph_payload = graph.json()
    assert graph_payload.get("elements") is not None or graph_payload.get("nodes") is not None

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
    assert query.json()["query"] == "What is the null hypothesis?"


def test_unknown_document_returns_404(client):
    assert client.get("/status/does-not-exist").status_code == 404
    assert client.get("/graph/does-not-exist").status_code == 404
    assert client.get("/knowledge-map/does-not-exist").status_code == 404


def test_logs_endpoint(client):
    response = client.get("/logs/recent?lines=10")
    assert response.status_code == 200
    assert "lines" in response.json()
