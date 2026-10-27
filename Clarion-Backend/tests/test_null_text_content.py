"""
Regression test for missing-text-content error handling in document analysis endpoints.
Verifies that analyzing a document with NULL or empty text_content in DB cleanly
returns HTTP 400 with a descriptive error message rather than crashing or returning HTTP 500.
"""

from datetime import datetime
import pytest
from fastapi.testclient import TestClient

from main import app
from models.document import Document, DocumentMetadata, DocumentStatus
from services.document_service import DocumentService
from services.embedding_service import EmbeddingService
from services.knowledge_map_service import LLMInterface
import routers.analyze as analyze_router_module


@pytest.fixture
def test_setup(isolated_settings, monkeypatch):
    """Test client with isolated settings and bound document service."""
    monkeypatch.setattr(EmbeddingService, "generate_embeddings", lambda self, chunks: [])
    monkeypatch.setattr(EmbeddingService, "embed_query", lambda self, q: [])
    monkeypatch.setattr(LLMInterface, "_is_service_available", lambda self: False)
    
    doc_service = DocumentService()
    monkeypatch.setattr(analyze_router_module, "document_service", doc_service)
    client = TestClient(app)
    return client, doc_service


def test_analyze_document_with_null_text_content_returns_400(test_setup):
    """
    Regression test: A document in DB with NULL text_content must return HTTP 400
    with a clear detail message ("Document text content is empty or NULL")
    and must not crash or trigger an unhandled HTTP 500.
    """
    client, doc_service = test_setup
    doc_id = "doc-test-null-text"
    doc = Document(
        id=doc_id,
        metadata=DocumentMetadata(
            filename="null_text.pdf",
            file_size=512,
            file_type="application/pdf",
            upload_time=datetime.now(),
            page_count=1,
            word_count=0,
        ),
        status=DocumentStatus.UPLOADED,
        text_content=None,
    )
    doc_service._save_to_db(doc)

    response = client.post(f"/analyze/{doc_id}")
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "Document text content is empty or NULL" in detail


def test_analyze_document_with_empty_text_content_returns_400(test_setup):
    """
    A document with an empty string text_content must return HTTP 400.
    """
    client, doc_service = test_setup
    doc_id = "doc-test-empty-text"
    doc = Document(
        id=doc_id,
        metadata=DocumentMetadata(
            filename="empty_text.pdf",
            file_size=512,
            file_type="application/pdf",
            upload_time=datetime.now(),
            page_count=1,
            word_count=0,
        ),
        status=DocumentStatus.UPLOADED,
        text_content="",
    )
    doc_service._save_to_db(doc)

    response = client.post(f"/analyze/{doc_id}")
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "Document text content is empty or NULL" in detail


def test_analyze_document_with_whitespace_text_content_returns_400(test_setup):
    """
    A document with whitespace-only text_content must also return HTTP 400.
    """
    client, doc_service = test_setup
    doc_id = "doc-test-whitespace-text"
    doc = Document(
        id=doc_id,
        metadata=DocumentMetadata(
            filename="spaces.pdf",
            file_size=512,
            file_type="application/pdf",
            upload_time=datetime.now(),
            page_count=1,
            word_count=0,
        ),
        status=DocumentStatus.UPLOADED,
        text_content="   \n\t  \n",
    )
    doc_service._save_to_db(doc)

    response = client.post(f"/analyze/{doc_id}")
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "Document text content is empty or NULL" in detail
