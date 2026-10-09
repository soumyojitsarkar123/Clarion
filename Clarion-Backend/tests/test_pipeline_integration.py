"""Integration test: upload -> chunking -> embedding -> mapping -> graph
-> evaluation -> hierarchy -> dataset capture, with all model paths
forced to deterministic fallbacks.
"""

import asyncio

import pytest

from services.document_service import DocumentService
from services.embedding_service import EmbeddingService
from services.knowledge_map_service import LLMInterface
from services.processing_pipeline import ProcessingPipeline
from utils.graph_store import load_graph_json


@pytest.fixture
def offline_pipeline(monkeypatch, isolated_settings):
    """Pipeline with live model paths replaced by deterministic fallbacks."""

    def fake_embeddings(self, chunks):
        self.using_fallback_embeddings = True
        return [self._fallback_embed_text(chunk.content) for chunk in chunks]

    def fake_embed_query(self, query_text):
        self.using_fallback_embeddings = True
        return self._fallback_embed_text(query_text)

    monkeypatch.setattr(EmbeddingService, "generate_embeddings", fake_embeddings)
    monkeypatch.setattr(EmbeddingService, "embed_query", fake_embed_query)
    monkeypatch.setattr(LLMInterface, "_is_service_available", lambda self: False)
    return ProcessingPipeline()


def test_full_pipeline_end_to_end(isolated_settings, stats_docx_bytes, offline_pipeline):
    document_service = DocumentService()
    document = document_service.upload_document(stats_docx_bytes, "stats_sample.docx")
    document_id = document.id
    assert str(document.status) == "uploaded"
    assert document.text_content

    results = asyncio.run(offline_pipeline.execute(document_id=document_id))

    assert results["success"] is True
    expected_stages = [
        "ingestion",
        "chunking",
        "embedding",
        "mapping",
        "graph_building",
        "evaluation",
        "hierarchy",
    ]
    for stage in expected_stages:
        assert stage in results["stages_completed"], f"stage {stage} not completed"

    artifacts = results["artifacts"]
    assert artifacts["chunks"]["count"] > 0
    assert artifacts["knowledge_map"]["concept_count"] > 0
    assert artifacts["knowledge_map"]["relation_count"] > 0
    assert artifacts["graph"]["node_count"] > 0
    assert artifacts["graph"]["edge_count"] > 0

    graph = load_graph_json(document_id)
    assert graph is not None
    assert graph.number_of_nodes() == artifacts["graph"]["node_count"]

    audit = results["audit"]
    assert audit["document_id"] == document_id
    for stage in audit["stages"]:
        assert stage["status"] in {"completed", "skipped"}

    updated = document_service.get_document(document_id)
    assert str(updated.status) == "analyzed"


def test_pipeline_reports_failure_for_missing_document(isolated_settings, offline_pipeline):
    with pytest.raises(Exception):
        asyncio.run(offline_pipeline.execute(document_id="missing-doc"))
