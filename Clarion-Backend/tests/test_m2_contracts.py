"""
Tests for Milestone 2 contracts and sanitization.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from core.llm.base import sanitize_llm_text
from core.llm.ollama_provider import OllamaProvider
from services.knowledge_map_service import LLMInterface
from services.processing_pipeline import ProcessingPipeline
from utils.config import settings


def test_sanitize_llm_text_reasoning_tags():
    """Verify that <think>...</think> tokens are stripped cleanly."""
    raw = "<think>Evaluating concept candidates...\nDiscarding noisy terms.</think>Machine Learning is a subset of AI."
    sanitized = sanitize_llm_text(raw)
    assert sanitized == "Machine Learning is a subset of AI."


def test_sanitize_llm_text_multiline_case_insensitive():
    """Verify multiline and case-insensitive think tags are handled."""
    raw = """<THINK>
    step 1: analyze input
    step 2: formulate response
    </THINK>
    {
        "status": "ok"
    }"""
    sanitized = sanitize_llm_text(raw)
    assert json.loads(sanitized) == {"status": "ok"}


def test_sanitize_llm_text_none_and_plain():
    """Verify handling of None and plain text."""
    assert sanitize_llm_text(None) == ""
    assert sanitize_llm_text("") == ""
    assert sanitize_llm_text("  Just plain text  ") == "Just plain text"


def test_ollama_timeout_configuration():
    """Verify that ollama_request_timeout_seconds is configured to 180 seconds."""
    assert settings.ollama_request_timeout_seconds == 180


def test_ollama_provider_sanitizes_structured_output():
    """Verify OllamaProvider strips reasoning tags before parsing structured JSON."""
    import asyncio
    provider = OllamaProvider(api_base="http://mock-ollama:11434/v1")
    
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "<think>Planning json output</think>{\"concepts\": [\"Neural Networks\"]}"
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    mock_response.model = "qwen3.5:4b"

    with patch.object(provider, "_client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        result = asyncio.run(provider.generate_structured(
            prompt="Extract concepts",
            output_schema={"type": "object", "properties": {"concepts": {"type": "array"}}}
        ))
        assert result == {"concepts": ["Neural Networks"]}


def test_knowledge_map_llm_interface_extract_json_fragment_with_think_tokens():
    """Verify LLMInterface._extract_json_fragment strips think tokens before parsing."""
    interface = LLMInterface()
    raw_response = "<think>Analyzing graph relations...</think>[{\"from\": \"A\", \"to\": \"B\"}]"
    parsed = interface._extract_json_fragment(raw_response)
    assert parsed == [{"from": "A", "to": "B"}]


def test_processing_pipeline_build_analysis_report_metrics():
    """Verify _build_analysis_report produces the 8-metric overview deck dictionary."""
    pipeline = ProcessingPipeline()
    results = {
        "document_id": "doc-m2-test",
        "started_at": "2026-10-09T08:00:00",
        "completed_at": "2026-10-09T08:00:15",
        "artifacts": {
            "document": {"id": "doc-m2-test", "word_count": 1250, "page_count": 5},
            "chunks": {"count": 18},
            "summary": {"section_count": 4},
        },
    }
    
    mock_km = MagicMock()
    mock_concept1 = MagicMock()
    mock_concept1.name = "Concept 1"
    mock_concept1.chunk_ids = ["c1", "c2"]
    mock_concept1.definition = "Def 1"
    mock_km.concepts = [mock_concept1]
    
    mock_rel1 = MagicMock()
    mock_rel1.relation_type.value = "is_a"
    mock_rel1.from_concept = "Concept 1"
    mock_rel1.to_concept = "Concept 2"
    mock_km.relations = [mock_rel1]

    report = pipeline._build_analysis_report(
        results=results,
        success=True,
        error=None,
        knowledge_map=mock_km,
        graph=None,
        evaluation={},
        hierarchy={},
        summary=None,
    )

    assert "metrics" in report
    metrics = report["metrics"]
    assert metrics["words"] == 1250
    assert metrics["pages"] == 5
    assert metrics["sections"] == 4
    assert metrics["chunks"] == 18
    assert metrics["concepts"] == 1
    assert metrics["candidate_relations"] == 1
    assert metrics["graph_nodes"] == 0
    assert metrics["graph_edges"] == 0
    assert report["total_duration_sec"] == 15.0
    assert report["status"] == "completed"
    assert report["analysis_mode"] == "llm"


def test_upload_list_includes_page_count():
    """Verify /upload/list includes page_count in document metadata response."""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    response = client.get("/upload/list")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    for doc in data["documents"]:
        assert "page_count" in doc
