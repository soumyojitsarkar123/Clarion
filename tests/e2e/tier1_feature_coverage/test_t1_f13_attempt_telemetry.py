"""Tier 1 Feature Coverage: F13 Detailed Structured Attempt Telemetry.

Specifications:
- T1.F13.1: Attempt Telemetry Record Creation
- T1.F13.2: Token Usage Logging in Attempt Telemetry
- T1.F13.3: Execution Duration Recording
- T1.F13.4: Failure Reason Logging on Error
- T1.F13.5: Aggregated Factory Statistics Endpoint
"""

import uuid
import pytest
from services.dataset_factory_service import DatasetFactoryService


def test_f13_attempt_record_creation(isolated_settings):
    """T1.F13.1: Attempt Telemetry Record Creation."""
    factory = DatasetFactoryService()
    att_id = str(uuid.uuid4())
    doc_id = "doc-f13-att"
    gen_id = "gen-f13-att"
    chk_id = "chk-f13-01"

    factory._log_generation_attempt(
        attempt_id=att_id,
        generation_id=gen_id,
        document_id=doc_id,
        chunk_id=chk_id,
        model="qwen3.5:4b",
        provider="ollama",
        prompt_tokens=40,
        completion_tokens=80,
        prompt_chars=160,
        completion_chars=320,
        request_duration_ms=1250.0,
        retry_count=0,
        status="completed",
        failure_reason=None,
        valid_record_count=1,
    )

    records = factory.get_generation_attempts(document_id=doc_id)
    assert records["total"] == 1
    att = records["attempts"][0]
    assert att["attempt_id"] == att_id
    assert att["document_id"] == doc_id
    assert att["chunk_id"] == chk_id
    assert att["status"] == "completed"


def test_f13_token_usage_logging(isolated_settings):
    """T1.F13.2: Token Usage Logging in Attempt Telemetry."""
    factory = DatasetFactoryService()
    att_id = str(uuid.uuid4())
    doc_id = "doc-f13-tokens"

    factory._log_generation_attempt(
        attempt_id=att_id,
        generation_id="gen-tokens",
        document_id=doc_id,
        chunk_id="chk-tokens",
        model="qwen3.5:4b",
        provider="ollama",
        prompt_tokens=55,
        completion_tokens=110,
        prompt_chars=220,
        completion_chars=440,
        request_duration_ms=800.0,
        retry_count=0,
        status="completed",
    )

    records = factory.get_generation_attempts(document_id=doc_id)
    assert records["total"] == 1
    att = records["attempts"][0]
    assert att["prompt_tokens"] == 55
    assert att["completion_tokens"] == 110


def test_f13_duration_recording(isolated_settings):
    """T1.F13.3: Execution Duration Recording."""
    factory = DatasetFactoryService()
    att_id = str(uuid.uuid4())
    doc_id = "doc-f13-dur"

    factory._log_generation_attempt(
        attempt_id=att_id,
        generation_id="gen-dur",
        document_id=doc_id,
        chunk_id="chk-dur",
        model="qwen3.5:4b",
        provider="ollama",
        prompt_tokens=10,
        completion_tokens=10,
        prompt_chars=40,
        completion_chars=40,
        request_duration_ms=1450.5,
        retry_count=0,
        status="completed",
    )

    records = factory.get_generation_attempts(document_id=doc_id)
    att = records["attempts"][0]
    assert att["request_duration_ms"] == pytest.approx(1450.5, rel=0.01)


def test_f13_failure_reason_logging(isolated_settings):
    """T1.F13.4: Failure Reason Logging on Error."""
    factory = DatasetFactoryService()
    att_id = str(uuid.uuid4())
    doc_id = "doc-f13-err"

    factory._log_generation_attempt(
        attempt_id=att_id,
        generation_id="gen-err",
        document_id=doc_id,
        chunk_id="chk-err",
        model="qwen3.5:4b",
        provider="ollama",
        prompt_tokens=20,
        completion_tokens=0,
        prompt_chars=80,
        completion_chars=0,
        request_duration_ms=500.0,
        retry_count=2,
        status="failed",
        failure_reason="ReadTimeout: Request timed out after 60.0s",
        valid_record_count=0,
    )

    records = factory.get_generation_attempts(document_id=doc_id)
    att = records["attempts"][0]
    assert att["status"] == "failed"
    assert "ReadTimeout" in att["failure_reason"]
    assert att["retry_count"] == 2


def test_f13_aggregated_factory_statistics(e2e_client):
    """T1.F13.5: Aggregated Factory Statistics Endpoint."""
    response = e2e_client.get("/dataset-factory/statistics")
    assert response.status_code == 200
    payload = response.json()

    assert "total_processed" in payload
    assert "total_samples" in payload
    assert "total_exported" in payload
    assert "quality_filtered" in payload
    assert "dedup_filtered" in payload
    assert "last_run" in payload
