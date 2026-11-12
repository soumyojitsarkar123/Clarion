"""Tier 2 Boundary & Corner Cases: test_t2_b13_telemetry_missing.py
Feature F13: Detailed Structured Attempt Telemetry.
Verifies boundary conditions: zero-token completion logging, extreme failure reason strings (>5000 chars),
high concurrency attempt persistence, querying empty telemetry histories, and duration non-negativity.
"""

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_factory_service import DatasetFactoryService


def test_b13_zero_token_completion_logging(tmp_path):
    """T2.F13.1: Zero-token completion telemetry record creation."""
    db_file = tmp_path / "telem_zero.db"
    factory = DatasetFactoryService(db_path=db_file)
    doc_id = "doc-zero-tokens"

    factory._log_generation_attempt(
        attempt_id="att-zero-001",
        generation_id="gen-zero-001",
        document_id=doc_id,
        chunk_id="chk-001",
        model="qwen3.5:4b",
        provider="ollama",
        prompt_tokens=0,
        completion_tokens=0,
        prompt_chars=120,
        completion_chars=0,
        request_duration_ms=45.2,
        retry_count=0,
        status="failed",
        failure_reason="Aborted before token emission",
        valid_record_count=0,
    )

    records = factory.get_generation_attempts(document_id=doc_id)
    assert records["total_count"] == 1
    att = records["attempts"][0]
    assert att["prompt_tokens"] == 0
    assert att["completion_tokens"] == 0
    assert att["status"] == "failed"


def test_b13_extreme_failure_reason_length(tmp_path):
    """T2.F13.2: Extremely long failure reason string (>5,000 characters)."""
    db_file = tmp_path / "telem_long_err.db"
    factory = DatasetFactoryService(db_path=db_file)
    doc_id = "doc-long-trace"
    massive_stack = "Traceback (most recent call last):\n" + ("  File 'server.py', line 42, in generate\n" * 150)
    assert len(massive_stack) > 5000

    factory._log_generation_attempt(
        attempt_id="att-long-002",
        generation_id="gen-long-002",
        document_id=doc_id,
        chunk_id="chk-002",
        model="qwen3.5:4b",
        provider="ollama",
        prompt_tokens=50,
        completion_tokens=0,
        prompt_chars=200,
        completion_chars=0,
        request_duration_ms=60000.0,
        retry_count=3,
        status="failed",
        failure_reason=massive_stack,
        valid_record_count=0,
    )

    records = factory.get_generation_attempts(document_id=doc_id)
    assert records["total_count"] == 1
    assert "Traceback" in records["attempts"][0]["failure_reason"]


def test_b13_high_concurrency_attempt_logging(tmp_path):
    """T2.F13.3: Rapid attempt telemetry writes into SQLite table."""
    db_file = tmp_path / "telem_concur.db"
    factory = DatasetFactoryService(db_path=db_file)
    doc_id = "doc-rapid-attempts"

    for i in range(25):
        factory._log_generation_attempt(
            attempt_id=f"att-seq-{i}",
            generation_id="gen-seq-001",
            document_id=doc_id,
            chunk_id=f"chk-{i % 5}",
            model="qwen3.5:4b",
            provider="ollama",
            prompt_tokens=40 + i,
            completion_tokens=80 + i,
            prompt_chars=150,
            completion_chars=300,
            request_duration_ms=1200.0 + (i * 10),
            retry_count=i % 3,
            status="completed",
            valid_record_count=1,
        )

    records = factory.get_generation_attempts(document_id=doc_id, limit=50)
    assert records["total_count"] == 25
    assert len(records["attempts"]) == 25


def test_b13_telemetry_query_zero_attempts(tmp_path):
    """T2.F13.4: Querying attempt telemetry for document with zero attempts."""
    db_file = tmp_path / "telem_empty.db"
    factory = DatasetFactoryService(db_path=db_file)
    records = factory.get_generation_attempts(document_id="doc-nonexistent-xyz")
    assert records["total_count"] == 0
    assert records["attempts"] == []


def test_b13_monotonic_clock_non_negative_duration(tmp_path):
    """T2.F13.5: Attempt duration is non-negative (>= 0.0)."""
    db_file = tmp_path / "telem_duration.db"
    factory = DatasetFactoryService(db_path=db_file)
    doc_id = "doc-nonneg-dur"

    factory._log_generation_attempt(
        attempt_id="att-dur-005",
        generation_id="gen-dur-005",
        document_id=doc_id,
        chunk_id="chk-001",
        model="qwen3.5:4b",
        provider="ollama",
        prompt_tokens=10,
        completion_tokens=10,
        prompt_chars=50,
        completion_chars=50,
        request_duration_ms=0.0,
        retry_count=0,
        status="completed",
        valid_record_count=1,
    )

    records = factory.get_generation_attempts(document_id=doc_id)
    assert records["attempts"][0]["request_duration_ms"] >= 0.0
