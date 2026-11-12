"""Tier 3 Cross-Feature Combination: test_t3_x05_export_gate_telemetry.py
Feature Interaction: F12 (Research Integrity & Gold Export Gate) + F13 (Detailed Structured Attempt Telemetry).
Validates that when gold export is refused under require_llm_for_export=True due to incomplete generation,
granular attempt telemetry accurately records the failure reason and execution metadata.
"""

import sys
from pathlib import Path
import pytest
from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_factory_service import DatasetFactoryService
from models.document import Document, DocumentMetadata


def test_t3_x05_export_gate_telemetry(tmp_path):
    """T3.INT.5: F12 + F13 Pairwise Integration."""
    db_file = tmp_path / "x05_telemetry.db"
    factory = DatasetFactoryService(db_path=db_file)
    factory.require_llm_for_export = True
    doc_id = "doc-x05-gate"

    doc = Document(id=doc_id, metadata=DocumentMetadata(filename="gate_test.pdf", file_type="pdf", file_size=1000))
    factory.document_service.save_document(doc)

    # Chunk 1 completed, chunk 2 failed after 3 retries
    factory._save_checkpoint_state("gen-x05", doc_id, "chk-1", status="completed", samples_data="[]")
    factory._save_checkpoint_state(
        "gen-x05", doc_id, "chk-2", status="failed", retry_count=3,
        error_message="ReadTimeout: Connection closed by server localhost:11434",
    )

    # Log attempt telemetry for the failed chunk
    factory._log_generation_attempt(
        attempt_id="att-x05-fail",
        generation_id="gen-x05",
        document_id=doc_id,
        chunk_id="chk-2",
        model="qwen3.5:4b",
        provider="ollama",
        prompt_tokens=150,
        completion_tokens=0,
        prompt_chars=500,
        completion_chars=0,
        request_duration_ms=60000.0,
        retry_count=3,
        status="failed",
        failure_reason="ReadTimeout: Connection closed by server localhost:11434",
        valid_record_count=0,
    )

    # F12: Attempting gold export must raise HTTP 409 Conflict
    with pytest.raises(HTTPException) as exc_info:
        factory.export_document_dataset_strict(doc_id)
    assert exc_info.value.status_code == 409
    assert "incomplete" in exc_info.value.detail.lower()

    # F13: Telemetry must be queryable and record the failure reason
    attempts = factory.get_generation_attempts(document_id=doc_id, chunk_id="chk-2")
    assert attempts["total_count"] == 1
    assert attempts["attempts"][0]["status"] == "failed"
    assert "ReadTimeout" in attempts["attempts"][0]["failure_reason"]
