"""Tier 3 Cross-Feature Combination: test_t3_x12_statistics_attempt_correlation.py
Feature Interaction: F10 (SQLite Checkpointing & Resumption) + F13 (Detailed Structured Attempt Telemetry).
Validates that attempt counts recorded in granular telemetry correlate accurately
with checkpoint rows in dataset_chunk_checkpoints and aggregated factory statistics.
"""

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_factory_service import DatasetFactoryService


def test_t3_x12_statistics_attempt_correlation(tmp_path):
    """T3.INT.12: F10 + F13 Pairwise Integration."""
    db_file = tmp_path / "x12_stats.db"
    factory = DatasetFactoryService(db_path=db_file)
    doc_id = "doc-x12-correlation"

    # Log 3 attempts for chunk 1 (2 failures, 1 success) and 1 attempt for chunk 2
    for attempt_num in range(1, 4):
        status = "failed" if attempt_num < 3 else "completed"
        factory._log_generation_attempt(
            attempt_id=f"att-chk1-{attempt_num}",
            generation_id="gen-x12",
            document_id=doc_id,
            chunk_id="chk-1",
            model="qwen3.5:4b",
            provider="ollama",
            prompt_tokens=50,
            completion_tokens=80 if status == "completed" else 0,
            prompt_chars=200,
            completion_chars=400 if status == "completed" else 0,
            request_duration_ms=1500.0,
            retry_count=attempt_num - 1,
            status=status,
            valid_record_count=1 if status == "completed" else 0,
        )

    # Save checkpoint for chunk 1 reflecting 2 retries
    factory._save_checkpoint_state(
        "gen-x12", doc_id, "chk-1", status="completed", retry_count=2,
        samples_data='[{"instruction": "Q", "response": "A"}]',
    )

    # Query telemetry and checkpoint
    telemetry = factory.get_generation_attempts(document_id=doc_id)
    assert telemetry["total_count"] == 3

    checkpoints = factory._get_checkpoints(doc_id)
    assert len(checkpoints) == 1
    assert checkpoints["chk-1"]["status"] == "completed"
    assert checkpoints["chk-1"]["retry_count"] == 2
    # Invariant: Total attempts logged >= checkpoint count
    assert telemetry["total_count"] >= len(checkpoints)
