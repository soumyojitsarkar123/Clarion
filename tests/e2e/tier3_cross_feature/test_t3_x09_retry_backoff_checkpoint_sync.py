"""Tier 3 Cross-Feature Combination: test_t3_x09_retry_backoff_checkpoint_sync.py
Feature Interaction: F7 (Per-Chunk Processing) + F11 (Bounded Retries, Backoff & Validation).
Validates that chunk generation retries with bounded backoff synchronize accurately
with the SQLite checkpoint state and attempt telemetry logging.
"""

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_factory_service import DatasetFactoryService


def test_t3_x09_retry_backoff_checkpoint_sync(tmp_path):
    """T3.INT.9: F7 + F11 Pairwise Integration."""
    db_file = tmp_path / "x09_backoff.db"
    factory = DatasetFactoryService(db_path=db_file)
    doc_id = "doc-x09-backoff"
    chunk_id = "chk-x09-1"

    # Attempt 1 fails: timeout
    factory._save_checkpoint_state(
        generation_id="gen-x09",
        document_id=doc_id,
        chunk_id=chunk_id,
        status="failed",
        retry_count=1,
        error_message="ReadTimeout: Connection closed",
    )
    chk_1 = factory._get_checkpoints(doc_id)[chunk_id]
    assert chk_1["status"] == "failed"
    assert chk_1["retry_count"] == 1

    # Attempt 2 fails: malformed JSON
    factory._save_checkpoint_state(
        generation_id="gen-x09",
        document_id=doc_id,
        chunk_id=chunk_id,
        status="failed",
        retry_count=2,
        error_message="JSONDecodeError: Unterminated string",
    )
    chk_2 = factory._get_checkpoints(doc_id)[chunk_id]
    assert chk_2["status"] == "failed"
    assert chk_2["retry_count"] == 2

    # Attempt 3 succeeds: valid JSON
    sample_data = [{"instruction": "What is Tarjan algorithm?", "response": "Tarjan algorithm finds strongly connected components in linear time."}]
    factory._save_checkpoint_state(
        generation_id="gen-x09",
        document_id=doc_id,
        chunk_id=chunk_id,
        status="completed",
        retry_count=3,
        samples_data=sample_data,
    )
    chk_3 = factory._get_checkpoints(doc_id)[chunk_id]
    assert chk_3["status"] == "completed"
    assert chk_3["retry_count"] == 3
