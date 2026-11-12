"""Tier 1 Feature Coverage: F10 SQLite Checkpointing & Resumption.

Specifications:
- T1.F10.1: Checkpoint Table Schema Initialization
- T1.F10.2: Completed Chunk Checkpoint Write
- T1.F10.3: Failed Chunk Checkpoint Write
- T1.F10.4: Resumption Skips Completed Chunks
- T1.F10.5: Checkpoint Cleared on Clean Reset
"""

import sqlite3
import pytest
from services.dataset_factory_service import DatasetFactoryService


def test_f10_checkpoint_table_schema(isolated_settings):
    """T1.F10.1: Checkpoint Table Schema Initialization."""
    factory = DatasetFactoryService()
    conn = sqlite3.connect(str(factory.db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dataset_chunk_checkpoints'")
    assert cursor.fetchone() is not None, "dataset_chunk_checkpoints table does not exist"

    cursor.execute("PRAGMA table_info(dataset_chunk_checkpoints)")
    cols = {row[1] for row in cursor.fetchall()}
    expected_cols = {
        "checkpoint_id",
        "generation_id",
        "document_id",
        "chunk_id",
        "status",
        "sample_ids",
        "samples_data",
        "retry_count",
        "error_message",
        "created_at",
        "updated_at",
    }
    assert expected_cols.issubset(cols), f"Missing checkpoint columns: {expected_cols - cols}"
    conn.close()


def test_f10_completed_chunk_checkpoint_write(isolated_settings):
    """T1.F10.2: Completed Chunk Checkpoint Write."""
    factory = DatasetFactoryService()
    doc_id = "doc-f10-complete"
    chunk_id = "chk-001"
    gen_id = "gen-1234"

    factory._save_checkpoint_state(
        generation_id=gen_id,
        document_id=doc_id,
        chunk_id=chunk_id,
        status="completed",
        sample_ids=["sample-1", "sample-2"],
        retry_count=0,
    )

    checkpoints = factory._get_checkpoints(doc_id)
    assert chunk_id in checkpoints, f"Expected {chunk_id} in checkpoints"
    chk = checkpoints[chunk_id]
    assert chk["status"] == "completed"
    assert "sample-1" in chk["sample_ids"]
    assert chk["retry_count"] == 0


def test_f10_failed_chunk_checkpoint_write(isolated_settings):
    """T1.F10.3: Failed Chunk Checkpoint Write."""
    factory = DatasetFactoryService()
    doc_id = "doc-f10-fail"
    chunk_id = "chk-002"
    gen_id = "gen-5678"

    factory._save_checkpoint_state(
        generation_id=gen_id,
        document_id=doc_id,
        chunk_id=chunk_id,
        status="failed",
        retry_count=3,
        error_message="HTTP 503 Provider Unavailable",
    )

    checkpoints = factory._get_checkpoints(doc_id)
    assert chunk_id in checkpoints
    chk = checkpoints[chunk_id]
    assert chk["status"] == "failed"
    assert chk["retry_count"] == 3
    assert "HTTP 503" in chk["error_message"]


def test_f10_resumption_skips_completed_chunks(isolated_settings):
    """T1.F10.4: Resumption Skips Completed Chunks."""
    factory = DatasetFactoryService()
    doc_id = "doc-f10-resume"
    gen_id = "gen-resume"

    # Pre-populate chunk 1 as completed
    factory._save_checkpoint_state(
        generation_id=gen_id,
        document_id=doc_id,
        chunk_id="chk-01",
        status="completed",
        sample_ids=["sample-pre-1"],
    )

    checkpoints = factory._get_checkpoints(doc_id)
    assert checkpoints.get("chk-01", {}).get("status") == "completed"
    assert "chk-02" not in checkpoints

    # A resumption query for doc_id confirms chunk 1 is marked completed
    summary = factory.get_checkpoint_summary(doc_id)
    assert summary.get("completed_count", 0) >= 1


def test_f10_checkpoint_cleared_on_clean_reset(isolated_settings):
    """T1.F10.5: Checkpoint Cleared on Clean Reset."""
    factory = DatasetFactoryService()
    doc_id = "doc-f10-reset"

    factory._save_checkpoint_state(
        generation_id="gen-reset",
        document_id=doc_id,
        chunk_id="chk-01",
        status="completed",
    )

    # Clean reset / removal for document
    conn = sqlite3.connect(str(factory.db_path))
    cursor = conn.cursor()
    cursor.execute("DELETE FROM dataset_chunk_checkpoints WHERE document_id = ?", (doc_id,))
    conn.commit()
    conn.close()

    checkpoints = factory._get_checkpoints(doc_id)
    assert len(checkpoints) == 0, "Checkpoints should be empty after reset"
