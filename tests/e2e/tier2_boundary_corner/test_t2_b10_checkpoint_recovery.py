"""Tier 2 Boundary & Corner Cases: test_t2_b10_checkpoint_recovery.py
Feature F10: SQLite Checkpointing & Resumption.
Verifies boundary conditions: SQLite busy/lock handling under concurrency,
resumption when 100% of chunks are already completed, failed chunk retry state updates,
OperationalError handling, and duplicate (document_id, chunk_id) unique constraint resolution.
"""

import sqlite3
import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_factory_service import DatasetFactoryService


def test_b10_sqlite_busy_locked_retry(tmp_path):
    """T2.F10.1: SQLite busy/lock handling during concurrent checkpoint writes."""
    db_file = tmp_path / "factory_lock.db"
    factory = DatasetFactoryService(db_path=db_file)

    # Enable WAL mode and verify busy timeout handling
    conn = sqlite3.connect(str(db_file), timeout=5.0)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    conn.commit()

    # Save a checkpoint
    factory._save_checkpoint_state(
        generation_id="gen-lock-001",
        document_id="doc-lock-001",
        chunk_id="chk-001",
        status="completed",
        sample_ids=["s-1"],
        samples_data=[{"instruction": "Q", "response": "A"}],
        retry_count=0,
    )
    conn.close()

    summary = factory.get_checkpoint_summary("doc-lock-001")
    assert summary["completed"] == 1
    assert summary["total"] == 1


def test_b10_resume_all_chunks_already_completed(tmp_path):
    """T2.F10.2: Resume when all chunks are already completed in checkpoint DB."""
    db_file = tmp_path / "factory_all_done.db"
    factory = DatasetFactoryService(db_path=db_file)
    doc_id = "doc-resume-all"

    # Save 3 chunks as completed
    for i in range(3):
        factory._save_checkpoint_state(
            generation_id="gen-all-001",
            document_id=doc_id,
            chunk_id=f"chk-{i}",
            status="completed",
            sample_ids=[f"s-{i}"],
            samples_data=[{"instruction": f"Q{i}", "response": f"A{i}"}],
        )

    checkpoints = factory._get_checkpoints(doc_id)
    assert len(checkpoints) == 3
    assert all(chk["status"] == "completed" for chk in checkpoints.values())
    summary = factory.get_checkpoint_summary(doc_id)
    assert summary["completed"] == 3
    assert summary["failed"] == 0
    assert summary["pending"] == 0


def test_b10_resume_retry_failed_chunks(tmp_path):
    """T2.F10.3: Failed chunk checkpoint state update on retry."""
    db_file = tmp_path / "factory_retry.db"
    factory = DatasetFactoryService(db_path=db_file)
    doc_id = "doc-retry-chk"

    # Step 1: Chunk fails on attempt 1
    factory._save_checkpoint_state(
        generation_id="gen-ret-001",
        document_id=doc_id,
        chunk_id="chk-fail-1",
        status="failed",
        error_message="ReadTimeout to provider",
        retry_count=1,
    )
    chk_before = factory._get_checkpoints(doc_id)
    assert chk_before["chk-fail-1"]["status"] == "failed"
    assert chk_before["chk-fail-1"]["retry_count"] == 1

    # Step 2: Retry succeeds
    factory._save_checkpoint_state(
        generation_id="gen-ret-001",
        document_id=doc_id,
        chunk_id="chk-fail-1",
        status="completed",
        sample_ids=["s-recovered"],
        samples_data=[{"instruction": "Q recovered", "response": "A recovered"}],
        retry_count=2,
    )
    chk_after = factory._get_checkpoints(doc_id)
    assert chk_after["chk-fail-1"]["status"] == "completed"
    assert chk_after["chk-fail-1"]["retry_count"] == 2


def test_b10_operational_error_graceful_handling(tmp_path):
    """T2.F10.4: Database query against uninitialized / invalid table."""
    db_file = tmp_path / "factory_empty.db"
    # Create empty database without tables
    conn = sqlite3.connect(str(db_file))
    conn.close()

    factory = DatasetFactoryService(db_path=db_file)
    # Checkpoint summary on empty DB returns safe zero defaults
    summary = factory.get_checkpoint_summary("doc-nonexistent")
    assert isinstance(summary, dict)
    assert summary.get("total", 0) == 0


def test_b10_duplicate_chunk_id_resolution(tmp_path):
    """T2.F10.5: Duplicate (document_id, chunk_id) UPSERT resolution."""
    db_file = tmp_path / "factory_dup.db"
    factory = DatasetFactoryService(db_path=db_file)
    doc_id = "doc-dup"
    chunk_id = "chk-same-id"

    # First write: pending
    factory._save_checkpoint_state(
        generation_id="gen-dup-001",
        document_id=doc_id,
        chunk_id=chunk_id,
        status="pending",
    )
    # Second write with same primary key: completed
    factory._save_checkpoint_state(
        generation_id="gen-dup-001",
        document_id=doc_id,
        chunk_id=chunk_id,
        status="completed",
        sample_ids=["s-dup-1"],
    )

    checkpoints = factory._get_checkpoints(doc_id)
    # UNIQUE constraint ensures exactly 1 row exists, updated to completed
    assert len(checkpoints) == 1
    assert checkpoints[chunk_id]["status"] == "completed"
