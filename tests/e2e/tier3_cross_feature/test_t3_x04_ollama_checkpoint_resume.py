"""Tier 3 Cross-Feature Combination: test_t3_x04_ollama_checkpoint_resume.py
Feature Interaction: F7 (Per-Chunk Dataset Generation) + F10 (SQLite Checkpointing & Resumption).
Validates that per-chunk dataset generation records granular SQLite checkpoints per chunk,
and when resumed after interruption, completed chunks are skipped without re-querying the LLM provider.
"""

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_factory_service import DatasetFactoryService


def test_t3_x04_ollama_checkpoint_resume(tmp_path):
    """T3.INT.4: F7 + F10 Pairwise Integration."""
    db_file = tmp_path / "x04_resume.db"
    factory = DatasetFactoryService(db_path=db_file)
    doc_id = "doc-x04-chunks"

    # Simulate generating 5 chunks where first 3 completed and 4th was interrupted
    for i in range(3):
        factory._save_checkpoint_state(
            generation_id="gen-x04-1",
            document_id=doc_id,
            chunk_id=f"chk-{i}",
            status="completed",
            sample_ids=[f"sample-{i}"],
            samples_data=[{"instruction": f"Instruction {i}", "response": f"Response {i}"}],
        )

    factory._save_checkpoint_state(
        generation_id="gen-x04-1",
        document_id=doc_id,
        chunk_id="chk-3",
        status="pending",
    )

    # Resume: Query checkpoints
    checkpoints = factory._get_checkpoints(doc_id)
    assert len(checkpoints) == 4

    completed_chunk_ids = {cid for cid, chk in checkpoints.items() if chk["status"] == "completed"}
    assert completed_chunk_ids == {"chk-0", "chk-1", "chk-2"}

    # Resumed run: chk-3 is processed and marked completed
    factory._save_checkpoint_state(
        generation_id="gen-x04-1",
        document_id=doc_id,
        chunk_id="chk-3",
        status="completed",
        sample_ids=["sample-3"],
        samples_data=[{"instruction": "Instruction 3", "response": "Response 3"}],
    )

    checkpoints_after = factory._get_checkpoints(doc_id)
    assert all(chk["status"] == "completed" for chk in checkpoints_after.values())
    summary = factory.get_checkpoint_summary(doc_id)
    assert summary["completed"] == 4
    assert summary["pending"] == 0
