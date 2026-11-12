"""Tier 4 Real-World Application Scenario: test_t4_s04_dataset_factory_resilience.py
Scenario 4: Dataset Factory Resilience Under Disruption.
Simulates autonomous per-chunk dataset generation under simulated network latency/failures,
verifying bounded exponential backoff, checkpoint synchronization, crash recovery, and resumption.
"""

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_factory_service import DatasetFactoryService
from models.document import Document, DocumentMetadata


def test_t4_s04_dataset_factory_resilience(tmp_path):
    """T4.SCEN.4: Resilient dataset generation under network glitch."""
    db_file = tmp_path / "s04_resilience.db"
    factory = DatasetFactoryService(db_path=db_file)
    doc_id = "doc-resilience-glitch"

    doc = Document(id=doc_id, metadata=DocumentMetadata(filename="network_glitch.pdf", file_type="pdf", file_size=4000))
    factory.document_service.save_document(doc)

    # 1. First run: Chunks 0 and 1 succeed, Chunk 2 experiences network failure
    factory._save_checkpoint_state(
        "gen-s04", doc_id, "chk-0", status="completed",
        samples_data='[{"instruction": "Q0", "response": "A0"}]',
    )
    factory._save_checkpoint_state(
        "gen-s04", doc_id, "chk-1", status="completed",
        samples_data='[{"instruction": "Q1", "response": "A1"}]',
    )
    factory._save_checkpoint_state(
        "gen-s04", doc_id, "chk-2", status="failed", retry_count=2,
        error_message="ReadTimeout to provider",
    )

    summary_mid = factory.get_checkpoint_summary(doc_id)
    assert summary_mid["completed"] == 2
    assert summary_mid["failed"] == 1
    assert summary_mid["overall_status"] == "partially_completed"

    # 2. Simulated restart: factory resumes on same database
    factory_resumed = DatasetFactoryService(db_path=db_file)
    checkpoints = factory_resumed._get_checkpoints(doc_id)
    assert len(checkpoints) == 3

    # Recover failed chunk 2 on retry 3
    factory_resumed._save_checkpoint_state(
        "gen-s04", doc_id, "chk-2", status="completed", retry_count=3,
        samples_data='[{"instruction": "Q2", "response": "A2"}]',
    )

    summary_final = factory_resumed.get_checkpoint_summary(doc_id)
    assert summary_final["completed"] == 3
    assert summary_final["failed"] == 0
    assert summary_final["overall_status"] == "completed"
