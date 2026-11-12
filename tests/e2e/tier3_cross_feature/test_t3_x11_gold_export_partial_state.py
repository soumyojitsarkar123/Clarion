"""Tier 3 Cross-Feature Combination: test_t3_x11_gold_export_partial_state.py
Feature Interaction: F7 (Per-Chunk Processing) + F12 (Research Integrity & Gold Export Gate).
Validates that when multi-chunk processing leaves 1 chunk permanently failed under require_llm_for_export=True,
the batch state is partially_completed and the gold export endpoint strictly refuses export with HTTP 409 Conflict.
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


def test_t3_x11_gold_export_partial_state(tmp_path):
    """T3.INT.11: F7 + F12 Pairwise Integration."""
    db_file = tmp_path / "x11_partial.db"
    factory = DatasetFactoryService(db_path=db_file)
    factory.require_llm_for_export = True
    doc_id = "doc-x11-partial-state"

    doc = Document(id=doc_id, metadata=DocumentMetadata(filename="partial.pdf", file_type="pdf", file_size=1200))
    factory.document_service.save_document(doc)

    # 4 chunks completed, 1 chunk failed
    for i in range(4):
        factory._save_checkpoint_state(
            "gen-x11", doc_id, f"chk-{i}", status="completed",
            samples_data='[{"instruction": "Q", "response": "A"}]',
        )
    factory._save_checkpoint_state(
        "gen-x11", doc_id, "chk-4", status="failed", retry_count=3,
        error_message="ReadTimeout after 3 attempts",
    )

    summary = factory.get_checkpoint_summary(doc_id)
    assert summary["completed"] == 4
    assert summary["failed"] == 1
    assert summary["overall_status"] == "partially_completed"

    # F12: Gold export MUST raise HTTP 409 Conflict
    with pytest.raises(HTTPException) as exc_info:
        factory.export_document_dataset_strict(doc_id)
    assert exc_info.value.status_code == 409
    assert "incomplete" in exc_info.value.detail.lower()
    assert "4/5 completed" in exc_info.value.detail or "1 failed" in exc_info.value.detail
