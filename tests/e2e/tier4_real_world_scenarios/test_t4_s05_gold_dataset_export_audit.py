"""Tier 4 Real-World Application Scenario: test_t4_s05_gold_dataset_export_audit.py
Scenario 5: Research Gold Export Audit.
Enforces research data integrity under require_llm_for_export=True,
validating strict refusal of incomplete datasets (HTTP 409 Conflict) and zero heuristic fallback.
"""

import json
import sys
from pathlib import Path
import pytest
from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_factory_service import DatasetFactoryService
from models.document import Document, DocumentMetadata


def test_t4_s05_gold_dataset_export_audit(tmp_path):
    """T4.SCEN.5: Full audit of gold dataset export gate."""
    db_file = tmp_path / "s05_gold_audit.db"
    factory = DatasetFactoryService(db_path=db_file)
    factory.require_llm_for_export = True
    doc_id = "doc-s05-gold-audit"

    doc = Document(id=doc_id, metadata=DocumentMetadata(filename="gold_study.pdf", file_type="pdf", file_size=3500))
    factory.document_service.save_document(doc)

    # 1. Test Incomplete Generation Refusal: 2 completed, 1 failed
    factory._save_checkpoint_state(
        "gen-s05", doc_id, "chk-1", status="completed",
        samples_data='[{"instruction": "Q1", "response": "A1", "metadata": {"generation_mode": "llm_ollama"}}]',
    )
    factory._save_checkpoint_state(
        "gen-s05", doc_id, "chk-2", status="completed",
        samples_data='[{"instruction": "Q2", "response": "A2", "metadata": {"generation_mode": "llm_ollama"}}]',
    )
    factory._save_checkpoint_state(
        "gen-s05", doc_id, "chk-3", status="failed", retry_count=3,
        error_message="Permanent provider failure",
    )

    with pytest.raises(HTTPException) as exc_info:
        factory.export_document_dataset_strict(doc_id)
    assert exc_info.value.status_code == 409
    assert "incomplete" in exc_info.value.detail.lower()

    # 2. Test Heuristic Record Refusal: Chunk 3 completed but in deterministic fallback mode
    factory._save_checkpoint_state(
        "gen-s05", doc_id, "chk-3", status="completed",
        samples_data='[{"instruction": "Q3", "response": "A3", "metadata": {"generation_mode": "deterministic"}}]',
    )

    with pytest.raises(HTTPException) as exc_info2:
        factory.export_document_dataset_strict(doc_id)
    assert exc_info2.value.status_code == 409
    assert "heuristic records detected" in exc_info2.value.detail.lower()

    # 3. Test Authorized Gold Export: Replace chunk 3 with genuine LLM record
    factory._save_checkpoint_state(
        "gen-s05", doc_id, "chk-3", status="completed",
        samples_data='[{"instruction": "Q3", "response": "A3", "metadata": {"generation_mode": "llm_ollama"}}]',
    )

    result = factory.export_document_dataset_strict(doc_id)
    assert result["status"] == "success"
    assert result["sample_count"] == 3
    assert Path(result["export_path"]).exists()
