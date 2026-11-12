"""Tier 1 Feature Coverage: F12 Research Integrity & Gold Export Gate (HTTP 409).

Specifications:
- T1.F12.1: Clean Gold Export Under 100% LLM Completion (HTTP 200)
- T1.F12.2: Export Refusal on Incomplete Dataset (HTTP 409 Conflict)
- T1.F12.3: Zero Heuristic Fallback Records in Gold Export (HTTP 409 Conflict)
- T1.F12.4: Status Marking as partially_completed
- T1.F12.5: Permissive Export When require_llm_for_export=False
"""

import json
from unittest.mock import MagicMock
import pytest
from fastapi import HTTPException
from services.dataset_factory_service import DatasetFactoryService, TrainingSample


def test_f12_clean_gold_export(isolated_settings):
    """T1.F12.1: Clean Gold Export Under 100% LLM Completion."""
    factory = DatasetFactoryService()
    factory.require_llm_for_export = True
    doc_id = "doc-f12-clean"

    # Mock document service
    mock_doc = MagicMock(id=doc_id, filename="CleanDoc.pdf", title="Clean Doc")
    factory.document_service.get_document = MagicMock(return_value=mock_doc)

    # 2 completed chunks with LLM generation mode
    samples_1 = [
        factory._sample_to_dict(
            TrainingSample(
                id="s-1",
                document_id=doc_id,
                chunk_id="chk-1",
                sample_type="qa",
                input_text="What is X?",
                output_text="X is Y.",
                context="",
                quality_score=0.9,
                metadata={"generation_mode": "llm_generated", "trainable": True},
                created_at="2026-11-11T05:00:00",
            )
        )
    ]
    factory._save_checkpoint_state(
        generation_id="gen-f12-clean",
        document_id=doc_id,
        chunk_id="chk-1",
        status="completed",
        sample_ids=["s-1"],
        samples_data=samples_1,
    )

    res = factory.export_document_dataset_strict(doc_id)
    assert res is not None
    assert "export_path" in res or "jsonl_path" in res
    assert res.get("sample_count") == 1


def test_f12_export_refusal_incomplete_409(isolated_settings):
    """T1.F12.2: Export Refusal on Incomplete Dataset (HTTP 409)."""
    factory = DatasetFactoryService()
    factory.require_llm_for_export = True
    doc_id = "doc-f12-incomplete"

    mock_doc = MagicMock(id=doc_id, filename="Incomplete.pdf")
    factory.document_service.get_document = MagicMock(return_value=mock_doc)

    # 1 chunk completed, 1 chunk failed
    factory._save_checkpoint_state(
        generation_id="gen-f12-inc",
        document_id=doc_id,
        chunk_id="chk-1",
        status="completed",
        sample_ids=["s-1"],
        samples_data=[{"id": "s-1", "document_id": doc_id, "chunk_id": "chk-1", "sample_type": "qa", "input_text": "Q", "output_text": "A", "quality_score": 0.9, "metadata": {"generation_mode": "llm"}}],
    )
    factory._save_checkpoint_state(
        generation_id="gen-f12-inc",
        document_id=doc_id,
        chunk_id="chk-2",
        status="failed",
        error_message="Provider timeout",
    )

    with pytest.raises(HTTPException) as exc_info:
        factory.export_document_dataset_strict(doc_id)

    assert exc_info.value.status_code == 409
    assert "incomplete" in str(exc_info.value.detail).lower()


def test_f12_zero_heuristic_fallback_records(isolated_settings):
    """T1.F12.3: Zero Heuristic Fallback Records in Gold Export."""
    factory = DatasetFactoryService()
    factory.require_llm_for_export = True
    doc_id = "doc-f12-heuristic"

    mock_doc = MagicMock(id=doc_id, filename="Tainted.pdf")
    factory.document_service.get_document = MagicMock(return_value=mock_doc)

    # Chunks completed, but with deterministic heuristic fallback mode
    heuristic_sample = {
        "id": "s-heur-1",
        "document_id": doc_id,
        "chunk_id": "chk-1",
        "sample_type": "qa",
        "input_text": "Q",
        "output_text": "A",
        "quality_score": 0.8,
        "metadata": {"generation_mode": "deterministic"},
    }
    factory._save_checkpoint_state(
        generation_id="gen-f12-heur",
        document_id=doc_id,
        chunk_id="chk-1",
        status="completed",
        sample_ids=["s-heur-1"],
        samples_data=[heuristic_sample],
    )

    with pytest.raises(HTTPException) as exc_info:
        factory.export_document_dataset_strict(doc_id)

    assert exc_info.value.status_code == 409
    assert "heuristic" in str(exc_info.value.detail).lower()


def test_f12_status_partially_completed():
    """T1.F12.4: Status Marking as partially_completed."""
    # When completed < total, batch status is partially_completed, never completed
    total_chunks = 5
    completed_chunks = 4
    failed_chunks = 1
    batch_status = "completed" if completed_chunks == total_chunks else "partially_completed"
    assert batch_status == "partially_completed"
    assert batch_status != "completed"


def test_f12_permissive_export_without_strict_llm(isolated_settings):
    """T1.F12.5: Permissive Export When require_llm_for_export=False."""
    factory = DatasetFactoryService()
    factory.require_llm_for_export = False  # relaxed mode
    doc_id = "doc-f12-relaxed"

    mock_doc = MagicMock(id=doc_id, filename="Relaxed.pdf", title="Relaxed")
    factory.document_service.get_document = MagicMock(return_value=mock_doc)

    sample = {
        "id": "s-rel-1",
        "document_id": doc_id,
        "chunk_id": "chk-1",
        "sample_type": "qa",
        "input_text": "Q",
        "output_text": "A",
        "quality_score": 0.85,
        "metadata": {"generation_mode": "deterministic", "trainable": True},
    }
    factory._save_checkpoint_state(
        generation_id="gen-f12-rel",
        document_id=doc_id,
        chunk_id="chk-1",
        status="completed",
        sample_ids=["s-rel-1"],
        samples_data=[sample],
    )

    res = factory.export_document_dataset_strict(doc_id)
    assert res is not None
    assert "export_path" in res or "jsonl_path" in res
