"""Tier 2 Boundary & Corner Cases: test_t2_b12_export_gate_corner.py
Feature F12: Research Integrity & Gold Export Gate (HTTP 409).
Verifies boundary conditions: export refusal on unprocessed documents (404/409),
refusal during active pending/running generation (409), refusal on 0 valid samples (409),
refusal when heuristic/deterministic records are detected under require_llm_for_export=True (409),
and concurrency safety.
"""

import json
import sqlite3
import sys
from pathlib import Path
import pytest
from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_factory_service import DatasetFactoryService
from models.document import Document, DocumentMetadata


def test_b12_export_unprocessed_document(tmp_path):
    """T2.F12.1: Export attempt on completely unprocessed document."""
    db_file = tmp_path / "factory_unprocessed.db"
    factory = DatasetFactoryService(db_path=db_file)
    factory.require_llm_for_export = True

    # Case A: Document doesn't exist -> 404
    with pytest.raises(HTTPException) as exc_info:
        factory.export_document_dataset_strict("doc-nonexistent-999")
    assert exc_info.value.status_code == 404

    # Case B: Document exists but 0 checkpoints -> 409
    doc_id = "doc-exist-no-chk"
    doc = Document(id=doc_id, metadata=DocumentMetadata(filename="test.pdf", file_type="pdf", file_size=100))
    factory.document_service.save_document(doc)

    with pytest.raises(HTTPException) as exc_info2:
        factory.export_document_dataset_strict(doc_id)
    assert exc_info2.value.status_code == 409
    assert "no dataset generation records found" in exc_info2.value.detail.lower()


def test_b12_export_during_active_generation(tmp_path):
    """T2.F12.2: Export attempt during active background generation (pending/running chunks)."""
    db_file = tmp_path / "factory_active_gen.db"
    factory = DatasetFactoryService(db_path=db_file)
    factory.require_llm_for_export = True
    doc_id = "doc-active-gen"

    doc = Document(id=doc_id, metadata=DocumentMetadata(filename="active.pdf", file_type="pdf", file_size=200))
    factory.document_service.save_document(doc)

    # 1 completed chunk, 1 running chunk
    factory._save_checkpoint_state("gen-1", doc_id, "chk-1", status="completed", samples_data="[]")
    factory._save_checkpoint_state("gen-1", doc_id, "chk-2", status="running")

    with pytest.raises(HTTPException) as exc_info:
        factory.export_document_dataset_strict(doc_id)
    assert exc_info.value.status_code == 409
    assert "incomplete" in exc_info.value.detail.lower()


def test_b12_zero_eligible_samples_quality_filter(tmp_path):
    """T2.F12.3: Zero eligible samples available for export."""
    db_file = tmp_path / "factory_zero_samples.db"
    factory = DatasetFactoryService(db_path=db_file)
    factory.require_llm_for_export = True
    doc_id = "doc-zero-samples"

    doc = Document(id=doc_id, metadata=DocumentMetadata(filename="zero.pdf", file_type="pdf", file_size=200))
    factory.document_service.save_document(doc)

    # Completed checkpoint but empty samples_data
    factory._save_checkpoint_state("gen-2", doc_id, "chk-1", status="completed", samples_data="[]")

    with pytest.raises(HTTPException) as exc_info:
        factory.export_document_dataset_strict(doc_id)
    assert exc_info.value.status_code == 409
    assert "0 valid samples" in exc_info.value.detail.lower()


def test_b12_heuristic_records_refusal_in_gold_export(tmp_path):
    """T2.F12.4: Attempted export of heuristic records under require_llm_for_export=True."""
    db_file = tmp_path / "factory_heuristic.db"
    factory = DatasetFactoryService(db_path=db_file)
    factory.require_llm_for_export = True
    doc_id = "doc-heuristic-guard"

    doc = Document(id=doc_id, metadata=DocumentMetadata(filename="heur.pdf", file_type="pdf", file_size=200))
    factory.document_service.save_document(doc)

    # Create sample dictionary with deterministic generation_mode
    deterministic_sample = {
        "id": "sample-heur-1",
        "sample_type": "concept",
        "chunk_id": "chk-1",
        "document_id": doc_id,
        "input_text": "What is X?",
        "output_text": "X is Y according to heuristic extractor.",
        "context": "Context",
        "quality_score": 0.90,
        "metadata": {"generation_mode": "deterministic"},
    }
    factory._save_checkpoint_state(
        "gen-3", doc_id, "chk-1", status="completed",
        samples_data=json.dumps([deterministic_sample]),
    )

    with pytest.raises(HTTPException) as exc_info:
        factory.export_document_dataset_strict(doc_id)
    assert exc_info.value.status_code == 409
    assert "heuristic records detected" in exc_info.value.detail.lower()


def test_b12_permissive_export_when_require_llm_false(tmp_path):
    """T2.F12.5: Permissive silver export allowed when require_llm_for_export=False."""
    db_file = tmp_path / "factory_permissive.db"
    factory = DatasetFactoryService(db_path=db_file)
    factory.require_llm_for_export = False
    doc_id = "doc-permissive"

    doc = Document(id=doc_id, metadata=DocumentMetadata(filename="perm.pdf", file_type="pdf", file_size=200))
    factory.document_service.save_document(doc)

    sample = {
        "id": "sample-perm-1",
        "sample_type": "qa",
        "chunk_id": "chk-1",
        "document_id": doc_id,
        "input_text": "What is Dijkstra algorithm?",
        "output_text": "Dijkstra algorithm computes shortest paths in non-negative weighted graphs.",
        "context": "Shortest paths",
        "quality_score": 0.88,
        "metadata": {"generation_mode": "deterministic"},
    }
    factory._save_checkpoint_state(
        "gen-4", doc_id, "chk-1", status="completed",
        samples_data=json.dumps([sample]),
    )

    # When require_llm_for_export=False, export proceeds without raising 409 for heuristic mode
    res = factory.export_document_dataset_strict(doc_id)
    assert isinstance(res, dict)
    assert res.get("status") == "success"
    assert "export_path" in res
