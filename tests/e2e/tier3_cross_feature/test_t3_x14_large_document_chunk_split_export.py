"""Tier 3 Cross-Feature Combination: test_t3_x14_large_document_chunk_split_export.py
Feature Interaction: F8 (Complete Provenance Preservation) + F12 (Research Integrity & Gold Export Gate).
Validates that exported gold dataset files maintain complete provenance tracing for every sample,
guaranteeing non-empty grounding excerpts, valid chunk_ids, and zero heuristic records.
"""

import json
import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_factory_service import DatasetFactoryService
from models.document import Document, DocumentMetadata


def test_t3_x14_large_document_chunk_split_export(tmp_path):
    """T3.INT.14: F8 + F12 Pairwise Integration."""
    db_file = tmp_path / "x14_export.db"
    factory = DatasetFactoryService(db_path=db_file)
    factory.require_llm_for_export = True
    doc_id = "doc-x14-gold-provenance"

    doc = Document(id=doc_id, metadata=DocumentMetadata(filename="gold_paper.pdf", file_type="pdf", file_size=5000))
    factory.document_service.save_document(doc)

    samples = [
        {
            "id": f"sample-x14-{i}",
            "sample_type": "qa",
            "chunk_id": f"chk-{i}",
            "document_id": doc_id,
            "input_text": f"Question about concept {i}?",
            "output_text": f"Detailed answer regarding concept {i}.",
            "context": f"Source text context for concept {i}.",
            "quality_score": 0.92,
            "metadata": {
                "generation_mode": "llm_ollama",
                "grounding_excerpt": f"Source text excerpt for concept {i}.",
                "section_title": f"Section {i}",
            },
        }
        for i in range(3)
    ]

    for i in range(3):
        factory._save_checkpoint_state(
            "gen-x14", doc_id, f"chk-{i}", status="completed",
            samples_data=json.dumps([samples[i]]),
        )

    # Gold export
    result = factory.export_document_dataset_strict(doc_id)
    assert result["status"] == "success"
    assert result["sample_count"] == 3

    # Inspect exported file
    export_json = Path(result["export_path"])
    assert export_json.exists()
    payload = json.loads(export_json.read_text(encoding="utf-8"))

    # F8 & F12 Invariants: Every sample has full provenance and non-deterministic generation mode
    assert len(payload["samples"]) == 3
    for s in payload["samples"]:
        assert s["chunk_id"].startswith("chk-")
        assert len(s["grounding_excerpt"]) > 0
        assert s["quality_score"] >= 0.80
    assert payload["generation_mode"].startswith("llm_")
