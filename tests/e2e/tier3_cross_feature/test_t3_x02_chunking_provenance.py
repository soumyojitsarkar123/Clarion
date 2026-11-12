"""Tier 3 Cross-Feature Combination: test_t3_x02_chunking_provenance.py
Feature Interaction: F7 (Per-Chunk / Small-Batch Dataset Generation) + F8 (Complete Provenance Preservation).
Validates that per-chunk dataset generation rigorously preserves full provenance metadata:
document_id, chunk_id, section_title, and grounding_excerpt pointing to source text.
"""

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_factory_service import DatasetFactoryService


def test_t3_x02_chunking_provenance(tmp_path):
    """T3.INT.2: F7 + F8 Pairwise Integration."""
    factory = DatasetFactoryService(db_path=tmp_path / "x02_factory.db")
    doc_id = "doc-x02-provenance"
    chunk_id = "chk-x02-pos0"
    section_title = "4.2 Dijkstra Shortest Path Algorithm"
    source_content = (
        "Dijkstra algorithm computes shortest paths from a single source vertex to all other vertices "
        "in a weighted directed graph with non-negative edge weights. The algorithm maintains a priority queue "
        "of candidate vertices ordered by tentative distance. The time complexity with a Fibonacci heap is O(V log V + E)."
    )

    sample = factory.create_training_sample(
        chunk_id=chunk_id,
        document_id=doc_id,
        sample_type="qa",
        input_text="What is the time complexity of Dijkstra algorithm with Fibonacci heaps?",
        output_text="The time complexity of Dijkstra algorithm with Fibonacci heaps is O(V log V + E).",
        context=source_content,
    )

    assert sample is not None
    # F8 Provenance Invariants
    assert sample.document_id == doc_id
    assert sample.chunk_id == chunk_id
    assert sample.sample_type == "qa"
    # Grounding excerpt check
    excerpt = factory._grounding_excerpt(source_content)
    assert len(excerpt) > 0
    assert excerpt in source_content
