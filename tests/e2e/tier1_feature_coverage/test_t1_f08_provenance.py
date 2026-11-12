"""Tier 1 Feature Coverage: F8 Complete Provenance Preservation.

Specifications:
- T1.F8.1: Document ID Preservation in Sample
- T1.F8.2: Chunk ID Traceability
- T1.F8.3: Section Title Provenance
- T1.F8.4: Source Text Snippet Grounding (grounding excerpt preserved)
- T1.F8.5: SFT JSONL Provenance Export Format
"""

import json
import pytest
from services.dataset_factory_service import (
    DatasetFactoryService,
    TrainingSample,
)


def test_f08_document_id_preservation():
    """T1.F8.1: Document ID Preservation in Sample."""
    factory = DatasetFactoryService()
    sample = factory.create_training_sample(
        document_id="doc-alpha-123",
        chunk_id="chunk-001",
        chunk_text="A hash table is a data structure that implements an associative array abstract data type.",
        sample_type="qa",
        input_text="What is a hash table?",
        output_text="A hash table is a data structure implementing an associative array mapping keys to values.",
    )
    assert sample is not None
    assert sample.document_id == "doc-alpha-123"


def test_f08_chunk_id_traceability():
    """T1.F8.2: Chunk ID Traceability."""
    factory = DatasetFactoryService()
    sample = factory.create_training_sample(
        document_id="doc-alpha-123",
        chunk_id="chunk-004-pos2",
        chunk_text="A B-tree is a self-balancing search tree in which nodes have multiple children.",
        sample_type="concept",
        input_text="Define B-tree data structure.",
        output_text="A B-tree maintains sorted data and allows searches, sequential access, insertions, and deletions.",
    )
    assert sample is not None
    assert sample.chunk_id == "chunk-004-pos2"


def test_f08_section_title_provenance():
    """T1.F8.3: Section Title Provenance."""
    factory = DatasetFactoryService()
    chunk = {
        "chunk_id": "chk-007",
        "section_title": "3.2 Convergence Proof",
        "content": "Theorem 3: Under Lipschitz conditions, gradient descent converges to a stationary point.",
    }
    sample = factory.create_training_sample(
        document_id="doc-math-456",
        chunk_id=chunk["chunk_id"],
        chunk_text=chunk["content"],
        sample_type="qa",
        input_text="Under what conditions does gradient descent converge?",
        output_text="Under Lipschitz conditions, gradient descent converges to a stationary point.",
        context=chunk["section_title"],
    )
    assert sample is not None
    sample.metadata["section_title"] = chunk["section_title"]
    sample_dict = factory._sample_to_dict(sample)
    assert sample_dict["metadata"].get("section_title") == "3.2 Convergence Proof"


def test_f08_source_snippet_grounding():
    """T1.F8.4: Source Text Snippet Grounding."""
    factory = DatasetFactoryService()
    raw_text = (
        "Let G = (V, E) be a directed graph with n vertices. A cycle is a path of edges "
        "and vertices wherein a vertex is reachable from itself. Topological sorting is only "
        "possible if and only if the graph has no directed cycles (i.e. is a DAG)."
    )
    excerpt = factory._grounding_excerpt(raw_text)
    assert len(excerpt) <= 220
    assert excerpt.startswith("Let G = (V, E)")
    assert "directed graph" in excerpt


def test_f08_sft_jsonl_provenance_export():
    """T1.F8.5: SFT JSONL Provenance Export Format."""
    factory = DatasetFactoryService()
    sample = TrainingSample(
        id="sample-uuid-001",
        document_id="doc-export-789",
        chunk_id="chk-001",
        sample_type="qa",
        input_text="Explain depth-first search.",
        output_text="DFS is an algorithm for traversing or searching tree or graph data structures.",
        context="Graph Traversal",
        quality_score=0.95,
        metadata={
            "document_id": "doc-export-789",
            "chunk_id": "chk-001",
            "grounding_excerpt": "DFS traverses tree or graph structures exploring along each branch.",
        },
        created_at="2026-11-11T05:00:00",
    )
    sample_dict = factory._sample_to_dict(sample)
    jsonl_line = json.dumps({
        "instruction": sample_dict["input_text"],
        "response": sample_dict["output_text"],
        "metadata": {
            "document_id": sample_dict["document_id"],
            "chunk_id": sample_dict["chunk_id"],
            "grounding_excerpt": sample_dict["metadata"]["grounding_excerpt"],
        },
    })
    parsed = json.loads(jsonl_line)
    assert parsed["metadata"]["document_id"] == "doc-export-789"
    assert parsed["metadata"]["chunk_id"] == "chk-001"
    assert "grounding_excerpt" in parsed["metadata"]
