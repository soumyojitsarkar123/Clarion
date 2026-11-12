"""Tier 1 Feature Coverage: F7 Per-Chunk / Small-Batch Dataset Generation.

Specifications:
- T1.F7.1: Single-Chunk Prompt Payload Construction (1 chunk per prompt)
- T1.F7.2: Text Length Bounding per Chunk Prompt (content bounded to 1200 chars)
- T1.F7.3: SFT Question-Answer Pair Generation (valid QA pairs from single chunk)
- T1.F7.4: Concept Definition Extraction per Chunk (structured definitions)
- T1.F7.5: Batch Progress Telemetry Tracking (statistics endpoint & tracking)
"""

import pytest
from services.dataset_factory_service import DatasetFactoryService


def test_f07_single_chunk_payload_construction():
    """T1.F7.1: Single-Chunk Prompt Payload Construction."""
    factory = DatasetFactoryService()
    chunk = {
        "chunk_id": "chunk-test-001",
        "section_title": "Graph Algorithms",
        "content": "Dijkstra's algorithm solves the single-source shortest path problem for non-negative weights.",
    }
    prompt = factory._build_per_chunk_prompt(chunk, "Algorithms Textbook")

    assert "chunk-test-001" in prompt, "Prompt must contain the target chunk_id"
    assert "Graph Algorithms" in prompt, "Prompt must contain section title"
    assert "Dijkstra" in prompt, "Prompt must contain chunk text"
    assert "Return a JSON object with this shape" in prompt


def test_f07_text_length_bounding():
    """T1.F7.2: Text Length Bounding per Chunk Prompt."""
    factory = DatasetFactoryService()
    long_content = "Word " * 2000  # ~10,000 characters
    chunk = {
        "chunk_id": "chunk-long-002",
        "section_title": "Asymptotics",
        "content": long_content,
    }
    prompt = factory._build_per_chunk_prompt(chunk, "Big Document")

    # Content section within the prompt must be bounded to 1200 characters
    assert len(long_content) > 1200
    assert prompt.count("Word ") <= 300, "Prompt must bound long chunk content to <= 1200 characters"


def test_f07_sft_qa_pair_generation():
    """T1.F7.3: SFT Question-Answer Pair Generation."""
    factory = DatasetFactoryService()
    chunk_text = (
        "In algorithmic graph theory, topological sorting of a directed acyclic graph (DAG) "
        "is a linear ordering of vertices such that for every directed edge u -> v, u comes before v. "
        "It can be computed in linear time using depth-first search (DFS) or Kahn's algorithm."
    )
    qa_pairs = factory.generate_qa_pairs(chunk_text, max_pairs=2)
    assert isinstance(qa_pairs, list), "Expected list of QA pairs"
    assert len(qa_pairs) > 0, "Expected at least 1 generated QA pair"
    for pair in qa_pairs:
        assert "question" in pair and len(pair["question"].strip()) > 0
        assert "answer" in pair and len(pair["answer"].strip()) > 0


def test_f07_concept_definition_extraction():
    """T1.F7.4: Concept Definition Extraction per Chunk."""
    factory = DatasetFactoryService()
    chunk_text = (
        "A binary search tree is a rooted binary tree data structure where each internal node stores a key. "
        "The left subtree contains keys smaller than the node, and the right subtree contains keys greater."
    )
    concepts = factory.extract_concept_definitions(chunk_text)
    assert isinstance(concepts, list)
    assert len(concepts) > 0, "Expected at least 1 extracted concept definition"
    first = concepts[0]
    assert "concept" in first and len(first["concept"]) > 0
    assert "definition" in first and len(first["definition"]) > 0


def test_f07_batch_progress_telemetry(e2e_client):
    """T1.F7.5: Batch Progress Telemetry Tracking."""
    response = e2e_client.get("/dataset-factory/statistics")
    assert response.status_code == 200
    stats = response.json()

    expected_keys = [
        "total_processed",
        "total_samples",
        "total_exported",
        "quality_filtered",
        "dedup_filtered",
        "last_run",
    ]
    for k in expected_keys:
        assert k in stats, f"Missing key in dataset factory statistics: {k}"
