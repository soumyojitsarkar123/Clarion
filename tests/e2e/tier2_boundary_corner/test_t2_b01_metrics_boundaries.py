"""Tier 2 Boundary & Corner Cases: test_t2_b01_metrics_boundaries.py
Feature F1: Genuine Stage Metrics Passthrough.
Verifies boundary conditions: zero-byte inputs, single-word documents, extreme token lengths,
and zero-edge/single-node graph density boundaries without division-by-zero errors.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import networkx as nx

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.pipeline_observability import PipelineStateTracker, estimate_size
from services.embedding_service import EmbeddingService
from services.chunking_service import ChunkingService
from routers.system import _single_graph_stats
from fastapi.testclient import TestClient
from main import app


def test_b01_zero_byte_ingestion():
    """T2.F1.1: Zero-byte ingestion boundary."""
    client = TestClient(app)
    # Test public API rejection of zero-byte file
    res = client.post(
        "/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert res.status_code == 400
    assert "empty" in res.json().get("detail", "").lower()

    # Test estimate_size on zero-byte payload
    assert estimate_size(b"") == 0
    assert estimate_size("") == 0
    assert estimate_size(None) == 0


def test_b01_single_word_chunking():
    """T2.F1.2: Single-word text document chunking boundary."""
    chunker = ChunkingService()
    single_word_text = "Mathematics"
    chunks = chunker.chunk_document("doc-single-word", single_word_text)
    assert len(chunks) == 1
    assert chunks[0].content.strip() == "Mathematics"
    assert estimate_size(chunks) == 1


def test_b01_extreme_subword_tokens_embedding():
    """T2.F1.3: Extreme subword token count (>512 tokens) in embedding."""
    service = EmbeddingService()
    extreme_text = "X" * 3000
    vec = service._fallback_embed_text(extreme_text)
    assert isinstance(vec, list)
    assert len(vec) == 1024
    norm = np.linalg.norm(vec)
    assert norm == pytest.approx(1.0, abs=1e-3)


def test_b01_disconnected_graph_zero_edges():
    """T2.F1.4: Disconnected knowledge graph with zero edges."""
    graph = nx.DiGraph()
    for i in range(5):
        graph.add_node(f"concept_{i}", node_type="concept", label=f"Concept {i}")
    stats = _single_graph_stats("doc-test-disc", graph)
    assert stats["node_count"] == 5
    assert stats["edge_count"] == 0
    assert stats["concept_count"] == 5
    assert stats["relation_count"] == 0
    assert stats["graph_density"] == 0.0


def test_b01_single_node_graph_density():
    """T2.F1.5: Single-node graph (|V| = 1) density boundary."""
    graph = nx.DiGraph()
    graph.add_node("concept_single", node_type="concept", label="Single")
    stats = _single_graph_stats("doc-test-single", graph)
    assert stats["node_count"] == 1
    assert stats["edge_count"] == 0
    assert stats["graph_density"] == 0.0
