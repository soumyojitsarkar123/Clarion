"""Tier 4 Real-World Application Scenario: test_t4_s03_textbook_chapter_pipeline.py
Scenario 3: Textbook Chapter (DAA Graph Algorithms).
Simulates ingestion of a dense academic textbook chapter: structure-aware chunking,
provenance preservation, knowledge graph construction, and graph density evaluation.
"""

import sys
from pathlib import Path
import networkx as nx
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.chunking_service import ChunkingService
from routers.system import _single_graph_stats


def test_t4_s03_textbook_chapter_pipeline(tmp_path):
    """T4.SCEN.3: Dense textbook chapter ingestion and knowledge graph construction."""
    doc_id = "doc-daa-chapter-graph-algos"
    chapter_text = (
        "Design and Analysis of Algorithms: Chapter 22 - Elementary Graph Algorithms\n\n"
        "22.1 Representations of Graphs\n"
        "There are two standard ways to represent a graph G = (V, E): as a collection of adjacency lists "
        "or as an adjacency matrix. The adjacency-list representation provides a compact way to represent sparse graphs.\n\n"
        "22.2 Breadth-First Search (BFS)\n"
        "Breadth-first search is one of the simplest algorithms for searching a graph. Given a graph G and a source s, "
        "BFS systematically explores the edges of G to discover every vertex reachable from s. It computes the distance "
        "(smallest number of edges) from s to each reachable vertex.\n\n"
        "22.3 Depth-First Search (DFS)\n"
        "Depth-first search searches deeper in the graph whenever possible. Edges are explored out of the most recently "
        "discovered vertex v that still has unexplored edges leaving it. DFS timestamps each vertex with discovery time d[v] "
        "and finishing time f[v]."
    )

    # 1. Chunking
    chunker = ChunkingService()
    chunks = chunker.chunk_document(doc_id, chapter_text)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.document_id == doc_id
        assert chunk.word_count > 0

    # 2. Graph Construction
    graph = nx.DiGraph()
    concepts = [
        ("Graph", "Fundamental structural concept"),
        ("Adjacency List", "Sparse graph representation"),
        ("Adjacency Matrix", "Dense graph representation"),
        ("BFS", "Breadth-first search algorithm"),
        ("DFS", "Depth-first search algorithm"),
    ]
    for name, desc in concepts:
        graph.add_node(name, node_type="concept", label=name, description=desc)

    relations = [
        ("Graph", "Adjacency List", "represented_by"),
        ("Graph", "Adjacency Matrix", "represented_by"),
        ("Graph", "BFS", "traversed_by"),
        ("Graph", "DFS", "traversed_by"),
    ]
    for u, v, r_type in relations:
        graph.add_edge(u, v, relation_type=r_type)

    # 3. Graph Analysis
    stats = _single_graph_stats(doc_id, graph)
    assert stats["concept_count"] == 5
    assert stats["relation_count"] == 4
    assert stats["node_count"] == 5
    assert stats["edge_count"] == 4
    assert stats["graph_density"] > 0.0
