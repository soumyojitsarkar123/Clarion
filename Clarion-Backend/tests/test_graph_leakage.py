"""
Regression tests for Graph Target Leakage prevention in GraphFeatureExtractor.

These tests mathematically verify that candidate relations being evaluated
cannot contaminate their own graph-structural features.
"""

import networkx as nx
import pytest

from core.experiments.graph_features import GraphFeatureExtractor, GraphEnrichedFeatureExtractor
from models.graph import GraphNodeType, GraphEdgeType


@pytest.fixture
def base_test_graph():
    """Build a synthetic graph with hierarchy and concept nodes."""
    G = nx.DiGraph()
    # Hierarchy
    G.add_node("doc-1", label="Root Document", node_type="document")
    G.add_node("top-1", label="Topic One", node_type="topic")
    G.add_edge("doc-1", "top-1", edge_type="hierarchy", relation_type="hierarchy")

    # Concepts under Topic One
    G.add_node("c-1", label="Concept Alpha", node_type="concept")
    G.add_node("c-2", label="Concept Beta", node_type="concept")
    G.add_node("c-3", label="Concept Gamma", node_type="concept")

    G.add_edge("top-1", "c-1", edge_type="hierarchy", relation_type="hierarchy")
    G.add_edge("top-1", "c-2", edge_type="hierarchy", relation_type="hierarchy")
    G.add_edge("top-1", "c-3", edge_type="hierarchy", relation_type="hierarchy")

    # A third-party bridge: Concept Alpha -> Concept Gamma
    G.add_edge("c-1", "c-3", edge_type="semantic", relation_type="prerequisite", weight=0.8)
    G.add_edge("c-3", "c-2", edge_type="semantic", relation_type="explanation", weight=0.7)

    return G


def test_shortest_path_not_trivial_one_when_edge_exists():
    """Verify shortest path is not 1.0 merely because the candidate edge exists."""
    G = nx.DiGraph()
    G.add_node("n1", label="Alpha", node_type="concept")
    G.add_node("n2", label="Beta", node_type="concept")
    # Only edge between them is the candidate edge itself
    G.add_edge("n1", "n2", relation_type="prerequisite")

    extractor = GraphFeatureExtractor(G)
    features = extractor.extract_features("Alpha", "Beta")

    # Without the candidate edge, there is no alternative path between n1 and n2
    assert features["shortest_path"] == -1.0, (
        f"Target leakage detected! Shortest path returned {features['shortest_path']} instead of -1.0"
    )
    assert features["is_connected"] == 0.0, "Target leakage detected! is_connected returned 1.0"


def test_shortest_path_reflects_true_alternative_distance(base_test_graph):
    """Verify shortest path accurately returns distance along alternative paths."""
    extractor = GraphFeatureExtractor(base_test_graph)
    
    # Path from Alpha (c-1) to Beta (c-2) exists via Gamma (c-3): length 2
    # Even if we insert candidate edge (c-1, c-2), shortest path must be 2.0 (via c-3), NOT 1.0
    base_test_graph.add_edge("c-1", "c-2", relation_type="cause-effect")
    
    features = extractor.extract_features("Concept Alpha", "Concept Beta")
    assert features["shortest_path"] == 2.0, (
        f"Expected path length 2.0 along alternative route, got {features['shortest_path']}"
    )


def test_degrees_and_centrality_do_not_count_candidate_edge():
    """Verify degrees and degree centrality exclude the candidate edge."""
    G = nx.DiGraph()
    G.add_node("n1", label="Alpha", node_type="concept")
    G.add_node("n2", label="Beta", node_type="concept")
    G.add_edge("n1", "n2", relation_type="prerequisite")

    extractor = GraphFeatureExtractor(G)
    features = extractor.extract_features("Alpha", "Beta")

    assert features["degree_a"] == 0.0, f"Expected degree 0.0, got {features['degree_a']}"
    assert features["degree_b"] == 0.0, f"Expected degree 0.0, got {features['degree_b']}"
    assert features["centrality_a"] == 0.0, f"Expected centrality 0.0, got {features['centrality_a']}"
    assert features["centrality_b"] == 0.0, f"Expected centrality 0.0, got {features['centrality_b']}"


def test_feature_invariance_before_and_after_edge_insertion(base_test_graph):
    """
    Critical Invariance Rule:
    Features extracted for candidate (A, B) must be IDENTICAL whether
    the edge (A, B) has already been inserted into the graph or not.
    """
    extractor = GraphFeatureExtractor(base_test_graph)

    # 1. Extract features when edge does NOT exist
    features_before = extractor.extract_features("Concept Alpha", "Concept Beta")

    # 2. Insert candidate edge into the graph
    base_test_graph.add_edge("c-1", "c-2", relation_type="prerequisite", weight=0.9)

    # 3. Extract features when edge DOES exist
    features_after = extractor.extract_features("Concept Alpha", "Concept Beta")

    # Check every feature key matches exactly
    for key, val_before in features_before.items():
        val_after = features_after[key]
        assert val_before == pytest.approx(val_after, abs=1e-5), (
            f"Feature invariance violated for key '{key}': "
            f"without edge = {val_before}, with edge = {val_after}"
        )


def test_common_neighbors_do_not_self_count():
    """Verify candidate nodes are not counted as their own common neighbors."""
    G = nx.DiGraph()
    G.add_node("n1", label="Alpha", node_type="concept")
    G.add_node("n2", label="Beta", node_type="concept")
    G.add_node("n3", label="Gamma", node_type="concept")

    # Gamma is a genuine common neighbor
    G.add_edge("n1", "n3")
    G.add_edge("n2", "n3")

    # Candidate edge between Alpha and Beta
    G.add_edge("n1", "n2")

    extractor = GraphFeatureExtractor(G)
    features = extractor.extract_features("Alpha", "Beta")

    # Common neighbors must be exactly 1 (Gamma), not 2 or 3
    assert features["common_neighbors"] == 1.0


def test_graph_enriched_feature_extractor_integration(base_test_graph):
    """Verify GraphEnrichedFeatureExtractor also inherits leak-free behavior."""
    enriched = GraphEnrichedFeatureExtractor(base_test_graph)
    base_record = {
        "llm_confidence": 0.85,
        "cooccurrence_score": 0.4,
        "semantic_similarity": 0.75,
    }

    # Add candidate edge
    base_test_graph.add_edge("c-1", "c-2", relation_type="prerequisite")

    features = enriched.extract_all_features(base_record, "Concept Alpha", "Concept Beta")

    # Shortest path must be 2.0 (via Gamma), NOT 1.0
    assert features["shortest_path"] == 2.0
    # Path-based boost must be 1 / (1 + 2) = 0.333..., NOT 1 / (1 + 1) = 0.5
    assert features["path_based_boost"] == pytest.approx(1.0 / 3.0, abs=1e-4)


def test_reciprocal_edge_leakage_prevention():
    """
    Verify that if a reciprocal edge (Beta -> Alpha) exists, it is also
    excluded when evaluating (Alpha -> Beta) so that an unvalidated bidirectional
    candidate edge does not artificially shorten the path to 1 or leak.
    """
    G = nx.DiGraph()
    G.add_node("n1", label="Alpha", node_type="concept")
    G.add_node("n2", label="Beta", node_type="concept")
    # Add candidate edge and reciprocal edge
    G.add_edge("n1", "n2", relation_type="related")
    G.add_edge("n2", "n1", relation_type="related")

    extractor = GraphFeatureExtractor(G)
    features = extractor.extract_features("Alpha", "Beta")

    # Both (n1, n2) and (n2, n1) should be excluded during evaluation
    assert features["degree_a"] == 0.0
    assert features["degree_b"] == 0.0
    assert features["shortest_path"] == -1.0
    assert features["is_connected"] == 0.0

    # Ensure graph was properly restored
    assert G.has_edge("n1", "n2")
    assert G.has_edge("n2", "n1")


def test_self_loop_edge_handling():
    """Verify that a candidate with node_a == node_b or self-loops does not crash or corrupt state."""
    G = nx.DiGraph()
    G.add_node("n1", label="Alpha", node_type="concept")
    G.add_edge("n1", "n1", relation_type="self")

    extractor = GraphFeatureExtractor(G)
    # Extract features for self candidate
    features = extractor.extract_features("Alpha", "Alpha")
    assert features is not None
    assert features["degree_a"] == 0.0
    assert G.has_edge("n1", "n1"), "Self-loop edge was not restored after context exit"


def test_batch_candidate_evaluation_isolation():
    """
    Verify that when evaluating a list of candidate edges from a batch,
    each candidate's feature extraction is isolated and deterministic.
    """
    G = nx.DiGraph()
    G.add_node("n1", label="Concept 1", node_type="concept")
    G.add_node("n2", label="Concept 2", node_type="concept")
    G.add_node("n3", label="Concept 3", node_type="concept")
    
    # Candidate batch: (1, 2) and (2, 3)
    # Baseline: Empty graph (no candidates inserted yet)
    extractor = GraphFeatureExtractor(G)
    feat_12_empty = extractor.extract_features("Concept 1", "Concept 2")
    feat_23_empty = extractor.extract_features("Concept 2", "Concept 3")

    # Now simulate candidates already inserted into the graph (e.g. speculative insertion)
    G.add_edge("n1", "n2", relation_type="rel1")
    G.add_edge("n2", "n3", relation_type="rel2")

    feat_12_inserted = extractor.extract_features("Concept 1", "Concept 2")
    # For (Concept 1, Concept 2), even if (2, 3) exists, path from 1 to 2 is -1.0 because there's no back path
    assert feat_12_inserted["shortest_path"] == -1.0
    assert feat_12_inserted["degree_a"] == 0.0
    # degree_b in the graph has edge (n2, n3), so degree of Concept 2 is 1.0 (from n2->n3)
    # but the candidate edge (n1, n2) is excluded.

