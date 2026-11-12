"""Unit tests for graph building, exporting, and hierarchy generation."""

import networkx as nx
import pytest

from graph.builder import GraphBuilder
from graph.exporter import GraphExporter
from graph.hierarchy import HierarchyGenerator
from models.knowledge_map import (
    Concept,
    KnowledgeMap,
    MainTopic,
    Relation,
    RelationType,
)
from models.graph import GraphEdgeType, GraphNodeType


def make_knowledge_map():
    concepts = [
        Concept(id="c-1", name="Hypothesis Testing"),
        Concept(id="c-2", name="Null Hypothesis"),
        Concept(id="c-3", name="Sampling Distribution"),
    ]
    relations = [
        Relation(
            id="r-1",
            from_concept="Hypothesis Testing",
            to_concept="Null Hypothesis",
            relation_type=RelationType.PREREQUISITE,
            confidence=0.8,
            description="Null hypothesis is required first.",
        ),
        Relation(
            id="r-2",
            from_concept="Sampling Distribution",
            to_concept="Hypothesis Testing",
            relation_type=RelationType.EXPLANATION,
            confidence=0.6,
        ),
    ]
    topics = [
        MainTopic(id="t-1", title="Statistics", concept_ids=["c-1", "c-2", "c-3"]),
    ]
    return KnowledgeMap(
        document_id="doc-1",
        concepts=concepts,
        relations=relations,
        main_topics=topics,
        metadata={"extraction_method": "heuristic"},
    )


def test_graph_builder_creates_nodes_and_edges(isolated_settings):
    builder = GraphBuilder()
    graph = builder.build_from_knowledge_map(make_knowledge_map())
    assert graph.number_of_nodes() >= 4  # doc + topic + 3 concepts
    assert graph.number_of_edges() >= 4  # hierarchy + semantic
    node_types = {data.get("node_type") for _, data in graph.nodes(data=True)}
    assert GraphNodeType.DOCUMENT.value in node_types
    assert GraphNodeType.CONCEPT.value in node_types
    edge_types = {data.get("relation_type") for _, _, data in graph.edges(data=True)}
    assert GraphEdgeType.HIERARCHY.value in edge_types


def test_graph_builder_falls_back_without_topics(isolated_settings):
    builder = GraphBuilder()
    knowledge_map = make_knowledge_map()
    knowledge_map.main_topics = []
    graph = builder.build_from_knowledge_map(knowledge_map)
    doc_nodes = [
        n for n in graph.nodes() if graph.nodes[n].get("node_type") == GraphNodeType.DOCUMENT.value
    ]
    assert doc_nodes
    hierarchy_edges = [
        (u, v) for u, v, d in graph.edges(data=True)
        if d.get("relation_type") == GraphEdgeType.HIERARCHY.value
    ]
    assert len(hierarchy_edges) >= len(knowledge_map.concepts)


def test_exporter_cytoscape_round_trip(isolated_settings, tmp_path):
    builder = GraphBuilder()
    graph = builder.build_from_knowledge_map(make_knowledge_map())
    exporter = GraphExporter()
    path = tmp_path / "graph_cytoscape.json"
    exporter.save_to_file(graph, path, format="cytoscape")
    assert path.exists()
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert "nodes" in payload["elements"]
    assert "edges" in payload["elements"]


def test_exporter_json_round_trip(isolated_settings, tmp_path):
    builder = GraphBuilder()
    graph = builder.build_from_knowledge_map(make_knowledge_map())
    exporter = GraphExporter()
    path = tmp_path / "graph.json"
    exporter.save_to_file(graph, path, format="json")
    assert path.exists()


def test_hierarchy_conceptual_layers(isolated_settings):
    builder = GraphBuilder()
    graph = builder.build_from_knowledge_map(make_knowledge_map())
    generator = HierarchyGenerator(graph)
    layers = generator.generate_conceptual_layers()
    assert layers
    for layer in layers:
        assert layer.level >= 1
        assert layer.name
        assert layer.concept_ids


def test_hierarchy_topic_tree(isolated_settings):
    builder = GraphBuilder()
    graph = builder.build_from_knowledge_map(make_knowledge_map())
    generator = HierarchyGenerator(graph)
    tree = generator.generate_topic_tree()
    assert tree is not None
    assert tree.node_type == GraphNodeType.DOCUMENT.value


def test_edge_provenance_and_calibration_in_cytoscape(isolated_settings):
    """Verify calibrated probability and evidence snippet are preserved in Cytoscape export."""
    km = make_knowledge_map()
    km.relations[0].calibrated_probability = 0.892
    km.relations[0].evidence_text = "The null hypothesis specifies that there is no effect."
    km.relations[0].signal_scores = {"entailment": 0.85, "cooc": 1.0}

    builder = GraphBuilder()
    graph = builder.build_from_knowledge_map(km)

    # Check graph edge attributes
    edge_data = None
    for u, v, d in graph.edges(data=True):
        if d.get("calibrated_probability") == 0.892:
            edge_data = d
            break

    assert edge_data is not None
    assert edge_data["evidence_text"] == "The null hypothesis specifies that there is no effect."
    # Calibration is a model score; the source relation remains a candidate
    # until the evidence itself is independently validated.
    assert edge_data["validated"] is False
    assert edge_data["is_candidate"] is True
    assert edge_data["signal_scores"]["entailment"] == 0.85

    # Check Cytoscape export
    exporter = GraphExporter()
    cyto = exporter.to_cytoscape_json(graph)
    cyto_edge = None
    for e in cyto["elements"]["edges"]:
        if e["data"].get("calibrated_probability") == 0.892:
            cyto_edge = e
            break

    assert cyto_edge is not None
    assert cyto_edge["data"]["evidence_text"] == "The null hypothesis specifies that there is no effect."
    assert cyto_edge["data"]["validated"] is False
    assert cyto_edge["data"]["is_candidate"] is True

