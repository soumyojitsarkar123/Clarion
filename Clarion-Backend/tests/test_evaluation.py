"""Unit tests for confidence scoring and hallucination detection."""

import networkx as nx
import pytest

from core.evaluation.confidence_scorer import ConfidenceScorer
from core.evaluation.hallucination_detector import (
    FlagSeverity,
    FlagType,
    HallucinationDetector,
)
from models.chunk import Chunk
from models.knowledge_map import Concept, KnowledgeMap, Relation, RelationType


def make_chunk(chunk_id, content, position=0):
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=content,
        position_index=position,
        word_count=len(content.split()),
    )


CHUNKS = [
    make_chunk("chunk-1", "Hypothesis testing evaluates the null hypothesis using a random sample."),
    make_chunk("chunk-2", "The sampling distribution approximates the sample mean distribution.", 10),
]


def make_relation(from_name, to_name, confidence=0.7, relation_type=RelationType.EXPLANATION):
    return Relation(
        id=f"r-{from_name}-{to_name}",
        from_concept=from_name,
        to_concept=to_name,
        relation_type=relation_type,
        confidence=confidence,
    )


def make_knowledge_map(document_id="doc-1", concepts=None, relations=None):
    return KnowledgeMap(
        document_id=document_id,
        concepts=concepts if concepts is not None else [
            Concept(id="c-1", name="Hypothesis Testing"),
            Concept(id="c-2", name="Null Hypothesis"),
            Concept(id="c-3", name="Sampling Distribution"),
            Concept(id="c-4", name="Random Sample"),
        ],
        relations=relations if relations is not None else [
            make_relation("Hypothesis Testing", "Null Hypothesis"),
            make_relation("Sampling Distribution", "Random Sample"),
        ],
    )


def test_scorer_composite_without_embeddings_and_graph():
    scorer = ConfidenceScorer()
    relation = make_relation("Hypothesis Testing", "Null Hypothesis")
    score = scorer.score_relation(relation, CHUNKS, concept_embeddings=None, graph=None)
    assert 0.0 <= score <= 1.0


def test_scorer_composite_with_graph():
    scorer = ConfidenceScorer()
    graph = nx.DiGraph()
    graph.add_node("a", name="Hypothesis Testing", pagerank=0.6)
    graph.add_node("b", name="Null Hypothesis", pagerank=0.4)
    relation = make_relation("Hypothesis Testing", "Null Hypothesis")
    score = scorer.score_relation(relation, CHUNKS, graph=graph)
    assert 0.0 <= score <= 1.0


def test_scorer_weights_are_normalized():
    scorer = ConfidenceScorer(weights={"llm_confidence": 1.0, "cooccurrence": 1.0})
    assert abs(sum(scorer.weights.values()) - 1.0) < 0.01


def test_scorer_score_breakdown():
    scorer = ConfidenceScorer()
    relation = make_relation("Hypothesis Testing", "Null Hypothesis")
    breakdown = scorer.get_score_breakdown(relation, CHUNKS)
    assert breakdown["final_score"] > 0
    assert "llm_confidence" in breakdown["components"]
    assert "cooccurrence" in breakdown["components"]


def test_detector_flags_unsupported_relation():
    detector = HallucinationDetector()
    knowledge_map = make_knowledge_map(
        relations=[make_relation("Ghost Concept", "Null Hypothesis")]
    )
    report = detector.detect(knowledge_map, CHUNKS)
    flagged_types = [flag.flag_type for flag in report.flags]
    assert FlagType.MISSING_SOURCE in flagged_types


def test_detector_flags_low_confidence():
    detector = HallucinationDetector(confidence_threshold=0.5)
    knowledge_map = make_knowledge_map(
        relations=[make_relation("Hypothesis Testing", "Null Hypothesis", confidence=0.2)]
    )
    report = detector.detect(knowledge_map, CHUNKS, confidence_scores={
        knowledge_map.relations[0].id: 0.2,
    })
    assert any(flag.flag_type == FlagType.LOW_CONFIDENCE for flag in report.flags)


def test_detector_flags_circular_prerequisite():
    detector = HallucinationDetector()
    knowledge_map = make_knowledge_map(
        relations=[
            make_relation("Hypothesis Testing", "Null Hypothesis", relation_type=RelationType.PREREQUISITE),
            make_relation("Null Hypothesis", "Hypothesis Testing", relation_type=RelationType.PREREQUISITE),
        ]
    )
    report = detector.detect(knowledge_map, CHUNKS)
    assert any(flag.flag_type == FlagType.CONTRADICTION for flag in report.flags)


def test_detector_risk_score_bounds():
    detector = HallucinationDetector()
    knowledge_map = make_knowledge_map(
        relations=[make_relation("Ghost Concept", "Null Hypothesis")]
    )
    report = detector.detect(knowledge_map, CHUNKS)
    assert 0.0 <= report.overall_risk_score <= 1.0
    assert report.total_relations == 1
