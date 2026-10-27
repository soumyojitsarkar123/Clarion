"""Regression tests for Milestone 3 deterministic fallback extraction quality.

Ensures that:
1. Obvious noise tokens (stopwords, code variable tokens, repetitive n-grams) are rejected.
2. Near-duplicate concepts (singular/plural) are deduplicated.
3. Relations require sentence-level evidence and reject negation patterns.
4. Summary candidate selection filters editorial/proceedings boilerplate.
5. Valid domain concepts remain preserved without collateral damage.
"""

import pytest
from models.chunk import Chunk
from models.knowledge_map import Concept, RelationType
from services.knowledge_map_service import KnowledgeMapService
from services.summary_service import SummaryService


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    words = text.split()
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-test",
        section_title=None,
        content=text,
        position_index=0,
        word_count=len(words),
    )


def test_reject_stopwords_and_generic_unigrams(isolated_settings):
    """Ensure stopwords like 'more', 'there', 'then', 'such', 'what' are rejected as concepts."""
    service = KnowledgeMapService()

    noisy_unigrams = ["more", "there", "then", "such", "what", "proceedings", "table", "figure"]
    for word in noisy_unigrams:
        assert service._is_generic_unigram(word) is True
        assert service._is_valid_concept_name(word) is False


def test_reject_code_tokens_and_malformed_ngrams(isolated_settings):
    """Ensure code tokens and malformed n-grams like 'Row Col' and 'Edge Dist Edge' are rejected."""
    service = KnowledgeMapService()

    # Code variable phrases
    assert service._is_valid_concept_name("Row Col") is False
    assert service._is_valid_concept_name("dist") is False
    assert service._is_valid_concept_name("Edge Dist Edge") is False
    assert service._is_valid_concept_name("Dist Edge Dist") is False
    assert service._is_valid_concept_name("idx ptr") is False
    assert service._is_valid_concept_name("min cost arr") is False

    # Repetitive n-grams (e.g. repeated words)
    assert service._is_valid_concept_name("node node") is False
    assert service._is_valid_concept_name("path shortest path") is False


def test_deduplicate_singular_and_plural_concepts(isolated_settings):
    """Ensure singular/plural near-duplicates ('Edge' vs 'Edges') are identified as redundant."""
    service = KnowledgeMapService()

    # If "Edges" is already selected, "Edge" should be redundant
    assert service._is_redundant_concept("Edge", ["Edges"]) is True
    # If "Edge" is already selected, "Edges" should be redundant
    assert service._is_redundant_concept("Edges", ["Edge"]) is True
    # If "Spanning Trees" is selected, "Spanning Tree" should be redundant
    assert service._is_redundant_concept("Spanning Tree", ["Spanning Trees"]) is True


def test_valid_domain_concepts_preserved(isolated_settings):
    """Ensure legitimate domain concepts are not filtered out."""
    service = KnowledgeMapService()

    valid_concepts = [
        "hypothesis testing",
        "sampling distribution",
        "shortest path",
        "dynamic programming",
        "spanning tree",
        "binary search",
        "gradient descent",
    ]
    for concept in valid_concepts:
        assert service._is_valid_concept_name(concept) is True, f"Failed for {concept}"


def test_cooccurrence_relations_require_sentence_evidence(isolated_settings):
    """Relations should not be emitted if concepts only co-occur across chunk boundaries without a shared sentence."""
    service = KnowledgeMapService()

    # Two concepts in different sentences in the same chunk without semantic indicator
    chunk_text = (
        "In this chapter we discuss binary search. "
        "Separately in the laboratory we observed significant heating."
    )
    chunks = [_make_chunk("c1", chunk_text)]
    concepts = [
        Concept(id="1", name="Binary Search", chunk_ids=["c1"]),
        Concept(id="2", name="Heating", chunk_ids=["c1"]),
    ]
    relations = service._detect_relations_by_cooccurrence(concepts, chunks)
    # Binary Search and Heating do not share a sentence
    assert len(relations) == 0


def test_cooccurrence_relations_reject_negated_evidence(isolated_settings):
    """Sentences with explicit negation between concepts must not emit affirmative candidate relations."""
    service = KnowledgeMapService()

    chunk_text = (
        "It is important to remember that binary search is not related to gradient descent."
    )
    chunks = [_make_chunk("c1", chunk_text)]
    concepts = [
        Concept(id="1", name="Binary Search", chunk_ids=["c1"]),
        Concept(id="2", name="Gradient Descent", chunk_ids=["c1"]),
    ]
    relations = service._detect_relations_by_cooccurrence(concepts, chunks)
    assert len(relations) == 0


def test_cooccurrence_relations_infer_typed_patterns(isolated_settings):
    """Sentence with structural indicator like 'is a component of' should yield a typed relation."""
    service = KnowledgeMapService()

    chunk_text = (
        "The priority queue is a component of Dijkstra algorithm used for vertex selection."
    )
    chunks = [_make_chunk("c1", chunk_text)]
    concepts = [
        Concept(id="1", name="Priority Queue", chunk_ids=["c1"]),
        Concept(id="2", name="Dijkstra Algorithm", chunk_ids=["c1"]),
    ]
    relations = service._detect_relations_by_cooccurrence(concepts, chunks)
    assert len(relations) == 1
    rel = relations[0]
    assert rel.relation_type == RelationType.EXPLANATION
    assert "part-of" in rel.description
    assert rel.evidence_text is not None
    assert "priority queue is a component of dijkstra algorithm" in rel.evidence_text.lower()


def test_summary_filters_boilerplate_and_proceedings(isolated_settings):
    """Summary candidate extraction should filter acknowledgment and proceedings boilerplate."""
    service = SummaryService()

    boilerplate_sentences = [
        "Proceedings of the 34th International Conference on Machine Learning.",
        "All rights reserved by the publisher in 2024.",
        "The authors thank the following reviewers for their constructive feedback.",
        "Abbreviations are used throughout the text according to standard conventions.",
    ]
    for s in boilerplate_sentences:
        assert service._looks_like_acknowledgment(s.lower()) is True

    technical_sentences = [
        "Dynamic programming solves problems by combining the solutions to subproblems.",
        "A shortest path between two vertices in a weighted graph minimizes the sum of edge weights.",
    ]
    for s in technical_sentences:
        assert service._looks_like_acknowledgment(s.lower()) is False
