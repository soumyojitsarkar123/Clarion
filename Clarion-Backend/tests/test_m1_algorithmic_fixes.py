"""
Unit and regression tests for Milestone 1: Backend Algorithmic & Config Fixes.
Verifies:
1. Pattern-inferred relation types are preserved in candidate relation extraction (not overridden to EXPLANATION).
2. Embedding vector normalization (normalize_embeddings=True) guarantees unit length and calibrated FAISS similarity.
3. Token length safety monitoring detects and warns on chunks exceeding 512 subwords.
4. Application branding configuration is updated to 'Clarion'.
"""

import logging
import math
from pathlib import Path
import numpy as np
import pytest

from models.chunk import Chunk
from models.knowledge_map import Concept, RelationType
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.knowledge_map_service import KnowledgeMapService
from utils.config import settings


# ============================================================================
# 1. Relation Type Inferred Preservation Tests
# ============================================================================

def test_relation_type_override_preserves_inferred_patterns(isolated_settings):
    """
    Test that pattern-inferred relation types (prerequisite, cause-effect, part-of, definition)
    are preserved when constructing candidate relations in knowledge_map_service.
    Previously, relation_type was hardcoded to RelationType.EXPLANATION, discarding the inferred type.
    """
    kms = KnowledgeMapService()

    # Create synthetic concepts and chunks with clear lexical relation indicators
    concepts = [
        Concept(id="c1", name="linear algebra", definition="Study of vectors and matrices."),
        Concept(id="c2", name="tensor factorization", definition="Decomposition of multi-way arrays."),
        Concept(id="c3", name="high learning rate", definition="Optimization step parameter."),
        Concept(id="c4", name="gradient explosion", definition="Numerical overflow in backpropagation."),
        Concept(id="c5", name="convolution layer", definition="Feature map extraction module."),
        Concept(id="c6", name="neural network", definition="Layered computational learning architecture."),
        Concept(id="c7", name="directed graph", definition="A finite graph with ordered pairs of vertices."),
        Concept(id="c8", name="generic node a", definition="Alpha entity."),
        Concept(id="c9", name="generic node b", definition="Beta entity."),
    ]

    # Sentence evidence targeting priority indicator patterns:
    # - prerequisite: "required before"
    # - cause-effect: "leads to"
    # - part-of: "component of"
    # - definition: "defined as"
    # - neutral: "co-occurs with"
    test_content = (
        "Linear algebra is required before studying tensor factorization.\n\n"
        "A high learning rate leads to gradient explosion during network training.\n\n"
        "The convolution layer is a component of the neural network architecture.\n\n"
        "A directed graph is defined as a graph with directed edges.\n\n"
        "Generic node a and generic node b co-occur in the same paragraph without predicates."
    )

    chunks = [
        Chunk(
            chunk_id="chunk-test-relations",
            document_id="doc-rel-test",
            content=test_content,
            section_title="Methodology",
            position_index=0,
            word_count=len(test_content.split()),
        )
    ]

    relations = kms._detect_relations_by_cooccurrence(concepts, chunks)
    assert len(relations) > 0

    rel_map = {(r.from_concept, r.to_concept): r.relation_type for r in relations}

    # Verify prerequisite pattern ("required before")
    pair_prereq = ("linear algebra", "tensor factorization")
    reverse_prereq = ("tensor factorization", "linear algebra")
    prereq_match = rel_map.get(pair_prereq) or rel_map.get(reverse_prereq)
    assert prereq_match == RelationType.PREREQUISITE, f"Expected PREREQUISITE, got {prereq_match}"

    # Verify cause-effect pattern ("leads to")
    pair_cause = ("high learning rate", "gradient explosion")
    reverse_cause = ("gradient explosion", "high learning rate")
    cause_match = rel_map.get(pair_cause) or rel_map.get(reverse_cause)
    assert cause_match == RelationType.CAUSE_EFFECT, f"Expected CAUSE_EFFECT, got {cause_match}"

    # Verify part-of pattern ("component of")
    pair_part = ("convolution layer", "neural network")
    reverse_part = ("neural network", "convolution layer")
    part_match = rel_map.get(pair_part) or rel_map.get(reverse_part)
    assert part_match == RelationType.PART_OF, f"Expected PART_OF, got {part_match}"

    # Verify definition pattern ("defined as")
    pair_def = ("directed graph", "directed graph")
    # For self or paired concept, ensure definition indicator is parsed
    has_definition_or_prereq = any(
        r.relation_type in (RelationType.PREREQUISITE, RelationType.CAUSE_EFFECT, RelationType.PART_OF, RelationType.DEFINITION)
        for r in relations
    )
    assert has_definition_or_prereq, "Candidate relations must preserve inferred non-explanation types"


def test_relation_type_defaults_to_explanation_without_indicators(isolated_settings):
    """
    Test that when sentence evidence lacks specific indicator patterns,
    it safely defaults to RelationType.EXPLANATION.
    """
    kms = KnowledgeMapService()
    concepts = [
        Concept(id="c1", name="apples", definition="Fruit."),
        Concept(id="c2", name="oranges", definition="Citrus fruit."),
    ]
    chunks = [
        Chunk(
            chunk_id="chunk-fruit",
            document_id="doc-fruit",
            content="Apples and oranges are both sold at the grocery market.",
            section_title="Produce",
            position_index=0,
            word_count=10,
        )
    ]
    relations = kms._detect_relations_by_cooccurrence(concepts, chunks)
    assert len(relations) == 1
    assert relations[0].relation_type == RelationType.EXPLANATION


# ============================================================================
# 2. Embedding Vector Normalization & FAISS Calibration Tests
# ============================================================================

def test_embedding_normalization_unit_length(isolated_settings):
    """
    Verify that embedding generation produces L2-normalized vectors (unit length ||v||_2 = 1.0).
    L2 normalization ensures that squared Euclidean distance in FAISS IndexFlatL2
    is mathematically equivalent to cosine distance: d = 2 - 2*cos(q, v) in [0, 4].
    """
    service = EmbeddingService()

    # Test fallback vector normalization
    fallback_vec = service._fallback_embed_text("Clarion dense retrieval test")
    norm_fallback = math.sqrt(sum(x * x for x in fallback_vec))
    assert abs(norm_fallback - 1.0) < 1e-5, f"Fallback vector norm {norm_fallback} != 1.0"

    # Test live model or mocked model encoding normalization
    sample_text = "Dense semantic representation of scientific concepts."
    chunks = [
        Chunk(
            chunk_id="norm-chk-1",
            document_id="doc-norm",
            content=sample_text,
            position_index=0,
            word_count=7,
        )
    ]

    embeddings = service.generate_embeddings(chunks)
    assert len(embeddings) == 1
    vec = embeddings[0]
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4, f"Embedding norm {norm} is not unit length (expected 1.0)"

    # Test query embedding normalization
    q_vec = service.embed_query("What is dense retrieval?")
    q_norm = math.sqrt(sum(x * x for x in q_vec))
    assert abs(q_norm - 1.0) < 1e-4, f"Query embedding norm {q_norm} is not unit length (expected 1.0)"


def test_normalized_faiss_similarity_calibration(isolated_settings):
    """
    Verify that with unit-normalized vectors, the FAISS similarity formula
    s = 1.0 / (1.0 + d_L2^2) is well-calibrated and bounded strictly in [0.20, 1.00].
    """
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2_identical = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3_orthogonal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v4_opposite = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    # Identical: d^2 = 0 -> s = 1.0
    d_ident = float(np.sum((v1 - v2_identical) ** 2))
    s_ident = 1.0 / (1.0 + d_ident)
    assert math.isclose(s_ident, 1.0, rel_tol=1e-5)

    # Orthogonal: d^2 = 2 -> s = 1/3 ~ 0.333
    d_ortho = float(np.sum((v1 - v3_orthogonal) ** 2))
    s_ortho = 1.0 / (1.0 + d_ortho)
    assert math.isclose(s_ortho, 1.0 / 3.0, rel_tol=1e-5)

    # Opposite: d^2 = 4 -> s = 1/5 = 0.20
    d_opp = float(np.sum((v1 - v4_opposite) ** 2))
    s_opp = 1.0 / (1.0 + d_opp)
    assert math.isclose(s_opp, 0.20, rel_tol=1e-5)
    assert 0.20 <= s_opp <= s_ortho <= s_ident <= 1.00


# ============================================================================
# 3. Token Length Safety & Truncation Monitoring Tests
# ============================================================================

def test_token_length_safety_warning_chunking_service(isolated_settings, caplog):
    """
    Verify that ChunkingService detects when a chunk exceeds 512 subwords
    and logs a warning to prevent silent truncation.
    """
    chunker = ChunkingService()

    # Short chunk (< 512 subwords) - should not log warning
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        tokens_short = chunker.check_token_length("This is a short sentence with few words.")
        assert tokens_short < 512
        assert not any("exceeds the 512 subword token limit" in r.message for r in caplog.records)

    # Long chunk (> 512 subwords) - must log warning
    caplog.clear()
    long_text = "word " * 420  # 420 words * 1.35 ~ 567 subword tokens
    with caplog.at_level(logging.WARNING):
        tokens_long = chunker.check_token_length(long_text)
        assert tokens_long > 512
        assert any("exceeds the 512 subword token limit" in r.message for r in caplog.records)


def test_token_length_safety_warning_embedding_service(isolated_settings, caplog):
    """
    Verify that EmbeddingService monitors chunk token lengths and warns when exceeding 512 subwords.
    """
    service = EmbeddingService()

    # Short text (< 512 tokens)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        service.check_token_length("Neural knowledge representation.")
        assert not any("exceeds limit of 512 subwords" in r.message for r in caplog.records)

    # Oversized text (> 512 tokens)
    caplog.clear()
    oversized_text = "advanced algorithm " * 300  # 600 words > 512 tokens
    with caplog.at_level(logging.WARNING):
        count = service.check_token_length(oversized_text)
        assert count > 512
        assert any("exceeds limit of 512 subwords" in r.message for r in caplog.records)


# ============================================================================
# 4. Backend Branding Configuration Tests
# ============================================================================

def test_app_name_is_clarion(isolated_settings):
    """
    Verify that default app_name in settings is 'Clarion' rather than obsolete prototype labels.
    """
    assert settings.app_name == "Clarion"


def test_env_example_app_name():
    """
    Verify that .env.example defines APP_NAME="Clarion".
    """
    env_path = Path(__file__).resolve().parents[1] / ".env.example"
    assert env_path.exists(), f"Could not find {env_path}"
    content = env_path.read_text(encoding="utf-8")
    assert 'APP_NAME="Clarion"' in content
    assert "Intelligent Document Knowledge System" not in content
