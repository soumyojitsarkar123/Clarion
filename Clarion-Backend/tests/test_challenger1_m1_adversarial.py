"""
Challenger 1 Empirical and Adversarial Verification Suite for Milestone 1.

Targets:
1. Adversarially test relation type preservation in knowledge_map_service.py:
   test sentences with each pattern indicator ('component of', 'leads to', 'prerequisite for',
   'such as', 'derived from', 'defined as') to verify they produce the corresponding RelationType
   and never collapse into EXPLANATION unless no pattern matched.
2. Adversarially test embedding normalization in embedding_service.py:
   verify numpy vectors have unit norm (||v||_2 = 1.0) and verify FAISS distance monotonicity
   and bounded distance-derived ranking score s in [0.20, 1.00].
3. Adversarially test token length safety monitoring in chunking_service.py and embedding_service.py
   for texts > 512 subwords.
4. Verify config.py app_name is 'Clarion'.
"""

import asyncio
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
from utils.config import Settings, settings


# ============================================================================
# 1. Adversarial Relation Type Preservation
# ============================================================================

class TestRelationTypePreservationAdversarial:
    """
    Adversarially challenge relation type preservation in KnowledgeMapService.
    Ensures lexical indicator patterns map to distinct non-EXPLANATION RelationTypes.
    """

    @pytest.mark.parametrize(
        "indicator_phrase,expected_rel_type",
        [
            ("is a component of", RelationType.PART_OF),
            ("part of", RelationType.PART_OF),
            ("leads to", RelationType.CAUSE_EFFECT),
            ("causes", RelationType.CAUSE_EFFECT),
            ("is a prerequisite for", RelationType.PREREQUISITE),
            ("is required before", RelationType.PREREQUISITE),
            ("such as", RelationType.EXAMPLE_OF),
            ("for example", RelationType.EXAMPLE_OF),
            ("is derived from", RelationType.DERIVES_FROM),
            ("calculated from", RelationType.DERIVES_FROM),
            ("is defined as", RelationType.DEFINITION),
            ("formally defined as", RelationType.DEFINITION),
        ],
    )
    def test_individual_indicator_patterns_produce_correct_relation_types(
        self, isolated_settings, indicator_phrase, expected_rel_type
    ):
        """
        Verify each required pattern indicator ('component of', 'leads to', 'prerequisite for',
        'such as', 'derived from', 'defined as') produces its exact corresponding RelationType.
        """
        kms = KnowledgeMapService()
        concept_a = Concept(id="c_alpha", name="Concept Alpha", definition="First entity.")
        concept_b = Concept(id="c_beta", name="Concept Beta", definition="Second entity.")

        content = f"Concept Alpha {indicator_phrase} Concept Beta in this theoretical formulation."
        chunk = Chunk(
            chunk_id="chk-indicator-test",
            document_id="doc-ind-test",
            content=content,
            section_title="Architecture",
            position_index=0,
            word_count=len(content.split()),
        )

        relations = kms._detect_relations_by_cooccurrence([concept_a, concept_b], [chunk])
        assert len(relations) >= 1, f"No relations extracted for phrase '{indicator_phrase}'"
        rel = relations[0]
        assert rel.relation_type == expected_rel_type, (
            f"Expected {expected_rel_type} for '{indicator_phrase}', but got {rel.relation_type}"
        )
        assert rel.relation_type != RelationType.EXPLANATION, (
            f"Relation type collapsed into EXPLANATION for indicator '{indicator_phrase}'!"
        )

    def test_neutral_sentence_defaults_to_explanation(self, isolated_settings):
        """
        Adversarially verify that when no lexical pattern matches,
        the relation type safely defaults to EXPLANATION.
        """
        kms = KnowledgeMapService()
        concept_a = Concept(id="c_alpha", name="Quantum State", definition="State vector.")
        concept_b = Concept(id="c_beta", name="Hilbert Space", definition="Complete inner product space.")

        # Neutral sentence with co-occurrence but no indicator pattern
        content = "Quantum State and Hilbert Space appear in modern physics literature."
        chunk = Chunk(
            chunk_id="chk-neutral-test",
            document_id="doc-neutral",
            content=content,
            section_title="Background",
            position_index=0,
            word_count=len(content.split()),
        )

        relations = kms._detect_relations_by_cooccurrence([concept_a, concept_b], [chunk])
        assert len(relations) == 1
        assert relations[0].relation_type == RelationType.EXPLANATION

    def test_priority_ordering_when_multiple_patterns_present(self, isolated_settings):
        """
        Test deterministic priority ordering when multiple pattern keywords appear.
        Priority: part-of > cause-effect > prerequisite > example-of > derives-from > similar-to > definition
        """
        kms = KnowledgeMapService()
        c1 = Concept(id="c1", name="Alpha", definition="A")
        c2 = Concept(id="c2", name="Beta", definition="B")

        # Sentence contains both 'is a component of' (part-of) and 'leads to' (cause-effect)
        content = "Alpha is a component of Beta which leads to system stability."
        chunk = Chunk(
            chunk_id="chk-priority-1",
            document_id="doc-pri-1",
            content=content,
            section_title="Design",
            position_index=0,
            word_count=len(content.split()),
        )
        relations = kms._detect_relations_by_cooccurrence([c1, c2], [chunk])
        assert len(relations) >= 1
        assert relations[0].relation_type == RelationType.PART_OF

    def test_case_insensitivity_and_punctuation_handling(self, isolated_settings):
        """
        Adversarially test mixed case, upper case, and punctuation around indicators.
        """
        kms = KnowledgeMapService()
        c1 = Concept(id="c1", name="Kernel Module", definition="OS component.")
        c2 = Concept(id="c2", name="Operating System", definition="System software.")

        content = "KERNEL MODULE IS A COMPONENT OF OPERATING SYSTEM."
        chunk = Chunk(
            chunk_id="chk-case-test",
            document_id="doc-case",
            content=content,
            section_title="Kernel",
            position_index=0,
            word_count=len(content.split()),
        )
        relations = kms._detect_relations_by_cooccurrence([c1, c2], [chunk])
        assert len(relations) >= 1
        assert relations[0].relation_type == RelationType.PART_OF


# ============================================================================
# 2. Adversarial Embedding Normalization & FAISS Calibration
# ============================================================================

class TestEmbeddingNormalizationAndFAISSAdversarial:
    """
    Adversarially challenge embedding vector normalization and FAISS metric calibration.
    """

    def test_fallback_embeddings_unit_norm(self, isolated_settings):
        """
        Verify that fallback embeddings consistently have ||v||_2 = 1.0 across diverse inputs.
        """
        service = EmbeddingService()
        test_strings = [
            "Clarion dense retrieval test",
            "Short",
            "A very long sentence with numerous words repeated over and over again to test hash distribution.",
            "Special characters: !@#$%^&*()_+=-~`{}[]|:;'<>,.?/",
            "1234567890 numbers and equations E = mc^2",
        ]
        for s in test_strings:
            vec = service._fallback_embed_text(s)
            norm = float(np.linalg.norm(np.array(vec, dtype=np.float32)))
            assert abs(norm - 1.0) < 1e-5, f"Fallback vector norm {norm} != 1.0 for string: {s[:20]}"

    def test_live_embedding_model_unit_norm(self, isolated_settings):
        """
        Verify that live SentenceTransformer model produces strictly unit-length vectors.
        """
        service = EmbeddingService()
        chunks = [
            Chunk(chunk_id="c1", document_id="d1", content="First technical chunk.", position_index=0, word_count=3),
            Chunk(chunk_id="c2", document_id="d1", content="Second mathematical formulation.", position_index=1, word_count=3),
            Chunk(chunk_id="c3", document_id="d1", content="Third experimental evaluation result.", position_index=2, word_count=4),
        ]
        embeddings = service.generate_embeddings(chunks)
        assert len(embeddings) == 3
        for idx, emb in enumerate(embeddings):
            v = np.array(emb, dtype=np.float32)
            norm = float(np.linalg.norm(v))
            assert abs(norm - 1.0) < 1e-4, f"Chunk {idx} embedding norm {norm} != 1.0"

    def test_query_embedding_unit_norm(self, isolated_settings):
        """
        Verify that query embeddings have unit norm.
        """
        service = EmbeddingService()
        queries = [
            "What is the architecture of Clarion?",
            "How does FAISS IndexFlatL2 compute distance?",
            "Why is embedding normalization required?",
        ]
        for q in queries:
            q_vec = np.array(service.embed_query(q), dtype=np.float32)
            norm = float(np.linalg.norm(q_vec))
            assert abs(norm - 1.0) < 1e-4, f"Query norm {norm} != 1.0 for query '{q}'"

    def test_async_embedding_unit_norm(self, isolated_settings):
        """
        Verify that asynchronous batch embedding generation enforces unit norm.
        """
        service = EmbeddingService()
        chunks = [
            Chunk(chunk_id="a1", document_id="d1", content="Async document chunk alpha.", position_index=0, word_count=4),
            Chunk(chunk_id="a2", document_id="d1", content="Async document chunk beta.", position_index=1, word_count=4),
            Chunk(chunk_id="a3", document_id="d1", content="Async document chunk gamma.", position_index=2, word_count=4),
        ]
        embeddings = asyncio.run(service.generate_embeddings_async(chunks))
        assert len(embeddings) == 3
        for idx, emb in enumerate(embeddings):
            v = np.array(emb, dtype=np.float32)
            norm = float(np.linalg.norm(v))
            assert abs(norm - 1.0) < 1e-4, f"Async chunk {idx} norm {norm} != 1.0"

    def test_faiss_squared_l2_monotonicity_and_bound_invariance(self):
        """
        Mathematical invariance proof and empirical check:
        For unit vectors u, v (||u|| = ||v|| = 1):
        ||u - v||^2 = ||u||^2 + ||v||^2 - 2 * <u, v> = 2 - 2 * cos(theta) in [0, 4].
        Similarity s = 1 / (1 + d^2) must be strictly monotonic and bounded in [0.20, 1.00].
        """
        # Step through angles theta from 0 to pi (cos from 1.0 down to -1.0)
        angles = np.linspace(0.0, np.pi, 100)
        prev_d2 = -1.0
        prev_s = 2.0

        for theta in angles:
            u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            v = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float32)

            d2 = float(np.sum((u - v) ** 2))
            s = 1.0 / (1.0 + d2)

            # Monotonicity checks
            assert d2 >= prev_d2 - 1e-6, f"Distance not monotonically increasing: {d2} < {prev_d2}"
            assert s <= prev_s + 1e-6, f"Similarity not monotonically decreasing: {s} > {prev_s}"

            # Strict calibration bounds
            assert 0.20 - 1e-6 <= s <= 1.00 + 1e-6, f"Similarity {s} outside [0.20, 1.00]"

            prev_d2 = d2
            prev_s = s

        # Boundary checks
        # Angle 0 (identical)
        assert math.isclose(1.0 / (1.0 + 0.0), 1.00, abs_tol=1e-5)
        # Angle pi/2 (orthogonal)
        assert math.isclose(1.0 / (1.0 + 2.0), 1.0 / 3.0, abs_tol=1e-5)
        # Angle pi (opposite)
        assert math.isclose(1.0 / (1.0 + 4.0), 0.20, abs_tol=1e-5)

    def test_random_spherical_fuzzing_calibration(self):
        """
        Fuzz test: 500 pairs of random points on unit 1024-sphere (BGE-Large dimension).
        All similarities must lie strictly in [0.20, 1.00].
        """
        np.random.seed(42)
        dim = 1024
        num_pairs = 500

        u_vecs = np.random.randn(num_pairs, dim).astype(np.float32)
        u_vecs /= np.linalg.norm(u_vecs, axis=1, keepdims=True)

        v_vecs = np.random.randn(num_pairs, dim).astype(np.float32)
        v_vecs /= np.linalg.norm(v_vecs, axis=1, keepdims=True)

        d2 = np.sum((u_vecs - v_vecs) ** 2, axis=1)
        sims = 1.0 / (1.0 + d2)

        assert np.all(d2 >= 0.0)
        assert np.all(d2 <= 4.0001)
        assert np.all(sims >= 0.20 - 1e-5)
        assert np.all(sims <= 1.00 + 1e-5)


# ============================================================================
# 3. Adversarial Token Length Safety Monitoring
# ============================================================================

class TestTokenLengthSafetyMonitoringAdversarial:
    """
    Adversarially challenge token length checking in chunking and embedding services.
    Verifies warnings are logged when texts exceed 512 subwords, preventing silent truncation.
    """

    def test_chunking_service_boundary_token_monitoring(self, isolated_settings, caplog):
        """
        Test exact boundary conditions around 512 subwords in ChunkingService.
        """
        chunker = ChunkingService()

        # Under limit: 350 words * 1.35 = 472 subwords -> No warning
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            t_under = chunker.check_token_length("word " * 350)
            assert t_under < 512
            assert not any("exceeds the 512 subword token limit" in r.message for r in caplog.records)

        # Over limit: 400 words * 1.35 = 540 subwords -> Warning logged
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            t_over = chunker.check_token_length("word " * 400)
            assert t_over > 512
            assert any("exceeds the 512 subword token limit" in r.message for r in caplog.records)

    def test_embedding_service_token_monitoring(self, isolated_settings, caplog):
        """
        Test token length monitoring in EmbeddingService for short vs oversized texts.
        """
        service = EmbeddingService()

        # Short text: No warning
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            service.check_token_length("Concise concept description.")
            assert not any("exceeds limit of 512 subwords" in r.message for r in caplog.records)

        # Oversized text: Warning logged
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            service.check_token_length("scientific observation " * 300)
            assert any("exceeds limit of 512 subwords" in r.message for r in caplog.records)

    def test_chunk_document_logs_warning_on_oversized_section(self, isolated_settings, caplog):
        """
        Verify that chunk_document triggers warning when creating chunks from large text.
        """
        isolated_settings.chunk_size = 512
        chunker = ChunkingService()
        oversized_text = "Dense academic prose with mathematical formulations and equations. " * 80  # ~640 words
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            chunks = chunker.chunk_document("doc-test-overflow", oversized_text)
            assert len(chunks) >= 1
            # Check that token length check was invoked during chunk_document
            has_warning = any("exceeds the 512 subword token limit" in r.message for r in caplog.records)
            assert has_warning, "Expected token length warning during chunking of oversized document"


# ============================================================================
# 4. Configuration Branding Verification
# ============================================================================

class TestConfigBrandingAdversarial:
    """
    Adversarially verify product name and branding configuration.
    """

    def test_default_settings_app_name_is_clarion(self):
        """
        Verify that default Settings instance has app_name == 'Clarion'.
        """
        s = Settings()
        assert s.app_name == "Clarion", f"Expected app_name 'Clarion', got '{s.app_name}'"

    def test_global_settings_app_name_is_clarion(self):
        """
        Verify that imported global settings object has app_name == 'Clarion'.
        """
        assert settings.app_name == "Clarion"

    def test_env_example_contains_clarion_and_no_obsolete_name(self):
        """
        Verify .env.example strictly contains APP_NAME="Clarion" and no prototype remnants.
        """
        env_file = Path(__file__).resolve().parents[1] / ".env.example"
        assert env_file.exists(), ".env.example does not exist"
        content = env_file.read_text(encoding="utf-8")
        assert 'APP_NAME="Clarion"' in content
        assert "Intelligent Document Knowledge System" not in content
