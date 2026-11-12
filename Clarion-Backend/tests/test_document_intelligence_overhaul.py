"""
Comprehensive unit and regression tests for Document Intelligence Overhaul (M1).
Tests chunking, front-matter bypassing, roster rejection, candidate relations,
multi-signal document classification, grounded summaries, LLM states, and embedding caching.
"""

import pytest
import sqlite3
import struct
from models.chunk import Chunk
from models.knowledge_map import KnowledgeMap, RelationType
from models.summary import StructuredSummary
from core.llm.base import LLMRuntimeState
from services.chunking_service import ChunkingService
from services.knowledge_map_service import KnowledgeMapService
from services.summary_service import SummaryService
from services.embedding_service import EmbeddingService
from services.pipeline_observability import PipelineStateTracker


# ============================================================================
# 1. Chunking Engine Overhaul Tests
# ============================================================================

def test_heading_detection_rejects_running_headers(isolated_settings):
    """Running headers repeated frequently throughout the document must not be treated as section headings."""
    service = ChunkingService()
    # "ADVANCED MACHINE LEARNING" appears on every page as a running header
    repeated_header = "ADVANCED MACHINE LEARNING"
    text = f"""
# Introduction
This is the opening text of the chapter.

{repeated_header}
Here is page one text.

{repeated_header}
Here is page two text.

{repeated_header}
Here is page three text.

{repeated_header}
Here is page four text.

# Real Chapter Section
Here is actual new content under a legitimate heading.
"""
    headings = service._detect_headings(text)
    detected_titles = [h[0] for h in headings]
    assert "Introduction" in detected_titles
    assert "Real Chapter Section" in detected_titles
    assert repeated_header not in detected_titles


def test_heading_detection_rejects_boilerplate_and_sentences(isolated_settings):
    """Page markers, copyright notices, and sentences with punctuation are not headings."""
    service = ChunkingService()
    text = """
Page 15 of 320
Copyright 2024 by Clarion Academic Press. All rights reserved.
1. This is a list item that ends with a period.
# Genuine Section
This is body text.
"""
    headings = service._detect_headings(text)
    detected_titles = [h[0] for h in headings]
    assert "Genuine Section" in detected_titles
    for title in detected_titles:
        assert "Page 15" not in title
        assert "Copyright" not in title
        assert "list item" not in title


def test_bidirectional_undersized_chunk_merging(isolated_settings):
    """Undersized chunks (<100 and <20 words) must never remain stranded in multi-chunk documents."""
    isolated_settings.chunk_size = 120
    isolated_settings.min_chunk_size = 50
    service = ChunkingService()

    # Section 1 has 10 words, Section 2 has 120 words, Section 3 has 8 words
    text = (
        "# Section A\n"
        + " ".join(["short"] * 10)
        + "\n\n# Section B\n"
        + " ".join(["body"] * 120)
        + "\n\n# Section C\n"
        + " ".join(["tail"] * 8)
    )
    chunks = service.chunk_document("doc-merge", text)
    assert len(chunks) >= 1
    for chunk in chunks:
        # Every chunk should be at least min_chunk_size or successfully absorbed
        assert chunk.word_count >= 50 or len(chunks) == 1


# ============================================================================
# 2. Multi-Signal Document Classification Tests
# ============================================================================

def test_classify_textbook_with_exercises(isolated_settings):
    """A textbook with practice exercises and question marks must be classified as textbook, not assessment."""
    summary_service = SummaryService()
    textbook_text = """
Chapter 1: Foundations of Artificial Intelligence
This chapter covers propositional logic, search algorithms, and heuristic estimation.
Table of Contents:
Chapter 1: Logic and Search
Chapter 2: Knowledge Representation
Chapter 3: Probabilistic Reasoning
Preface:
This textbook is intended for graduate students.
Exercises:
1. What is the time complexity of A* search?
2. How does alpha-beta pruning improve minimax?
3. Compare depth-first and breadth-first search.
What are the heuristic requirements?
""" + " expository content explaining logic and probability" * 200

    doc_type = summary_service.classify_document(text=textbook_text)
    assert doc_type == "textbook"


def test_classify_actual_exam(isolated_settings):
    """An examination paper with instructions and marks must be classified as assessment."""
    summary_service = SummaryService()
    exam_text = """
Final Examination - Computer Science 301
Time Allowed: 3 Hours
Maximum Marks: 100
Instructions to candidates:
Answer any four questions. Each question carries equal marks.
1. What is the difference between TCP and UDP? (10 marks)
2. Explain Dijkstra's shortest path algorithm. (15 marks)
3. Describe the ACID properties of database transactions. (15 marks)
4. What is virtual memory and paging? (10 marks)
"""
    doc_type = summary_service.classify_document(text=exam_text)
    assert doc_type == "assessment"


def test_classify_technical_documentation(isolated_settings):
    """Technical docs with installation, API reference, parameters and usage must be classified as technical_documentation."""
    summary_service = SummaryService()
    docs_text = """
Clarion Backend API Reference and Developer Guide
Installation:
pip install clarion-core
Quickstart and Usage:
from clarion import ProcessingPipeline
Configuration:
Parameters:
- document_id (str): Unique document identifier.
- timeout (int): Request timeout in seconds.
Returns:
Dictionary containing pipeline status and execution metrics.
Raises:
PipelineStageError on stage failure.
Endpoint: POST /api/v1/analyze
"""
    doc_type = summary_service.classify_document(text=docs_text)
    assert doc_type == "technical_documentation"


def test_classify_academic_paper(isolated_settings):
    """An academic paper with abstract, methodology, and citations must be classified as academic_paper."""
    summary_service = SummaryService()
    paper_text = """
Neural Graph Synthesis: A New Approach
Abstract
In this work, we propose a graph neural architecture for relational induction.
1. Introduction
Recent work [1] has demonstrated significant gains in representation learning.
2. Methodology and Methods
We formulate the relation scoring as a bilinear tensor factorization.
3. Experiments and Results
Our benchmark on the standard dataset shows state-of-the-art results.
4. Conclusion
References:
[1] Vaswani et al., 2017. Attention is all you need.
"""
    doc_type = summary_service.classify_document(text=paper_text)
    assert doc_type == "academic_paper"


def test_classify_unknown(isolated_settings):
    """Unstructured text without dominant signals should return unknown rather than a false positive."""
    summary_service = SummaryService()
    doc_type = summary_service.classify_document(text="hello world sunshine tomorrow morning")
    assert doc_type == "unknown"


# ============================================================================
# 3. Front-Matter Bypassing & Editorial Roster Rejection Tests
# ============================================================================

def test_front_matter_detection(isolated_settings):
    """Chunks with copyright, ISBN, and editorial boards are correctly recognized as front matter."""
    kms = KnowledgeMapService()
    chunk_front = Chunk(
        chunk_id="c-front",
        document_id="doc-1",
        content="Copyright 2024. All rights reserved. ISBN 978-0-123456-47-2. Published by Academic Press.",
        section_title="Copyright and Acknowledgments",
        position_index=0,
        word_count=15,
    )
    chunk_body = Chunk(
        chunk_id="c-body",
        document_id="doc-1",
        content="Gradient descent optimizes parameters by moving in the direction of steepest descent.",
        section_title="Optimization Algorithms",
        position_index=1,
        word_count=12,
    )
    assert kms._is_front_matter_chunk(chunk_front) is True
    assert kms._is_front_matter_chunk(chunk_body) is False


def test_roster_sentence_rejection(isolated_settings):
    """Contributor name rosters are identified and excluded from concept extraction."""
    kms = KnowledgeMapService()
    roster_sentence = "John Smith, Alice Cooper, Bob Dylan, Charlie Parker, David Bowie, Frank Sinatra"
    narrative_sentence = "Alan Turing introduced the Turing test in 1950 to evaluate machine intelligence."

    assert kms._is_roster_sentence(roster_sentence) is True
    assert kms._is_roster_sentence(narrative_sentence) is False


# ============================================================================
# 4. Candidate Relations & Grounding Tests
# ============================================================================

def test_candidate_relations_have_grounding_and_flags(isolated_settings, sample_chunks):
    """Heuristic relations must be marked as candidate relations with grounding chunk IDs."""
    kms = KnowledgeMapService()
    km = kms.build_knowledge_map("doc-test", sample_chunks)
    assert len(km.relations) > 0
    for rel in km.relations:
        assert rel.is_candidate is True
        assert rel.source_chunk_id is not None
        assert isinstance(rel.source_chunk_ids, list)


def test_concepts_have_frequency_and_supporting_sections(isolated_settings, sample_chunks):
    """Concepts must include frequency counts and supporting section titles."""
    kms = KnowledgeMapService()
    km = kms.build_knowledge_map("doc-test", sample_chunks)
    assert len(km.concepts) > 0
    for concept in km.concepts:
        assert concept.frequency >= 1
        assert isinstance(concept.supporting_sections, list)


# ============================================================================
# 5. Grounded Summary Generation Tests
# ============================================================================

def test_summary_deterministic_fallback_notice(isolated_settings, sample_chunks):
    """Deterministic fallback summaries must expose grounded text without template boilerplate, and set metadata flags."""
    summary_service = SummaryService()
    summary = summary_service.generate_summary(
        document_id="doc-test",
        title="Statistics Guide",
        chunks=sample_chunks,
    )
    assert isinstance(summary, StructuredSummary)
    assert summary.overall_summary and len(summary.overall_summary.strip()) > 0
    assert "Deterministic fallback analysis" not in summary.overall_summary
    assert "assessment-style document focused on" not in summary.overall_summary
    assert summary.metadata["fallback_used"] is True
    assert summary.metadata["analysis_mode"] == "deterministic_fallback"
    assert summary.metadata["llm_summary_accepted"] is False
    assert summary.document_type in [
        "textbook", "academic_paper", "report", "assessment",
        "technical_documentation", "notes", "unknown"
    ]


# ============================================================================
# 6. LLM Runtime States & Degraded Pipeline Status Tests
# ============================================================================

def test_llm_runtime_state_enum():
    """Enum values match required states."""
    assert LLMRuntimeState.LLM_AVAILABLE.value == "LLM_AVAILABLE"
    assert LLMRuntimeState.LLM_DEGRADED.value == "LLM_DEGRADED"
    assert LLMRuntimeState.LLM_UNAVAILABLE.value == "LLM_UNAVAILABLE"
    assert LLMRuntimeState.FALLBACK_USED.value == "FALLBACK_USED"


def test_pipeline_tracker_reports_degraded_and_contract_stages(isolated_settings):
    """When fallback occurs, pipeline reports status='degraded' and adheres to interface contract."""
    tracker = PipelineStateTracker()
    tracker.start_pipeline("doc-deg")
    tracker.start_stage("doc-deg", "ingestion", input_size=100)
    tracker.complete_stage("doc-deg", "ingestion", output_size=100)
    tracker.start_stage("doc-deg", "chunking", input_size=100)
    tracker.complete_stage("doc-deg", "chunking", output_size=2)
    tracker.start_stage("doc-deg", "embedding", input_size=2)
    tracker.complete_stage("doc-deg", "embedding", output_size=2)
    tracker.start_stage("doc-deg", "mapping", input_size=2)
    tracker.complete_stage("doc-deg", "mapping", output_size=4)
    tracker.start_stage("doc-deg", "graph_building", input_size=4)
    tracker.complete_stage("doc-deg", "graph_building", output_size=6, quality_metrics={"node_count": 4, "edge_count": 3})

    tracker.complete_pipeline(
        "doc-deg",
        is_degraded=True,
        analysis_mode="deterministic_fallback",
        active_model="qwen3.5:4b",
    )

    status = tracker.get_status("doc-deg")
    assert status["status"] == "degraded"
    assert status["analysis_mode"] == "deterministic_fallback"
    assert status["active_model"] == "qwen3.5:4b"

    stages = status["stages"]
    assert "ingestion" in stages
    assert "chunking" in stages
    assert "embedding" in stages
    assert "mapping" in stages
    assert "graph_building" in stages
    assert "evaluation" in stages
    assert "hierarchy" in stages
    assert "llm_analysis" not in stages

    assert stages["mapping"]["status"] == "fallback"
    assert stages["mapping"]["model"] == "qwen3.5:4b"
    assert stages["chunking"]["chunk_count"] == 2
    assert stages["graph_building"]["node_count"] == 4
    assert stages["graph_building"]["edge_count"] == 3


# ============================================================================
# 7. Embedding Performance & Caching Tests
# ============================================================================

def test_embedding_fallback_dimension_is_1024(isolated_settings):
    """Fallback embedding dimension must be 1024 to match BAAI/bge-large-en-v1.5."""
    emb_service = EmbeddingService()
    assert emb_service.fallback_dimension == 1024
    vec = emb_service._fallback_embed_text("test content")
    assert len(vec) == 1024


def test_embedding_sqlite_cache_roundtrip(isolated_settings, monkeypatch):
    """Fallback vectors are cached under their own identity and reused."""
    emb_service = EmbeddingService()
    emb_service.model = None
    monkeypatch.setattr(emb_service, "_load_model", lambda: None)
    chunks = [
        Chunk(
            chunk_id="c1",
            document_id="d1",
            content="Testing sqlite caching mechanism for vectors.",
            position_index=0,
            word_count=6,
        )
    ]
    # First call generates and caches
    vectors_1 = emb_service.generate_embeddings(chunks)
    assert len(vectors_1) == 1
    assert len(vectors_1[0]) == 1024

    # Verify directly in SQLite table
    conn = sqlite3.connect(emb_service.db_path)
    cur = conn.cursor()
    cache_model_id = emb_service.active_model_id
    assert cache_model_id == "clarion-hash-fallback-v1"
    cur.execute("SELECT content_hash, vector_blob FROM chunk_embeddings WHERE model_name = ?", (cache_model_id,))
    rows = cur.fetchall()
    conn.close()
    assert len(rows) >= 1

    # Fail if the second call recomputes instead of using the persisted vector.
    monkeypatch.setattr(
        emb_service,
        "_fallback_embed_text",
        lambda _text: pytest.fail("fallback vector should be served from SQLite cache"),
    )
    vectors_2 = emb_service.generate_embeddings(chunks)
    assert len(vectors_1) == len(vectors_2)
    assert all(
        len(first) == len(second)
        and all(abs(left - right) <= 1e-6 for left, right in zip(first, second))
        for first, second in zip(vectors_1, vectors_2)
    )
