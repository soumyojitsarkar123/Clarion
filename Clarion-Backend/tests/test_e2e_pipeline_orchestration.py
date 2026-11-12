"""End-to-End Application-Level Integration Tests for Workflow A.

Validates the complete document upload and ProcessingPipeline orchestration
workflow from upload through retrieval while maintaining content and provenance
integrity in an isolated temporary storage environment.

Real vs Mocked Components:
- REAL: Text extraction (DOCX/PDF), structure-aware chunking, fast tokenizer offset
        segmentation, SentenceTransformer BGE model inference (BAAI/bge-large-en-v1.5,
        1024-dim), SQLite persistence, FAISS IndexFlatL2 vector storage, NetworkX graph
        building, confidence scoring, topological hierarchy, and segment-aware retrieval.
- MOCKED (CONTROLLED FAKE): External LLM generation calls only (concept extraction,
        relation extraction, topic hierarchy, summary, and answer synthesis) to ensure
        fully deterministic integration testing without external network dependencies.
"""

import asyncio
import io
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import pytest
from fastapi.testclient import TestClient

from main import app
from models.chunk import Chunk
from models.document import DocumentStatus
from services.background_service import BackgroundService
from services.chunking_service import ChunkingService
from services.document_service import DocumentService
from services.embedding_service import EmbeddingService
from services.knowledge_map_service import KnowledgeMapService, LLMInterface
from services.processing_pipeline import PipelineStageError, ProcessingPipeline
from services.rag_audit_service import RagAuditService
from services.relation_dataset_service import RelationDatasetService
from services.retrieval_service import RetrievalService
from tests.conftest import STATS_TEXT, make_docx_bytes
from utils.config import settings
from utils.file_handler import extract_text
from vectorstore import VectorStore


def controlled_fake_llm_generate(self, prompt: str) -> str:
    """Deterministic controlled LLM generator for external generation prompts only."""
    prompt_lower = prompt.lower()
    if (
        "extract key concepts" in prompt
        or "significant knowledge concepts" in prompt
        or "concepts in this text" in prompt
        or "most significant" in prompt
    ):
        return json.dumps(
            [
                {
                    "name": "Hypothesis Testing",
                    "definition": (
                        "A mathematical framework for statistical decision making "
                        "from empirical sample observations."
                    ),
                },
                {
                    "name": "Null Hypothesis",
                    "definition": (
                        "The baseline proposition assumed true until sample "
                        "evidence indicates otherwise."
                    ),
                },
                {
                    "name": "Sampling Distribution",
                    "definition": (
                        "The probability distribution of a given statistic based on "
                        "repeated random samples."
                    ),
                },
                {
                    "name": "Type I Error",
                    "definition": (
                        "The incorrect rejection of a true null hypothesis with "
                        "probability alpha."
                    ),
                },
                {
                    "name": "Confidence Interval",
                    "definition": (
                        "An estimated interval containing the true population "
                        "parameter with specified confidence."
                    ),
                },
            ]
        )
    elif "analyze the relationship between" in prompt or "relationship between these two" in prompt:
        return json.dumps(
            {
                "relation_type": "prerequisite",
                "description": (
                    "Understanding the null hypothesis is a prerequisite for "
                    "conducting hypothesis testing."
                ),
                "confidence": 0.95,
            }
        )
    elif "analyze relationships between these concepts" in prompt:
        return json.dumps(
            [
                {
                    "from_concept": "Null Hypothesis",
                    "to_concept": "Hypothesis Testing",
                    "relation_type": "part-of",
                    "description": (
                        "Forms the baseline assumption tested in statistical "
                        "hypothesis testing."
                    ),
                    "confidence": 0.92,
                },
                {
                    "from_concept": "Sampling Distribution",
                    "to_concept": "Hypothesis Testing",
                    "relation_type": "prerequisite",
                    "description": (
                        "Sampling distributions provide the reference distribution "
                        "for computing test statistics."
                    ),
                    "confidence": 0.90,
                },
            ]
        )
    elif "main topics and their hierarchy" in prompt or "identify the main topics" in prompt:
        return json.dumps(
            {
                "main_topics": [
                    {
                        "id": "topic_stats",
                        "title": "Statistical Inference",
                        "description": (
                            "Foundational principles and mathematical frameworks "
                            "of statistical inference."
                        ),
                        "subtopic_ids": ["sub_hyp", "sub_dist"],
                    }
                ],
                "subtopics": [
                    {
                        "id": "sub_hyp",
                        "title": "Hypothesis Testing",
                        "parent_topic_id": "topic_stats",
                    },
                    {
                        "id": "sub_dist",
                        "title": "Sampling Distributions",
                        "parent_topic_id": "topic_stats",
                    },
                ],
            }
        )
    elif "executive summary" in prompt_lower or "concise, factual executive summary" in prompt_lower:
        return (
            "Hypothesis testing provides a systematic mathematical framework to make inferences "
            "regarding population characteristics from empirical sample observations. Key concepts "
            "include null hypotheses, sampling distributions, and rigorous type-one error rate controls."
        )
    elif "structured summary" in prompt_lower or "comprehensive structured summary" in prompt_lower:
        return (
            "This document systematically explores statistical hypothesis testing methodologies. "
            "It establishes core mathematical definitions including the null hypothesis, sampling "
            "distributions, and decision boundaries for statistical inference."
        )
    elif "based on the following context" in prompt_lower or "answer:" in prompt_lower:
        return (
            "Hypothesis testing is a statistical method used to evaluate claims about a population "
            "based on sample data, balancing significance levels and error rates."
        )
    else:
        return "Controlled fake LLM response for testing pipeline orchestration."


@pytest.fixture
def isolated_orchestration_env(tmp_path, monkeypatch):
    """Isolate storage directories and rebind router singletons for genuine pipeline tests."""
    temp_data = tmp_path / "data"
    temp_vectorstore = temp_data / "vectorstore"
    temp_logs = tmp_path / "logs"
    temp_datasets = temp_data / "datasets"
    temp_graphs = temp_data / "graphs"
    temp_reports = temp_data / "reports"

    for directory in (
        temp_data,
        temp_vectorstore,
        temp_logs,
        temp_datasets,
        temp_graphs,
        temp_reports,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    prod_db_path = Path(__file__).resolve().parents[1] / "data" / "clarion.db"
    prod_db_stat_before = prod_db_path.stat() if prod_db_path.exists() else None

    # Redirect global settings
    monkeypatch.setattr(settings, "data_dir", temp_data)
    monkeypatch.setattr(settings, "vectorstore_dir", temp_vectorstore)
    monkeypatch.setattr(settings, "logs_dir", temp_logs)
    monkeypatch.setattr(settings, "dataset_dir", temp_datasets)
    isolated_db_path = temp_data / "clarion.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{isolated_db_path}")

    # Rebind module-level router singletons to isolated storage
    import routers.analyze as analyze_mod
    import routers.query as query_mod
    import routers.upload as upload_mod

    monkeypatch.setattr(upload_mod, "document_service", DocumentService())
    monkeypatch.setattr(analyze_mod, "document_service", DocumentService())
    monkeypatch.setattr(analyze_mod, "background_service", BackgroundService(db_path=isolated_db_path))
    monkeypatch.setattr(analyze_mod, "pipeline", ProcessingPipeline())
    monkeypatch.setattr(query_mod, "document_service", DocumentService())
    monkeypatch.setattr(query_mod, "retrieval_service", RetrievalService())
    monkeypatch.setattr(query_mod, "knowledge_map_service", KnowledgeMapService())
    monkeypatch.setattr(query_mod, "rag_audit_service", RagAuditService(db_path=isolated_db_path))

    # Configure controlled fake LLM ONLY for external generation calls
    monkeypatch.setattr(LLMInterface, "_is_service_available", lambda self: True)
    monkeypatch.setattr(LLMInterface, "generate", controlled_fake_llm_generate)
    monkeypatch.setattr(
        RelationDatasetService, "_workspace_dataset_dir", lambda self: None
    )

    env = {
        "tmp_path": tmp_path,
        "data_dir": temp_data,
        "vectorstore_dir": temp_vectorstore,
        "db_path": isolated_db_path,
        "client": TestClient(app),
        "upload_mod": upload_mod,
        "analyze_mod": analyze_mod,
        "query_mod": query_mod,
    }

    yield env

    # Teardown assertion: Production database was never modified
    if prod_db_path.exists() and prod_db_stat_before:
        prod_db_stat_after = prod_db_path.stat()
        assert (
            prod_db_stat_after.st_mtime == prod_db_stat_before.st_mtime
        ), "FATAL: Production clarion.db modification time changed!"
        assert (
            prod_db_stat_after.st_size == prod_db_stat_before.st_size
        ), "FATAL: Production clarion.db size changed during integration test!"


def build_multilingual_technical_docx() -> bytes:
    """Build a multi-section technical document fixture with non-ASCII and Unicode symbols.
    
    Contains > 1,000 words across 5 sections, ensuring multiple chunks and tokenizer-aware
    segmentation across context windows.
    """
    paragraphs = [
        (
            "# 1. Foundations of Statistical Hypothesis Testing and Inference\n\n"
            "Hypothesis testing is an authoritative mathematical and statistical procedure used to make "
            "principled decisions regarding population distributions from empirical sample observations. "
            "In modern algorithmic data analysis, statistical hypotheses are structured into two competing "
            "propositions: the null hypothesis denoted by H₀, and the alternative hypothesis denoted by H₁. "
            "The null hypothesis asserts the baseline proposition that no systematic phenomenon, treatment "
            "effect, or parameter disparity exists within the measured population. Conversely, the alternative "
            "hypothesis represents the operational claim under empirical scrutiny. When evaluating a sample "
            "mean vector μ against an established benchmark μ₀, rigorous statistical estimators require "
            "unbiased variance properties and bounded error propagation across dimensional projections."
        ),
        (
            "# 2. Mathematical Formalism and Non-ASCII Operational Constraints\n\n"
            "Let {x₁, x₂, ..., xₙ} represent independent and identically distributed random observations "
            "drawn from an underlying normal distribution N(μ, σ²). The sample mean estimator is formally "
            "expressed by the finite summation: x̄ = (1/n) · ∑_{i=1}^n x_i. When the true population "
            "variance σ² is known, the standardized test statistic follows the standard normal distribution: "
            "Z = (x̄ - μ₀) / (σ / √n), where Z ~ N(0, 1). In the generalized multidimensional setting, "
            "the quadratic risk function satisfies the inequality: R(θ, δ) ≤ ∫ L(θ, δ(x)) f(x; θ) dx. "
            "Decision boundaries partition the sample space into an acceptance region A and a critical rejection "
            "region C such that the intersection satisfies A ∩ C = ∅ and A ∪ C = Ω. Box-drawing notation "
            "for parameter partition matrices is represented as: ╔═════════════╦═════════════╗ with borders ║ "
            "and closing corners ╚═════════════╩═════════════╝. Furthermore, mathematical inequalities "
            "strictly enforce α ≤ 0.05, β ≤ 0.20, and test power 1 - β ≥ 0.80."
        ),
        (
            "# 3. Sampling Distributions and Asymptotic Central Limit Convergence\n\n"
            "The asymptotic behavior of sample estimators is governed by the Lindeberg-Lévy Central Limit "
            "Theorem. For any sequence of independent random variables possessing finite expectation E[X_i] = μ "
            "and non-zero finite variance Var(X_i) = σ², the standardized cumulative difference converges in "
            "distribution to the standard normal distribution as sample cardinality tends toward infinity: "
            "lim_{n → ∞} P( (∑ X_i - nμ) / (σ√n) ≤ z ) = Φ(z) = ∫_{-∞}^z (1/√(2π)) e^{-t²/2} dt. "
            "This universal asymptotic convergence provides the fundamental mathematical justification for "
            "constructing confidence intervals and hypothesis tests in high-dimensional engineering systems. "
            "Confidence intervals quantify estimation uncertainty by generating a stochastic interval "
            "[L(X), U(X)] such that P(L(X) ≤ θ ≤ U(X)) = 1 - α for any prescribed confidence coefficient 1 - α. "
            "Typical operational confidence levels include 95% (z_{0.025} ≈ 1.960) and 99% (z_{0.005} ≈ 2.576)."
        ),
        (
            "# 4. Decision Risk, Error Classification, and Test Power Optimization\n\n"
            "Statistical decision theory categorizes testing errors into two distinct failure modes. A Type I "
            "error occurs when a true null hypothesis H₀ is erroneously rejected by the decision rule. The "
            "maximum permissible probability of committing a Type I error is defined as the significance level "
            "α = P(Reject H₀ | H₀ is true). Conversely, a Type II error occurs when a false null hypothesis "
            "is erroneously retained, denoted with probability β = P(Fail to reject H₀ | H₁ is true). "
            "The statistical power of a test, defined as 1 - β = P(Reject H₀ | H₁ is true), measures the "
            "operational sensitivity of the test procedure in correctly detecting true effects. The Neyman-Pearson "
            "lemma establishes that the likelihood ratio test maximizes power among all level-α tests for "
            "simple hypothesis pairs. Balancing these error probabilities requires careful sample size planning: "
            "n ≥ ( (z_α + z_β) · σ / δ )², ensuring that statistical significance is not conflated with "
            "mere sample scale."
        ),
        (
            "# 5. Multi-Lingual Nomenclature, Accented Expressions, and Global Standards\n\n"
            "Academic and engineering discourse incorporates multilingual technical terminology across "
            "international computational venues. French terminology frequently references the analytical "
            "“café de la recherche”, highlighting the “naïve Bayes” classifier and architectural “façade” "
            "patterns in software design. German statistical literature emphasizes “Grundwahrheit” (ground truth) "
            "and “Stichprobenverteilung” (sampling distribution). In Devanagari script, foundational concepts "
            "include नमस्ते (salutations) and सांख्यिकी (statistics). Mathematical texts incorporate Greek letters "
            "extensively: α (alpha), β (beta), γ (gamma), δ (delta), ε (epsilon), θ (theta), λ (lambda), "
            "μ (mu), π (pi), σ (sigma), and Ω (omega). Typography preserves smart quotes (“ ”), apostrophes (‘ ’), "
            "em-dashes (—), and ellipsis (…). Robust tokenizer-aware embedding and indexing pipelines must "
            "preserve every single one of these Unicode characters losslessly without silent truncation or corruption."
        ),
    ]
    return make_docx_bytes(paragraphs)


# ---------------------------------------------------------------------------
# Test 1: Full Application-Level End-to-End Workflow A Integration
# ---------------------------------------------------------------------------
def test_e2e_upload_analyze_query_workflow_a(isolated_orchestration_env):
    """Execute real upload, synchronous pipeline analysis, and retrieval through public routes."""
    client: TestClient = isolated_orchestration_env["client"]
    db_path = isolated_orchestration_env["db_path"]
    vectorstore_dir = isolated_orchestration_env["vectorstore_dir"]

    # 1. Genuine Upload Route: POST /upload
    docx_bytes = build_multilingual_technical_docx()
    upload_response = client.post(
        "/upload",
        files={
            "file": (
                "multilingual_technical_stats.docx",
                docx_bytes,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert upload_response.status_code == 200, f"Upload failed: {upload_response.text}"
    upload_data = upload_response.json()
    assert "document_id" in upload_data
    doc_id = upload_data["document_id"]
    assert upload_data["status"] == "uploaded"
    assert upload_data["filename"] == "multilingual_technical_stats.docx"

    # Verify SQLite record created in documents table
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    doc_row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    conn.close()
    assert doc_row is not None, "Document not found in isolated SQLite documents table"
    assert doc_row["status"] == DocumentStatus.UPLOADED.value
    assert doc_row["word_count"] >= 700
    assert "Hypothesis" in doc_row["text_content"]
    assert "∑" in doc_row["text_content"], "Non-ASCII mathematical symbol missing from extracted text"

    # 2. Genuine Analysis Route: POST /analyze/{document_id} with wait_for_completion=True
    analyze_response = client.post(
        f"/analyze/{doc_id}",
        json={
            "generate_hierarchy": True,
            "run_evaluation": True,
            "force_reanalyze": False,
            "wait_for_completion": True,
            "timeout_seconds": 180,
        },
    )
    assert analyze_response.status_code == 200, f"Analysis failed: {analyze_response.text}"
    analyze_data = analyze_response.json()
    assert analyze_data["document_id"] == doc_id
    assert analyze_data["status"] == "completed"

    pipeline_results = analyze_data.get("results", {})
    assert pipeline_results.get("success") is True
    stages_completed = pipeline_results.get("stages_completed", [])
    expected_stages = [
        "ingestion",
        "chunking",
        "embedding",
        "mapping",
        "graph_building",
        "evaluation",
        "hierarchy",
    ]
    for stage in expected_stages:
        assert stage in stages_completed, f"Stage {stage} was not completed: {stages_completed}"

    # Verify isolated SQLite state post-analysis
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Document status updated to analyzed
    doc_row_after = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    assert doc_row_after["status"] == DocumentStatus.ANALYZED.value

    # Chunks persisted
    chunks = conn.execute("SELECT * FROM chunks WHERE document_id = ?", (doc_id,)).fetchall()
    assert len(chunks) >= 4, f"Expected at least 4 chunks, got {len(chunks)}"

    # Segments persisted in embedding_segments table
    segments = conn.execute(
        "SELECT * FROM embedding_segments WHERE document_id = ? ORDER BY segment_id", (doc_id,)
    ).fetchall()
    assert len(segments) >= len(chunks), "Expected at least as many segments as chunks"

    # Embeddings metadata persisted
    emb_rows = conn.execute("SELECT * FROM embeddings WHERE document_id = ?", (doc_id,)).fetchall()
    assert len(emb_rows) == len(segments), "Embeddings table count must match segment count"

    # Knowledge map persisted
    km_row = conn.execute("SELECT * FROM knowledge_maps WHERE document_id = ?", (doc_id,)).fetchone()
    assert km_row is not None, "Knowledge map was not persisted to SQLite"

    conn.close()

    # Resolve the model/revision-scoped index through retrieval's DB-map check.
    vector_store = EmbeddingService().get_vector_store(doc_id)
    assert vector_store is not None
    index_file = vector_store.index_path
    meta_file = vector_store.metadata_path
    assert index_file.exists(), f"FAISS index file missing at {index_file}"
    assert meta_file.exists(), f"FAISS meta file missing at {meta_file}"

    with open(meta_file, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    faiss_vector_ids = meta_data["vector_ids"]
    assert faiss_vector_ids == vector_store.chunk_ids
    assert len(meta_data["vector_ids"]) == len(segments)

    # Verify FAISS vector count matches segment count and dimensions are 1024
    faiss_index = faiss.read_index(str(index_file))
    assert faiss_index.ntotal == len(segments)
    assert faiss_index.d == 1024

    # 3. Genuine Retrieval Route: POST /query/{document_id}
    query_response = client.post(
        f"/query/{doc_id}",
        json={
            "query": "What is the null hypothesis and how is it tested?",
            "top_k": 3,
            "include_knowledge_map": True,
        },
    )
    assert query_response.status_code == 200, f"Query failed: {query_response.text}"
    query_data = query_response.json()
    assert query_data["query"] == "What is the null hypothesis and how is it tested?"
    results = query_data.get("results", [])
    assert len(results) >= 1, "Expected at least 1 retrieval result"

    first_result = results[0]
    assert "chunk_id" in first_result
    assert "content" in first_result
    assert "score" in first_result
    assert first_result["score"] >= 0.5, f"Expected score >= 0.5, got {first_result['score']}"

    # Verify returned chunk_id maps directly to a parent chunk in SQLite chunks table
    conn = sqlite3.connect(db_path)
    parent_chunk = conn.execute(
        "SELECT * FROM chunks WHERE chunk_id = ?", (first_result["chunk_id"],)
    ).fetchone()
    conn.close()
    assert parent_chunk is not None, f"Returned chunk_id {first_result['chunk_id']} not in chunks table"

    # Verify grounded answer synthesis was returned
    assert len(query_data.get("response", "")) > 20
    assert query_data.get("knowledge_map") is not None


# ---------------------------------------------------------------------------
# Test 2: Content & Provenance Integrity (R4 Lossless Coverage & Token Limit)
# ---------------------------------------------------------------------------
def test_lossless_segmentation_exact_string_cover_and_token_ceiling(isolated_orchestration_env):
    """Verify lossless concatenation and strict 512-subword token ceiling."""
    embedding_service = EmbeddingService()
    embedding_service._load_model()
    tokenizer = embedding_service._ensure_offset_tokenizer()
    limit = embedding_service._model_token_limit()
    assert limit == 512, f"Expected BGE sequence limit of 512, got {limit}"

    # Construct multiple synthetic chunks of varying lengths (small, medium, oversized)
    long_text = (
        "Statistical Learning Theory and Empirical Risk Minimization. "
        + "Let the feature space X and target space Y define a joint probability distribution P(X, Y). "
        + "A learning algorithm observes training samples and seeks a predictor minimizing expected risk. "
    ) * 40  # ~880 words, ~1,300 tokens -> must segment into multiple spans

    chunks = [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id="doc_integrity_test",
            section_title="Long Algorithmic Theory",
            content=long_text,
            position_index=0,
            word_count=len(long_text.split()),
        ),
        Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id="doc_integrity_test",
            section_title="Compact Summary",
            content=STATS_TEXT,
            position_index=len(long_text.split()),
            word_count=len(STATS_TEXT.split()),
        ),
    ]

    segments = embedding_service.prepare_embedding_segments(chunks)
    assert len(segments) > len(chunks), "Oversized chunk must generate multiple segments"

    # Group segments by parent chunk ID
    by_parent: Dict[str, List[Chunk]] = {}
    for seg in segments:
        parent_id, _ = seg.chunk_id.rsplit(":seg:", 1)
        by_parent.setdefault(parent_id, []).append(seg)

    for chunk in chunks:
        parent_segments = by_parent[chunk.chunk_id]
        # Guarantee 1: Exact lossless string reconstruction
        reconstructed = "".join(s.content for s in parent_segments)
        assert reconstructed == chunk.content, (
            f"Lossless violation for chunk {chunk.chunk_id}: "
            f"expected len {len(chunk.content)}, got len {len(reconstructed)}"
        )

        # Guarantee 2: Every segment strictly respects the 512-subword ceiling
        for seg in parent_segments:
            tokens = tokenizer.encode(seg.content, add_special_tokens=True, truncation=False)
            assert len(tokens) <= limit, (
                f"Token ceiling violation on {seg.chunk_id}: "
                f"token count {len(tokens)} exceeds {limit}"
            )

        # Guarantee 3: Exact non-overlapping character span tracking
        for seg in parent_segments:
            span = embedding_service._segment_spans[seg.chunk_id]
            start, end = span
            assert chunk.content[start:end] == seg.content, (
                f"Character span mismatch on {seg.chunk_id}: "
                f"content[{start}:{end}] != segment.content"
            )

        # Guarantee 4: Contiguity of character spans
        spans = [embedding_service._segment_spans[s.chunk_id] for s in parent_segments]
        assert spans[0][0] == 0
        assert spans[-1][1] == len(chunk.content)
        for i in range(len(spans) - 1):
            assert spans[i][1] == spans[i + 1][0], f"Gap between segment spans at index {i}"


# ---------------------------------------------------------------------------
# Test 3: Real Repository PDF Fixture (DAA FINAL.pdf) Provenance & Retrieval
# ---------------------------------------------------------------------------
def test_daa_final_pdf_fixture_ingestion_provenance_and_retrieval(isolated_orchestration_env):
    """Validate genuine PDF text extraction, segmentation, and retrieval on DAA FINAL.pdf."""
    pdf_path = Path(__file__).resolve().parents[1] / "data" / "DAA FINAL.pdf"
    assert pdf_path.exists(), f"DAA FINAL.pdf fixture not found at {pdf_path}"

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Genuine text extraction
    text, meta = extract_text(pdf_bytes, "DAA FINAL.pdf")
    assert meta["page_count"] >= 80, f"Expected >= 80 pages, got {meta['page_count']}"
    words = text.split()
    assert len(words) >= 15000, f"Expected >= 15,000 words, got {len(words)}"

    # Verify non-ASCII characters exist in extracted PDF text
    has_non_ascii = any(ord(c) > 127 for c in text)
    assert has_non_ascii, "DAA FINAL.pdf must contain non-ASCII characters"
    assert any(c in text for c in ["╔", "║", "╚", "—", "•"]), "Expected box-drawing or Unicode characters"

    # Upload via DocumentService
    document_service = DocumentService()
    doc = document_service.upload_document(pdf_bytes, "DAA FINAL.pdf")
    assert doc.id is not None
    assert doc.metadata.word_count >= 15000

    # Execute chunking on a representative 3,000-word algorithmic slice
    chunking_service = ChunkingService()
    slice_text = " ".join(words[:3000])
    chunks = chunking_service.chunk_document(doc.id, slice_text)
    assert len(chunks) >= 5, f"Expected >= 5 chunks for 3,000 words, got {len(chunks)}"

    # Generate real BGE embeddings and FAISS index
    embedding_service = EmbeddingService()
    embedding_chunks, embeddings = embedding_service.generate_embedding_segments(chunks)
    assert len(embedding_chunks) >= len(chunks)
    assert len(embeddings) == len(embedding_chunks)
    assert len(embeddings[0]) == 1024

    # Verify unit norm of generated embeddings
    for emb in embeddings[:5]:
        norm = sum(x * x for x in emb) ** 0.5
        assert abs(norm - 1.0) < 1e-4, f"Embedding norm {norm} is not unit length"

    vector_store = embedding_service.create_vector_index(doc.id, embedding_chunks, embeddings)
    assert vector_store is not None

    # Retrieve against the indexed algorithmic document
    retrieval_service = RetrievalService()
    results = retrieval_service.retrieve(doc.id, "dynamic programming algorithm complexity", top_k=3)
    assert len(results) >= 1, "Expected at least 1 retrieval result for algorithmic query"

    first_hit = results[0]
    assert first_hit.chunk_id in [c.chunk_id for c in chunks]
    assert first_hit.score >= 0.5
    assert len(first_hit.content) > 20


# ---------------------------------------------------------------------------
# Test 4: SQLite-FAISS Consistency and 3-Tier Hierarchy Audit
# ---------------------------------------------------------------------------
def test_sqlite_faiss_consistency_and_three_tier_hierarchy(isolated_orchestration_env):
    """Verify strict 1:1 mapping between SQLite embedding_segments and FAISS index vectors."""
    db_path = isolated_orchestration_env["db_path"]
    vectorstore_dir = isolated_orchestration_env["vectorstore_dir"]

    docx_bytes = build_multilingual_technical_docx()
    doc_service = DocumentService()
    doc = doc_service.upload_document(docx_bytes, "hierarchy_test.docx")

    pipeline = ProcessingPipeline()
    result = asyncio.run(pipeline.execute(doc.id, skip_existing=False))
    assert result.get("success") is True

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # 1. Root level: documents table
    doc_row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc.id,)).fetchone()
    assert doc_row is not None
    assert doc_row["id"] == doc.id

    # 2. Tier 2: chunks table
    chunks = conn.execute(
        "SELECT * FROM chunks WHERE document_id = ? ORDER BY position_index", (doc.id,)
    ).fetchall()
    assert len(chunks) >= 4
    chunk_ids = {c["chunk_id"] for c in chunks}

    # 3. Tier 3: embedding_segments table (ordered by vector_index to match FAISS)
    segments = conn.execute(
        """SELECT s.*, e.vector_index 
           FROM embedding_segments s 
           JOIN embeddings e ON s.segment_id = e.chunk_id 
           WHERE s.document_id = ? 
           ORDER BY e.vector_index""",
        (doc.id,),
    ).fetchall()
    assert len(segments) >= len(chunks)

    # 4. Tier 3: embeddings table
    embeddings_rows = conn.execute(
        "SELECT * FROM embeddings WHERE document_id = ? ORDER BY vector_index", (doc.id,)
    ).fetchall()
    assert len(embeddings_rows) == len(segments)

    # Verify the model/revision-scoped index and its validated SQLite mapping.
    vector_store = EmbeddingService().get_vector_store(doc.id)
    assert vector_store is not None
    meta_path = vector_store.metadata_path
    index_path = vector_store.index_path
    assert meta_path.exists()
    assert index_path.exists()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    faiss_vector_ids = meta_data["vector_ids"]
    assert faiss_vector_ids == vector_store.chunk_ids
    assert len(faiss_vector_ids) == len(segments)

    faiss_index = faiss.read_index(str(index_path))
    assert faiss_index.ntotal == len(segments)

    # Verify 1:1 vector_index correspondence across FAISS and SQLite
    for idx, seg in enumerate(segments):
        # Deterministic naming: {chunk_id}:seg:{idx}
        assert ":seg:" in seg["segment_id"]
        source_id = seg["source_chunk_id"]
        assert source_id in chunk_ids, f"Segment {seg['segment_id']} points to nonexistent parent chunk"
        assert seg["char_end"] > seg["char_start"]

        # Content matches exact character span in parent chunk
        parent_chunk = next(c for c in chunks if c["chunk_id"] == source_id)
        expected_span = parent_chunk["content"][seg["char_start"]:seg["char_end"]]
        assert seg["content"] == expected_span

        # FAISS metadata alignment
        assert faiss_vector_ids[idx] == seg["segment_id"]
        assert embeddings_rows[idx]["chunk_id"] == seg["segment_id"]
        assert embeddings_rows[idx]["vector_index"] == idx

    conn.close()


# ---------------------------------------------------------------------------
# Test 5: Simulated Stage Error Cleanup and Containment
# ---------------------------------------------------------------------------
def test_failed_processing_cleanup_and_error_containment(isolated_orchestration_env, monkeypatch):
    """Verify that a simulated pipeline stage failure marks status='failed' without orphan state."""
    db_path = isolated_orchestration_env["db_path"]

    docx_bytes = build_multilingual_technical_docx()
    doc_service = DocumentService()
    doc = doc_service.upload_document(docx_bytes, "failed_doc_test.docx")

    # Inject failure into Stage 4 (mapping)
    async def failing_stage_mapping(document_id, chunks, use_existing=True):
        raise RuntimeError("Simulated stage error: mapping component failure")

    pipeline = ProcessingPipeline()
    monkeypatch.setattr(pipeline, "_stage_mapping", failing_stage_mapping)

    with pytest.raises(PipelineStageError) as exc_info:
        asyncio.run(pipeline.execute(doc.id, skip_existing=False))
    assert "Simulated stage error" in str(exc_info.value)

    # Verify SQLite document record was transitioned to failed status
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    doc_row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc.id,)).fetchone()
    assert doc_row["status"] == DocumentStatus.FAILED.value
    assert "Simulated stage error" in doc_row["error_message"]

    # Verify failure report was saved in analysis_reports
    report_row = conn.execute("SELECT * FROM analysis_reports WHERE document_id = ?", (doc.id,)).fetchone()
    assert report_row is not None
    report_data = json.loads(report_row["data"])
    assert report_data.get("success") is False
    assert "Simulated stage error" in report_data.get("error", "")

    conn.close()


# ---------------------------------------------------------------------------
# Test 6: Reprocessing Deduplication with force_reanalyze=True
# ---------------------------------------------------------------------------
def test_reprocessing_deduplication_with_force_reanalyze(isolated_orchestration_env):
    """Verify that reprocessing with force_reanalyze=True purges old records, preventing duplicates."""
    db_path = isolated_orchestration_env["db_path"]
    vectorstore_dir = isolated_orchestration_env["vectorstore_dir"]

    docx_bytes = build_multilingual_technical_docx()
    doc_service = DocumentService()
    doc = doc_service.upload_document(docx_bytes, "reprocess_test.docx")

    pipeline = ProcessingPipeline()

    # Pass 1: Initial analysis
    res1 = asyncio.run(pipeline.execute(doc.id, skip_existing=False))
    assert res1.get("success") is True

    conn = sqlite3.connect(db_path)
    initial_chunks = conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc.id,)).fetchone()[0]
    initial_segs = conn.execute("SELECT COUNT(*) FROM embedding_segments WHERE document_id = ?", (doc.id,)).fetchone()[0]
    initial_embs = conn.execute("SELECT COUNT(*) FROM embeddings WHERE document_id = ?", (doc.id,)).fetchone()[0]
    conn.close()

    vector_store = EmbeddingService().get_vector_store(doc.id)
    assert vector_store is not None
    index_file = vector_store.index_path
    initial_faiss_ntotal = faiss.read_index(str(index_file)).ntotal
    assert initial_faiss_ntotal == initial_segs

    # Pass 2: Reprocess with force_reanalyze=True (skip_existing=False)
    res2 = asyncio.run(pipeline.execute(doc.id, skip_existing=False))
    assert res2.get("success") is True

    conn = sqlite3.connect(db_path)
    final_chunks = conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc.id,)).fetchone()[0]
    final_segs = conn.execute("SELECT COUNT(*) FROM embedding_segments WHERE document_id = ?", (doc.id,)).fetchone()[0]
    final_embs = conn.execute("SELECT COUNT(*) FROM embeddings WHERE document_id = ?", (doc.id,)).fetchone()[0]
    conn.close()

    final_vector_store = EmbeddingService().get_vector_store(doc.id)
    assert final_vector_store is not None
    assert final_vector_store.index_path == index_file
    final_faiss_ntotal = faiss.read_index(str(final_vector_store.index_path)).ntotal

    # Deduplication assertion: Counts must NOT double!
    assert final_chunks == initial_chunks, f"Chunk duplication detected: {final_chunks} != {initial_chunks}"
    assert final_segs == initial_segs, f"Segment duplication detected: {final_segs} != {initial_segs}"
    assert final_embs == initial_embs, f"Embedding duplication detected: {final_embs} != {initial_embs}"
    assert final_faiss_ntotal == initial_faiss_ntotal, f"FAISS vector duplication: {final_faiss_ntotal} != {initial_faiss_ntotal}"


# ---------------------------------------------------------------------------
# Test 7: Historical Unsegmented Data Compatibility Fallback
# ---------------------------------------------------------------------------
def test_historical_unsegmented_data_backward_compatibility(isolated_orchestration_env):
    """Verify RetrievalService fallback when searching historical indexes lacking embedding_segments."""
    db_path = isolated_orchestration_env["db_path"]

    # Create document and chunk directly in SQLite without embedding_segments
    doc_id = "doc_historical_legacy"
    chunk_id = "chunk_legacy_001"
    content = "Legacy document content without tokenizer-aware segmentation spans."

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO documents (id, filename, file_size, file_type, upload_time, page_count, word_count, status, text_content) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (doc_id, "legacy.docx", 1000, ".docx", "2025-01-01T00:00:00", 1, 10, "analyzed", content),
    )
    conn.execute(
        "INSERT INTO chunks (chunk_id, document_id, section_title, content, position_index, word_count) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (chunk_id, doc_id, "Legacy Section", content, 0, 10),
    )
    conn.commit()
    conn.close()

    # Create FAISS index directly mapping chunk_id
    embedding_service = EmbeddingService()
    legacy_chunk = Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        section_title="Legacy",
        content=content,
        position_index=0,
        word_count=10,
    )
    vecs = embedding_service.generate_embeddings([legacy_chunk])
    embedding_service.create_vector_index(doc_id, [legacy_chunk], vecs)

    # Delete any embedding_segments entries for this document to simulate legacy store
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM embedding_segments WHERE document_id = ?", (doc_id,))
    conn.commit()
    conn.close()

    # Execute retrieval: Must fall back gracefully to raw chunk_id
    retrieval_service = RetrievalService()
    results = retrieval_service.retrieve(doc_id, "Legacy document content", top_k=1)
    assert len(results) == 1, "Expected 1 result from legacy fallback retrieval"
    assert results[0].chunk_id == chunk_id
    assert results[0].content == content
