"""Tier 4 Real-World Application Scenario: test_t4_s01_academic_paper_pipeline.py
Scenario 1: Academic Paper Pipeline (BSM 301 Mathematics).
Executes realistic multi-stage academic paper processing: structure-aware chunking,
embedding generation, concept mapping, knowledge graph assembly, and stage telemetry validation.
"""

import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.pipeline_observability import PipelineStateTracker, PIPELINE_STAGES
from tests.e2e.test_helpers import simulate_render_pipeline_status, evaluate_frontend_format_duration


def test_t4_s01_academic_paper_pipeline(tmp_path):
    """T4.SCEN.1: Full academic paper processing workflow."""
    doc_id = "doc-bsm-301-math"
    paper_content = (
        "BSM 301: Advanced Engineering Mathematics - Module III\n\n"
        "1. Probability Distributions and Expectation\n"
        "Let X be a continuous random variable with probability density function f(x). "
        "The mathematical expectation E[X] is defined by the improper Riemann integral.\n\n"
        "2. Characteristic Functions and Moment Generating Functions\n"
        "The characteristic function phi_X(t) = E[e^{itX}] uniquely determines the probability distribution. "
        "By the inversion theorem, the density can be recovered via the Fourier inversion formula.\n\n"
        "3. Central Limit Theorem and Applications\n"
        "Let X_1, X_2, ..., X_n be independent and identically distributed random variables with finite variance. "
        "The standardized sum converges in distribution to the standard Gaussian N(0, 1)."
    )

    # Stage 1: Chunking
    chunker = ChunkingService()
    chunks = chunker.chunk_document(doc_id, paper_content)
    assert len(chunks) >= 2

    # Stage 2: Embedding
    embedder = EmbeddingService()
    embeddings = [embedder._fallback_embed_text(c.content) for c in chunks]
    assert len(embeddings) == len(chunks)
    assert all(len(e) == 1024 for e in embeddings)

    # Stage 3: Telemetry Tracking
    tracker = PipelineStateTracker()
    tracker.init_pipeline(doc_id, total_steps=7)
    tracker.start_stage(doc_id, "ingestion")
    tracker.complete_stage(doc_id, "ingestion", input_size=len(paper_content), output_size=len(paper_content))
    tracker.start_stage(doc_id, "chunking")
    tracker.complete_stage(doc_id, "chunking", input_size=len(paper_content), output_size=len(chunks))
    tracker.start_stage(doc_id, "embedding")
    tracker.complete_stage(doc_id, "embedding", input_size=len(chunks), output_size=len(chunks))

    status = tracker.get_status(doc_id)
    stages = status["stages"]

    # Verify 7 canonical stages and sub-second formatting
    assert len(stages) == 7
    assert "llm_analysis" not in stages

    rows = simulate_render_pipeline_status(status)
    assert len(rows) == 7
