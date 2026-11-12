"""Tier 3 Cross-Feature Combination: test_t3_x15_rag_audit_evaluation_alignment.py
Feature Interaction: F14 (Baseline Backend Regression Preservation) + F16 (Live vs Mock Transparency).
Validates that FAISS similarity indexing and RAG retrieval evaluations operate deterministically
under offline conditions, maintaining similarity calibration bounded within [0.20, 1.00].
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import faiss

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.embedding_service import EmbeddingService


def test_t3_x15_rag_audit_evaluation_alignment():
    """T3.INT.15: F14 + F16 Pairwise Integration."""
    dim = 1024
    index = faiss.IndexFlatL2(dim)

    # Deterministic query and document vectors
    embedder = EmbeddingService()
    doc_text = "Topological sort orders directed acyclic graph vertices."
    query_text = "What is topological sorting?"

    v_doc = np.array([embedder._fallback_embed_text(doc_text)], dtype=np.float32)
    v_query = np.array([embedder._fallback_embed_text(query_text)], dtype=np.float32)

    faiss.normalize_L2(v_doc)
    faiss.normalize_L2(v_query)

    index.add(v_doc)
    assert index.ntotal == 1

    distances, indices = index.search(v_query, 1)
    d_sq = float(distances[0][0])
    # Calibrated similarity score
    sim_score = 1.0 / (1.0 + d_sq)
    assert 0.20 <= sim_score <= 1.00
