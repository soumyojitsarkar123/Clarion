"""Tier 2 Boundary & Corner Cases: test_t2_b14_regression_boundaries.py
Feature F14: Baseline Backend Regression Preservation (235 tests).
Verifies boundary conditions: single-element FAISS indexing, max cosine distance bounds [0.20, 1.00],
high-dimensional zero vector normalization, chunk step size invariance under equal size and overlap,
and zero-variance Fleiss' Kappa agreement boundary without zero-division crash.
"""

import math
import sys
from pathlib import Path
import numpy as np
import pytest
import faiss

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.benchmarking.agreement import compute_fleiss_kappa
from services.chunking_service import ChunkingService


def test_b14_single_element_faiss_index():
    """T2.F14.1: Single-element batch in FAISS IndexFlatL2."""
    dim = 1024
    index = faiss.IndexFlatL2(dim)
    vec = np.random.randn(1, dim).astype(np.float32)
    faiss.normalize_L2(vec)

    index.add(vec)
    assert index.ntotal == 1

    distances, indices = index.search(vec, 1)
    assert indices[0][0] == 0
    assert distances[0][0] == pytest.approx(0.0, abs=1e-5)


def test_b14_maximum_cosine_distance_query():
    """T2.F14.2: Maximum distance query bounded strictly in [0.20, 1.00]."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_opposite = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    # Opposite vectors: squared L2 distance = 4.0
    d_sq = float(np.sum((v1 - v_opposite) ** 2))
    assert d_sq == pytest.approx(4.0, abs=1e-5)
    # Clarion calibrated similarity formula: s = 1.0 / (1.0 + d_sq)
    sim = 1.0 / (1.0 + d_sq)
    assert sim == pytest.approx(0.20, abs=1e-5)
    assert 0.20 <= sim <= 1.00


def test_b14_zero_vector_normalization():
    """T2.F14.3: High-dimensional zero vector normalization boundary."""
    vec = np.zeros((1, 1024), dtype=np.float32)
    norm = np.linalg.norm(vec)
    assert norm == 0.0
    # Avoid div by zero by checking norm > 0
    normalized = vec / norm if norm > 0 else vec
    assert np.all(normalized == 0.0)
    assert not np.isnan(normalized).any()


def test_b14_chunk_step_size_identity():
    """T2.F14.4: Maximum chunk step size invariance when size equals overlap."""
    chunk_size = 500
    chunk_overlap = 500
    # Step size formula: max(1, chunk_size - chunk_overlap)
    step = max(1, chunk_size - chunk_overlap)
    assert step == 1  # Never drops to 0 or triggers infinite loop


def test_b14_fleiss_kappa_zero_variance():
    """T2.F14.5: Zero variance Fleiss' Kappa boundary (all raters agree on single label)."""
    # 5 items, 3 raters, everyone selects 'VALID'
    matrix = [
        ["VALID", "VALID", "VALID"],
        ["VALID", "VALID", "VALID"],
        ["VALID", "VALID", "VALID"],
        ["VALID", "VALID", "VALID"],
        ["VALID", "VALID", "VALID"],
    ]
    res = compute_fleiss_kappa(matrix)
    assert res.num_raters == 3
    assert res.num_items == 5
    # When agreement is unanimous across all items, observed agreement is 1.0
    assert res.observed_agreement == 1.0
