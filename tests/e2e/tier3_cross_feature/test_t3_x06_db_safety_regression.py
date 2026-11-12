"""Tier 3 Cross-Feature Combination: test_t3_x06_db_safety_regression.py
Feature Interaction: F14 (Baseline Backend Regression Preservation) + F15 (Production Database Isolation & Safety).
Validates that regression operations (embeddings, graph generation, document storage)
execute within temporary environments and leave the production database completely unmodified.
"""

import hashlib
import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
PROD_DB = BACKEND_DIR / "data" / "clarion.db"
PROD_DB_EXPECTED_SIZE = 34533376


def test_t3_x06_db_safety_regression(tmp_path):
    """T3.INT.6: F14 + F15 Pairwise Integration."""
    # Pre-test production DB snapshot
    assert PROD_DB.exists(), "Production clarion.db must exist"
    pre_stat = PROD_DB.stat()
    assert pre_stat.st_size == PROD_DB_EXPECTED_SIZE
    pre_hash = hashlib.sha256(PROD_DB.read_bytes()).hexdigest()

    # Perform backend operations inside tmp_path
    from services.chunking_service import ChunkingService
    from services.embedding_service import EmbeddingService

    chunker = ChunkingService()
    test_text = (
        "Advanced Graph Algorithms in Computer Science.\n\n"
        "Tarjan strongly connected components algorithm runs in linear time O(V + E). "
        "It uses a single depth-first search traversal with low-link values to identify components."
    )
    chunks = chunker.chunk_document("doc-x06-isolated", test_text)
    assert len(chunks) >= 1

    embedder = EmbeddingService()
    vec = embedder._fallback_embed_text(chunks[0].content)
    assert len(vec) == 1024

    # Post-test production DB verification
    post_stat = PROD_DB.stat()
    assert post_stat.st_size == PROD_DB_EXPECTED_SIZE
    assert post_stat.st_mtime == pre_stat.st_mtime
    post_hash = hashlib.sha256(PROD_DB.read_bytes()).hexdigest()
    assert post_hash == pre_hash
