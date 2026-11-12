"""Unit tests for deterministic embedding fallback and vector store round-trip."""

import math

import numpy as np
import pytest

from models.chunk import Chunk
from services.embedding_service import EmbeddingService
from vectorstore import VectorStore


def make_chunks(count=3):
    return [
        Chunk(
            chunk_id=f"chunk-{i}",
            document_id="doc-1",
            content=f"Hypothesis testing content number {i} with null hypothesis and sample mean.",
            position_index=i,
            word_count=10,
        )
        for i in range(count)
    ]


def test_fallback_embedding_is_deterministic(isolated_settings):
    service = EmbeddingService()
    first = service._fallback_embed_text("hypothesis testing sample")
    second = service._fallback_embed_text("hypothesis testing sample")
    assert first == second
    assert len(first) == service.fallback_dimension
    norm = math.sqrt(sum(value * value for value in first))
    assert abs(norm - 1.0) < 1e-6


def test_fallback_embedding_differs_by_text(isolated_settings):
    service = EmbeddingService()
    first = service._fallback_embed_text("hypothesis testing")
    second = service._fallback_embed_text("sampling distribution")
    assert first != second


def test_vector_store_round_trip(isolated_settings):
    service = EmbeddingService()
    chunks = make_chunks()
    embeddings = [service._fallback_embed_text(chunk.content) for chunk in chunks]

    vector_store = service.create_vector_index("doc-1", chunks, embeddings)
    assert vector_store.index is not None

    reloaded = service.get_vector_store("doc-1")
    assert reloaded is not None
    assert reloaded.chunk_ids == [chunk.chunk_id for chunk in chunks]

    query = service._fallback_embed_text("null hypothesis")
    results = reloaded.search(query, k=2)
    assert len(results) == 2
    for chunk_id, distance in results:
        assert chunk_id in reloaded.chunk_ids
        assert distance >= 0


def test_delete_embeddings(isolated_settings):
    service = EmbeddingService()
    chunks = make_chunks(1)
    embeddings = [service._fallback_embed_text(chunk.content) for chunk in chunks]
    service.create_vector_index("doc-1", chunks, embeddings)
    service.delete_embeddings("doc-1")
    assert service.get_vector_store("doc-1") is None
    assert service.get_vector_store("doc-1") is None
