"""Unit tests for structure-aware, heading-sensitive, word-based chunking."""

import pytest

from models.chunk import Chunk
from services.chunking_service import ChunkingService


def make_service(isolated_settings, chunk_size=8, overlap=2, min_size=3):
    isolated_settings.chunk_size = chunk_size
    isolated_settings.chunk_overlap = overlap
    isolated_settings.min_chunk_size = min_size
    return ChunkingService()


def test_short_text_produces_single_chunk(isolated_settings):
    service = make_service(isolated_settings)
    chunks = service.chunk_document("doc-1", "a small piece of text")
    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].word_count == 5
    assert chunks[0].position_index == 0


def test_long_text_is_split_with_word_slices(isolated_settings):
    service = make_service(isolated_settings, chunk_size=8, overlap=2)
    words = ["word"] * 50
    text = " ".join(words)
    chunks = service.chunk_document("doc-2", text)

    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert 3 <= chunk.word_count <= 8
    # The last chunk may absorb a trailing remainder smaller than min_chunk_size.
    assert chunks[-1].word_count <= 8 + (8 - 2)
    joined_words = " ".join(chunk.content for chunk in chunks).split()
    assert len(joined_words) >= 50 * 0.95
    assert all(chunk.document_id == "doc-2" for chunk in chunks)


def test_trailing_undersized_slice_is_merged(isolated_settings):
    service = make_service(isolated_settings, chunk_size=10, overlap=0, min_size=4)
    words = [f"w{i}" for i in range(13)]
    chunks = service.chunk_document("doc-3", " ".join(words))

    # 13 words -> one 10-word chunk plus a 3-word remainder that is below
    # min_chunk_size, so the remainder is merged into the previous chunk.
    assert len(chunks) == 1
    assert chunks[0].word_count == 13
    assert chunks[0].content.endswith("w12")


def test_merge_decision_uses_word_count_not_character_length(isolated_settings):
    # A trailing slice of 20 short words is ~80 characters. The old
    # character-based check treated it as large enough to keep standalone;
    # the word-based check must merge it because 20 < min_chunk_size.
    service = make_service(isolated_settings, chunk_size=100, overlap=0, min_size=50)
    words = [f"w{i}" for i in range(120)]
    chunks = service.chunk_document("doc-4", " ".join(words))

    assert len(chunks) == 1
    assert chunks[0].word_count == 120


def test_section_heading_is_preserved(isolated_settings):
    service = make_service(isolated_settings, chunk_size=100, overlap=0)
    text = "# Introduction\n\nThis is the introduction section content.\n\n# Methods\n\nThis is the methods section content."
    chunks = service.chunk_document("doc-5", text)
    titles = [chunk.section_title for chunk in chunks]
    assert "Introduction" in titles
    assert "Methods" in titles


def test_numbered_heading_detection(isolated_settings):
    service = make_service(isolated_settings)
    headings = service._detect_headings("1. First Section\n2. Second Section\nplain line has a period.")
    assert len(headings) == 2
    assert headings[0][0] == "First Section"
    assert headings[1][0] == "Second Section"


def test_chunks_persist_and_load(isolated_settings):
    service = make_service(isolated_settings)
    chunks = service.chunk_document("doc-6", "persisted chunk text for the database")
    loaded = service.get_chunks("doc-6")
    assert [chunk.chunk_id for chunk in loaded] == [chunk.chunk_id for chunk in chunks]
    assert service.get_chunk(chunks[0].chunk_id) is not None


def test_delete_chunks(isolated_settings):
    service = make_service(isolated_settings)
    service.chunk_document("doc-7", "text to delete afterwards")
    service.delete_chunks("doc-7")
    assert service.get_chunks("doc-7") == []


def test_undersized_sections_are_merged(isolated_settings):
    service = make_service(isolated_settings, chunk_size=20, overlap=0, min_size=5)
    # Section 1 has only 3 words (below min_size=5), Section 2 has 8 words.
    text = "# Section 1\nTiny bit here.\n\n# Section 2\nHere is a longer section with plenty of words."
    chunks = service.chunk_document("doc-8", text)
    # The undersized Section 1 should be merged with Section 2 rather than existing as an isolated 3-word chunk.
    assert len(chunks) == 1
    assert chunks[0].word_count >= 8


def test_tail_chunk_overlap_deduplication(isolated_settings):
    """Verify that merging an undersized remainder does not repeat overlap words."""
    service = make_service(isolated_settings, chunk_size=10, overlap=3, min_size=6)
    words = [f"token{i}" for i in range(12)]
    chunks = service.chunk_document("doc-tail-dedup", " ".join(words))
    assert len(chunks) == 1
    content_tokens = chunks[0].content.split()
    assert len(content_tokens) == 12
    assert len(set(content_tokens)) == 12
    assert content_tokens == words

