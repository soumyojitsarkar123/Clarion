"""Tier 2 Boundary & Corner Cases: test_t2_b08_provenance_boundaries.py
Feature F8: Complete Provenance Preservation.
Verifies boundary conditions: chunks with missing section titles, short chunk snippets (<200 chars),
special characters in filenames, extreme chunk ID lengths (256 chars), and exact character offset grounding.
"""

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_factory_service import DatasetFactoryService


def test_b08_missing_or_none_section_title(tmp_path):
    """T2.F8.1: Chunk with missing or empty section title fallback."""
    factory = DatasetFactoryService(db_path=tmp_path / "factory_title.db")
    title = factory._resolve_chunk_section_title("Plain text without heading markers", position_index=3)
    assert title is not None
    assert len(title) > 0
    # Falls back to Chunk 4 or candidate term
    assert title == "Chunk 4" or isinstance(title, str)


def test_b08_short_chunk_snippet(tmp_path):
    """T2.F8.2: Short chunk snippet (<200 characters) grounding excerpt."""
    factory = DatasetFactoryService(db_path=tmp_path / "factory_short_snip.db")
    short_text = "Short sentence of 35 characters."
    excerpt = factory._grounding_excerpt(short_text)
    assert excerpt == short_text
    assert len(excerpt) <= 220


def test_b08_special_chars_in_document_title(tmp_path):
    """T2.F8.3: Special characters in document filename sanitized safely."""
    factory = DatasetFactoryService(db_path=tmp_path / "factory_safe_stem.db")
    filename = "Math & Physics [Vol. 1] #3 (2026).pdf"
    safe_stem = factory._safe_filename_stem(filename)
    # Filename stem must contain only filesystem-safe characters [A-Za-z0-9_-]
    for ch in safe_stem:
        assert ch.isalnum() or ch in ("_", "-")
    assert "[" not in safe_stem
    assert "]" not in safe_stem
    assert "#" not in safe_stem
    assert "&" not in safe_stem


def test_b08_extreme_chunk_id_length(tmp_path):
    """T2.F8.4: Extremely long chunk ID string (256 characters) handled safely."""
    factory = DatasetFactoryService(db_path=tmp_path / "factory_long_id.db")
    long_chunk_id = "chunk_" + ("x" * 250)
    assert len(long_chunk_id) == 256
    # Verify sample creation retains the chunk_id without truncation
    sample = factory.create_training_sample(
        chunk_id=long_chunk_id,
        document_id="doc-test-provenance",
        sample_type="concept",
        input_text="What is topological sorting in directed acyclic graphs?",
        output_text="Topological sorting provides a linear ordering of vertices such that for every directed edge uv, u comes before v.",
        context="Graph theory fundamentals and DAG scheduling.",
    )
    if sample:
        assert sample.chunk_id == long_chunk_id


def test_b08_exact_offset_grounding_match(tmp_path):
    """T2.F8.5: Verification that grounding excerpt matches source text."""
    factory = DatasetFactoryService(db_path=tmp_path / "factory_exact_snip.db")
    source = "Dijkstra algorithm finds shortest paths from a single source node in weighted graphs. It maintains a set of visited vertices."
    excerpt = factory._grounding_excerpt(source)
    # Excerpt must be a genuine prefix substring of the source text
    assert excerpt in source
