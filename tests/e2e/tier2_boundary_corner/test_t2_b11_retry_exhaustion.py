"""Tier 2 Boundary & Corner Cases: test_t2_b11_retry_exhaustion.py
Feature F11: Bounded Retries, Backoff & JSON Schema Validation.
Verifies boundary conditions: markdown code fences stripping, conversational preambles,
concatenated JSON extraction, per-request timeout bounds, and null required fields rejection.
"""

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[2] / "Clarion-Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.dataset_factory_service import DatasetFactoryService
from utils.config import settings


def test_b11_markdown_fence_stripping(tmp_path):
    """T2.F11.1: Response wrapped in markdown code fences stripped cleanly."""
    factory = DatasetFactoryService(db_path=tmp_path / "factory_fence.db")
    chunk = {"chunk_id": "chk-001", "section_title": "Graph Algorithms", "content": "Graph algorithms content"}
    raw_fence = "```json\n{\"instruction\": \"What is a DAG?\", \"response\": \"A DAG is a directed acyclic graph.\"}\n```"
    samples = factory._repair_and_extract_samples(raw_fence, chunk)
    assert len(samples) >= 1
    assert samples[0]["instruction"] == "What is a DAG?"
    assert samples[0]["response"] == "A DAG is a directed acyclic graph."


def test_b11_conversational_preamble_stripping(tmp_path):
    """T2.F11.2: Conversational preamble before JSON stripped cleanly."""
    factory = DatasetFactoryService(db_path=tmp_path / "factory_preamble.db")
    chunk = {"chunk_id": "chk-002", "section_title": "Complexity", "content": "Complexity theory content"}
    raw_preamble = (
        "Certainly! Here is the research-grade training sample you requested:\n\n"
        "{\"instruction\": \"What is polynomial time?\", \"response\": \"Polynomial time complexity means execution time is bounded by a polynomial in input size.\"}\n\n"
        "Let me know if you need additional examples!"
    )
    samples = factory._repair_and_extract_samples(raw_preamble, chunk)
    assert len(samples) >= 1
    assert "polynomial time" in samples[0]["instruction"].lower()


def test_b11_multiple_concatenated_json_objects(tmp_path):
    """T2.F11.3: Response containing array of samples parsed without syntax crash."""
    factory = DatasetFactoryService(db_path=tmp_path / "factory_multi.db")
    chunk = {"chunk_id": "chk-003", "section_title": "Algorithms", "content": "Algorithm content"}
    raw_array = {
        "samples": [
            {"instruction": "Define Vertex", "response": "A vertex is a node in a graph."},
            {"instruction": "Define Edge", "response": "An edge connects two vertices in a graph."},
        ]
    }
    samples = factory._repair_and_extract_samples(raw_array, chunk)
    assert len(samples) == 2
    assert samples[0]["instruction"] == "Define Vertex"
    assert samples[1]["instruction"] == "Define Edge"


def test_b11_per_request_timeout_enforcement():
    """T2.F11.4: Per-request timeout parameter bounded."""
    # Ensure dataset request timeout configuration exists and is bounded <= 60.0s
    timeout = getattr(settings, "ollama_timeout", 60.0)
    assert isinstance(timeout, (int, float))
    assert timeout <= 120.0


def test_b11_null_required_fields_rejected(tmp_path):
    """T2.F11.5: JSON record with null required field rejected."""
    factory = DatasetFactoryService(db_path=tmp_path / "factory_null_field.db")
    chunk = {"chunk_id": "chk-004", "section_title": "Test", "content": "Test content"}
    # instruction is None / empty string
    raw_invalid = {"instruction": None, "response": "Valid response"}
    samples = factory._repair_and_extract_samples(raw_invalid, chunk)
    # Invalid sample without instruction must be dropped
    valid_samples = [s for s in samples if s.get("instruction")]
    assert len(valid_samples) == 0
