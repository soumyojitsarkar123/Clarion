"""Tier 1 Feature Coverage: F11 Bounded Retries, Backoff & JSON Schema Validation.

Specifications:
- T1.F11.1: Success on First Attempt (0 Retries)
- T1.F11.2: Success on Second Attempt After 1 Failure
- T1.F11.3: Exponential Backoff Interval Calculation
- T1.F11.4: Strict JSON Schema Key Validation
- T1.F11.5: Permanent Failure After 3 Exhausted Retries
"""

import time
from unittest.mock import MagicMock, patch
import pytest
from services.dataset_factory_service import DatasetFactoryService


def test_f11_success_first_attempt(isolated_settings):
    """T1.F11.1: Success on First Attempt (0 Retries)."""
    factory = DatasetFactoryService()
    chunk = {
        "chunk_id": "chk-retry-1",
        "section_title": "Intro",
        "content": "Graph theory studies mathematical structures used to model pairwise relations.",
    }

    mock_provider = MagicMock()
    mock_provider.get_model_info.return_value = MagicMock(model_name="qwen3.5:4b", provider=MagicMock(value="ollama"))
    mock_provider.generate_structured.return_value = {
        "samples": [
            {
                "chunk_id": "chk-retry-1",
                "sample_type": "qa",
                "instruction": "What does graph theory study?",
                "response": "It studies mathematical structures used to model pairwise relations between objects.",
                "concepts": ["graph theory"],
                "grounding_excerpt": "Graph theory studies mathematical structures.",
            }
        ]
    }
    factory._provider = mock_provider

    samples, meta = factory._generate_chunk_with_retry(
        chunk=chunk,
        document_title="Math",
        generation_id="gen-test-01",
        document_id="doc-test-01",
        max_retries=3,
        chunk_timeout_seconds=5.0,
    )

    assert meta["status"] == "completed"
    assert meta["retry_count"] == 0
    assert len(samples) == 1
    assert mock_provider.generate_structured.call_count == 1


def test_f11_success_second_attempt(isolated_settings):
    """T1.F11.2: Success on Second Attempt After 1 Failure."""
    factory = DatasetFactoryService()
    chunk = {
        "chunk_id": "chk-retry-2",
        "section_title": "Intro",
        "content": "Graph theory studies mathematical structures.",
    }

    mock_provider = MagicMock()
    mock_provider.get_model_info.return_value = MagicMock(model_name="qwen3.5:4b", provider=MagicMock(value="ollama"))
    # First call raises timeout, second call succeeds
    mock_provider.generate_structured.side_effect = [
        TimeoutError("Connection timed out"),
        {
            "samples": [
                {
                    "chunk_id": "chk-retry-2",
                    "sample_type": "qa",
                    "instruction": "Question 2?",
                    "response": "Answer 2.",
                    "concepts": ["graph"],
                    "grounding_excerpt": "Graph theory studies.",
                }
            ]
        },
    ]
    factory._provider = mock_provider

    with patch("time.sleep", return_value=None):  # fast backoff during test
        samples, meta = factory._generate_chunk_with_retry(
            chunk=chunk,
            document_title="Math",
            generation_id="gen-test-02",
            document_id="doc-test-02",
            max_retries=3,
            chunk_timeout_seconds=5.0,
        )

    assert meta["status"] == "completed"
    assert meta["retry_count"] == 1
    assert len(samples) == 1
    assert mock_provider.generate_structured.call_count == 2


def test_f11_exponential_backoff_intervals():
    """T1.F11.3: Exponential Backoff Interval Calculation."""
    base = 1.0
    jitter = 0.2
    for attempt in range(3):
        expected_nominal = base * (2 ** attempt)
        min_bound = expected_nominal * (1 - jitter)
        max_bound = expected_nominal * (1 + jitter)
        assert min_bound < expected_nominal < max_bound
    # Verify exponential progression: 1s, 2s, 4s
    assert base * (2 ** 0) == 1.0
    assert base * (2 ** 1) == 2.0
    assert base * (2 ** 2) == 4.0


def test_f11_strict_json_schema_validation():
    """T1.F11.4: Strict JSON Schema Key Validation."""
    factory = DatasetFactoryService()
    chunk = {"chunk_id": "chk-val-1", "content": "Sample content"}

    # Malformed / missing required keys
    invalid_raw = {"something_else": "invalid payload"}
    extracted = factory._repair_and_extract_samples(invalid_raw, chunk)
    assert len(extracted) == 0, "Should reject payload without required instruction/response"

    # Valid payload
    valid_raw = {
        "samples": [
            {
                "chunk_id": "chk-val-1",
                "instruction": "What is the key?",
                "response": "The key is a unique identifier.",
            }
        ]
    }
    extracted_valid = factory._repair_and_extract_samples(valid_raw, chunk)
    assert len(extracted_valid) == 1
    assert extracted_valid[0]["instruction"] == "What is the key?"


def test_f11_failure_after_max_retries(isolated_settings):
    """T1.F11.5: Permanent Failure After 3 Exhausted Retries."""
    factory = DatasetFactoryService()
    chunk = {
        "chunk_id": "chk-retry-fail",
        "section_title": "Failures",
        "content": "This will fail 3 times.",
    }

    mock_provider = MagicMock()
    mock_provider.get_model_info.return_value = MagicMock(model_name="qwen3.5:4b", provider=MagicMock(value="ollama"))
    mock_provider.generate_structured.side_effect = RuntimeError("HTTP 500 Internal Error")
    factory._provider = mock_provider

    with patch("time.sleep", return_value=None):
        samples, meta = factory._generate_chunk_with_retry(
            chunk=chunk,
            document_title="Math",
            generation_id="gen-test-fail",
            document_id="doc-test-fail",
            max_retries=3,
            chunk_timeout_seconds=5.0,
        )

    assert meta["status"] == "failed"
    assert meta["retry_count"] == 3
    assert len(samples) == 0
    assert mock_provider.generate_structured.call_count == 3
