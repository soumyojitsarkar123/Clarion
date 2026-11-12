"""
Tests for Milestone M3: Dataset Factory Reliability, Resilience, Checkpointing, and Strict Export Gate.

All tests use the isolated_settings fixture to ensure Clarion-Backend/data/clarion.db
is never accessed or modified.
"""

import asyncio
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from core.llm.base import (
    BaseLLMProvider,
    LLMProviderError,
    LLMResponse,
    ModelInfo,
    ProviderType,
    TokenUsage,
)
from services.dataset_factory_service import DatasetFactoryService
from utils.config import settings


class MockLLMProvider(BaseLLMProvider):
    """Controllable mock LLM provider for deterministic resilience testing."""

    def __init__(self, responses=None, side_effects=None, **kwargs):
        kwargs.setdefault("api_key", "test-key")
        super().__init__(**kwargs)
        self.responses = list(responses or [])
        self.side_effects = list(side_effects or [])
        self.call_count = 0
        self.structured_call_count = 0
        self.calls = []

    def _initialize_client(self):
        pass

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        self.call_count += 1
        self.calls.append({"prompt": prompt, "kwargs": kwargs})

        # Quality scoring prompts return high quality score
        if "rate the quality" in prompt.lower() or "dataset quality rater" in str(kwargs).lower():
            return LLMResponse(
                content="0.95",
                model="mock-qwen",
                usage=TokenUsage(prompt_tokens=15, completion_tokens=5, total_tokens=20),
            )

        if self.side_effects:
            eff = self.side_effects.pop(0)
            if isinstance(eff, Exception):
                raise eff
            return eff

        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            if isinstance(resp, str):
                return LLMResponse(
                    content=resp,
                    model="mock-qwen",
                    usage=TokenUsage(prompt_tokens=25, completion_tokens=50, total_tokens=75),
                )
            return resp

        return LLMResponse(content="0.95", model="mock-qwen", usage=TokenUsage())

    async def generate_structured(
        self, prompt: str, output_schema: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        self.call_count += 1
        self.structured_call_count += 1
        self.calls.append({"prompt": prompt, "output_schema": output_schema, "kwargs": kwargs})

        if self.side_effects:
            eff = self.side_effects.pop(0)
            if isinstance(eff, Exception):
                raise eff
            return eff

        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            if isinstance(resp, (dict, list)):
                return resp
            return json.loads(resp)

        return {"samples": []}

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            provider=ProviderType.OLLAMA,
            model_name="mock-qwen3.5:4b",
            max_tokens=2000,
            context_window=8192,
            supports_functions=False,
            supports_json_mode=True,
        )


def _init_test_document_and_chunks(db_path: Path, doc_id: str = "doc-test-1", num_chunks: int = 2):
    """Seed an isolated SQLite database with test document and substantial chunks."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            file_type TEXT NOT NULL,
            upload_time TEXT NOT NULL,
            page_count INTEGER,
            word_count INTEGER,
            status TEXT NOT NULL,
            text_content TEXT,
            error_message TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            content TEXT NOT NULL,
            position_index INTEGER NOT NULL
        )
    """)

    cursor.execute("""
        INSERT OR REPLACE INTO documents
        (id, filename, file_size, file_type, upload_time, page_count, word_count, status, text_content, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        doc_id,
        "machine_learning_notes.pdf",
        1024,
        "pdf",
        datetime.now().isoformat(),
        1,
        250,
        "analyzed",
        "Full document content for testing.",
        None,
    ))

    chunk_templates = [
        (
            f"chunk-{i}",
            doc_id,
            (
                f"Section {i}: Gradient Descent Optimization in Machine Learning. "
                "Stochastic gradient descent optimizes an objective function with suitable smoothness properties. "
                f"In step {i}, the parameter update rule theta_{{t+1}} = theta_t - eta * grad(L) computes parameter steps. "
                "The learning rate determines convergence speed and stability across iterations. "
                "Excessive learning rates cause oscillations while very small values lead to slow convergence."
            ),
            i,
        )
        for i in range(num_chunks)
    ]

    for c in chunk_templates:
        cursor.execute("""
            INSERT OR REPLACE INTO chunks (chunk_id, document_id, content, position_index)
            VALUES (?, ?, ?, ?)
        """, c)

    conn.commit()
    conn.close()


def _valid_sample_payload(
    chunk_id: str,
    excerpt: str,
    instruction: Optional[str] = None,
    response: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a valid sample structure matching the generation schema."""
    if not instruction:
        if "1" in chunk_id:
            instruction = f"How does the learning rate influence convergence velocity and oscillation behavior in {chunk_id}?"
        else:
            instruction = f"What is the parameter update rule in stochastic gradient descent for {chunk_id}?"

    if not response:
        if "1" in chunk_id:
            response = f"In {chunk_id}, excessive learning rates cause dangerous oscillations while very small values lead to extremely slow convergence toward the optimal objective minimum."
        else:
            response = f"The parameter update rule in stochastic gradient descent for {chunk_id} updates parameters using theta_{{t+1}} = theta_t - eta * grad(L). This controls step sizes across iterations."

    return {
        "samples": [
            {
                "chunk_id": chunk_id,
                "sample_type": "qa",
                "instruction": instruction,
                "response": response,
                "concepts": ["Gradient Descent", "Learning Rate"],
                "grounding_excerpt": excerpt,
            }
        ]
    }


# ============================================================================
# Tests
# ============================================================================

def test_per_chunk_processing_preserves_provenance(isolated_settings):
    """
    Test 1: Provenance preservation.
    Verifies that samples retain document_id, chunk_id, section_title,
    source_text_snippet, and grounding_excerpt throughout generation and export.
    """
    doc_id = "doc-prov-001"
    _init_test_document_and_chunks(isolated_settings.data_dir / "clarion.db", doc_id=doc_id, num_chunks=1)

    chunk_excerpt = "the parameter update rule theta_{t+1} = theta_t - eta * grad(L)"
    mock_provider = MockLLMProvider(responses=[_valid_sample_payload("chunk-0", chunk_excerpt)])

    service = DatasetFactoryService()
    service._provider = mock_provider

    batch = service.process_document(doc_id)

    assert batch.status == "completed"
    assert batch.sample_count == 1
    sample = batch.samples[0]

    # Verify provenance on TrainingSample
    assert sample.document_id == doc_id
    assert sample.chunk_id == "chunk-0"
    assert sample.metadata["section_title"] != ""
    assert sample.metadata["source_text_snippet"].startswith("Section 0: Gradient Descent")
    assert sample.metadata["grounding_excerpt"] == chunk_excerpt
    assert sample.metadata["generation_mode"].startswith("llm_")

    # Verify export files contain provenance block
    assert len(batch.exported_files) >= 2
    json_path = Path(batch.exported_files[0])
    jsonl_path = Path(batch.exported_files[1])

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    sample_entry = json_data["samples"][0]
    assert "provenance" in sample_entry
    assert sample_entry["provenance"]["source_text_snippet"].startswith("Section 0:")
    assert sample_entry["provenance"]["grounding_excerpt"] == chunk_excerpt

    with open(jsonl_path, "r", encoding="utf-8") as f:
        line = json.loads(f.readline())
    assert "provenance" in line["metadata"]
    assert line["metadata"]["provenance"]["grounding_excerpt"] == chunk_excerpt


def test_oversized_dataset_prompt_is_rejected_once_before_provider_call(isolated_settings):
    service = DatasetFactoryService()
    provider = MockLLMProvider()
    service._provider = provider
    result, state = service._generate_chunk_with_retry(
        chunk={
            "chunk_id": "oversized-chunk",
            "content": "source " * 6000,
            "section_title": "Large section",
            "context": "",
        },
        document_title="Budget test",
        generation_id="budget-generation",
        document_id="budget-document",
        max_retries=4,
    )

    assert result == []
    assert state["budget_rejected"] is True
    assert state["retry_count"] == 0
    assert provider.structured_call_count == 0
    attempt = service.get_generation_attempts(document_id="budget-document")["attempts"][0]
    assert attempt["budget_rejected"] == 1
    assert attempt["prompt_tokens_exact"] == 0
    assert attempt["token_count_method"] == "utf8_byte_upper_bound_estimate"


def test_dataset_attempt_telemetry_additive_migration(isolated_settings):
    database = isolated_settings.data_dir / "clarion.db"
    conn = sqlite3.connect(str(database))
    conn.execute("""
        CREATE TABLE dataset_generation_attempts (
            attempt_id TEXT PRIMARY KEY, generation_id TEXT NOT NULL,
            document_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
            model TEXT NOT NULL, provider TEXT NOT NULL,
            prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,
            prompt_chars INTEGER NOT NULL, completion_chars INTEGER NOT NULL,
            request_duration_ms REAL NOT NULL, retry_count INTEGER NOT NULL,
            status TEXT NOT NULL, failure_reason TEXT,
            valid_record_count INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

    DatasetFactoryService()
    conn = sqlite3.connect(str(database))
    columns = {row[1] for row in conn.execute("PRAGMA table_info(dataset_generation_attempts)")}
    conn.close()
    assert {
        "token_count_method", "prompt_tokens_exact", "completion_tokens_exact",
        "actual_prompt_tokens", "actual_completion_tokens", "context_limit",
        "reserved_output_tokens", "budget_rejected",
    }.issubset(columns)


def test_bounded_retries_and_exponential_backoff(isolated_settings, monkeypatch):
    """
    Test 2: Bounded retries and exponential backoff.
    Mock provider fails on attempts 0 and 1 with TimeoutError, then succeeds on attempt 2.
    Verifies 3 attempts logged in dataset_generation_attempts and checkpoint retry count.
    """
    doc_id = "doc-retry-002"
    _init_test_document_and_chunks(isolated_settings.data_dir / "clarion.db", doc_id=doc_id, num_chunks=1)

    chunk_excerpt = "The parameter update rule theta_{t+1} = theta_t - eta * grad(L)"
    mock_provider = MockLLMProvider(
        side_effects=[
            asyncio.TimeoutError("Request timed out after 60s"),
            LLMProviderError("Transient connection reset"),
            _valid_sample_payload("chunk-0", chunk_excerpt),
        ]
    )

    # Use fast backoff for instant test execution
    monkeypatch.setattr(settings, "dataset_backoff_base_seconds", 0.01)
    monkeypatch.setattr(settings, "dataset_max_retries_per_chunk", 3)

    service = DatasetFactoryService()
    service._provider = mock_provider

    batch = service.process_document(doc_id)

    assert batch.status == "completed"
    assert batch.sample_count == 1

    # Verify telemetry records
    attempts_result = service.get_generation_attempts(document_id=doc_id)
    assert attempts_result["total"] == 3
    # Attempts ordered by created_at DESC
    attempts = attempts_result["attempts"]
    assert attempts[0]["status"] == "completed"
    assert attempts[0]["retry_count"] == 2
    assert attempts[1]["status"] == "failed"
    assert attempts[1]["retry_count"] == 1
    assert attempts[2]["status"] == "failed"
    assert attempts[2]["retry_count"] == 0

    # Verify checkpoint
    checkpoints = service._get_checkpoints(doc_id)
    assert checkpoints["chunk-0"]["status"] == "completed"
    assert checkpoints["chunk-0"]["retry_count"] == 2


def test_per_request_timeout_enforcement(isolated_settings, monkeypatch):
    """
    Test 3: Per-request timeout exhaustion.
    Provider continuously raises TimeoutError.
    Verifies all 3 retries fail, attempts logged, and chunk checkpoint marked failed.
    """
    doc_id = "doc-timeout-003"
    _init_test_document_and_chunks(isolated_settings.data_dir / "clarion.db", doc_id=doc_id, num_chunks=1)

    mock_provider = MockLLMProvider(
        side_effects=[
            asyncio.TimeoutError("Timeout on attempt 0"),
            asyncio.TimeoutError("Timeout on attempt 1"),
            asyncio.TimeoutError("Timeout on attempt 2"),
        ]
    )

    monkeypatch.setattr(settings, "dataset_backoff_base_seconds", 0.01)
    monkeypatch.setattr(settings, "dataset_max_retries_per_chunk", 3)

    service = DatasetFactoryService()
    service._provider = mock_provider

    batch = service.process_document(doc_id)

    assert batch.status == "failed"
    assert batch.sample_count == 0

    attempts_result = service.get_generation_attempts(document_id=doc_id)
    assert attempts_result["total"] == 3
    for att in attempts_result["attempts"]:
        assert att["status"] == "failed"
        assert "Timeout" in (att["failure_reason"] or "")

    checkpoints = service._get_checkpoints(doc_id)
    assert checkpoints["chunk-0"]["status"] == "failed"
    assert checkpoints["chunk-0"]["retry_count"] == 3


def test_multi_stage_json_repair(isolated_settings):
    """
    Test 4: 7-stage JSON repair pipeline.
    Tests extraction across multiple noisy model output formats:
    - Thinking tags (<think>...</think>)
    - Markdown code fences (```json ... ```)
    - Conversational preamble
    - Direct JSON string
    - Root list
    - Root dict without 'samples' key
    """
    service = DatasetFactoryService()
    chunk = {
        "chunk_id": "chk-parse-1",
        "section_title": "Parsing Test Section",
        "content": "This is a detailed technical chunk discussing machine learning optimization and gradient algorithms in depth. Convergence guarantees apply to smooth convex functions under appropriate step sizes.",
        "context": "",
    }

    # Case A: Thinking tags + markdown fences + preamble
    raw_noisy = """
<think>
Step 1: Read the chunk.
Step 2: Generate QA pair.
</think>
Here is the JSON result:
```json
{
  "samples": [
    {
      "chunk_id": "chk-parse-1",
      "sample_type": "qa",
      "instruction": "What convergence guarantees apply to gradient algorithms?",
      "response": "Convergence guarantees apply to smooth convex functions when appropriate step sizes are selected for optimization routines and gradient iterations across training epochs.",
      "concepts": ["Convergence", "Gradient"],
      "grounding_excerpt": "Convergence guarantees apply to smooth convex functions"
    }
  ]
}
```
"""
    extracted_a = service._repair_and_extract_samples(raw_noisy, chunk)
    assert len(extracted_a) == 1
    assert extracted_a[0]["instruction"].startswith("What convergence")
    assert extracted_a[0]["sample_type"] == "qa"

    # Case B: Root list format
    raw_list = [
        {
            "chunk_id": "chk-parse-1",
            "sample_type": "concept",
            "instruction": "Define convergence guarantees in machine learning optimization.",
            "response": "Convergence guarantees define the mathematical conditions under which an optimization algorithm reaches the global or local minimum reliably across training epochs.",
            "concepts": ["Convergence"],
            "grounding_excerpt": "Convergence guarantees apply to smooth convex functions",
        }
    ]
    extracted_b = service._repair_and_extract_samples(raw_list, chunk)
    assert len(extracted_b) == 1
    assert extracted_b[0]["sample_type"] == "concept"

    # Case C: Single sample dict format (without 'samples' root key)
    raw_single = {
        "chunk_id": "chk-parse-1",
        "sample_type": "qa",
        "instruction": "How do step sizes impact algorithm convergence?",
        "response": "Step sizes determine whether the optimization algorithm converges smoothly or diverges completely during iterative gradient updates across training epochs.",
        "concepts": ["Step Sizes"],
        "grounding_excerpt": "Convergence guarantees apply to smooth convex functions",
    }
    extracted_c = service._repair_and_extract_samples(raw_single, chunk)
    assert len(extracted_c) == 1
    assert extracted_c[0]["instruction"].startswith("How do step sizes")


def test_checkpoint_storage_and_resume(isolated_settings, monkeypatch):
    """
    Test 5: Checkpointing and incremental resume.
    1st run: chunk-0 succeeds, chunk-1 fails -> status is partially_completed.
    2nd run with resume=True: chunk-0 is skipped, chunk-1 retries and succeeds -> status becomes completed.
    """
    doc_id = "doc-resume-005"
    _init_test_document_and_chunks(isolated_settings.data_dir / "clarion.db", doc_id=doc_id, num_chunks=2)

    chunk0_excerpt = "In step 0, the parameter update rule theta_{t+1}"
    chunk1_excerpt = "In step 1, the parameter update rule theta_{t+1}"

    # Provider for Run 1: chunk-0 succeeds, chunk-1 fails 3 times
    provider_run1 = MockLLMProvider(
        side_effects=[
            _valid_sample_payload("chunk-0", chunk0_excerpt),
            asyncio.TimeoutError("Timeout on chunk 1 attempt 0"),
            asyncio.TimeoutError("Timeout on chunk 1 attempt 1"),
            asyncio.TimeoutError("Timeout on chunk 1 attempt 2"),
        ]
    )

    monkeypatch.setattr(settings, "dataset_backoff_base_seconds", 0.01)
    monkeypatch.setattr(settings, "dataset_max_retries_per_chunk", 3)

    service1 = DatasetFactoryService()
    service1._provider = provider_run1

    batch1 = service1.process_document(doc_id, resume=True)
    assert batch1.status == "partially_completed"
    assert batch1.completed_chunk_count == 1
    assert batch1.sample_count == 1

    # Verify checkpoints in SQLite
    chk_summary1 = service1.get_checkpoint_summary(doc_id)
    assert chk_summary1["completed_chunks"] == 1
    assert chk_summary1["failed_chunks"] == 1
    assert chk_summary1["overall_status"] == "partially_completed"

    # Provider for Run 2: chunk-1 now succeeds.
    # Note: provider_run2 should ONLY be called for chunk-1, NEVER for chunk-0!
    provider_run2 = MockLLMProvider(
        responses=[_valid_sample_payload("chunk-1", chunk1_excerpt)]
    )

    service2 = DatasetFactoryService()
    service2._provider = provider_run2

    batch2 = service2.process_document(doc_id, resume=True)

    # Chunk generation should have been invoked exactly ONCE (for chunk-1 only)
    assert provider_run2.structured_call_count == 1
    chunk_calls = [c for c in provider_run2.calls if "output_schema" in c]
    assert len(chunk_calls) == 1
    assert "chunk-1" in chunk_calls[0]["prompt"]
    assert "chunk-0" not in chunk_calls[0]["prompt"]

    assert batch2.status == "completed"
    assert batch2.completed_chunk_count == 2
    assert batch2.sample_count == 2

    chk_summary2 = service2.get_checkpoint_summary(doc_id)
    assert chk_summary2["completed_chunks"] == 2
    assert chk_summary2["failed_chunks"] == 0
    assert chk_summary2["overall_status"] == "completed"


def test_strict_export_gate_refusal_on_partial(isolated_settings, monkeypatch):
    """
    Test 6: Strict gold export refusal on incomplete dataset.
    Under require_llm_for_export=True, an uncompleted or partially completed document
    must raise HTTP 409 Conflict when export_document_dataset_strict is invoked.
    """
    doc_id = "doc-gate-006"
    _init_test_document_and_chunks(isolated_settings.data_dir / "clarion.db", doc_id=doc_id, num_chunks=2)

    chunk0_excerpt = "In step 0, the parameter update rule theta_{t+1}"
    provider = MockLLMProvider(
        side_effects=[
            _valid_sample_payload("chunk-0", chunk0_excerpt),
            asyncio.TimeoutError("Timeout on chunk 1"),
            asyncio.TimeoutError("Timeout on chunk 1"),
            asyncio.TimeoutError("Timeout on chunk 1"),
        ]
    )

    monkeypatch.setattr(settings, "dataset_backoff_base_seconds", 0.01)
    monkeypatch.setattr(settings, "dataset_max_retries_per_chunk", 3)

    service = DatasetFactoryService()
    service._provider = provider
    service.require_llm_for_export = True

    batch = service.process_document(doc_id, resume=True)
    assert batch.status == "partially_completed"
    # Automatic export suppressed during batch processing
    assert batch.exported_files == []

    # Direct export attempt must be strictly refused with HTTP 409 Conflict
    with pytest.raises(HTTPException) as exc_info:
        service.export_document_dataset_strict(doc_id)

    assert exc_info.value.status_code == 409
    assert "Export refused" in exc_info.value.detail
    assert "incomplete" in exc_info.value.detail.lower()


def test_no_heuristic_fallback_when_require_llm(isolated_settings):
    """
    Test 7: No heuristic records substituted when require_llm_for_export=True.
    When LLM provider is unavailable, 0 deterministic samples are created,
    batch status is 'failed', and export refuses with HTTP 409.
    """
    doc_id = "doc-no-fallback-007"
    _init_test_document_and_chunks(isolated_settings.data_dir / "clarion.db", doc_id=doc_id, num_chunks=2)

    service = DatasetFactoryService()
    # Force provider to be unavailable
    service._provider = None
    service._provider_error = "Connection refused"
    service.require_llm_for_export = True

    batch = service.process_document(doc_id)

    assert batch.status == "failed"
    assert batch.sample_count == 0
    assert batch.generation_mode == "llm_unavailable"
    assert batch.exported_files == []

    # Export must be refused with 409
    with pytest.raises(HTTPException) as exc_info:
        service.export_document_dataset_strict(doc_id)

    assert exc_info.value.status_code == 409
    assert "Export refused" in exc_info.value.detail


def test_successful_strict_export_when_all_completed(isolated_settings, monkeypatch):
    """
    Test 8: Successful strict export when all chunks completed with valid LLM data.
    """
    monkeypatch.setattr(settings, "dataset_backoff_base_seconds", 0.01)
    monkeypatch.setattr(settings, "dataset_max_retries_per_chunk", 3)

    doc_id = "doc-success-008"
    _init_test_document_and_chunks(isolated_settings.data_dir / "clarion.db", doc_id=doc_id, num_chunks=2)

    chunk0_excerpt = "In step 0, the parameter update rule theta_{t+1}"
    chunk1_excerpt = "In step 1, the parameter update rule theta_{t+1}"

    provider = MockLLMProvider(
        responses=[
            _valid_sample_payload("chunk-0", chunk0_excerpt),
            _valid_sample_payload("chunk-1", chunk1_excerpt),
        ]
    )

    service = DatasetFactoryService()
    service._provider = provider
    service.require_llm_for_export = True

    batch = service.process_document(doc_id)
    assert batch.status == "completed"
    assert batch.sample_count == 2
    assert len(batch.exported_files) >= 2

    # Direct export succeeds
    result = service.export_document_dataset_strict(doc_id)
    assert result["status"] == "success"
    assert result["document_id"] == doc_id
    assert result["sample_count"] == 2
    assert Path(result["export_path"]).exists()
    assert Path(result["jsonl_path"]).exists()
